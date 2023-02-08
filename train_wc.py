from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.evaluator import eval_wc
from model.predictor import predict_wc #NEW

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools
import random

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--emb_file', default='./embedding/glove.6B.100d.txt', help='path to pre-trained embedding')
    parser.add_argument('--train_file', nargs='+', default='./data/ner2003/eng.train.iobes', help='path to training file')
    parser.add_argument('--dev_file', nargs='+', default='./data/ner2003/eng.testa.iobes', help='path to development file')
    parser.add_argument('--test_file', nargs='+', default='./data/ner2003/eng.testb.iobes', help='path to test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=300, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_false', help='fine tune the diction of word embedding or not')
    parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--lambda0', type=float, default=1, help='lambda0')
    parser.add_argument('--co_train', action='store_true', help='cotrain language model')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--output_annotation', action='store_true', help='output annotation results or not')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')
    file_num = len(args.train_file)
    lines = []
    dev_lines = []
    test_lines = []
    for i in range(file_num):
        with codecs.open(args.train_file[i], 'r', 'utf-8') as f:
            lines0 = f.readlines()
        lines.append(lines0)
    for i in range(file_num):
        with codecs.open(args.dev_file[i], 'r', 'utf-8') as f:
            dev_lines0 = f.readlines()
        dev_lines.append(dev_lines0)
    for i in range(file_num):
        with codecs.open(args.test_file[i], 'r', 'utf-8') as f:
            test_lines0 = f.readlines()
        test_lines.append(test_lines0)

    dataset_loader = []
    dev_dataset_loader = []
    test_dataset_loader = []
    f_map = dict()
    l_map = dict()
    char_count = dict()
    train_features = []
    dev_features = []
    test_features = []
    train_labels = []
    dev_labels = []
    test_labels = []
    train_features_tot = []
    test_word = []

    for i in range(file_num):
        dev_features0, dev_labels0 = utils.read_corpus(dev_lines[i])
        test_features0, test_labels0 = utils.read_corpus(test_lines[i])

        dev_features.append(dev_features0)
        test_features.append(test_features0)
        dev_labels.append(dev_labels0)
        test_labels.append(test_labels0)

        if args.output_annotation: #NEW
            test_word0 = utils.read_features(test_lines[i])
            test_word.append(test_word0)

        if args.load_check_point:
            if os.path.isfile(args.load_check_point):
                print("loading checkpoint: '{}'".format(args.load_check_point))
                checkpoint_file = torch.load(args.load_check_point)
                args.start_epoch = checkpoint_file['epoch']
                f_map = checkpoint_file['f_map']
                l_map = checkpoint_file['l_map']
                c_map = checkpoint_file['c_map']
                in_doc_words = checkpoint_file['in_doc_words']
                train_features, train_labels = utils.read_corpus(lines[i])
            else:
                print("no checkpoint found at: '{}'".format(args.load_check_point))
        else:
            print('constructing coding table')
            train_features0, train_labels0, f_map, l_map, char_count = utils.generate_corpus_char(lines[i], f_map, l_map, char_count, c_thresholds=args.mini_count, if_shrink_w_feature=False)
        
        train_features.append(train_features0)
        train_labels.append(train_labels0)

        train_features_tot += train_features0

    shrink_char_count = [k for (k, v) in iter(char_count.items()) if v >= args.mini_count]
    char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}

    char_map['<u>'] = len(char_map)  # unk for char
    char_map[' '] = len(char_map)  # concat for char
    char_map['\n'] = len(char_map)  # eof for char

    f_set = {v for v in f_map}
    dt_f_set = f_set
    f_map = utils.shrink_features(f_map, train_features_tot, args.mini_count)

    l_set = set()

    for i in range(file_num):
                   
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features[i]), dt_f_set)
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features[i]), dt_f_set)

        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels[i]), l_set)
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels[i]), l_set)

    if not args.rand_embedding:
        print("feature size: '{}'".format(len(f_map)))
        print('loading embedding')
        if args.fine_tune:  # which means does not do fine-tune
            f_map = {'<eof>': 0}
        f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', f_map, dt_f_set, args.caseless, args.unk, args.word_dim, shrink_to_corpus=args.shrink_embedding)
        print("embedding size: '{}'".format(len(f_map)))

    for label in l_set:
        if label not in l_map:
            l_map[label] = len(l_map)

    print('constructing dataset')
    for i in range(file_num):
        # construct dataset
        dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(train_features[i], train_labels[i], l_map, char_map, f_map, args.caseless)
        dev_dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(dev_features[i], dev_labels[i], l_map, char_map, f_map, args.caseless)
        test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(test_features[i], test_labels[i], l_map, char_map, f_map, args.caseless)
        
        dataset_loader.append([torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset])
        dev_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset])
        test_dataset_loader.append([torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset])

    # build model
    print('building model')
    ner_model = LM_LSTM_CRF(len(l_map), len(char_map), args.char_dim, args.char_hidden, args.char_layers, args.word_dim, args.word_hidden, args.word_layers, len(f_map), args.drop_out, file_num, large_CRF=args.small_crf, if_highway=args.high_way, in_doc_words=in_doc_words, highway_layers = args.highway_layers)

    if args.load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not args.rand_embedding:
            ner_model.load_pretrained_word_embedding(embedding_tensor)
        ner_model.rand_init(init_word_embedding=args.rand_embedding)

    if args.update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit_lm = nn.CrossEntropyLoss()
    crit_ner = CRFLoss_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

    if args.gpu >= 0:
        if_cuda = True
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        crit_ner.cuda()
        crit_lm.cuda()
        ner_model.cuda()
        packer = CRFRepack_WC(len(l_map), True)
    else:
        if_cuda = False
        packer = CRFRepack_WC(len(l_map), False)

    tot_length = sum(map(lambda t: len(t), dataset_loader))

    best_f1 = []
    for i in range(file_num):
        best_f1.append(float('-inf'))

    best_pre = []
    for i in range(file_num):
        best_pre.append(float('-inf'))

    best_rec = []
    for i in range(file_num):
        best_rec.append(float('-inf'))

    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    evaluator = eval_wc(packer, l_map, args.eva_matrix)

    predictor = predict_wc(if_cuda, f_map, char_map, l_map, f_map['<eof>'], char_map['\n'], l_map['<pad>'], l_map['<start>'], True, args.batch_size, args.caseless) #NEW

    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        sample_num = 1

        epoch_loss = 0
        ner_model.train()

        for sample_id in tqdm( range(sample_num) , mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stdout):

            file_no = random.randint(0, file_num-1)            
            cur_dataset = dataset_loader[file_no]
            
            for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in itertools.chain.from_iterable(cur_dataset):

                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)

                ner_model.zero_grad()
                scores = ner_model(f_f, f_p, b_f, b_p, w_f, file_no)
                loss = crit_ner(scores, tg_v, mask_v)

                epoch_loss += utils.to_scalar(loss)
                if args.co_train:
                    cf_p = f_p[0:-1, :].contiguous()
                    cb_p = b_p[1:, :].contiguous()
                    cf_y = w_f[1:, :].contiguous()
                    cb_y = w_f[0:-1, :].contiguous()
                    cfs, _ = ner_model.word_pre_train_forward(f_f, cf_p)
                    loss = loss + args.lambda0 * crit_lm(cfs, cf_y.view(-1))
                    cbs, _ = ner_model.word_pre_train_backward(b_f, cb_p)
                    loss = loss + args.lambda0 * crit_lm(cbs, cb_y.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm(ner_model.parameters(), args.clip_grad)
                optimizer.step()
        
        epoch_loss /= tot_length

        # update lr
        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        # eval & save check_point
        if 'f' in args.eva_matrix:
            dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(ner_model, dev_dataset_loader[file_no], file_no)

            if dev_f1 > best_f1[file_no]:
                patience_count = 0
                best_f1[file_no] = dev_f1
                best_pre[file_no] = dev_pre
                best_rec[file_no] = dev_rec

                test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(ner_model, test_dataset_loader[file_no], file_no)

                track_list.append(
                    {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'test_f1': test_f1,
                     'test_acc': test_acc})

                print(
                    '(loss: %.4f, epoch: %d, dataset: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, F1 on test = %.4f, pre on test= %.4f, rec on test= %.4f), saving...' %
                    (epoch_loss,
                     args.start_epoch,
                     file_no,
                     dev_f1,
                     dev_pre,
                     dev_rec,
                     test_f1,
                     test_pre,
                     test_rec))

                if args.output_annotation: #NEW
                    print('annotating')
                    with open('output'+str(file_no)+'.txt', 'w') as fout:
                        predictor.output_batch(ner_model, test_word[file_no], fout, file_no)

                try:
                    utils.save_checkpoint({
                        'epoch': args.start_epoch,
                        'state_dict': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': l_map,
                        'c_map': char_map,
                        'in_doc_words': in_doc_words
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, args.checkpoint + 'cwlm_lstm_crf')
                except Exception as inst:
                    print(inst)

            else:
                patience_count += 1
                print('(loss: %.4f, epoch: %d, dataset: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f)' %
                      (epoch_loss,
                       args.start_epoch,
                       file_no,
                       dev_f1,
                       dev_pre,
                       dev_rec))
                track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= args.patience and args.start_epoch >= args.least_iters:
            break
