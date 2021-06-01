"""
.. module:: crf
    :synopsis: conditional random field
 
.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.sparse as sparse
import model.utils as utils


class CRF_L(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Ma et al. 2016, has more parameters than CRF_S
 
    args: 
        hidden_dim : input dim size 
        tagset_size: target_set_size 
        if_biase: whether allow bias in linear trans    
    """
    

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_L, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size, bias=if_bias)

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        return self.hidden2tag(feats).view(-1, self.tagset_size, self.tagset_size)


class CRF_S(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args: 
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
 
    """
    
    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_S, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        
        ins_num = feats.size(0)
        scores = self.hidden2tag(feats)
        crf_scores = scores.view(-1, self.tagset_size, 1).expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores

class CRFRepack:
    """Packer for word level model
    
    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
    """

    def __init__(self, tagset_size, if_cuda):
        
        self.tagset_size = tagset_size
        self.if_cuda = if_cuda

    def repack_vb(self, feature, target, mask):
        """packer for viterbi loss

        args: 
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
            mask (Seq_len, Batch_size): padding mask
        return:
            feature (Seq_len, Batch_size), target (Seq_len, Batch_size), mask (Seq_len, Batch_size)
        """
        
        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            tg_v = autograd.Variable(target.transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = autograd.Variable(mask.transpose(0, 1)).cuda()
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            tg_v = autograd.Variable(target.transpose(0, 1)).contiguous().unsqueeze(2)
            mask_v = autograd.Variable(mask.transpose(0, 1)).contiguous()
        return fea_v, tg_v, mask_v

    def repack_gd(self, feature, target, current):
        """packer for greedy loss

        args: 
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
            current (Seq_len, Batch_size): current state
        return:
            feature (Seq_len, Batch_size), target (Seq_len * Batch_size), current (Seq_len * Batch_size, 1, 1)
        """
        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            ts_v = autograd.Variable(target.transpose(0, 1)).cuda().view(-1)
            cs_v = autograd.Variable(current.transpose(0, 1)).cuda().view(-1, 1, 1)
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            ts_v = autograd.Variable(target.transpose(0, 1)).contiguous().view(-1)
            cs_v = autograd.Variable(current.transpose(0, 1)).contiguous().view(-1, 1, 1)
        return fea_v, ts_v, cs_v

    def convert_for_eval(self, target):
        """convert target to original decoding

        args: 
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size


class CRFRepack_WC:
    """Packer for model with char-level and word-level

    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
        
    """

    def __init__(self, tagset_size, if_cuda):
        
        self.tagset_size = tagset_size
        self.if_cuda = if_cuda

    def repack_vb(self, f_f, f_p, b_f, b_p, w_f, target, mask, len_b):
        """packer for viterbi loss

        args: 
            f_f (Char_Seq_len, Batch_size) : forward_char input feature 
            f_p (Word_Seq_len, Batch_size) : forward_char input position
            b_f (Char_Seq_len, Batch_size) : backward_char input feature
            b_p (Word_Seq_len, Batch_size) : backward_char input position
            w_f (Word_Seq_len, Batch_size) : input word feature
            target (Seq_len, Batch_size) : output target
            mask (Word_Seq_len, Batch_size) : padding mask
            len_b (Batch_size, 2) : length of instances in one batch
        return:
            f_f (Char_Reduced_Seq_len, Batch_size), f_p (Word_Reduced_Seq_len, Batch_size), b_f (Char_Reduced_Seq_len, Batch_size), b_p (Word_Reduced_Seq_len, Batch_size), w_f (size Word_Seq_Len, Batch_size), target (Reduced_Seq_len, Batch_size), mask  (Word_Reduced_Seq_len, Batch_size)

        """
        mlen, _ = len_b.max(0)
        mlen = mlen.squeeze()
        ocl = b_f.size(1)
        if self.if_cuda:
            f_f = autograd.Variable(f_f[:, 0:mlen[0]].transpose(0, 1)).cuda()
            f_p = autograd.Variable(f_p[:, 0:mlen[1]].transpose(0, 1)).cuda()
            b_f = autograd.Variable(b_f[:, -mlen[0]:].transpose(0, 1)).cuda()
            b_p = autograd.Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1)).cuda()
            w_f = autograd.Variable(w_f[:, 0:mlen[1]].transpose(0, 1)).cuda()
            tg_v = autograd.Variable(target[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = autograd.Variable(mask[:, 0:mlen[1]].transpose(0, 1)).cuda()
        else:
            f_f = autograd.Variable(f_f[:, 0:mlen[0]].transpose(0, 1))
            f_p = autograd.Variable(f_p[:, 0:mlen[1]].transpose(0, 1))
            b_f = autograd.Variable(b_f[:, -mlen[0]:].transpose(0, 1))
            b_p = autograd.Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1))
            w_f = autograd.Variable(w_f[:, 0:mlen[1]].transpose(0, 1))
            tg_v = autograd.Variable(target[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2)
            mask_v = autograd.Variable(mask[:, 0:mlen[1]].transpose(0, 1))
        return f_f, f_p, b_f, b_p, w_f, tg_v, mask_v

    def convert_for_eval(self, target):
        """convert for eval

        args: 
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size


class CRFLoss_gd(nn.Module):
    """loss for greedy decode loss, i.e., although its for CRF Layer, we calculate the loss as 

    .. math::
        \sum_{j=1}^n \log (p(\hat{y}_{j+1}|z_{j+1}, \hat{y}_{j}))

    instead of 
    
    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
    
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_gd, self).__init__()
        self.tagset_size = tagset_size
        self.average_batch = average_batch
        self.crit = nn.CrossEntropyLoss(size_average=self.average_batch)

    def forward(self, scores, target, current):
        """
        args: 
            scores (Word_Seq_len, Batch_size, target_size_from, target_size_to): crf scores
            target (Word_Seq_len, Batch_size): golden list
            current (Word_Seq_len, Batch_size): current state
        return:
            crf greedy loss
        """
        ins_num = current.size(0)
        current = current.expand(ins_num, 1, self.tagset_size)
        scores = scores.view(ins_num, self.tagset_size, self.tagset_size)
        current_score = torch.gather(scores, 1, current).squeeze()
        return self.crit(current_score, target)


class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        # print(scores)
        # print(target)
        # print(mask)

        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # calculate sentence score
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
                  # (bat_size * from_target * to_target) -> (bat_size * to_target)
            
            partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            # partition = utils.switch(partition, cur_partition,
            #                          mask[idx].contiguous().view(bat_size, 1).expand(bat_size, self.tagset_size)).contiguous().view(bat_size, -1)
            # the following two may achieve higher speed, but raise run-time error
            # new_partition = partition.clone()
            # new_partition.masked_scatter_(mask[idx].view(-1, 1).expand(bat_size, self.tagset_size), cur_partition)  #0 for partition, 1 for cur_partition
            # partition = new_partition
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = (partition - tg_energy)

        return loss

class CRFDecode_vb():
    """Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer
        return decode_idx