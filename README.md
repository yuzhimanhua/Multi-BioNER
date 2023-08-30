# Cross-type Biomedical Named Entity Recognition with Deep Multi-task Learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the source code for [**Cross-type Biomedical Named Entity Recognition with Deep Multi-Task Learning**](https://arxiv.org/abs/1801.09851).

The implementation is based on the PyTorch library. Our model collectively trains different biomedical entity types to build a unified model that benefits the training of each single entity type and achieves a significantly better performance compared with the state-of-the-art BioNER systems.

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Running](#running)
- [Benchmarks](#benchmarks)
- [Prediction](#prediction)
- [Citation](#citation)

## Installation

For training, a GPU is strongly recommended.

#### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/). 

#### Dependencies

The code is written in Python 3.6. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:
```
pip3 install -r requirements.txt
```
**Note: PyTorch 0.3.1 and CUDA 9.0 were used during development. If your PyTorch version is 0.4.0 or higher, the code may not work.**

## Quick Start

To reproduce the results in our [paper](https://arxiv.org/abs/1801.09851), you can first download the corpora and the embedding file **[here](https://drive.google.com/file/d/1JHQJ9DKaEeSGZdA0Nmz9KCdtjUoJKXCb/view?usp=sharing)**, unzip the folder ```data_bioner_5/``` and put it under the main folder ```./```. Then the following running script can be used to run the model.
```
./run_lm-lstm-crf5.sh
```

## Data

We use five biomedical corpora collected by Crichton et al. for biomedical NER. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). The details of each dataset are listed below:

|Dataset | Entity Type | Dataset Size | 
| ------------- |-------------| -----|
| BC2GM | Gene/Protein | 20,000 sentences |
| BC4CHEMD | Chemical | 10,000 abstracts |
| BC5CDR | Chemical, Disease | 1,500 articles |
| NCBI-disease | Disease | 793 abstracts |
| JNLPBA | Gene/Protein, DNA, Cell Type, Cell Line, RNA | 2,404 abstracts |

#### Note
**In our paper, we merge the original training set and development set to be the new training set, as many teams did in the challenge. Some previous work (e.g., [Luo et al., Bioinformatics 2017](https://github.com/lingluodlut/Att-ChemdNER), [Lu et al., Journal  of
cheminformatics 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4331694/pdf/1758-2946-7-S1-S4.pdf) and [Leaman and Lu, Bioinformatics 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5018376/pdf/btw343.pdf)) also preprocessed data in this way. If you want to reproduce our results, please follow the same way.**

#### Format

Users may want to use other datasets. We assume the corpus is formatted as same as the CoNLL 2003 NER dataset.

More specifically, **empty lines** are used as separators between sentences, and the separator between documents is a special line as below.
```
-DOCSTART- -X- -X- -X- O
```
Other lines contains words, labels and other fields. **Word** must be the **first** field, **label** must be the **last**. For example,
```
-DOCSTART- -X- -X- -X- O

Selegiline	S-Chemical
-	O
induced	O
postural	B-Disease
hypotension	E-Disease
in	O
Parkinson	B-Disease
'	I-Disease
s	I-Disease
disease	E-Disease
:	O
a	O
longitudinal	O
study	O
on	O
the	O
effects	O
of	O
drug	O
withdrawal	O
.	O
```

#### Embedding
We initialize the word embedding matrix with pre-trained word vectors from Pyysalo et al., 2013. These word vectors are
trained using the skip-gram model on the PubMed abstracts together with all the full-text articles
from PubMed Central (PMC) and a Wikipedia dump. You can download the embedding file [here](https://drive.google.com/file/d/1JHQJ9DKaEeSGZdA0Nmz9KCdtjUoJKXCb/view?usp=sharing).

## Running

```train_wc.py``` is the script for our multi-task LSTM-CRF model.
The usages of it can be accessed by
```
python train_wc.py -h
```

The default running commands are:
```
python3 train_wc.py --train_file [training file 1] [training file 2] ... [training file N] \
                    --dev_file [developing file 1] [developing file 2] ... [developing file N] \
                    --test_file [testing file 1] [testing file 2] ... [testing file N] \
                    --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200
```

Users may incorporate an arbitrary number of corpora into the training process. In each epoch, our model randomly selects one dataset _i_. We use training set _i_ to learn the parameters and developing set _i_ to evaluate the performance. If the current model achieves the best performance for dataset _i_ on the developing set, we will then calculate the precision, recall and F1 on testing set _i_.

## Benchmarks

Here we compare our model with recent state-of-the-art models on the five datasets mentioned above. We use F1 score as the evaluation metric.

|Model | [BC2GM](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOBES) | [BC4CHEMD](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC4CHEMD-IOBES) | [BC5CDR](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-IOBES) | [NCBI-disease](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/NCBI-disease-IOBES) | [JNLPBA](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/JNLPBA-IOBES) |
| ------------- |-------------| -----| -----| -----| ---- |
| Dataset Benchmark | - | 88.06 | 86.76 | 82.90 | 72.55 |
| [Crichton et al. 2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) | 73.17 | 83.02 | 83.90 | 80.37 | 70.09 |
| [Lample et al. 2016](https://github.com/glample/tagger) | 80.51 | 87.74 | 86.92 | 85.80 | 73.48 |
| [Ma and Hovy 2016](https://github.com/XuezheMax/LasagneNLP) | 78.48 | 86.84 | 86.65 | 82.62 | 72.68 |
| [Liu et al. 2018](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF) | 80.00 | 88.75 | 86.96 | 83.92 | 72.17 |
| Our Model | **80.74** | **89.37** | **88.78** | **86.14** | **73.52** |


## Prediction
Our ```train_wc.py``` provides an option to directly output the annotation results during the training process by the parameter ````--output_annotation````, i.e.,
```
python3 train_wc.py --train_file [training file 1] [training file 2] ... [training file N] \
                    --dev_file [developing file 1] [developing file 2] ... [developing file N] \
                    --test_file [testing file 1] [testing file 2] ... [testing file N] \
                    --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --output_annotation --word_dim 200 --gpu 0
```

If users do not use ````--output_annotation````, the best performing model during the training process will be saved in ```./checkpoint/```. 

#### Pre-trained Model
**We have released our pre-trained model. You can download the [Arg](https://drive.google.com/file/d/1CxW75H1NwnUCfnBVWQFdZD9TNbuayUAQ/view?usp=sharing) file and the [Model](https://drive.google.com/file/d/1aBoIUDzU6_DcB0c1Y1t0AoKmcVik0YO1/view?usp=sharing) file and put them in ```./checkpoint/```.**

Using the saved model, ```seq_wc.py``` can be applied to annotate raw text. Its usage can be accessed by command 
```
python seq_wc.py -h
```
and a running command example is provided below:
```
python3 seq_wc.py --load_arg checkpoint/cwlm_lstm_crf.json --load_check_point checkpoint/cwlm_lstm_crf.model --input_file test.tsv --output_file annotate/output --gpu 0
```
The annotation results will be in ```./annotate/```.

The input format is similar to CoNLL, but each line is required to contain only one field, token. For example, an input file could be:
```
The
severe
anemia
(
hemoglobin
1
.
2
g
/
dl
)
appeared
to
be
the
primary
etiologic
factor
.
```
and the corresponding output is:
```
The O
severe O
anemia O
( O
hemoglobin B-GENE
1 I-GENE
. I-GENE
2 I-GENE
g I-GENE
/ I-GENE
dl E-GENE
) O
appeared O
to O
be O
the O
primary O
etiologic O
factor O
. O 
```

## Citation
If you find this repository useful, please cite the following paper:
```
@article{wang2018cross,
  title={Cross-type biomedical named entity recognition with deep multi-task learning},
  author={Wang, Xuan and Zhang, Yu and Ren, Xiang and Zhang, Yuhao and Zitnik, Marinka and Shang, Jingbo and Langlotz, Curtis and Han, Jiawei},
  journal={Bioinformatics},
  volume={35},
  number={10},
  pages={1745--1752},
  year={2019},
  publisher={Oxford University Press}
}
```
