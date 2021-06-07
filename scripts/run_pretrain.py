import argparse
from distutils.util import strtobool as _bool
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import numpy as np
import random
from tqdm.auto import tqdm
import pickle
import gc
import sys
import os
sys.path.append(os.getcwd())
from src.functions import *


############################## Argparse ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='56789')
# parser.add_argument('--max_epoch', type=int, default=)
# parser.add_argument('--stack_num', type=int, default=)
# parser.add_argument('--batch_size', type=int, default=)
parser.add_argument('--max_seq_len', type=int, default=256)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--att_head_num', type=int, default=12)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--initializer_range', type=float, default=0.2)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--restart', type=_bool, default=False)
parser.add_argument('--restart_epoch', type=int, default=0)
args = parser.parse_args()

log_dir = f'log/bert_{args.max_seq_len}seq_{args.d_model}H_{args.layer_num}L_{args.att_head_num}A'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f)


############################## Hyperparameter ##############################
# load vocabulary
with open('tokenizer/pretrain_all_30k.vocab', 'r', encoding='utf8') as f:
    data = f.readlines()
    vocab = []
    for word in data:
        vocab.append(word.split('\t')[0])
inverse_vocab = {}
for i, word in enumerate(vocab):
    inverse_vocab[word] = i
V = len(vocab)


# define symbol
PAD = '<pad>'
UNK = '<unk>'
MASK = '[MASK]'
CLS = '[CLS]'
SEP = '[SEP]'

pad_id = inverse_vocab['<pad>']
unk_id = inverse_vocab['<unk>']
mask_id = inverse_vocab['[MASK]']
cls_id = inverse_vocab['[CLS]']
sep_id = inverse_vocab['[SEP]']

# set hyperparameter
d_ff = int(args.d_model * 4)
warmup_steps = 10000
beta1 = 0.9
beta2 = 0.999
l2_weight_decay = 0.01
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


############################## Tensorboard ##############################
tb_dir = os.path.join(log_dir, 'tb')
ckpt_dir = os.path.join(log_dir, 'ckpt')
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
tb_writer = SummaryWriter(tb_dir)


############################## Input data ##############################
print("Loading pre-train Datasets...", end=' ')
print("pairs...", end=' ')
with open('datasets/preprocessed/huggingface/concatenated/concatenated.pair.pkl', 'rb') as fr:
    pairs = pickle.load(fr)
print("soles...", end=' ')
with open('datasets/preprocessed/huggingface/concatenated/concatenated.sole.pkl', 'rb') as fr:
    soles = pickle.load(fr)
print("pool...", end=' ')
with open('datasets/preprocessed/huggingface/concatenated/concatenated.pool.pkl', 'rb') as fr:
    pool = pickle.load(fr)
print('Complete..!')

IsNext_pairs, NotNext_pairs = load_pairs()
model_file = 'tokenizer/pretrain_all_30k.model'
tokenizer = spm.SentencePieceProcessor(model_file=model_file)


############################## Pretrain ##############################
