import numpy as np
import torch
import pickle
from tqdm.auto import tqdm
import sentencepiece as spm
import sys
import os
sys.path.append(os.getcwd())


def load_pairs():
    log_dir = 'datasets/preprocessed/huggingface/concatenated/'
    with open(os.path.join(log_dir, 'IsNext_pairs.pkl'), 'rb') as fr:
        IsNext_pairs = pickle.load(fr)
    with open(os.path.join(log_dir, 'NotNext_pairs.pkl'), 'rb') as fr:
        NotNext_pairs = pickle.load(fr)
    return IsNext_pairs, NotNext_pairs


def get_pad_mask(tensor, gpu, cuda):
    """
    tensor = (batch_size, seq_len) == target or source input
    """
    batch_size, seq_len = tensor.size()
    zero_pad_mask = torch.zeros_like(tensor)
    if gpu:
        zero_pad_mask = zero_pad_mask.to(torch.device(f'cuda:{cuda}'))
    zero_pad_mask = torch.eq(tensor, zero_pad_mask)
    return zero_pad_mask.view(batch_size, 1, 1, seq_len)        # zero_pad_mask = (batch_size, 1, 1, seq_len)


def get_seg_embedding(tensor, sep_id, gpu, cuda):
    """
    tensor = (batch_size, seq_len) == target or source input
    """
    seg_embedding = torch.zeros_like(tensor)
    if gpu:
        seg_embedding = seg_embedding.to(torch.device(f'cuda:{cuda}'))
    for i, sentence in enumerate(tensor):     # per sentence
        if sep_id in sentence:
            sep = int(torch.where(sentence == sep_id)[0])
            seg_embedding[i][sep+1:] += 1
    return seg_embedding
