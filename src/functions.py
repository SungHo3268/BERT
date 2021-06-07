import torch
import pickle
from tqdm.auto import tqdm
import numpy as np
import sys
import os
sys.path.append(os.getcwd())


def load_pairs():
    log_dir = 'datasets/preprocessed/huggingface/concatenated/'

    print("Loading pre-train Datasets...", end=' ')
    print("IsNext pairs...", end=' ')
    with open(os.path.join(log_dir, 'IsNext_pairs.pkl'), 'rb') as fr:
        IsNext_pairs = pickle.load(fr)
    print("NotNext pairs...", end=' ')
    with open(os.path.join(log_dir, 'NotNext_pairs.pkl'), 'rb') as fr:
        NotNext_pairs = pickle.load(fr)
    print('Complete..!')

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


def convert_to_ids(pairs, cls_id, sep_id, max_seq_len, tokenizer):
    corpus = []
    for pair in tqdm(pairs, total=len(pairs), desc="Convert string to ids...", bar_format='{l_bar}{r_bar}'):
        temp = [cls_id]
        temp.extend(tokenizer.encode(pair[0]))
        temp.append(sep_id)
        temp.extend(tokenizer.encode(pair[1]))
        if len(temp) < 128:
            pad_len = 128-len(temp)
            temp += [tokenizer.pad_id]*pad_len
        else:
            temp = temp[:max_seq_len]
        corpus.append(temp)
    return np.array(corpus)
