import torch
import pickle
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


def count_tokens(corpus):
    mask = np.zeros_like(corpus)
    eq = np.not_equal(corpus, mask)
    token_nums = np.sum(eq)
    return token_nums


def replace_mask(inputs, vocab_size=30000, cls_id=2, sep_id=3, mask_id=4):
    mask_loc = []
    mask_label = []
    for i, sequence in enumerate(inputs):
        for j, token in enumerate(sequence):
            if (token != 0) and (token != cls_id) and (token != sep_id):
                prob = np.random.random()
                if prob >= 0.15:
                    continue
                elif prob >= 0.135:  # random   1.5%
                    mask_loc.append([i, j])
                    mask_label.append(token)
                    inputs[i, j] = np.random.randint(low=5, high=vocab_size, size=1, dtype=int)
                elif prob >= 0.12:   # same     1.5%
                    mask_loc.append([i, j])
                    mask_label.append(token)
                else:               # MASK      12%
                    mask_loc.append([i, j])
                    mask_label.append(token)
                    inputs[i, j] = mask_id
    return None
