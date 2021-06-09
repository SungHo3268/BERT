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


def get_seg_embedding(inputs, sep_id, gpu, cuda):
    """
    tensor = (batch_size, seq_len) == target or source input
    """
    seg_embedding = torch.zeros_like(inputs)
    if gpu:
        seg_embedding = seg_embedding.to(torch.device(f'cuda:{cuda}'))
    for i, sentence in enumerate(inputs):     # per sentence
        if sep_id in sentence:
            sep = torch.where(sentence == sep_id)[0]
            if len(sep) == 1:
                seg_embedding[i][sep[0]+1:] += 1
            else:
                seg_embedding[i][sep[0]+1: sep[1]+1] += 1
    return seg_embedding.unsqueeze(-1)


def count_tokens(corpus):
    count = 0
    for sequence in corpus:
        count += len(sequence)
    return count


def replace_mask(inputs, vocab_size=30000, cls_id=2, sep_id=3, mask_id=4):
    """
    inputs = list[tensor([ids]),
                  tensor([ids]),
                    ...,
                  tensor([ids])]
    """
    mask_loc = []
    mask_label = []
    for i, sequence in enumerate(inputs):       # sequence = tensor([id1, id2, ... ])
        for j, token in enumerate(sequence):
            if token == 0:
                break
            elif (token != cls_id) and (token != sep_id):
                prob = np.random.random()
                if prob >= 0.15:
                    continue
                elif prob >= 0.135:  # random   1.5%
                    mask_loc.append((i, j))
                    mask_label.append(int(token))
                    inputs[i][j] = torch.as_tensor(np.random.randint(low=5, high=vocab_size, size=1, dtype=int),
                                                   dtype=torch.int32)
                elif prob >= 0.12:   # same     1.5%
                    mask_loc.append((i, j))
                    mask_label.append(int(token))
                else:               # MASK      12%
                    mask_loc.append((i, j))
                    mask_label.append(int(token))
                    inputs[i][j] = torch.as_tensor(mask_id, dtype=torch.int32)
    return mask_loc, mask_label


def collate_fn(data):
    """
    :param data: the batch input comprised of tuples which is like (input, cls_label),      data = (batch, 2, seq_len)
    :return: corpus = list[tensor([ids]),
                           tensor([ids]),
                           ...,
                           tensor([ids])]
             c_labels = list[tensor([label1, ...])]
             mask_loc
    """
    max_seq_len = 128
    corpus = []
    c_labels = []
    for inputs, label in data:      # per batch
        """
        inputs = list[id1, id2, id3, ...]
        label = scalar
        """
        inputs += [0] * (max_seq_len - len(inputs))
        corpus.append(torch.tensor(inputs, dtype=torch.int32))
        c_labels.append(torch.tensor(label, dtype=torch.int32))
    mask_loc, mask_label = replace_mask(corpus)
    return corpus, c_labels, mask_loc, mask_label


def get_mask_out(mlm_out, mask_loc):
    mask_out = []
    for loc in mask_loc:
        mask_hidden = mlm_out[loc]      # (3000, )
        mask_out.append(mask_hidden)
    return torch.stack(mask_out, dim=0)                     # (mask_num, 3000)
