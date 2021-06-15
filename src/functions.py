import torch
from datasets import load_dataset
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
    # inputs are replaced

    inputs = list[tensor([ids]),
                  tensor([ids]),
                    ...,
                  tensor([ids])]

             tensor([ids]) = (128, )
    """
    mask_loc = torch.zeros_like(inputs)
    mask_label = torch.full_like(input=inputs, fill_value=-100, dtype=torch.long)
    for i, sequence in enumerate(inputs):       # sequence = tensor([id1, id2, ... ])
        for j, token in enumerate(sequence):
            if token == 0:
                break
            elif (token != cls_id) and (token != sep_id):
                prob = np.random.random()
                if prob >= 0.15:
                    continue
                elif prob >= 0.135:  # random   1.5%
                    mask_loc[i][j] = 1
                    mask_label[i][j] = token
                    inputs[i][j] = torch.as_tensor(np.random.randint(low=5, high=vocab_size, size=1, dtype=int),
                                                   dtype=torch.long)
                elif prob >= 0.12:   # same     1.5%
                    mask_loc[i][j] = 1
                    mask_label[i][j] = token
                else:               # MASK      12%
                    mask_loc[i][j] = 1
                    mask_label[i][j] = token
                    inputs[i][j] = mask_id
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
        corpus.append(torch.tensor(inputs, dtype=torch.long))
        c_labels.append(torch.tensor(label, dtype=torch.long))
    corpus = torch.stack(corpus, dim=0)
    c_labels = torch.stack(c_labels, dim=0)
    mask_loc, mask_label = replace_mask(corpus)
    return corpus, c_labels, mask_loc, mask_label


def get_mask_out(mlm_out, mask_loc):
    mask_out = []
    for loc in mask_loc:
        mask_hidden = mlm_out[loc]      # (3000, )
        mask_out.append(mask_hidden)
    return torch.stack(mask_out, dim=0)                     # (mask_num, 3000)


def load_ft_dataset(task, tokenizer, cls_id, sep_id, max_seq_len):
    dataset = load_dataset('glue', task)

    single_dataset = ['sst2', 'cola']
    pair_dataset = ['mnli', 'mnli_matched', 'mnli_mismatched', 'qqp', 'qnli', 'stsb', 'mrpc', 'rte']

    if task in single_dataset:
        print('Loading the train dataset...')
        train_inputs, train_labels = dataset['train']['sentence'], dataset['train']['label']
        train_inputs, train_labels = preprocess_ft_dataset(train_inputs, train_labels, tokenizer,
                                                           cls_id, sep_id, task, max_seq_len)
        print('Loading the validation dataset...')
        val_inputs, val_labels = dataset['validation']['sentence'], dataset['validation']['label']
        val_inputs, val_labels = preprocess_ft_dataset(val_inputs, val_labels, tokenizer,
                                                       cls_id, sep_id, task, max_seq_len)
        return train_inputs, train_labels, val_inputs, val_labels

    elif task in pair_dataset:
        print('Loading the train dataset...')
        if task == 'qnli':
            question, sentence, train_labels = dataset['train']['question'], \
                                               dataset['train']['sentence'], \
                                               dataset['train']['label']
            train_inputs = [question, sentence]
        elif task == 'qqp':
            question1, question2, train_labels = dataset['train']['question1'], \
                                                 dataset['train']['question2'], \
                                                 dataset['train']['label']
            train_inputs = [question1, question2]
        elif task == 'mnli':
            premise, hypothesis, train_labels = dataset['train']['premise'], \
                                                dataset['train']['hypothesis'], \
                                                dataset['train']['label']
            train_inputs = [premise, hypothesis]
        else:
            train_input1, train_input2, train_labels = dataset['train']['sentence1'], \
                                                       dataset['train']['sentence2'], \
                                                       dataset['train']['label']
            train_inputs = [train_input1, train_input2]
        train_inputs, train_labels = preprocess_ft_dataset(train_inputs, train_labels, tokenizer,
                                                           cls_id, sep_id, task, max_seq_len)

        print('Loading the validation dataset...')
        if task == 'qnli':
            question, sentence, val_labels = dataset['validation']['question'], \
                                             dataset['validation']['sentence'], \
                                             dataset['validation']['label']
            val_inputs = [question, sentence]
            val_inputs, val_labels = preprocess_ft_dataset(val_inputs, val_labels, tokenizer,
                                                           cls_id, sep_id, task, max_seq_len)
            return train_inputs, train_labels, val_inputs, val_labels
        elif task == 'qqp':
            question1, question2, val_labels = dataset['validation']['question1'], \
                                               dataset['validation']['question2'], \
                                               dataset['validation']['label']
            val_inputs = [question1, question2]
            val_inputs, val_labels = preprocess_ft_dataset(val_inputs, val_labels, tokenizer,
                                                           cls_id, sep_id, task, max_seq_len)
            return train_inputs, train_labels, val_inputs, val_labels
        elif task == 'mnli':
            m_premise, m_hypothesis, m_val_labels = dataset['validation_matched']['premise'], \
                                                    dataset['validation_matched']['hypothesis'], \
                                                    dataset['validation_matched']['label']
            val_matched_inputs = [m_premise, m_hypothesis]

            mm_premise, mm_hypothesis, mm_val_labels = dataset['validation_mismatched']['premise'], \
                                                       dataset['validation_mismatched']['hypothesis'], \
                                                       dataset['validation_mismatched']['label']
            val_mismatched_inputs = [mm_premise, mm_hypothesis]
            val_matched_inputs, val_matched_labels = preprocess_ft_dataset(val_matched_inputs, m_val_labels, tokenizer,
                                                                           cls_id, sep_id, task, max_seq_len)
            val_mismatched_inputs, val_mismatched_labels = preprocess_ft_dataset(val_mismatched_inputs, mm_val_labels,
                                                                                 tokenizer, cls_id, sep_id, task,
                                                                                 max_seq_len)
            return train_inputs, train_labels, val_matched_inputs, val_matched_labels, \
                   val_mismatched_inputs, val_mismatched_labels

        else:
            val_input1, val_input2, val_labels = dataset['validation']['sentence1'], \
                                                 dataset['validation']['sentence2'], \
                                                 dataset['validation']['label']
            val_inputs = [val_input1, val_input2]
            val_inputs, val_labels = preprocess_ft_dataset(val_inputs, val_labels, tokenizer,
                                                           cls_id, sep_id, task, max_seq_len)
            return train_inputs, train_labels, val_inputs, val_labels


def preprocess_ft_dataset(inputs, labels, tokenizer, cls_id, sep_id, task, max_seq_len):
    # preprocessing
    single_dataset = ['sst2', 'cola']
    pair_dataset = ['mnli', 'mnli_matched', 'mnli_mismatched', 'qqp', 'qnli', 'stsb', 'mrpc', 'rte']
    print("Preprocess dataset (Convert_to_id, CLS, SEP, Padding, Shuffling)...")
    truncated = 0
    new_input = []
    if task in single_dataset:
        for i, sentence in enumerate(inputs):
            temp = [cls_id]
            temp += tokenizer.EncodeAsIds(sentence)
            if len(temp) <= max_seq_len:
                temp += [0] * (max_seq_len - len(temp))
            else:
                temp = temp[:max_seq_len]
                truncated += 1
            new_input.append(temp)
    elif task in pair_dataset:
        sentences1 = inputs[0]
        sentences2 = inputs[1]
        for sentence1, sentence2 in zip(sentences1, sentences2):
            temp = [cls_id]
            temp += tokenizer.EncodeAsIds(sentence1)
            temp += [sep_id]
            temp += tokenizer.EncodeAsIds(sentence2)
            if len(temp) <= max_seq_len:
                temp += [0] * (max_seq_len - len(temp))
            else:
                temp = temp[:max_seq_len]
                truncated += 1
            new_input.append(temp)
    if truncated > 0:
        print(f"{truncated}/{len(new_input)} sentences are truncated by {max_seq_len}seq_len.")
    else:
        print("There is no truncated sentence...", end=' ')

    # Shuffling
    per = np.arange(len(new_input))
    new_input = np.array(new_input)[per]
    labels = np.array(labels)[per]
    print('Complete.\n')
    return new_input, labels

