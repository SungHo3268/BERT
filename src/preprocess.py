from datasets import load_from_disk
import pickle
from tqdm.auto import tqdm
import random
import sys
import os
sys.path.append(os.getcwd())


def split_to_article():
    data_list = ['wiki', 'bookcorpus', 'concatenated']
    for data in data_list:
        # load_ dataset
        print(f"Loading {data} dataset...")
        pretrain_dataset = load_from_disk(f'datasets/raw/huggingface/{data}')
        dataset = pretrain_dataset['text']

        # split the dataset based on '\n' and discard the sentences less then 10 tokens and more than 4192 tokens.
        split_dataset = []
        for article in tqdm(dataset, total=len(dataset), desc="Split the article based on '\\n'...",
                            bar_format='{l_bar}{bar}{r_bar}'):
            document = [sentence for sentence in article.split('\n')
                        if (len(sentence.split()) >= 10) and (len(sentence.split()) <= 4192)]
            if len(document) == 0:
                continue
            split_dataset.append(document)

        # count the number of sentences
        sen_count = 0
        for document in tqdm(split_dataset, desc="Counting #senteces: "):
            sen_count += len(document)
        print(sen_count, " sentences")

        # count the number of tokens
        token_count = 0
        for document in tqdm(split_dataset, desc="Counting #tokens: "):
            for sen in document:
                token_count += len(sen.split())
        print(token_count, " tokens")

        # Save the split_dataset as type of pickle
        with open(f'datasets/preprocessed/huggingface/{data}/{data}.split.pkl', 'wb') as f:
            pickle.dump(split_dataset, f)

        if data == 'concatenated':
            # Save the split_dataset as type of txt
            with open(f'tokenizer/{data}.split.forSPM.txt', 'w') as f:
                for document in tqdm(split_dataset, total=len(split_dataset), desc='Saving to txt...'):
                    for line in document:
                        f.write(line + ' \n')


def make_pair():
    # load wiki
    with open('datasets/preprocessed/huggingface/wiki/wiki.split.pkl', 'rb') as fr:
        wiki_dataset = pickle.load(fr)

    # load bookcorpus
    with open('datasets/preprocessed/huggingface/bookcorpus/bookcorpus.split.pkl', 'rb') as fr:
        bookcorpus_dataset = pickle.load(fr)

    # make next_pair in each document-level
    pairs = []
    soles = []
    for document in tqdm(wiki_dataset, desc='Pairing for wiki dataset...', bar_format='{l_bar}{r_bar}'):
        if len(document) == 1:
            soles.append(document[0])
        else:
            soles.append(document[-1])
            for i in range(len(document)-1):
                pair = [document[i], document[i+1]]
                pairs.append(pair)
    for i in tqdm(range(len(bookcorpus_dataset)-1), total=len(bookcorpus_dataset)-1,
                  desc='Pairing for bookcorpus...', bar_format='{l_bar}{r_bar}'):
        pair = [bookcorpus_dataset[i][0], bookcorpus_dataset[i+1][0]]
        pairs.append(pair)
    soles.append(bookcorpus_dataset[-1][0])

    pool = soles.copy()
    for pair in tqdm(pairs, desc='Making a whole pool...', total=len(pairs), bar_format='{l_bar}{r_bar}'):
        pool.append(pair[0])

    with open('datasets/preprocessed/huggingface/concatenated/concatenated.pair.pkl', 'wb') as fw:
        pickle.dump(pairs, fw)
    with open('datasets/preprocessed/huggingface/concatenated/concatenated.sole.pkl', 'wb') as fw:
        pickle.dump(soles, fw)
    with open('datasets/preprocessed/huggingface/concatenated/concatenated.pool.pkl', 'wb') as fw:
        pickle.dump(pool, fw)


def make_next_pairs():
    # Load preprocessed dataset
    print("Loading the pair dataset...", end=' ')
    print("pairs...", end=' ')
    with open('datasets/preprocessed/huggingface/concatenated/concatenated.pair.pkl', 'rb') as fr:
        pairs = pickle.load(fr)
    print("soles...", end=' ')
    with open('datasets/preprocessed/huggingface/concatenated/concatenated.sole.pkl', 'rb') as fr:
        soles = pickle.load(fr)
    print("pool...", end=' ')
    with open('datasets/preprocessed/huggingface/concatenated/concatenated.pool.pkl', 'rb') as fr:
        pool = pickle.load(fr)
    print("Complete..!")

    print("Shuffling the pairs...", end=' ')
    random.shuffle(pairs)  # multi-dimensional arrays are only shuffled along the first axis:
    print("soles...", end=' ')
    random.shuffle(soles)
    print("pool...", end=' ')
    random.shuffle(pool)
    print('Complete..!')

    # IsNext
    half = int(len(pool) / 2)
    print('Making IsNext pairs...')
    IsNext_pairs = pairs[:half]
    for pair in tqdm(pairs[half:], total=len(pairs[half:]), desc='', bar_format='{l_bar}{r_bar}'):
        soles.append(pair[0])

    # NotNext
    NotNext_pairs = []  # NotNext pairs
    for i, sole in tqdm(enumerate(soles), total=len(soles),
                        desc='Making NotNext pairs...', bar_format='{l_bar}{r_bar}'):
        pair = [sole, pool[i]]
        NotNext_pairs.append(pair)

    # save the files
    print("Saving the pairs...", end=' ')
    print("IsNext_pairs...", end=' ')
    with open('datasets/preprocessed/huggingface/concatenated/IsNext_pairs.pkl', 'wb') as fw:
        pickle.dump(IsNext_pairs, fw)
    print("NotNext_pairs...", end=' ')
    with open('datasets/preprocessed/huggingface/concatenated/NotNext_pairs.pkl', 'wb') as fw:
        pickle.dump(NotNext_pairs, fw)
    print("Complete..!")


if __name__ == '__main__':
    split_to_article()      # with saving file
    make_pair()             # with saving file
    make_next_pairs()
