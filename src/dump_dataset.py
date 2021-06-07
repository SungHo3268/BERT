from datasets import load_dataset, concatenate_datasets, load_from_disk
import numpy as np
import pickle
from tqdm.auto import tqdm
import sys
import os
sys.path.append(os.getcwd())


def load_from_huggingface():
    # en-wiki dataset
    wiki = load_dataset('wikipedia', '20200501.en', split='train')
    wiki = wiki.remove_columns("title")

    # book corpus dataset
    bookcorpus = load_dataset('bookcorpus', split='train')

    # concatenate two corpus
    assert bookcorpus.features.type == wiki.features.type  # is the same structure type?
    concatenated = concatenate_datasets([wiki, bookcorpus])
    return wiki, bookcorpus, concatenated


def save_dataset(wiki, bookcorpus, concatenated):
    # Save the datasets.
    wiki.save_to_disk('datasets/raw/huggingface/wiki')
    bookcorpus.save_to_disk('datasets/raw/huggingface/bookcorpus')
    concatenated.save_to_disk('datasets/raw/huggingface/concatenated')
    return None


if __name__ == '__main__':
    wiki, bookcorpus, concatenated = load_from_huggingface()
    save_dataset(wiki, bookcorpus, concatenated)
