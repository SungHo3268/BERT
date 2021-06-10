This project is about the reimplementation of the paper, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

# Datasets
- English Wikipedia dataset
: Use Dump file from "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2", and using WikiExtractor (https://github.com/attardi/wikiextractor) clean up the raw dump wiki data.

- BookCorpus
: Use old ver BookCorpus dataset from huggingface (here: "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip")



# Tokenizer
: To make tokenizer, I use the sentencepiece module of Google with the command like below.
spm_train --input=datasets/raw/pretrain_dataset_all.txt --model_prefix=dataset_all_30k --vocab_size=30000 --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 --control_symbols=[CLS],[SEP],[MASK] --user_defined_symbols="(,),\",-,.,–,£,€" --shuffle_input_sentence=true --character_coverage=0.99995 --model_type=word

