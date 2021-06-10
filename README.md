This project is about the reimplementation of the paper, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

# Datasets
- English Wikipedia dataset
: Use Dump file from "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2", and using WikiExtractor (https://github.com/attardi/wikiextractor) clean up the raw dump wiki data.

- BookCorpus
: Use old ver BookCorpus dataset from huggingface (here: "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip")

[dataset_info.txt](https://github.com/SungHo3268/BERT/files/6629607/dataset_info.txt)



# Tokenizer
: To make tokenizer, I use the sentencepiece module of Google with the command like below.

spm_train --input=tokenizer/concatenated.split.forSPM.txt 
          --model_prefix=pretrain_all_30k
          --vocab_size=30000 
          --pad_id=0 
          --unk_id=1 
          --eos_id=-1 
          --bos_id=-1 
          --control_symbols=[CLS],[SEP],[MASK] 
          --user_defined_symbols='(,),\",-,.,–,£,€' 
          --shuffle_input_sentence=true 
          --character_coverage=0.99995 
          --model_type=bpe



# Preprocess
: Since I just implemente the tiny and small model of BERT, I just set the batch size as 128 and 128 max_seq_len = tokens.
: so I delt with 128x128 tokens as a step.



# Functions
- collate_fn
: input = the list of sequences expressed in indices.
          (one sequence which is one sole sentence or the pair of two sentences per line)
          
          

# Remark for model
: Due to lack of memory and time, I just implemented the tiny and small model of BERT in this project.
: If you wanna do get a base or large model, just change the arguments of 'run_pretrain.py' -- args.hidden_layer_num, args.att_head_num, args.d_model, args
