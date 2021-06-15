import sentencepiece as spm
import sys
import os
sys.path.append(os.getcwd())


# sentencepiece - bpe
"""
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
"""

with open('tokenizer/pretrain_all_30k.vocab', 'r', encoding='utf8') as f:
    data = f.readlines()
    vocab = []
    for word in data:
        vocab.append(word.split('\t')[0])
inverse_vocab = {}
for i, word in enumerate(vocab):
    inverse_vocab[word] = i

model_file = 'tokenizer/pretrain_all_30k.model'
sp = spm.SentencePieceProcessor(model_file=model_file)
de = sp.EncodeAsIds(
    ["I'm developing with pycharm locally on my windows 10 machine. And have a guest VM machine with ubuntu "
     "which contain my remote deployment server. I'm using SFTP for syncing between the two environments. "])
print(de)
print(sp.Decode(de))
