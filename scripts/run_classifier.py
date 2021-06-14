from datasets import load_dataset
import argparse
from distutils.util import strtobool as _bool
import json
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import sys
import os
sys.path.append(os.getcwd())
from src.preprocess import *
from src.models import *


############################## Argparse ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='56789')
parser.add_argument('--task', type=str, default='rte',
                    help="ax, cola, mnli, mnli_matched,mnli_mismatched, mrpc, qnli, qqp, rte, sst2, stsb, wnli")
parser.add_argument('--max_epoch', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--hidden_layer_num', type=int, default=2)
parser.add_argument('--att_head_num', type=int, default=2)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--initializer_range', type=float, default=0.2)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=100000)
parser.add_argument('--initial_lr', type=float, default=2e-5)
parser.add_argument('--pretraining', type=_bool, default=True)
parser.add_argument('--max_steps', type=int, default=1000000)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--restart', type=_bool, default=False)
parser.add_argument('--restart_epoch', type=int, default=0)
args = parser.parse_args()

log_dir = f'log/bert_{args.max_seq_len}seq_{args.d_model}H_{args.hidden_layer_num}L_{args.att_head_num}A/'
ft_dir = f'log/bert_{args.max_seq_len}seq_{args.d_model}H_{args.hidden_layer_num}L_{args.att_head_num}A/fine_tuning'
if not os.path.exists(ft_dir):
    os.mkdir(ft_dir)
with open(os.path.join(ft_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f)


############################## Hyperparameter ##############################
# load vocabulary
with open('tokenizer/pretrain_all_30k.vocab', 'r', encoding='utf8') as f:
    data = f.readlines()
    vocab = []
    for word in data:
        vocab.append(word.split('\t')[0])
inverse_vocab = {}
for i, word in enumerate(vocab):
    inverse_vocab[word] = i
V = len(vocab)

# Load the tokenizer made by sentencepiece
model_file = 'tokenizer/pretrain_all_30k.model'
tokenizer = spm.SentencePieceProcessor(model_file=model_file)


# define symbol
PAD = '<pad>'
UNK = '<unk>'
MASK = '[MASK]'
CLS = '[CLS]'
SEP = '[SEP]'

pad_id = inverse_vocab['<pad>']
unk_id = inverse_vocab['<unk>']
mask_id = inverse_vocab['[MASK]']
cls_id = inverse_vocab['[CLS]']
sep_id = inverse_vocab['[SEP]']

# set hyperparameter
d_ff = int(args.d_model * 4)
warmup_steps = 10000
beta1 = 0.9
beta2 = 0.999
l2_weight_decay = 0.01
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


############################## Load dataset ##############################
dataset = load_dataset('glue', args.task)

single_dataset = ['sst2', 'cola']
pair_dataset = ['mnli', 'mnli_matched', 'mnli_mismatched', 'qqp', 'qnli', 'stsb', 'mrpc', 'rte']

if args.task in single_dataset:
    print('Loading the train dataset...')
    train_inputs, train_labels = dataset['train']['sentence'], dataset['train']['label']
    train_inputs, train_labels = preprocess_ft_dataset(train_inputs, train_labels, tokenizer,
                                                       cls_id, sep_id, args.task, args.max_seq_len)
    print('Loading the validation dataset...')
    val_inputs, val_labels = dataset['validation']['sentence'], dataset['validation']['label']
    val_inputs, val_labels = preprocess_ft_dataset(val_inputs, val_labels, tokenizer,
                                                   cls_id, sep_id, args.task, args.max_seq_len)
elif args.task in pair_dataset:
    print('Loading the train dataset...')
    train_input1, train_input2, train_labels = dataset['train']['sentence1'],\
                                               dataset['train']['sentence2'],\
                                               dataset['train']['label']
    train_inputs = [train_input1, train_input2]
    train_inputs, train_labels = preprocess_ft_dataset(train_inputs, train_labels, tokenizer,
                                                       cls_id, sep_id, args.task, args.max_seq_len)
    print('Loading the validation dataset...')
    val_input1, val_input2, val_labels = dataset['validation']['sentence1'], \
                                         dataset['validation']['sentence2'], \
                                         dataset['validation']['label']
    val_inputs = [val_input1, val_input2]
    val_inputs, val_labels = preprocess_ft_dataset(val_inputs, val_labels, tokenizer,
                                                   cls_id, sep_id, args.task, args.max_seq_len)

class_num = len(set(train_labels))
train_dataset = []
for i in range(len(train_inputs)):
    train_dataset.append([train_inputs[i], train_labels[i]])
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

val_dataset = []
for i in range(len(val_inputs)):
    val_dataset.append([val_inputs[i], val_labels[i]])
validation_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)


############################## Init Net ##############################
embed_weight = nn.parameter.Parameter(torch.empty(V, args.d_model), requires_grad=True)
model = BERT(V, args.d_model, embed_weight, args.max_seq_len, args.dropout, args.hidden_layer_num, d_ff,
             args.att_head_num, args.pretraining, args.gpu, args.cuda)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params=model.parameters(), lr=args.initial_lr,
                        betas=(beta1, beta2), weight_decay=l2_weight_decay)
scaler = amp.GradScaler()


# Load pre-trained model
print("Loading the model...", end=' ')
ckpt_dir = os.path.join(log_dir, 'ckpt')
model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.ckpt'),
                                 map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))
model.pretraining = False
scaler.load_state_dict(torch.load(os.path.join(ckpt_dir, 'scaler.ckpt'),
                                  map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))
print("Complete.")

classifier = Classifier(model, class_num, args.d_model)
classifier.init_param(args.initializer_range)
device = None
if args.gpu:
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    criterion.to(device)


############################## Start FineTuning ##############################
stack = 0
total_loss = 0
total_acc = 0
loss_list = []
acc_list = []

val_acc = 0
count = 0
best_acc = [0, 0.0]       # [epoch, acc]
for epoch in range(args.max_epoch):
    classifier.train()
    for train_input, train_label in tqdm(train_loader, desc=f'Fine tuning {args.task}...',
                                         total=len(train_loader), bar_format='{l_bar}{r_bar}'):
        if args.gpu:
            train_input = train_input.to(device)
            train_label = train_label.to(device)

        optimizer.zero_grad()
        with amp.autocast():
            # forward
            out = classifier(train_input, sep_id)        # out = (batch_size, max_seq_len, class_num)
            cls_out = out[:, 0]  # cls = (batch_size, class_num)
            loss = criterion(cls_out, train_label)
            total_loss += loss

        # acc
        correct = torch.sum(torch.argmax(cls_out, dim=-1) == train_label)
        acc = correct / len(train_label)
        total_acc += acc

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        stack += 1
        if stack == args.eval_interval:         # update
            avg_loss = total_loss/stack
            avg_acc = total_acc/stack
            loss_list.append(avg_loss)
            acc_list.append(avg_acc)
            # print(f"loss: {avg_loss}    |    acc: {avg_acc*100} [%]")
            stack = 0
            total_loss = 0
            total_acc = 0

    classifier.eval()
    for val_input, val_label in tqdm(validation_loader, desc=f'Evaluation...',
                                         total=len(validation_loader), bar_format='{l_bar}{r_bar}'):
        if args.gpu:
            val_input = val_input.to(device)
            val_label = val_label.to(device)

        count += 1
        with amp.autocast():
            # evaluation
            acc = classifier.predict(val_input, val_label, sep_id)  # out = (batch_size, max_seq_len, class_num)
            val_acc += acc

    validation_acc = val_acc / count * 100
    if best_acc[1] < validation_acc and epoch != 0:
        best_acc[1] = validation_acc
        best_acc[0] = epoch+1
    print(f"{epoch+1}/{args.max_epoch}epoch valid_accuracy: {validation_acc}[%]")
print(f"best_acc: {best_acc[1]}[%]   at {best_acc[0]}")
