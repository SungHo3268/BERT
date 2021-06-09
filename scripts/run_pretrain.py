import argparse
from distutils.util import strtobool as _bool
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import time
import sys
import os
sys.path.append(os.getcwd())
from src.preprocess import *
from src.functions import *
from src.models import BERT


############################## Argparse ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='56789')
parser.add_argument('--max_epoch', type=int, default=11)
parser.add_argument('--stack_num', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--hidden_layer_num', type=int, default=2)
parser.add_argument('--att_head_num', type=int, default=2)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--initializer_range', type=float, default=0.2)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--initial_lr', type=float, default=1e-4)
parser.add_argument('--pretraining', type=_bool, default=True)
parser.add_argument('--max_steps', type=int, default=1000000)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--restart', type=_bool, default=False)
parser.add_argument('--restart_epoch', type=int, default=0)
args = parser.parse_args()

log_dir = f'log/bert_{args.max_seq_len}seq_{args.d_model}H_{args.hidden_layer_num}L_{args.att_head_num}A'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUD_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.cuda}'


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


############################## Tensorboard ##############################
tb_dir = os.path.join(log_dir, 'tb')
ckpt_dir = os.path.join(log_dir, 'ckpt')
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
tb_writer = SummaryWriter(tb_dir)


############################## Init Net ##############################
# Load the tokenizer made by sentencepiece
model_file = 'tokenizer/pretrain_all_30k.model'
tokenizer = spm.SentencePieceProcessor(model_file=model_file)

embed_weight = nn.parameter.Parameter(torch.empty(V, args.d_model), requires_grad=True)
nn.init.trunc_normal_(embed_weight, std=args.initializer_range)
model = BERT(V, args.d_model, embed_weight, args.max_seq_len, args.dropout, args.hidden_layer_num, d_ff,
             args.att_head_num, args.pretraining, args.gpu, 0)
model.init_param(args.initializer_range)
mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)
nsp_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), betas=(beta1, beta2), weight_decay=l2_weight_decay)
scaler = amp.GradScaler()


if args.restart:
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.ckpt'),
                                     map_location=f'cuda:0' if args.gpu else 'cpu'))
    optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'),
                                         map_location=f'cuda:0' if args.gpu else 'cpu'))
    scaler.load_state_dict(torch.load(os.path.join(ckpt_dir, 'scaler.ckpt'),
                                      map_location=f'cuda:0' if args.gpu else 'cpu'))
    with open(os.path.join(log_dir, 'ckpt/step_num.pkl'), 'rb') as fr:
        re_step_num = pickle.load(fr)

device = None
if args.gpu:
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)


############################## Start Pretrain ##############################
step_num = 0 if not args.restart else re_step_num
start_e = 0 if not args.restart else args.restart_epoch-1
stack = 0
total_loss = 0
mlm_loss_sum = 0
nsp_loss_sum = 0
mlm_acc_list = []
nsp_acc_list = []
for epoch in range(start_e, args.max_epoch):
    data_seg_num = 100
    # for shuffling
    IsNext_order = np.arange(data_seg_num)
    NotNext_order = np.arange(data_seg_num)
    np.random.shuffle(IsNext_order)
    np.random.shuffle(NotNext_order)
    for i in range(data_seg_num):
        print('\n')
        print(f"epoch: {epoch+1}/{args.max_epoch}, data_seg: {i+1}/{data_seg_num}")

        # Load input datasets
        print("Loading the input dataset...", end=' ')
        print(f"IsNext...", end=' ')
        with open(f'datasets/preprocessed/huggingface/corpus/IsNextre/IsNext_{i}.pkl', 'rb') as fr:
            corpus1 = pickle.load(fr)
            c_label1 = [0] * len(corpus1)
        print(f"NotNext...", end=' ')
        with open(f'datasets/preprocessed/huggingface/corpus/NotNextre/NotNext_{i}.pkl', 'rb') as fr:
            corpus2 = pickle.load(fr)
            c_label2 = [1] * len(corpus2)
        print('Complete.')

        corpus = corpus1 + corpus2
        c_label = c_label1 + c_label2
        dataset = []
        print('Concatenating...', end=' ')
        for j in range(len(corpus)):
            temp = (corpus[j], c_label[j])
            dataset.append(temp)
        print("Applying padding and Making batch...")
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                 num_workers=4, drop_last=True)
        for inputs, cls_label, mask_loc, mask_label in tqdm(data_loader, desc='pre-training...',
                                                            total=len(data_loader), bar_format='{l_bar}{r_bar}'):
            t0 = time.time()
            if args.gpu:
                inputs = torch.stack(inputs, dim=0).to(device)
                mask_label = torch.LongTensor(mask_label).to(device)
                cls_label = torch.LongTensor(cls_label).to(device)

            t1 = time.time()
            with amp.autocast():
                # forward
                mlm_out, nsp_out = model(inputs, sep_id)        # mlm_out = (batch_size, max_seq_len, V)
                t2 = time.time()
                                                                # nsp_out = (batch_size, 2)
                mask_out = get_mask_out(mlm_out, mask_loc)      # mask_out = (mask_num, V)
                t3 = time.time()
                # acc
                # MLM
                correct = torch.sum(torch.argmax(mask_out, dim=-1) == mask_label)
                mlm_acc = int(correct) / len(mask_label)
                mlm_acc_list.append(mlm_acc)
                t4 = time.time()
                # CLS
                correct = torch.sum(torch.argmax(nsp_out, dim=-1) == cls_label)
                nsp_acc = int(correct) / len(cls_label)
                nsp_acc_list.append(nsp_acc)
                t5 = time.time()
                # loss
                mlm_loss = mlm_criterion(mask_out, mask_label)
                nsp_loss = nsp_criterion(nsp_out, cls_label)
                t6 = time.time()
                mlm_loss /= args.stack_num
                nsp_loss /= args.stack_num
                loss = mlm_loss + nsp_loss
            scaler.scale(loss).backward()
            total_loss += loss
            mlm_loss_sum += mlm_loss
            nsp_loss_sum += nsp_loss
            stack += 1
            t7 = time.time()
            if stack == args.stack_num:         # update
                stack = 0
                step_num += 1
                if step_num < warmup_steps:
                    optimizer.param_groups[0]['lr'] += args.initial_lr / warmup_steps
                else:
                    optimizer.param_groups[0]['lr'] -= args.initial_lr / (args.max_steps - warmup_steps)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                t8 = time.time()
                tb_writer.add_scalar('total_loss/step', total_loss, step_num)
                tb_writer.add_scalar('mlm_loss/step', mlm_loss_sum, step_num)
                tb_writer.add_scalar('nsp_loss/step', nsp_loss_sum, step_num)
                tb_writer.add_scalar('mlm_acc/step', np.mean(mlm_acc_list), step_num)
                tb_writer.add_scalar('nsp_acc/step', np.mean(nsp_acc_list), step_num)
                tb_writer.add_scalar('lr/step', optimizer.param_groups[0]['lr'], step_num)
                tb_writer.flush()
                t9 = time.time()
                print('\n')
                print("Load to gpu", t1 - t0)
                print("forward", t2 - t1)
                print("get mask vec", t3 - t2)
                print("mlm acc", t4 - t3)
                print("nsp acc", t5 - t4)
                print("cal loss", t6 - t5)
                print("save, merge loss", t7 - t6)
                print("update lr and optimizer", t8 - t7)
                print("tensorboard", t9 - t8)

                total_loss = 0
                mlm_loss_sum = 0
                nsp_loss_sum = 0
                mlm_acc_list = []
                nsp_acc_list = []

                if step_num == args.max_steps:
                    break

    print("Saving the model...", end=' ')
    torch.save(model.state_dict(), os.path.join(log_dir, 'ckpt/model.ckpt'))
    print("optimizer...", end=' ')
    torch.save(optimizer.state_dict(), os.path.join(log_dir, 'ckpt/optimizer.ckpt'))
    print('scaler...', end=' ')
    torch.save(scaler.state_dict(), os.path.join(log_dir, 'ckpt/scaler.ckpt'))
    print('step_num...', end=' ')
    with open(os.path.join(log_dir, 'ckpt/step_num.pkl'), 'wb') as fw:
        pickle.dump(step_num, fw)
    print("Complete.\n")
