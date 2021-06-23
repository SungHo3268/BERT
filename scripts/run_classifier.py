import argparse
from distutils.util import strtobool as _bool
import json
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import sys
import os
sys.path.append(os.getcwd())
from src.preprocess import *
from src.models import *


############################## Argparse ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='56789')
parser.add_argument('--task', type=str, default='mrpc',
                    help="cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb")
parser.add_argument('--max_epoch', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--hidden_layer_num', type=int, default=4)
parser.add_argument('--att_head_num', type=int, default=8)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--initializer_range', type=float, default=0.2)
parser.add_argument('--trained_steps', type=str, default='900k')
parser.add_argument('--eval_interval', type=int, default=10)
parser.add_argument('--initial_lr', type=float, default=1e-04)
parser.add_argument('--pretraining', type=_bool, default=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--gpu', type=_bool, default=True)
parser.add_argument('--cuda', type=int, default=1)
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
if args.task == 'mnli':
    train_inputs, train_labels, val_matched_inputs, val_matched_labels, val_mismatched_inputs, val_mismatched_labels \
        = load_ft_dataset(args.task, tokenizer, cls_id, sep_id, args.max_seq_len)
    # make train dataset loader
    class_num = len(set(train_labels))
    train_dataset = []
    for i in range(len(train_inputs)):
        train_dataset.append([train_inputs[i], train_labels[i]])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                              drop_last=True)
    # make validation matched dataset loader
    val_matched_dataset = []
    for i in range(len(val_matched_inputs)):
        val_matched_dataset.append([val_matched_inputs[i], val_matched_labels[i]])
    validation_matched_loader = DataLoader(dataset=val_matched_dataset, batch_size=args.batch_size, num_workers=4,
                                           shuffle=True, drop_last=True)
    # make validation mismatched dataset loader
    val_mismatched_dataset = []
    for i in range(len(val_matched_inputs)):
        val_mismatched_dataset.append([val_mismatched_inputs[i], val_mismatched_labels[i]])
    validation_mismatched_loader = DataLoader(dataset=val_mismatched_dataset, batch_size=args.batch_size, num_workers=4,
                                              shuffle=True, drop_last=True)
else:
    train_inputs, train_labels, val_inputs, val_labels \
        = load_ft_dataset(args.task, tokenizer, cls_id, sep_id, args.max_seq_len)
    # make train dataset loader
    class_num = len(set(train_labels))
    if args.task == 'stsb':
        class_num = 1
    train_dataset = []
    for i in range(len(train_inputs)):
        train_dataset.append([train_inputs[i], train_labels[i]])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                              drop_last=True)
    # make validation dataset loader
    val_dataset = []
    for i in range(len(val_inputs)):
        val_dataset.append([val_inputs[i], val_labels[i]])
    validation_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                   drop_last=True)


############################## Init Net ##############################
embed_weight = nn.parameter.Parameter(torch.empty(V, args.d_model), requires_grad=True)
model = BERT(V, args.d_model, embed_weight, args.max_seq_len, args.dropout, args.hidden_layer_num, d_ff,
             args.att_head_num, args.pretraining, args.gpu, args.cuda)
criterion = nn.CrossEntropyLoss()
stsb_criterion = nn.MSELoss()
optimizer = optim.AdamW(params=model.parameters(), lr=args.initial_lr,
                        betas=(beta1, beta2), weight_decay=l2_weight_decay)
scaler = amp.GradScaler()


# Load pre-trained model
ckpt_dir = os.path.join(log_dir, 'ckpt')
model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'model_{args.trained_steps}.ckpt'),
                                 map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))
model.pretraining = False
scaler.load_state_dict(torch.load(os.path.join(ckpt_dir, f'scaler_{args.trained_steps}.ckpt'),
                                  map_location=f'cuda:{args.cuda}' if args.gpu else 'cpu'))

classifier = Classifier(model, class_num, args.d_model)
classifier.init_param(args.initializer_range)
device = None
if args.gpu:
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    criterion.to(device)


############################## Start FineTuning ##############################
print(f"{args.task}_fine_tuning_{args.batch_size}B_{args.initial_lr}lr")
for epoch in range(args.max_epoch):
    classifier.train()
    for train_input, train_label in tqdm(train_loader, desc='Fine tuning...',
                                         total=len(train_loader), bar_format='{l_bar}{r_bar}'):
        if args.gpu:
            train_input = train_input.to(device)
            train_label = train_label.to(device)
            if args.task == 'stsb':
                train_label = train_label.to(torch.float32)

        optimizer.zero_grad()
        with amp.autocast():
            # forward
            out = classifier(train_input, sep_id)        # out = (batch_size, max_seq_len, class_num)
            cls_out = out[:, 0]  # cls = (batch_size, class_num)
            if args.task == 'stsb':
                cls_out = torch.sigmoid(cls_out.squeeze()) * 5
                loss = stsb_criterion(cls_out, train_label)
            elif args.task == 'qnli':
                loss = criterion(cls_out, train_label.to(torch.long))
            else:
                loss = criterion(cls_out, train_label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    classifier.eval()
    val_acc = 0
    val_f1 = 0
    pearson_corr = 0
    spearman_corr = 0
    count = 0
    if args.task == 'mnli':         # two accuracy
        matched_acc = 0
        mismatched_acc = 0
        for val_input, val_label in validation_matched_loader:
            if args.gpu:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
            count += 1
            with amp.autocast():
                # evaluation
                acc = classifier.predict(val_input, val_label, sep_id)  # out = (batch_size, max_seq_len, class_num)
                val_acc += acc
        matched_acc = val_acc / count * 100

        for val_input, val_label in validation_mismatched_loader:
            if args.gpu:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
            count += 1
            with amp.autocast():
                # evaluation
                acc = classifier.predict(val_input, val_label, sep_id)  # out = (batch_size, max_seq_len, class_num)
                val_acc += acc
        mismatched_acc = val_acc / count * 100
        print(f"{epoch+1}/{args.max_epoch}epoch, validation acc (matched/ mismatched): "
              f"{float(matched_acc).__round__(1)}/ {float(mismatched_acc).__round__(1)} [%]\n")

    elif args.task == 'qqp' or args.task == 'mrpc':         # F1 score and Accuracy
        total_out = []
        total_label = []
        for val_input, val_label in validation_loader:
            if args.gpu:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
            count += 1
            with amp.autocast():
                # evaluation
                out = classifier(val_input, sep_id)  # out = (batch_size, max_seq_len, class_num)
            cls = out[:, 0]  # cls = (batch_size, class_num)
            max_cls = torch.argmax(cls, dim=-1)     # cls = (batch_size, )

            total_out.append(max_cls)
            total_label.append(val_label)
        # get f1 score
        total_out = torch.cat(total_out)
        total_label = torch.cat(total_label)
        validation_f1 = f1_score(total_out.to('cpu'), total_label.to('cpu')) * 100
        # get accuracy
        correct = torch.sum(total_out == total_label)
        validation_acc = correct/len(total_out) * 100
        print(f"{epoch+1}/{args.max_epoch}epoch, validation f1/ acc: "
              f"{validation_f1.__round__(1)}/ {float(validation_acc).__round__(1)} [%]\n")

    elif args.task == 'cola':       # Metthew's Corr
        total_out = []
        total_label = []
        for val_input, val_label in validation_loader:
            if args.gpu:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
            count += 1
            with amp.autocast():
                # evaluation
                out = classifier(val_input, sep_id)  # out = (batch_size, max_seq_len, class_num)
            cls = out[:, 0]  # cls = (batch_size, class_num)
            max_cls = torch.argmax(cls, dim=-1)  # cls = (batch_size, )

            total_out.append(max_cls)
            total_label.append(val_label)
        # get f1 score
        total_out = torch.cat(total_out)
        total_label = torch.cat(total_label)
        validation_matthew = matthews_corrcoef(total_out.to('cpu'), total_label.to('cpu')) * 100
        print(f"{epoch + 1}/{args.max_epoch}epoch, validation Matthews_corr: {validation_matthew.__round__(1)} [%]\n")

    elif args.task == 'stsb':       # Pearson-Spearman Corr
        for val_input, val_label in validation_loader:
            if args.gpu:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
            count += 1
            with amp.autocast():
                # evaluation
                out = classifier(val_input, sep_id)  # out = (batch_size, max_seq_len, class_num)

            cls = out[:, 0].squeeze()  # cls = (batch_size, class_num)
            pearson_corr += np.corrcoef(cls.to('cpu').detach().numpy(), val_label.to('cpu'))[0][1]
            spearman_corr += stats.spearmanr(cls.to('cpu').detach().numpy(), val_label.to('cpu'))[0]
        val_pearson_corr = pearson_corr / count * 100
        val_spearman_corr = spearman_corr / count * 100
        print(f"{epoch + 1}/{args.max_epoch}epoch, validation pearson_corr/ spearman_corr: "
              f"{val_pearson_corr.__round__(1)}/ {val_spearman_corr.__round__(1)} [%]\n")

    else:
        total_out = []
        total_label = []
        for val_input, val_label in validation_loader:
            if args.gpu:
                val_input = val_input.to(device)
                val_label = val_label.to(device)
            count += 1
            with amp.autocast():
                # evaluation
                out = classifier(val_input, sep_id)  # out = (batch_size, max_seq_len, class_num)
            cls = out[:, 0]  # cls = (batch_size, class_num)
            max_cls = torch.argmax(cls, dim=-1)  # cls = (batch_size, )
            total_out.append(max_cls)
            total_label.append(val_label)
        # get accuracy
        total_out = torch.cat(total_out)
        total_label = torch.cat(total_label)
        correct = torch.sum(total_out == total_label)
        validation_acc = correct/len(total_out) * 100
        print(f"{epoch + 1}/{args.max_epoch}epoch, validation acc: {float(validation_acc).__round__(1)} [%]\n")
