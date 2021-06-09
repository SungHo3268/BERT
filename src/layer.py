import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())
from src.functions import *


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model, gpu, cuda):
        super(PositionalEmbedding, self).__init__()
        self.pos_encoding = torch.empty(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(d_model//2):
                exponent = pos / (10000**(2*i/d_model))
                exponent = torch.FloatTensor([exponent])
                self.pos_encoding[pos][2*i] = torch.sin(exponent)
                self.pos_encoding[pos][2*i+1] = torch.cos(exponent)
        if gpu:
            self.pos_encoding = self.pos_encoding.to(torch.device(f'cuda:{cuda}'))

    def forward(self, x):
        """
        :param x: It is a output of the embedding layer. x = (batch_size, max_seq_len, embedding_dim)
        :return: x = (batch_size, max_seq_len, embedding_dim)
        """
        x += self.pos_encoding
        return x


class InputLayer(nn.Module):
    def __init__(self, d_model, embed_weight, max_seq_len, dropout, gpu, cuda):
        super(InputLayer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding.from_pretrained(embed_weight, freeze=False, padding_idx=0)
        self.positional = PositionalEmbedding(max_seq_len, d_model, gpu, cuda)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seg_embedding):
        """
        :param x: source_input = (batch_size, max_seq_len)
        :param seg_embedding: segment embedding comprised of 0 or 1 = (batch_size, max_seq_len)
        :return: x = (batch_size, max_seq_len, embed_dim)
        """
        x = self.embedding(x)       # x = (batch_size, max_seq_len, embed_dim)
        x *= np.sqrt(self.d_model)
        x = self.positional(x)
        x += seg_embedding
        x = self.dropout(x)
        return x


class ScaledDotProdAtt(nn.Module):
    def __init__(self, d, gpu, cuda):
        super(ScaledDotProdAtt, self).__init__()
        self.d = d
        self.gpu = gpu
        self.cuda = cuda

    def forward(self, q, k, v, pad_mask=None):
        """
        :param q: (batch_size, head_num, seq_len(max_sen_len), d_k)
        :param k: (batch_size, head_num, seq_len(max_sen_len), d_k)
        :param v: (batch_size, head_num, seq_len(max_sen_len), d_v)
        :param pad_mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        :return:
        """
        att = torch.matmul(q, k.transpose(2, 3))        # att = (batch_size, head_num, seq_len, seq_len)
        att = att / np.sqrt(self.d)
        if pad_mask is not None:
            att += pad_mask * (-1e+9)
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, v)                      # att = (batch_size, head_num, seq_len, d_v)
        return att


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num, dropout, gpu, cuda):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.d = int(d_model / head_num)
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProdAtt(self.d, gpu, cuda)
        self.linear_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, pad_mask):
        """
        :param q: (batch_size, seq_len (max_seq_len), embed_dim)
        :param k: (batch_size, seq_len (max_seq_len), embed_dim)
        :param v: (batch_size, seq_len (max_seq_len), embed_dim)
        :param pad_mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        :return: out = (batch_size, seq_len(max_seq_len), d_model)
        """
        batch_size, seq_len, _ = q.size()
        residual = q.clone()
        q = self.linear_q(q).view(batch_size, seq_len, self.head_num, self.d)
        k = self.linear_k(k).view(batch_size, seq_len, self.head_num, self.d)
        v = self.linear_v(v).view(batch_size, seq_len, self.head_num, self.d)

        q = q.transpose(1, 2)                       # q = (batch_size, head_num, seq_len (max_seq_len), d_q)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = self.attention(q, k, v, pad_mask)        # att = (batch_size, head_num, seq_len (max_seq_len), d_v)
        att = att.transpose(1, 2)                   # att = (batch_size, seq_len(max_seq_len), head_num, d_v)
        att = att.contiguous().view(batch_size, seq_len, -1)     # att = (batch_size, seq_len(max_seq_len), d_model)
        out = self.linear_o(att)                      # out = (batch_size, seq_len(max_seq_len), d_model)
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        return out
    
    def init_param(self, initializer_range):
        nn.init.trunc_normal_(self.linear_q.weight, std=initializer_range)
        nn.init.trunc_normal_(self.linear_k.weight, std=initializer_range)
        nn.init.trunc_normal_(self.linear_v.weight, std=initializer_range)
        nn.init.trunc_normal_(self.linear_o.weight, std=initializer_range)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        :param x: the output of the MultiHeadAttention layer. x= (batch_size, seq_len(max_seq_len), d_model)
        :return: out = (batch_size, seq_len(max_seq_len), d_model)
        """
        residual = x.clone()
        out = self.linear1(x)           # out = (batch_size, seq_len(max_seq_len), d_ff)
        out = self.gelu(out)
        out = self.linear2(out)         # out = (batch_size, seq_len(max_seq_len), d_model)
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        return out
    
    def init_param(self, initializer_range):
        nn.init.trunc_normal_(self.linear1.weight, std=initializer_range)
        nn.init.trunc_normal_(self.linear2.weight, std=initializer_range)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)


class SubLayers(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout, gpu, cuda):
        super(SubLayers, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, head_num, dropout, gpu, cuda)
        self.pos_feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x, pad_mask):
        """
        :param x: (the output of the embedding layer) or (the output of the previous sublayer)
               x = (batch_size, max_seq_len, d_model)
        :param pad_mask: (batch_size, 1, 1, seq_len)
        :return: out = (batch_size, seq_len(max_seq_len), d_model)
        """
        att = self.multi_head_attention(x, x, x, pad_mask)        # x = (batch_size, seq_len(max_seq_len), d_model)
        out = self.pos_feed_forward(att)                # out = (batch_size, seq_len(max_seq_len), d_model)
        return out

    def init_param(self, initializer_range):
        self.multi_head_attention.init_param(initializer_range)
        self.pos_feed_forward.init_param(initializer_range)
