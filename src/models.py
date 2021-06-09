import sys
import os
sys.path.append(os.getcwd())
from src.layer import *
from src.functions import *


class BERT(nn.Module):
    def __init__(self, V, d_model, embed_weight, max_seq_len, dropout, hidden_layer_num, d_ff, head_num,
                 pretraining, gpu, cuda):
        """
        :param V: Vocabulary size
        :param d_model: embedding and hidden dimension size
        :param embed_weight: shared weight matrix with Embedding, and linear layer before softmax.
        :param max_seq_len: the maximum length of the sentence
        :param dropout: the dropout ratio
        :param hidden_layer_num: the number of hidden layers
        :param d_ff: position wise feed forward network hidden dimension size
        :param head_num: the number of head in multi-head attention
        :param pretraining: Is pretraining or fine tuning?
        :param gpu: (bool) do you want to use gpu computation?
        :param cuda: (int) gpu number
        """
        super(BERT, self).__init__()
        self.embed_weight = embed_weight
        self.pretraining = pretraining
        self.gpu = gpu
        self.cuda = cuda
        self.input_layer = InputLayer(d_model, self.embed_weight, max_seq_len, dropout, gpu, cuda)
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(SubLayers(d_model, d_ff, head_num, dropout, gpu, cuda))
        if self.pretraining:
            self.mlm_layer1 = nn.Linear(d_model, d_model, bias=False)
            self.gelu = nn.GELU()
            self.layer_norm = nn.LayerNorm(d_model)
            self.mlm_layer2 = nn.Linear(d_model, V, bias=True)
            self.nsp_layer = nn.Linear(d_model, 2, bias=True)

    def forward(self, inputs, sep_id):
        """
        :param inputs: (batch_size, max_seq_len)
        :param sep_id: the vocabulary id of '[SEP]' token
        """
        pad_mask = get_pad_mask(inputs, self.gpu, self.cuda)
        seg_embedding = get_seg_embedding(inputs, sep_id, self.gpu, self.cuda)
        hs = self.input_layer(inputs, seg_embedding)
        for sub_layer in self.sub_layers:
            hs = sub_layer(hs, pad_mask)  # hs = (batch_size, max_seq_len, d_model)
        if self.pretraining:
            mlm_out = self.mlm_layer1(hs)               # mlm_out = (batch_size, max_seq_len, d_model)
            mlm_out = self.gelu(mlm_out)
            mlm_out = self.layer_norm(mlm_out)
            mlm_out = self.mlm_layer2(mlm_out)          # mlm_out = (batch_size, max_seq_len, V)
            nsp_out = self.nsp_layer(hs[:, 0])          # nsp_out = (batch_size, 2)
            return mlm_out, nsp_out
        else:
            return hs

    def init_param(self, initializer_range):
        # self.sub_layers.init_param(initializer_range)
        for sub_layer in self.sub_layers:
            sub_layer.init_param(initializer_range)
        if self.pretraining:
            nn.init.trunc_normal_(self.mlm_layer1.weight, std=initializer_range)
            self.mlm_layer2.weight_ = self.embed_weight.T
            nn.init.trunc_normal_(self.nsp_layer.weight, std=initializer_range)
            nn.init.constant_(self.mlm_layer2.bias, 0)
            nn.init.constant_(self.nsp_layer.bias, 0)
