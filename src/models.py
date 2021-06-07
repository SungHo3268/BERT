import sys
import os
sys.path.append(os.getcwd())
from src.layer import *
from src.functions import *


class BERT(nn.Module):
    def __init__(self, V, d_model, embed_weight, max_seq_len, dropout, hidden_layer_num, d_ff, head_num, gpu, cuda):
        """
        :param V: Vocabulary size
        :param d_model: embedding and hidden dimension size
        :param embed_weight: shared weight matrix with Encoder, Decoder, and linear layer before softmax.
        :param max_seq_len: the maximum length of the sentence
        :param dropout: the dropout ratio
        :param hidden_layer_num: the number of Encoder and Decoder layers
        :param d_ff: position wise feed forward network hidden dimension size
        :param head_num: the number of head in multi-head attention
        :param gpu: (bool) do you want to use gpu computation?
        :param cuda: (int) gpu number
        """
        super(BERT, self).__init__()
        self.gpu = gpu
        self.cuda = cuda
        self.input_layer = InputLayer(d_model, embed_weight, max_seq_len, dropout, gpu, cuda)
        self.sub_layers = nn.ModuleList()
        for _ in range(hidden_layer_num):
            self.sub_layers.append(SubLayers(d_model, d_ff, head_num, dropout, gpu, cuda))
        self.output_layer = nn.Linear(d_model, V)

    def forward(self, src_input, sep_id):
        """
        :param src_input: (batch_size, max_seq_len)
        :param sep_id: the vocabulary id of '[SEP]' token
        :return:
        """
        pad_mask = get_pad_mask(src_input, self.gpu, self.cuda)
        seg_embedding = get_seg_embedding(src_input, sep_id, self.gpu, self.cuda)
        src = self.input_layer(src_input, seg_embedding)
        hs = self.encoder(src, pad_mask)            # hs = (batch_size, max_seq_len, d_model)
        out = self.output_layer(hs)                 # out = (batch_size, max_seq_len, V)
        return out

    def init_param(self, initializer_range):
        self.input_layer.init_param(initializer_range)
        self.sub_layers.init_param(initializer_range)
        # for sub_layer in self.sub_layers:
        #     sub_layer.init_param(initializer_range)
        nn.init.trunc_normal_(self.output_layer.weight, std=initializer_range)
