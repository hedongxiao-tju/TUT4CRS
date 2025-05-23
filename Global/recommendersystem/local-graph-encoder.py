import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from conv import GeneralConv
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)   #XW
        output = torch.sparse.mm(adj, support)   #AXW
        if self.bias is not None:
            return output + self.bias
        else:
            return output




class GraphEncoder(Module):
    def __init__(self, input_x, edge_index, kg, hidden_size=100, conv_name = 'gcn', layers=2):
        super(GraphEncoder, self).__init__()
        self.entity = input_x.shape[0]
        self.emb_size = input_x.shape[1]
        # self.embedding = nn.Embedding(self.entity, self.emb_size, padding_idx=entity - 1)
        self.embedding = nn.Embedding(self.entity, self.emb_size)
        if input_x is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(input_x, freeze=True)
        self.layers = layers
        self.user_num = len(kg.G['user'])
        self.item_num = len(kg.G['item'])
        self.gcn = conv_name  # args action='store_false', help='use GCN or not'

        # self.fc1 = nn.Linear(hidden_size, hidden_size)

        indim, outdim = self.emb_size, hidden_size
        self.gnns = nn.ModuleList()
        for l in range(layers):
            self.gnns.append(GraphConvolution(indim, outdim))
            indim = outdim
        print('GraphEncoder init ok...')

    def forward(self, b_state):
        """
        :param b_state [N]
        :return: [N x L x d]
        """
        batch_output = []
        for s in b_state:
            # neighbors, adj = self.get_state_graph(s)
            neighbors, adj = s['neighbors'].to(self.device), s['adj'].to(self.device)
            input_state = self.embedding(neighbors)
            for gnn in self.gnns:
                output_state = gnn(input_state, adj)
                input_state = output_state
            batch_output.append(output_state)

        seq_embeddings = []
        for s, o in zip(b_state, batch_output):
            seq_embeddings.append(o[:len(s['cur_node']), :][None, :])
        if len(batch_output) > 1:
            seq_embeddings = self.padding_seq(seq_embeddings)
        seq_embeddings = torch.cat(seq_embeddings, dim=0)  # [N x L x d]

        if self.seq == 'rnn':
            _, h = self.rnn(seq_embeddings)
            seq_embeddings = h.permute(1, 0, 2)  # [N*1*D]
        elif self.seq == 'transformer':
            seq_embeddings = torch.mean(self.transformer(seq_embeddings), dim=1, keepdim=True)
        elif self.seq == 'mean':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)


        seq_embeddings = F.relu(self.fc1(seq_embeddings))

        return seq_embeddings

    def padding_seq(self, seq):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size, :] = s[0]
            padded_seq.append(new_s[None, :])
        return padded_seq
