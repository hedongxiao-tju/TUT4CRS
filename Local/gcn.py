import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from utils import *

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, conv_name='gcn', n_heads=1):
        super(GraphConvolution, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_features, out_features)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_features, out_features // n_heads, heads=n_heads)
        elif self.conv_name == 'sage':
            self.base_conv = SAGEConv(in_features, out_features)
        else:
            print("no predefined conv layer {} !".format(conv_name))

    def forward(self, input_x, edge_index):
        if self.conv_name == 'gcn':
            return self.base_conv(input_x, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(input_x, edge_index)
        elif self.conv_name == 'sage':
        	return self.base_conv(input_x, edge_index)


TIME_NUM_DICT = {
    MOVIE : 94,
    YELP_STAR : 65
}

TIME_DIM_NUM_DICT = {
    MOVIE : 5,
    YELP_STAR : 4
}


class GraphEncoder(Module):
    def __init__(self, device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True, hidden_size=100, layers=1, rnn_layer=1, conv_name='gat', n_heads = 1, time_emb_16 = None, data_name = MOVIE):
        super(GraphEncoder, self).__init__()
        self.data_name = data_name
        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity-1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings,freeze=fix_emb)

        self.time_embedding = nn.Embedding(TIME_NUM_DICT[self.data_name], time_emb_16.shape[1])
        if time_emb_16 is not None:
            print(f"pre-trained embeddings-16 with time")
            self.time_embedding = self.time_embedding.from_pretrained(time_emb_16, freeze=fix_emb)

        self.time_transfer = Parameter(torch.FloatTensor(time_emb_16.shape[1], emb_size))
        nn.init.xavier_uniform_(self.time_transfer)

        self.layers = layers
        self.user_num = len(kg.G['user'])
        self.item_num = len(kg.G['item'])
        self.PADDING_ID = entity-1
        self.device = device
        self.seq = seq
        self.gcn = gcn

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        elif self.seq == 'transformer':
            self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400), num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            print(f'conv_name = {conv_name};  n_heads = {n_heads};  layers = {layers}')
            for l in range(layers):
                self.gnns.append(GraphConvolution(indim, outdim, conv_name=conv_name, n_heads=n_heads))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)

    def forward(self, b_state):
        """
        :param b_state [N]
        :return: [N x L x d]
        """
        batch_output = []
        for s in b_state:
            #  neighbors = cur_node + user + cand_items + reachable_feature +   self.time_node
            neighbors, adj = s['neighbors'].to(self.device), s['adj'].to(self.device)    #neighbors：LongTensor   adj: LongTensor
            # uif full_id    times short_id
            uifs,times = neighbors[:-TIME_DIM_NUM_DICT[self.data_name]].to(self.device),  (neighbors[-TIME_DIM_NUM_DICT[self.data_name]:]-(s['user_length'] + s['item_length'] + s['feature_length'])).to(self.device)  # acc_att, user, cand_items, reachable_feature

            uifs_emb = self.embedding(uifs)
            time_emb = torch.matmul(self.time_embedding(times),self.time_transfer)

            input_state = torch.cat((uifs_emb, time_emb), dim=0)
            if self.gcn:
                for gnn in self.gnns:
                    output_state = gnn(input_state, adj)
                    input_state = output_state
                batch_output.append(output_state)
            else:
                output_state = F.relu(self.fc2(input_state))
                batch_output.append(output_state)

        seq_embeddings = []
        for s, o in zip(b_state, batch_output):
            seq_embeddings.append(o[:len(s['cur_node'])+1,:][None,:])   #acc_fea and user  TransPart

        if len(batch_output) > 1:
            seq_embeddings = self.padding_seq(seq_embeddings)
        seq_embeddings = torch.cat(seq_embeddings, dim=0)  # [N x L x d]

        if self.seq == 'rnn':
            _, h = self.rnn(seq_embeddings)
            seq_embeddings = h.permute(1,0,2) #[N*1*D]
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
            new_s[:cur_size,:] = s[0]
            padded_seq.append(new_s[None,:])
        return padded_seq
