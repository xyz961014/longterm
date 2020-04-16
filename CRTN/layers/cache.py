import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math
from copy import deepcopy

from CRTN.layers.attention import DotProductAttention
from CRTN.layers.transformer import PositionalEmbedding

class Cache(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        self.args = deepcopy(args)
        self.corpus = corpus
        #self.demo = self.args.demo
        if self.args.summary_method == "no_summary":
            self.dk = self.args.num_steps * self.args.nhid
        elif self.args.summary_method in ["sum", "max", "mean", "last_state"]:
            self.dk = self.args.nhid
        elif self.args.summary_method == "weighted_sum":
            self.dk = self.args.nhid
            self.summary = nn.Linear(args.num_steps, 1)
        elif self.args.summary_method == "linear":
            self.dk = self.args.cache_dk
            self.summary = nn.Linear(args.nhid * args.num_steps, args.cache_dk)

        self.attn = DotProductAttention()

        self.batch_size = args.batch_size
        self.L = self.args.num_steps
        self.N = self.args.cache_N
        self.dv = self.args.nhid
        self.topk = self.args.cache_k
        self.theta = self.args.cache_theta

        self.pos_emb = PositionalEmbedding(args.nhid)


    def new_key_and_values(self):
        cache_key = torch.zeros(self.N, 
                                self.batch_size,
                                self.dk).cuda()
        nn.init.normal_(cache_key, std=self.args.init_std)
        cache_value = torch.zeros(self.N, 
                                  self.batch_size,
                                  self.L,
                                  self.dv * (self.args.nlayers + 1)).cuda()

        return cache_key, cache_value


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, query, keys):

        query = query.transpose(1, 2)
        query_len, bsz = query.size(0), query.size(1)
        query = query.reshape(query_len, bsz, -1)
        
        keys = keys.transpose(0, 1)
        
        #print(query.device, keys.device, values.device)
        if self.args.max_pooling:
            query = query.view(-1, self.args.num_steps, self.args.nhid)
            pooling_keys = keys.view(-1, self.N, self.args.num_steps, self.args.nhid)
            attention = torch.einsum("bih,bnjh->bijn", query, pooling_keys)
            attention = attention.view(-1, self.args.num_steps ** 2, 1, self.N)
            attention = attention.max(1)[0]
        else:
            attention = self.attn(query, keys, scale=self.theta) 

        _, topk_indices = attention.topk(self.topk, dim=-1)
        topk_indices = topk_indices.permute(2, 0, 1)

        return attention, topk_indices


    def renew(self, inputs, words=None, cache_info=None, keys=None, values=None):

        if keys is None and values is None:
            keys, values = self.new_key_and_values()

        keys, values = keys.detach(), values.detach()
        
        if self.args.merge:
            key_blocks, value_blocks, cache_info = self.merge(keys, values, cache_info)
        elif self.args.discard_worst:
            key_blocks, value_blocks, cache_info = self.discard_worst(keys, 
                                                                      values, 
                                                                      cache_info)
        else:
            key_blocks, value_blocks = self.eliminate_last(keys, values)

        if self.args.summary_method == "no_summary":
            new_key = inputs[-1].reshape(self.batch_size, -1)
        elif self.args.summary_method == "sum":
            new_key = inputs[-1].sum(dim=1)
        elif self.args.summary_method == "max":
            new_key, _ = inputs[-1].max(dim=1)
        elif self.args.summary_method == "mean":
            new_key = inputs[-1].mean(dim=1)
        elif self.args.summary_method == "last_state":
            new_key = inputs[-1,:,-1,:]
        elif self.args.summary_method == "weighted_sum":
            new_key = F.sigmoid(self.summary(inputs[-1].transpose(1, 2)).squeeze(-1))
        elif self.args.summary_method == "linear":
            new_key = F.sigmoid(self.summary(inputs[-1].reshape(-1, self.L * self.dv)))

        new_value = torch.einsum("mblh->lbmh", 
                                 inputs
                                 ).reshape(self.L, -1, (self.args.nlayers+1) * self.dv)

        key_blocks[-1] = new_key.unsqueeze(0).detach()
        value_blocks[-1] = new_value.transpose(0, 1).unsqueeze(0).detach()
        
        keys = torch.cat(key_blocks, 0)
        values = torch.cat(value_blocks, 0)
        
        return keys, values, cache_info.detach()


    def eliminate_last(self, keys, values):

        key_blocks = list(keys.split(1))
        value_blocks = list(values.split(1))
        for i in range(self.N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])

        return key_blocks, value_blocks

    def merge(self, keys, values, cache_info):
        
        key_blocks = list(keys.split(1))
        value_blocks = list(values.split(1))

        eli_key = key_blocks[0]
        eli_value = value_blocks[0]

        device = eli_key.device
        alpha = self.args.merge_alpha

        key_blocks[0] = alpha * eli_key + (1 - alpha) * key_blocks[1]
        value_blocks[0] = alpha * eli_value + (1 - alpha) * value_blocks[1]

        for i in range(1, self.N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])

        pos, recall = cache_info.split([1,2], dim=-1)
        pos = pos.squeeze(-1)
        merge_matrix = torch.eye(pos.size(0),
                                 pos.size(0) - 1,
                                 device=pos.device)
        merge_matrix = torch.cat((merge_matrix.new_zeros(pos.size(0), 1), 
                                  merge_matrix), dim=1)
        merge_matrix[0][0], merge_matrix[0][1] = alpha, 1 - alpha
        pos = torch.matmul(merge_matrix, pos + 1)
        pos = pos.unsqueeze(-1)
        cache_info = torch.cat((pos, recall), dim=-1)
        return key_blocks, value_blocks, cache_info

    def discard_worst(self, keys, values, cache_info):

        key_blocks = list(keys.split(1))
        value_blocks = list(values.split(1))

        pos, recalls, queries = cache_info.chunk(3, dim=-1)
        recall_mean = recalls / queries

        # discard the least used block
        eli_indice = recall_mean.squeeze(-1).argmin(dim=0)
        for b, eli_index in enumerate(eli_indice):
            for i in range(eli_index, self.N - 1):
                pos[i,b,:] = pos[i+1,b,:]
                recalls[i,b,:] = recalls[i+1,b,:]
                queries[i,b,:] = queries[i+1,b,:]
                key_blocks[i][:,b,:] = key_blocks[i+1][:,b,:]
                value_blocks[i][:,b,:,:] = value_blocks[i+1][:,b,:,:]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])
        pos = pos + 1
        pos[-1] = torch.ones_like(pos[0])
        recalls[-1] = torch.zeros_like(recalls[0])
        queries[-1] = torch.zeros_like(queries[0])

        cache_info = torch.cat((pos, recalls, queries), dim=-1)

        return key_blocks, value_blocks, cache_info

