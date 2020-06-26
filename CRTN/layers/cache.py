import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math
import numpy as np
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
            self.dk = args.cache_L * args.nhid
        elif self.args.summary_method in ["sum", "max", "mean", "last_state"]:
            self.dk = args.nhid
        elif self.args.summary_method == "weighted_sum":
            self.dk = args.nhid
            self.summary = nn.Linear(args.cache_L, 1, bias=False)
        elif self.args.summary_method == "conv":
            self.dk = args.nhid
            self.summary = nn.Conv1d(args.nhead, args.nhead, args.nhid // args.nhead, 
                                     stride=args.cache_L, 
                                     padding=math.ceil((args.nhid // args.nhead - args.nhead) / 2))
        elif self.args.summary_method == "linear":
            self.dk = args.cache_dk
            self.summary = nn.Sequential(nn.Linear(args.nhid * args.cache_L, args.cache_dk), nn.Tanh())

        self.attn = DotProductAttention()

        self.batch_size = args.batch_size
        self.L = self.args.cache_L
        self.N = self.args.cache_N
        self.dv = self.args.nhid
        self.topk = self.args.cache_k
        self.theta = self.args.cache_theta

        self.pos_emb = PositionalEmbedding(args.nhid)


    def new_key_and_values(self, text):
        batch_size = text.size(1)
        cache_key = torch.zeros(self.N, 
                                batch_size,
                                self.dk).cuda()
        nn.init.normal_(cache_key, std=self.args.init_std)
        cache_value = torch.zeros(self.N, 
                                  batch_size,
                                  self.L,
                                  self.dv * (self.args.nlayers + 1)).cuda()

        return cache_key, cache_value

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, query, keys):

        if self.args.summary_method == "no_summary" and not self.L == query.size(1):
            padding = query.new_zeros(query.size(0), 
                                      self.L - query.size(1), 
                                      *query.size()[2:])
            query = torch.cat((query, padding), dim=1)
        query = query.transpose(1, 2)
        query_len, bsz = query.size(0), query.size(1)
        query = query.reshape(query_len, bsz, -1)
        
        keys = keys.transpose(0, 1)
        
        if self.args.max_pooling:
            pooling_keys = keys.view(-1, self.N, self.L, self.args.nhid)
            attention = torch.einsum("ibh,bnjh->ijbn", query, pooling_keys)
            attention = attention.max(1)[0]
            attention = F.softmax(self.theta * attention / np.sqrt(pooling_keys.size(-1)), 2)
        else:
            attention = self.attn(query, keys, scale=self.theta) 

        _, topk_indices = attention.topk(self.topk, dim=-1)
        topk_indices = topk_indices.permute(2, 0, 1)

        return attention, topk_indices


    def renew(self, inputs, words=None, cache_info=None, keys=None, values=None, cache_eos=None):

        if keys is None and values is None:
            keys, values = self.new_key_and_values(inputs)

        keys, values = keys.detach(), values.detach()
        
        input_blocks = inputs.split(self.L, dim=2)
    
        key_blocks = list(keys.split(1))
        value_blocks = list(values.split(1))

        if cache_eos is not None:
            sent_infos = cache_eos - torch.cat((torch.zeros_like(cache_eos[0]).unsqueeze(0), cache_eos[:-1]))
            sent_infos = sent_infos.split(1, dim=0)
        else:
            sent_infos = [cache_info.new_ones(1, cache_info.size(1)) * self.L for i in input_blocks]

        if self.args.sentence_cache:
            # compute distance of sentences
            if len(sent_infos) <= self.args.neighbor_len:
                # all sentences into cache is a part of neighbor
                sent_infos = cache_info[self.N:self.N+len(sent_infos),:,:2]
            else:
                # some sentences into cache is a part of input
                sent_infos_nei = cache_info[self.N:,:,:2]
                sent_infos_inp = torch.cat(sent_infos[self.args.neighbor_len:], dim=0)
                sent_dis_inp = -torch.matmul(
                    torch.tril(sent_infos_inp.new_ones(sent_infos_inp.size(0), sent_infos_inp.size(0)), diagonal=-1),
                    sent_infos_inp)
                sent_infos_inp = torch.cat((sent_infos_inp.unsqueeze(-1), sent_dis_inp.unsqueeze(-1)), dim=-1)
                sent_infos = torch.cat((sent_infos_nei, sent_infos_inp.to(sent_infos_nei)), dim=0)
        else:
            nei_len = cache_info[-1,0,0].item()
            cache_len = inputs.size(2)
            sent_len = torch.cat(sent_infos, dim=0).unsqueeze(-1)
            sent_dis = torch.arange(nei_len, nei_len - cache_len, step=-self.L).to(sent_len)
            sent_infos = torch.cat((sent_len, sent_dis[:,None,None].expand_as(sent_len)), dim=-1)

        sent_infos = sent_infos.split(1, dim=0)
            

        for input_block, sent_info in list(zip(input_blocks, sent_infos)):    
            if self.args.merge:
                key_blocks, value_blocks, cache_info = self.merge(key_blocks, 
                                                                  value_blocks, 
                                                                  cache_info, sent_info)
            elif self.args.discard_worst:
                key_blocks, value_blocks, cache_info = self.discard_worst(key_blocks, 
                                                                          value_blocks, 
                                                                          cache_info,
                                                                          sent_info)
            else:
                key_blocks, value_blocks, cache_info = self.eliminate_last(key_blocks, 
                                                                           value_blocks, 
                                                                           cache_info,
                                                                           sent_info)

            if self.args.summary_method == "no_summary":
                new_key = input_block[-1].reshape(self.batch_size, -1)
            elif self.args.summary_method == "sum":
                new_key = input_block[-1].sum(dim=1)
            elif self.args.summary_method == "max":
                new_key, _ = input_block[-1].max(dim=1)
            elif self.args.summary_method == "mean":
                new_key = input_block[-1].mean(dim=1)
            elif self.args.summary_method == "last_state":
                new_key = input_block[-1,:,-1,:]
            elif self.args.summary_method == "weighted_sum":
                new_key = self.summary(input_block[-1].transpose(1, 2)).squeeze(-1)
            elif self.args.summary_method == "conv":
                key_base = input_block[-1].reshape(*input_block.size()[1:3], self.args.nhead, -1)
                key_base = key_base.transpose(1, 2)
                key_base = key_base.reshape(*key_base.size()[:2], -1)
                new_key = self.summary(key_base).reshape(key_base.size(0), -1)
            elif self.args.summary_method == "linear":
                new_key = self.summary(input_block[-1].reshape(-1, self.L * self.dv))

            new_value = input_block.permute(2, 1, 0, 3).reshape(self.L, -1, (self.args.nlayers+1) * self.dv)

            key_blocks[-1] = new_key.unsqueeze(0).detach()
            value_blocks[-1] = new_value.transpose(0, 1).unsqueeze(0).detach()
            
        keys = torch.cat(key_blocks, 0)
        values = torch.cat(value_blocks, 0)
        
        return keys, values, cache_info.detach()


    def eliminate_last(self, key_blocks, value_blocks, cache_info, sent_info):

        for i in range(self.N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
            cache_info[i] = cache_info[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])
        cache_info[self.N-1,:,:2] = sent_info

        return key_blocks, value_blocks, cache_info

    def merge(self, key_blocks, value_blocks, cache_info, sent_info):
       
        # merge is not compatible with sentence_cache, since sentences are not in same length, we do not update length when merge, the length stay at L.
        eli_key = key_blocks[0]
        eli_value = value_blocks[0]

        device = eli_key.device
        alpha = self.args.merge_alpha

        key_blocks[0] = alpha * eli_key + (1 - alpha) * key_blocks[1]
        value_blocks[0] = alpha * eli_value + (1 - alpha) * value_blocks[1]

        for i in range(1, self.N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
            cache_info[i] = cache_info[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])

        length, pos, recall, = cache_info.split([1,1,2], dim=-1)
        pos = pos.squeeze(-1)
        merge_matrix = torch.eye(pos.size(0),
                                 pos.size(0) - 1,
                                 device=pos.device)
        merge_matrix = torch.cat((merge_matrix.new_zeros(pos.size(0), 1), 
                                  merge_matrix), dim=1)
        merge_matrix[0][0], merge_matrix[0][1] = alpha, 1 - alpha
        pos = torch.matmul(merge_matrix, pos + 1)
        pos = pos.unsqueeze(-1)
        cache_info = torch.cat((length, pos, recall), dim=-1)
 
        cache_info[self.N-1,:,:2] = sent_info
        cache_info[0,:,0] = torch.ones_like(sent_info) * self.L

        return key_blocks, value_blocks, cache_info

    def discard_worst(self, key_blocks, value_blocks, cache_info, sent_info):

        nei_info = cache_info[self.N:]
        lengths, pos, recalls, queries = cache_info[:self.N].chunk(4, dim=-1)

        recall_mean = recalls / queries

        # discard the least used block
        eli_indice = recall_mean.squeeze(-1).argmin(dim=0)
        for b, eli_index in enumerate(eli_indice):
            for i in range(eli_index, self.N - 1):
                lengths[i,b,:] = lengths[i+1,b,:]
                pos[i,b,:] = pos[i+1,b,:]
                recalls[i,b,:] = recalls[i+1,b,:]
                queries[i,b,:] = queries[i+1,b,:]
                key_blocks[i][:,b,:] = key_blocks[i+1][:,b,:]
                value_blocks[i][:,b,:,:] = value_blocks[i+1][:,b,:,:]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])

        lengths[-1] = sent_info.squeeze(0)[:,:1]
        pos[-1] = sent_info.squeeze(0)[:,1:]
        recalls[-1] = torch.zeros_like(recalls[0])
        queries[-1] = torch.zeros_like(queries[0])

        cache_info = torch.cat((lengths, pos, recalls, queries), dim=-1)
        cache_info = torch.cat((cache_info, nei_info), dim=0)

        return key_blocks, value_blocks, cache_info

