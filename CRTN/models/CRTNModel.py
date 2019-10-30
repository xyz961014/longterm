import sys
import math
import ipdb
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
from CRTN.utils.adaptive import AdaptiveEmbedding

from CRTN.layers.transformer import TransformerLM
from CRTN.layers.cache import Cache

import torchsnooper

class CRTNModel(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.corpus = corpus
        self.demo = self.args.demo

        self.drop = nn.Dropout(args.dropout)

        self.cache = Cache(args, corpus)

        self.encoder = TransformerLM(args, corpus)

        if args.query_method == "linear":
            if args.farnear:
                self.shorten = nn.Linear(self.args.num_steps + self.args.neighbor_len, 
                                         self.args.num_steps)
            else:
                self.shorten = nn.Linear(2 * self.args.num_steps, self.args.num_steps)

    def to(self, device):
        super().to(device)
        self.cache.to(device)
        self.encoder.to(device)

    def set_batch_size(self, batch_size):
        self.cache.set_batch_size(batch_size)
        self.encoder.set_batch_size(batch_size)
        self.args.batch_size = batch_size

    def forward(self, inputs, key_num, cache_key, cache_value, draw=False, renew=True, neighbor_mem=None):
        seq_len = self.args.num_steps
        bsz = inputs.size(1)
        nhid = self.args.nhid

        # get cache key and value
        self.cache.init_key_and_value(cache_key, cache_value)

        #self.cache.testtensor.copy_(self.cache.testtensor + 1)
        #print(self.cache.testtensor)
        #if self.cache.testtensor[0,0].item() > 5:
        #    print(self.cache.test2) 
        #self.cache.register_buffer("test2", self.cache.testtensor)
        #print(self.cache.key4[0][0])
        
        if self.args.farnear:
            nei_len = self.args.neighbor_len
            if neighbor_mem is not None:
                neighbor_mem = neighbor_mem.reshape(self.args.nlayers+1,
                                                    self.args.neighbor_len,
                                                    bsz, nhid)
            else:
                neighbor_mem = torch.zeros(self.args.nlayers+1,
                                           self.args.neighbor_len,
                                           bsz, nhid, device=inputs.device)

        if self.args.wise_summary:
            if self.args.query_method == "vanilla":
                _, wise_inputs, _ = self.encoder(inputs)
                query = wise_inputs[-1]
                query = query.expand(query.size(0), -1, -1, -1)
                mask = torch.triu(query.new_ones(seq_len, seq_len), diagonal=1)
                if torch.__version__ < "1.2.0":
                    mask = mask.byte()[:,:,None,None]
                else:
                    mask = mask.bool()[:,:,None,None]
                query = query.masked_fill(mask, 0)
            elif self.args.query_method == "fixed_length":
                if self.args.farnear:
                    if self.args.neighbor_len < self.args.num_steps:
                        raise ValueError("neighbor_len < num_steps, "
                                         "not compatible with method fixed_length")
                    else:
                        prev = neighbor_mem[0].split([nei_len-seq_len, 
                                                      seq_len], dim=0)[1]
                else:
                    prev = self.cache._get_values()[-1]
                    prev = prev.view(seq_len, -1, self.args.nlayers+1, nhid)
                    prev = prev[:,:,0,:]
                inputs = self.encoder.embedding(inputs)
                new_input = torch.cat((prev, inputs), 0)
                index_matrix = torch.arange(seq_len, 
                                            device=new_input.device).unsqueeze(0)
                index_matrix = index_matrix.expand(seq_len, -1)
                index_matrix = index_matrix.t() + index_matrix + index_matrix.new_ones(seq_len, seq_len) 
                index_matrix = index_matrix.view(-1, 1, 1)
                index_matrix = index_matrix.expand(-1, bsz, nhid)
                query = torch.gather(new_input, 0, index_matrix)
                query = query.view(seq_len, seq_len, -1, nhid)
            #elif self.args.query_method == "fixed_length_1":
            #    prev = self.cache._get_values()[-1]
            #    prev = prev.view(seq_len, -1, self.args.nlayers+1, nhid)
            #    prev = prev[:,:,0,:]
            #    inputs = self.encoder.embedding(inputs)
            #    new_input = torch.cat((prev, inputs), 0)
            #    index_matrix = torch.arange(seq_len, device=new_input.device).unsqueeze(0)
            #    index_matrix = index_matrix.expand(seq_len, -1)
            #    index_matrix = index_matrix.t() + index_matrix + index_matrix.new_ones(seq_len, seq_len) 
            #    index_matrix = index_matrix.view(-1, 1, 1)
            #    index_matrix = index_matrix.expand(-1, bsz, nhid)
            #    query = torch.gather(new_input, 0, index_matrix)
            #    query = query.view(seq_len, seq_len, bsz, nhid)
            #    query = query.transpose(0, 1).contiguous()
            #    query = query.view(seq_len, seq_len * bsz, nhid)
            #    _, query, _ = self.encoder(query)
            #    query = query[-1].view(seq_len, seq_len, bsz, nhid)
            #elif self.args.query_method == "fixed_length_2":
            #    new_inputs = inputs.expand(seq_len, -1, -1)
            #    prev_value = self.cache._get_values()[-1]
            #    prev_value.unsqueeze_(0)
            #    prev = prev_value.expand(seq_len, -1, -1, -1)
            #    input_mask = torch.triu(new_inputs.new_ones(seq_len, seq_len), 
            #                                                    diagonal=1)
            #    memory_mask = torch.tril(prev.new_ones(seq_len, seq_len), diagonal=0)
            #    input_mask = input_mask.bool()[:,:,None]
            #    memory_mask = memory_mask.bool()[:,:,None,None]
            #    new_inputs.masked_fill_(input_mask, 0)
            #    prev.masked_fill_(memory_mask, 0)
            #    new_inputs = new_inputs.transpose(0, 2).contiguous()
            #    new_inputs = new_inputs.view(seq_len, seq_len * bsz)
            #    prev = prev.transpose(1, 2).contiguous()
            #    prev = prev.view(1, seq_len * bsz, seq_len, (self.args.nlayers+1)*nhid)
            #    prev_indice = torch.zeros_like(new_inputs).view(-1)
            #    prev_indice.unsqueeze_(0)
            #    _, wise_inputs, _ = self.encoder(new_inputs, values=prev_value, 
            #                                        indices=prev_indice)
            #    wise_inputs = wise_inputs[-1]
            #    prev = prev.view(-1, seq_len, self.args.nlayers+1, nhid)
            #    prev = prev[:,:,-1,:]
            #    prev.transpose_(0, 1)
            #    query_base = torch.cat((prev, wise_inputs), 0)
            #    query_base = query_base.view(query_base.size(0), seq_len, -1, nhid)
            #    query_base = query_base.transpose(0, 1).contiguous()
            #    query_base = query_base.view(query_base.size(0) * seq_len, -1, nhid)
            #    index_range = torch.arange(seq_len, device=new_inputs.device).unsqueeze(0)
            #    index_matrix = index_range.expand(seq_len, -1)
            #    index_matrix = index_matrix.t() + index_matrix + index_matrix.new_ones(seq_len, seq_len) + index_range.t().expand(-1, seq_len) * 2 * seq_len 
            #    index_matrix = index_matrix.view(-1, 1, 1)
            #    index_matrix = index_matrix.expand(-1, query_base.size(1), 
            #                                        query_base.size(2))
            #    query = torch.gather(query_base, 0, index_matrix)
            #    
            #    query = query.view(seq_len, seq_len, -1, nhid)
            else:
                if self.args.farnear:
                    prev_value = torch.einsum("nlbh->lbnh", neighbor_mem)
                    prev_value = prev_value.reshape(1, self.args.neighbor_len, bsz, 
                                                    (self.args.nlayers+1) * nhid)
                    _, wise_inputs, _ = self.encoder(inputs, 
                                                     neighbor_mem=neighbor_mem)
                else:
                    prev_value = self.cache._get_values()[-1]
                    prev_value.unsqueeze_(0)
                    prev_indice = torch.zeros_like(inputs).view(-1)
                    prev_indice.unsqueeze_(0)
                    _, wise_inputs, _ = self.encoder(inputs, values=prev_value,
                                                     indices=prev_indice)
                if self.args.query_method == "last_l":
                    query = wise_inputs[-1]
                    query = query.expand(seq_len, seq_len, bsz, nhid)
                    #query = torch.einsum("lbd,k->klbd",query, 
                    #                     torch.ones_like(query[:,0,0]))
                    mask = torch.triu(query.new_ones(seq_len, seq_len), diagonal=1)
                    if torch.__version__ < "1.2.0":
                        mask = mask.byte()[:,:,None,None]
                    else:
                        mask = mask.bool()[:,:,None,None]
                    query = query.masked_fill(mask, 0)
                elif self.args.query_method == "middle_l":
                    if self.args.farnear and self.args.neighbor_len < seq_len:
                        raise ValueError("neighbor_len < num_steps, "
                                         "not compatible with method middle_l")
                    wise_inputs = wise_inputs[-1]
                    prev = prev_value.transpose(1, 2).contiguous()
                    prev = prev.view(-1, prev.size(2), self.args.nlayers+1, nhid)
                    prev = prev[:,:,-1,:]
                    prev.transpose_(0, 1)
                    query_base = torch.cat((prev, wise_inputs), 0)
                    index_range = torch.arange(seq_len, 
                                               device=inputs.device).unsqueeze(0)
                    index_matrix = index_range.expand(seq_len, -1)
                    index_matrix = (index_matrix.t() + index_matrix + 
                                    index_matrix.new_ones(seq_len, seq_len)) 
                    index_matrix = index_matrix.view(-1, 1, 1)
                    index_matrix = index_matrix.expand(-1, query_base.size(1), 
                                                       query_base.size(2))
                    if self.args.farnear:
                        index_matrix = index_matrix + nei_len - seq_len
                    query = torch.gather(query_base, 0, index_matrix)
                    query = query.view(seq_len, seq_len, -1, nhid)
                elif self.args.query_method == "linear":
                    wise_inputs = wise_inputs[-1]
                    prev = prev_value.transpose(1, 2).contiguous()
                    prev = prev.view(bsz, prev.size(2), self.args.nlayers+1, nhid)
                    prev = prev[:,:,-1,:]
                    prev = prev.transpose(0, 1)
                    query_base = torch.cat((prev, wise_inputs), 0)
                    query_base = query_base.transpose(0, 2)
                    query_base = query_base.expand(seq_len, -1, -1, -1)
                    query = torch.sigmoid(self.shorten(query_base))
                    query = torch.einsum("khbl->klbh", query)
        else:
            query = self.encoder.embedding(inputs)
            query = query.expand(query.size(0), -1, -1, -1)
            mask = torch.triu(query.new_ones(seq_len, seq_len), diagonal=1)
            if torch.__version__ < "1.2.0":
                mask = mask.byte()[:,:,None,None]
            else:
                mask = mask.bool()[:,:,None,None]
            query.masked_fill_(mask, 0)

        if self.demo:
            weights, indices, words = self.cache(query)
        else:
            #weights, indices, zones = self.cache(query)
            weights, indices = self.cache(query)
            words = None

        if key_num is None:
            key_num = list(range(self.args.cache_N - 1, -1, -1))

        #output, mems, attn_map = self.encoder(inputs, zones, weights, indices, words, draw)
        values = self.cache._get_values()

        if self.args.not_weighted:
            weights = None

        output, mems, attn_map = self.encoder(inputs, key_num, values, weights, 
                                              indices, words, draw, neighbor_mem)
        
        if self.args.farnear:
            total_mem = torch.cat((neighbor_mem, mems), 1)
            mems, neighbor_mem = total_mem.split([seq_len, self.args.neighbor_len], 
                                                 dim=1)
            #mems = total_mem[:,:seq_len,:,:]
            #neighbor_mem = total_mem[:,seq_len:,:,:]
            neighbor_mem = neighbor_mem.reshape(-1, bsz, nhid)

        if renew:
            new_key_num = self.cache.renew(mems, inputs, key_num)

        #print("after:", self.cache.key4[0][0])

        keys = self.cache._get_keys()
        values = self.cache._get_values()
        values.transpose_(1, 2)

        if self.args.farnear:
            return output, neighbor_mem, new_key_num, (keys, values)
        else:
            return output, new_key_num, (keys, values) 



