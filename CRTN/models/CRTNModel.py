import sys
import math
import ipdb
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from CRTN.utils.adaptive import AdaptiveEmbedding

from CRTN.layers.transformer import TransformerLM
from CRTN.layers.cache import Cache

class CRTNModel(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.corpus = corpus
        self.demo = self.args.demo
        self.name = "CRTN"

        self.cache = Cache(args, corpus)

        self.encoder = TransformerLM(args, corpus)

        self.theta = args.theta
        self.theta_alpha = args.theta_annealing_alpha

        if args.query_method == "linear":
            if args.farnear:
                if args.summary_method == "no_summary":
                    self.shorten = nn.Linear(args.num_steps + args.neighbor_len, 
                                             args.cache_L)
                elif args.summary_method == "linear":
                    self.shorten = nn.Linear((args.num_steps + args.neighbor_len) * args.nhid, 
                                              args.cache_dk) 
                else:
                    self.shorten = nn.Linear(args.num_steps + args.neighbor_len, 1) 
            else:
                self.shorten = nn.Linear(2 * args.num_steps, args.num_steps)
        elif args.query_method == "single_linear":
            self.shorten = nn.Linear(args.nhid, args.cache_dk)

    def to(self, device):
        super().to(device)
        self.cache.to(device)
        self.encoder.to(device)

    def theta_annealing_step(self):
        self.theta = self.theta * self.theta_alpha
        self.cache.theta = self.theta
        self.encoder.theta = self.theta

    def set_theta(self, cache_theta=1.0, attn_theta=1.0):
        self.cache.theta = cache_theta
        self.encoder.theta = attn_theta

    def set_batch_size(self, batch_size):
        self.cache.set_batch_size(batch_size)
        self.encoder.set_batch_size(batch_size)
        self.args.batch_size = batch_size

    def forward(self, inputs, cache_key, cache_value, cache_info=None, draw=False, neighbor_mem=None, inf_ind=None, inf_blocks=None, **kwargs):
        """
        inputs: num_steps * batch_size, with padding behind in sentence_cache mode
        cache_key: cache_N * batch_size * (num_steps * nhid)
        cache_value: cache_N * batch_size * num_steps * ((nlayers+1) * nhid)
        cache_info: cache_N * batch_size * 3 [pos, recalls, queries]
        neighbor_mem: (nlayers+1) * batch_size * nei_len * nhid
        inf_blocks: (nlayers+1) * batch_size * num_steps * nhid
        """
        ### preparing ###
        seq_len, bsz = inputs.size(0), inputs.size(1)
        nhid = self.args.nhid

        if cache_key is None and cache_value is None:
            cache_key, cache_value = self.cache.new_key_and_values(inputs)
        
        if self.args.farnear:
            nei_len = self.args.neighbor_len
            if neighbor_mem is not None:
                neighbor_mem = neighbor_mem.reshape(self.args.nlayers+1,
                                                    -1,
                                                    bsz, nhid)
            else:
                if not self.args.sentence_cache:
                    neighbor_mem = torch.zeros(self.args.nlayers+1,
                                               nei_len,
                                               bsz, nhid, device=inputs.device)
                else:
                    neighbor_mem = torch.zeros(self.args.nlayers+1,
                                               0,
                                               bsz, nhid, device=inputs.device)

        ### computing query ###

        if self.args.query_method == "vanilla":
            near_output, wise_inputs, _ = self.encoder(inputs, cache_info=cache_info)
            query = wise_inputs[-1]
            mask = torch.triu(query.new_ones(seq_len, seq_len), diagonal=1)
            if inf_ind is None:
                query = query.expand(query.size(0), -1, -1, -1)
            else:
                query = query.unsqueeze(0)
                mask = mask[inf_ind].unsqueeze(0)
            mask = mask.bool()[:,:,None,None]
            query = query.masked_fill(mask, 0)
        else:
            if self.args.farnear:
                near_output, wise_inputs, _ = self.encoder(inputs,
                                                           cache_info=cache_info,
                                                           neighbor_mem=neighbor_mem)
            else:
                prev_value = cache_value[-1].transpose(0, 1)
                prev_value.unsqueeze_(0)
                prev_indice = torch.zeros_like(inputs).view(-1)
                prev_indice.unsqueeze_(0)
                near_output, wise_inputs, _ = self.encoder(inputs,
                                                           cache_info=cache_info,
                                                           values=prev_value,
                                                           indices=prev_indice)
            if self.args.query_method == "last_l":
                query = wise_inputs[-1]
                mask = torch.triu(query.new_ones(seq_len, seq_len), diagonal=1)
                if inf_ind is None:
                    query = query.expand(seq_len, seq_len, bsz, nhid)
                else:
                    query = query.unsqueeze(0)
                    mask = mask[inf_ind].unsqueeze(0)
                mask = mask.bool()[:,:,None,None]
                query = query.masked_fill(mask, 0)
            elif self.args.query_method == "middle_l":
                if self.args.farnear and self.args.neighbor_len < seq_len:
                    raise ValueError("neighbor_len < num_steps, "
                                     "not compatible with method middle_l")
                wise_inputs = wise_inputs[-1]
                if not self.args.farnear:
                    prev = prev_value.transpose(1, 2).contiguous()
                    prev = prev.view(-1, prev.size(2), self.args.nlayers+1, nhid)
                    prev = prev[:,:,-1,:]
                    prev.transpose_(0, 1)
                else:
                    prev = neighbor_mem[-1]
                query_base = torch.cat((prev, wise_inputs), 0)
                index_range = torch.arange(seq_len, 
                                           device=inputs.device).unsqueeze(0)
                if inf_ind is None:
                    index_matrix = index_range.expand(seq_len, -1)
                    index_matrix = (index_matrix.t() + index_matrix + 
                                    index_matrix.new_ones(seq_len, seq_len)) 
                else:
                    index_matrix = index_range + inf_ind + 1
                index_matrix = index_matrix.view(-1, 1, 1)
                index_matrix = index_matrix.expand(-1, query_base.size(1), 
                                                   query_base.size(2))
                if self.args.farnear:
                    index_matrix = index_matrix + nei_len - seq_len
                query = torch.gather(query_base, 0, index_matrix)
                if inf_ind is None:
                    query = query.view(seq_len, seq_len, -1, nhid)
                else:
                    query = query.view(1, seq_len, -1, nhid)
            elif self.args.query_method == "linear":
                wise_inputs = wise_inputs[-1]
                if not self.args.farnear:
                    prev = prev_value.transpose(1, 2).contiguous()
                    prev = prev.view(bsz, prev.size(2), self.args.nlayers+1, nhid)
                    prev = prev[:,:,-1,:]
                    prev = prev.transpose(0, 1)
                else:
                    prev = neighbor_mem[-1]
                query_base = torch.cat((prev, wise_inputs), 0)
                query_base = query_base.transpose(0, 2)
                mask = torch.triu(query_base.new_ones(
                                    seq_len, query_base.size(-1)), 
                                    diagonal=1+query_base.size(-1)-seq_len)
                if inf_ind is None:
                    query_base = query_base.expand(seq_len, -1, -1, -1)
                else:
                    query_base = query_base.unsqueeze(0)
                    mask = mask[inf_ind].unsqueeze(0)
                mask = mask.bool()[:,None,None,:]
                query_base = query_base.masked_fill(mask, 0)
                if self.args.summary_method == "linear":
                    query_base = query_base.permute(0, 2, 3, 1).reshape(seq_len, bsz, -1)
                query = torch.tanh(self.shorten(query_base))
                if self.args.summary_method == "linear":
                    query = query.unsqueeze(1)
                else:
                    query = torch.einsum("khbl->klbh", query)
            elif self.args.query_method == "single":
                wise_inputs = wise_inputs[-1][:,None,:,:]
                if inf_ind is not None:
                    wise_inputs = wise_inputs[inf_ind].unsqueeze(0)
                query = wise_inputs
            elif self.args.query_method == "single_linear":
                wise_inputs = wise_inputs[-1][:,None,:,:]
                query = torch.tanh(self.shorten(wise_inputs))

        ### look up from cache ###
        
        if self.demo:
            weights, indices, words = self.cache(query, cache_key)
        else:
            weights, indices = self.cache(query, cache_key)
            words = None


        if self.args.not_weighted:
            weights = None

        values = cache_value.transpose(1, 2).contiguous()

        output, hidden, attn_map = self.encoder(inputs, cache_info, values, weights, 
                                                indices, words, draw, neighbor_mem,
                                                inf_ind, inf_blocks)

        
        if self.args.farnear and inf_ind is None:
            if not self.args.sentence_cache:
                total_mem = torch.cat((neighbor_mem, hidden), dim=1)
                if self.cache.L > total_mem.size(1):
                    #print("current segment ({}) not long enough,"
                    #      " padding ({}) added".format(total_mem.size(1), self.cache.L - total_mem.size(1)))
                    padding = total_mem.new_zeros(total_mem.size(0), 
                                                  self.cache.L - total_mem.size(1), 
                                                  *total_mem.size()[2:])
                    total_mem = torch.cat((padding, total_mem), dim=1)

                cache_blocks = max(1, (total_mem.size(1) - nei_len) // self.cache.L)
                cache_len = cache_blocks * self.cache.L
                nei_len = total_mem.size(1) - cache_len
                hidden, neighbor_mem = total_mem.split([cache_len, nei_len], dim=1)
            else:
                hidden = hidden.transpose(1, 2)

        hidden = hidden.transpose(1, 2)
        neighbor_mem = neighbor_mem.reshape(-1, bsz, nhid)

        if self.args.farnear:
            return output, hidden.detach(), neighbor_mem.detach()
        else:
            return output, hidden



