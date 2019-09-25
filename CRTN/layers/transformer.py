import sys
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
from CRTN.utils.adaptive import AdaptiveEmbedding

import ipdb
import torchsnooper
import visdom

class PostionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
            
        self.d_model = d_model

        inverse_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inverse_freq", inverse_freq)

    def forward(self, pos_seq):

        sinusoid = torch.einsum("bi,j->ibj", pos_seq, self.inverse_freq)

        #sinusoid = torch.einsum("i,j->ij", pos_seq, self.inverse_freq)

        pos_embedding = torch.cat((sinusoid.sin(), sinusoid.cos()), -1)

        return pos_embedding[:,:,None,:]



class PostionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.FFNet = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
                )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        
        output = self.FFNet(inputs)
        output = self.layer_norm(inputs + output)

        return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_head, d_model, d_head, dropout):
        super().__init__()

        self.num_head = num_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)

        self.lin_qkv = nn.Linear(d_model, 3 * num_head * d_head, bias=False)
        self.lin_o = nn.Linear(num_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, x, pos_emb, mask=None, memory=None, indices=None, weights=None):

        seq_len = x.size(0)
        
        x = x + pos_emb[-seq_len:]

        if memory is not None:
            c = torch.cat((memory, x), 0)
        else:
            c = x


        total_len, batch_size = c.size(0), c.size(1)

        heads_matrix = self.lin_qkv(c)

        heads_q, heads_k, heads_v = torch.chunk(heads_matrix, 3, dim=-1)

        heads_q = heads_q.view(total_len, batch_size, self.num_head, self.d_head)[-seq_len:]
        heads_k = heads_k.view(total_len, batch_size, self.num_head, self.d_head)
        heads_v = heads_v.view(total_len, batch_size, self.num_head, self.d_head)

        attn_score = torch.einsum("ibnd,jbnd->ijbn", heads_q, heads_k)
        attn_score.mul_(self.scale)

        if mask is not None and mask.any().item():
            attn_score.masked_fill_(mask[:,:,:,None], -float('inf'))

        attn_prob = F.softmax(attn_score, 1)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, heads_v)
        attn_vec = attn_vec.reshape(seq_len, batch_size, self.num_head * self.d_head)

        attn_out = self.lin_o(attn_vec)
        attn_out = self.drop(attn_out)

        output = self.layer_norm(x + attn_out)

        return output

class LearnableMultiheadSelfAttention(nn.Module):
    def __init__(self, num_head, d_model, d_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)

        self.lin_q = nn.Linear(d_model, num_head * d_head, bias=False)
        self.lin_kv = nn.Linear(d_model, 2 * num_head * d_head, bias=False)
        self.lin_o = nn.Linear(num_head * d_head, d_model, bias=False)
        self.lin_relemb = nn.Linear(d_model, num_head * d_head, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)


    def _rel_shift(self, x):
        x_inp = x.reshape(x.size(0), -1, *x.size()[-2:])
        zero_pad = x_inp.new_zeros((x_inp.size(0), 1, *x_inp.size()[2:]))
        x_padded = torch.cat([zero_pad, x_inp], dim=1)

        x_padded = x_padded.view(x_inp.size(1) + 1, x_inp.size(0), *x_inp.size()[2:])

        x = x_padded[1:].view_as(x)

        return x


    def forward(self, x, pos_emb, pos_bias_u, pos_bias_v, mask=None, memory=None, indice_bool=None, weights=None, neighbor_mem=None):
        x_len, batch_size, nhid = x.size(0), x.size(1), x.size(2)
        #if indices is not None:
        #    ipdb.set_trace()
        #seq_len = x_len

        #x_seq = x.unsqueeze(1)
        #x_seq = x_seq.expand(-1, seq_len, -1, -1).contiguous()
        #x_seq = x_seq.view(seq_len, -1, self.d_model)

        #if memory is not None:
        #    if batch_size == memory.size(1):
        #        c = torch.cat((memory, x), 0)
        #        seq_len = 1
        #        vanilla = True
        #    else:
        #        c = torch.cat((memory, x_seq), 0)
        #        vanilla = False
        #else:
        #    c = x
        #    seq_len = 1
        #    vanilla = True

        if memory is not None:
            mem_num = memory.size(0)
            memory = memory.view(mem_num * memory.size(1), -1, nhid)
            #memory = memory.detach()
            if not batch_size == memory.size(1):
                memory.unsqueeze_(1)
                memory = memory.expand(-1, batch_size // memory.size(2), -1, -1)
                memory = memory.reshape(memory.size(0), -1, memory.size(-1))
            c = torch.cat((memory, x), 0)
        else:
            c = x



        heads_matrix = self.lin_kv(c)
        rel_emb_matrix = self.lin_relemb(pos_emb)

        heads_k, heads_v = torch.chunk(heads_matrix, 2, dim=-1)

        heads_q = self.lin_q(x)

        if neighbor_mem is not None:
            nei_len = neighbor_mem.size(0)
            nei_matrix = self.lin_kv(neighbor_mem)
            nei_k, nei_v = torch.chunk(nei_matrix, 2, dim=-1)

            rel_cache, rel_nei, rel_inp = rel_emb_matrix.split([memory.size(0), 
                                                                nei_len, x_len])
            rel_emb_matrix = torch.cat((rel_cache, rel_inp), 0)

        #if indices is not None:
        #    indice_len = indices.size(0)
        #    total_len = indice_len * x_len 

        #    indice_g = indices.view(-1, batch_size)
        #    indice_g = indice_g[:,None,:,None]
        #    indice_g = indice_g.expand(-1, x_len, -1, nhid)

        #    heads_k = heads_k.view(-1, x_len, batch_size, nhid)
        #    heads_k = torch.gather(heads_k, 0, indice_g)
        #    heads_k = heads_k.view(indice_len, -1, x_len, batch_size, nhid)
        #    heads_k = heads_k.transpose(0, 1).contiguous()
        #    heads_k = heads_k.view(-1, indice_len * x_len, batch_size, nhid)

        #    heads_v = heads_v.view(-1, x_len, batch_size, nhid)
        #    heads_v = torch.gather(heads_v, 0, indice_g)
        #    heads_v = heads_v.view(indice_len, -1, x_len, batch_size, nhid)
        #    heads_v = heads_v.transpose(0, 1).contiguous()
        #    if weights is not None:
        #        weights = torch.cat((weights, torch.ones_like(weights[:,:,0]).view(weights.size(0), 1, -1)), 2)
        #        weights.squeeze_()
        #        weights = weights.view(-1, batch_size, indice_len)
        #        heads_v = torch.einsum("jkibh,jbk->jkibh", heads_v, weights)
        #        #heads_v = heads_v.view(-1, indice_len, x_len, batch_size, self.num_head, self.d_head)
        #    heads_v = heads_v.view(-1, indice_len * x_len, batch_size, nhid)

        #    rel_emb_matrix = rel_emb_matrix.view(-1, x_len, batch_size, nhid)
        #    rel_emb_matrix = torch.gather(rel_emb_matrix, 0, indice_g)
        #    rel_emb_matrix = rel_emb_matrix.view(indice_len, -1, x_len, batch_size, nhid)
        #    rel_emb_matrix = rel_emb_matrix.transpose(0, 1).contiguous()
        #    rel_emb_matrix = rel_emb_matrix.view(-1, indice_len * x_len, batch_size, nhid)
        #else:
        #    total_len = c.size(0)


        #if weights is not None:
        #    weights = torch.cat((weights, torch.ones_like(weights[:,:,0]).view(weights.size(0), 1, -1)), 2)
        #    weights = torch.einsum("bik,l->bkl", weights, torch.ones(x_len, device=weights.device))
        #    weights = weights.view(weights.size(0), -1)
        #    heads_v = torch.einsum("bl,lbnd->lbnd", weights, heads_v)

        #heads_v = heads_v.view(total_len, batch_size, self.num_head, self.d_head)


        heads_q = heads_q.view(x_len, batch_size, self.num_head, self.d_head)
        heads_qu = heads_q + pos_bias_u
        heads_qv = heads_q + pos_bias_v

        if indice_bool is None:
            total_len = c.size(0)
            rel_emb_matrix = rel_emb_matrix.view(total_len, batch_size, 
                                                 self.num_head, self.d_head)

            heads_k = heads_k.view(total_len, batch_size, self.num_head, self.d_head)
            heads_v = heads_v.view(total_len, batch_size, self.num_head, self.d_head)
            AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, heads_k))
            BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_emb_matrix))
        else:
            #indice_len = indices.size(0)
            #indices = indices.view(indice_len, -1, batch_size)
            #tfbase = torch.eye(mem_num + 1, device=indices.device)
            #indice_bool = torch.index_select(tfbase, 0, indices.view(-1))
            #indice_bool = indice_bool.view(indice_len, -1, batch_size, mem_num + 1)
            #indice_bool = indice_bool.sum(0)
            pre_AC = torch.einsum("ibnd,ibk->kibnd", heads_qu, indice_bool)
            pre_BD = torch.einsum("ibnd,ibk->kibnd", heads_qv, indice_bool)
            pre_k = heads_k.view(mem_num + 1, x_len, batch_size, 
                                 self.num_head, self.d_head)
            pre_v = heads_v.view(mem_num + 1, x_len, batch_size, 
                                 self.num_head, self.d_head)
            if weights is not None:
                #weights = weights.view(x_len, batch_size, -1)
                #weights = torch.cat((weights, torch.ones_like(weights[:,:,0,None]) * -float("inf")), 2)
                #weights.masked_fill_((1 - indice_bool).bool(), -float("inf"))
                #weights = F.softmax(weights, 2)
                #weights = weights.index_fill(2, (weights.new_ones(1) * mem_num).long(), 1.0)
                pre_v = torch.einsum("kjbnd,jbk->kjbnd", pre_v, weights)

            pre_rel = rel_emb_matrix.view(mem_num + 1, x_len, 
                                          batch_size, self.num_head, self.d_head)
            AC = torch.einsum("kibnd,kjbnd->ikjbn", pre_AC, pre_k)
            BD = torch.einsum("kibnd,kjbnd->ikjbn", pre_BD, pre_rel)

            if neighbor_mem is not None:
                nei_k = nei_k.view(nei_len, batch_size, self.num_head, self.d_head)
                nei_v = nei_v.view(nei_len, batch_size, self.num_head, self.d_head)
                rel_nei = rel_nei.view(nei_len, batch_size, self.num_head, self.d_head)
                nei_AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, nei_k))
                nei_BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_nei))
                
                AC_cache, AC_inp = AC.split(mem_num, dim=1)
                BD_cache, BD_inp = BD.split(mem_num, dim=1)

                AC = torch.cat((AC_cache.reshape(x_len, -1, batch_size, self.num_head),
                                nei_AC,
                                AC_inp.squeeze()), 1)
                BD = torch.cat((BD_cache.reshape(x_len, -1, batch_size, self.num_head),
                                nei_BD,
                                BD_inp.squeeze()), 1)
        
        BD = self._rel_shift(BD)


        attn_score = AC + BD
        attn_score.mul_(self.scale)
        attn_score = attn_score.reshape(x_len, -1, batch_size, self.num_head)

        if mask is not None:
            attn_score.masked_fill_(mask[:,:,:,None], -float('inf'))

        #if memory is not None:
        #    attn_score = attn_score.view(x_len, mem_num + 1, x_len, batch_size, self.num_head)
        #    indice_len = indices.size(0)
        #    indices = indices.view(-1, batch_size)
        #    score_indices = indices[None,:,None,:,None]
        #    score_indices = score_indices.expand(attn_score.size(0), -1, attn_score.size(2), -1, attn_score.size(4))
        #    sel_attn_score = torch.gather(attn_score, 1, score_indices)
        #    sel_attn_score = sel_attn_score.view(x_len, indice_len, -1, x_len, batch_size, self.num_head)
        #    sel_attn_score = sel_attn_score.transpose(2, 3).contiguous()
        #    sel_attn_score = sel_attn_score.view(x_len, -1, x_len, batch_size, self.num_head)
        #    heads_v = heads_v.view(-1, x_len, batch_size, self.num_head, self.d_head)
        #    v_indices = indices[:,None,:,None,None]
        #    v_indices = v_indices.expand(-1, heads_v.size(1), -1, heads_v.size(3), heads_v.size(4))
        #    sel_heads_v = torch.gather(heads_v, 0, v_indices)
        #    sel_heads_v = sel_heads_v.view(indice_len, -1, x_len, batch_size, self.num_head, self.d_head)
        #    sel_heads_v = sel_heads_v.transpose(1, 2).contiguous()
        #    sel_heads_v = sel_heads_v.view(indice_len, x_len, -1, self.num_head, self.d_head)
        #    if weights is not None:
        #        weights = torch.cat((weights, torch.ones_like(weights[:,:,0]).view(weights.size(0), 1, -1)), 2)
        #        weights.squeeze_()
        #        sel_heads_v = torch.einsum("jibnh,bj->jibnh", sel_heads_v, weights)
        #        sel_heads_v = sel_heads_v.view(-1, x_len, batch_size, self.num_head, self.d_head)
        #else:
        #    sel_attn_score = attn_score
        #    sel_heads_v = heads_v

        attn_prob = F.softmax(attn_score, 1)
        attn_prob = self.drop(attn_prob)
        attn_matrix = attn_prob.mean((2, 3))

        #if memory is None:
        #    attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, heads_v)
        #else:
        #    attn_vec = torch.einsum("ijbn,ijbnd->ibnd", attn_prob, heads_v)
        if memory is None:
            attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, heads_v)
        else:
            if neighbor_mem is not None:
                prob_cache, prob_nei, prob_inp = attn_prob.split([memory.size(0),
                                                                  nei_len,
                                                                  x_len], dim=1)
                attn_prob = torch.cat((prob_cache, prob_inp), 1)
                nei_vec = torch.einsum("ilbn,lbnd->ibnd", prob_nei, nei_v)

            attn_prob = attn_prob.reshape(x_len, -1, x_len, batch_size, self.num_head)
            attn_vec = torch.einsum("ikjbn,kjbnd->ibnd", attn_prob, pre_v)

            if neighbor_mem is not None:
                attn_vec += nei_vec

        attn_vec = attn_vec.reshape(x_len, batch_size, self.num_head * self.d_head)

        attn_out = self.lin_o(attn_vec)
        attn_out = self.drop(attn_out)

        output = self.layer_norm(x + attn_out)

        return output, attn_matrix






class TransformerUnit(nn.Module):
    def __init__(self, num_head, d_model, d_head, d_ff, attn_type, dropout):
        super().__init__()

        self.attn_type = attn_type

        if attn_type == 0:
            self.attn = MultiheadSelfAttention(num_head, d_model, d_head, dropout)
        elif attn_type == 1:
            self.attn = LearnableMultiheadSelfAttention(num_head, d_model, 
                                                            d_head, dropout)


        self.pos_ff = PostionwiseFF(d_model, d_ff, dropout)

    def forward(self, inputs, pos_emb, pos_bias_u=None, pos_bias_v=None, mask=None, memory=None, indices=None, weights=None, neighbor_mem=None):
        
        if self.attn_type == 0:
            output = self.attn(inputs, pos_emb, mask=mask, memory=memory, 
                                indice_bool=indices, weights=weights)
        elif self.attn_type == 1:
            output, attn_matrix = self.attn(inputs, pos_emb, pos_bias_u, pos_bias_v, 
                                            mask=mask, memory=memory, 
                                            indice_bool=indices, 
                                            weights=weights, 
                                            neighbor_mem=neighbor_mem)

        output = self.pos_ff(output)

        return output, attn_matrix


class TransformerLM(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        self.args = deepcopy(args)
        self.batch_size = self.args.batch_size

        vocab_size = self.args.vocab_size
        d_embedding = self.args.emsize
        d_model = self.args.nhid
        d_ff = self.args.d_ff
        d_head = self.args.nhid // args.nhead
        num_head = self.args.nhead
        num_layer = self.args.nlayers
        num_steps = self.args.num_steps
        mem_len = self.args.mem_len
        attn_type = self.args.attn_type
        dropout = self.args.dropout
        cutoffs = self.args.cutoffs
        div_val = self.args.div_val
        init_std = self.args.init_std

        attn_type = self.args.attn_type
        adaptive = self.args.adaptive
        
        self.corpus = corpus
        self.demo = self.args.demo


        self.decoder = nn.Linear(d_model, vocab_size, bias=False) 

        if adaptive:
            self.embedding = AdaptiveEmbedding(vocab_size, d_embedding, d_model, 
                                               cutoffs, div_val=div_val, 
                                               init_std=init_std)
        else:
            if args.tied:
                self.embedding = nn.Embedding(vocab_size, d_embedding, 
                                              padding_idx=0).from_pretrained(
                                                      self.decoder.weight)
                self.embedding.weight = self.decoder.weight
            else:
                self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=0)

        self.pos_emb = PostionalEmbedding(d_model)



        #establish pos_emb_bank

        #pos_unit = torch.arange((self.args.cache_N+1)*num_steps-1, -1, -1.0) 
        #pos_unit = pos_unit.view(self.args.cache_N+1, -1)
        #batch = torch.ones(self.batch_size * num_steps)
        #pos_emb_bank = torch.einsum('nl,b->bnl', pos_unit, batch)
        #self.register_buffer("pos_emb_bank", pos_emb_bank)

        if attn_type == 1:
            self.pos_bias_u = nn.Parameter(torch.Tensor(num_head, d_head))
            self.pos_bias_v = nn.Parameter(torch.Tensor(num_head, d_head))

        if args.stat:
            self.select_stat = nn.Parameter(torch.zeros(args.cache_N), 
                                            requires_grad=False)
            self.viz = visdom.Visdom()
            assert self.viz.check_connection()

        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList()

        for i in range(num_layer):
            self.layers.append(TransformerUnit(
                num_head=num_head,
                d_model=d_model,
                d_head=d_head,
                d_ff=d_ff,
                attn_type=attn_type,
                dropout=dropout))


        self.init_weights(init_std)
        #self.criterion = criterion

    def init_weights(self, init_std):
        if self.args.attn_type == 1:
            nn.init.normal_(self.pos_bias_u, 0.0, init_std)
            nn.init.normal_(self.pos_bias_v, 0.0, init_std)
        if not self.args.adaptive:
            nn.init.normal_(self.embedding.weight, 0.0, init_std)
            nn.init.normal_(self.decoder.weight, 0.0, init_std)

    def init_hidden(self, batch_size):
        return self.init_memory(batch_size)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def init_memory(self, batch_size):
        if self.args.mem_len > 0:
            param = next(self.parameters())
            return torch.empty(self.args.nlayers+1, self.args.mem_len, 
                               batch_size, self.args.nhid, 
                               dtype=param.dtype, device=param.device)
        else:
            return None

    def forward(self, inputs, values=None, weights=None, indices=None, words=None, draw=False, neighbor_mem=None):
        #input shape should be seq_len * bsz or seq_len * bsz * emsize
        if inputs.dim() == 2:
            word_emb = self.embedding(inputs)
            seq_len, batch_size = inputs.size()
        else:
            word_emb = inputs
            seq_len, batch_size, _ = inputs.size()

        #length = torch.tensor([seq_len] * batch_size, dtype=torch.int64)
        #inputs = pack_padded_sequence(inputs, length)



        #if zones is not None:
        #    zone_bsz = zones.size(1)
        #    zones = zones.transpose(1, 2).contiguous()
        #    zones = zones.view(-1, zone_bsz, self.args.nlayers+1, self.args.nhid)
        #    zones = torch.einsum("mblh->lmbh", zones)
        #    mem_len = zones[0].size(0)
        #else:
        #    mem_len = 0
        #    zone_bsz = batch_size

        if indices is not None:
            #mem_len = seq_len * indices.size(0)
            mem_len = values.size(0) * values.size(1)
            zone_bsz = indices.size(1)
            values = values.view(values.size(0), values.size(1), -1, 
                                 self.args.nlayers+1, self.args.nhid)
        else:
            mem_len = 0
            zone_bsz = batch_size


        if self.args.not_weighted:
            weights = None

        if neighbor_mem is not None:
            total_len = seq_len + mem_len + neighbor_mem.size(1)
        else:
            total_len = seq_len + mem_len


        mask = torch.triu(word_emb.new_ones(seq_len, total_len), diagonal=1+mem_len) 
        mask = mask.bool()[:,:,None]

        if indices is not None:
            #pos_seq
            pos_indices = torch.cat((indices, 
                                    (torch.ones_like(indices[0]) * values.size(0)
                                    ).view(1, -1)))

            pos_seq = torch.einsum("b,k->bk", 
                                   torch.ones(batch_size, device=inputs.device), 
                                   torch.arange(total_len-1, -1, -1.0, 
                                   device=inputs.device))

            #one-hot pos_indices
            mem_num = values.size(0)
            indice_len = pos_indices.size(0)
            pos_indices = pos_indices.view(indice_len, -1, batch_size)
            tfbase = torch.eye(mem_num + 1, device=pos_indices.device)
            indice_bool = torch.index_select(tfbase, 0, pos_indices.view(-1))
            indice_bool = indice_bool.view(indice_len, -1, batch_size, mem_num + 1)
            indice_bool = indice_bool.sum(0)

            if self.args.stat:
                stat = indice_bool.sum((0, 1))
                self.select_stat += stat[:-1]
                self.viz.bar(self.select_stat, win="select stat")

            if weights is not None:
                x_len = inputs.size(0)
                weights = weights.view(x_len, batch_size, -1)
                weights = torch.cat((weights, 
                                    torch.ones_like(
                                        weights[:,:,0,None]) * -float("inf")), 2)
                weights.masked_fill_((1 - indice_bool).bool(), -float("inf"))
                weights = F.softmax(weights, 2)
                weights = weights.index_fill(
                            2, (weights.new_ones(1) * mem_num).long(), 1.0)
            
            #zones
            #values.unsqueeze_(2)
            #values = values.expand(-1, -1, seq_len, -1, -1)
            #values = values.reshape(values.size(0), seq_len, -1, self.args.nlayers+1, self.args.nhid)
            #values.transpose_(1, 2)
            #indices = indices[:,:,None,None,None]
            #indices = indices.expand(-1, -1, values.size(-3), values.size(-2), values.size(-1))
            #zones = torch.gather(values, 0, indices)
            #zones = torch.einsum("kblyh->yklbh", zones)
            #zones = zones.reshape(zones.size(0), -1, zone_bsz, zones.size(-1))
        else:
            pos_seq = torch.einsum("b,k->bk", 
                                   torch.ones(batch_size, device=inputs.device), 
                                   torch.arange(self.args.num_steps-1, -1, -1.0, 
                                   device=inputs.device))
            pos_indices = indices
            indice_bool = None
        pos_seq = pos_seq.view(batch_size, -1)

        pos_emb = self.pos_emb(pos_seq)

        pos_emb = self.drop(pos_emb)
        core_out = self.drop(word_emb)
        
        memory = word_emb.clone()
        memories = [memory]

        if self.demo and weights is not None:
            demo_display = tuple(zip(indice_bool.squeeze(), weights.squeeze()))

        for i, layer in enumerate(self.layers):
            #zone_i = None if zones is None else zones[i]
            if indices is None:
                value_i = None
            else:
                value_i = values[:,:,:,i,:]

            if neighbor_mem is None:
                neighbor_mem_i = None
            else:
                neighbor_mem_i = neighbor_mem[i]

            if self.args.attn_type == 0:
                core_out = layer(core_out, pos_emb, mask=mask, memory=value_i, 
                                 indices=pos_indices, weights=weights)
            elif self.args.attn_type == 1:
                core_out, attn_matrix = layer(core_out, pos_emb, self.pos_bias_u, 
                                              self.pos_bias_v, 
                                              mask=mask, 
                                              memory=value_i, 
                                              indices=indice_bool, 
                                              weights=weights,
                                              neighbor_mem=neighbor_mem_i
                                              )


            mem = core_out
            memories.append(mem)

        memories = torch.cat(memories, 0)
        memories = memories.view(self.args.nlayers+1, seq_len, -1, self.args.nhid)

        if draw:
            attn_map = attn_matrix
        else:
            attn_map = None



        core_out = self.drop(core_out)
        
        if not self.args.adaptive:
            output = self.decoder(core_out)
            output = self.drop(output)
        else:
            output = core_out

        #output = pad_packed_sequence(output)
        if self.demo and weights is not None:
            id2w = self.corpus.vocabulary.index2word
            words = torch.cat((words.squeeze(), inputs.t()), 0)
            for idis, (ind, weight) in enumerate(demo_display):
                print("-" * 89 + "\n")
                print("Current Segment: ", end="")
                for iinp, wd in enumerate(inputs):
                    if idis == iinp:
                        print("\033[1;32m %s \033[0m" % id2w[wd.item()], end=" ")
                    else:
                        print(id2w[wd.item()], end=" ")
                print("\n")
                for i, wt in enumerate(ind * weight):
                    if i + 1 == len(weight):
                        continue
                    print("SEGMENT %s | weight: %.3f" % (i, wt.item()))
                    for wd in words[i].view(-1):
                        print(id2w[wd.item()], end=" ")
                    print("\n")
            #print("Current input: ")
            #for word in inputs[:,0]:
            #    print(self.corpus.vocabulary.index2word[word.item()], end=" ")
            #print("")
            return output, memories, (attn_map, demo_display)


        return output, memories, attn_map
