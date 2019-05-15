import sys
import math
import ipdb
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
from CRTN.utils.adaptive import AdaptiveEmbedding


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

    def forward(self, x, pos_emb, mask=None, memory=None, weights=None):

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
        attn_vec = attn_vec.contiguous().view(seq_len, batch_size, self.num_head * self.d_head)

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

        self.lin_qkv = nn.Linear(d_model, 3 * num_head * d_head, bias=False)
        self.lin_o = nn.Linear(num_head * d_head, d_model, bias=False)
        self.lin_relemb = nn.Linear(d_model, num_head * d_head, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)


    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        return x


    def forward(self, x, pos_emb, pos_bias_u, pos_bias_v, mask=None, memory=None, weights=None):
        seq_len = x.size(0)

        if memory is not None:
            c = torch.cat((memory, x), 0)
        else:
            c = x

        total_len, batch_size = c.size(0), c.size(1)

        heads_matrix = self.lin_qkv(c)
        rel_emb_matrix = self.lin_relemb(pos_emb)

        heads_q, heads_k, heads_v = torch.chunk(heads_matrix, 3, dim=-1)

        heads_q = heads_q.view(total_len, batch_size, self.num_head, self.d_head)[-seq_len:]
        heads_k = heads_k.view(total_len, batch_size, self.num_head, self.d_head)
        heads_v = heads_v.view(total_len, batch_size, self.num_head, self.d_head)

        if weights is not None:
            weights = torch.cat((weights, torch.ones_like(weights[:,:,0]).view(batch_size, 1, -1)),2)
            weights = torch.einsum("bik,l->bkl", weights, torch.ones(seq_len, device=weights.device))
            weights = weights.view(batch_size, -1)
            heads_v = torch.einsum("bl,lbnd->lbnd", weights, heads_v)

        rel_emb_matrix = rel_emb_matrix.view(total_len, batch_size, self.num_head, self.d_head)

        heads_qu = heads_q + pos_bias_u
        heads_qv = heads_q + pos_bias_v

        AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, heads_k))
        BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_emb_matrix))
        
        BD = self._rel_shift(BD)


        attn_score = AC + BD
        attn_score.mul_(self.scale)

        if mask is not None and mask.any().item():
            attn_score.masked_fill_(mask[:,:,:,None], -float('inf'))

        attn_prob = F.softmax(attn_score, 1)
        attn_prob = self.drop(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, heads_v)
        attn_vec = attn_vec.contiguous().view(seq_len, batch_size, self.num_head * self.d_head)

        attn_out = self.lin_o(attn_vec)
        attn_out = self.drop(attn_out)

        output = self.layer_norm(x + attn_out)

        return output






class TransformerUnit(nn.Module):
    def __init__(self, num_head, d_model, d_head, d_ff, attn_type, dropout):
        super().__init__()

        self.attn_type = attn_type

        if attn_type == 0:
            self.attn = MultiheadSelfAttention(num_head, d_model, d_head, dropout)
        elif attn_type == 1:
            self.attn = LearnableMultiheadSelfAttention(num_head, d_model, d_head, dropout)


        self.pos_ff = PostionwiseFF(d_model, d_ff, dropout)

    def forward(self, inputs, pos_emb, pos_bias_u=None, pos_bias_v=None, mask=None, memory=None, weights=None):
        
        if self.attn_type == 0:
            output = self.attn(inputs, pos_emb, mask=mask, memory=memory, weights=weights)
        elif self.attn_type == 1:
            output = self.attn(inputs, pos_emb, pos_bias_u, pos_bias_v, mask=mask, memory=memory, weights=weights)

        output = self.pos_ff(output)

        return output


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
            self.embedding = AdaptiveEmbedding(vocab_size, d_embedding, d_model, cutoffs, div_val=div_val, init_std=init_std)
        else:
            if args.tied:
                self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=0).from_pretrained(self.decoder.weight)
                self.embedding.weight = self.decoder.weight
            else:
                self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=0)

        self.pos_emb = PostionalEmbedding(d_model)



        #establish pos_emb_bank
        pos_unit = torch.arange((self.args.cache_N+1)*num_steps-1, -1, -1.0) 
        pos_unit = pos_unit.view(self.args.cache_N+1, -1)
        batch = torch.ones(self.batch_size)
        pos_emb_bank = torch.einsum('nl,b->bnl', pos_unit, batch)
        self.register_buffer("pos_emb_bank", pos_emb_bank)

        if attn_type == 1:
            self.pos_bias_u = nn.Parameter(torch.Tensor(num_head, d_head))
            self.pos_bias_v = nn.Parameter(torch.Tensor(num_head, d_head))

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
        #if not self.adaptive:
        #    nn.init.normal_(self.embedding.weight, 0.0, init_std)
        #    nn.init.normal_(self.decoder.weight, 0.0, init_std)

    def init_hidden(self, batch_size):
        return self.init_memory(batch_size)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def init_memory(self, batch_size):
        if self.args.mem_len > 0:
            param = next(self.parameters())
            return torch.empty(self.args.nlayers+1, self.args.mem_len, batch_size, self.args.nhid, dtype=param.dtype, device=param.device)
        else:
            return None

    def forward(self, inputs, zones=None, weights=None, indices=None, words=None):
        seq_len, batch_size = inputs.size()
        length = torch.tensor([seq_len] * batch_size, dtype=torch.int64)

        #inputs = pack_padded_sequence(inputs, length)

        if self.demo:
            print("-" * 89)
            display = tuple(zip(indices.view(-1), weights.view(-1), words))
            display = sorted(display, key=lambda x:x[0].item())


        if zones is None:
            zones = self.init_memory(batch_size)

        if zones is not None:
            zones = zones.transpose(1, 2).contiguous()
            if self.demo:
                zones = zones.view(self.args.cache_N * self.args.num_steps, -1, self.args.nlayers+1, self.args.nhid)
            else:
                zones = zones.view(self.args.cache_k * self.args.num_steps, -1, self.args.nlayers+1, self.args.nhid)
            zones = torch.einsum("mblh->lmbh", zones)
            mem_len = zones[0].size(0)
        else:
            mem_len = 0

        total_len = seq_len + mem_len

        word_emb = self.embedding(inputs)

        mask = torch.triu(word_emb.new_ones(seq_len, total_len), diagonal=1+mem_len)
        #mask = torch.cat((word_emb.new_ones(mem_len, total_len), mask), 0)
        mask = mask.byte()[:,:,None]

        ### POSITION SCHEME ###
        indices = torch.cat((indices, (torch.ones(self.batch_size, dtype=indices.dtype, device=indices.device)-2).view(1,-1)))

        if self.demo:
            batch = torch.einsum('k,b->kb',torch.ones(self.args.cache_N+1, dtype=torch.long), torch.arange(self.batch_size))
        else:
            batch = torch.einsum('k,b->kb',torch.ones(self.args.cache_k+1, dtype=torch.long), torch.arange(self.batch_size))

        pos_seq = self.pos_emb_bank[batch, indices].transpose(0, 1).contiguous()
        pos_seq = pos_seq.view(self.batch_size, -1)

        pos_emb = self.pos_emb(pos_seq)

        pos_emb = self.drop(pos_emb)
        core_out = self.drop(word_emb)
        
        if zones is not None:
            memory = word_emb.clone().detach()

        memories = [memory]
        for i, layer in enumerate(self.layers):
            zone_i = None if zones is None else zones[i]
            if self.args.attn_type == 0:
                core_out = layer(core_out, pos_emb, mask=mask, memory=zone_i, weights=weights)
            elif self.args.attn_type == 1:
                core_out = layer(core_out, pos_emb, self.pos_bias_u, self.pos_bias_v, mask=mask, memory=zone_i, weights=weights)

            if zones is not None:
                mem = core_out.detach()
                memories.append(mem)
        memories = torch.cat(memories, 0).view(self.args.nlayers+1, seq_len, -1, self.args.nhid)

        if self.demo:
            for ind, weight, words in display:
                print("SEGMENT %s | weight: %.3f" % (ind.item(), weight.item()))
                for word in words.view(-1):
                    print(self.corpus.vocabulary.index2word[word.item()], end=" ")
                print("\n")
            print("当前输入序列：")
            for word in inputs[:,0]:
                print(self.corpus.vocabulary.index2word[word.item()], end=" ")
            print("")


        core_out = self.drop(core_out)
        
        if not self.args.adaptive:
            output = self.decoder(core_out)
            output = self.drop(output)
        else:
            output = core_out

        #output = pad_packed_sequence(output)


        return output, memories
