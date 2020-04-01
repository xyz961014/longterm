import sys
import math
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
sys.path.append("../..")
from CRTN.utils.adaptive import AdaptiveEmbedding
from CRTN.utils.fancy_dropout import WeightDropLinear
from torchnlp.nn import LockedDropout
try:
    from apex.normalization import FusedLayerNorm
except:
    print("No apex package found")


class PostionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        inverse_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inverse_freq", inverse_freq)

    def forward(self, pos_seq, batch_size=None):
        sinusoid = torch.einsum("i,j->ij", pos_seq, self.inverse_freq)
        pos_embedding = torch.cat((sinusoid.sin(), sinusoid.cos()), -1)

        if batch_size is not None:
            return pos_embedding[:,None,:].expand(-1, batch_size, -1)
        else:
            return pos_embedding[:,None,:]


def bmm_einsum(tensor1, tensor2, eqn="ibnd,jbnd->ijbn"):
    """
    bmm version of 
    ibnd,jbnd->ijbn
    ilbn,lbnd->ibnd
    kibnd,kjbnd->ikjbn

    this version is used to be compatible with apex
    """
    if eqn == "ibnd,jbnd->ijbn":
        assert tensor1.shape[1:] == tensor2.shape[1:]
        bmm1 = tensor1.reshape(tensor1.size(0), -1, tensor1.size(-1))
        bmm2 = tensor2.reshape(tensor2.size(0), -1, tensor2.size(-1))
        bmm1 = bmm1.permute(1,0,2)
        bmm2 = bmm2.permute(1,2,0)
        ret = torch.bmm(bmm1,bmm2)
        ret = ret.view(tensor1.size(1), tensor1.size(2), *ret.shape[1:])
        return ret.permute(2,3,0,1)
    elif eqn == "ilbn,lbnd->ibnd":
        assert tensor1.size(1) == tensor2.size(0) and tensor1.shape[2:] == tensor2.shape[1:3] 
        bmm1 = tensor1.reshape(tensor1.size(0), tensor1.size(1), -1)
        bmm2 = tensor2.reshape(tensor2.size(0), -1, tensor2.size(-1))
        bmm1 = bmm1.permute(2,0,1)
        bmm2 = bmm2.permute(1,0,2)
        ret = torch.bmm(bmm1,bmm2)
        ret = ret.view(tensor1.size(2), tensor1.size(3), *ret.shape[1:])
        return ret.permute(2,0,1,3)
    elif eqn == "kibnd,kjbnd->ikjbn":
        assert tensor1.size(0) == tensor2.size(0) and tensor1.shape[2:] == tensor2.shape[2:]
        bmm1 = tensor1.permute(0,2,3,1,4)
        bmm2 = tensor2.permute(0,2,3,4,1)
        bmm1 = bmm1.reshape(-1, bmm1.size(-2), bmm1.size(-1))
        bmm2 = bmm2.reshape(-1, bmm2.size(-2), bmm2.size(-1))
        ret = torch.bmm(bmm1,bmm2)
        ret = ret.view(tensor1.size(0), tensor1.size(2), tensor1.size(3), *ret.shape[1:])
        return ret.permute(3,0,4,1,2)



class PostionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, drophid, apex):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.FFNet = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(inplace=True),
                LockedDropout(drophid),
                nn.Linear(d_ff, d_model),
                LockedDropout(drophid)
                )

        if apex:
            self.layer_norm = FusedLayerNorm(d_model)
        else:
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

    def forward(self, x, pos_emb, mask=None, memory=None):

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
    def __init__(self, num_head, d_model, d_head, dropout, dropatt, dropwei, drophid, apex=False):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.d_head = d_head
        self.apex = apex

        self.dropout = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.drophid = LockedDropout(drophid)

        self.lin_qkv = WeightDropLinear(d_model, 3 * num_head * d_head, bias=False, 
                                        weight_dropout=dropwei)
        self.lin_relemb = WeightDropLinear(d_model, num_head * d_head, bias=False, 
                                           weight_dropout=dropwei)
        #self.lin_qkv = nn.Linear(d_model, 3 * num_head * d_head, bias=False) 
        #self.lin_relemb = nn.Linear(d_model, num_head * d_head, bias=False,)
        self.lin_o = nn.Linear(num_head * d_head, d_model, bias=False)

        if apex:
            self.layer_norm = FusedLayerNorm(d_model)
        else:
            self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)


    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        return x


    def forward(self, x, pos_emb, pos_bias_u, pos_bias_v, mask=None, memory=None):
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

        rel_emb_matrix = rel_emb_matrix.view(total_len, self.num_head, self.d_head)

        heads_qu = heads_q + pos_bias_u
        heads_qv = heads_q + pos_bias_v

        rel_emb_matrix = rel_emb_matrix.unsqueeze(1)
        rel_emb_matrix = rel_emb_matrix.expand(-1, heads_qv.size(1), -1, -1)
        if self.apex:
            AC = bmm_einsum(heads_qu, heads_k, "ibnd,jbnd->ijbn")
            BD = bmm_einsum(heads_qv, rel_emb_matrix, "ibnd,jbnd->ijbn")
        else:
            AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, heads_k))
            BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_emb_matrix))
        
        BD = self._rel_shift(BD)


        attn_score = AC + BD
        attn_score.mul_(self.scale)

        if mask is not None and mask.any().item():
            attn_score.masked_fill_(mask[:,:,:,None], -float('inf'))

        attn_prob = F.softmax(attn_score, 1)
        attn_prob = self.dropatt(attn_prob)

        if self.apex:
            attn_vec = bmm_einsum(attn_prob, heads_v, "ilbn,lbnd->ibnd")
        else:
            attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, heads_v)
        attn_vec = attn_vec.contiguous().view(seq_len, batch_size, self.num_head * self.d_head)

        attn_out = self.lin_o(attn_vec)
        attn_out = self.drophid(attn_out)

        output = self.layer_norm(x + attn_out)

        return output






class TransformerUnit(nn.Module):
    def __init__(self, num_head, d_model, d_head, d_ff, dropout, dropatt, dropwei, drophid, apex):
        super().__init__()

        self.attn = LearnableMultiheadSelfAttention(num_head, d_model, d_head, dropout, dropatt, dropwei, drophid, apex)

        self.pos_ff = PostionwiseFF(d_model, d_ff, drophid, apex)

    def forward(self, inputs, pos_emb, pos_bias_u=None, pos_bias_v=None, mask=None, memory=None):
        
        output = self.attn(inputs, pos_emb, pos_bias_u, pos_bias_v, mask=mask, memory=memory)

        output = self.pos_ff(output)

        return output


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layer, num_head, d_model, d_head, d_ff, d_embedding, tied_weights, num_steps, mem_len, clamp_len, same_length, init_std, adaptive=True, div_val=1, cutoffs=[], dropout=0.0, dropatt=0.0, dropemb=0.0, dropinp=0.0, dropwei=0.0, drophid=0.0, apex=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.num_head = num_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_ff = d_ff
        self.d_embedding = d_embedding
        self.tied_weights = tied_weights
        self.num_steps = num_steps
        self.mem_len = mem_len
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.init_std = init_std
        self.adaptive = adaptive
        self.div_val = div_val
        self.cutoffs = cutoffs

        #self.dropout = dropout
        #self.dropatt = dropatt
        #self.dropemb = dropemb
        #self.dropinp = dropinp
        #self.dropwei = dropwei
        #self.drophid = drophid

        self.adaptive = adaptive

        if adaptive:
            self.embedding = AdaptiveEmbedding(vocab_size, d_embedding, d_model, cutoffs, div_val=div_val, init_std=init_std, dropemb=dropemb)
        else:
            self.decoder = nn.Linear(d_model, vocab_size, bias=False) 
            if tied_weights:
                self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=0).from_pretrained(self.decoder.weight)
                self.embedding.weight = self.decoder.weight
            else:
                self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=0)

        self.pos_emb = PostionalEmbedding(d_model)

        self.pos_bias_u = nn.Parameter(torch.Tensor(num_head, d_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_head, d_head))

        self.dropout = LockedDropout(dropout)
        self.dropinp = LockedDropout(dropinp)
        #self.drophid = LockedDropout(drophid)

        self.layers = nn.ModuleList()

        for i in range(num_layer):
            self.layers.append(TransformerUnit(
                num_head=num_head,
                d_model=d_model,
                d_head=d_head,
                d_ff=d_ff,
                dropout=dropout,
                dropatt=dropatt,
                dropwei=dropwei,
                drophid=drophid,
                apex=apex))

        self.init_weights(init_std)


    def init_weights(self, init_std):
        nn.init.normal_(self.pos_bias_u, 0.0, init_std)
        nn.init.normal_(self.pos_bias_v, 0.0, init_std)


    def init_hidden(self, batch_size):
        return self.init_memory(batch_size)

    def init_memory(self, batch_size):
        if self.mem_len > 0:
            param = next(self.parameters())
            return torch.zeros(self.num_layer+1, 
                               self.mem_len, 
                               batch_size, 
                               self.d_model, 
                               dtype=param.dtype, 
                               device=param.device)
        else:
            return None

    def forward(self, inputs, memory=None):
        seq_len, batch_size = inputs.size()

        if memory is None:
            memory = self.init_memory(batch_size)
        else:
            memory = memory.transpose(1, 2)

        if memory is not None:
            mem_len = memory.size(1)
        else:
            mem_len = 0

        total_len = seq_len + mem_len

        word_emb = self.embedding(inputs)

        if self.same_length:
            all_ones = word_emb.new_ones(seq_len, total_len)
            simple_mask = torch.triu(all_ones, diagonal=1+mem_len)
            mask = simple_mask + torch.tril(all_ones, diagonal=0)
        else:
            mask = torch.triu(word_emb.new_ones(seq_len, total_len), 
                              diagonal=1+mem_len)
        mask = mask.bool()[:,:,None]

        pos_seq = torch.arange(total_len-1, -1, -1.0, 
                               device=word_emb.device, 
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq = pos_seq.clamp(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        pos_emb = self.dropinp(pos_emb)
        core_out = self.dropinp(word_emb)
        
        if memory is not None:
            memories = [core_out.unsqueeze(0)]

        for i, layer in enumerate(self.layers):
            memory_i = None if memory is None else memory[i]
            core_out = layer(core_out, 
                             pos_emb, 
                             self.pos_bias_u, 
                             self.pos_bias_v, 
                             mask=mask, 
                             memory=memory_i)

            if memory is not None:
                memories.append(core_out.unsqueeze(0))
        memories = torch.cat(memories, dim=0)

        core_out = self.dropout(core_out)

        if memory is not None:
            whole_seq = torch.cat((memory, memories), dim=1)
            new_memory = whole_seq[:,-self.mem_len:,:,:].detach()
        else:
            new_memory = None
        
        if not self.adaptive:
            output = self.decoder(core_out)
            output = self.dropout(output)
        else:
            output = core_out

        #output = pad_packed_sequence(output)


        return output, new_memory.transpose(1, 2)
