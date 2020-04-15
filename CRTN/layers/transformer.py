import sys
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from CRTN.utils.adaptive import AdaptiveEmbedding
from CRTN.utils.fancy_dropout import WeightDropLinear
from torchnlp.nn import LockedDropout

import ipdb
import visdom
import time
try:
    from apex.normalization import FusedLayerNorm
except:
    print("No apex package found")

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
            
        self.d_model = d_model

        inverse_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inverse_freq", inverse_freq)

    def forward(self, pos_seq):

        sinusoid = torch.einsum("bi,j->ibj", pos_seq, self.inverse_freq)

        pos_embedding = torch.cat((sinusoid.sin(), sinusoid.cos()), -1)

        return pos_embedding

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
    def __init__(self, d_model, d_ff, dropfor, apex=False):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.FFNet = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(inplace=True),
                LockedDropout(dropfor),
                nn.Linear(d_ff, d_model),
                LockedDropout(dropfor)
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
    def __init__(self, num_head, d_model, d_head, dropatt, dropwei, apex=False):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.d_head = d_head
        self.apex = apex

        self.dropatt = nn.Dropout(dropatt)
        self.dropattout = LockedDropout(dropatt)

        #self.lin_q = nn.Linear(d_model, num_head * d_head, bias=False)
        #self.lin_kv = nn.Linear(d_model, 2 * num_head * d_head, bias=False)
        #self.lin_relemb = nn.Linear(d_model, num_head * d_head, bias=False)
        self.lin_q = WeightDropLinear(d_model, num_head * d_head, bias=False,
                                      weight_dropout=dropwei)
        self.lin_kv = WeightDropLinear(d_model, 2 * num_head * d_head, bias=False,
                                       weight_dropout=dropwei)
        self.lin_relemb = WeightDropLinear(d_model, num_head * d_head, bias=False,
                                           weight_dropout=dropwei)
        self.lin_o = nn.Linear(num_head * d_head, d_model, bias=False)

        if apex:
            self.layer_norm = FusedLayerNorm(d_model)
        else:
            self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)


    def _rel_shift(self, x):
        x_inp = x.reshape(x.size(0), -1, *x.size()[-2:])
        zero_pad = x_inp.new_zeros((x_inp.size(0), 1, *x_inp.size()[2:]))
        x_padded = torch.cat([zero_pad, x_inp], dim=1)

        x_padded = x_padded.view(x_inp.size(1) + 1, x_inp.size(0), *x_inp.size()[2:])

        x = x_padded[1:].view_as(x)

        return x


    def forward(self, x, pos_emb, pos_bias_u, pos_bias_v, mask=None, cache=None, indice_bool=None, weights=None, neighbor_mem=None, inf_ind=None, theta=1.0):
        """
        3 usage: 
            - compute query
            - compute hidden state
            - inference: ind
        """
        x_len, batch_size, nhid = x.size(0), x.size(1), x.size(2)

        if inf_ind is not None:
            assert inf_ind >= 0 and inf_ind < x_len
            x = x[inf_ind].unsqueeze(0)

        if cache is not None:
            cache_num, cache_ulen = cache.size(0), cache.size(1)
            cache_len = cache_num * cache_ulen
            cache = cache.view(cache_num * cache.size(1), -1, nhid)
            if not batch_size == cache.size(1):
                cache.unsqueeze_(1)
                cache = cache.expand(-1, batch_size // cache.size(2), -1, -1)
                cache = cache.reshape(cache.size(0), -1, cache.size(-1))
            cache_matrix = self.lin_kv(cache)
            cache_k, cache_v = torch.chunk(cache_matrix, 2, dim=-1)
        else:
            cache_num = 0
            cache_len = 0

        if neighbor_mem is not None:
            nei_len = neighbor_mem.size(0)
            nei_matrix = self.lin_kv(neighbor_mem)
            nei_k, nei_v = torch.chunk(nei_matrix, 2, dim=-1)
        else:
            nei_len = 0

        heads_q = self.lin_q(x)
        heads_q = heads_q.view(heads_q.size(0), batch_size, self.num_head, self.d_head)
        heads_qu = heads_q + pos_bias_u
        heads_qv = heads_q + pos_bias_v

        inp_matrix = self.lin_kv(x)
        inp_k, inp_v = torch.chunk(inp_matrix, 2, dim=-1)
            
        rel_emb_matrix = self.lin_relemb(pos_emb)
        rel_cache, rel_nei, rel_inp = rel_emb_matrix.split([cache_len, nei_len, x_len],
                                                            dim=0)

        rel_inp = rel_inp.view(x_len, batch_size, self.num_head, self.d_head)
        inp_k = inp_k.view(x_len, batch_size, self.num_head, self.d_head)
        inp_v = inp_v.view(x_len, batch_size, self.num_head, self.d_head)
        if self.apex:
            AC = bmm_einsum(heads_qu, inp_k)
            BD = bmm_einsum(heads_qv, rel_inp)
        else:
            AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, inp_k))
            BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_inp))

        if neighbor_mem is not None:
            nei_k = nei_k.view(nei_len, batch_size, self.num_head, self.d_head)
            nei_v = nei_v.view(nei_len, batch_size, self.num_head, self.d_head)
            rel_nei = rel_nei.view(nei_len, batch_size, self.num_head, self.d_head)
            if self.apex:
                nei_AC = bmm_einsum(heads_qu, nei_k)
                nei_BD = bmm_einsum(heads_qv, rel_nei)
            else:
                nei_AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, nei_k))
                nei_BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_nei))

            # if neighbor_mem is empty(0), mask it

            if indice_bool is None:
                if neighbor_mem.eq(0).sum() == neighbor_mem.numel():
                    nei_mask = torch.cat((mask.new_ones(nei_len), mask.new_zeros(x_len)), 0)
                    mask = mask + nei_mask.expand(mask.size(0), -1).unsqueeze(-1)

            AC = torch.cat((nei_AC, AC), dim=1)
            BD = torch.cat((nei_BD, BD), dim=1)
            
 
        if indice_bool is not None:
            pre_AC = torch.einsum("ibnd,ibk->kibnd", heads_qu, indice_bool)
            pre_BD = torch.einsum("ibnd,ibk->kibnd", heads_qv, indice_bool)
            cache_k = cache_k.view(cache_num, cache_ulen, batch_size, self.num_head, self.d_head)
            cache_v = cache_v.view(cache_num, cache_ulen, batch_size, self.num_head, self.d_head)

            rel_cache = rel_cache.view(cache_num, cache_ulen, batch_size, self.num_head, self.d_head)
            if self.apex:
                cache_AC = bmm_einsum(pre_AC, cache_k, "kibnd,kjbnd->ikjbn")
                cache_BD = bmm_einsum(pre_BD, rel_cache, "kibnd,kjbnd->ikjbn")
            else:
                cache_AC = torch.einsum("kibnd,kjbnd->ikjbn", pre_AC, cache_k)
                cache_BD = torch.einsum("kibnd,kjbnd->ikjbn", pre_BD, rel_cache)
            AC_mask = indice_bool.eq(0).transpose(1, 2)[:,:,None,:,None]
            cache_AC.masked_fill_(AC_mask, -float("inf")) 

            # if neighbor_mem is empty(0) or cache key is empty(0), mask it

            if cache.sum(dim=[1,2]).eq(0).sum() > 0:
                nei_mask = neighbor_mem.eq(0).sum().eq(neighbor_mem.numel()).expand(nei_len)
                cache_mask = cache.eq(0).reshape(cache_len, -1).min(dim=-1)[0]
                mask = mask + torch.cat((cache_mask, nei_mask, mask.new_zeros(x_len)), 0).expand(mask.size(0), -1).unsqueeze(-1)

            cache_AC = cache_AC.reshape(cache_AC.size(0), -1, batch_size, self.num_head)
            cache_BD = cache_BD.reshape(cache_BD.size(0), -1, batch_size, self.num_head)
            AC = torch.cat((cache_AC, AC), dim=1)
            BD = torch.cat((cache_BD, BD), dim=1)
       
            
  
        #if indice_bool is None and neighbor_mem is None:
        #    rel_emb_matrix = rel_emb_matrix.view(x_len, batch_size, 
        #                                         self.num_head, self.d_head)
        #    heads_k = heads_k.view(x_len, batch_size, self.num_head, self.d_head)
        #    heads_v = heads_v.view(x_len, batch_size, self.num_head, self.d_head)
        #    AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, heads_k))
        #    BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_emb_matrix))
        #else:
        #    if indice_bool is not None:
        #        cache_AC = torch.einsum("ibnd,ibk->kibnd", heads_qu, indice_bool)
        #        cache_BD = torch.einsum("ibnd,ibk->kibnd", heads_qv, indice_bool)
        #        cache_k = heads_k.view(cache_num + 1, x_len, batch_size, 
        #                             self.num_head, self.d_head)
        #        cache_v = heads_v.view(cache_num + 1, x_len, batch_size, 
        #                             self.num_head, self.d_head)

        #        cache_rel = rel_emb_matrix.view(cache_num + 1, x_len, 
        #                                      batch_size, self.num_head, self.d_head)
        #        if self.apex:
        #            AC = bmm_einsum(cache_AC, cache_k, "kibnd,kjbnd->ikjbn")
        #            BD = bmm_einsum(cache_BD, cache_rel, "kibnd,kjbnd->ikjbn")
        #        else:
        #            AC = torch.einsum("kibnd,kjbnd->ikjbn", cache_AC, cache_k)
        #            BD = torch.einsum("kibnd,kjbnd->ikjbn", cache_BD, cache_rel)
        #        AC_mask = indice_bool.eq(0).transpose(1, 2)[:,:,None,:,None]
        #        AC.masked_fill_(AC_mask, -float("inf")) 

        #        # if neighbor_mem is empty(0) or cache key is empty(0), mask it
        #        if cache.sum(dim=[1,2]).eq(0).sum() > 0:
        #            nei_mask = neighbor_mem.eq(0).sum().eq(neighbor_mem.numel()).expand(nei_len)
        #            cache_mask = cache.eq(0).reshape(cache_len, -1).min(dim=-1)[0]
        #            mask = mask + torch.cat((cache_mask, nei_mask, mask.new_zeros(x_len)), 0).expand(mask.size(0), -1).unsqueeze(-1)
        #    else:
        #        rel_emb_matrix = rel_emb_matrix.view(x_len, batch_size, 
        #                                             self.num_head, self.d_head)
        #        heads_k = heads_k.view(x_len, batch_size, self.num_head, 
        #                               self.d_head)
        #        heads_v = heads_v.view(x_len, batch_size, self.num_head, 
        #                               self.d_head)
        #        if self.apex:
        #            AC = bmm_einsum(heads_qu, heads_k)
        #            BD = bmm_einsum(heads_qv, rel_emb_matrix)
        #        else:
        #            AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, heads_k))
        #            BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_emb_matrix))

        #        # if neighbor_mem is empty(0), mask it
        #        if neighbor_mem.eq(0).sum() == neighbor_mem.numel():
        #            nei_mask = torch.cat((mask.new_ones(nei_len), mask.new_zeros(x_len)), 0)
        #            mask = mask + nei_mask.expand(mask.size(0), -1).unsqueeze(-1)
        #        
        #        AC.unsqueeze_(1)
        #        BD.unsqueeze_(1)

        #    if neighbor_mem is not None:
        #        nei_k = nei_k.view(nei_len, batch_size, self.num_head, self.d_head)
        #        nei_v = nei_v.view(nei_len, batch_size, self.num_head, self.d_head)
        #        rel_nei = rel_nei.view(nei_len, batch_size, self.num_head, self.d_head)
        #        if self.apex:
        #            nei_AC = bmm_einsum(heads_qu, nei_k)
        #            nei_BD = bmm_einsum(heads_qv, rel_nei)
        #        else:
        #            nei_AC = torch.einsum("ibnd,jbnd->ijbn", (heads_qu, nei_k))
        #            nei_BD = torch.einsum("ibnd,jbnd->ijbn", (heads_qv, rel_nei))
        #        
        #        AC_cache, AC_inp = AC.split([cache_num, 1], dim=1)
        #        BD_cache, BD_inp = BD.split([cache_num, 1], dim=1)

        #        AC = torch.cat((AC_cache.reshape(
        #                            AC_cache.size(0), -1, batch_size, self.num_head),
        #                        nei_AC,
        #                        AC_inp.squeeze(1)), 1)
        #        BD = torch.cat((BD_cache.reshape(
        #                            BD_cache.size(0), -1, batch_size, self.num_head),
        #                        nei_BD,
        #                        BD_inp.squeeze(1)), 1)
        
        if inf_ind is None:
            BD = self._rel_shift(BD)
        else:
            valid_len = BD.size(1) - (x_len - 1 - inf_ind)
            BD[:,:valid_len,:,:] = BD.clone()[:,x_len-1-inf_ind:,:,:]


        attn_score = AC + BD
        attn_score.mul_(self.scale)
        attn_score.mul_(theta)
        attn_score = attn_score.reshape(attn_score.size(0), -1, 
                                        batch_size, self.num_head)

        if inf_ind is not None:
            mask = mask[inf_ind].unsqueeze(0)
        if mask is not None:
            attn_score.masked_fill_(mask[:,:,:,None], -float('inf'))

        attn_prob = F.softmax(attn_score, 1)
        attn_prob = self.dropatt(attn_prob)
        attn_matrix = attn_prob.mean((2, 3))

        prob_cache, prob_nei, prob_inp = attn_prob.split([cache_len,
                                                          nei_len,
                                                          x_len], dim=1)
        if self.apex:
            attn_vec = bmm_einsum(prob_inp, inp_v, "ilbn,lbnd->ibnd")
        else:
            attn_vec = torch.einsum("ilbn,lbnd->ibnd", prob_inp, inp_v)

        if nei_len > 0:
            if self.apex:
                nei_vec = bmm_einsum(prob_nei, nei_v, "ilbn,lbnd->ibnd")
            else:
                nei_vec = torch.einsum("ilbn,lbnd->ibnd", prob_nei, nei_v)
            attn_vec = attn_vec + nei_vec

        if cache_len > 0:
            if weights is not None:
                prob_cache = prob_cache.reshape(prob_cache.size(0), -1, cache_ulen, 
                                                batch_size, self.num_head)
                prob_cache = torch.einsum("ikjbn,ibk->ikjbn", prob_cache, weights)
            prob_cache = prob_cache.view(prob_cache.size(0), -1, *prob_cache.shape[3:])
            cache_v = cache_v.view(-1, *cache_v.shape[2:])
            if self.apex:
                cache_vec = bmm_einsum(attn_prob, cache_v, "ilbn,lbnd->ibnd")
            else:
                cache_vec = torch.einsum("ilbn,lbnd->ibnd", prob_cache, cache_v)
            attn_vec = attn_vec + cache_vec
            

        #if cache is None and neighbor_mem is None:
        #    attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, heads_v)
        #else:
        #    if neighbor_mem is not None and nei_len > 0:
        #        prob_cache, prob_nei, prob_inp = attn_prob.split([cache_len,
        #                                                          nei_len,
        #                                                          x_len], dim=1)
        #        attn_prob = torch.cat((prob_cache, prob_inp), 1)
        #        if self.apex:
        #            nei_vec = bmm_einsum(prob_nei, nei_v, "ilbn,lbnd->ibnd")
        #        else:
        #            nei_vec = torch.einsum("ilbn,lbnd->ibnd", prob_nei, nei_v)

        #    if indice_bool is not None: 
        #        attn_prob = attn_prob.reshape(attn_prob.size(0), -1, x_len,
        #                                      batch_size, self.num_head)
        #        if weights is not None:
        #            attn_prob = torch.einsum("ikjbn,ibk->ikjbn", attn_prob, weights)
        #        attn_prob = attn_prob.view(attn_prob.size(0), -1, *attn_prob.shape[3:])
        #        cache_v = cache_v.view(-1, *cache_v.shape[2:])
        #        if self.apex:
        #            attn_vec = bmm_einsum(attn_prob, cache_v, "ilbn,lbnd->ibnd")
        #        else:
        #            attn_vec = torch.einsum("ilbn,lbnd->ibnd", attn_prob, cache_v)
        #        if neighbor_mem is not None and nei_len > 0:
        #            attn_vec = attn_vec + nei_vec
        #    else:
        #        if self.apex:
        #            attn_vec = bmm_einsum(attn_prob, heads_v, "ilbn,lbnd->ibnd")
        #        else:
        #            attn_vec = torch.einsum("ilbn,lbnd->ibnd", attn_prob, heads_v)
        #        attn_vec = attn_vec + nei_vec

        attn_vec = attn_vec.reshape(attn_vec.size(0), batch_size, 
                                    self.num_head * self.d_head)

        attn_out = self.lin_o(attn_vec)
        attn_out = self.dropattout(attn_out)

        output = self.layer_norm(x + attn_out)

        #if indice_bool is not None:
        #    if inf_ind is not None:
        #        print("inf time: %.2f ms" % ((time.time() - start_time) * 1000))
        #    else:
        #        print("train time: %.2f ms" % ((time.time() - start_time) * 1000))

        return output, attn_matrix






class TransformerUnit(nn.Module):
    def __init__(self, num_head, d_model, d_head, d_ff, dropatt, dropwei, dropfor, apex):
        super().__init__()


        self.attn = LearnableMultiheadSelfAttention(num_head, d_model, d_head, dropatt, 
                                                    dropwei, apex)


        self.pos_ff = PostionwiseFF(d_model, d_ff, dropfor, apex)

    def forward(self, inputs, pos_emb, pos_bias_u=None, pos_bias_v=None, mask=None, cache=None, indices=None, weights=None, neighbor_mem=None, inf_ind=None, theta=1.0):
        
        output, attn_matrix = self.attn(inputs, pos_emb, pos_bias_u, pos_bias_v, 
                                        mask=mask, cache=cache, 
                                        indice_bool=indices, 
                                        weights=weights, 
                                        neighbor_mem=neighbor_mem,
                                        inf_ind=inf_ind,
                                        theta=theta)

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
        cutoffs = self.args.cutoffs
        div_val = self.args.div_val
        init_std = self.args.init_std

        adaptive = self.args.adaptive

        
        self.corpus = corpus
        self.demo = self.args.demo
        self.same_length = args.same_length
        self.same_length_query = args.same_length_query

        self.theta = self.args.theta

        if adaptive:
            self.embedding = AdaptiveEmbedding(vocab_size, d_embedding, d_model, 
                                               cutoffs, div_val=div_val, 
                                               init_std=init_std,
                                               dropemb=args.dropemb)
        else:
            self.decoder = nn.Linear(d_model, vocab_size, bias=False) 
            if args.tied:
                self.embedding = nn.Embedding(vocab_size, 
                                              d_embedding, 
                                              padding_idx=1).from_pretrained(
                                                      self.decoder.weight)
                self.embedding.weight = self.decoder.weight
            else:
                self.embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=1)

        self.pos_emb = PositionalEmbedding(d_model)

        self.drophid = LockedDropout(args.drophid)
        self.dropout = LockedDropout(args.dropout)
        self.dropinp = LockedDropout(args.dropinp)


        self.pos_bias_u = nn.Parameter(torch.Tensor(num_head, d_head))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_head, d_head))

        if args.stat:
            self.select_stat = nn.Parameter(torch.zeros(args.cache_N), 
                                            requires_grad=False)
            self.viz = visdom.Visdom()
            assert self.viz.check_connection()

        self.layers = nn.ModuleList()

        for i in range(num_layer):
            self.layers.append(TransformerUnit(
                num_head=num_head,
                d_model=d_model,
                d_head=d_head,
                d_ff=d_ff,
                dropatt=args.dropatt,
                dropwei=args.dropwei,
                dropfor=args.dropfor,
                apex=args.apex))


        self.init_weights(init_std)

    def init_weights(self, init_std):
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

    def forward(self, inputs, cache_info=None, values=None, weights=None, indices=None, words=None, draw=False, neighbor_mem=None, inf_ind=None, inf_blocks=None):
        #input shape should be seq_len * bsz or seq_len * bsz * emsize

        if inputs.dim() == 2:
            word_emb = self.embedding(inputs)
            seq_len, batch_size = inputs.size()
        else:
            word_emb = inputs
            seq_len, batch_size, _ = inputs.size()

        if indices is not None:
            mem_len = values.size(0) * values.size(1)
            values = values.view(values.size(0), values.size(1), -1, 
                                 self.args.nlayers+1, self.args.nhid)
        else:
            mem_len = 0
            #zone_bsz = batch_size

        if inf_blocks is not None:
            inf_blocks = inf_blocks.transpose(1, 2)

        if neighbor_mem is not None:
            nei_len = neighbor_mem.size(1)
            total_len = seq_len + mem_len + nei_len
        else:
            nei_len = 0
            total_len = seq_len + mem_len

        

        if indices is not None:
            #pos_seq
            pos_indices = torch.cat((indices, 
                                    (torch.ones_like(indices[0]) * values.size(0)
                                    ).unsqueeze(0)))

            pos_seq = torch.arange(total_len-1, -1, -1.0, device=inputs.device)
            if self.args.merge_shift:
                alpha = self.args.merge_alpha
                if alpha == 1.0:
                    alpha -= 1e-10
                pos_shift = pos_seq.new_ones(seq_len)
                pos_shift *= seq_len * alpha / (1 - alpha)
                pos_pad = pos_seq.new_zeros(total_len-seq_len)
                seq_shift = torch.cat((pos_shift, pos_pad), 0)
                pos_seq += seq_shift

            if self.args.real_pos:
                if cache_info is None:
                    pos = torch.arange(indices.size(0) - 1, -1, -1, 
                                           dtype=torch.float,
                                           device=inputs.device)
                    pos = pos.expand(batch_size, -1)
                    pos.transpose_(0, 1)
                    pos_key = pos 
                else:
                    pos_key = cache_info[:,:,0]
                pos_start = torch.einsum("ib,j->bij", pos_key, 
                                         pos_key.new_ones(seq_len) * seq_len)
                if self.args.farnear:
                    pos_start += self.args.neighbor_len
                pos_seq = pos_start + torch.arange(seq_len - 1, -1, -1, 
                                                   dtype=pos_key.dtype, 
                                                   device=pos_key.device)
                pos_seq = pos_seq.reshape(batch_size, -1)
                if self.args.farnear:
                    pos_tail = torch.arange(seq_len + nei_len - 1, -1, -1,
                                            dtype=pos_key.dtype,
                                            device=pos_key.device)
                else:
                    pos_tail = torch.arange(seq_len - 1, -1, -1,
                                            dtype=pos_key.dtype,
                                            device=pos_key.device)
                pos_tail = pos_tail.expand(batch_size, -1)
                pos_seq = torch.cat((pos_seq, pos_tail), dim=1)
            else:
                pos_seq = pos_seq.expand(batch_size, -1)

            #one-hot pos_indices
            mem_num = values.size(0)
            indice_len = indices.size(0)
            tfbase = torch.eye(mem_num, device=indices.device)
            indice_bool = torch.index_select(tfbase, 0, indices.reshape(-1))
            indice_bool = indice_bool.view(indice_len, -1, batch_size, mem_num)
            indice_bool = indice_bool.sum(0)
            if self.args.discard_worst:
                # update recalls
                query_len = indice_bool.size(0)
                recall = indice_bool.sum(0).t()[:-1,:]
                pos, recalls, queries = cache_info.chunk(3, dim=-1)
                recalls += recall.unsqueeze(-1)
                queries += query_len

            if self.args.stat:
                stat = indice_bool.sum((0, 1))
                self.select_stat += stat[:-1]
                self.viz.bar(self.select_stat, win="select stat")

            if weights is not None:
                x_len = inputs.size(0) if inf_ind is None else 1
                #weights = torch.cat((weights, 
                #                    torch.ones_like(
                #                        weights[:,:,0,None]) * -float("inf")), 2)
                weights = weights.masked_fill(indice_bool.eq(0), -float("inf"))
                weights = F.softmax(weights, 2) * self.args.cache_k
                #weights = weights.index_fill(2, (weights.new_ones(1) * mem_num).long(), 1.0)
            
        else:
            pos_seq = torch.arange(total_len-1, -1, -1.0, device=inputs.device)
            pos_seq = pos_seq.expand(batch_size, -1)
            pos_indices = indices
            indice_bool = None

        if self.same_length_query and indices is None:
            all_ones = word_emb.new_ones(seq_len, total_len)
            simple_mask = torch.triu(all_ones, diagonal=1+nei_len)
            mask = simple_mask + torch.tril(all_ones, diagonal=-1)
        else:
            mask = torch.triu(word_emb.new_ones(seq_len, total_len), 
                              diagonal=1+mem_len+nei_len) 
        mask = mask.bool()[:,:,None]


        pos_seq = pos_seq.view(batch_size, -1)

        if self.args.clamp_len > 0:
            pos_seq = pos_seq.clamp(max=self.args.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        pos_emb = self.dropinp(pos_emb)
        core_out = self.dropinp(word_emb)
        
        memory = core_out.clone()
        if inf_ind is None:
            memories = [memory]
        else:
            memories = [memory[inf_ind].unsqueeze(0)]
            core_out = core_out[inf_ind].unsqueeze(0)

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

            if inf_blocks is None:
                block_i = None
            else:
                block_i = inf_blocks[i]

            if inf_ind is None:
                core_out, attn_matrix = layer(core_out, pos_emb, self.pos_bias_u, 
                                              self.pos_bias_v, 
                                              mask=mask, 
                                              cache=value_i, 
                                              indices=indice_bool, 
                                              weights=weights,
                                              neighbor_mem=neighbor_mem_i,
                                              theta=self.theta)
            else:
                block_i[inf_ind] = core_out.squeeze(0)
                core_out, attn_matrix = layer(block_i, pos_emb, self.pos_bias_u, 
                                              self.pos_bias_v, 
                                              mask=mask, 
                                              cache=value_i, 
                                              indices=indice_bool, 
                                              weights=weights,
                                              neighbor_mem=neighbor_mem_i,
                                              inf_ind=inf_ind,
                                              theta=self.theta)


            if i < len(self.layers) - 1:
                core_out = self.drophid(core_out)
            else:
                core_out = self.dropout(core_out)
            memories.append(core_out)

        memories = torch.cat(memories, 0)
        memories = memories.view(self.args.nlayers+1, core_out.size(0), -1, 
                                 self.args.nhid)

        if draw:
            attn_map = attn_matrix
        else:
            attn_map = None

        if not self.args.adaptive:
            output = self.decoder(core_out)
            output = self.dropout(output)
        else:
            output = core_out

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
            return output, memories, (attn_map, demo_display)


        return output, memories, attn_map
