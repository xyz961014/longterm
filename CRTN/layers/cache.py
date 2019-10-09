import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math
from copy import deepcopy

from CRTN.layers.attention import DotProductAttention

class Cache(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        self.args = deepcopy(args)
        self.corpus = corpus
        self.demo = self.args.demo
        if self.args.no_summary:
            self.dk = self.args.num_steps * self.args.nhid
        else:
            self.dk = self.args.cache_dk

        batch_size = self.args.batch_size // len(self.args.devices)

        self.keys = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros(batch_size, self.dk),
                                 requires_grad=False) 
                                    for i in range(args.cache_N)
        })
        self.values = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros(args.num_steps, batch_size, 
                                             (args.nlayers+1) * args.nhid),
                                requires_grad=False) 
                                    for i in range(args.cache_N)
        })
        if corpus is not None:
            self.words = nn.ParameterDict({
                str(i): nn.Parameter(torch.zeros(args.num_steps, batch_size, 
                                                 dtype=torch.long), 
                                                 requires_grad=False)                                     
                    for i in range(args.cache_N)
        })

        self.init_keys(args.init_std)

        self.renew_place = args.cache_N - 1
        self.attn = DotProductAttention()
        if not args.no_summary:
            self.summary = nn.Linear(args.nhid * args.num_steps, args.cache_dk)

        self.batch_size = batch_size
        self.L = self.args.num_steps
        self.N = self.args.cache_N
        self.dv = self.args.nhid
        self.topk = self.args.cache_k


    def to(self, device):
        super().to(device)
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        if self.demo:
            self.words = self.words.to(device)

    def init_keys(self, init_std):
        for key in self.keys.keys():
            nn.init.normal_(self.keys[key])

    def _get_keys(self):
        keys = self.keys.values()
        keys = torch.cat(tuple(keys), 0)
        keys = keys.view(self.N, -1, self.dk)
        return keys

    def _get_values(self):
        values = self.values.values()
        values = torch.cat(tuple(values), 0)
        values = values.view(self.N, self.L, -1, self.dv * (self.args.nlayers+1))
        return values

    def _get_words(self):
        words = self.words.values()
        words = torch.cat(tuple(words), 0)
        words = words.view(-1, self.N, self.L)
        return words

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        device = self._get_keys().device 
        self.keys.clear()
        self.values.clear()
        self.keys.update({
            str(i): nn.Parameter(torch.zeros(batch_size, self.dk),
                                 requires_grad=False)
                        for i in range(self.N)
        })
        self.values.update({
            str(i): nn.Parameter(torch.zeros(self.L, batch_size, 
                                 (self.args.nlayers+1) * self.dv), 
                                 requires_grad=False) 
                        for i in range(self.N)
        })
        if self.demo:
            self.words.update({
                str(i): nn.Parameter(torch.zeros(self.L, batch_size, 
                                        dtype=torch.long), requires_grad=False) 
                            for i in range(self.N)
            })

        self.init_keys(self.args.init_std)

        self.renew_place = self.N - 1
        self.to(device)

    def forward(self, query):

        query = query.transpose(1, 2)
        query_len, bsz = query.size(0), query.size(1)
        if self.args.no_summary:
            query = query.reshape(query_len, bsz, -1)
        else:
            query = self.summary(query.reshape(query_len, -1, self.L * self.dv))
        keys = self._get_keys()
        values = self._get_values()
        
        if self.demo:
            words = self._get_words()
            
        #keys = keys.transpose(0, 1).contiguous()
        keys.transpose_(0, 1)
        values = torch.einsum("klbh->bklh", values)

        #keys = keys.expand(query_len, -1, -1, -1)
        #values = values.expand(query_len, -1, -1, -1, -1)

        #query = query.reshape(-1, self.L * self.dv)
        #keys = keys.reshape(-1, self.N, self.L * self.dv)
        #values = values.reshape(-1, self.N, self.L, self.dv * (self.args.nlayers + 1))

        
        if self.args.max_pooling:
            query = query.view(-1, self.args.num_steps, self.args.nhid)
            pooling_keys = keys.view(-1, self.N, self.args.num_steps, self.args.nhid)
            attention = torch.einsum("bih,bnjh->bijn", query, pooling_keys)
            attention = attention.view(-1, self.args.num_steps ** 2, 1, self.N)
            attention = attention.max(1)[0]
        else:
            attention, _ = self.attn(query, keys, values)

        attention = attention.view(-1, 1, attention.size(-1))
        
        _, topk_indices = attention.topk(self.topk)
        topk_indices = topk_indices.squeeze().t()
        #topk_indices = topk_indices.transpose(0, 2).reshape(self.topk, -1)
        #outputs = values[batch, topk_indices]

        #values.transpose_(0, 1)
        #indices = topk_indices[:,:,None,None]
        #indices = indices.expand(-1, -1, values.size(-2), values.size(-1))
        #outputs = torch.gather(values, 0, indices)

        if self.demo:
            #words = words.transpose(0, 1).contiguous()
            #indices = topk_indices[:,:,None]
            #indices = indices.expand(-1, -1, values.size(-1))
            #word_output = torch.gather(words, 0, indices)
            return attention, topk_indices, words
        else:
            return attention, topk_indices


    def renew(self, inputs, words=None):
        #inputs = inputs.detach()
        inputs = inputs.transpose(1, 2)
        n = self.renew_place
        
        if n >= self.N:
            self.eliminate_last()

        if self.args.no_summary:
            new_key = inputs[-1].reshape(self.batch_size, -1)
        else:
            new_key = self.summary(inputs[-1].reshape(-1, self.L * self.dv))

        new_value = torch.einsum("mblh->lbmh", 
                                 inputs
                                 ).reshape(self.L, -1, (self.args.nlayers+1) * self.dv)
        self.keys.update({
            str(n): nn.Parameter(new_key, requires_grad=False)
            })
        self.values.update({
            str(n): nn.Parameter(new_value, requires_grad=False)
            })
        if self.demo:
            self.words.update({
                str(n): nn.Parameter(words, requires_grad=False)
                })
        
        n += 1
        self.renew_place = n

    def eliminate_last(self):

        keys_keys = list(self.keys.keys())
        keys_values = list(self.values.keys())

        keys_keys = sorted(list(map(int, keys_keys)))
        keys_values = sorted(list(map(int, keys_values)))

        eli_key = self.keys.pop(str(keys_keys[0]))
        eli_value = self.values.pop(str(keys_values[0]))

        device = eli_key.device

        self.keys.update({
            str(keys_keys[-1]+1): nn.Parameter(torch.zeros(self.batch_size, self.dk, 
                                                            device=device))
            })
        self.values.update({
            str(keys_values[-1]+1): nn.Parameter(torch.zeros(self.L, self.batch_size, 
                                    self.dv * (self.args.nlayers+1), device=device))
            })
        if self.demo:
            keys_words = list(self.words.keys())
            keys_words = sorted(list(map(int, keys_words)))
            eli_word = self.words.pop(str(keys_words[0]))
            self.words.update({
                str(keys_words[-1]+1): nn.Parameter(torch.zeros(self.L, 
                        self.batch_size, dtype=torch.long), requires_grad=False)
                })
 



