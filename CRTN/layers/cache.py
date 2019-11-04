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

        self.mem_start = 0
        self.mem_end = args.cache_N - 1

        #self.keys = nn.ParameterDict({
        #    str(i): nn.Parameter(torch.zeros(batch_size, self.dk),
        #                         requires_grad=False) 
        #                            for i in range(args.cache_N)
        #})
        #self.keys = dict({
        #    str(i): torch.zeros(batch_size, self.dk, requires_grad=False) 
        #        for i in range(args.cache_N)
        #    })
        #self.values = nn.ParameterDict({
        #    str(i): nn.Parameter(torch.zeros(args.num_steps, batch_size, 
        #                                     (args.nlayers+1) * args.nhid),
        #                        requires_grad=False) 
        #                            for i in range(args.cache_N)
        #})
        #self.values = dict({
        #    str(i): torch.zeros(args.num_steps, batch_size, 
        #                        (args.nlayers+1) * args.nhid, 
        #                        requires_grad=False) 
        #                            for i in range(args.cache_N)
        #})

        #self.register_buffer("testtensor", torch.zeros([3,3]))

        for i in range(self.mem_start, self.mem_end + 1):
            self.register_buffer("key" + str(i), torch.zeros(batch_size, self.dk))
            self.register_buffer("value" + str(i), torch.zeros(args.num_steps,
                                                               batch_size,
                                                               ((args.nlayers + 1) 
                                                                * args.nhid)))


        if corpus is not None:
            self.words = nn.ParameterDict({
                str(i): nn.Parameter(torch.zeros(args.num_steps, batch_size, 
                                                 dtype=torch.long), 
                                                 requires_grad=False)                                     
                    for i in range(args.cache_N)
        })

        self.init_keys(args.init_std)

        self.attn = DotProductAttention()
        if not args.no_summary:
            self.summary = nn.Linear(args.nhid * args.num_steps, args.cache_dk)

        if args.p_discard:
            self.p_content = nn.Linear(args.nhid * args.num_steps, 1)
            self.p_pos = nn.Linear(args.nhid * args.num_steps, 1)

        self.batch_size = batch_size
        self.L = self.args.num_steps
        self.N = self.args.cache_N
        self.dv = self.args.nhid
        self.topk = self.args.cache_k

    #def cache2buffer(self):
    #    for ik, key in enumerate(self.keys.keys()):
    #        self.register_buffer("key" + str(ik), self.keys[key])
    #    for ik, key in enumerate(self.values.keys()):
    #        self.register_buffer("value" + str(ik), self.values[key])

    def to(self, device):
        super().to(device)
        #self.keys = self.keys.to(device)
        #for key in self.keys.keys():
        #    self.keys[key] = self.keys[key].to(device)
        #self.values = self.values.to(device)
        #for key in self.values.keys():
        #    self.values[key] = self.values[key].to(device)
        if self.demo:
            self.words = self.words.to(device)

    def init_keys(self, init_std):
        for i in range(self.mem_start, self.mem_end + 1):
            nn.init.normal_(getattr(self, "key" + str(i)))

    def _get_keys(self):
        keys = [getattr(self, "key" + str(i)) 
                for i in range(self.mem_start, self.mem_end + 1)]
        keys = torch.cat(keys, 0)
        keys = keys.view(self.N, -1, self.dk)
        return keys

    def init_key_and_value(self, key, value):
        if key is not None:
            key = key[1]
            for i in range(self.mem_start, self.mem_end + 1):
                idx = torch.tensor(i).to(key.device)
                getattr(self, "key" + str(i)).copy_(key.index_select(0, idx).squeeze())
        if value is not None:
            value.transpose_(1, 2)
            for i in range(self.mem_start, self.mem_end + 1):
                idx = torch.tensor(i).to(value.device)
                getattr(self, 
                        "value" + str(i)).copy_(value.index_select(0, idx).squeeze())

    def _get_values(self):
        values = [getattr(self, "value" + str(i)) 
                  for i in range(self.mem_start, self.mem_end + 1)]
        values = torch.cat(values, 0)
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

        self.mem_start = 0
        self.mem_end = self.args.cache_N - 1

        for i in range(self.mem_start, self.mem_end + 1):
            setattr(self, "key" + str(i), torch.zeros(batch_size, self.dk))
            setattr(self, "value" + str(i), torch.zeros(self.args.num_steps,
                                                         batch_size,
                                                         ((self.args.nlayers + 1) 
                                                         * self.args.nhid)))
        #self.keys.clear()
        #self.values.clear()
        #self.keys.update({
        #    str(i): nn.Parameter(torch.zeros(batch_size, self.dk),
        #                         requires_grad=False)
        #                for i in range(self.N)
        #})
        #self.keys.update({
        #    str(i): torch.zeros(batch_size, self.dk, requires_grad=False) 
        #        for i in range(self.N)
        #    })
        #self.values.update({
        #    str(i): nn.Parameter(torch.zeros(self.L, batch_size, 
        #                         (self.args.nlayers+1) * self.dv), 
        #                         requires_grad=False) 
        #                for i in range(self.N)
        #})
        #self.values.update({
        #    str(i): torch.zeros(self.L, batch_size, (self.args.nlayers+1) * self.dv, 
        #                        requires_grad=False)
        #                for i in range(self.N)
        #    })
        
        if self.demo:
            self.words.update({
                str(i): nn.Parameter(torch.zeros(self.L, batch_size, 
                                        dtype=torch.long), requires_grad=False) 
                            for i in range(self.N)
            })

        self.init_keys(self.args.init_std)

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

        #print(query.device, keys.device, values.device, self.testtensor.device, self.named_buffers("key0").device, self.keys["0"].device)
        
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


    def renew(self, inputs, words=None, key_num=None):
        #inputs = inputs.detach()
        inputs = inputs.transpose(1, 2)
        
        if self.args.merge:
            key_num = self.merge(key_num)
        elif self.args.p_discard:
            key_num = self.p_discard(key_num)
        else:
            self.eliminate_last()

        n = self.mem_end

        if self.args.no_summary:
            new_key = inputs[-1].reshape(self.batch_size, -1)
        else:
            new_key = self.summary(inputs[-1].reshape(-1, self.L * self.dv))

        new_value = torch.einsum("mblh->lbmh", 
                                 inputs
                                 ).reshape(self.L, -1, (self.args.nlayers+1) * self.dv)

        getattr(self, "key" + str(n)).copy_(new_key.detach())
        getattr(self, "value" + str(n)).copy_(new_value.detach())
        #self.keys.update({
        #    str(n): nn.Parameter(new_key, requires_grad=False)
        #    })
        #self.keys.update({
        #    str(n): new_key.detach() 
        #    })
        #self.values.update({
        #    str(n): nn.Parameter(new_value, requires_grad=False)
        #    })
        #self.values.update({
        #    str(n): new_value.detach()
        #    })
        if self.demo:
            self.words.update({
                str(n): nn.Parameter(words, requires_grad=False)
                })

        return key_num


    def eliminate_last(self):

        device = getattr(self, "key" + str(self.mem_start)).device

        for i in range(self.mem_start, self.mem_end):
            getattr(self, "key" + str(i)).copy_(getattr(self, "key" + str(i+1)))
            getattr(self, "value" + str(i)).copy_(getattr(self, "value" + str(i+1)))

        getattr(self, "key" + str(self.mem_end)).copy_(torch.zeros(
                                                                self.batch_size,
                                                                self.dk,
                                                                device=device))
        getattr(self, "value" + str(self.mem_end)).copy_(torch.zeros(
                                                                self.L,
                                                                self.batch_size,
                                                                (self.dv * 
                                                                (self.args.nlayers+1)),
                                                                device=device))

        #keys_keys = list(self.keys.keys())
        #keys_values = list(self.values.keys())

        #keys_keys = sorted(list(map(int, keys_keys)))
        #keys_values = sorted(list(map(int, keys_values)))

        #eli_key = self.keys.pop(str(keys_keys[0]))
        #eli_value = self.values.pop(str(keys_values[0]))

        #device = eli_key.device

        #self.keys.update({
        #    str(keys_keys[-1]+1): nn.Parameter(torch.zeros(self.batch_size, self.dk, 
        #                                                   device=device))
        #    })
        #self.keys.update({
        #    str(keys_keys[-1]+1): torch.zeros(self.batch_size, self.dk, device=device,
        #                                      requires_grad=False)
        #    })

        #self.values.update({
        #    str(keys_values[-1]+1): nn.Parameter(
        #                                torch.zeros(self.L, self.batch_size, 
        #                                            self.dv * (self.args.nlayers+1), 
        #                                            device=device))
        #    })
        #self.values.update({
        #    str(keys_values[-1]+1): torch.zeros(self.L, self.batch_size, 
        #                                        self.dv * (self.args.nlayers+1), 
        #                                        device=device,
        #                                        requires_grad=False)
        #    })


        if self.demo:
            keys_words = list(self.words.keys())
            keys_words = sorted(list(map(int, keys_words)))
            eli_word = self.words.pop(str(keys_words[0]))
            self.words.update({
                str(keys_words[-1]+1): nn.Parameter(torch.zeros(self.L, 
                                                                self.batch_size, 
                                                                dtype=torch.long), 
                                                    requires_grad=False)
                })

    def merge(self, key_num):
        
        eli_key = getattr(self, "key" + str(self.mem_start))
        eli_value = getattr(self, "value" + str(self.mem_start))
        device = eli_key.device
        alpha = self.args.merge_alpha

        getattr(self, "key" + str(self.mem_start)).copy_((
            alpha * eli_key
            + (1 - alpha) * getattr(self, "key" + str(self.mem_start+1))))
        getattr(self, "value" + str(self.mem_start)).copy_((
            alpha * eli_value
            + (1 - alpha) * getattr(self, "value" + str(self.mem_start+1))))
        for i in range(self.mem_start + 1, self.mem_end):
            getattr(self, "key" + str(i)).copy_(getattr(self, "key" + str(i+1)))
            getattr(self, "value" + str(i)).copy_(getattr(self, "value" + str(i+1)))

        getattr(self, "key" + str(self.mem_end)).copy_(torch.zeros(
                                                                self.batch_size,
                                                                self.dk,
                                                                device=device))
        getattr(self, "value" + str(self.mem_end)).copy_(torch.zeros(
                                                                self.L,
                                                                self.batch_size,
                                                                (self.dv * 
                                                                (self.args.nlayers+1)),
                                                                device=device))

        key_num[0] = alpha * key_num[0].item() + (1 - alpha) * key_num[1].item() + 1
        return key_num

    def p_discard(self, key_num):
        ipdb.set_trace()
        pass



