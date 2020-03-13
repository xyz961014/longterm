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
        if self.args.no_summary:
            self.dk = self.args.num_steps * self.args.nhid
        else:
            self.dk = self.args.cache_dk


        #self.mem_start = 0
        #self.mem_end = args.cache_N - 1

        #for i in range(self.mem_start, self.mem_end + 1):
        #    self.register_buffer("key" + str(i), torch.zeros(batch_size, self.dk))
        #    self.register_buffer("value" + str(i), torch.zeros(args.num_steps,
        #                                                       batch_size,
        #                                                       ((args.nlayers + 1) 
        #                                                        * args.nhid)))


        #if corpus is not None:
        #    self.words = nn.ParameterDict({
        #        str(i): nn.Parameter(torch.zeros(args.num_steps, batch_size, 
        #                                         dtype=torch.long), 
        #                                         requires_grad=False)                                     
        #            for i in range(args.cache_N)
        #})
        #self.init_keys(args.seed, args.init_std)

        self.attn = DotProductAttention()
        if not args.no_summary:
            self.summary = nn.Linear(args.nhid * args.num_steps, args.cache_dk)

        #if args.p_discard:
        #    self.p_content = nn.Linear(args.nhid * args.num_steps, 1)
        #    self.p_pos = nn.Linear(args.nhid * args.num_steps, 1)

        self.batch_size = args.batch_size
        self.L = self.args.num_steps
        self.N = self.args.cache_N
        self.dv = self.args.nhid
        self.topk = self.args.cache_k

        self.pos_emb = PositionalEmbedding(args.nhid)

    #def to(self, device):
    #    super().to(device)
    #    if self.demo:
    #        self.words = self.words.to(device)

    #def init_keys(self, seed=0, init_std=1.0):

    #    for i in range(self.mem_start, self.mem_end + 1):
    #        nn.init.normal_(getattr(self, "key" + str(i)), std=init_std)

    #def _get_keys(self):
    #    keys = [getattr(self, "key" + str(i)) 
    #            for i in range(self.mem_start, self.mem_end + 1)]
    #    keys = torch.cat(keys, 0)
    #    keys = keys.view(self.N, -1, self.dk)
    #    return keys

    #def init_key_and_value(self, key, value):
    #    if key is not None:
    #        for i in range(self.mem_start, self.mem_end + 1):
    #            setattr(self, "key" + str(i), key[i])

    #    if value is not None:
    #        value = value.transpose(1, 2)
    #        for i in range(self.mem_start, self.mem_end + 1):
    #            setattr(self, "value" + str(i), value[i])

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
        

    #def _get_values(self):
    #    values = [getattr(self, "value" + str(i)) 
    #              for i in range(self.mem_start, self.mem_end + 1)]
    #    values = torch.cat(values, 0)
    #    values = values.view(self.N, self.L, -1, self.dv * (self.args.nlayers+1))
    #    return values

    #def _get_words(self):
    #    words = self.words.values()
    #    words = torch.cat(tuple(words), 0)
    #    words = words.view(-1, self.N, self.L)
    #    return words

    #def detach_memory(self):
    #    for i in range(self.N):
    #        setattr(self, "key" + str(i), getattr(self, "key" + str(i)).detach())
    #        setattr(self, "value" + str(i), getattr(self, "value" + str(i)).detach())

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        #device = self.key0.device 

        #self.mem_start = 0
        #self.mem_end = self.args.cache_N - 1


        #keys = torch.zeros(self.args.cache_N, batch_size, self.dk, device=device)
        #values = torch.zeros(self.args.cache_N, self.args.num_steps, batch_size, 
        #                     (self.args.nlayers + 1) * self.args.nhid, device=device)
        #for i in range(self.mem_start, self.mem_end + 1):
        #    setattr(self, "key" + str(i), keys[i])
        #    setattr(self, "value" + str(i), values[i])
        
        #if self.demo:
        #    self.words.update({
        #        str(i): nn.Parameter(torch.zeros(self.L, batch_size, 
        #                                dtype=torch.long), requires_grad=False) 
        #                    for i in range(self.N)
        #    })

        #self.init_keys(self.args.seed, self.args.init_std)

    def forward(self, query, keys, values):

        query = query.transpose(1, 2)
        query_len, bsz = query.size(0), query.size(1)
        if self.args.no_summary:
            query = query.reshape(query_len, bsz, -1)
        else:
            query = self.summary(query.reshape(query_len, -1, self.L * self.dv))
        
        #if self.demo:
        #    words = self._get_words()
            
        keys = keys.transpose(0, 1)
        values = torch.einsum("klbh->bklh", values)
        
        #print(query.device, keys.device, values.device)
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
        topk_indices = topk_indices.squeeze(1).t()

        #if self.demo:
        #    #words = words.transpose(0, 1).contiguous()
        #    #indices = topk_indices[:,:,None]
        #    #indices = indices.expand(-1, -1, values.size(-1))
        #    #word_output = torch.gather(words, 0, indices)
        #    return attention, topk_indices, words
        #else:
        #    return attention, topk_indices

        return attention, topk_indices


    def renew(self, inputs, words=None, key_num=None, keys=None, values=None):
        inputs = inputs.transpose(1, 2)

        if keys is None and values is None:
            keys, values = self.new_key_and_values()

        keys, values = keys.detach(), values.detach()
        
        if self.args.merge:
            key_blocks, value_blocks, key_num = self.merge(keys, values, key_num)
        #elif self.args.p_discard:
        #    keys, values, key_num = self.p_discard(keys, values, key_num)
        else:
            key_blocks, value_blocks = self.eliminate_last(keys, values)

        if self.args.no_summary:
            new_key = inputs[-1].reshape(self.batch_size, -1)
        else:
            new_key = self.summary(inputs[-1].reshape(-1, self.L * self.dv))

        new_value = torch.einsum("mblh->lbmh", 
                                 inputs
                                 ).reshape(self.L, -1, (self.args.nlayers+1) * self.dv)

        #setattr(self, "key" + str(n), new_key.detach())
        #setattr(self, "value" + str(n), new_value.detach())
        
        key_blocks[-1] = new_key.unsqueeze(0).detach()
        value_blocks[-1] = new_value.transpose(0, 1).unsqueeze(0).detach()
        
        keys = torch.cat(key_blocks, 0)
        values = torch.cat(value_blocks, 0)
        

        #if self.demo:
        #    self.words.update({
        #        str(n): nn.Parameter(words, requires_grad=False)
        #        })

        return keys, values, key_num.detach()


    def eliminate_last(self, keys, values):

        #device = getattr(self, "key" + str(self.mem_start)).device

        #for i in range(self.mem_start, self.mem_end):
        #    setattr(self, "key" + str(i), getattr(self, "key" + str(i+1)))
        #    setattr(self, "value" + str(i), getattr(self, "value" + str(i+1)))

        #setattr(self, "key" + str(self.mem_end), 
        #        torch.zeros(self.batch_size, self.dk).cuda())
        #setattr(self, "value" + str(self.mem_end), 
        #        torch.zeros(self.L,
        #                    self.batch_size,
        #                    (self.dv * (self.args.nlayers + 1))).cuda())

        key_blocks = list(keys.split(1))
        value_blocks = list(values.split(1))
        for i in range(self.N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])

        #if self.demo:
        #    keys_words = list(self.words.keys())
        #    keys_words = sorted(list(map(int, keys_words)))
        #    eli_word = self.words.pop(str(keys_words[0]))
        #    self.words.update({
        #        str(keys_words[-1]+1): nn.Parameter(torch.zeros(self.L, 
        #                                                        self.batch_size, 
        #                                                        dtype=torch.long), 
        #                                            requires_grad=False)
        #        })

        return key_blocks, value_blocks

    def merge(self, keys, values, key_num):
        
        key_blocks = list(keys.split(1))
        value_blocks = list(values.split(1))

        eli_key = key_blocks[0]
        eli_value = value_blocks[0]

        #eli_key = getattr(self, "key" + str(self.mem_start))
        #eli_value = getattr(self, "value" + str(self.mem_start))
        device = eli_key.device
        alpha = self.args.merge_alpha

        key_blocks[0] = alpha * eli_key + (1 - alpha) * key_blocks[1]
        value_blocks[0] = alpha * eli_value + (1 - alpha) * value_blocks[1]

        for i in range(1, self.N - 1):
            key_blocks[i] = key_blocks[i+1]
            value_blocks[i] = value_blocks[i+1]
        key_blocks[-1] = torch.zeros_like(key_blocks[0])
        value_blocks[-1] = torch.zeros_like(value_blocks[0])
        #setattr(self, "key" + str(self.mem_start), (
        #    alpha * eli_key
        #    + (1 - alpha) * getattr(self, "key" + str(self.mem_start+1))))
        #setattr(self, "value" + str(self.mem_start), (
        #    alpha * eli_value
        #    + (1 - alpha) * getattr(self, "value" + str(self.mem_start+1))))
        #for i in range(self.mem_start + 1, self.mem_end):
        #    setattr(self, "key" + str(i), getattr(self, "key" + str(i+1)))
        #    setattr(self, "value" + str(i), getattr(self, "value" + str(i+1)))

        #setattr(self, "key" + str(self.mem_end), torch.zeros(self.batch_size,
        #                                                     self.dk,
        #                                                     device=device))
        #setattr(self, "value" + str(self.mem_end), torch.zeros(self.L,
        #                                                       self.batch_size,
        #                                                       (self.dv * 
        #                                                       (self.args.nlayers+1)),
        #                                                       device=device))

        merge_matrix = torch.eye(key_num.size(0),
                                 key_num.size(0) - 1,
                                 device=key_num.device)
        merge_matrix = torch.cat((merge_matrix.new_zeros(key_num.size(0), 1), 
                                  merge_matrix), dim=1)
        merge_matrix[0][0], merge_matrix[0][1] = alpha, 1 - alpha
        key_num = torch.matmul(merge_matrix, key_num + 1)
        return key_blocks, value_blocks, key_num

    #def #p_discard(self, keys, values, key_num):
    #    #keys = self._get_keys()
    #    pos_start = torch.einsum("ib,j->ibj", 
    #                             key_num, key_num.new_ones(self.L) * (self.L))
    #    pos_seq = pos_start + torch.arange(self.L - 1, -1, -1, 
    #                                       dtype=key_num.dtype,
    #                                       device=key_num.device)
    #    pos_seq = pos_seq.reshape(-1, self.L)
    #    pos_emb = self.pos_emb(pos_seq)
    #    pos_emb = pos_emb.transpose(0, 1).reshape(self.N, -1, self.L * self.dv)
    #    probs = F.sigmoid(self.p_content(keys) + self.p_pos(pos_emb))
    #    probs = probs.transpose(0, 1)
    #    probs = probs.squeeze()
    #    probs = F.gumbel_softmax(probs, hard=True, dim=1)
    #    bsz, klen = probs.size(0), probs.size(1)

    #    _, indices = probs.topk(1, dim=1)
    #    move_matrix = torch.einsum("bi,ij->bij",
    #                                1.0 - probs, 
    #                                torch.eye(klen, device=probs.device))
    #    move_matrix = move_matrix.reshape(-1, klen)

    #    ind_sel = list(range(bsz * klen))

    #    for i in range(bsz):
    #        ind_del = i * klen + indices[i].item()
    #        ind_last = (i + 1) * klen - 1
    #        ind_sel.insert(ind_last + 1, ind_sel[ind_del])
    #        del ind_sel[ind_del]

    #    ind_sel = torch.tensor(ind_sel, dtype=torch.long, device=probs.device)
    #    move_matrix = move_matrix.index_select(0, ind_sel)
    #    move_matrix = move_matrix.reshape(bsz, klen, klen)

    #    key_num = torch.einsum("bij,jb->ib", move_matrix, key_num)
    #    key_num = key_num + 1.0

    #    keys = self._get_keys()
    #    values = self._get_values()

    #    keys = torch.einsum("bij,jbh->ibh", move_matrix, keys)
    #    values = torch.einsum("bij,jlbh->ilbh", move_matrix, values)

    #    for i in range(klen):
    #        setattr(self, "key" + str(i), keys[i])
    #        setattr(self, "value" + str(i), values[i])



    #    #for j, discard in enumerate(discards):
    #    #    i = discard[0]
    #    #    for pos in range(i, key_num.size(0) - 1):

    #    #        key_num[pos][j] = key_num[pos+1][j]

    #    #        getattr(self, "key" + str(pos)).index_copy_(
    #    #                    0, 
    #    #                    key_num.new_ones(1).to(torch.long) * j,
    #    #                    getattr(self, "key" + str(pos+1))[j].unsqueeze(0))
    #    #        getattr(self, "value" + str(pos)).index_copy_(
    #    #                    1, 
    #    #                    key_num.new_ones(1).to(torch.long) * j,
    #    #                    getattr(self, "value" + str(pos+1))[:,j,:].unsqueeze(1))

    #    #key_num.index_fill_(0, 
    #    #                    key_num.new_ones(1).to(torch.long) * (key_num.size(0) - 1),
    #    #                    0.)

    #    
    #    return key_num



