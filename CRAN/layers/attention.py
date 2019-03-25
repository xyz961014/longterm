import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, keys, values, mask=None):
        keys_T = keys.transpose(-2, -1)
        keys_T = keys_T.view(-1, keys_T.size(-2), keys_T.size(-1))
        values_attn = values.view(list(values.size())[:2]+[-1])
        query_attn = query.view(values_attn.size(0), -1, query.size(-1))
        try:
            assert (query_attn.size(0) == keys_T.size(0) and query_attn.size(0) == values_attn.size(0) and keys_T.size(0) == values_attn.size(0)), "Batch size not justified, please check"
        except:
            print("Q:%s, K:%s, V:%s" % (query_attn.shape, keys_T.shape, values_attn.shape))
        assert (query_attn.size(0) == keys_T.size(0) and query_attn.size(0) == values_attn.size(0) and keys_T.size(0) == values_attn.size(0)), "Batch size not justified, please check"
        if mask is not None:
            mask = torch.cat(tuple(mask.view([1]+list(mask.size())) for _ in range(keys_T.size(0))), 0)
            keys_T = torch.matmul(keys_T, mask)
        weights = F.softmax(torch.matmul(query_attn, keys_T) / np.sqrt(keys_T.size(-2)), 2)
        #print(query_attn, keys_T, weights)
        outputs = torch.matmul(weights, values_attn)
        outputs = outputs.view([-1]+list(values.size())[2:])
        return weights, outputs


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_head, num_steps, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.drop = nn.Dropout(dropout)
        self.attn = DotProductAttention()
        self.LinO = nn.Linear(d_model, d_model)
        self.LinQs = nn.ModuleList([nn.Linear(d_model, d_model // num_head) for i in range(num_head)])
        self.LinKs = nn.ModuleList([nn.Linear(d_model, d_model // num_head) for i in range(num_head)])
        self.LinVs = nn.ModuleList([nn.Linear(d_model, d_model // num_head) for i in range(num_head)])

        self.mask_matrix = torch.tensor([[1.0 if j < i else 0.0 for j in range(num_steps)] for i in range(num_steps)])

    def forward(self, inputs, mask=False):
        inputs = inputs.transpose(0, 1).contiguous()
        if mask:
            self.mask_matrix = self.mask_matrix.to(inputs.device)
            heads = tuple(self.attn(self.LinQs[i](self.drop(inputs)), self.LinKs[i](self.drop(inputs)), self.LinVs[i](self.drop(inputs)), self.mask_matrix)[1] for i in range(self.num_head))
        else:
            heads = tuple(self.attn(self.LinQs[i](self.drop(inputs)), self.LinKs[i](self.drop(inputs)), self.LinVs[i](self.drop(inputs)))[1] for i in range(self.num_head))
        concat = torch.cat(heads, -1)
        return self.LinO(concat)




class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_head, d_ff, num_steps, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.attn = MultiheadSelfAttention(d_model, num_head, num_steps, dropout)
        self.drop = nn.Dropout(dropout)
        self.FFN_1 = nn.Linear(d_model, d_ff)
        self.FFN_2 = nn.Linear(d_ff, d_model)
        self.pos = 0

    def forward(self, x, leftward=False):
        size = x.size()
        x = self.positional_encoding(x)
        for i in range(self.num_layers):
            if leftward:
                x = self.attn(x, mask=True).view(size)
            else:
                x = self.attn(x).view(size)
            x = self.FFN_2(F.relu(self.FFN_1(x)))
        return x

    def positional_encoding(self, x):

        def pe_item(pos, i, d_model):
            denominator = math.pow(10000, 2 * i / d_model)
            if i % 2 == 0:
                return math.sin(pos / denominator)
            else:
                return math.cos(pos / denominator)
            
        length, batch_size, hidden_size = x.size()
        pos_enc = torch.tensor([[pe_item(pos, i, hidden_size) for i in range(hidden_size)] for pos in range(self.pos, self.pos+length)], device=x.device)
        pos_enc = torch.cat(tuple(pos_enc.view([1]+list(pos_enc.size())) for _ in range(batch_size)), 0)
        x = x + pos_enc.transpose(0, 1).contiguous()
        self.pos += length

        return x
