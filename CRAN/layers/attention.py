import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, keys, values):
        keys_T = keys.transpose(-2, -1)
        keys_T = keys_T.view(-1, keys_T.size(-2), keys_T.size(-1))
        values_attn = values.view(list(values.size())[:2]+[-1])
        query_attn = query.view(values_attn.size(0), -1, query.size(-1))
        try:
            assert (query_attn.size(0) == keys_T.size(0) and query_attn.size(0) == values_attn.size(0) and keys_T.size(0) == values_attn.size(0)), "Batch size not justified, please check"
        except:
            print("Q:%s, K:%s, V:%s" % (query_attn.shape, keys_T.shape, values_attn.shape))
        assert (query_attn.size(0) == keys_T.size(0) and query_attn.size(0) == values_attn.size(0) and keys_T.size(0) == values_attn.size(0)), "Batch size not justified, please check"
        weights = F.softmax(torch.matmul(query_attn, keys_T) / np.sqrt(keys_T.size(-2)), 2)
        #print(query_attn, keys_T, weights)
        outputs = torch.matmul(weights, values_attn)
        outputs = outputs.view([-1]+list(values.size())[2:])
        return weights, outputs


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.drop = nn.Dropout(dropout)
        self.attn = DotProductAttention()
        self.LinO = nn.Linear(d_model, d_model)
        self.LinQs = nn.ModuleList([nn.Linear(d_model, d_model // num_head) for i in range(num_head)])
        self.LinKs = nn.ModuleList([nn.Linear(d_model, d_model // num_head) for i in range(num_head)])
        self.LinVs = nn.ModuleList([nn.Linear(d_model, d_model // num_head) for i in range(num_head)])

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1).contiguous()
        heads = tuple(self.attn(self.LinQs[i](self.drop(inputs)), self.LinKs[i](self.drop(inputs)), self.LinVs[i](self.drop(inputs)))[1] for i in range(self.num_head))
        concat = torch.cat(heads, -1)
        return self.LinO(concat)




class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_head, d_ff, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.attn = MultiheadSelfAttention(d_model, num_head, dropout)
        self.drop = nn.Dropout(dropout)
        self.FFN_1 = nn.Linear(d_model, d_ff)
        self.FFN_2 = nn.Linear(d_ff, d_model)


    def forward(self, x):
        size = x.size()
        for i in range(self.num_layers):
            x = self.attn(x).view(size)
            x = self.FFN_2(F.relu(self.FFN_1(x)))
        return x



