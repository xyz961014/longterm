import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, keys, values):
        keys_T = keys.transpose(-2, -1)
        query_attn = query.view(-1, 1, query.size(-1))
        keys_T = keys_T.view(-1, keys_T.size(-2), keys_T.size(-1))
        values_attn = values.view(list(values.size())[:2]+[-1])
        #print(query_attn.shape, keys_T.shape, values_attn.shape)
        assert (query_attn.size(0) == keys_T.size(0) and query_attn.size(0) == values_attn.size(0) and keys_T.size(0) == values_attn.size(0)), "Batch size not justified, please check"
        weights = F.softmax(torch.matmul(query_attn, keys_T) / np.sqrt(keys_T.size(-2)), 2)
        #print(query_attn, keys_T, weights)
        outputs = torch.matmul(weights, values_attn)
        outputs = outputs.view([-1]+list(values.size())[2:])
        return weights, outputs
