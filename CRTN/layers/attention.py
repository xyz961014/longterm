import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, keys, mask=None, scale=1.0):
        weights = torch.einsum("...bh,bnh->...bn", query, keys)
        weights = F.softmax(scale * weights / np.sqrt(keys.size(-1)), 2)

        return weights


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
        x = self.positional_encoding(x)
        for i in range(self.num_layers):
            if leftward:
                x = self.attn(x, mask=True)
            else:
                x = self.attn(x)
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
