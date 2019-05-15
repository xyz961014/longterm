import sys
import math
import ipdb
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
from CRTN.utils.adaptive import AdaptiveEmbedding

from CRTN.layers.attention import DotProductAttention
from CRTN.layers.transformer import TransformerLM
from CRTN.layers.cache import Cache

class CRTNModel(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.corpus = corpus
        self.demo = self.args.demo

        self.drop = nn.Dropout(args.dropout)
        self.attn = DotProductAttention()

        self.cache = Cache(args, corpus)

        self.encoder = TransformerLM(args, corpus)

    def to(self, device):
        super().to(device)
        self.cache.to(device)
        self.encoder.to(device)

    def set_batch_size(self, batch_size):
        self.cache.set_batch_size(batch_size)
        self.encoder.set_batch_size(batch_size)


    def forward(self, inputs):

        if self.args.wise_summary:
            _, wise_inputs = self.encoder(inputs)
            query = wise_inputs[-1]
        else:
            query = self.encoder.embedding(inputs)
        
        if self.demo:
            weights, indices, zones, words = self.cache(query)
        else:
            weights, indices, zones = self.cache(query)
            words=None

        output, mems = self.encoder(inputs, zones, weights, indices, words)

        self.cache.renew(mems, inputs)

        return output 



