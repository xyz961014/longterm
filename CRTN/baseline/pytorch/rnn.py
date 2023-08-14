import time
import argparse

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchnlp.nn import LockedDropout
import os
import sys
import ipdb

class RNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.embedding = nn.Embedding(args.vocab_size, args.emsize)
        self.rnn_type = args.model
        if args.model in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, args.model)(
                    input_size=args.emsize,
                    hidden_size=args.nhid,
                    num_layers=args.nlayers,
                    dropout=args.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.model]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(args.emsize, args.nhid, args.nlayers, nonlinearity=nonlinearity, dropout=args.dropout)

        self.hidden2tag = nn.Linear(args.nhid, args.vocab_size)

        self.init_weights()
        self.nhid = args.nhid
        self.nlayers = args.nlayers

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.zero_()
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.embedding(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        
        tag = self.hidden2tag(output)
        
        return tag, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)


