import time
import argparse

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import os
import sys
sys.path.append("../..")
from CRAN.data import dataloader
from CRAN.layers.attention import Transformer

from tensorboardX import SummaryWriter

import ipdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/xyz/Documents/Dataset/ptb_sample',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--rnn', action='store_true', 
                        help='enable rnn mode, disable transformer mode')
    parser.add_argument('--emsize', type=int, default=512,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='number of heads')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of feed-forward')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, 
                        help='eval batch size')
    parser.add_argument('--num_steps', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args()
    return args

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class TransformerLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.drop = nn.Dropout(args.dropout)
        self.embedding = nn.Embedding(args.vocab_size, args.emsize)
        self.encoder = Transformer(
                num_layers=args.nlayers,
                d_model=args.nhid,
                num_head=args.nhead,
                d_ff=args.d_ff,
                dropout=args.dropout)
        self.decoder = nn.Linear(args.nhid, args.vocab_size)

        self.init_weights()

    def forward(self, x, is_training=True):
        emb = self.drop(self.embedding(x))
        output = self.encoder(emb)
        if is_training:
            tag = self.decoder(output)
        else:
            tag = self.decoder(output[:,-1])
        return tag

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        return 0





class RNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
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

def train(model, train_loader, criterion, optimizer, args, epoch):
    model.train()
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    total_loss = 0.

    for batch, (data, targets) in enumerate(train_loader):
        data, targets = data.transpose(0, 1).contiguous(), targets.transpose(0, 1).contiguous()
        model.zero_grad()
        if args.rnn:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        else:
            output = model(data)
        loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_loader), args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_loader, criterion, args):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i, (data, targets) in enumerate(eval_loader):
            data, targets = data.transpose(0, 1).contiguous(), targets.transpose(0, 1).contiguous()
            if args.rnn:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            else:
                output = model(data)
            output_flat = output.view(-1, args.vocab_size)
            total_loss += criterion(output_flat, targets.view(-1)).item()

    return total_loss / (len(eval_loader) - 1)



def main(args):
    torch.manual_seed(args.seed)

    corpus = dataloader.Corpus(args.data)

    eval_batch_size = args.eval_batch_size
    train_loader = corpus.get_train_loader(batch_size=args.batch_size, num_steps=args.num_steps)
    valid_loader = corpus.get_valid_loader(batch_size=eval_batch_size, num_steps=args.num_steps)
    test_loader = corpus.get_test_loader(batch_size=eval_batch_size, num_steps=args.num_steps)

    args.vocab_size = corpus.vocabulary.num_words

    if args.rnn:
        model = RNNModel(args).to(device)
    else:
        model = TransformerLM(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(model, train_loader, criterion, optimizer, args, epoch)
        eval_loss = evaluate(model, valid_loader, criterion, args)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           eval_loss, math.exp(eval_loss)))
        print('-' * 89)

    test_loss = evaluate(model, test_loader, criterion, args)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)



if __name__ == "__main__":
    args = parse_args()
    main(args)
