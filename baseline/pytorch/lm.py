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
from CRAN.utils.adaptive import ProjectedAdaptiveLogSoftmax
from transformer import TransformerLM

from tensorboardX import SummaryWriter

import ipdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='ptb',
                        help='data corpus name')
    parser.add_argument('--demo', action='store_true',
                        help='demo mode')
    parser.add_argument('--adam', action='store_true',
                        help='adam optimizer')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--rnn', action='store_true', 
                        help='enable rnn mode, disable transformer mode')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=4,
                        help='number of heads')
    parser.add_argument('--d_ff', type=int, default=1000,
                        help='dimension of feed-forward')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='length of memory')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'constant'],
                        help='lr scheduler to use')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=60, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, 
                        help='eval batch size')
    parser.add_argument('--num_steps', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0.0, init_std)')
    parser.add_argument('--tied', action="store_true",
                        help='tied embedding weights')
    parser.add_argument('--attn_type', type=int, default=0, choices=[0, 1],
                        help='attention type, 0 for vaswani;1 for transformer-xl')
    parser.add_argument('--multi_gpu', action="store_true",
                        help='enable multiple gpus')
    parser.add_argument('--adaptive', action="store_true",
                        help='use adaptive embedding and softmax')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adaptive input and softmax')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model',
                        help='path to save the final model')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    args = parser.parse_args()
    return args

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


#class TransformerLM(nn.Module):
#    def __init__(self, args):
#        super().__init__()
#        self.args = args
#        self.drop = nn.Dropout(args.dropout)
#        self.embedding = nn.Embedding(args.vocab_size, args.emsize)
#        self.encoder = Transformer(
#                num_layers=args.nlayers,
#                d_model=args.nhid,
#                num_head=args.nhead,
#                d_ff=args.d_ff,
#                num_steps=args.num_steps,
#                dropout=args.dropout)
#        self.decoder = nn.Linear(args.nhid, args.vocab_size)
#
#        self.init_weights()
#
#    def forward(self, x, is_training=True):
#        emb = self.drop(self.embedding(x))
#        output = self.encoder(emb, leftward=True)
#        output = self.drop(output)
#        if is_training:
#            tag = self.decoder(output)
#        else:
#            tag = self.decoder(output[:,-1])
#        return tagpack_padded_sequence
#
#    def init_weights(self):
#        initrange = 0.1
#        self.embedding.weight.data.uniform_(-initrange, initrange)
#        self.decoder.bias.data.zero_()
#        self.decoder.weight.data.uniform_(-initrange, initrange)
#
#    def init_hidden(self, batch_size):
#        return 0





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

def train(model, train_loader, criterion, args, epoch, optimizer, scheduler):
    model.train()
    start_time = time.time()
    if args.rnn:
        hidden = model.init_hidden(args.batch_size)
    memory = None
    total_loss = 0.

    for batch, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        data, targets = data.t().contiguous(), targets.t().contiguous()
        model.zero_grad()
        if args.rnn:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        else:
            output, memory = model(data, memory)
        if args.adaptive:
            loss = criterion(output.view(-1, args.nhid), targets.view(-1))
            loss = loss.mean()
        else:
            loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if args.scheduler == "cosine":
            scheduler.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.3f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_loader), optimizer.state_dict()["param_groups"][0]["lr"],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0.
            start_time = time.time()


def evaluate(model, eval_loader, criterion, args):
    model.eval()
    total_loss = 0.
    if args.rnn:
        hidden = model.init_hidden(args.eval_batch_size)
    memory = None
    with torch.no_grad():
        for i, (data, targets) in enumerate(eval_loader):
            data, targets = data.to(device), targets.to(device)
            data, targets = data.transpose(0, 1).contiguous(), targets.transpose(0, 1).contiguous()
            if args.rnn:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            else:
                output, memory = model(data, memory)

            if args.adaptive:
                loss = criterion(output.view(-1, args.nhid), targets.view(-1))
                loss = loss.mean()
            else:
                loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))

            total_loss += loss

    return total_loss / len(eval_loader)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        if args.dataset == "ptb":
            cutoffs = [2000, 4000, 8000]
            tie_projs += [True] * 3
        elif args.dataset == "wt103":
            cutoffs = [20000, 40000, 80000]
            tie_projs += [True] * 3

    ### Load Data ###
    
    print("Loading data from %s" % args.data)
    datatime_begin = time.time()

    corpus = dataloader.Corpus(args.data)
    args.vocab_size = corpus.vocabulary.num_words
    eval_batch_size = args.eval_batch_size

    train_loader = corpus.get_train_loader(batch_size=args.batch_size, num_steps=args.num_steps)
    valid_loader = corpus.get_valid_loader(batch_size=eval_batch_size, num_steps=args.num_steps)
    test_loader = corpus.get_test_loader(batch_size=eval_batch_size, num_steps=args.num_steps)

    print("Data loading finished. time: {:.3f} s".format(time.time() - datatime_begin))

    if args.load:
        checkpoint = torch.load(args.load)
        model_args = checkpoint["model_args"]
        model_args.demo = args.demo
        args.rnn = model_args.rnn
        if args.demo:
            model_args.batch_size = 1
        if model_args.rnn:
            model = RNNModel(model_args).to(device)
        else:
            model = TransformerLM(
                    vocab_size=model_args.vocab_size,
                    num_layer=model_args.nlayers,
                    num_head=model_args.nhead,
                    d_model=model_args.nhid,
                    d_head=model_args.nhid // model_args.nhead,
                    d_ff=model_args.d_ff,
                    d_embedding=model_args.emsize,
                    tied_weights=model_args.tied,
                    mem_len=model_args.mem_len,
                    attn_type=model_args.attn_type,
                    init_std=model_args.init_std,
                    adaptive=model_args.adaptive,
                    div_val=model_args.div_val,
                    cutoffs=cutoffs,
                    dropout=model_args.dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if args.rnn:
            model = RNNModel(args).to(device)
        else:
            model = TransformerLM(
                    vocab_size=args.vocab_size,
                    num_layer=args.nlayers,
                    num_head=args.nhead,
                    d_model=args.nhid,
                    d_head=args.nhid // args.nhead,
                    d_ff=args.d_ff,
                    d_embedding=args.emsize,
                    tied_weights=args.tied,
                    mem_len=args.mem_len,
                    attn_type=args.attn_type,
                    init_std=args.init_std,
                    adaptive=args.adaptive,
                    div_val=args.div_val,
                    cutoffs=cutoffs,
                    dropout=args.dropout)


    
    if args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, args.emsize, args.nhid, cutoffs, div_val=args.div_val, init_std=args.init_std) 
        if args.tied:
            for i in range(len(criterion.out_layers)):
                criterion.out_layers[i].weight = model.embedding.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and args.div_val == 1 and args.nhid != args.emsize:
                    criterion.out_projs[i] = model.embedding.emb_projs[0]
                elif tie_proj and args.div_val != 1:
                    criterion.out_projs[i] = model.embedding.emb_projs[i]

    else:
        criterion = nn.CrossEntropyLoss()
        
    model = model.to(device)
    criterion = criterion.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model, dim=1)

    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)
    elif args.scheduler == "constant":
        scheduler = None


    try:
        best_eval_loss = float('inf')
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(model, train_loader, criterion, args, epoch, optimizer, scheduler)
            eval_loss = evaluate(model, valid_loader, criterion, args)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               eval_loss, math.exp(eval_loss)))
            print('-' * 89)
            if eval_loss < best_eval_loss:
                torch.save({
                    "model_args": args,
                    "model_state_dict": model.state_dict(),
                    }, "save/" + args.save + "_" + str(epoch) + ".pt")
                with open("save/" + args.save + "_best.pt", "wb") as f:
                    torch.save(model, f)
                best_eval_loss = eval_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    with open("save/" + args.save + "_best.pt", "rb") as f:
        model = torch.load(f)
    test_loss = evaluate(model, test_loader, criterion, args)
    print('=' * 89)
    print('| best valid loss {:5.2f} | best valid ppl {:8.2f}'.format(
        best_eval_loss, math.exp(best_eval_loss)))
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)



if __name__ == "__main__":
    args = parse_args()
    main(args)
