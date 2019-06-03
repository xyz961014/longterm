import argparse
import math
import time
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import ipdb
import visdom
viz = visdom.Visdom()
assert viz.check_connection()


import os
import sys
sys.path.append("..")
sys.path.append("../..")

                    
from data import dataloader
from data.dataloader import textDataset

from torch.utils.data import DataLoader

from models.CRTNModel import CRTNModel
from utils.adaptive import ProjectedAdaptiveLogSoftmax

from baseline.pytorch.transformer import TransformerLM

from baseline.pytorch.rnn import RNNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--model_names", nargs="+", type=str, choices=['LSTM', 'XL', 'CRTN'], help="model names. 'LSTM' for LSTM model; 'XL' for Transformer-XL model; CRTN for this work")
    parser.add_argument("--model_paths", nargs="+", type=str, help="model paths")
    parser.add_argument("--initc", type=int, default=10, help="initial c")
    parser.add_argument("--initr", type=float, default=1.0, help="initial r")
    parser.add_argument("--delta", type=int, default=10, help="step of increasing c")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--bias", type=int, default=0, help="bias")
    return parser.parse_args()

def loss(model, criterion, corpus, c, batch_size=5, bias=0):
    txt_len = corpus.valid_data.data.view(-1).size(0)
    raw_data = corpus.valid_data.data.view(-1).narrow(0, 0, (txt_len // batch_size) * batch_size)
    raw_data = raw_data.view(batch_size, -1)
    data, target = raw_data[:,-c-2-bias:-2-bias], raw_data[:,-c-1-bias:-1-bias]
    data, target = data.t().contiguous(), target.t().contiguous()
    seg_len = c
    losses = []
    with torch.no_grad():
        model.eval()
        criterion.eval()
        if model.name == "LSTM":
            datas = [data[i*seg_len:(i+1)*seg_len] for i in range(c // seg_len)]
            targets = [target[i*seg_len:(i+1)*seg_len] for i in range(c // seg_len)]
            if c % seg_len:
                datas.append(data[(c // seg_len):])
                targets.append(target[(c // seg_len):])

            hidden = model.init_hidden(batch_size)
            for data, target in list(zip(datas, targets)):
                data, target = data.to(device), target.to(device)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(seg_len * batch_size, -1), target.view(-1))
                loss = loss.view(seg_len, batch_size)
                losses.append(loss)
            return torch.cat(losses, 0).mean(1)
        elif model.name == "XL":
            seg_len = model.num_steps
            datas = [data[i*seg_len:(i+1)*seg_len] for i in range(c // seg_len)]
            targets = [target[i*seg_len:(i+1)*seg_len] for i in range(c // seg_len)]
            if c % seg_len:
                datas.append(data[(c // seg_len) * seg_len:])
                targets.append(target[(c // seg_len) * seg_len:])

            memory = None
            for data, target in list(zip(datas, targets)):
                data, target = data.to(device), target.to(device)
                cur_len = data.size(0)
                output, memory = model(data, memory)
                loss = criterion(output.view(cur_len * batch_size, -1), target.view(-1))
                loss = loss.view(cur_len, batch_size)
                losses.append(loss)
            return torch.cat(losses, 0).mean(1)
        elif model.name == "CRTN":
            seg_len = model.args.num_steps
            datas = [data[i*seg_len:(i+1)*seg_len] for i in range(c // seg_len)]
            targets = [target[i*seg_len:(i+1)*seg_len] for i in range(c // seg_len)]

            model.set_batch_size(batch_size)
            for data, target in list(zip(datas, targets)):
                data, target = data.to(device), target.to(device)
                cur_len = data.size(0)
                output = model(data)
                loss = criterion(output.view(cur_len * batch_size, -1), target.view(-1))
                loss = loss.view(cur_len, batch_size)
                losses.append(loss)
            return torch.cat(losses, 0).mean(1)
        
        




def relative_loss(model, criterion, corpus, base, c, cp, tau):
    losscp = loss(model, criterion, corpus, cp)
    func = torch.cat((base[tau], losscp[tau])).view(-1, tau.size(0))
    func = func.t().contiguous()
    func = func.min(1)[0]
    return func.mean().item()

def relative_gain(model, criterion, corpus, base, c, cp, tau):
    fbase = relative_loss(model, criterion, corpus, base, c, c, tau)
    fp = relative_loss(model, criterion, corpus, base, c, cp, tau)
    return (fbase - fp) / fbase

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    corpus = dataloader.Corpus(args.data)
    models = []

    for path, name in list(zip(args.model_paths, args.model_names)):
        checkpoint = torch.load(path)
        model_args = checkpoint["model_args"]
        model_state_dict = checkpoint["model_state_dict"]

        if name == 'LSTM':
            model = RNNModel(model_args)
            criterion = nn.CrossEntropyLoss()
        elif name == 'XL':
            model = TransformerLM(
                    vocab_size=model_args.vocab_size,
                    num_layer=model_args.nlayers,
                    num_head=model_args.nhead,
                    d_model=model_args.nhid,
                    d_head=model_args.nhid // model_args.nhead,
                    d_ff=model_args.d_ff,
                    d_embedding=model_args.emsize,
                    tied_weights=model_args.tied,
                    num_steps=model_args.num_steps,
                    mem_len=model_args.mem_len,
                    attn_type=model_args.attn_type,
                    init_std=model_args.init_std,
                    adaptive=model_args.adaptive,
                    div_val=model_args.div_val,
                    cutoffs=model_args.cutoffs,
                    dropout=model_args.dropout)
        elif name == 'CRTN':
            model = CRTNModel(model_args)
            keys = model_state_dict.copy().keys()
            for key in keys:
                if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key) or re.match(r"encoder.pos_emb_bank", key):
                    model_state_dict.pop(key)

        if model_args.adaptive:
            criterion = ProjectedAdaptiveLogSoftmax(model_args.vocab_size, model_args.emsize, model_args.nhid, model_args.cutoffs, div_val=model_args.div_val, init_std=model_args.init_std) 
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
        model.load_state_dict(model_state_dict, strict=False)
        criterion.load_state_dict(checkpoint["criterion"])
        model.name = name

        model.to(device)
        criterion.to(device)

        models.append((model, criterion))

    
    for i, (model, criterion) in enumerate(models):
        c = args.initc
        delta = args.delta
        cp = c
        gain = 1.0
        ite = 0
        while gain >= 0.01:
            c = cp
            cp = cp + delta
            losses = []
            for j, (mod, crit) in enumerate(models):
                l = loss(mod, crit, corpus, c)
                losses.append(l)

            #l = loss(model, criterion, corpus, c)
            #losses.append(l)

            lossc = torch.cat(losses).view(-1, c).t().contiguous()
            base = lossc.min(1)[0]

            basec, tau = base.topk(round(c * args.initr))
            tau = tau - c
            gain = relative_gain(model, criterion, corpus, base, c, cp, tau)
            print(gain)
            ite += 1
            viz.line(np.array([[np.NaN] * i + [gain] + [np.NaN] * (len(args.model_names) - i - 1)]), np.array([c]), opts={
                'legend': args.model_names,
                'showlegend': True
                }, win="gain", update="append")
        print("%s RECL: %s" % (model.name, c))


if __name__ == "__main__":
    args = parse_args()
    main(args)
