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
import torchtext

from models.CRTNModel import CRTNModel
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from data.dataloader import TextDataset, ExistingDataset

from baseline.pytorch.transformer import TransformerLM

from baseline.pytorch.rnn import RNNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "c_primeu")

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample/',
                        help='location of the data corpus')
    parser.add_argument('--datasets', type=str, choices=["fromfile", "ptb", "wt103"], 
                        default="ptb", help='load datasets from torchtext')
    parser.add_argument('--vocab_size', type=int, default=10000)
    # models
    parser.add_argument("--model_paths", nargs="+", type=str, 
                        help="model paths")
    # xl settings
    parser.add_argument('--num_steps', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--mem_len', type=int, default=140,
                        help='length of memory')
    # model settings
    parser.add_argument('--num_steps', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--neighbor_len', type=int, default=80,
                        help='length of memory')
    parser.add_argument("--cache_N", type=int, default=20, 
                        help="size of Cache, default: 10")
    parser.add_argument("--cache_k", type=int, default=8, 
                        help="select top k values, default: 8")
    parser.add_argument("--cache_L", type=int, default=20, 
                        help="length of segments in cache, default: 20")
    # recl settings
    parser.add_argument("--init_c", type=int, default=10, 
                        help="initial c")
    parser.add_argument("--top_r", type=float, default=1.0, 
                        help="ratio r of worst words to attention")
    parser.add_argument("--delta", type=int, default=10, 
                        help="step of increasing c")
    parser.add_argument("--threshold", type=float, default=0.01, 
                        help="addition ratio threshold to stop")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="batch size")
    parser.add_argument("--bias", type=int, default=0, help="bias")
    # setting
    parser.add_argument("--seed", type=int, default=1111, 
                        help="random seed")
    parser.add_argument('--device', type=int, default=0,
                        help='device number')
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
        
        




def relative_loss(model, criterion, corpus, base, c, c_prime, tau):
    lossc_prime = loss(model, criterion, corpus, c_prime)
    func = torch.cat((base[tau], lossc_prime[tau])).view(-1, tau.size(0))
    func = func.t().contiguous()
    func = func.min(1)[0]
    return func.mean().item()

def relative_gain(model, criterion, corpus, base, c, c_prime, tau):
    fbase = relative_loss(model, criterion, corpus, base, c, c, tau)
    fp = relative_loss(model, criterion, corpus, base, c, c_prime, tau)
    return (fbase - fp) / fbase

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    models = []

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("c_primeu")
    torch.cuda.set_device(device)


    for path in args.model_paths:
        checkpoint = torch.load(path, map_location=device)
        model_args = checkpoint["model_args"]
        name = model_args.name

        # redefine hyperparams

        if not model_args.num_steps == args.num_steps:
            print("REDEFINE num_steps: {} --> {}".format(model_args.num_steps, 
                                                         args.num_steps))
            model_args.num_steps = args.num_steps
        if not model_args.mem_len == args.mem_len:
            print("REDEFINE mem_len: {} --> {}".format(model_args.mem_len, 
                                                       args.mem_len))
            model_args.mem_len = args.mem_len
        if not model_args.neighbor_len == args.neighbor_len:
            print("REDEFINE neighbor_len: {} --> {}".format(model_args.neighbor_len, 
                                                            args.neighbor_len))
            model_args.neighbor_len = args.neighbor_len
        if not model_args.cache_N == args.cache_N:
            print("REDEFINE cache_N: {} --> {}".format(model_args.cache_N, 
                                                       args.cache_N))
            model_args.cache_N = args.cache_N
        if not model_args.cache_k == args.cache_k:
            print("REDEFINE cache_k: {} --> {}".format(model_args.cache_k, 
                                                       args.cache_k))
            model_args.cache_k = args.cache_k
        if not model_args.cache_L == args.cache_L:
            print("REDEFINE cache_L: {} --> {}".format(model_args.cache_L, 
                                                       args.cache_L))
            model_args.cache_L = args.cache_L

        model_args.device = args.device
        model_args.batch_size = args.batch_size


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
                    num_steps=args.num_steps,
                    mem_len=model_args.mem_len,
                    clamp_len=model_args.clamp_len,
                    same_length=model_args.same_length,
                    init_std=model_args.init_std,
                    adaptive=model_args.adaptive,
                    div_val=model_args.div_val,
                    cutoffs=model_args.cutoffs,
                    dropout=model_args.dropout,
                    dropatt=model_args.dropatt,
                    dropemb=model_args.dropemb,
                    dropinp=model_args.dropinp,
                    dropwei=model_args.dropwei,
                    dropfor=model_args.dropfor,
                    drophid=model_args.drophid,
                    theta=model_args.theta,
                    theta_alpha=model_args.theta_annealing_alpha,
                    apex=model_args.apex,
                    no_pos_bias=model_args.no_pos_bias
                    )
            model.load_state_dict(checkpoint["model_state_dict"])
        elif name == 'CRTN':
            model = CRTNModel(model_args)
            model.load_state_dict(checkpoint["model_state_dict"])

        if args.adaptive:
            criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, 
                                                    args.emsize, 
                                                    args.nhid, 
                                                    args.cutoffs, 
                                                    div_val=args.div_val, 
                                                    init_std=args.init_std,
                                                    proj_init_std=args.proj_init_std,
                                                    mos=args.mos,
                                                    n_experts=args.n_experts,
                                                    dropmos=0
                                                    ) 
            criterion.load_state_dict(checkpoint["criterion"])
        else:
            criterion = nn.CrossEntropyLoss()

        model.to(device)
        criterion.to(device)

        models.append((model, criterion))

    c = args.init_c
    delta = args.delta
    threshold = args.threshold
    for i, (model, criterion) in enumerate(models):
        c_prime = c
        gain = 1.0
        iters = 0
        while gain >= threshold:
            c = c_prime
            c_prime = c_prime + delta
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
            gain = relative_gain(model, criterion, corpus, base, c, c_prime, tau)
            print(gain)
            iters += 1
            viz.line(np.array([[np.NaN] * i + [gain] + [np.NaN] * (len(args.model_names) - i - 1)]), np.array([c]), opts={
                'legend': args.model_names,
                'showlegend': True
                }, win="gain", update="append")
        print("%s RECL: %s" % (model.name, c))


if __name__ == "__main__":
    args = parse_args()
    main(args)
