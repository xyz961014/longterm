import argparse
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ipdb

import os
import sys
import re
sys.path.append("..")
from data import dataloader
import visdom
viz = visdom.Visdom()
assert viz.check_connection()

from utils.adaptive import ProjectedAdaptiveLogSoftmax

from models.CRTNModel import CRTNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args(args=None):
    #Arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/xyz/Documents/Dataset/ptb_sample", help="data path")
    parser.add_argument("--demo", action="store_true", help="demo mode")
    parser.add_argument("--load", type=str, default="",  help="load model from saved models")
    parser.add_argument("--cache_N", type=int, default=5,  help="cache size N")
    parser.add_argument("--cache_k", type=int, default=3,  help="choose k segment from cache")
    parser.add_argument("--func", type=str, default="", choices=["attention_map", "demo_words"], help="function to use, choices: attention_map, demo_words")
    parser.add_argument("--init_word", type=str, default="you", help="initial word for generation")
    parser.add_argument("--length", type=int, default=70,  help="generation text length")

    return parser.parse_args(args)

def evaluate(model, eval_data, criterion, args):
    model.set_batch_size(model.args.eval_batch_size)
    model.to(device)
    criterion.to(device)
    model.eval() 
    total_loss = 0.

    with torch.no_grad():
        for data, targets in eval_data:
            data, targets = data.to(device), targets.to(device)
            data, targets = data.t().contiguous(), targets.t().contiguous()

            output, _ = model(data)
            
            if model.args.adaptive:
                loss = criterion(output.view(-1, model.args.nhid), targets.view(-1))
                loss = loss.mean()
            else:
                loss = criterion(output.view(-1, model.args.vocab_size), targets.view(-1))
    
            total_loss += loss

    model.set_batch_size(model.args.batch_size)

    return total_loss / len(eval_data)

def attention_map(model, criterion, corpus, loader, seg_num=200):
    model.set_batch_size(model.args.eval_batch_size)
    model.to(device)
    criterion.to(device)
    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data, target = data.t().contiguous(), target.t().contiguous()
            
            if not i == seg_num:
                output, (attn_map, display) = model(data, draw=True)
                display = sorted(display, key=lambda x:x[0].item())
                weights = torch.cat([torch.tensor([d[1]], device=data.device) for d in display])
                weights = torch.cat((weights, torch.ones_like(weights[0]).view(1)))
                seq_len = data.size(0)
                weights = torch.einsum("k,l->kl", weights, torch.ones(seq_len, device=weights.device)).view(-1)

                column = [corpus.vocabulary.index2word[w.item()] for d in display for w in d[2].view(-1)]
                row = [corpus.vocabulary.index2word[w.item()] for w in data.view(-1)]
                column = column + row
                row = row[::-1]
                disdup = []
                for ind, word in enumerate(column):
                    while word in disdup:
                        word = word + " "
                        column[ind] = word
                    disdup.append(column[ind])
                for ind, word in enumerate(row):
                    while word in disdup:
                        word = " " + word
                        row[ind] = word
                    disdup.append(row[ind])
                opts = dict(columnnames=column,rownames=row, colormap="Electric")
                attn_map = attn_map.flip(0)
                viz.heatmap(attn_map * weights, opts=opts)
                ipdb.set_trace()
            else:
                output, _ = model(data)


def demo_words(model, criterion, corpus, init_word, length=100):
    model.set_batch_size(1)
    model.to(device)
    criterion.to(device)
    model.eval()
    
    num_steps = model.args.num_steps
    vocab_size = model.args.vocab_size
    cutoffs = model.args.cutoffs

    init_ind = corpus.vocabulary.word2index[init_word]
    print(corpus.vocabulary.index2word[init_ind], end= " ")
    sequence = [init_ind] + [0 for _ in range(num_steps-1)]
    sequence = torch.tensor(sequence).view(-1, 1).to(device)
    for i in range(length):
        if i == length - 1:
            output, _ = model(sequence)
        else:
            output, _ = model(sequence, renew=False)
        head_prob, tails = criterion(output.view(-1, model.args.nhid), sequence.view(-1), output=True)
        word_ind = head_prob.max(1)[1][(i+1) % num_steps - 1]
        if word_ind >= cutoffs[0]:
            cluster = word_ind - cutoffs[0]
            word_ind = tails[cluster].max(1)[1][(i+1) % num_steps - 1] + cutoffs[cluster]
        
        word = corpus.vocabulary.index2word[word_ind.item()]
        if word == "<eos>":
            print(".", end=" ")
        elif word == "<pad>" or word == "<sos>":
            pass
        else:
            print(word, end=" ")
        if i + 1 < num_steps:
            sequence[i+1] = word_ind
        else:
            sequence = [word_ind.item()] + [0 for _ in range(num_steps-1)]
            sequence = torch.tensor(sequence).view(-1, 1).to(device)
    print("")


def main(args):
    

    checkpoint = torch.load(args.load)
    model_args = checkpoint["model_args"]
    model_args.cache_N = args.cache_N
    model_args.cache_k = args.cache_k
    model_args.demo = args.demo
    if args.demo:
        model_args.eval_batch_size = 1

    model_state_dict = checkpoint["model_state_dict"]
    keys = model_state_dict.copy().keys()
    for key in keys:
        if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key) or re.match(r"encoder.pos_emb_bank", key):
            model_state_dict.pop(key)



    print("Loading data from %s" % args.data)
    datatime_begin = time.time()

    corpus = dataloader.Corpus(args.data)
    args.vocab_size = corpus.vocabulary.num_words
    eval_batch_size = model_args.eval_batch_size

    valid_loader = corpus.get_valid_loader(batch_size=eval_batch_size, num_steps=model_args.num_steps)
    test_loader = corpus.get_test_loader(batch_size=eval_batch_size, num_steps=model_args.num_steps)

    print("Data loading finished. time: {:.3f} s".format(time.time() - datatime_begin))
   
    if args.demo:
        model = CRTNModel(model_args, corpus)
    else:
        model = CRTNModel(model_args)
    model.load_state_dict(model_state_dict, strict=False)

    cutoffs = model.args.cutoffs
    tie_projs = model.args.tie_projs
    #if model.args.adaptive:
    #    if model.args.dataset == "ptb":
    #        cutoffs = [20000, 40000, 80000]
    #        tie_projs += [True] * 3
    #    elif model.args.dataset == "wt103":
    #        cutoffs = [20000, 40000, 80000]
    #        tie_projs += [True] * 3

    if model.args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(model.args.vocab_size, model.args.emsize, model.args.nhid, cutoffs, div_val=model.args.div_val, init_std=model.args.init_std) 
        if model.args.tied:
            for i in range(len(criterion.out_layers)):
                criterion.out_layers[i].weight = model.encoder.embedding.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and model.args.div_val == 1 and model.args.nhid != model.args.emsize:
                    criterion.out_projs[i] = model.encoder.embedding.emb_projs[0]
                elif tie_proj and model.args.div_val != 1:
                    criterion.out_projs[i] = model.encoder.embedding.emb_projs[i]
        criterion.load_state_dict(checkpoint["criterion"])

    else:
        criterion = nn.CrossEntropyLoss()


    if args.func == "attention_map":
        attention_map(model, criterion, corpus, valid_loader)
    elif args.func == "demo_words":
        init_word = args.init_word
        length = args.length
        while True:
            demo_words(model, criterion, corpus, init_word=init_word, length=length)
            init_word = input("Input initial word:")
            length = int(input("Input text length:"))
    else:
        valid_loss = evaluate(model, valid_loader, criterion, args)
        print('=' * 89)
        print('| valid loss {:5.2f} |valid ppl {:8.2f}'.format(
            valid_loss, math.exp(valid_loss)))
        test_loss = evaluate(model, test_loader, criterion, args)
        print('=' * 89)
        print('| test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == "__main__":
    args = parse_args()
    main(args)
