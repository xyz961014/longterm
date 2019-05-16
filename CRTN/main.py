import time
import re
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
sys.path.append("..")
from data import dataloader
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from models.CRTNModel import CRTNModel

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
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_dk", type=int, default=200, 
                        help="dimension of key, default: 200")
    parser.add_argument("--cache_k", type=int, default=3, 
                        help="select top k values, default: 3")
    parser.add_argument('--multi_gpu', action="store_true",
                        help='enable multiple gpus')
    parser.add_argument('--adaptive', action="store_true",
                        help='use adaptive embedding and softmax')
    parser.add_argument('--no_summary', action="store_true",
                        help='use the output of the transformer layer as key')
    parser.add_argument('--wise_summary', action="store_true",
                        help='use encoder function(transformer-xl) to summary the key')
    parser.add_argument('--not_weighted', action="store_true",
                        help='use not-weighted values directly as memory')
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


def train(model, train_loader, criterion, args, epoch, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.

    for batch, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        data, targets = data.t().contiguous(), targets.t().contiguous()
        model.zero_grad()

        output = model(data)

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
    model.set_batch_size(args.eval_batch_size)
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, (data, targets) in enumerate(eval_loader):
            data, targets = data.to(device), targets.to(device)
            data, targets = data.t().contiguous(), targets.t().contiguous()
                
            output = model(data)

            if args.adaptive:
                loss = criterion(output.view(-1, args.nhid), targets.view(-1))
                loss = loss.mean()
            else:
                loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))

            total_loss += loss

    model.set_batch_size(args.batch_size)

    return total_loss / len(eval_loader)



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        if args.dataset == "ptb":
            cutoffs = [20000, 40000, 80000]
            tie_projs += [True] * 3
        elif args.dataset == "wt103":
            cutoffs = [20000, 40000, 80000]
            tie_projs += [True] * 3

    if args.demo:
        args.batch_size = 1
        args.eval_batch_size = 1

    args.cutoffs = cutoffs
    args.tie_projs = tie_projs



    if args.load:
        checkpoint = torch.load(args.load)
        model_args = checkpoint["model_args"]
        model_args.demo = args.demo
        model_args.load = args.load
        model_args.adam = args.adam
        model_args.lr = args.lr
        model_args.scheduler = args.scheduler
        model_args.clip = args.clip
        model_args.epochs = args.epochs
        model_args.multi_gpu = args.multi_gpu
        model_args.save = args.save
        if args.demo:
            model_args.batch_size = 1
            model_args.eval_batch_size = 1
        args = model_args
        
    ### Load Data ###
    
    print("Loading data from %s" % args.data)
    datatime_begin = time.time()

    corpus = dataloader.Corpus(args.data)
    args.vocab_size = corpus.vocabulary.num_words
    eval_batch_size = args.eval_batch_size

    args.mem_len = args.cache_k * args.num_steps

    train_loader = corpus.get_train_loader(batch_size=args.batch_size, num_steps=args.num_steps)
    valid_loader = corpus.get_valid_loader(batch_size=eval_batch_size, num_steps=args.num_steps)
    test_loader = corpus.get_test_loader(batch_size=eval_batch_size, num_steps=args.num_steps)


    print("Data loading finished. time: {:.3f} s".format(time.time() - datatime_begin))
    if args.load:
        # clear cache
        keys = checkpoint["model_state_dict"].copy().keys()
        for key in keys:
            if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key) or re.match(r"encoder.pos_emb_bank", key):
                popitem = checkpoint["model_state_dict"].pop(key)

        if args.demo:
            model = CRTNModel(model_args, corpus=corpus)
        else:keys = checkpoint["model_state_dict"].copy().keys()
        for key in keys:
            if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key) or re.match(r"encoder.pos_emb_bank", key):
                popitem = checkpoint["model_state_dict"].pop(key)


            model = CRTNModel(model_args)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        #create model
        if args.demo:
            model = CRTNModel(args, corpus=corpus)
        else:
            model = CRTNModel(args)


    
    if args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, args.emsize, args.nhid, cutoffs, div_val=args.div_val, init_std=args.init_std) 
        if args.tied:
            for i in range(len(criterion.out_layers)):
                criterion.out_layers[i].weight = model.encoder.embedding.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and args.div_val == 1 and args.nhid != args.emsize:
                    criterion.out_projs[i] = model.encoder.embedding.emb_projs[0]
                elif tie_proj and args.div_val != 1:
                    criterion.out_projs[i] = model.encoder.embedding.emb_projs[i]
        if args.load:
            criterion.load_state_dict(checkpoint["criterion"])

    else:
        criterion = nn.CrossEntropyLoss()

        
    model.to(device)
    criterion.to(device)
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

    #writer.add_embedding(model.encoder.embedding.emb_layers[0], corpus.vocabulary.index2word)

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
            writer.add_scalar("valid/ppl", math.exp(eval_loss), epoch)
            if eval_loss < best_eval_loss:
                torch.save({
                    "model_args": model.args,
                    "model_state_dict": model.state_dict(),
                    "criterion": criterion.state_dict()
                    }, "save/" + args.save + "/" + args.save + "_best" + ".pt")
                #with open("save/" + args.save + "/" + args.save + "_best.pt", "wb") as f:
                #    torch.save(model, f)
                #with open("save/" + args.save + "/" + args.save + "_crit.pt", "wb") as f:
                #    torch.save(criterion, f)
                best_eval_loss = eval_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    #with open("save/" + args.save + "/" + args.save + "_best.pt", "rb") as f:
    #    model = torch.load(f)
    #with open("save/" + args.save + "/" + args.save + "_crit.pt", "rb") as f:
    #    criterion = torch.load(f)
    eval_checkpoint = torch.load("save/" + args.save + "/" + args.save + "_best.pt")
    model_state_dict = eval_checkpoint["model_state_dict"]
    keys = model_state_dict.copy().keys()
    for key in keys:
        if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key) or re.match(r"encoder.pos_emb_bank", key):
            model_state_dict.pop(key)
    model.load_state_dict(model_state_dict, strict=False)
    if args.adaptive:
        criterion.load_state_dict(eval_checkpoint["criterion"])
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
    
    if not os.path.exists("./log/" + args.save):
        os.mkdir("./log/" + args.save)
    if not os.path.exists("./save/" + args.save):
        os.mkdir("./save/" + args.save)
    writer = SummaryWriter("./log/" + args.save)
    main(args)
