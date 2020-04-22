import time
from datetime import datetime
import os
import sys
import argparse
from tqdm import tqdm
from itertools import chain
import pickle as pkl

sys.path.append("../..")
sys.path.append("../../CRTN/")

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from torch.utils.data import DataLoader

from CRTN.data.dataloader import TextDataset, ExistingDataset
from CRTN.utils.adaptive import ProjectedAdaptiveLogSoftmax
from CRTN.utils.visual import TargetText
from transformer import TransformerLM

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample/',
                        help='location of the data corpus')
    parser.add_argument('--datasets', type=str, choices=["fromfile", "ptb", "wt103"], 
                        default="fromfile", help='load datasets from torchtext')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of vocabulary, excluding special chars')
    # optimization
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='initial learning rate')
    # hyperparams
    parser.add_argument('--num_steps', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--mem_len', type=int, default=70,
                        help='length of memory')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='batch size for gradient statistics')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='batch size for evaluation')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    # dynamic setting
    parser.add_argument('--lamb', type=float, default=0.02,
                        help='decay parameter lambda')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='stabilization parameter epsilon')
    # eval setting
    parser.add_argument('--device', type=int, default=0,
                        help='device number')
    parser.add_argument('--eval_temperature', type=float, default=1.0, 
                        help='eval temperature, divide logits.')
    parser.add_argument('--eval_temp_search', action="store_true",
                        help='search best temperature on valid set during test. 1-1.2/0.02')
    parser.add_argument('--eval_theta_search', action="store_true",
                        help='search best theta on valid set during test. 0.7-1/0.02')
    # setting
    parser.add_argument("--theta", type=float, default=1.0, 
                        help="attention theta, default: 1.0")
    parser.add_argument('--word_loss', action="store_true",
                        help='output loss of every word')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    return parser.parse_args()

def gradstat(model, train_loader, criterion, args):

    model.train()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    total_loss = 0.
    total_len = len(train_loader)
    memory = None

    for param in chain(model.parameters(), criterion.parameters()):
        param.MS = torch.zeros_like(param.data)

    with tqdm(total=total_len) as pbar:
        for i, data in enumerate(train_loader):

            text, targets = data.text.to(device), data.target.to(device)

            model.zero_grad()

            output, memory = model(text, memory)

            if args.adaptive:
                loss_tensor = criterion(output, targets,
                                        temperature=args.eval_temperature)
                loss = loss_tensor.sum()
            else:
                loss = criterion(output.view(-1, args.vocab_size), 
                                             targets.view(-1))

            loss.backward()
            total_loss += loss.item()

            for param in chain(model.parameters(), criterion.parameters()):
                param.MS = param.MS + param.grad.data.pow(2)

            pbar.update(1)
    
    gsum = 0
    for param in chain(model.parameters(), criterion.parameters()):
        param.MS = torch.sqrt(param.MS)
        gsum += torch.mean(param.MS)

    for param in chain(model.parameters(), criterion.parameters()):
        param.decrate = param.MS / gsum

def evaluate(model, eval_loader, criterion, args):

    model.train()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    total_loss = 0.
    len_eval = 0
    total_len = len(eval_loader)
    memory = None

    model_prm, crit_prm = dict(), dict()
    for prm in model.parameters():
        model_prm[prm] = prm.data.clone()
    for prm in criterion.parameters():
        crit_prm[prm] = prm.data.clone()

    for param in chain(model.parameters(), criterion.parameters()):
        param.decrate = param.decrate.clamp(max=1/args.lamb)
        param.data0 = param.data.clone()

    with tqdm(total=total_len) as pbar:
        for i, data in enumerate(eval_loader):
                           
            text, targets = data.text.to(device), data.target.to(device)
            len_eval += targets.view(-1).size(0)

            model.zero_grad()

            output, memory = model(text, memory)

            if args.adaptive:
                loss_tensor = criterion(output, targets,
                                        temperature=args.eval_temperature)
                loss = loss_tensor.sum()
            else:
                loss = criterion(output.view(-1, args.vocab_size), 
                                             targets.view(-1))
            loss.backward()
            total_loss += loss.item()

            for param in chain(model.parameters(), criterion.parameters()):
                dW = args.lamb * param.decrate * (param.data0 - param.data) - \
                     args.lr * param.grad.data / (param.MS + args.epsilon)
                param.data += dW

            pbar.update(1)

    ppl = math.exp(total_loss / len_eval)

    for prm in model.parameters():
        prm.data.copy_(model_prm[prm])
    for prm in criterion.parameters():
        prm.data.copy_(crit_prm[prm])

    return ppl



def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")
    torch.cuda.set_device(device)

    ### Load Data ###
 
    datatime_begin = time.time()
    
    if args.datasets == "ptb":
        print("Loading %s dataset from torchtext" % args.datasets)
        train_loader, _, _ = torchtext.datasets.PennTreebank.iters(
                batch_size=args.batch_size, 
                device=torch.device("cpu"),
                bptt_len=args.num_steps)
        _, valid_loader, test_loader = torchtext.datasets.PennTreebank.iters(
                batch_size=args.eval_batch_size, 
                device=torch.device("cpu"),
                bptt_len=args.num_steps)
        vocab = train_loader.dataset.fields["text"].vocab
        args.vocab_size = len(vocab.itos)
    elif args.datasets == "wt103":
        print("Loading %s dataset from torchtext" % args.datasets)
        train_loader, _, _ = torchtext.datasets.WikiText103.iters(
                batch_size=args.batch_size, 
                device=torch.device("cpu"),
                bptt_len=args.num_steps)
        _, valid_loader, test_loader = torchtext.datasets.WikiText103.iters(
                batch_size=args.eval_batch_size, 
                device=torch.device("cpu"),
                bptt_len=args.num_steps)
        vocab = train_loader.dataset.fields["text"].vocab
        args.vocab_size = len(vocab.itos)
    elif args.datasets == "fromfile":
        print("Loading data from %s" % args.data)
        corpus = TextDataset(args.data, args.vocab_size, args.num_steps)
        args.vocab_size = len(corpus.TEXT.vocab.itos)

        train_loader = corpus.get_train_loader(args.batch_size)
        valid_loader = corpus.get_valid_loader(args.eval_batch_size)
        test_loader = corpus.get_test_loader(args.eval_batch_size)

    datatime_end = time.time()

    ### Load model ###

    checkpoint = torch.load(args.load, map_location=device)
    model_args = checkpoint["model_args"]

    model_args.load = args.load
    model_args.lr = args.lr
    model_args.device = args.device

    model_args.lamb = args.lamb
    model_args.epsilon = args.epsilon
    
    if not model_args.num_steps == args.num_steps:
        print("REDEFINE num_steps: {} --> {}".format(model_args.num_steps, 
                                                     args.num_steps))
        model_args.num_steps = args.num_steps
    if not model_args.mem_len == args.mem_len:
        print("REDEFINE mem_len: {} --> {}".format(model_args.mem_len, 
                                                   args.mem_len))
        model_args.mem_len = args.mem_len
    if not model_args.clamp_len == args.clamp_len:
        print("REDEFINE clamp_len: {} --> {}".format(model_args.clamp_len, 
                                                     args.clamp_len))
        model_args.clamp_len = args.clamp_len
    if not model_args.theta == args.theta:
        print("REDEFINE theta: {} --> {}".format(model_args.theta, 
                                                 args.theta))
        model_args.theta = args.theta
    model_args.same_length = args.same_length

    model_args.batch_size = args.batch_size
    model_args.eval_batch_size = args.eval_batch_size
    model_args.eval_temperature = args.eval_temperature
    model_args.eval_temp_search = args.eval_temp_search
    model_args.eval_theta_search = args.eval_theta_search

    args = model_args

    #Print Params
    for argk, argv in args.__dict__.items():
        print("{}: {}".format(argk, argv))
    print("")
    print("Data loading finished. time: {:.3f} s".format(datatime_end-datatime_begin))

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
            dropout=0,
            dropatt=0,
            dropemb=0,
            dropinp=0,
            dropwei=0,
            dropfor=0,
            drophid=0,
            theta=model_args.theta,
            theta_alpha=model_args.theta_annealing_alpha,
            apex=model_args.apex,
            no_pos_bias=model_args.no_pos_bias
            )

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

    gradstat(model, train_loader, criterion, args)

    if args.eval_temp_search:
        best_temp_ppl = float("inf")
        best_temp = 1.0
        print("temperature search")
        for temp in np.arange(1.0, 1.2, 0.02):
            args.eval_temperature = temp
            temp_ppl = evaluate(model, valid_loader, criterion, args)
            if temp_ppl < best_temp_ppl:
                best_temp_ppl = temp_ppl
                best_temp = temp
                print("UPDATE best temp {:5.2f} | valid ppl {:8.2f}".format(temp, temp_ppl))
            else:
                break
        args.eval_temperature = best_temp

    if args.eval_theta_search:
        best_theta_ppl = float("inf")
        best_theta = 1.0
        print("theta search")
        for theta in np.arange(1.0, 0.7, -0.02):
            model.theta = theta
            theta_ppl = evaluate(model, valid_loader, criterion, args)
            if theta_ppl < best_theta_ppl:
                best_theta_ppl = theta_ppl
                best_theta = theta
                print("UPDATE best theta {:5.2f} | valid ppl {:8.2f}".format(theta, theta_ppl))
            else:
                break
        model.theta = best_theta

    best_eval_ppl = evaluate(model, valid_loader, criterion, args)

    test_ppl = evaluate(model, test_loader, criterion, args)

    print('=' * 89)
    print('| End of training | best valid ppl {:8.2f}'.format(best_eval_ppl))
    print('=' * 89)
    print('| test ppl {:8.2f}'.format(test_ppl))
    print('=' * 89)





if __name__ == "__main__":
    args = parse_args()
    main(args)



