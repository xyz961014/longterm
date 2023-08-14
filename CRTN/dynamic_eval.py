import time
from datetime import datetime
import os
import sys
import argparse
from tqdm import tqdm
from itertools import chain
import pickle as pkl

sys.path.append("..")
sys.path.append("../baseline")
sys.path.append("../baseline/pytorch")

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from CRTN.data.dataloader import TextDataset, ExistingDataset
from CRTN.utils.adaptive import ProjectedAdaptiveLogSoftmax
from models.CRTNModel import CRTNModel

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample/',
                        help='location of the data corpus')
    parser.add_argument('--datasets', type=str, choices=["fromfile", "ptb", "wt2", "wt103"], 
                        default="ptb", help='load datasets from torchtext')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of vocabulary, excluding special chars')
    # optimization
    parser.add_argument('--lr', type=float, default=0.002,
                        help='initial learning rate')
    # hyperparams
    parser.add_argument('--num_steps', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--neighbor_len', type=int, default=70,
                        help='length of memory')
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_k", type=int, default=2, 
                        help="select top k values, default: 2")
    parser.add_argument("--cache_L", type=int, default=20, 
                        help="length of segments in cache, default: 20")
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
    parser.add_argument("--cache_theta", type=float, default=1.0, 
                        help="cache query theta, default: 1.0")
    parser.add_argument("--attn_theta", type=float, default=1.0, 
                        help="attention theta, default: 1.0")
    parser.add_argument('--word_loss', action="store_true",
                        help='output loss of every word')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    return parser.parse_args()

def init_cache_info(args, device, evaluate=False):
    """
    store relative position of cahce chunk and its recall times
    [pos, recall times, all query times]
    """
    batch_size = args.eval_batch_size if evaluate else args.batch_size
    pos = torch.arange(args.cache_N, 0, -1, 
                       dtype=torch.float,
                       device=device)
    recall_query = torch.zeros((args.cache_N, 2), dtype=torch.float, device=device)
    cache_info = torch.cat((pos.unsqueeze(-1), recall_query), dim=-1).unsqueeze(1)
    cache_info = cache_info.expand(-1, batch_size, -1).contiguous()
    return cache_info

def update_cache(model, batch_size, key, value, hidden, text, cache_info):
    
    keys, values, cache_info = model.cache.renew(hidden, 
                                                 text, 
                                                 cache_info, 
                                                 keys=key, 
                                                 values=value)
    return model, cache_info, keys, values


def gradstat(model, train_loader, criterion, args):

    model.train()
    criterion.train()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    total_loss = 0.
    total_len = len(train_loader)
    key, value = None, None
    if args.farnear:
        mem = None
    cache_info = init_cache_info(args, device, True)

    for param in chain(model.parameters(), criterion.parameters()):
        param.MS = torch.zeros_like(param.data)

    with tqdm(total=total_len) as pbar:
        for i, data in enumerate(train_loader):

            text, targets = data.text.to(device), data.target.to(device)

            model.zero_grad()
            criterion.zero_grad()

            if args.farnear:
                if mem is not None:
                    mem = mem.detach()
                output, hidden, mem = model(text, key, value, 
                                            neighbor_mem=mem, 
                                            cache_info=cache_info)
            else:
                output, hidden = model(text, key, value, cache_info=cache_info)

            model, cache_info, key, value = update_cache(model, 
                                                          text.size(1), 
                                                          key, value, 
                                                          hidden, 
                                                          text, 
                                                          cache_info)

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
    criterion.train()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    total_loss = 0.
    len_eval = 0
    total_len = len(eval_loader)
    key, value = None, None
    if args.farnear:
        mem = None
    cache_info = init_cache_info(args, device, True)


    for param in chain(model.parameters(), criterion.parameters()):
        param.decrate = param.decrate.clamp(max=1/args.lamb)
        param.data0 = param.data.clone()

    with tqdm(total=total_len) as pbar:
        for i, data in enumerate(eval_loader):
                           
            text, targets = data.text.to(device), data.target.to(device)
            len_eval += targets.view(-1).size(0)

            eval_batch_size = text.size(1)
            model.set_batch_size(eval_batch_size)

            model.zero_grad()
            criterion.zero_grad()

            if args.farnear:
                if mem is not None:
                    mem = mem.detach()
                output, hidden, mem = model(text, key, value, 
                                            neighbor_mem=mem, 
                                            cache_info=cache_info)
            else:
                output, hidden = model(text, key, value, cache_info=cache_info)

            model, cache_info, key, value = update_cache(model, 
                                                          text.size(1), 
                                                          key, value, 
                                                          hidden, 
                                                          text, 
                                                          cache_info)

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

    for param in chain(model.parameters(), criterion.parameters()):
        param.data.copy_(param.data0)

    model.set_batch_size(args.batch_size)
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
    elif args.datasets == "wt2":
        print("Loading %s dataset from torchtext" % args.datasets)
        train_loader, _, _ = torchtext.datasets.WikiText2.iters(
                batch_size=args.batch_size, 
                device=torch.device("cpu"),
                bptt_len=args.num_steps)
        _, valid_loader, test_loader = torchtext.datasets.WikiText2.iters(
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

    assert args.num_steps >= args.cache_L, "cache_L should <= num_steps"

    checkpoint = torch.load(args.load, map_location=device)
    model_args = checkpoint["model_args"]

    model_args.load = args.load
    model_args.lr = args.lr
    model_args.device = args.device

    model_args.lamb = args.lamb
    model_args.epsilon = args.epsilon

    # set dropout to 0
    model_args.dropout = 0
    model_args.dropatt = 0
    model_args.dropinp = 0
    model_args.dropemb = 0
    model_args.dropwei = 0
    model_args.dropfor = 0
    model_args.dropmos = 0
    model_args.drophid = 0
    

    if not model_args.num_steps == args.num_steps:
        print("REDEFINE num_steps: {} --> {}".format(model_args.num_steps, 
                                                     args.num_steps))
        model_args.num_steps = args.num_steps
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
    if not model_args.clamp_len == args.clamp_len:
        print("REDEFINE clamp_len: {} --> {}".format(model_args.clamp_len, 
                                                     args.clamp_len))
        model_args.clamp_len = args.clamp_len
    if hasattr(model_args, "cache_theta"):
        if not model_args.cache_theta == args.cache_theta:
            print("REDEFINE cache_theta: {} --> {}".format(model_args.cache_theta, 
                                                     args.cache_theta))
            model_args.cache_theta = args.cache_theta
    else:
        model_args.cache_theta = args.cache_theta
    if hasattr(model_args, "attn_theta"):
        if not model_args.attn_theta == args.attn_theta:
            print("REDEFINE attn_theta: {} --> {}".format(model_args.attn_theta, 
                                                     args.attn_theta))
            model_args.attn_theta = args.attn_theta
    else:
        model_args.attn_theta = args.attn_theta


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

    model = CRTNModel(args)
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
        best_atheta_ppl = float("inf")
        best_atheta = 1.0
        print("attn theta search")
        for atheta in np.arange(1.0, 0.7, -0.02):
            model.set_theta(1.0, atheta)
            atheta_ppl = evaluate(model, valid_loader, criterion, args)
            if atheta_ppl < best_atheta_ppl:
                best_atheta_ppl = atheta_ppl
                best_atheta = atheta
                print("UPDATE best attn theta {:5.2f} | valid ppl {:8.2f}".format(atheta, atheta_ppl))
            else:
                break
        model.set_theta(1.0, best_atheta)

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



