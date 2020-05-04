import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import ipdb
import visdom
#viz = visdom.Visdom()
#assert viz.check_connection()


import os
import sys
sys.path.append("..")
sys.path.append("../..")

                    
from data import dataloader
from data.dataloader import textDataset

import torchtext
from torchtext import datasets

from models.CRTNModel import CRTNModel
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from data.dataloader import TextDataset, ExistingDataset

from baseline.pytorch.transformer import TransformerLM

from baseline.pytorch.rnn import RNNModel


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample/',
                        help='location of the data corpus')
    parser.add_argument('--datasets', type=str, choices=["fromfile", "ptb", "wt103"], 
                        default="ptb", help='load datasets from torchtext')
    parser.add_argument('--vocab_size', type=int, default=10000)
    # model
    parser.add_argument("--model_path", type=str, 
                        help="model path")
    # xl or model settings
    parser.add_argument('--num_steps', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--mem_len', type=int, default=140,
                        help='length of memory')
    parser.add_argument('--neighbor_len', type=int, default=80,
                        help='length of memory')
    parser.add_argument("--cache_N", type=int, default=20, 
                        help="size of Cache, default: 10")
    parser.add_argument("--cache_k", type=int, default=8, 
                        help="select top k values, default: 8")
    parser.add_argument("--cache_L", type=int, default=20, 
                        help="length of segments in cache, default: 20")
    # amid settings
    parser.add_argument("--sample_k", type=int, default=20, 
                        help="number of samples")
    parser.add_argument("--range", type=int, default=20, 
                        help="largest range to compute mutual information")
    parser.add_argument("--largest_range", type=int, default=200, 
                        help="largest range to load data")
    parser.add_argument("--target_len", type=int, default=1, 
                        help="target length")
    parser.add_argument("--end_bias", type=int, default=0, 
                        help="last word pos bias when loading data")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="batch size")
    # setting
    parser.add_argument("--seed", type=int, default=1111, 
                        help="random seed")
    parser.add_argument('--device', type=int, default=0,
                        help='device number')
    parser.add_argument("--debug", action="store_true",
                        help="display in debug mode")
    return parser.parse_args()

def init_cache_info(args):
    """
    store relative position of cahce chunk and its recall times
    [pos, recall times, all query times]
    """
    batch_size = args.batch_size
    pos = torch.arange(args.cache_N, 0, -1, dtype=torch.float).cuda()
    recall_query = torch.zeros((args.cache_N, 2), dtype=torch.float).cuda()
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

def get_logprob(model, criterion, texts, args, ind=None):
    model.eval()
    criterion.eval()
    assert ind is None or ind >= 0
    probs = []
    if model.name == "Transformer-XL":
        memory = None
    elif model.name == "CRTN":
        mem = None
        cache_info = init_cache_info(args)
        key, value = None, None
    with torch.no_grad():
        pos = 0
        for text in texts:
            target = torch.zeros_like(text)
            pos += text.size(0)
            if model.name == "Transformer-XL":
                output, memory = model(text, memory)
                if model.adaptive:
                    head_prob, tails = criterion(output, target, output=True)
                    prob = get_logprob_from_head_and_tails(head_prob, tails)
                else:
                    prob = F.log_softmax(output, dim=1)
            elif model.name == "CRTN":
                if mem is not None:
                    mem = mem.detach()
                output, hidden, mem = model(text, key, value,
                                            neighbor_mem=mem,
                                            cache_info=cache_info)

                model, cache_info, key, value = update_cache(model, 
                                                             text.size(1), 
                                                             key, value, 
                                                             hidden, 
                                                             text, 
                                                             cache_info)
                if model.adaptive:
                    head_prob, tails = criterion(output, target, output=True)
                    prob = get_logprob_from_head_and_tails(head_prob, tails)
                else:
                    prob = F.log_softmax(output, dim=1)

            prob = prob.reshape(*text.size(), -1)
            probs.append(prob)

            if ind is not None:
                if pos > ind:
                    break

        prob = torch.cat(probs, dim=0)
    if ind is None:
        return prob
    else:
        return prob[ind]

def get_logprob_from_head_and_tails(head_prob, tails):
    tail_len = len(tails)
    get_tails = []
    for i in range(tail_len):
        base_prob = head_prob[:,-i-1].unsqueeze(1)
        get_tails.append(tails[i] + base_prob)
    real_prob = torch.cat((head_prob[:,:-tail_len], *get_tails), 1)
    return real_prob



def mutual_information(model, criterion, texts, distance, args):
    text_len = texts.size(0)
    ind_i, ind_j = text_len - distance - 1, text_len - 1
    text_blocks = texts.split(args.num_steps, dim=0)
    # compute to distribution of Yi to sample from
    sample_prob = get_logprob(model, criterion, text_blocks, args, ind=ind_i).exp()
    # sample k times
    yi_samples = torch.multinomial(sample_prob, args.sample_k, replacement=True)
    # form new condition given samples
    sample_texts = texts.expand(args.sample_k, -1, -1).contiguous()
    sample_texts[:,ind_i,:] = yi_samples.t()
    # compute sample conditioned prob
    yj_probs = []
    oa_probs = []
    for texts in sample_texts:
        text_blocks = texts.split(args.num_steps, dim=0)
        all_logprob = get_logprob(model, criterion, text_blocks, args)
        yj_logprob = all_logprob[ind_j]
        oa_all_logprob = all_logprob[ind_i+1:ind_j]
        oa_text = texts[ind_i+1:ind_j]
        oa_logprob = torch.gather(oa_all_logprob, dim=2, index=oa_text.unsqueeze(-1))
        oa_logprob = oa_logprob.sum(dim=0)

        yj_probs.append(yj_logprob.unsqueeze(0))
        oa_probs.append(oa_logprob)

    logoa_prob = torch.cat(oa_probs, dim=1).t()
    wk = F.softmax(logoa_prob, dim=0)
    logp_yj = torch.cat(yj_probs, dim=0)

    # compute mutual information I(Yi, Yj)
    weighted_pyj = (wk.unsqueeze(-1) * logp_yj.exp()).sum(dim=0)
    H_yj = -(weighted_pyj * weighted_pyj.log()).sum(dim=1)

    meta_hyj_givenyi = -(logp_yj.exp() * logp_yj).sum(dim=2)
    H_yj_givenyi = (wk * meta_hyj_givenyi).sum(dim=0)

    mi = H_yj - H_yj_givenyi
    #if mi < 0 or torch.isnan(mi):
    #    ipdb.set_trace()

    return mi.mean().item()


def averaged_split(mutual_infos):
    mi_sum = sum(mutual_infos)
    if mi_sum > 0:
        k = 0
        temp_sum = 0
        while True:
            if temp_sum + mutual_infos[k] < mi_sum / 2:
                temp_sum = temp_sum + mutual_infos[k]
                k += 1
            else:
                break
        amid = k + (mi_sum / 2 - temp_sum) / mutual_infos[k+1]
    else:
        amid = 0

    return amid

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")
    torch.cuda.set_device(device)

    ### Load Data ###

    if args.largest_range < args.range:
        args.largest_range = args.range
    
    if args.datasets == "ptb":
        print("Loading %s dataset from torchtext" % args.datasets)
        corpus = ExistingDataset("ptb", args.num_steps)
    elif args.datasets == "wt103":
        print("Loading %s dataset from torchtext" % args.datasets)
        corpus = ExistingDataset("wt103", args.num_steps)
    elif args.datasets == "fromfile":
        print("Loading data from %s" % args.data)
        corpus = TextDataset(args.data, args.vocab_size, args.num_steps)
    data_loader = corpus.recl_loader(args.batch_size, args.target_len, args.largest_range, end_bias=args.end_bias)

    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint["model_args"]
    name = model_args.name

    # redefine hyperparams

    if not model_args.num_steps == args.num_steps:
        print("REDEFINE num_steps: {} --> {}".format(model_args.num_steps, 
                                                     args.num_steps))
        model_args.num_steps = args.num_steps
    if hasattr(model_args, "mem_len"):
        if not model_args.mem_len == args.mem_len:
            print("REDEFINE mem_len: {} --> {}".format(model_args.mem_len, 
                                                       args.mem_len))
            model_args.mem_len = args.mem_len
    if hasattr(model_args, "neighbor_len"):
        if not model_args.neighbor_len == args.neighbor_len:
            print("REDEFINE neighbor_len: {} --> {}".format(model_args.neighbor_len, 
                                                            args.neighbor_len))
            model_args.neighbor_len = args.neighbor_len
    if hasattr(model_args, "cache_N"):
        if not model_args.cache_N == args.cache_N:
            print("REDEFINE cache_N: {} --> {}".format(model_args.cache_N, 
                                                       args.cache_N))
            model_args.cache_N = args.cache_N
    if hasattr(model_args, "cache_k"):
        if not model_args.cache_k == args.cache_k:
            print("REDEFINE cache_k: {} --> {}".format(model_args.cache_k, 
                                                       args.cache_k))
            model_args.cache_k = args.cache_k
    if hasattr(model_args, "cache_L"):
        if not model_args.cache_L == args.cache_L:
            print("REDEFINE cache_L: {} --> {}".format(model_args.cache_L, 
                                                       args.cache_L))
            model_args.cache_L = args.cache_L

    model_args.device = args.device
    model_args.batch_size = args.batch_size


    if name == 'LSTM':
        model = RNNModel(model_args)
        criterion = nn.CrossEntropyLoss()
    elif name == 'Transformer-XL':
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

    if model_args.adaptive:
        model.adaptive = model_args.adaptive
        criterion = ProjectedAdaptiveLogSoftmax(model_args.vocab_size, 
                                                model_args.emsize, 
                                                model_args.nhid, 
                                                model_args.cutoffs, 
                                                div_val=model_args.div_val, 
                                                init_std=model_args.init_std,
                                                proj_init_std=model_args.proj_init_std,
                                                mos=model_args.mos,
                                                n_experts=model_args.n_experts,
                                                dropmos=model_args.dropmos
                                                ) 
        criterion.load_state_dict(checkpoint["criterion"])
    else:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    if args.debug:
        vocab = data_loader.dataset.fields["text"].vocab
    
    mutual_infos = [0 for _ in range(args.range)]
    with tqdm(total=args.target_len) as pbar:
        for data in data_loader:
            texts, targets = data.text.to(device), data.target.to(device)

            if args.debug:
                for i, text in enumerate(texts.t()):
                    words = [vocab.itos[w] for w in text]
                    print("Batch %s: " % i, " ".join(words))

            for dis in range(1, args.range + 1):
                mutual_info = mutual_information(model, criterion, texts, dis, args)
                mutual_infos[dis-1] += mutual_info
                if args.debug:
                    print("dis %s mi %.3e" % (dis, mutual_info))
            pbar.update(1)

    amid = averaged_split(mutual_infos)
    print("-" * 89)
    print("Averaged Mutual Information Distance of {}: {:.3f}".format(model.name, amid))
    print("-" * 89)




if __name__ == "__main__":
    args = parse_args()
    main(args)