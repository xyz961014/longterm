import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import ipdb
import visdom
import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append("..")
sys.path.append("../../awd-lstm-lm")

                    
from data import dataloader
from data.dataloader import textDataset

import torchtext
from torchtext import datasets

from models.CRTNModel import CRTNModel
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from data.dataloader import TextDataset, ExistingDataset
from utils.utils import repackage_hidden

from baseline.pytorch.transformer import TransformerLM

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample/',
                        help='location of the data corpus')
    parser.add_argument('--datasets', type=str, choices=["fromfile", "ptb", "wt2", "wt103"], 
                        default="ptb", help='load datasets from torchtext')
    parser.add_argument('--vocab_size', type=int, default=10000)
    # models
    parser.add_argument("--model_path", nargs="+", type=str, 
                        help="model paths")
    # xl and model settings
    parser.add_argument('--num_steps', type=int, default=80,
                        help='sequence length')
    parser.add_argument('--mem_len', type=int, default=80,
                        help='length of memory')
    parser.add_argument('--neighbor_len', type=int, default=80,
                        help='length of memory')
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_k", type=int, default=2, 
                        help="select top k values, default: 2")
    parser.add_argument("--cache_L", type=int, default=80, 
                        help="length of segments in cache, default: 80")
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument('--same_length_query', action='store_true',
                        help='use the same attn length for all tokens in query')
    # recl settings
    parser.add_argument("--target_len", type=int, default=30, 
                        help="target length")
    parser.add_argument("--end_bias", type=int, default=0,
                        help="last word pos bias when loading data")
    parser.add_argument("--init_c", type=int, default=20, 
                        help="initial c")
    parser.add_argument("--top_r", type=float, default=1.0, 
                        help="ratio r of worst words to attention")
    parser.add_argument("--delta", type=int, default=20, 
                        help="step of increasing c")
    parser.add_argument("--threshold", type=float, default=0.001, 
                        help="addition ratio threshold to stop")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="batch size")
    # select long-term words
    parser.add_argument("--select_long", action="store_true",
                        help="switch of select long-term words")
    parser.add_argument("--context_len", type=int, default=100, 
                        help="fixed context length to compare with")
    parser.add_argument("--compared_len", type=int, default=-1, 
                        help="long context length to compare with, -1 to disable")
    parser.add_argument("--parallel_num", type=int, default=1, 
                        help="computing loss in parallel")
    parser.add_argument("--select_ratio", type=float, default=0.05, 
                        help="ratio r to select long-term words")
    parser.add_argument("--long_save", type=str, default=".data/longterm_idx.txt",
                        help="path to save selected long-term word indexes")
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

def evaluate(model, criterion, data_loader, args):
    model.eval()
    criterion.eval()
    losses = []
    if args.debug:
        texts = []
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            pbar.set_description(model.name)

            # initial model
            if model.name == "AWD-LSTM":
                hidden = model.init_hidden(args.batch_size)
            elif model.name == "Transformer-XL":
                memory = None
            elif model.name == "CRTN":
                mem = None
                cache_info = init_cache_info(args)
                key, value = None, None

            for data in data_loader:
                text, target = data.text.cuda(), data.target.cuda()
                if args.debug:
                    texts.append(text)

                if model.name == "AWD-LSTM":
                    target = target.view(-1)
                    hidden = repackage_hidden(hidden)
                    output, hidden = model(text, hidden)
                    loss_tensor = criterion(model.decoder.weight, model.decoder.bias, output, target, output=True)
                elif model.name == "Transformer-XL":
                    output, memory = model(text, memory)
                    if model.adaptive:
                        loss_tensor = criterion(output, target,
                                                keep_order=True)
                    else:
                        loss_tensor = criterion(output, target, reduction='none')
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
                        loss_tensor = criterion(output, target,
                                                keep_order=True)
                    else:
                        loss_tensor = criterion(output, target, reduction='none')
                loss_tensor = loss_tensor.reshape_as(target)
                losses.append(loss_tensor)
                pbar.update(1)
    loss = torch.cat(losses, dim=0)
    e = args.end_bias
    l = args.target_len
    selected_loss = loss[-1 - l - e:-1 - e]

    if args.debug:
        vocab = data_loader.dataset.fields["text"].vocab
        texts = torch.cat(texts, dim=0)
        selected_text = texts[-1-l-e:-1-e,0]
        print("batch_len: %s" % loss.size(0))
        print("infinite context Batch 0:")
        for i, w in enumerate(selected_text):
            print("%s|%.4f" % (vocab.itos[w], selected_loss[i, 0]), end=" ")
        print("")

    return selected_loss, loss.size(0) - 1 - l - e



def loss(model, criterion, data_loader, args):
    model.eval()
    criterion.eval()
    losses = []
    p_texts, p_targets = [], []
    if args.debug:
        txts = []
    with torch.no_grad():
        with tqdm(total=args.target_len) as pbar:
            pbar.set_description(model.name)
            for idata, data in enumerate(data_loader):
                text, target = data.text.cuda(), data.target.cuda()
                if args.debug:
                    txts.append(text[-1][0].item())
                p_texts.append(text)
                p_targets.append(target)
                if len(p_texts) < args.parallel_num and not idata + 1 >= args.target_len:
                    continue
                parallel_num = len(p_texts)
                texts = torch.cat(p_texts, dim=1)
                targets = torch.cat(p_targets, dim=1)
                texts = texts.split(args.num_steps, dim=0)
                targets = targets.split(args.num_steps, dim=0)
                p_texts, p_targets = [], []


                # initial model
                if model.name == "AWD-LSTM":
                    hidden = model.init_hidden(args.batch_size)
                elif model.name == "Transformer-XL":
                    memory = None
                elif model.name == "CRTN":
                    mem = None
                    cache_info = init_cache_info(args)
                    key, value = None, None


                for text, target in list(zip(texts, targets)):

                    if model.name == "AWD-LSTM":
                        hidden = repackage_hidden(hidden)
                        output, hidden = model(text, hidden)
                        loss_tensor = criterion(model.decoder.weight, model.decoder.bias, output, target, output=True)
                    elif model.name == "Transformer-XL":
                        output, memory = model(text, memory)
                        if model.adaptive:
                            loss_tensor = criterion(output, target,
                                                    keep_order=True)
                        else:
                            loss_tensor = criterion(output, target, reduction='none')
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
                            loss_tensor = criterion(output, target,
                                                    keep_order=True)
                        else:
                            loss_tensor = criterion(output, target, reduction='none')
                    loss_tensor = loss_tensor.reshape_as(target)
                losses = losses + list(loss_tensor[-1].unsqueeze(0).split(args.batch_size, dim=1))
                pbar.update(parallel_num)

    loss = torch.cat(losses[::-1], dim=0)
    if args.debug:
        vocab = data_loader.dataset.fields["text"].vocab
        txts = txts[::-1]
        print("fixed context %s Batch 0:" % data_loader.context_len)
        for i, w in enumerate(txts):
            print("%s|%.4f" % (vocab.itos[w], loss[i, 0]), end=" ")
        print("")
        
    return loss



def relative_loss(model_loss, base_loss, tau):
    base_tau, pos_tau = base_loss.topk(tau, dim=0)
    model_tau = model_loss.index_select(0, pos_tau)
    cat_tau = torch.cat((base_tau.unsqueeze(0), model_tau.unsqueeze(0)), dim=0)
    min_tau = cat_tau.min(0)[0]
    return min_tau.mean().item()

def relative_gain(model_loss, base_loss, r):
    model_loss = model_loss.reshape(-1)
    base_loss = base_loss.reshape(-1)
    tau = math.ceil(r * base_loss.size(0))

    f_base = relative_loss(base_loss, base_loss, tau)
    f_prime = relative_loss(model_loss, base_loss, tau)
    return (f_base - f_prime) / f_base

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    models = []

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")
    torch.cuda.set_device(device)

    ### Load Data ###
    
    if args.datasets == "ptb":
        print("Loading %s dataset from torchtext" % args.datasets)
        corpus = ExistingDataset("ptb", args.num_steps)
    elif args.datasets == "wt2":
        print("Loading %s dataset from torchtext" % args.datasets)
        corpus = ExistingDataset("wt2", args.num_steps)
    elif args.datasets == "wt103":
        print("Loading %s dataset from torchtext" % args.datasets)
        corpus = ExistingDataset("wt103", args.num_steps)
    elif args.datasets == "fromfile":
        print("Loading data from %s" % args.data)
        corpus = TextDataset(args.data, args.vocab_size, args.num_steps)



    for path in args.model_path:
        checkpoint = torch.load(path, map_location=device)
        if type(checkpoint) == list:
            # model from AWD-LSTM
            name = "AWD-LSTM"
        else:
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
            if hasattr(model_args, "same_length"):
                if not model_args.same_length == args.same_length:
                    print("REDEFINE same_length: {} --> {}".format(model_args.same_length, 
                                                               args.same_length))
                    model_args.same_length = args.same_length
            if hasattr(model_args, "same_length_query"):
                if not model_args.same_length_query == args.same_length_query:
                    print("REDEFINE same_length_query: {} --> {}".format(model_args.same_length_query, 
                                                               args.same_length_query))
                    model_args.same_length_query = args.same_length_query

            model_args.device = args.device
            model_args.batch_size = args.batch_size
            if not hasattr(model_args, "d_head"):
                model_args.d_head = model_args.nhid // model_args.nhead


        if name == 'AWD-LSTM':
            model, criterion, _ = checkpoint
        elif name == 'Transformer-XL':
            model = TransformerLM(
                    vocab_size=model_args.vocab_size,
                    num_layer=model_args.nlayers,
                    num_head=model_args.nhead,
                    d_model=model_args.nhid,
                    d_head=model_args.d_head,
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

        if not name == "AWD-LSTM":
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

        model.cuda()
        criterion.cuda()

        models.append((model, criterion))

    if args.select_long:
        assert len(models) == 1, "plz only use one model to select long-term words"
        assert args.batch_size == 1, "plz set batch_size = 1"
        model, criterion = models[0]
        if args.compared_len == -1:
            valid_loader = corpus.get_valid_loader(args.batch_size)
        else:
            valid_loader = corpus.recl_loader(args.batch_size, args.target_len, args.compared_len, 
                                              end_bias=args.end_bias)
        fixed_loader = corpus.recl_loader(args.batch_size, args.target_len, args.context_len, end_bias=args.end_bias)
        text = valid_loader.dataset.examples[0].text

        if args.compared_len == -1:
            valid_loss, start_idx = evaluate(model, criterion, valid_loader, args)
        else:
            valid_loss = loss(model, criterion, valid_loader, args) 
            start_idx = len(text) - 2 - args.end_bias - args.target_len
        fixed_loss = loss(model, criterion, fixed_loader, args)
        min_loss = torch.cat((valid_loss.unsqueeze(0), fixed_loss.unsqueeze(0)), dim=0).min(dim=0)[0]
        ppl_gain = (fixed_loss.exp() - min_loss.exp()) / fixed_loss.exp()


        tau = math.ceil(args.select_ratio * ppl_gain.size(0))
        selected_gain, idx = ppl_gain.topk(tau, dim=0, sorted=False)

        long_index = []
        batch_idx = idx[:, 0]
        gain = ppl_gain[:, 0]
        target_text = text[start_idx: start_idx + args.target_len]
        for i, w in enumerate(target_text):
            if i in batch_idx:
                long_index.append(start_idx + i)
                if args.debug:
                    print("\033[1;31m %s\033[0m|%.4f" % (w, gain[i].item()), end=" ")
            else:
                if args.debug:
                    print(w, end=" ")
        print("\n" + "-" * 80)
        compared_len = float("inf") if args.compared_len == -1 else args.compared_len
        print("WHOLE TEXT long-term %s PPL: %.2f | fixed context %s PPL: %.2f" % (compared_len, 
                                                                                  valid_loss.mean().exp(), 
                                                                                  args.context_len, 
                                                                                  fixed_loss.mean().exp()))
        valid_selected_loss = valid_loss.squeeze().index_select(0, idx.squeeze())
        fixed_selected_loss = fixed_loss.squeeze().index_select(0, idx.squeeze())
        print("SELECTED WORDS INFO: {}/{}".format(idx.size(0), args.target_len))
        print("MIN/MEAN/MAX gain: {:.4f}/{:.4f}/{:.4f}".format(selected_gain.min(), selected_gain.mean(), selected_gain.max()))
        print("SELECTED WORDS long-term PPL: %.2f | fixed context %s PPL: %.2f" % (valid_selected_loss.mean().exp(), args.context_len, fixed_selected_loss.mean().exp()))
        with open(args.long_save, "w") as f:
            f.write(" ".join([str(p) for p in long_index]))

    else:
        c = args.init_c
        r = args.top_r
        delta = args.delta
        threshold = args.threshold
        gain_stops = [False for model in models]

        c_prime = c
        base_loss = None
        while False in gain_stops:

            c = c_prime
            c_prime = c + delta
            print("c: {}\tc': {}".format(c, c_prime))

            if base_loss is None:
                base_loader = corpus.recl_loader(args.batch_size, args.target_len, c, end_bias=args.end_bias)
                base_losses = []
                for model, criterion in models:
                    model_loss = loss(model, criterion, base_loader, args)
                    base_losses.append(model_loss.unsqueeze(0))
                base_loss = torch.cat(base_losses, dim=0)
                base_loss = base_loss.min(0)[0]
            print("base loss: {:.3f}".format(base_loss.mean()))

            prime_loader = corpus.recl_loader(args.batch_size, args.target_len, c_prime, end_bias=args.end_bias)
            prime_losses = [base_loss.unsqueeze(0)]
            for idx in range(len(gain_stops)):
                model, criterion = models[idx]
                gain_stop = gain_stops[idx]
                model_loss = loss(model, criterion, prime_loader, args)
                prime_losses.append(model_loss.unsqueeze(0))
                if not gain_stop:
                    gain = relative_gain(model_loss, base_loss, r)
                    print("{} loss:{:.3f} gain: {:.4f}".format(model.name, model_loss.mean(), gain))
                    if gain < threshold:
                        gain_stops[idx] = True
                        print("{} RECL: {}".format(model.name, c))


            base_loss = torch.cat(prime_losses, dim=0)
            base_loss = base_loss.min(0)[0]

if __name__ == "__main__":
    args = parse_args()
    main(args)
