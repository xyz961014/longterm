import argparse
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import re
sys.path.append("..")
sys.path.append("../baseline/pytorch")
from data import dataloader
import visdom
viz = visdom.Visdom()
assert viz.check_connection()

from utils.adaptive import ProjectedAdaptiveLogSoftmax

from models.CRTNModel import CRTNModel

from transformer import TransformerLM

import ipdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args(args=None):
    #Arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/xyz/Documents/Dataset/ptb_sample", help="data path")
    parser.add_argument("--demo", action="store_true", help="demo mode")
    parser.add_argument("--load", type=str, default="",  help="load model from saved models")
    parser.add_argument("--loadbaseline", type=str, default="",  help="load baseline model from saved models")
    parser.add_argument("--cache_N", type=int, default=5,  help="cache size N")
    parser.add_argument("--cache_k", type=int, default=3,  help="choose k segment from cache")
    parser.add_argument("--func", type=str, default="", choices=["attention_map", "demo_words", "contrast"], help="function to use, choices: attention_map, demo_words, contrast")
    parser.add_argument("--init_word", type=str, default="you", help="initial word for generation")
    parser.add_argument("--length", type=int, default=70,  help="generation text length")

    return parser.parse_args(args)

def evaluate(model, eval_data, criterion, model_args=None):
    if model_args is None:
        model.set_batch_size(model.args.eval_batch_size)
    else:
        model.args = model_args
    model.to(device)
    criterion.to(device)
    model.eval() 
    total_loss = 0.
    memory = None
    cache_info = torch.arange(model_args.cache_N, 0, -1, 
                           dtype=torch.float,
                           device=device)
    cache_info = cache_info.expand(model_args.eval_batch_size, -1)
    cache_info.transpose_(0, 1)
    key = None
    value = None
    len_eval = len(eval_data)



    with torch.no_grad():
        for data, targets in eval_data:
            data, targets = data.to(device), targets.to(device)
            data, targets = data.t(), targets.t()

            #if model_args is None:
            #    if model.args.farnear:
            #        output, memory, _ = model(data, neighbor_mem=memory)
            #    else:
            #        output, _ = model(data)
            #else:
            #    output, memory = model(data, memory)

            if args.farnear:
                if mem is not None:
                    mem = mem.detach()
                output, mems, mem, cache_info = model(text, key, value, 
                                                   neighbor_mem=mem, 
                                                   cache_info=cache_info)
            else:
                output, mems, cache_info = model(text, key, value, cache_info=cache_info)

            model.cache.set_batch_size(args.eval_batch_size)
            model.cache.init_key_and_value(key, value)
            model.cache.detach_memory()
            cache_info = model.cache.renew(mems, text, cache_info)
            key, value = (model.cache._get_keys(), 
                          model.cache._get_values().transpose(1, 2))
            model.cache.set_batch_size(args.eval_batch_size // len(args.devices))


            
            if model.args.adaptive:
                loss = criterion(output.view(-1, model.args.nhid), targets.view(-1))
                loss = loss.mean()
            else:
                loss = criterion(output.view(-1, model.args.vocab_size), targets.view(-1))
    
            total_loss += loss
    
    if model_args is None:
        model.set_batch_size(model.args.batch_size)

    return total_loss / len_eval


def contrast(model, base_model, criterion, base_criterion, dataloader, corpus):
    model.set_batch_size(model.args.eval_batch_size)
    model.to(device)
    criterion.to(device)
    mem = None

    base_model.to(device)
    base_criterion.to(device)
    memory = None

    id2w = corpus.vocabulary.index2word

    with torch.no_grad():
        prev_data = None
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            data, targets = data.t().contiguous(), targets.t().contiguous()
            seq_len, bsz = data.size(0), data.size(1)

            if model.args.farnear:
                output, mem, _ = model(data, neighbor_mem=mem)
            else:
                output, _ = model(data)
            base_output, memory = base_model(data, memory)

            head_prob, tails = criterion(output.view(-1, model.args.nhid), targets.view(-1), output=True)
            base_head_prob, base_tails = base_criterion(base_output.view(-1, base_model.d_model), targets.view(-1), output=True)

            def resize(obj, seq_len, bsz):
                if type(obj) == torch.Tensor:
                    return obj.reshape(seq_len, bsz, -1)
                else:
                    return [resize(o, seq_len, bsz) for o in obj]

            head_prob = resize(head_prob, seq_len, bsz)
            tails = resize(tails, seq_len, bsz)
            base_head_prob = resize(base_head_prob, seq_len, bsz)
            base_tails = resize(base_tails, seq_len, bsz)


            def display_candidates(head_prob, tail_prob, cutoffs, corpus, topk=5):
                probs, inds = head_prob.topk(topk)
                probs, inds = probs[:,0,:], inds[:,0,:]

                seg_words = []
                seg_probs = []
                for i, (prob, ind) in enumerate(list(zip(probs, inds))):

                    for iw in range(topk):
                        word_prob = prob[iw]
                        word_ind = ind[iw]
                        if word_ind.item() >= cutoffs[0]:
                            cluster = word_ind - cutoffs[0]
                            t_probs, t_inds = tail_prob[cluster][i,0,:].topk(topk)
                            prob = torch.cat((prob, t_probs + word_prob), 0)
                            ind = torch.cat((ind, t_inds + cutoffs[cluster]))
                            prob[iw] = -float("inf")
                    indice_choice = prob.topk(topk)[1]
                    word_choice = ind.index_select(0, indice_choice)
                    prob_choice = prob.index_select(0, indice_choice).exp()
                    word_choice = [id2w[w.item()] for w in word_choice]
                    seg_words.append(word_choice)
                    seg_probs.append(prob_choice)
        

                return seg_words, seg_probs

            cand_model, prob_model = display_candidates(head_prob, tails, model.args.cutoffs, corpus)
            cand_base, prob_base = display_candidates(base_head_prob, base_tails, base_model.cutoffs, corpus)
            showdata, showtgt = data[:,0], targets[:,0]
            showdata = [id2w[w.item()] for w in showdata]
            showtgt = [id2w[w.item()] for w in showtgt]

            if prev_data is not None:
                prev_words = prev_data[:,0]
                prev_words = [id2w[w.item()] for w in prev_words]


            for i in range(len(showdata)):
                print("-" * 89)
                print("Current word to predict:", end="")
                if prev_data is not None:
                    print(" ".join(prev_words), end="")
                print("%s \033[1;31m %s \033[0m" % (" ".join(showdata[:i+1]), showtgt[i]))

                print("baseline model prediction: ", end="")
                for word, prob in tuple(zip(cand_base[i], prob_base[i])):
                    if word.strip() == showtgt[i].strip(): 
                        print("\033[1;32m%s\033[0m|%.2f" % (word, prob), end=" ")
                    else:
                        print("%s|%.2f" % (word, prob), end=" ")
                print("")

                print("our crtn model prediction: ", end="")
                for word, prob in tuple(zip(cand_model[i], prob_model[i])):
                    if word.strip() == showtgt[i].strip(): 
                        print("\033[1;32m%s\033[0m|%.2f" % (word, prob), end=" ")
                    else:
                        print("%s|%.2f" % (word, prob), end=" ")
                print("\n")
                input("Enter to continue")

            prev_data = data

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
    model.set_batch_size(2)
    model.to(device)
    criterion.to(device)
    model.eval()
    
    num_steps = model.args.num_steps
    vocab_size = model.args.vocab_size
    cutoffs = model.args.cutoffs

    if not init_word in corpus.vocabulary.word2index.keys():
        print("initial word out of vocabulary")
        return
    init_ind = corpus.vocabulary.word2index[init_word]
    print(corpus.vocabulary.index2word[init_ind], end= " ")
    sequence = [init_ind] + [0 for _ in range(num_steps-1)]
    sequence = torch.tensor(sequence).view(-1, 1).to(device)
    sequence = sequence.expand(-1, 2)

    cache_info = torch.arange(model.args.cache_N, 0, -1, 
                           dtype=torch.float,
                           device=device)
    cache_info = cache_info.expand(2, -1)
    cache_info.transpose_(0, 1)
    key = None
    value = None
    if model.args.farnear:
        mem = None

    for i in range(length):
        if model.args.farnear:
            if mem is not None:
                mem = mem.detach()
            output, mems, mem, cache_info = model(sequence, key, value, 
                                               neighbor_mem=mem, 
                                               cache_info=cache_info)
        else:
            output, mems, cache_info = model(sequence, key, value, cache_info=cache_info)
        if not i == length - 1:
            model.cache.init_key_and_value(key, value)
            model.cache.detach_memory()
            cache_info = model.cache.renew(mems, sequence, cache_info)
            key, value = (model.cache._get_keys(), 
                          model.cache._get_values().transpose(1, 2))

            head_prob, tails = criterion(output[:,0,:].view(-1, model.args.nhid), sequence[:,0].view(-1), output=True)
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

    if args.loadbaseline:
        base_ckp = torch.load(args.loadbaseline)
        base_args = base_ckp["model_args"]
        base_args.demo = args.demo
        if args.demo:
            base_args.batch_size = 1

        base_model = TransformerLM(
                vocab_size=base_args.vocab_size,
                num_layer=base_args.nlayers,
                num_head=base_args.nhead,
                d_model=base_args.nhid,
                d_head=base_args.nhid // base_args.nhead,
                d_ff=base_args.d_ff,
                d_embedding=base_args.emsize,
                tied_weights=base_args.tied,
                num_steps=base_args.num_steps,
                mem_len=base_args.mem_len,
                attn_type=base_args.attn_type,
                init_std=base_args.init_std,
                adaptive=base_args.adaptive,
                div_val=base_args.div_val,
                cutoffs=base_args.cutoffs,
                dropout=base_args.dropout)
        base_model.load_state_dict(base_ckp["model_state_dict"])

        if base_args.adaptive:
            base_criterion = ProjectedAdaptiveLogSoftmax(base_args.vocab_size, base_args.emsize, base_args.nhid, base_args.cutoffs, div_val=base_args.div_val, init_std=base_args.init_std) 
            if base_args.tied:
                for i in range(len(base_criterion.out_layers)):
                    base_criterion.out_layers[i].weight = base_model.embedding.emb_layers[i].weight

            if base_args.tie_projs:
                for i, tie_proj in enumerate(base_args.tie_projs):
                    if tie_proj and base_args.div_val == 1 and base_args.nhid != base_args.emsize:
                        base_criterion.out_projs[i] = base_model.embedding.emb_projs[0]
                    elif tie_proj and base_args.div_val != 1:
                        base_criterion.out_projs[i] = base_model.embedding.emb_projs[i]
            base_criterion.load_state_dict(base_ckp["criterion"])

        else:
            base_criterion = nn.CrossEntropyLoss()

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
    elif args.func == "contrast":
        contrast(model, base_model, criterion, base_criterion, valid_loader, corpus)
    else:
        print('=' * 89)
        if args.loadbaseline:
            base_valid_loss = evaluate(base_model, valid_loader, base_criterion, base_args)
            print('| baseline valid loss {:5.2f} | baseline valid ppl {:8.2f}'.format(
                base_valid_loss, math.exp(base_valid_loss)))
            print('-' * 89)
        valid_loss = evaluate(model, valid_loader, criterion)
        print('| valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            valid_loss, math.exp(valid_loss)))
        print('=' * 89)
        if args.loadbaseline:
            base_test_loss = evaluate(base_model, test_loader, base_criterion, base_args)
            print('| baseline test loss {:5.2f} | baseline test ppl {:8.2f}'.format(
                base_test_loss, math.exp(base_test_loss)))
            print('-' * 89)
        test_loss = evaluate(model, test_loader, criterion)
        print('| test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == "__main__":
    args = parse_args()
    main(args)
