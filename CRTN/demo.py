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

from utils.adaptive import ProjectedAdaptiveLogSoftmax

from models.CRTNModel import CRTNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args(args=None):
    #Arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/xyz/Documents/Dataset/ptb_sample", help="data path")
    parser.add_argument("--load", type=str, default="",  help="load model from saved models")
    parser.add_argument("--func", type=str, default="attention_map", choices=["attention_map", "demo_words"], help="function to use, choices: attention_map, demo_words")
    return parser.parse_args(args)

def main(args):
    checkpoint = torch.load(args.load)
    model_args = checkpoint["model_args"]
    model_args.demo = True
    model_args.eval_batch_size = 1
    model_state_dict = checkpoint["model_state_dict"]

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

    model = CRTNModel(model_args, corpus)
    model.load_state_dict(model_state_dict)

    cutoffs, tie_projs = [], [False]
    if model.args.adaptive:
        cutoffs = [20000, 40000, 80000]
        tie_projs += [True] * 3
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




if __name__ == "__main__":
    args = parse_args()
    main(args)


