import argparse
import time

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


from models.CRANModel import CRANModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parseargs(args=None):
    #Arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path")
    parser.add_argument("--demo", action="store_true", help="demo mode")
    parser.add_argument("--load", type=str, default="",  help="load model from saved models")
    return parser.parse_args(args)

def evaluate(model, eval_data, criterion, arg):
    model.set_batch_size(arg.eval_batch_size)
    model.to(device)
    model.eval()
    total_loss = 0.
    hiddens = torch.zeros(arg.eval_batch_size, arg.hidden_size).to(device)

    with torch.no_grad():
        for i, (data, targets) in enumerate(eval_data):
            data, targets = data.to(device), targets.to(device)
            data, targets = data.t().contiguous(), targets.t().contiguous()

            logits, hiddens = model(data, hiddens)
            loss = criterion(logits.view(-1, model.args.vocab_size), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(eval_data)

def main(args):
    corpus = dataloader.Corpus(args.data_path)
    checkpoint = torch.load(args.load)
    keys = checkpoint["model_state_dict"].copy().keys()
    model_args = checkpoint["model_args"]
    model_args.demo = args.demo
    if args.demo:
        model_args.batch_size = 1
    args.batch_size = model_args.batch_size
    for key in keys:
        if re.match(r"cranunit.cache.keys", key) or re.match(r"cranunit.cache.values", key):
            popitem = checkpoint["model_state_dict"].pop(key)
            update_key = key.split(".")
            update_key[-1] = str(int(update_key[-1]) % model_args.cache_N)
            update_key = ".".join(update_key)
            if re.match(r"cranunit.cache.keys", key):
                checkpoint["model_state_dict"].update({update_key: torch.zeros(model_args.batch_size, model_args.cache_dk, device=popitem.device)})
                if model_args.demo:
                    update_key = key.split(".")
                    update_key[2] = "words"
                    update_key[-1] = str(int(update_key[-1]) % model_args.cache_N)
                    update_key = ".".join(update_key)
                    checkpoint["model_state_dict"].update({update_key: torch.zeros(model_args.cache_L, model_args.batch_size, device=popitem.device)})
            elif re.match(r"cranunit.cache.values", key):
                checkpoint["model_state_dict"].update({update_key: torch.zeros(model_args.cache_L, model_args.batch_size, model_args.cache_dk, device=popitem.device)})
    if args.demo:
        model = CRANModel(model_args, corpus)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = CRANModel(model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    test_loader = corpus.get_test_loader(batch_size=1, num_steps=model_args.num_steps)

    test_loss = evaluate(model, test_loader, criterion, model_args)
    print('test ppl {:8.2f}'.format(np.exp(test_loss)))

if __name__ == "__main__":
    args = parseargs()
    main(args)




