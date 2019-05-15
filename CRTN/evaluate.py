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
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--demo", action="store_true", help="demo mode")
    parser.add_argument("--load", type=str, default="",  help="load model from saved models")
    return parser.parse_args(args)

def evaluate(model, eval_data, criterion, args):
    model.set_batch_size(model.args.eval_batch_size)
    model.eval() 
    total_loss = 0.

    with torch.no_grad():
        for data, targets in eval_data:
            data, targets = data.to(device), targets.to(device)
            data, targets = data.t(),contiguous(), targets.t().contiguous()

            output = model(data)
            
            if model.args.adaptive:
                loss = criterion(output.view(-1, model.args.nhid), targets.view(-1))
                loss = loss.mean()
            else:
                loss = criterion(output.view(-1, model.args.vocab_size), targets.view(-1))
    
        total_loss += loss

    model.set_batch_size(model.args.batch_size)

    return total_loss / len(eval_data)


def main(args):
    
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
