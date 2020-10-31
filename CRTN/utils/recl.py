import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

import os
import sys
sys.path.append("..")

from data import dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--model_paths", nargs="+", type=str, help="model paths")
    parser.add_argument("--initc", type=int, default=10, help="initial c")
    parser.add_argument("--initr", type=float, default=1.0, help="initial r")
    parser.add_argument("--delta", type=int, default=10, help="step of increasing c")
    return parser.parse_args()

def main(args):
    ipdb.set_trace()

if __name__ == "__main__":
    args = parse_args()
    main(args)
