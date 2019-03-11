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
from tensorboardX import SummaryWriter


from models.CRANModel import CRANModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parseargs(args=None):
    #Arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path")
    parser.add_argument("--demo", action="store_true", help="demo mode")
    parser.add_argument("--name", type=str, default="default", help="experiment name, default: default")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size, default: 20")
    parser.add_argument("--disp_freq", type=int, default=50, help="display frequency, default: 50")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="evaluation batch size, default: 1")
    parser.add_argument("--num_steps", type=int, default=35, help="num of words in one step, default: 35")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocabulary size, default: 10000")
    parser.add_argument("--embedding_dim", type=int, default=300, help="dimension of embedding, default: 300")
    parser.add_argument("--hidden_size", type=int, default=300, help="size of hidden state, default: 300")
    parser.add_argument("--normc", type=float, default=10, help="norm constant multiplied when normalize the weight, default: 10")
    parser.add_argument("--cache_N", type=int, default=5, help="size of Cache, default: 5")
    parser.add_argument("--cache_dk", type=int, default=300, help="dimension of key, default: 300")
    parser.add_argument("--cache_L", type=int, default=10, help="max length of a sequence in one value, default: 10")
    parser.add_argument("--cache_k", type=int, default=3, help="select top k values, default: 3")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate, default: 0.1")
    parser.add_argument("--max_epoch", type=int, default=10, help="max number of training epochs, default: 10")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout, default: 0.1")
    parser.add_argument("--seed", type=int, default=1111, help="random seed, default: 1111")
    parser.add_argument("--update", type=str, default="standard", choices=["standard", "gated"], help="method of updating hidden state, options: standard, gated. default: standard")
    parser.add_argument("--load", type=str, default="",  help="load model from saved models")
    return parser.parse_args(args)
    
def repackage_state(s):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(s, torch.Tensor):
        return s.detach()
    else:
        return list(repackage_state(v) for v in s)

def train(model, data_loader, criterion, optimizer, epoch, arg):
    model.set_batch_size(arg.batch_size)
    model.to(device)
    model.train()
    total_loss = 0.
    start_time = time.time()
    hiddens = model.init_hidden().to(device)


    for i, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        data, targets = data.t().contiguous(), targets.t().contiguous()
        model.zero_grad()

        logits, hiddens = model(data, hiddens)
        #print(logits.shape, targets.shape)
        loss = criterion(logits.view(-1, model.args.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % args.disp_freq == 0 and i > 0:
            cur_loss = total_loss / args.disp_freq
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(data_loader), args.lr,
                elapsed * 1000 / args.disp_freq, cur_loss, np.exp(cur_loss)))
            writer.add_scalar("train/loss", cur_loss, len(data_loader)*(epoch-1)+i)
            writer.add_scalar("train/ppl", np.exp(cur_loss), len(data_loader)*(epoch-1)+i)
            total_loss = 0.
            start_time = time.time()

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
    torch.manual_seed(args.seed)

    corpus = dataloader.Corpus(args.data_path)
    args.vocab_size = corpus.vocabulary.num_words

    if args.load:
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
                update_key[-1] = str(int(update_key[-1]) % 5)
                update_key = ".".join(update_key)
                if re.match(r"cranunit.cache.keys", key):
                    checkpoint["model_state_dict"].update({update_key: torch.zeros(model_args.batch_size, model_args.cache_dk, device=popitem.device)})
                    if model_args.demo:
                        update_key = key.split(".")
                        update_key[2] = "words"
                        update_key[-1] = str(int(update_key[-1]) % 5)
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
    else:
        if args.demo:
            args.batch_size = 1
            #args.num_steps = 1
            model = CRANModel(args, corpus)
        else:
            model = CRANModel(args)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


    train_loader = corpus.get_train_loader(batch_size=args.batch_size, num_steps=args.num_steps)
    valid_loader = corpus.get_valid_loader(batch_size=args.eval_batch_size, num_steps=args.num_steps)
    test_loader = corpus.get_test_loader(batch_size=args.eval_batch_size, num_steps=args.num_steps)


    for epoch in range(1, args.max_epoch+1):
        epoch_start_time = time.time()
        train(model, train_loader, criterion, optimizer, epoch, args)
        valid_loss = evaluate(model, valid_loader, criterion, args)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),valid_loss, np.exp(valid_loss)))
        print('-' * 89)
        writer.add_scalar("valid/ppl", np.exp(valid_loss), epoch)
        ipdb.set_trace()
        torch.save({
            "model_args": model.args,
            "model_state_dict": model.state_dict(),
            }, "save/" + args.name + "_" + str(epoch) + ".pt")
    test_loss = evaluate(model, test_loader, criterion, args)
    print('-' * 89)
    print('| end of training | test ppl {:8.2f}'.format(np.exp(test_loss)))
    print('-' * 89)


if __name__ == "__main__":
    args = parseargs()
    if not os.path.exists("./log/" + args.name):
        os.mkdir("./log/" + args.name)
    writer = SummaryWriter("log/" + args.name)
    main(args)
