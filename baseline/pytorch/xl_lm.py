import time
from datetime import datetime
import os
import sys
import argparse
from tqdm import tqdm

#ignore future warning from tensorboard
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")

sys.path.append("../..")

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from torch.utils.data import DataLoader

from CRTN.data.dataloader import TextDataset
from CRTN.utils.adaptive import ProjectedAdaptiveLogSoftmax
from CRTN.utils.visual import TargetText
from transformer import TransformerLM

from torch.utils.tensorboard import SummaryWriter

import ipdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample/',
                        help='location of the data corpus')
    parser.add_argument('--datasets', type=str, choices=["fromfile", "ptb", "wt103"], 
                        default="fromfile", help='load datasets from torchtext')
    parser.add_argument('--eval', action='store_true',
                        help='skip training')
    parser.add_argument('--demo', action='store_true',
                        help='demo mode')
    parser.add_argument('--adam', action='store_true',
                        help='adam optimizer')
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='number of heads')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='dimension of feed-forward')
    parser.add_argument('--mem_len', type=int, default=60,
                        help='length of memory')
    parser.add_argument('--lr', type=float, default=25e-5,
                        help='initial learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'constant'],
                        help='lr scheduler to use')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, 
                        help='eval batch size')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0.0, init_std)')
    parser.add_argument('--tied', action="store_true",
                        help='tied embedding weights')
    parser.add_argument('--attn_type', type=int, default=0, choices=[0, 1],
                        help='attention type, 0 for vaswani;1 for transformer-xl')
    parser.add_argument('--multi_gpu', action="store_true",
                        help='enable multiple gpus')
    parser.add_argument('--devices', type=int, default=[0], nargs="+",
                        help='device list')
    parser.add_argument('--adaptive', action="store_true",
                        help='use adaptive embedding and softmax')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of vocabulary, excluding special chars')
    parser.add_argument('--cutoffs', type=int, 
                        default=[2000, 4000, 6000], nargs="+",
                        help='cutoffs for adaptive embedding')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adaptive input and softmax')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--eval_steps', type=int, default=2000, metavar='N',
                        help='evaluation steps')
    parser.add_argument('--save', type=str, default='model',
                        help='path to save the final model')
    parser.add_argument('--word_loss', action="store_true",
                        help='output loss of every word')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    args = parser.parse_args()
    return args

class DataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, 
                 find_unused_parameters=True):
        super().__init__(module, device_ids, output_device, dim)

    def set_batch_size(self, batch_size):
        batch_size = batch_size // len(self.device_ids)
        self.module.set_batch_size(batch_size)
        self.batch_size = batch_size



def train(model, train_loader, valid_loader, criterion, 
          args, epoch, optimizer, scheduler, best_eval_ppl):

    model.train()
    start_time = time.time()
    total_loss = 0.
    module = model.module if args.multi_gpu else model
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.devices[0]))
    else:
        device = torch.device("cpu")

    memory = None

    for batch, data in enumerate(train_loader):
        text, targets = data.text, data.target
        if not text.size(0) == args.num_steps:
            continue
        text, targets = text.to(device), targets.to(device)

        model.zero_grad()

        output, memory = model(text, memory)

        if args.adaptive:
            loss = criterion(output.view(-1, args.nhid), targets.view(-1))
            loss = loss.mean()
        else:
            loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if args.scheduler == "cosine":
            scheduler.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:1d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                    'ms/batch {:4.0f} | loss {:4.2f} | ppl {:5.2f}'.format(
                epoch, batch, len(train_loader), 
                optimizer.state_dict()["param_groups"][0]["lr"],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            writer.add_scalar("train/ppl", math.exp(cur_loss), 
                              batch + (epoch - 1) * len(train_loader))
            writer.flush()
            total_loss = 0.
            start_time = time.time()

        if batch % args.eval_steps == 0 and batch > 0:
            eval_ppl = evaluate(model, valid_loader, criterion, args)
            print('| eval at step {:3d} | eval ppl {:5.2f}'.format(batch, eval_ppl))
            if eval_ppl < best_eval_ppl: 
                best_eval_ppl = eval_ppl
                torch.save({
                    "model_args": args,
                    "model_state_dict": module.state_dict(),
                    "criterion": criterion.state_dict()
                    }, 
                    savepath + "/" + args.save + "_best.pt")
                print("save best model")
            print('-' * 60)
            start_time = time.time()

    return best_eval_ppl



def evaluate(model, eval_loader, criterion, args):

    model.eval()
    module = model.module if args.multi_gpu else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.devices[0]))
    else:
        device = torch.device("cpu")

    total_loss = 0.
    len_eval = 0
    total_len = len(eval_loader)
    memory = None

    with torch.no_grad():
        with tqdm(total=total_len) as pbar:
            for i, data in enumerate(eval_loader):
                text, targets = data.text, data.target
                if not text.size(0) == args.num_steps:
                    pbar.update(1)
                    continue
                text, targets = text.to(device), targets.to(device)
                eval_batch_size = text.size(1)
                len_eval += eval_batch_size

                output, memory = model(text, memory)

                if args.adaptive:
                    loss = criterion(output.view(-1, args.nhid), targets.view(-1))
                    loss = loss.sum()
                else:
                    loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))

                total_loss += loss.item()
                pbar.update(1)

    ppl = math.exp(total_loss / (len_eval * args.num_steps))
    model.train()
    return ppl


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        devices = [torch.device("cuda:" + str(i)) for i in args.devices]

        args.batch_size = len(devices) * (args.batch_size // len(devices))
        args.eval_batch_size = len(devices) * (args.eval_batch_size // len(devices))
    else:
        devices = [torch.device("cpu")]

    if args.adaptive:
        args.tie_projs = [False] + [True] * 3

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
    else:
        print("Loading data from %s" % args.data)
        corpus = TextDataset(args.data, args.vocab_size, args.num_steps)
        args.vocab_size = len(corpus.TEXT.vocab.itos)

        train_loader = corpus.get_train_loader(args.batch_size)
        valid_loader = corpus.get_valid_loader(args.eval_batch_size)
        test_loader = corpus.get_test_loader(args.eval_batch_size)

    datatime_end = time.time()

    if args.load:
        # Load Model
        checkpoint = torch.load(args.load, map_location=devices[0])
        model_args = checkpoint["model_args"]

        model_args.demo = args.demo
        model_args.data = args.data
        model_args.eval = args.eval
        model_args.load = args.load
        model_args.adam = args.adam
        model_args.lr = args.lr
        model_args.scheduler = args.scheduler
        model_args.clip = args.clip
        model_args.epochs = args.epochs
        model_args.multi_gpu = args.multi_gpu
        model_args.devices = args.devices
        model_args.save = args.save

        model_args.batch_size = args.batch_size
        model_args.eval_batch_size = args.eval_batch_size

        model_args.log_interval = args.log_interval
        model_args.eval_steps = args.eval_steps
        model_args.word_loss = args.word_loss

        args = model_args

    #Print Params
    for argk, argv in args.__dict__.items():
        print("{}: {}".format(argk, argv))
    print("")
    print("Data loading finished. time: {:.3f} s".format(datatime_end-datatime_begin))

    if args.eval:
        print("SKIP TRAINING")
    else:
        print("TRAINING......")

    if args.load:

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
                attn_type=model_args.attn_type,
                init_std=model_args.init_std,
                adaptive=model_args.adaptive,
                div_val=model_args.div_val,
                cutoffs=model_args.cutoffs,
                dropout=model_args.dropout)

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # create model
        model = TransformerLM(
                vocab_size=args.vocab_size,
                num_layer=args.nlayers,
                num_head=args.nhead,
                d_model=args.nhid,
                d_head=args.nhid // args.nhead,
                d_ff=args.d_ff,
                d_embedding=args.emsize,
                tied_weights=args.tied,
                num_steps=args.num_steps,
                mem_len=args.mem_len,
                attn_type=args.attn_type,
                init_std=args.init_std,
                adaptive=args.adaptive,
                div_val=args.div_val,
                cutoffs=args.cutoffs,
                dropout=args.dropout)


    
    if args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, 
                                                args.emsize, 
                                                args.nhid, 
                                                args.cutoffs, 
                                                div_val=args.div_val, 
                                                init_std=args.init_std) 
        if args.tied:
            for i in range(len(criterion.out_layers)):
                criterion.out_layers[i].weight = model.embedding.emb_layers[i].weight

        if args.tie_projs:
            for i, tie_proj in enumerate(args.tie_projs):
                if tie_proj and args.div_val == 1 and args.nhid != args.emsize:
                    criterion.out_projs[i] = model.embedding.emb_projs[0]
                elif tie_proj and args.div_val != 1:
                    criterion.out_projs[i] = model.embedding.emb_projs[i]

    else:
        criterion = nn.CrossEntropyLoss()
        
    model = model.to(devices[0])
    criterion = criterion.to(devices[0])
    if args.multi_gpu:
        model = DataParallel(model, device_ids=devices, dim=1)

    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         len(train_loader) 
                                                         * args.epochs)
    elif args.scheduler == "constant":
        scheduler = None

    ### Training ###

    if not args.eval:
        try:
            best_eval_ppl = float('inf')
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                best_eval_ppl = train(model, train_loader, 
                                      valid_loader, criterion, 
                                      args, epoch, optimizer, 
                                      scheduler, best_eval_ppl)
                eval_ppl = evaluate(model, valid_loader, criterion, args)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | '
                        'valid ppl {:8.2f}'.format(epoch, 
                                                   (time.time() - epoch_start_time),
                                                   eval_ppl))
                print('-' * 89)

                writer.add_scalar("valid/ppl", eval_ppl, 
                                  epoch * len(train_loader))
                writer.flush()

                module = model.module if args.multi_gpu else model

                if eval_ppl < best_eval_ppl:
                    best_eval_ppl = eval_ppl
                    torch.save({
                        "model_args": args,
                        "model_state_dict": module.state_dict(),
                        "criterion": criterion.state_dict()
                        }, 
                        savepath + "/" + args.save + "_best.pt")
                print("save best model")

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    ### Reload best model

    if args.eval:
        best_model = args.load
    else:
        best_model = savepath + "/" + args.save + "_best.pt"

    eval_checkpoint = torch.load(best_model, map_location=devices[0])
    model_state_dict = eval_checkpoint["model_state_dict"]

    module = model.module if args.multi_gpu else model
    module.load_state_dict(model_state_dict)

    if args.adaptive:
        criterion.load_state_dict(eval_checkpoint["criterion"])

    print("=" * 89)
    print("experiment name: {}".format(args.save))
    print("saved in: {}".format(os.path.abspath(savepath)))

    best_eval_ppl = evaluate(model, valid_loader, criterion, args)

    test_ppl = evaluate(model, test_loader, criterion, args)
    print('=' * 89)
    print('| End of training | best valid ppl {:8.2f}'.format(best_eval_ppl))
    print('=' * 89)
    print('| test ppl {:8.2f}'.format(test_ppl))
    print('=' * 89)



if __name__ == "__main__":
    args = parse_args()
    savepath = "../../../../experiment/crtn/save/"
    timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")
    savepath += args.save + timestr
    
    if not os.path.exists("./log/" + args.save + timestr):
        os.mkdir("./log/" + args.save + timestr)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    writer = SummaryWriter("./log/" + args.save + timestr)

    main(args)
