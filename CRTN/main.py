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

sys.path.append("..")

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from torch.utils.data import DataLoader

from data.dataloader import TextDataset
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from utils.visual import TargetText
from models.CRTNModel import CRTNModel

if torch.__version__ < "1.2.0":
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter
import ipdb

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
    parser.add_argument('--stat', action='store_true',
                        help='stat memory choices')
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
    parser.add_argument('--attn_type', type=int, default=1, choices=[0, 1],
                        help='attention type, 0 for vaswani;1 for transformer-xl')
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_dk", type=int, default=240, 
                        help="dimension of key, default: 240")
    parser.add_argument("--cache_k", type=int, default=3, 
                        help="select top k values, default: 3")
    parser.add_argument('--multi_gpu', action="store_true",
                        help='enable multiple gpus')
    parser.add_argument('--adaptive', action="store_true",
                        help='use adaptive embedding and softmax')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of vocabulary, excluding special chars')
    parser.add_argument('--cutoffs', type=int, 
                        default=[2000, 4000, 6000], nargs="+",
                        help='cutoffs for adaptive embedding')
    parser.add_argument('--no_summary', action="store_true",
                        help='use the output of the transformer layer as key')
    parser.add_argument('--wise_summary', action="store_true",
                        help='use encoder function(transformer-xl) to summary the key')
    parser.add_argument('--max_pooling', action="store_true",
                        help='use max pooling to justice importance' 
                        'of segments in the cache')
    parser.add_argument('--query_method', type=str, default='vanilla', 
                        choices=['fixed_length', 'last_l', 'middle_l', 
                                 'linear', 'single_linear', 'single_sum', 'vanilla'],
                        help='query method to use. vanilla indicates just use '
                        'current segment to query, other methods link previous '
                        'segment. last_l and middle_l only work in wise_summary '
                        'mode')
    parser.add_argument('--not_weighted', action="store_true",
                        help='use not-weighted values directly as memory')
    parser.add_argument('--farnear', action="store_true",
                        help='split history into two parts,'
                        ' near to compute query and attention; far to be queried')
    parser.add_argument("--neighbor_len", type=int, default=50,
                        help="length of near neighbor; only use in farnear mode")
    parser.add_argument('--p_discard', action="store_true",
                        help='discard segment according to a computed posibility')
    parser.add_argument('--merge', action="store_true",
                        help='merge history instead of discarding')
    parser.add_argument('--merge_shift', action="store_true",
                        help='shift positioning encoding when merge')
    parser.add_argument("--merge_alpha", type=float, default=0.5,
                        help="ratio of retaining old information when merging")
    parser.add_argument('--real_pos', action="store_true",
                        help='compute position encoding according to realtime pos')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adaptive input and softmax')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--devices', type=int, default=[0], nargs="+",
                        help='device list')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model',
                        help='path to save the final model')
    parser.add_argument('--eval_steps', type=int, default=2000, metavar='N',
                        help='evaluation steps')
    parser.add_argument('--word_loss', action="store_true",
                        help='output loss of every word')
    parser.add_argument('--compare_farnear', action="store_true",
                        help='compare loss between far and near')
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

def init_key_num(args, device, evaluate=False):
    batch_size = args.eval_batch_size if evaluate else args.batch_size
    key_num = torch.arange(args.cache_N, 0, -1, 
                           dtype=torch.float,
                           device=device)
    key_num = key_num.expand(batch_size, -1)
    key_num.transpose_(0, 1)
    return key_num

def update_cache(model, batch_size, key, value, hidden, text, key_num):
    
    hidden = hidden.transpose(1, 2)

    model.cache.set_batch_size(batch_size)
    model.cache.init_key_and_value(key, value)
    model.cache.detach_memory()
    key_num = model.cache.renew(hidden, text, key_num)
    key, value = (model.cache._get_keys(), 
                  model.cache._get_values().transpose(1, 2))
    model.cache.set_batch_size(batch_size // len(model.args.devices))
    return model, key_num, key, value


def train(model, train_loader, valid_loader, criterion, 
          args, epoch, optimizer, scheduler, best_eval_ppl):

    model.train()
    start_time = time.time()
    total_loss = 0.
    module = model.module if args.multi_gpu else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[0]))
    else:
        device = torch.device("cpu")
    
    key = None
    value = None
    if args.farnear:
        mem = None
    key_num = init_key_num(args, device)

    for batch, data in enumerate(train_loader):
        text, targets = data.text, data.target
        if not text.size(0) == args.num_steps:
            continue
        text, targets = text.to(device), targets.to(device)
        model.zero_grad()
        
        if args.farnear:
            if mem is not None:
                mem = mem.detach()
            output, hidden, mem = model(text, key, value, 
                                        neighbor_mem=mem, 
                                        key_num=key_num)
        else:
            output, hidden = model(text, key, value, key_num=key_num)


        module, key_num, key, value = update_cache(module, args.batch_size, 
                                                   key, value, hidden, text, key_num)

        if args.adaptive:
            loss = criterion(output.reshape(-1, args.nhid), targets.reshape(-1))
            loss = loss.mean()
        else:
            loss = criterion(output.reshape(-1, args.vocab_size), targets.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if args.scheduler == "cosine":
            scheduler.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:1d} | {:5d}/{:5d} batches | lr {:02.2e} | '
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
                    "model_args": module.args,
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
        device = torch.device("cuda:" + str(module.args.devices[0]))
    else:
        device = torch.device("cpu")
    
    total_loss = 0.
    len_eval = 0
    total_len = len(eval_loader)
    key = None                  
    value = None                
    if args.farnear:            
        mem = None              
    key_num = init_key_num(args, device, True)
                               
    with torch.no_grad():      
        with tqdm(total=total_len) as pbar:
            pbar.set_description("evaluating")
                               
            for batch, data in enumerate(eval_loader):
                text, targets = data.text, data.target
                if not text.size(0) == args.num_steps:
                    pbar.update(1)
                    continue
                text, targets = text.to(device), targets.to(device)
                eval_batch_size = text.size(1)
                len_eval += eval_batch_size
                model.set_batch_size(eval_batch_size)
                               
                               
                if args.farnear:
                    if mem is not None:
                        mem = mem.detach()
                    output, hidden, mem = model(text, key, value, 
                                                neighbor_mem=mem, 
                                                key_num=key_num)
                else:
                    output, hidden = model(text, key, value, key_num=key_num)

                module, key_num, key, value = update_cache(module, 
                                                           eval_batch_size, 
                                                           key, value, 
                                                           hidden, 
                                                           text, 
                                                           key_num)

                if args.adaptive:
                    loss = criterion(output.view(-1, args.nhid), targets.view(-1))
                    loss = loss.sum()
                else:
                    loss = criterion(output.view(-1, args.vocab_size), 
                                     targets.view(-1))

                total_loss += loss.item()
                pbar.update(1)


    ppl = math.exp(total_loss / (len_eval * args.num_steps))
    model.set_batch_size(args.batch_size)
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

    if args.demo:
        args.batch_size = 1
        args.eval_batch_size = 1

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

        model_args.data = args.data
        model_args.demo = args.demo
        model_args.stat = args.stat
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

        if args.demo:
            batch_size = 1
            model_args.eval_batch_size = 1
        else:
            batch_size = args.batch_size
            model_args.eval_batch_size = args.eval_batch_size

        model_args.log_interval = args.log_interval
        model_args.eval_steps = args.eval_steps
        model_args.word_loss = args.word_loss

        args = model_args
        
    args.mem_len = args.cache_k * args.num_steps

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
        # load state_dict

        if args.demo:
            model = CRTNModel(model_args, corpus=corpus)
        else:
            model = CRTNModel(model_args)

        args.batch_size = batch_size

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.set_batch_size(batch_size)
    else:
        #create model
        if args.demo:
            model = CRTNModel(args, corpus=corpus)
        else:
            model = CRTNModel(args)

    
    if args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, 
                                                args.emsize, 
                                                args.nhid, 
                                                args.cutoffs, 
                                                div_val=args.div_val, 
                                                init_std=args.init_std) 
        if args.tied:
            for i in range(len(criterion.out_layers)):
                criterion.out_layers[i].weight = model.encoder.embedding.emb_layers[i].weight

        if args.tie_projs:
            for i, tie_proj in enumerate(args.tie_projs):
                if tie_proj and args.div_val == 1 and args.nhid != args.emsize:
                    criterion.out_projs[i] = model.encoder.embedding.emb_projs[0]
                elif tie_proj and args.div_val != 1:
                    criterion.out_projs[i] = model.encoder.embedding.emb_projs[i]
        if args.load:
            criterion.load_state_dict(checkpoint["criterion"])

    else:
        criterion = nn.CrossEntropyLoss()

    model.to(devices[0])
    criterion.to(devices[0])
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
                best_eval_ppl = train(model, train_loader, valid_loader, 
                                      criterion, args, epoch, optimizer, scheduler,
                                      best_eval_ppl)
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
                        "model_args": module.args,
                        "model_state_dict": module.state_dict(),
                        "criterion": criterion.state_dict()
                        }, 
                        savepath + "/" + args.save + "_best.pt")
                    print("save best model")

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    ### Reload the best model

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
    savepath = "../../../experiment/crtn/save/"
    timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")
    savepath += args.save + timestr
    
    if not os.path.exists("./log/" + args.save + timestr):
        os.mkdir("./log/" + args.save + timestr)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    writer = SummaryWriter("./log/" + args.save + timestr)

    main(args)
