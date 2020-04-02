import time
from datetime import datetime
import os
import sys
import argparse
import socket
from copy import copy
from tqdm import tqdm

#ignore future warning from tensorboard
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")

sys.path.append("..")
sys.path.append("../baseline")
sys.path.append("../baseline/pytorch")

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

import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as apexDDP
    class ApexDataParallel(apexDDP):
        def __init__(self, module, **kwargs):
            super().__init__(module, **kwargs)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
    
        def set_batch_size(self, batch_size):
            batch_size = batch_division(batch_size, 
                                        self.rank, 
                                        self.world_size,
                                        single_value=True)
            self.batch_size = batch_size
            self.module.set_batch_size(batch_size)
except:
    print("No apex package found")

#if torch.__version__ < "1.2.0":
#    from tensorboardX import SummaryWriter
#else:
#    from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'constant'],
                        help='lr scheduler to use')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='lr_min for cosine scheduler')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='linear warmup steps')
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
    parser.add_argument('--dropatt', type=float, default=0.2,
                        help='dropout applied to attention (0 = no dropout)')
    parser.add_argument('--dropemb', type=float, default=0.1,
                        help='embedding dropout, random remove whole words')
    parser.add_argument('--dropinp', type=float, default=0.65,
                        help='input layer dropout')
    parser.add_argument('--dropwei', type=float, default=0.5,
                        help='linear weight dropout')
    parser.add_argument('--drophid', type=float, default=0.3,
                        help='hidden layers dropout')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0.0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0.0, proj_init_std)')
    parser.add_argument('--tied', action="store_true",
                        help='tied embedding weights')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_dk", type=int, default=240, 
                        help="dimension of key, default: 240")
    parser.add_argument("--cache_k", type=int, default=3, 
                        help="select top k values, default: 3")
    parser.add_argument("--cache_theta", type=float, default=1.0, 
                        help="cache theta, default: 1.0")
    parser.add_argument("--theta_annealing_alpha", type=float, default=1.0, 
                        help="cache theta annealing alpha, default: 1.0")
    parser.add_argument("--theta_annealing_steps", type=int, default=200, 
                        help="cache theta annealing steps, default: 200")
    parser.add_argument('--distributed', action="store_true",
                        help='enable distributed multiple gpus')
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
    parser.add_argument('--merge', action="store_true",
                        help='merge history instead of discarding')
    parser.add_argument('--merge_shift', action="store_true",
                        help='shift positioning encoding when merge')
    parser.add_argument("--merge_alpha", type=float, default=0.5,
                        help="ratio of retaining old information when merging")
    parser.add_argument('--discard_worst', action="store_true",
                        help='discard the least used section')
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
    parser.add_argument('--rank', type=int, default=0,
                        help='rank in nccl')
    parser.add_argument('--apex', action="store_true",
                        help='use apex to train')
    parser.add_argument('--opt_level', type=str, default='O1',
                        help='apex opt level')
    args = parser.parse_args()
    return args


class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, 
                 find_unused_parameters=True, **kwargs):
        super().__init__(module, device_ids, output_device, dim, 
                         find_unused_parameters=find_unused_parameters, 
                         **kwargs)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def set_batch_size(self, batch_size):
        batch_size = batch_division(batch_size, 
                                    self.rank, 
                                    self.world_size,
                                    single_value=True)
        self.batch_size = batch_size
        self.module.set_batch_size(batch_size)


def batch_division(batch_size, rank=0, world_size=None, single_value=False):
    if world_size is None:
        world_size = dist.get_world_size()
    batch_div = batch_size // world_size
    if rank < world_size - 1:
        if not single_value:
            return batch_div * rank, batch_div * (rank + 1)
        else:
            return batch_div
    elif rank == world_size - 1:
        if not single_value:
            return batch_div * rank, batch_size
        else:
            return batch_size - batch_div * rank

def init_cache_info(args, device, evaluate=False):
    """
    store relative position of cahce chunk and its recall times
    [pos, recall times, all query times]
    """
    batch_size = args.eval_batch_size if evaluate else args.batch_size
    pos = torch.arange(args.cache_N, 0, -1, 
                       dtype=torch.float,
                       device=device)
    recall_query = torch.zeros((args.cache_N, 2), dtype=torch.float, device=device)
    cache_info = torch.cat((pos.unsqueeze(-1), recall_query), dim=-1).unsqueeze(1)
    cache_info = cache_info.expand(-1, batch_size, -1).contiguous()
    return cache_info

def update_cache(model, batch_size, key, value, hidden, text, cache_info):
    
    hidden = hidden.transpose(1, 2)

    keys, values, cache_info = model.cache.renew(hidden, 
                                              text, 
                                              cache_info, 
                                              keys=key, 
                                              values=value)
    return model, cache_info, keys, values


def train(model, train_loader, valid_loader, criterion, scheduler, 
          args, epoch, step, optimizer, best_eval_ppl, writer):

    model.train()
    start_time = time.time()
    total_loss = 0.
    module = model.module if args.distributed else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[args.rank]))
    else:
        device = torch.device("cpu")
    
    key = None
    value = None
    if args.farnear:
        mem = None
    cache_info = init_cache_info(args, device)

    for batch, data in enumerate(train_loader):

        if not data.text.size(0) == args.num_steps:
            continue

        # load data
        if args.distributed:
            batch_start, batch_end = batch_division(data.target.size(1), 
                                                    args.rank)
            text, targets = (data.text[:,batch_start:batch_end].to(device), 
                             data.target[:,batch_start:batch_end].to(device))
        else:
            text, targets = data.text.to(device), data.target.to(device)

        model.zero_grad()
       
        # train
        if args.farnear:
            if mem is not None:
                mem = mem.detach()
            output, hidden, mem = model(text, key, value, 
                                        neighbor_mem=mem, 
                                        cache_info=cache_info)
        else:
            output, hidden = model(text, key, value, cache_info=cache_info)

        module, cache_info, key, value = update_cache(module, text.size(1), 
                                                   key, value, hidden, text, cache_info)

        if args.adaptive:
            loss = criterion(output.reshape(-1, args.nhid), targets.reshape(-1))
            loss = loss.mean()
        else:
            loss = criterion(output.reshape(-1, args.vocab_size), targets.reshape(-1))

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        step += 1
        if step <= args.warmup_steps:
            curr_lr = args.lr * step / args.warmup_steps
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            if args.scheduler == "cosine":
                scheduler.step()

        if step % args.theta_annealing_steps == 0 and args.theta_annealing_alpha < 1:
            module.cache.theta_annealing_step()
            print("STEP {:5d}, annealing theta to {:3.4f}".format(step, module.cache.theta))


        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            if args.distributed:
                cur_loss = torch.tensor([cur_loss]).cuda()
                dist.reduce(cur_loss, 0)
                cur_loss = cur_loss.item() / dist.get_world_size()
            elapsed = time.time() - start_time
            if args.rank == 0:
                print('| epoch {:1d} | {:5d}/{:5d} batches | lr {:02.2e} | '
                      'ms/batch {:4.0f} | loss {:4.2f} | ppl {:5.2f}'.format(
                    epoch, batch, len(train_loader), 
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    elapsed * 1000 / args.log_interval, 
                    cur_loss, 
                    math.exp(cur_loss)))
            writer.add_scalar("train/ppl", math.exp(cur_loss), 
                                batch + (epoch - 1) * len(train_loader))
            writer.flush()
            total_loss = 0.
            start_time = time.time()

        if batch % args.eval_steps == 0 and batch > 0:
            eval_ppl = evaluate(model, valid_loader, criterion, writer, args)
            if args.rank == 0:
                print('| eval at step {:3d} | eval ppl {:5.2f}'.format(batch, 
                                                                       eval_ppl))
                if eval_ppl < best_eval_ppl: 
                    best_eval_ppl = eval_ppl
                    torch.save({
                        "model_args": module.args,
                        "model_state_dict": module.state_dict(),
                        "criterion": criterion.state_dict()
                        }, 
                       args.savepath + "/" + args.save + "_best.pt")
                    print("save best model")
                print('-' * 60)
            start_time = time.time()

    return best_eval_ppl, step


def evaluate(model, eval_loader, criterion, writer, args):

    model.eval()
    module = model.module if args.distributed else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[args.rank]))
    else:
        device = torch.device("cpu")
    
    total_loss = 0.
    len_eval = 0
    total_len = len(eval_loader)
    key = None                  
    value = None                
    if args.farnear:            
        mem = None              
    cache_info = init_cache_info(args, device, True)

    if args.word_loss and args.rank == 0:
        vocab = eval_loader.dataset.fields["text"].vocab 
        loss_file = open(args.savepath + "/" + args.save + "_word_loss.pkl", "wb")
        loss_obj = TargetText(batch_size=args.eval_batch_size, 
                              num_steps=args.num_steps)         
        loss_obj.clear()                
                               
    with torch.no_grad():      
        with tqdm(total=total_len) as pbar:
            pbar.set_description("evaluating")
                               
            for batch, data in enumerate(eval_loader):
                if not data.text.size(0) == args.num_steps:
                    pbar.update(1)
                    continue
                               
                eval_batch_size = data.text.size(1)
                model.set_batch_size(eval_batch_size)

                if args.distributed:
                    batch_start, batch_end = batch_division(eval_batch_size, 
                                                            args.rank)
                    
                    text, targets = (data.text[:,batch_start:batch_end].to(device), 
                                     data.target[:,batch_start:batch_end].to(device))
                else:
                    text, targets = data.text.to(device), data.target.to(device)
                               
                eval_batch_size = text.size(1)
                len_eval += targets.view(-1).size(0)

                if args.farnear:
                    if mem is not None:
                        mem = mem.detach()
                    output, hidden, mem = model(text, key, value, 
                                                neighbor_mem=mem, 
                                                cache_info=cache_info)
                else:
                    output, hidden = model(text, key, value, cache_info=cache_info)

                module, cache_info, key, value = update_cache(module, 
                                                           text.size(1), 
                                                           key, value, 
                                                           hidden, 
                                                           text, 
                                                           cache_info)

                if args.adaptive:
                    loss_tensor = criterion(output.view(-1, args.nhid), 
                                            targets.view(-1),
                                            keep_order=True)
                    loss = loss_tensor.sum()
                else:
                    loss = criterion(output.view(-1, args.vocab_size), 
                                     targets.view(-1))

                total_loss += loss.item()

                if args.word_loss:
                    if args.distributed:
                        targets_list = [targets.new_zeros(targets.size(0), batch_division(data.target.size(1), r, single_value=True)) for r in range(dist.get_world_size())]
                        loss_list = [loss_tensor.new_zeros(targets.size(0) * batch_division(data.target.size(1), r, single_value=True)) for r in range(dist.get_world_size())]
                        dist.all_gather(targets_list, targets)
                        dist.all_gather(loss_list, loss_tensor)
                        targets = torch.cat(targets_list, dim=1)
                        loss_tensor = torch.cat(loss_list, dim=0)
                    if args.rank == 0:
                        words = [vocab.itos[w] for w in targets.view(-1)]
                        word_loss = [l.item() for l in loss_tensor]
                        loss_obj.add_words(words)
                        loss_obj.add_losss(word_loss)

                pbar.update(1)

    if args.word_loss and args.rank == 0:
        pkl.dump(loss_obj, loss_file)
        loss_file.close()

    if args.distributed:
        total_loss = torch.tensor([total_loss]).cuda()
        len_eval = torch.tensor([len_eval]).cuda()
        dist.reduce(total_loss, 0)
        dist.reduce(len_eval, 0)
        total_loss = total_loss.item()
        len_eval = len_eval.item()
    ppl = math.exp(total_loss / len_eval)
    model.set_batch_size(args.batch_size)
    model.train()
    return ppl

def load_dataset(args):
    datatime_begin = time.time()
    
    if args.datasets == "ptb":
        if args.rank == 0:
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
        vocab_size = len(vocab.itos)
    elif args.datasets == "wt103":
        if args.rank == 0:
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
        vocab_size = len(vocab.itos)
    else:
        if args.rank == 0:
            print("Loading data from %s" % args.data)
        corpus = TextDataset(args.data, args.vocab_size, args.num_steps)
        vocab_size = len(corpus.TEXT.vocab.itos)

        train_loader = corpus.get_train_loader(args.batch_size)
        valid_loader = corpus.get_valid_loader(args.eval_batch_size)
        test_loader = corpus.get_test_loader(args.eval_batch_size)

    datatime_end = time.time()
    datatime = datatime_end - datatime_begin

    return (train_loader, valid_loader, test_loader), vocab_size, datatime




def main(args):

    def init_weights(model):
        classname = model.__class__.__name__
        if classname in ["WeightDropLinear", "Embedding"]:
            if hasattr(model, 'weight') and model.weight is not None:
                nn.init.normal_(model.weight, 0.0, args.init_std)
            if hasattr(model, 'bias') and model.bias is not None:
                nn.init.constant_(model.bias, 0.0)
        elif classname == "LayerNorm":
            if hasattr(model, 'weight') and model.weight is not None:
                nn.init.normal_(model.weight, 1.0, args.init_std)
            if hasattr(model, 'bias') and model.bias is not None:
                nn.init.constant_(model.bias, 0.0)
        elif classname == "Linear":
            if hasattr(model, 'weight') and model.weight is not None:
                nn.init.normal_(model.weight, 0.0, args.init_std)
            #if hasattr(model, 'bias') and model.bias is not None:
            #    nn.init.constant_(model.bias, 0.0)


    writer = SummaryWriter("./log/" + args.save + args.timestr)

    if torch.cuda.is_available():
        devices = [torch.device("cuda:" + str(i)) for i in args.devices]
    else:
        devices = [torch.device("cpu")]

    device = devices[args.rank]
    torch.cuda.set_device(device)

    if args.distributed:
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.rank,
                                world_size=len(devices))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.adaptive:
        args.tie_projs = [False] + [True] * 3

    if args.demo:
        args.batch_size = 1
        args.eval_batch_size = 1

    ### Load Data ###
    datasets, vocab_size, data_time = load_dataset(args)
    args.vocab_size = vocab_size
    print("Data loading finished. time: {:.3f} s".format(data_time))
    train_loader, valid_loader, test_loader = datasets

    total_steps = len(train_loader) * args.epochs

    if args.load:
        # Load Model
        checkpoint = torch.load(args.load, map_location=device)
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
        model_args.distributed = args.distributed
        model_args.apex = args.apex
        model_args.devices = args.devices
        model_args.save = args.save

        model_args.rank = args.rank

        if args.demo:
            batch_size = 1
            model_args.eval_batch_size = 1
        else:
            batch_size = args.batch_size
            model_args.eval_batch_size = args.eval_batch_size

        if not model_args.num_steps == args.num_steps:
            print("REDEFINE num_steps: {} --> {}".format(model_args.num_steps, 
                                                            args.num_steps))
            model_args.num_steps = args.num_steps
        if not model_args.neighbor_len == args.neighbor_len:
            print("REDEFINE neighbor_len: {} --> {}".format(model_args.neighbor_len, 
                                                            args.neighbor_len))
            model_args.neighbor_len = args.neighbor_len
        if not model_args.cache_N == args.cache_N:
            print("REDEFINE cache_N: {} --> {}".format(model_args.cache_N, 
                                                       args.cache_N))
            model_args.cache_N = args.cache_N
        if not model_args.cache_k == args.cache_k:
            print("REDEFINE cache_k: {} --> {}".format(model_args.cache_k, 
                                                       args.cache_k))
            model_args.cache_k = args.cache_k
        if not model_args.clamp_len == args.clamp_len:
            print("REDEFINE clamp_len: {} --> {}".format(model_args.clamp_len, 
                                                         args.clamp_len))
            model_args.clamp_len = args.clamp_len
        if not model_args.cache_theta == args.cache_theta:
            print("REDEFINE cache_theta: {} --> {}".format(model_args.cache_theta, 
                                                         args.cache_theta))
            model_args.cache_theta = args.cache_theta
        model_args.same_length = args.same_length

        model_args.log_interval = args.log_interval
        model_args.eval_steps = args.eval_steps
        model_args.word_loss = args.word_loss

        args = model_args
        
    args.mem_len = args.cache_k * args.num_steps
    if not args.eval:
        args.cache_theta *= (1 / args.theta_annealing_alpha) ** (total_steps // args.theta_annealing_steps)

    #Print Params
    if args.rank == 0:
        for argk, argv in args.__dict__.items():
            print("{}: {}".format(argk, argv))
        print("")



    if args.load:
        # load state_dict

        if args.demo:
            model = CRTNModel(model_args, corpus=corpus)
        else:
            model = CRTNModel(model_args)

        args.batch_size = batch_size

        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_batch_size(batch_size)
    else:
        #create model
        if args.demo:
            model = CRTNModel(args, corpus=corpus)
        else:
            model = CRTNModel(args)

    if args.rank == 0:
        all_param = sum([p.numel() for p in model.parameters()])
        nonemb_param = sum([p.numel() for p in model.encoder.layers.parameters()])
        print("#model params = {}".format(all_param))
        print('#non emb params = {}'.format(nonemb_param))

        if args.eval:
            print("SKIP TRAINING")
        else:
            print("TRAINING......")
    
    if args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, 
                                                args.emsize, 
                                                args.nhid, 
                                                args.cutoffs, 
                                                div_val=args.div_val, 
                                                init_std=args.init_std,
                                                proj_init_std=args.proj_init_std
                                                ) 
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

    model.apply(init_weights)

    model.cuda()
    criterion.cuda()

    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.distributed:
        if args.apex:
            model = ApexDataParallel(model) 
        else:
            model = DistributedDataParallel(model, 
                                            device_ids=[device], 
                                            dim=1)
        model.set_batch_size(args.batch_size)
    
    if args.scheduler == "cosine":
        scheduler_steps = total_steps - args.warmup_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=scheduler_steps,
                                                         eta_min=args.eta_min)
    elif args.scheduler == "constant":
        scheduler = None


    ### Training ###

    if not args.eval:
        try:
            best_eval_ppl = float('inf')
            best_eval_ppls = []
            train_step = 0
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                best_eval_ppl, train_step = train(model, 
                                                  train_loader, 
                                                  valid_loader, 
                                                  criterion,
                                                  scheduler,
                                                  args, 
                                                  epoch,
                                                  train_step,
                                                  optimizer, 
                                                  best_eval_ppl, 
                                                  writer)

                eval_ppl = evaluate(model, valid_loader, criterion, writer, args)

                if args.rank == 0:
                    module = model.module if args.distributed else model
                    if eval_ppl < best_eval_ppl:
                        best_eval_ppl = eval_ppl
                        torch.save({
                            "model_args": args,
                            "model_state_dict": module.state_dict(),
                            "criterion": criterion.state_dict()
                            }, 
                            args.savepath + "/" + args.save + "_best.pt")
                        print("save best model")

                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid ppl '
                          '{:8.2f}'.format(epoch, 
                                           (time.time() - epoch_start_time),
                                           eval_ppl))
                    print('-' * 89)

                    writer.add_scalar("valid/ppl", eval_ppl, 
                                      epoch * len(train_loader))
                    writer.flush()
                
                    best_eval_ppls.append(eval_ppl)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    ### Reload the best model

    if args.rank == 0:
        if args.eval:
            best_model = args.load
        else:
            best_model = args.savepath + "/" + args.save + "_best.pt"

        eval_checkpoint = torch.load(best_model, map_location=device)
        model_state_dict = eval_checkpoint["model_state_dict"]

        module = model.module if args.distributed else model
        module.load_state_dict(model_state_dict)

        if args.adaptive:
            criterion.load_state_dict(eval_checkpoint["criterion"])

        print("=" * 89)
        print("experiment name: {}".format(args.save))
        print("saved in: {}".format(os.path.abspath(args.savepath)))

    if args.distributed:
        broadcast(model)
        broadcast(criterion)


    best_eval_ppl = evaluate(model, valid_loader, criterion, writer, args)

    test_ppl = evaluate(model, test_loader, criterion, writer, args)

    if args.rank == 0:
        print('=' * 89)
        print('| End of training | best valid ppl {:8.2f}'.format(best_eval_ppl))
        print('=' * 89)
        print('| test ppl {:8.2f}'.format(test_ppl))
        print('=' * 89)

    if args.distributed:
        dist.destroy_process_group()


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


if __name__ == "__main__":

    args = parse_args()
    savepath = "../../../experiment/crtn/save/"
    timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")
    savepath += args.save + timestr

    args.savepath = savepath
    args.timestr = timestr
    
    if not os.path.exists("./log/" + args.save + timestr):
        os.mkdir("./log/" + args.save + timestr)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    writer = SummaryWriter("./log/" + args.save + timestr)

    #### Load Data ###
    #datasets, vocab_size, data_time = load_dataset(args)
    #args.vocab_size = vocab_size
    #print("Data loading finished. time: {:.3f} s".format(data_time))

    world_size = len(args.devices)
    if world_size == 1 and not args.apex:
        args.distributed = False

    if args.distributed:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            args.url = url

        from utils import dist_scripts
        process_fn = dist_scripts.process_lm_fn
        mp.spawn(process_fn, args=(args, ), nprocs=world_size)
    else:
        main(args)
