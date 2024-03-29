import time
from datetime import datetime
import os
import sys
import argparse
import socket
import re
from itertools import chain
from tqdm import tqdm
from copy import copy, deepcopy
import pickle as pkl

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

from data.dataloader import TextDataset, ExistingDataset
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from utils.visual import TargetText
from utils.utils import init_cache_info, batch_division, param_in
from utils.utils import padding_hidden, padding_cache, maintain_checkpoints, Logger
from models.CRTNModel import CRTNModel

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

logger = Logger("log")

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data', type=str,
                        default='',
                        help='path of the data corpus')
    parser.add_argument('--datasets', type=str, 
                        default="ptb", help='dataset name, load datasets from torchtext: ptb, wt2, wt103')
    # optimization
    parser.add_argument('--adam', action='store_true',
                        help='adam optimizer')
    parser.add_argument('--lr', type=float, default=25e-5,
                        help='initial learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'constant'],
                        help='lr scheduler to use')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='lr_min for cosine scheduler')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='linear warmup steps')
    parser.add_argument('--emb_mult', type=float, default=2,
                        help='multiplier for the learning rate of embeddings')
    parser.add_argument('--ema_lr_mult', type=float, default=0.5,
                        help='lr multiplier when switching to EMA.')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    # regularization
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='alpha L2 regularization on activation')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='beta slowness regularization applied on activiation')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropatt', type=float, default=0.2,
                        help='dropout applied to attention (0 = no dropout)')
    parser.add_argument('--dropemb', type=float, default=0.1,
                        help='embedding dropout, random remove whole words')
    parser.add_argument('--dropinp', type=float, default=0.6,
                        help='input layer dropout')
    parser.add_argument('--dropwei', type=float, default=0.1,
                        help='linear weight dropout')
    parser.add_argument('--dropfor', type=float, default=0.2,
                        help='forward layers dropout')
    parser.add_argument('--drophid', type=float, default=0.0,
                        help='hidden layers dropout')
    parser.add_argument('--dropmos', type=float, default=0.2,
                        help='MoS latent layers dropout')
    # hyperparams
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='number of heads')
    parser.add_argument('--d_head', type=int, default=40,
                        help='dimension of single head')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='dimension of feed-forward')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size')
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
    parser.add_argument('--same_length_query', action='store_true',
                        help='use the same attn length for all tokens in query')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='sequence length')
    parser.add_argument("--neighbor_len", type=int, default=50,
                        help="length of near neighbor; only use in farnear mode")
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_dk", type=int, default=240, 
                        help="dimension of key, default: 240")
    parser.add_argument("--cache_k", type=int, default=3, 
                        help="select top k values, default: 3")
    parser.add_argument("--cache_L", type=int, default=20, 
                        help="length of segments in cache, default: 20 "
                             "it is the max sentence length in sentence_cache mode")
    parser.add_argument('--adaptive', action="store_true",
                        help='use adaptive embedding and softmax')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of vocabulary, excluding special chars')
    parser.add_argument('--cutoffs', type=int, 
                        default=[], nargs="+",
                        help='cutoffs for adaptive embedding')
    parser.add_argument('--max_pooling', action="store_true",
                        help='use max pooling to justice importance' 
                        'of segments in the cache')
    parser.add_argument('--query_method', type=str, default='single', 
                        choices=['vanilla', 'last_l', 'middle_l', 'linear', 'single', 
                                 'single_linear'],
                        help='method to compute query of words. vanilla indicates '
                        'use current segment to query, other methods link previous '
                        'segment.')
    parser.add_argument('--summary_method', type=str, default='no_summary', 
                        choices=['no_summary', 'max', 'mean', 'sum', 
                                 'weighted_sum', 'last_state', 'linear', 'conv'],
                        help='method to summary key of segments')
    parser.add_argument('--sentence_cache', action="store_true",
                        help='store a sentence in per cache slot, in this mode, cache_L is the max sentence length')
    parser.add_argument('--not_weighted', action="store_true",
                        help='use not-weighted values directly as memory')
    parser.add_argument('--no_pos', action="store_true",
                        help='disable pos embedding')
    parser.add_argument('--no_pos_bias', action="store_true",
                        help='disable pos bias u and v')
    parser.add_argument('--farnear', action="store_true",
                        help='split history into two parts,'
                        ' near to compute query and attention; far to be queried')
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
    parser.add_argument('--mos', action='store_true',
                        help='use mixture of softmaxes (Yang et al. 2018)')
    parser.add_argument('--n_experts', type=int, default=15,
                        help='number of experts in mos')
    # training setting
    parser.add_argument('--std_epochs', type=int, default=150,
                        help='number of epochs of standard training')
    parser.add_argument('--ema_epochs', type=int, default=50,
                        help='number of epochs with ema of params')
    parser.add_argument('--nonmono', type=int, default=-1,
                        help='non mono epochs to skip std epochs, -1 to disable it')
    parser.add_argument('--update_cycle', type=int, default=1,
                        help='gradient update cycle, use for simullate large batch_size')
    parser.add_argument('--mu', type=float, default=-1,
                        help='mu used for EMA. set to -1 to use 1 / step.')
    parser.add_argument("--theta_annealing_alpha", type=float, default=1.0, 
                        help="attention theta annealing alpha, default: 1.0")
    parser.add_argument("--theta_annealing_steps", type=int, default=200, 
                        help="attention theta annealing steps, default: 200")
    parser.add_argument('--random_seq_len', action="store_true",
                        help='random sequence length rather than fixed')
    parser.add_argument('--partial_shuffle', action="store_true",
                        help='partial shuffle training text use Press et al. 2019')
    parser.add_argument('--distributed', action="store_true",
                        help='enable distributed multiple gpus')
    parser.add_argument('--devices', type=int, default=[0], nargs="+",
                        help='device list')
    # eval setting
    parser.add_argument('--eval_batch_size', type=int, default=10, 
                        help='eval batch size')
    parser.add_argument('--eval_index', type=str, default='none',
                        help='file including indexes of words to compute PPL')
    parser.add_argument('--eval_steps', type=int, default=2000, metavar='N',
                        help='evaluation steps')
    parser.add_argument('--eval_temperature', type=float, default=1.0, 
                        help='eval temperature, divide logits.')
    parser.add_argument('--eval_temp_search', action="store_true",
                        help='search best temperature on valid set during test. 1-1.2/0.02')
    parser.add_argument('--eval_theta_search', action="store_true",
                        help='search best theta on valid set during test. 0.7-1/0.02')
    # setting
    parser.add_argument('--eval_metric', type=str, default='ppl', choices=["ppl", "bpc"],
                        help='evaluation metric. use ppl for word level models and bpc for char level models')
    parser.add_argument("--cache_theta", type=float, default=1.0, 
                        help="cache query theta, default: 1.0")
    parser.add_argument("--attn_theta", type=float, default=1.0, 
                        help="attention theta, default: 1.0")
    parser.add_argument('--eval', action='store_true',
                        help='skip training')
    parser.add_argument('--demo', action='store_true',
                        help='demo mode')
    parser.add_argument('--stat', action='store_true',
                        help='stat memory choices')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model',
                        help='path to save the final model')
    parser.add_argument('--save_checkpoints', type=int, default=10,
                        help='number of checkpoints remained')
    parser.add_argument('--word_loss', action="store_true",
                        help='output loss of every word')
    parser.add_argument('--compare_farnear', action="store_true",
                        help='compare loss between far and near')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    parser.add_argument('--load_xl', type=str, default='',
                        help='path to load the transformer-xl model')
    parser.add_argument('--rank', type=int, default=0,
                        help='rank in nccl')
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


def train(model, train_loader, valid_loader, criterion, scheduler, 
          args, epoch, step, optimizer, best_eval_score, writer, ema=None):

    model.train()
    criterion.train()
    start_time = time.time()
    total_loss = 0.
    len_train = 0
    module = model.module if args.distributed else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[args.rank]))
    else:
        device = torch.device("cpu")

    vocab = train_loader.dataset.fields["text"].vocab
    pad_token = vocab.stoi["<pad>"]

    params = [p for group in optimizer.param_groups for p in group["params"]]
    
    keys = [None for _ in range(args.update_cycle)]
    values = [None for _ in range(args.update_cycle)]
    mems = [None for _ in range(args.update_cycle)]
    nei_eoss = [None for _ in range(args.update_cycle)]
    mem = None
    cache_info = init_cache_info(args)
    cache_infos = cache_info.chunk(args.update_cycle, dim=1)

    for batch, data in enumerate(train_loader):

        #if not data.text.size(0) == args.num_steps:
        #    continue

        # load data
        text, target = data.text.cuda(), data.target.cuda()
        if args.distributed:
            dist.broadcast(text, 0)
            dist.broadcast(target, 0)
            batch_start, batch_end = batch_division(target.size(1), args.rank)
            text, target = (text[:,batch_start:batch_end], 
                            target[:,batch_start:batch_end])
            if args.sentence_cache:
                eos = data.eos[:,batch_start:batch_end]
        else:
            if args.sentence_cache:
                eos = data.eos

        model.zero_grad()
        criterion.zero_grad()
        texts = text.chunk(args.update_cycle, dim=1)
        targets = target.chunk(args.update_cycle, dim=1)
        if args.sentence_cache:
            eoss = eos.chunk(args.update_cycle, dim=1)
        else:
            eoss = [None for _ in range(args.update_cycle)]

        key_chunks, value_chunks, mem_chunks, info_chunks, nei_eos_chunks = [], [], [], [], []
        for text, target, eos, nei_eos, key, value, mem, cache_info in \
            list(zip(texts, targets, eoss, nei_eoss, keys, values, mems, cache_infos)):

            # train

            len_target = target.reshape(-1).size(0) - torch.nonzero(target.eq(pad_token)).size(0)
            len_train += len_target

            if args.farnear:
                output, hidden, mem = model(text, key, value, 
                                            neighbor_mem=mem, 
                                            cache_info=cache_info)
            else:
                output, hidden = model(text, key, value, cache_info=cache_info)

            if args.sentence_cache:
                # turn indice of eos to sentence length by +1
                if nei_eos is None:
                    total_mem = hidden
                    total_eos = torch.cat((eos.new_zeros(args.neighbor_len, eos.size(1)), (eos + 1)), dim=0)
                else:
                    total_mem = torch.cat((mem.reshape(hidden.size(0), -1, *mem.size()[1:]), hidden), dim=1)
                    rel_eos = eos + nei_eos[-1].expand_as(eos) + 1
                    total_eos = torch.cat((nei_eos, rel_eos), dim=0)
                    
                total_num = total_eos.size(0)
                nei_num = args.neighbor_len
                cache_num = total_num - nei_num

                split_idx = total_eos[cache_num-1]
                hiddens, mems = [], []
                for isp, idx in enumerate(split_idx):
                    batch_mem = total_mem[:,:,isp,:]
                    batch_mem_text = text[:,isp]

                    real_mem_len = total_eos[args.neighbor_len-1,isp] 
                    seq_len = real_mem_len + batch_mem_text.size(0) - batch_mem_text.eq(pad_token).sum()
                    if nei_eos is not None:
                        mem_len = mem.size(0) // hidden.size(0)
                        start = mem_len - real_mem_len
                    else:
                        start = 0
                    hidden_unit, mem_unit = batch_mem[:,start:start+seq_len,:].split([idx, seq_len-idx], dim=1)

                    hiddens.append(hidden_unit)
                    mems.append(mem_unit)
                
                cache_eos = total_eos[:cache_num,:]
                nei_eos = total_eos[cache_num:,:]
                nei_eos = nei_eos - cache_eos[-1].expand_as(nei_eos)

                mem = padding_hidden(mems)
                hidden = padding_cache(hiddens, cache_eos, args.cache_L)

                mem = mem.permute(1, 2, 0, 3).reshape(-1, mem.size(0), mem.size(-1))
                hidden = hidden.permute(2, 0, 1, 3, 4).reshape(hidden.size(2), hidden.size(0), -1, hidden.size(-1))

                # renew cache
                key, value, cache_info = module.cache.renew(hidden, text, cache_info, key, value, 
                                                            cache_eos=cache_eos)

                # distance translation 
                cache_info[:,:,1] += (eos[-1] + 1).expand_as(cache_info[:,:,1]).to(cache_info)
                # update near info
                if args.farnear:
                    # length
                    len_sents = nei_eos - torch.cat((torch.zeros_like(nei_eos[0]).unsqueeze(0), nei_eos[:-1]))
                    cache_info[args.cache_N:,:,0] = len_sents
                    # distance
                    cache_info[args.cache_N:,:,1] = torch.matmul(torch.triu(len_sents.new_ones(len_sents.size(0),
                                                                                               len_sents.size(0))),
                                                                 len_sents)
            else:
                # renew cache
                key, value, cache_info = module.cache.renew(hidden, text, cache_info, key, value)

                # distance translation 
                cache_info[:,:,1] += text.size(0)
                # update near info
                if args.farnear:
                    nei_len = mem.size(0) / (args.nlayers + 1)
                    cache_info[-1,:,:2] = torch.ones_like(cache_info[-1,:,:2]) * nei_len 
            
            key_chunks.append(key)
            value_chunks.append(value)
            mem_chunks.append(mem)
            info_chunks.append(cache_info)
            nei_eos_chunks.append(nei_eos)

            if args.adaptive:
                loss = criterion(output, target, keep_order=True)
            else:
                loss = criterion(output.reshape(-1, args.vocab_size), target.reshape(-1),
                                 reduction="none")

            if args.sentence_cache:
                loss = loss.reshape_as(target)
                pad_mask = target.eq(pad_token)
                loss = loss.masked_fill(pad_mask, 0).sum() / len_target
            else:
                loss = loss.mean() 


            # Activiation Regularization
            if args.alpha:
                loss = loss + args.alpha * output.pow(2).mean()

            # Temporal Activation Regularization (slowness)
            if args.beta:
                loss = loss + args.beta * (output[1:] - output[:-1]).pow(2).mean()

            loss.backward()

            total_loss += loss.item() * len_target

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(1.0 / args.update_cycle)

        torch.nn.utils.clip_grad_norm_(params, args.clip)

        optimizer.step()
        step += 1

        keys, values, mems, cache_infos, nei_eoss = key_chunks, value_chunks, mem_chunks, info_chunks, nei_eos_chunks


        if ema is not None:
            # parameters average
            if args.mu < 0:
                ema_mu = 1 / max(1, step - args.decay_steps)
            else:
                ema_mu = args.mu
            for p in params:
                ema[p].add_(p.data.sub(ema[p]).mul(ema_mu))
        else:
            if step <= args.warmup_steps:
                # warmup steps
                curr_lr = args.lr * step / args.warmup_steps
                optimizer.param_groups[0]['lr'] = curr_lr
                optimizer.param_groups[1]['lr'] = curr_lr * args.emb_mult
            else:
                if args.scheduler == "cosine":
                    scheduler.step()

        if step % args.theta_annealing_steps == 0 and args.theta_annealing_alpha < 1:
            module.theta_annealing_step()
            if args.rank == 0:
                logger.log("STEP {:5d}, annealing theta to {:3.4f}".format(step, module.theta))


        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / len_train
            if args.distributed:
                cur_loss = torch.tensor([cur_loss]).cuda()
                dist.reduce(cur_loss, 0)
                cur_loss = cur_loss.item() / dist.get_world_size()
            elapsed = time.time() - start_time
            if args.rank == 0:
                if args.eval_metric == "ppl":
                    cur_score = math.exp(cur_loss)
                elif args.eval_metric == "bpc":
                    cur_score = cur_loss / math.log(2)
                logger.log('| epoch {:1d} | {:5d}/{:5d} batches | lr {:02.2e} | '
                      'ms/batch {:4.0f} | loss {:4.2f} | {} {:5.2f}'.format(
                    epoch, batch, len(train_loader), 
                    optimizer.param_groups[0]["lr"],
                    elapsed * 1000 / args.log_interval, 
                    cur_loss, args.eval_metric, 
                    cur_score))
            writer.add_scalar("train/" + args.eval_metric, math.exp(cur_loss), 
                                batch + (epoch - 1) * len(train_loader))
            writer.flush()
            total_loss = 0.
            len_train = 0
            start_time = time.time()

        if batch % args.eval_steps == 0 and batch > 0:
            eval_score = evaluate(model, valid_loader, criterion, writer, args)
            if args.rank == 0:
                logger.log('| eval at step {:3d} | eval {} {:5.2f}'.format(batch, args.eval_metric, eval_score))
                save_model(args, module, criterion, str(step), eval_score)
                if eval_score < best_eval_score: 
                    best_eval_score = eval_score
                    save_model(module.args, module, criterion)
                    logger.log("save best model")
                logger.log('-' * 60)
            start_time = time.time()

    if ema is not None:
        return best_eval_score, step, ema
    else:
        return best_eval_score, step


def evaluate(model, eval_loader, criterion, writer, args):

    model.eval()
    criterion.eval()
    module = model.module if args.distributed else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[args.rank]))
    else:
        device = torch.device("cpu")

    vocab = eval_loader.dataset.fields["text"].vocab
    pad_token = vocab.stoi["<pad>"]
    
    total_loss = 0.
    len_eval = 0
    total_len = len(eval_loader)
    key = None                  
    value = None                
    if args.farnear:            
        mem = None              
    cache_info = init_cache_info(args, True)

    nei_eos = None

    if args.word_loss and args.rank == 0:
        vocab = eval_loader.dataset.fields["text"].vocab 
        loss_file = open(args.savepath + "/" + args.save + "_word_loss.pkl", "wb")
        loss_obj = TargetText(batch_size=args.eval_batch_size, 
                              num_steps=args.num_steps)         
        loss_obj.clear()                

    if not args.eval_index == "none":
        assert args.eval_batch_size == 1
        losses = []
        with open(args.eval_index, "r") as f:
            line = f.readline()
            idxs = torch.tensor([int(x) for x in line.split()])
                               
    with torch.no_grad():      
        with tqdm(total=total_len) as pbar:
            pbar.set_description("evaluating")
                               
            for batch, data in enumerate(eval_loader):
                #if not data.text.size(0) == args.num_steps:
                #    pbar.update(1)
                #    continue
                               
                eval_batch_size = data.text.size(1)
                model.set_batch_size(eval_batch_size)

                if args.distributed:
                    batch_start, batch_end = batch_division(eval_batch_size, 
                                                            args.rank)
                    
                    text, target = (data.text[:,batch_start:batch_end].to(device), 
                                     data.target[:,batch_start:batch_end].to(device))
                    if args.sentence_cache:
                        eos = data.eos[:,batch_start:batch_end]
                else:
                    text, target = data.text.to(device), data.target.to(device)
                    if args.sentence_cache:
                        eos = data.eos

                eval_batch_size = text.size(1)
                len_eval += (target.reshape(-1).size(0) - torch.nonzero(target.eq(pad_token)).size(0))

                if args.farnear:
                    if mem is not None:
                        mem = mem.detach()
                    output, hidden, mem = model(text, key, value, 
                                                neighbor_mem=mem, 
                                                cache_info=cache_info)
                else:
                    output, hidden = model(text, key, value, cache_info=cache_info)

                if args.sentence_cache:
                    # turn indice of eos to sentence length by +1
                    if nei_eos is None:
                        total_mem = hidden
                        total_eos = torch.cat((eos.new_zeros(args.neighbor_len, eos.size(1)), (eos + 1)), dim=0)
                    else:
                        total_mem = torch.cat((mem.reshape(hidden.size(0), -1, *mem.size()[1:]), hidden), dim=1)
                        rel_eos = eos + nei_eos[-1].expand_as(eos) + 1
                        total_eos = torch.cat((nei_eos, rel_eos), dim=0)
                        
                    total_num = total_eos.size(0)
                    nei_num = args.neighbor_len
                    cache_num = total_num - nei_num

                    split_idx = total_eos[cache_num-1]
                    hiddens, mems = [], []
                    for isp, idx in enumerate(split_idx):
                        batch_mem = total_mem[:,:,isp,:]
                        batch_mem_text = text[:,isp]

                        real_mem_len = total_eos[args.neighbor_len-1,isp] 
                        seq_len = real_mem_len + batch_mem_text.size(0) - batch_mem_text.eq(pad_token).sum()
                        if nei_eos is not None:
                            mem_len = mem.size(0) // hidden.size(0)
                            start = mem_len - real_mem_len
                        else:
                            start = 0
                        hidden_unit, mem_unit = batch_mem[:,start:start+seq_len,:].split([idx, seq_len-idx], dim=1)

                        hiddens.append(hidden_unit)
                        mems.append(mem_unit)
                    
                    cache_eos = total_eos[:cache_num,:]
                    nei_eos = total_eos[cache_num:,:]
                    nei_eos = nei_eos - cache_eos[-1].expand_as(nei_eos)

                    mem = padding_hidden(mems)
                    hidden = padding_cache(hiddens, cache_eos, args.cache_L)

                    mem = mem.permute(1, 2, 0, 3).reshape(-1, mem.size(0), mem.size(-1))
                    hidden = hidden.permute(2, 0, 1, 3, 4).reshape(hidden.size(2), hidden.size(0), -1, hidden.size(-1))

                    # renew cache
                    key, value, cache_info = module.cache.renew(hidden, text, cache_info, key, value, 
                                                                cache_eos=cache_eos)

                    # distance translation 
                    cache_info[:,:,1] += (eos[-1] + 1).expand_as(cache_info[:,:,1]).to(cache_info)
                    # update near info
                    if args.farnear:
                        # length
                        len_sents = nei_eos - torch.cat((torch.zeros_like(nei_eos[0]).unsqueeze(0), nei_eos[:-1]))
                        cache_info[args.cache_N:,:,0] = len_sents
                        # distance
                        cache_info[args.cache_N:,:,1] = torch.matmul(torch.triu(len_sents.new_ones(len_sents.size(0),
                                                                                                   len_sents.size(0))),
                                                                     len_sents)
                else:
                    # renew cache
                    key, value, cache_info = module.cache.renew(hidden, text, cache_info, key, value)

                    # distance translation 
                    cache_info[:,:,1] += text.size(0)
                    # update near info
                    if args.farnear:
                        nei_len = mem.size(0) / (args.nlayers + 1)
                        cache_info[-1,:,:2] = torch.ones_like(cache_info[-1,:,:2]) * nei_len 

                if args.adaptive:
                    loss_tensor = criterion(output, target,
                                            keep_order=True,
                                            temperature=args.eval_temperature)
                else:
                    loss_tensor = criterion(output.reshape(-1, args.vocab_size), target.reshape(-1), 
                                            reduction="none")

                if args.sentence_cache:
                    loss = loss_tensor.reshape_as(target)
                    pad_mask = target.eq(pad_token)
                    loss = loss.masked_fill(pad_mask, 0).sum()
                else:
                    loss = loss_tensor.sum() 

                if not args.eval_index == "none":
                    losses.append(loss_tensor)

                total_loss += loss.item()

                if args.word_loss:
                    if args.distributed:
                        target_list = [target.new_zeros(target.size(0), batch_division(data.target.size(1), r, single_value=True)) for r in range(dist.get_world_size())]
                        loss_list = [loss_tensor.new_zeros(target.size(0) * batch_division(data.target.size(1), r, single_value=True)) for r in range(dist.get_world_size())]
                        dist.all_gather(target_list, target)
                        dist.all_gather(loss_list, loss_tensor)
                        target = torch.cat(target_list, dim=1)
                        loss_tensor = torch.cat(loss_list, dim=0)
                    if args.rank == 0:
                        words = [vocab.itos[w] for w in target.reshape(-1)]
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
    if not args.eval_index == "none":
        loss = torch.cat(losses, dim=0)
        mean_loss = loss.index_select(0, idxs.cuda()).mean()
        ppl = mean_loss.exp().item()
        bpc = mean_loss.item() / math.log(2)
    else:
        ppl = math.exp(total_loss / len_eval)
        bpc = total_loss / len_eval / math.log(2)
    model.set_batch_size(args.batch_size)
    model.train()
    criterion.train()
    if args.eval_metric == "ppl":
        return ppl
    elif args.eval_metric == "bpc":
        return bpc

def load_dataset(args):
    datatime_begin = time.time()

    corpus_path = "../.data/" + args.datasets + ".pt"

    if args.datasets == "ptb":
        if args.rank == 0:
            logger.log("Loading %s dataset from torchtext" % args.datasets)
        if args.random_seq_len or args.partial_shuffle or args.sentence_cache:
            if os.path.exists(corpus_path):
                corpus = torch.load(corpus_path)
            else:
                corpus = ExistingDataset("ptb", args.num_steps)
                torch.save(corpus, corpus_path)
            if args.random_seq_len:
                train_loader = corpus.randomlen_train_loader(args.batch_size, 
                                                             mem_len=args.neighbor_len,
                                                             partial_shuffled=args.partial_shuffle)
                valid_loader = corpus.get_valid_loader(args.eval_batch_size)
                test_loader = corpus.get_test_loader(args.eval_batch_size)
            elif args.sentence_cache:
                train_loader = corpus.sentence_train_loader(args.batch_size, args.num_steps, 
                                                            partial_shuffled=args.partial_shuffle,
                                                            max_sent_len=args.cache_L)
                valid_loader = corpus.sentence_valid_loader(args.eval_batch_size, args.num_steps,
                                                            max_sent_len=args.cache_L)
                test_loader = corpus.sentence_test_loader(args.eval_batch_size, args.num_steps,
                                                          max_sent_len=args.cache_L)
            else:
                train_loader = corpus.partial_shuffle_loader(args.batch_size)
                valid_loader = corpus.get_valid_loader(args.eval_batch_size)
                test_loader = corpus.get_test_loader(args.eval_batch_size)
        else:
            train_loader, _, _ = torchtext.datasets.PennTreebank.iters(
                    root="../.data",
                    batch_size=args.batch_size, 
                    device=torch.device("cpu"),
                    bptt_len=args.num_steps)
            _, valid_loader, test_loader = torchtext.datasets.PennTreebank.iters(
                    root="../.data",
                    batch_size=args.eval_batch_size, 
                    device=torch.device("cpu"),
                    bptt_len=args.num_steps)
        vocab = train_loader.dataset.fields["text"].vocab
        vocab_size = len(vocab.itos)
    elif args.datasets == "wt2":
        if args.rank == 0:
            logger.log("Loading %s dataset from torchtext" % args.datasets)
        if args.random_seq_len or args.partial_shuffle:
            corpus = ExistingDataset("wt2", args.num_steps)
            if args.random_seq_len:
                train_loader = corpus.randomlen_train_loader(args.batch_size, 
                                                             mem_len=args.neighbor_len,
                                                             partial_shuffled=args.partial_shuffle)
            else:
                train_loader = corpus.partial_shuffle_loader(args.batch_size)
            valid_loader = corpus.get_valid_loader(args.eval_batch_size)
            test_loader = corpus.get_test_loader(args.eval_batch_size)
        else:
            train_loader, _, _ = torchtext.datasets.WikiText2.iters(
                    root="../.data",
                    batch_size=args.batch_size, 
                    device=torch.device("cpu"),
                    bptt_len=args.num_steps)
            _, valid_loader, test_loader = torchtext.datasets.WikiText2.iters(
                    root="../.data",
                    batch_size=args.eval_batch_size, 
                    device=torch.device("cpu"),
                    bptt_len=args.num_steps)
        vocab = train_loader.dataset.fields["text"].vocab
        vocab_size = len(vocab.itos)
    elif args.datasets == "wt103":
        if args.rank == 0:
            logger.log("Loading %s dataset from torchtext" % args.datasets)
        if args.random_seq_len or args.partial_shuffle:
            corpus = ExistingDataset("wt103", args.num_steps)
            if args.random_seq_len:
                train_loader = corpus.randomlen_train_loader(args.batch_size, 
                                                             mem_len=args.neighbor_len,
                                                             partial_shuffled=args.partial_shuffle)
            else:
                train_loader = corpus.partial_shuffle_loader(args.batch_size)
            valid_loader = corpus.get_valid_loader(args.eval_batch_size)
            test_loader = corpus.get_test_loader(args.eval_batch_size)
        else:
            train_loader, _, _ = torchtext.datasets.WikiText103.iters(
                    root="../.data",
                    batch_size=args.batch_size, 
                    device=torch.device("cpu"),
                    bptt_len=args.num_steps)
            _, valid_loader, test_loader = torchtext.datasets.WikiText103.iters(
                    root="../.data",
                    batch_size=args.eval_batch_size, 
                    device=torch.device("cpu"),
                    bptt_len=args.num_steps)
        vocab = train_loader.dataset.fields["text"].vocab
        vocab_size = len(vocab.itos)
    else:
        if args.rank == 0:
            logger.log("Loading data from %s" % args.data)
        corpus = TextDataset(args.data, args.vocab_size, args.num_steps)
        if args.random_seq_len or args.partial_shuffle:
            if args.random_seq_len:
                train_loader = corpus.randomlen_train_loader(args.batch_size, 
                                                             mem_len=args.neighbor_len,
                                                             partial_shuffled=args.partial_shuffle)
            else:
                train_loader = corpus.partial_shuffle_loader(args.batch_size)
        else:
            train_loader = corpus.get_train_loader(args.batch_size)
        valid_loader = corpus.get_valid_loader(args.eval_batch_size)
        test_loader = corpus.get_test_loader(args.eval_batch_size)

        vocab_size = len(corpus.TEXT.vocab.itos)

    datatime_end = time.time()
    datatime = datatime_end - datatime_begin

    return (train_loader, valid_loader, test_loader), vocab_size, datatime


def save_model(args, model, criterion, lattix="best", valid_score=None):
    savename = args.savepath + "/" + args.save + "_" + lattix + ".pt"
    torch.save({
        "model_args": args,
        "model_state_dict": model.state_dict(),
        "criterion": criterion.state_dict()
        }, 
        savename)
    if valid_score is not None:
        with open("saved_model_scores", "a") as f:
            f.write(savename + "\t" + str(valid_score) + "\n")
        maintain_checkpoints(args.save_checkpoints)


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

    writer = SummaryWriter("../log/" + args.save + args.timestr)

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
        args.tie_projs = [False] + [True] * len(args.cutoffs)

    if args.demo:
        args.batch_size = 1
        args.eval_batch_size = 1

    ### Load Data ###
    datasets, vocab_size, data_time = load_dataset(args)
    args.vocab_size = vocab_size
    logger.log("Data loading finished. time: {:.3f} s".format(data_time))
    train_loader, valid_loader, test_loader = datasets

    decay_steps = len(train_loader) * args.std_epochs
    total_steps = len(train_loader) * args.epochs
    args.decay_steps = decay_steps

    if not args.sentence_cache:
        assert args.num_steps >= args.cache_L, "cache_L should <= num_steps"
    else:
        assert args.num_steps <= args.cache_N, "cache_N should >= num_steps in sentence_cache mode"

    if args.load:
        # Load Model
        checkpoint = torch.load(args.load, map_location=device)
        model_args = checkpoint["model_args"]

        model_args.data = args.data
        model_args.demo = args.demo
        model_args.stat = args.stat
        model_args.eval = args.eval
        model_args.load = args.load
        model_args.load_xl = args.load_xl
        model_args.adam = args.adam
        model_args.lr = args.lr
        model_args.warmup_steps = args.warmup_steps
        model_args.emb_mult = args.emb_mult
        model_args.ema_lr_mult = args.ema_lr_mult
        model_args.scheduler = args.scheduler
        model_args.clip = args.clip
        model_args.std_epochs = args.std_epochs
        model_args.ema_epochs = args.ema_epochs
        model_args.mu = args.mu
        model_args.distributed = args.distributed
        model_args.devices = args.devices
        model_args.save = args.save

        model_args.rank = args.rank

        if args.demo:
            batch_size = 1
            model_args.eval_batch_size = 1
        else:
            batch_size = args.batch_size
            model_args.eval_batch_size = args.eval_batch_size

        if not hasattr(model_args, "d_head"):
            model_args.d_head = model_args.nhid // model_args.nhead

        if not model_args.num_steps == args.num_steps:
            logger.log("REDEFINE num_steps: {} --> {}".format(model_args.num_steps, 
                                                         args.num_steps))
            model_args.num_steps = args.num_steps
        if not model_args.neighbor_len == args.neighbor_len:
            logger.log("REDEFINE neighbor_len: {} --> {}".format(model_args.neighbor_len, 
                                                            args.neighbor_len))
            model_args.neighbor_len = args.neighbor_len
        if not model_args.cache_N == args.cache_N:
            logger.log("REDEFINE cache_N: {} --> {}".format(model_args.cache_N, 
                                                       args.cache_N))
            model_args.cache_N = args.cache_N
        if not model_args.cache_k == args.cache_k:
            logger.log("REDEFINE cache_k: {} --> {}".format(model_args.cache_k, 
                                                       args.cache_k))
            model_args.cache_k = args.cache_k
        if not model_args.cache_L == args.cache_L:
            logger.log("REDEFINE cache_L: {} --> {}".format(model_args.cache_L, 
                                                       args.cache_L))
            model_args.cache_L = args.cache_L
        if not model_args.clamp_len == args.clamp_len:
            logger.log("REDEFINE clamp_len: {} --> {}".format(model_args.clamp_len, 
                                                         args.clamp_len))
            model_args.clamp_len = args.clamp_len
        if hasattr(model_args, "cache_theta"):
            if not model_args.cache_theta == args.cache_theta:
                logger.log("REDEFINE cache_theta: {} --> {}".format(model_args.cache_theta, 
                                                         args.cache_theta))
                model_args.cache_theta = args.cache_theta
        else:
            model_args.cache_theta = args.cache_theta

        if hasattr(model_args, "attn_theta"):
            if not model_args.attn_theta == args.attn_theta:
                logger.log("REDEFINE attn_theta: {} --> {}".format(model_args.attn_theta, 
                                                         args.attn_theta))
                model_args.attn_theta = args.attn_theta
        else:
            model_args.attn_theta = args.attn_theta
        model_args.same_length = args.same_length
        model_args.same_length_query = args.same_length_query

        model_args.log_interval = args.log_interval
        model_args.eval_steps = args.eval_steps
        model_args.word_loss = args.word_loss
        model_args.eval_metric = args.eval_metric
        model_args.eval_temperature = args.eval_temperature
        model_args.eval_temp_search = args.eval_temp_search
        model_args.eval_theta_search = args.eval_theta_search
        model_args.eval_index = args.eval_index

        args = model_args

    elif args.load_xl:
        checkpoint = torch.load(args.load_xl, map_location=device)
        weight_keys = list(checkpoint["model_state_dict"].keys())
        for key in weight_keys:
            layers = key.split(".")
            if "lin_qkv" in layers:
                idx = layers.index("lin_qkv")
                weight = checkpoint["model_state_dict"].pop(key)
                weight_q, weight_k, weight_v = weight.chunk(3, dim=0)
                weight_kv = torch.cat((weight_k, weight_v), dim=0)
                layers_q = copy(layers)
                layers_kv = copy(layers)
                layers_q[idx] = "lin_q"
                layers_kv[idx] = "lin_kv"
                checkpoint["model_state_dict"][".".join(layers_q)] = weight_q
                checkpoint["model_state_dict"][".".join(layers_kv)] = weight_kv
        
    args.mem_len = args.cache_k * args.cache_L
    if not args.eval:
        args.theta = (1 / args.theta_annealing_alpha) ** (total_steps // args.theta_annealing_steps)
    else:
        args.theta = 1.0

    #Print Params
    if args.rank == 0:
        for argk, argv in args.__dict__.items():
            logger.log("{}: {}".format(argk, argv))
        logger.log("")



    if args.load:
        # load state_dict

        if args.demo:
            model = CRTNModel(model_args, corpus=corpus)
        else:
            model = CRTNModel(model_args)

        args.batch_size = batch_size

        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_batch_size(batch_size)
    elif args.load_xl:
        model = CRTNModel(args)
        model.encoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        #create model
        if args.demo:
            model = CRTNModel(args, corpus=corpus)
        else:
            model = CRTNModel(args)

        model.apply(init_weights)

    
    if args.adaptive:
        criterion = ProjectedAdaptiveLogSoftmax(args.vocab_size, 
                                                args.emsize, 
                                                args.nhid, 
                                                args.cutoffs, 
                                                div_val=args.div_val, 
                                                init_std=args.init_std,
                                                proj_init_std=args.proj_init_std,
                                                mos=args.mos,
                                                n_experts=args.n_experts,
                                                dropmos=args.dropmos
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
        if args.load or args.load_xl:
            criterion.load_state_dict(checkpoint["criterion"])

    else:
        criterion = nn.CrossEntropyLoss()

    
    emb_param = list(model.encoder.embedding.parameters())
    nonemb_param = [p for p in model.parameters() if not param_in(p, emb_param)] + \
                   [p for p in criterion.parameters() if not param_in(p, emb_param)]
    if args.rank == 0:
        nonemb_param_num = sum([p.numel() for p in nonemb_param])
        emb_param_num = sum([p.numel() for p in emb_param])
        logger.log("#model params = {}".format(nonemb_param_num + emb_param_num))
        logger.log('#non emb params = {}'.format(nonemb_param_num))
        logger.log('#emb params = {}'.format(emb_param_num))

        if args.eval:
            logger.log("SKIP TRAINING")
        else:
            logger.log("TRAINING......")

    model.cuda()
    criterion.cuda()

    param_list = [nonemb_param, emb_param]
    lr_list = [args.lr, args.lr * args.emb_mult]
    if args.adam:
        optimizer = optim.Adam([{"params": p, "lr": lr} for p, lr in zip(param_list, lr_list)], 
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD([{"params": p, "lr": lr} for p, lr in zip(param_list, lr_list)], 
                               weight_decay=args.weight_decay)
    
    if args.distributed:
        model = DistributedDataParallel(model, 
                                        device_ids=[device], 
                                        dim=1)
        model.set_batch_size(args.batch_size)
    
    if args.scheduler == "cosine":
        scheduler_steps = decay_steps - args.warmup_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=scheduler_steps,
                                                         eta_min=args.eta_min)
    elif args.scheduler == "constant":
        scheduler = None


    ### Training ###

    if not args.eval:
        try:
            best_eval_score = float('inf')
            eval_scores = []
            train_step = 0
            ema = dict()
            module = model.module if args.distributed else model
            for epoch in range(1, args.std_epochs+1):
                epoch_start_time = time.time()
                best_eval_score, train_step = train(model, 
                                                  train_loader, 
                                                  valid_loader, 
                                                  criterion,
                                                  scheduler,
                                                  args, 
                                                  epoch,
                                                  train_step,
                                                  optimizer, 
                                                  best_eval_score, 
                                                  writer)

                eval_score = evaluate(model, valid_loader, criterion, writer, args)

                if args.rank == 0:

                    logger.log('-' * 89)

                    logger.log('| end of epoch {:3d} | time: {:5.2f}s | valid {} '
                               '{:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                args.eval_metric, eval_score))
                    # save model
                    save_model(args, module, criterion, str(train_step), eval_score)
                    if eval_score < best_eval_score:
                        # save best model
                        best_eval_score = eval_score
                        save_model(args, module, criterion)
                        logger.log("save best model")
                    logger.log('-' * 89)

                    writer.add_scalar("valid/" + args.eval_metric, eval_score, 
                                      epoch * len(train_loader))
                    writer.flush()
                
                if args.nonmono > 0:
                    if len(eval_scores) > args.nonmono:
                        if eval_score > min(eval_scores[:-args.nonmono]):
                            break
                    eval_scores.append(eval_score)

            ema_start = epoch
            if args.ema_epochs > 0:
                logger.log("Starting EMA at epoch {}".format(epoch))
                for p in chain(model.parameters(), criterion.parameters()):
                    ema[p] = p.data.clone()
                for k in range(len(optimizer.param_groups)):
                    optimizer.param_groups[k]["lr"] *= args.ema_lr_mult

            for epoch in range(ema_start+1, ema_start+args.ema_epochs+1):
                epoch_start_time = time.time()
                best_eval_score, train_step, ema = train(model, 
                                                       train_loader, 
                                                       valid_loader, 
                                                       criterion,
                                                       scheduler,
                                                       args, 
                                                       epoch, 
                                                       train_step,
                                                       optimizer, 
                                                       best_eval_score, 
                                                       writer,
                                                       ema=ema)
                tmp = dict()

                # load ema params
                for prm in chain(model.parameters(), criterion.parameters()):
                    tmp[prm] = prm.data.clone()
                    prm.data.copy_(ema[prm])

                eval_score = evaluate(model, valid_loader, criterion, writer, args)

                if args.rank == 0:
                    logger.log('-' * 89)
                    logger.log('| end of epoch {:3d} | time: {:5.2f}s | valid {} '
                          '{:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           args.eval_metric, eval_score))
                    # save model
                    save_model(args, module, criterion, str(train_step), eval_score)
                    if eval_score < best_eval_score:
                        best_eval_score = eval_score
                        save_model(args, module, criterion)
                        logger.log("save averaged model")

                    logger.log('-' * 89)

                    writer.add_scalar("valid/" + args.eval_metric, eval_score, 
                                      epoch * len(train_loader))
                    writer.flush()                

                # restore params
                for prm in chain(model.parameters(), criterion.parameters()):
                    prm.data.copy_(tmp[prm])


        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

    ### Reload the best model

    if args.rank == 0:
        if args.eval:
            if args.load:
                best_model = args.load
            elif args.load_xl:
                best_model = args.load_xl
        else:
            best_model = args.savepath + "/" + args.save + "_best.pt"

        eval_checkpoint = torch.load(best_model, map_location=device)
        model_state_dict = eval_checkpoint["model_state_dict"]

        module = model.module if args.distributed else model
        if args.load_xl and args.eval:
            weight_keys = list(model_state_dict.keys())
            for key in weight_keys:
                layers = key.split(".")
                if "lin_qkv" in layers:
                    idx = layers.index("lin_qkv")
                    weight = model_state_dict.pop(key)
                    weight_q, weight_k, weight_v = weight.chunk(3, dim=0)
                    weight_kv = torch.cat((weight_k, weight_v), dim=0)
                    layers_q = copy(layers)
                    layers_kv = copy(layers)
                    layers_q[idx] = "lin_q"
                    layers_kv[idx] = "lin_kv"
                    model_state_dict[".".join(layers_q)] = weight_q
                    model_state_dict[".".join(layers_kv)] = weight_kv
 
            module.encoder.load_state_dict(model_state_dict)
        else:
            module.load_state_dict(model_state_dict)

        if args.adaptive:
            criterion.load_state_dict(eval_checkpoint["criterion"])

        logger.log("=" * 89)
        logger.log("experiment name: {}".format(args.save))
        logger.log("saved in: {}".format(os.path.abspath(args.savepath)))

    if args.distributed:
        broadcast(model)
        broadcast(criterion)

    if args.eval_temp_search:
        best_temp_score = float("inf")
        best_temp = 1.0
        logger.log("temperature search")
        for temp in np.arange(1.0, 1.2, 0.02):
            args.eval_temperature = temp
            temp_score = evaluate(model, valid_loader, criterion, writer, args)
            if temp_score < best_temp_score:
                best_temp_score = temp_score
                best_temp = temp
                logger.log("UPDATE best temp {:5.2f} | valid {} {:8.2f}".format(temp, args.eval_metric, 
                                                                                temp_score))
            else:
                break
        args.eval_temperature = best_temp

    if args.eval_theta_search:
        module = model.module if args.distributed else model
        best_atheta_score = float("inf")
        best_atheta = 1.0
        logger.log("attn theta search")
        for atheta in np.arange(1.0, 0.7, -0.02):
            module.set_theta(1.0, atheta)
            atheta_score = evaluate(model, valid_loader, criterion, writer, args)
            if atheta_score < best_atheta_score:
                best_atheta_score = atheta_score
                best_atheta = atheta
                logger.log("UPDATE best attn theta {:5.2f} | valid {} {:8.2f}".format(atheta, args.eval_metric,
                                                                                      atheta_score))
            else:
                break
        module.set_theta(1.0, best_atheta)

    best_eval_score = evaluate(model, valid_loader, criterion, writer, args)

    if args.eval_index == "none":
        test_score = evaluate(model, test_loader, criterion, writer, args)

    if args.rank == 0:
        logger.log('=' * 89)
        if not args.eval_index == "none":
            logger.log('| End of training | best valid {} {:8.2f} on {}'.format(best_eval_score, args.eval_metric,
                                                                                args.eval_index))
        else:
            logger.log('| End of training | best valid {} {:8.2f}'.format(args.eval_metric, best_eval_score))
            logger.log('=' * 89)
            logger.log('| test {} {:8.2f}'.format(args.eval_metric, test_score))
        logger.log('=' * 89)

    if args.distributed:
        dist.destroy_process_group()


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)

def process_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    main(local_args)


if __name__ == "__main__":

    args = parse_args()
    savepath = "."
    timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")

    args.name = "CRTN"
    args.savepath = savepath
    args.timestr = timestr
    args.epochs = args.std_epochs + args.ema_epochs

    
    if not os.path.exists("../log/"):
        os.mkdir("../log/")
    if not os.path.exists("../log/" + args.save + timestr):
        os.mkdir("../log/" + args.save + timestr)
    writer = SummaryWriter("../log/" + args.save + timestr)

    #### Load Data ###
    #datasets, vocab_size, data_time = load_dataset(args)
    #args.vocab_size = vocab_size
    #logger.log("Data loading finished. time: {:.3f} s".format(data_time))

    world_size = len(args.devices)

    if args.distributed:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            args.url = url

        mp.spawn(process_fn, args=(args, ), nprocs=world_size)
    else:
        process_fn(0, args)
