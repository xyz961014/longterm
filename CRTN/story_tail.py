import time
from datetime import datetime
import os
import sys
import re
import argparse
import copy
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
from nltk.translate.bleu_score import sentence_bleu

#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel

from data.tail_loader import TailDataset, ROCDataset
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from utils.visual import TargetText
from models.CRTNModel import CRTNModel

if torch.__version__ < "1.2.0":
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

import ipdb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/writingprompts/toy/',
                        help='location of the data corpus')
    parser.add_argument('--eval', action='store_true',
                        help='skip training')
    parser.add_argument('--demo', action='store_true',
                        help='demo mode')
    parser.add_argument('--stat', action='store_true',
                        help='stat memory choices')
    parser.add_argument('--adam', action='store_true',
                        help='adam optimizer')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='size of vocabulary, excluding special chars')
    parser.add_argument('--trgmax', type=int, default=100,
                        help='max len of generated tail')
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
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
    parser.add_argument('--beam_size', type=int, default=4,
                        help='beam size of beam search when inferencing')
    parser.add_argument('--decode_alpha', type=float, default=0.6,
                        help='length punishment when decoding: ((5+l)/6)^alpha')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, 
                        help='eval batch size')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0.0, init_std)')
    parser.add_argument('--tied', action="store_true",
                        help='tied embedding weights')
    parser.add_argument('--attn_type', type=int, default=1, choices=[0, 1],
                        help='attention type, 0 for vaswani; 1 for transformer-xl')
    parser.add_argument("--cache_N", type=int, default=5, 
                        help="size of Cache, default: 5")
    parser.add_argument("--cache_dk", type=int, default=256, 
                        help="dimension of key, default: 256")
    parser.add_argument("--cache_k", type=int, default=3, 
                        help="select top k values, default: 3")
    parser.add_argument('--multi_gpu', action="store_true",
                        help='enable multiple gpus')
    parser.add_argument('--adaptive', action="store_true",
                        help='use adaptive embedding and softmax')
    parser.add_argument('--cutoffs', type=int, 
                        default=[20000, 40000, 60000], nargs="+",
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
    parser.add_argument("--neighbor_len", type=int, default=20,
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
    parser.add_argument('--eval_steps', type=int, default=2000, metavar='N',
                        help='evaluation steps')
    parser.add_argument('--eval_part', type=float, default=1.0,
                        help='only use a part of validation in eval during training')
    parser.add_argument('--eval_on_train', action="store_true",
                        help='use part of training data to do evaluation')
    parser.add_argument('--eval_bleu', action="store_true",
                        help='compute bleu during evaluation')
    parser.add_argument('--word_loss', action="store_true",
                        help='output loss of every word')
    parser.add_argument('--compare_farnear', action="store_true",
                        help='compare loss between far and near')
    parser.add_argument('--rocstories', action="store_true",
                        help='choose ROCStories task')
    parser.add_argument('--save', type=str, default='model',
                        help='path to save the final model')
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    parser.add_argument('--rank', type=int, default=0,
                        help='rank in nccl')
    args = parser.parse_args()
    return args

#class DataParallel(DistributedDataParallel):
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

def batch_bleu(vocab, pred, trg):
    bleu = 0.
    batch_size = pred.size(1)
    pred_sentences = []
    trg_sentences = []
    for i in range(batch_size):
        pred_s = []
        for w in pred[:,i]:
            if vocab.itos[w] == "<eos>":
                break
            if not vocab.itos[w] == "<pad>":
                pred_s.append(vocab.itos[w])
        pred_sentences.append(pred_s)

        trg_s = []
        for w in trg[:,i]:
            if vocab.itos[w] == "<eos>":
                break
            if not vocab.itos[w] == "<pad>":
                trg_s.append(vocab.itos[w])
        trg_sentences.append(trg_s)

    for p, t in list(zip(pred_sentences, trg_sentences)):
        bleu += sentence_bleu([t], p)

    return (bleu, [" ".join(p) for p in pred_sentences], 
                  [" ".join(t) for t in trg_sentences])

def get_real_ind_and_prob(head_prob, tails, beam_size, padding_idx=1):
    tail_len = len(tails)
    get_tails = []
    for i in range(tail_len):
        base_prob = head_prob[:,-i-1].unsqueeze(1)
        get_tails.append(tails[i] + base_prob)
    real_prob = torch.cat((head_prob[:,:-tail_len], *get_tails), 1)
    real_prob[:,padding_idx] = -float("inf")
    word_prob, word_ind = real_prob.topk(beam_size)
    return word_ind, word_prob

def variance_prob(head_prob, tails):
    tail_len = len(tails)
    get_tails = []
    for i in range(tail_len):
        base_prob = head_prob[:,-i-1].unsqueeze(1)
        get_tails.append(tails[i] + base_prob)
    real_prob = torch.cat((head_prob[:,:-tail_len], *get_tails), 1)
    var = torch.var(real_prob, 1)
    return var

def beam_search(candidates, criterion, vocab, block, block_start, ind, model, args, update=False):
    """
    content in candidates:
        generated word indices
        sequence probability
        sequence eos
        nlayers representation of predicted words block
        key
        valueoutput, memory = model(ppl_block, memory)

                outputs = torch.cat((outputs, output), 0)

                for trg_ppl_block in trg_lasts:
                    output, memory = model(trg_ppl_block, memory)

                    outputs = torch.cat((outputs, output), 0)

                for batch, tail_len in enumerate(tail_lens):
                    batch_pred = outputs[ind-1:ind+tail_len-1,batch,:]
                    batch_trg = trg[:tail_len,batch]
                    loss_tensor = criterion(batch_pred, batch_trg, keep_order=True)
                    head_prob, tail_probs = criterion(batch_pred, 
                                                      batch_trg, 
                                                      keep_order=True,
                                                      output=True)
                    variances = variance_prob(head_prob, tail_probs)


        neighbor_mem
        key_num
        output of prediction of word
    """
    module = model.module if args.multi_gpu else model

    # record all searched results
    ind_tensor = block.new_ones(0)
    eos_tensor = block.new_ones(0).bool() # True represents sentence is not end.
    prob_tensor = candidates[0][-1].new_zeros(0)
    blocks_tensor = prob_tensor.new_zeros(0)
    keys_tensor = prob_tensor.new_zeros(0)
    values_tensor = prob_tensor.new_zeros(0)
    mems_tensor = prob_tensor.new_zeros(0)
    key_nums_tensor = prob_tensor.new_zeros(0)

    def length_punish(alpha, length):
        return ((5 + length) / 6) ** alpha

    for old_inds, old_probs, old_eos, old_block, old_key, old_value, old_mem, old_key_num, out in candidates:
        # get probablity
        head_prob, tails = criterion(out, block[0], output=True)
        word_ind, word_prob = get_real_ind_and_prob(head_prob, tails, args.beam_size)
        # process eos 
        word_ind.masked_fill_(old_eos.eq(False), 1)
        word_prob.masked_fill_(old_eos.eq(False), -float("inf"))
        # remain only one eos
        if old_eos.eq(False).sum().item() > 0:
            for eos_ind, eos_it in enumerate(old_eos):
                if not eos_it:
                    word_prob[eos_ind][0] = 0.0
        # concat old ind and new ones
        old_inds = old_inds.unsqueeze(1).expand(-1, args.beam_size, -1)
        word_inds = word_ind.unsqueeze(-1)
        word_inds = torch.cat((old_inds, word_inds), -1)
        # decode alpha scaling
        scale_len = word_inds.size(-1)
        return_tensor = torch.ones_like(word_prob) * length_punish(args.decode_alpha, 
                                                                   scale_len - 1)
        scale_tensor = torch.ones_like(word_prob) / length_punish(args.decode_alpha, 
                                                                  scale_len)
        return_tensor.masked_fill_(old_eos.eq(False), 1.0)
        scale_tensor.masked_fill_(old_eos.eq(False), 1.0)
        word_prob = old_probs.mul(return_tensor) + word_prob
        word_prob.mul_(scale_tensor)
        # mark eos
        eos = old_eos * word_ind.eq(vocab.stoi["<eos>"]).eq(False)
        # record current result
        ind_tensor = torch.cat((ind_tensor, word_inds), 1)
        prob_tensor = torch.cat((prob_tensor, word_prob), 1)
        eos_tensor = torch.cat((eos_tensor, eos), 1)
        # accumulate old tensors to gather from
        old_block = old_block.unsqueeze(0)
        blocks_tensor = torch.cat((blocks_tensor, old_block), 0)
        old_key = old_key.unsqueeze(0)
        keys_tensor = torch.cat((keys_tensor, old_key), 0)
        old_value = old_value.unsqueeze(0)
        values_tensor = torch.cat((values_tensor, old_value), 0)
        old_mem = old_mem.unsqueeze(0)
        mems_tensor = torch.cat((mems_tensor, old_mem), 0)
        old_key_num = old_key_num.unsqueeze(0)
        key_nums_tensor = torch.cat((key_nums_tensor, old_key_num), 0)

    chosen_prob, chosen_ind = prob_tensor.topk(args.beam_size)
    chosen_eos = torch.gather(eos_tensor, 1, chosen_ind)
    chosen_cand = chosen_ind.t() / args.beam_size
    gather_id = chosen_ind.unsqueeze(-1).expand(-1, -1, ind_tensor.size(-1))
    chosen_ind = torch.gather(ind_tensor, 1, gather_id)
        
    # gather corresponding tensors 
    block_ind = chosen_cand[:,None,:,None,None]
    block_ind = block_ind.expand(-1, blocks_tensor.size(1), -1, blocks_tensor.size(-2),
                                 blocks_tensor.size(-1))
    chosen_inf_blocks = torch.gather(blocks_tensor, 0, block_ind)

    key_ind = chosen_cand[:,None,:,None]
    key_ind = key_ind.expand(-1, keys_tensor.size(1), -1, keys_tensor.size(-1))
    chosen_keys = torch.gather(keys_tensor, 0, key_ind)

    value_ind = chosen_cand[:,None,:,None,None]
    value_ind = value_ind.expand(-1, values_tensor.size(1), -1, values_tensor.size(-2),
                                 values_tensor.size(-1))
    chosen_values = torch.gather(values_tensor, 0, value_ind)

    mem_ind = chosen_cand[:,None,None,:,None]
    mem_ind = mem_ind.expand(-1, mems_tensor.size(1), mems_tensor.size(2), -1,
                             mems_tensor.size(-1))
    chosen_mems = torch.gather(mems_tensor, 0, mem_ind)

    key_num_ind = chosen_cand[:,None,:]
    key_num_ind = key_num_ind.expand(-1, key_nums_tensor.size(1), -1)
    chosen_key_nums = torch.gather(key_nums_tensor, 0, key_num_ind)

    # demo: print words in batch0
    #for words, prob in list(zip(chosen_ind[0], chosen_prob[0])):
    #    print(" ".join([vocab.itos[w] for w in words]), end=" ")
    #    print("%.4f" % prob.item())
    #print("")

    new_candidates = []
    for i in range(args.beam_size): 
        inf_blocks = chosen_inf_blocks[i]
        key = chosen_keys[i]
        value = chosen_values[i]
        mem = chosen_mems[i]
        key_num = chosen_key_nums[i]
        eos = chosen_eos[:,i]

        cand = [chosen_ind[:,i,:], 
                chosen_prob[:,i].unsqueeze(-1), 
                chosen_eos[:,i].unsqueeze(-1)]

        block[block_start:ind+1] = chosen_ind[:,i,block_start-ind-1:].t()
        
        if not eos.eq(False).sum() == eos.size(0):
            if args.farnear:

                output, hidden, new_mem = model(block, key, value, neighbor_mem=mem, 
                                                key_num=key_num, inf_ind=ind, 
                                                inf_blocks=inf_blocks)

                #output, hidden, new_mem = model(block, key, value, neighbor_mem=mem, 
                #                                key_num=key_num)
                #output = output[ind].unsqueeze(0)
                #update_hidden = hidden.clone()
                #hidden = new_mem.view_as(mem)[:,ind-args.num_steps,:,:].unsqueeze(1)
                #hidden = hidden.transpose(1, 2)
            else:
                output, hidden = model(block, key, value, key_num=key_num)

            hidden = hidden.transpose(1, 2)
            new_inf_blocks = inf_blocks.clone()
            new_inf_blocks[:,:,ind,:] = hidden.squeeze(1)

            if update:
                if args.farnear:
                    mem = mem.reshape(args.nlayers+1, args.neighbor_len, -1, args.nhid)
                    total_mem = torch.cat((mem, new_inf_blocks.transpose(1, 2)), 1)
                    update_hidden, new_mem = total_mem.split([args.num_steps, 
                                                              args.neighbor_len], 
                                                              dim=1)
                module, key_num, key, value = update_cache(module, block.size(1), 
                                                          key, value, update_hidden, 
                                                          block, key_num)
                mem = new_mem.view_as(mem)

            cand.append(new_inf_blocks)
            cand += [key, value, mem, key_num]
            cand.append(output.squeeze(0))
        else:
            cand += [inf_blocks.clone(), key, value, mem, key_num,
                     inf_blocks.new_zeros(inf_blocks.size(1), inf_blocks.size(-1))]
        new_candidates.append(cand)
    
    return new_candidates 

def save_pred(savepath, name, preds, trgs):
    with open(savepath + "/" + name + ".pred", "w") as fw:
        for p in preds:
            fw.write(p)
            fw.write("\n")
    with open(savepath + "/" + name + ".trg", "w") as fw:
        for t in trgs:
            fw.write(t)
            fw.write("\n")

def train(model, train_loader, valid_loader, criterion, 
          args, epoch, optimizer, scheduler, best_eval_ppl):

    model.train()
    start_time = time.time()
    total_loss = 0.
    module = model.module if args.multi_gpu else model
    #module.encoder.embedding.emb_layers[0].weight[vocab.stoi["<pad>"]].zero_()
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.devices[0]))
    else:
        device = torch.device("cpu")
    
    key = None
    value = None
    if args.farnear:
        mem = None
    key_num = init_key_num(args, device)

    for batch, data in enumerate(train_loader):
        text, target = data.text.to(device), data.target.to(device)
        if not text.size(0) == args.num_steps:
            continue

        model.zero_grad()
        
        if args.farnear:
            if mem is not None:
                mem = mem.detach()
            if args.compare_farnear:
                output, hidden, mem, near_output = model(text, key, value, 
                                                         neighbor_mem=mem, 
                                                         key_num=key_num)
            else:
                output, hidden, mem = model(text, key, value, neighbor_mem=mem, 
                                                              key_num=key_num)
        else:
            output, hidden = model(text, key, value, key_num=key_num)

        module, key_num, key, value = update_cache(module, args.batch_size, 
                                                   key, value, hidden, text, key_num)

        if args.adaptive:
            loss = criterion(output.reshape(-1, args.nhid), target.reshape(-1))
            loss = loss.mean()
        else:
            loss = criterion(output.reshape(-1, args.vocab_size), target.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if args.scheduler == "cosine":
            scheduler.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:1d} |{:5d}/{:5d} batches | lr {:02.2e} | '
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
            (eval_bleu, 
             eval_ppl, 
             eval_preds, 
             eval_trgs,
             eval_nearppl) = evaluate(model, 
                                      valid_loader, 
                                      criterion, 
                                      args, 
                                      args.eval_part)
            print('| eval at step {:3d} | eval bleu {:5.2f} |'
                  ' eval ppl {:5.2f}'.format(batch, 
                                             eval_bleu * 100,
                                             eval_ppl))
            save_pred(savepath, 
                      "eval_" + str(epoch) + "_" + str(batch // args.eval_steps), 
                      eval_preds, eval_trgs)
            if eval_ppl < best_eval_ppl: 
                best_eval_ppl = eval_ppl
                torch.save({
                    "model_args": module.args,
                    "model_state_dict": module.state_dict(),
                    "criterion": criterion.state_dict()
                    }, 
                    savepath + "/" + args.save + "_best.pt")
                print("save best model for better ppl")
            print('-' * 60)
            start_time = time.time()
    
    return best_eval_ppl

def evaluate(model, eval_loader, criterion, args, eval_part=1.0):
    model.eval()
    module = model.module if args.multi_gpu else model
    # add eos token
    eval_loader.dataset.fields["trg"].eos_token = "<eos>"

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.devices[0]))
    else:
        device = torch.device("cpu")

    losses = 0.
    near_losses = 0.
    if args.word_loss:
        loss_file = open(savepath + "/" + args.save + "_word_loss.pkl", "wb")
        loss_obj = TargetText()
        loss_obj.clear()

    
    bleu = 0.
    len_eval = 0
    total_len = math.ceil(len(eval_loader) * eval_part)

    vocab = eval_loader.dataset.fields["trg"].vocab
    pad_idx = vocab.stoi["<pad>"]
    preds = []
    trgs = []


    with torch.no_grad():
        with tqdm(total=total_len) as pbar:
            pbar.set_description("evaluating")

            for batch, data in enumerate(eval_loader):
                if batch >= total_len:
                    break
                src, trg = data.src.to(device), data.trg.to(device)
                eval_batch_size = src.size(1)
                len_eval += eval_batch_size
                srcs = src.split(module.args.num_steps)

                model.set_batch_size(eval_batch_size)
                key_num = init_key_num(args, device, True)
                key = None
                value = None
                pred = []
                if args.farnear:
                    mem = None

                # load history
                for i, block in enumerate(srcs):
                    if args.farnear:
                        if mem is not None:
                            mem = mem.detach()
                        
                        if args.compare_farnear:
                            output, hidden, mem, near_output = model(block, 
                                                                     key, 
                                                                     value, 
                                                                     neighbor_mem=mem,
                                                                     key_num=key_num)
                        else:
                            output, hidden, mem = model(block, key, value, 
                                                        neighbor_mem=mem, 
                                                        key_num=key_num)
                    else:
                        output, hidden = model(block, key, value, key_num=key_num)

                    if i < len(srcs) - 1:
                        module, key_num, key, value = update_cache(module, 
                                                                   eval_batch_size, 
                                                                   key, value, 
                                                                   hidden, 
                                                                   block, 
                                                                   key_num)

                # begin to evaluate

                hidden = hidden.transpose(1, 2)

                if args.farnear:
                    mem = mem.reshape(args.nlayers+1, args.neighbor_len, -1, 
                                      args.nhid)
                    total_mem = torch.cat((hidden, mem), 1)
                    mem, inf_blocks = total_mem.split([args.neighbor_len, 
                                                           args.num_steps], 
                                                           dim=1)
                if args.eval_bleu:
                    inf_blocks = inf_blocks.transpose(1, 2)

                    candidates = [[block.new_ones(eval_batch_size, 0), 
                                  output.new_zeros(eval_batch_size, 1),
                                  block.new_ones(eval_batch_size, 1).bool(),
                                  inf_blocks,
                                  key, value, mem, key_num]]

                    # generated index
                    # probability
                    # eos
                    # previous input


                for ind in range(args.num_steps):
                    if block[ind][0].item() == pad_idx:
                        break
                else:
                    ind += 1
                block_start = ind

                if args.eval_bleu:
                    candidates[0].append(output[ind-1])

                # compute ppl of trg

                ppl_block = block.clone()
                outputs = output.new_zeros(0)
                trg_len = trg.size(0)
                tail_lens = []
                if args.compare_farnear:
                    near_outputs = output.new_zeros(0)

                for story in trg.t():
                    for wid, w in enumerate(story):
                        if w.item() == vocab.stoi["<eos>"]:
                            tail_lens.append(wid + 1)
                            break
                if trg_len - args.num_steps + ind > 0:
                    trg_fill, trg_last = trg.split([args.num_steps - ind, 
                                                    trg_len - args.num_steps + ind])
                    ppl_block[ind:] = trg_fill
                    trg_lasts = list(trg_last.split(args.num_steps))
                    last_trg_len = trg_lasts[-1].size(0)
                    trg_lasts[-1] = torch.cat((trg_lasts[-1], 
                                               trg_lasts[-1].new_ones(
                                                   args.num_steps - last_trg_len, 
                                                   eval_batch_size)))
                else:
                    ppl_block[ind:ind+trg_len] = trg
                    trg_lasts = []
                if args.farnear:
                    if args.compare_farnear:
                        output, hidden, mem, near_output = model(ppl_block, 
                                                                 key, 
                                                                 value, 
                                                                 neighbor_mem=mem, 
                                                                 key_num=key_num)
                        near_outputs = torch.cat((near_outputs, near_output), 0)
                    else:
                        output, hidden, mem = model(ppl_block, key, value, 
                                                    neighbor_mem=mem, 
                                                    key_num=key_num)
                outputs = torch.cat((outputs, output), 0)

                module, key_num, key, value = update_cache(module, 
                                                           eval_batch_size, 
                                                           key, value, hidden, 
                                                           ppl_block, 
                                                           key_num)
                for trg_ppl_block in trg_lasts:
                    if args.compare_farnear:
                        output, hidden, mem, near_output = model(trg_ppl_block, 
                                                                 key, 
                                                                 value, 
                                                                 neighbor_mem=mem, 
                                                                 key_num=key_num)
                        near_outputs = torch.cat((near_outputs, near_output), 0)
                    else:
                        output, hidden, mem = model(trg_ppl_block, key, value, 
                                                    neighbor_mem=mem, 
                                                    key_num=key_num)

                    outputs = torch.cat((outputs, output), 0)

                    module, key_num, key, value = update_cache(module, 
                                                               eval_batch_size, 
                                                               key, value, hidden, 
                                                               ppl_block, 
                                                               key_num)
                for batch, tail_len in enumerate(tail_lens):
                    batch_pred = outputs[ind-1:ind+tail_len-1,batch,:]
                    batch_trg = trg[:tail_len,batch]
                    if args.compare_farnear:
                        near_batch_pred = near_outputs[ind-1:ind+tail_len-1,batch,:]
                        near_loss_tensor = criterion(near_batch_pred, 
                                                     batch_trg, 
                                                     keep_order=True)
                        near_loss = near_loss_tensor.mean()
                        near_losses += near_loss.item()
                    loss_tensor = criterion(batch_pred, 
                                            batch_trg, 
                                            keep_order=True)
                    loss = loss_tensor.mean()
                    losses += loss.item()

                    if args.word_loss:
                        head_prob, tail_probs = criterion(batch_pred, 
                                                          batch_trg, 
                                                          keep_order=True,
                                                          output=True)
                        variances = variance_prob(head_prob, tail_probs)
                        variances = [v.item() for v in variances]

                        words = [vocab.itos[w] for w in batch_trg]    
                        word_loss = [l.item() for l in loss_tensor] 
                        if args.compare_farnear:
                            near_word_loss = [l.item() for l in near_loss_tensor] 
                        else:
                            near_word_loss = []
                        loss_obj.add_words(words)
                        loss_obj.add_losss(word_loss)
                        loss_obj.add_variances(variances)
                        loss_obj.add_near_losss(near_word_loss)


                # end of ppl


                if args.eval_bleu:
                    # complete unfilled block
                    while ind < args.num_steps:
                        candidates = beam_search(candidates, criterion, vocab, block, 
                                                 block_start, ind, model, args,
                                                 ind == args.num_steps - 1)

                        ind += 1

                    # start new blocks
                    step = 0
                    block_start = 0
                    block = torch.ones_like(block) * vocab.stoi["<pad>"]
                    while step < args.trgmax - args.num_steps:
                        ind = step % args.num_steps
                        candidates = beam_search(candidates, criterion, vocab, block, 
                                                 block_start, ind, model, args, 
                                                 ind == args.num_steps - 1)
                        eos_bool = torch.cat([c[2] for c in candidates], 0)
                        if eos_bool.equal(torch.zeros_like(eos_bool)):
                            break
                        step += 1

                    final_ind = torch.cat([x[0].unsqueeze(0) for x in candidates], 0)
                    final_prob = torch.cat([x[1] for x in candidates], 1)
                    _, max_ind = final_prob.max(1)
                    max_ind = max_ind.unsqueeze(-1).expand(-1, final_ind.size(-1)).unsqueeze(0)
                    pred_tensor = torch.gather(final_ind, 0, max_ind).squeeze(0).t()
                    b_bleu, b_pred, b_trg = batch_bleu(vocab, pred_tensor, trg)
                    bleu += b_bleu
                    preds += b_pred
                    trgs += b_trg

                pbar.update(1)


    loss_mean = losses / len_eval
    ppl = math.exp(loss_mean)
    near_loss_mean = near_losses / len_eval
    near_ppl = math.exp(near_loss_mean)
    if args.word_loss:
        pkl.dump(loss_obj, loss_file)
        loss_file.close()

    model.train()
    model.set_batch_size(args.batch_size)
    return bleu / len_eval, ppl, preds, trgs, near_ppl

def roc_evaluate(model, eval_loader, criterion, args):
    model.eval()
    module = model.module if args.multi_gpu else model

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.devices[0]))
    else:
        device = torch.device("cpu")

    vocab = eval_loader.dataset.fields["InputText"].vocab
    pad_idx = vocab.stoi["<pad>"]
    total_len = len(eval_loader)
    len_eval = 0
    corrects = []

    with torch.no_grad():
        with tqdm(total=total_len) as pbar:
            pbar.set_description("testing")

            for batch, data in enumerate(eval_loader):
                (src, ans1, ans2), ansidx = data
                src, ans1, ans2 = src.to(device), ans1.to(device), ans2.to(device)
                ansidx = ansidx.to(device)
                eval_batch_size = src.size(1)
                len_eval += eval_batch_size
                srcs = src.split(module.args.num_steps)

                model.set_batch_size(eval_batch_size)
                key_num = init_key_num(args, device, True)
                key = None
                value = None

                if args.farnear:
                    mem = None

                # load history
                for i, block in enumerate(srcs):
                    if args.farnear:
                        if mem is not None:
                            mem = mem.detach()
                        output, hidden, mem = model(block, key, value, 
                                                    neighbor_mem=mem, 
                                                    key_num=key_num)
                    else:
                        output, hidden = model(block, key, value, key_num=key_num)

                    if i < len(srcs) - 1:
                        module, key_num, key, value = update_cache(module, 
                                                                   eval_batch_size, 
                                                                   key, value, 
                                                                   hidden, 
                                                                   block, 
                                                                   key_num)

                # begin to evaluate

                hidden = hidden.transpose(1, 2)

                if args.farnear:
                    mem = mem.reshape(args.nlayers+1, args.neighbor_len, -1, 
                                      args.nhid)
                    total_mem = torch.cat((hidden, mem), 1)
                    mem, inf_blocks = total_mem.split([args.neighbor_len, 
                                                       args.num_steps], dim=1)
 
                for ind in range(args.num_steps):
                    if block[ind][0].item() == pad_idx:
                        break
                else:
                    ind += 1
                block_start = ind


                # compute ppl of cands

                losses = []
                for trg in [ans1, ans2]:
                    ppl_block = block.clone()
                    key_num_clone= key_num.clone()
                    key_clone = key.clone()
                    value_clone = value.clone()
                    mem_clone = mem.clone()
                    outputs = output.new_zeros(0)
                    trg_len = trg.size(0)
                    tail_lens = []
                    loss_list = []
                    for story in trg.t():
                        for wid, w in enumerate(story):
                            if w.item() == vocab.stoi["<eos>"]:
                                tail_lens.append(wid + 1)
                                break
                    if trg_len - args.num_steps + ind > 0:
                        trg_fill, trg_last = trg.split(
                                [args.num_steps - ind, 
                                 trg_len - args.num_steps + ind])
                        ppl_block[ind:] = trg_fill
                        trg_lasts = list(trg_last.split(args.num_steps))
                        last_trg_len = trg_lasts[-1].size(0)
                        trg_lasts[-1] = torch.cat((trg_lasts[-1], 
                                                   trg_lasts[-1].new_ones(
                                                       args.num_steps - last_trg_len, 
                                                       eval_batch_size)))
                    else:
                        ppl_block[ind:ind+trg_len] = trg
                        trg_lasts = []
                    if args.farnear:
                        output, hidden, mem_clone = model(ppl_block, 
                                                          key_clone, 
                                                          value_clone, 
                                                          neighbor_mem=mem_clone, 
                                                          key_num=key_num_clone)
                    outputs = torch.cat((outputs, output), 0)

                    module, key_num_clone, key_clone, value_clone = update_cache(
                                                               module, 
                                                               eval_batch_size, 
                                                               key_clone, 
                                                               value_clone, 
                                                               hidden, 
                                                               ppl_block, 
                                                               key_num_clone)
                    for trg_ppl_block in trg_lasts:
                        output, hidden, mem_clone = model(trg_ppl_block, 
                                                          key_clone, 
                                                          value_clone, 
                                                          neighbor_mem=mem_clone, 
                                                          key_num=key_num_clone)

                        outputs = torch.cat((outputs, output), 0)

                        module, key_num_clone, key_clone, value_clone = update_cache(
                                                                   module, 
                                                                   eval_batch_size, 
                                                                   key_clone, 
                                                                   value_clone, 
                                                                   hidden, 
                                                                   ppl_block, 
                                                                   key_num_clone)
                    for batch, tail_len in enumerate(tail_lens):
                        batch_pred = outputs[ind-1:ind+tail_len-1,batch,:]
                        batch_trg = trg[:tail_len,batch]
                        loss_tensor = criterion(batch_pred, 
                                                batch_trg, 
                                                keep_order=True)
                        loss_list.append(loss_tensor.mean().unsqueeze(0))
                    losses.append(torch.cat(loss_list, 0))

                predict = losses[0].gt(losses[1]) + 1
                correct = predict.eq(ansidx)

                corrects.append(correct)

                pbar.update(1)

        corrects = torch.cat(corrects, 0)
        accuracy = corrects.sum().item() / len_eval

    model.train()
    model.set_batch_size(args.batch_size)
    return accuracy


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

    print("Loading data from %s" % args.data)
    datatime_begin = time.time()

    
    if args.rocstories:
        rocdata = ROCDataset(args.data, args.vocab_size, args.num_steps)
        args.vocab_size = len(rocdata.TRG.vocab.itos)
        train_loader = rocdata.get_train_loader(args.batch_size)
        valid_loader = rocdata.get_valid_loader(args.eval_batch_size)
        test_loader = rocdata.get_test_loader(args.eval_batch_size)
        dicriminate_loader = rocdata.get_discriminate_loader(args.eval_batch_size)
    else:
        corpus = TailDataset(args.data, args.vocab_size, args.num_steps)
        args.vocab_size = len(corpus.TRG.vocab.itos)

        train_loader = corpus.get_train_loader(args.batch_size)
        train_valid_loader = corpus.get_train_valid_loader(args.eval_batch_size)
        valid_loader = corpus.get_valid_loader(args.eval_batch_size)
        test_loader = corpus.get_test_loader(args.eval_batch_size)


    datatime_end = time.time()

    if args.load:
        # Load Model
        checkpoint = torch.load(args.load, map_location=devices[0])
        model_args = checkpoint["model_args"]

        # inject params for this time
        model_args.data = args.data
        model_args.demo = args.demo
        model_args.stat = args.stat
        model_args.eval = args.eval
        model_args.beam_size = args.beam_size
        model_args.decode_alpha = args.decode_alpha
        model_args.load = args.load
        model_args.adam = args.adam
        model_args.lr = args.lr
        model_args.scheduler = args.scheduler
        model_args.clip = args.clip
        model_args.epochs = args.epochs
        model_args.multi_gpu = args.multi_gpu
        model_args.devices = args.devices
        model_args.save = args.save

        batch_size = args.batch_size
        model_args.eval_batch_size = args.eval_batch_size

        model_args.log_interval = args.log_interval
        model_args.eval_steps = args.eval_steps
        model_args.eval_part = args.eval_part
        model_args.eval_on_train = args.eval_on_train
        model_args.eval_bleu = args.eval_bleu
        model_args.word_loss = args.word_loss
        model_args.compare_farnear = args.compare_farnear
        model_args.rocstories = args.rocstories

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
            best_eval_bleu = 0.
            best_eval_ppl = float("inf")
            best_eval_preds, best_eval_trgs = [], []
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                best_eval_ppl = train(model, train_loader, valid_loader, criterion, 
                                      args, epoch, optimizer, scheduler, 
                                      best_eval_ppl)

                if args.eval_on_train:
                    (eval_bleu, 
                     eval_ppl, 
                     eval_preds, 
                     eval_trgs,
                     eval_nearppl) = evaluate(model, 
                                              train_valid_loader,
                                              criterion, 
                                              args,
                                              args.eval_part)
                else:
                    (eval_bleu, 
                     eval_ppl, 
                     eval_preds, 
                     eval_trgs,
                     eval_nearppl) = evaluate(model, valid_loader, criterion, args)

                save_pred(savepath, "eval_" + str(epoch), eval_preds, eval_trgs)

                print('-' * 89)
                print('| end of epoch {:3d}'
                      ' | time: {:5.2f}s '.format(epoch,
                                                  time.time() - epoch_start_time),
                      end="")
                print('| valid ppl {:5.2f} '.format(eval_ppl), end="")
                if args.compare_farnear:
                    print('| valid near ppl {:5.2f} '.format(eval_nearppl))
                if args.eval_bleu:
                    print('| valid bleu {:5.2f} '.format(eval_bleu * 100))

                writer.add_scalar("valid/bleu", eval_bleu, 
                                  epoch * len(train_loader))
                writer.flush()

                module = model.module if args.multi_gpu else model

                torch.save({
                    "model_args": module.args,
                    "model_state_dict": module.state_dict(),
                    "criterion": criterion.state_dict()
                    }, 
                    savepath + 
                    "/" + args.save + "_" + str(epoch) + ".pt")

                # 0 if lower ppl then save model
                # 1 if higher bleu then save model
                if eval_ppl < best_eval_ppl:
                    best_eval_ppl = eval_ppl
                    torch.save({
                        "model_args": module.args,
                        "model_state_dict": module.state_dict(),
                        "criterion": criterion.state_dict()
                        }, 
                        savepath + "/" + args.save + "_best.pt")
                    print("save best model for better ppl")
                    if eval_bleu > best_eval_bleu:
                        best_eval_bleu = eval_bleu
                    best_eval_preds, best_eval_trgs = eval_preds, eval_trgs
                    # save prediction
                    save_pred(savepath, "eval_best", best_eval_preds, best_eval_trgs)
                else:
                    if eval_bleu > best_eval_bleu:
                        torch.save({
                            "model_args": module.args,
                            "model_state_dict": module.state_dict(),
                            "criterion": criterion.state_dict()
                            }, 
                            savepath + "/" + args.save + "_best.pt")
                        print("save best model for better bleu")
                        best_eval_bleu = eval_bleu
                        best_eval_preds, best_eval_trgs = eval_preds, eval_trgs
                        # save prediction
                        save_pred(savepath, "eval_best", best_eval_preds, 
                                  best_eval_trgs)
                print('-' * 89)

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

    if args.eval_on_train:
        (best_eval_bleu, 
         best_eval_ppl, 
         best_eval_preds, 
         best_eval_trgs,
         best_eval_nearppl) = evaluate(model, 
                                       train_valid_loader,
                                       criterion, 
                                       args,
                                       args.eval_part)
    else:
        (best_eval_bleu, 
         best_eval_ppl, 
         best_eval_preds, 
         best_eval_trgs,
         best_eval_nearppl) = evaluate(model, valid_loader, criterion, args)

    save_pred(savepath, "eval_best", best_eval_preds, best_eval_trgs)

    print('=' * 89)
    print('| End of training '
          '| best valid ppl {:5.2f} '.format(best_eval_ppl), end="")
    if args.compare_farnear:
        print('| best valid near ppl {:5.2f} '.format(best_eval_nearppl))
    else:
        print("")
    if args.eval_bleu:
        print('| best valid bleu {:5.2f} '.format(best_eval_bleu * 100))

    print('=' * 89)

    if args.rocstories:
        test_accuracy = roc_evaluate(model, dicriminate_loader, criterion, args) 
        print('| ROCStories test accuracy {:5.2f} % |'.format(test_accuracy * 100))
        print('=' * 89)

    (test_bleu, 
     test_ppl, 
     test_preds, 
     test_trgs, 
     test_nearppl) = evaluate(model, test_loader, criterion, args)

    # save prediction
    save_pred(savepath, "test", test_preds, test_trgs)
    print('| test ppl {:5.2f} '.format(test_ppl), end="")
    if args.compare_farnear:
        print('| test near ppl {:5.2f} '.format(test_nearppl))
    else:
        print("")
    if args.eval_bleu:
        print('| test bleu {:5.2f} '.format(test_bleu * 100))
    print('=' * 89)




if __name__ == "__main__":

    args = parse_args()
    savepath = "../../../experiment/crtn/story/"
    timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")
    savepath += args.save + timestr

    #os.environ["RANK"] = str(args.rank)
    #os.environ["WORLD_SIZE"] = str(len(args.devices))
    #os.environ["MASTER_ADDR"] = "127.0.0.1"
    #os.environ["MASTER_PORT"] = "23456"
    #dist.init_process_group("nccl")
    
    if not os.path.exists("./log/" + args.save + timestr):
        os.mkdir("./log/" + args.save + timestr)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    writer = SummaryWriter("./log/" + args.save + timestr)

    main(args)
