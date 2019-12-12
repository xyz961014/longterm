import time
from datetime import datetime
import re
import argparse

#ignore future warning from tensorboard
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from torch.utils.data import DataLoader

import os
import sys
sys.path.append("..")
from data import dataloader
from utils.adaptive import ProjectedAdaptiveLogSoftmax
from models.CRTNModel import CRTNModel

if torch.__version__ < "1.2.0":
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='/home/xyz/Documents/Dataset/ptb_sample',
                        help='location of the data corpus')
    parser.add_argument('--data_from_torchtext', action='store_true',
                        help='load data from torchtext')
    parser.add_argument('--dataset', type=str, default='ptb',
                        help='data corpus name')
    parser.add_argument('--eval', action='store_true',
                        help='skip training')
    parser.add_argument('--demo', action='store_true',
                        help='demo mode')
    parser.add_argument('--stat', action='store_true',
                        help='stat memory choices')
    parser.add_argument('--adam', action='store_true',
                        help='adam optimizer')
    parser.add_argument('--emsize', type=int, default=240,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=240,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=15,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='number of heads')
    parser.add_argument('--d_ff', type=int, default=1300,
                        help='dimension of feed-forward')
    parser.add_argument('--lr', type=float, default=27e-5,
                        help='initial learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'constant'],
                        help='lr scheduler to use')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, 
                        help='eval batch size')
    parser.add_argument('--num_steps', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.45,
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
    parser.add_argument('--load', type=str, default='',
                        help='path to load the model')
    args = parser.parse_args()
    return args

class DataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids, output_device, dim)

    def set_batch_size(self, batch_size):
        batch_size = batch_size // len(self.device_ids)
        self.module.set_batch_size(batch_size)
        self.batch_size = batch_size

def init_key_num(args, device):
    key_num = torch.arange(args.cache_N, 0, -1, 
                           dtype=torch.float,
                           device=device)
    key_num = key_num.expand(args.batch_size, -1)
    key_num.transpose_(0, 1)
    return key_num

def update_cache(model, args, key, value, hidden, text, key_num, evaluate=False):
    batch_size = args.eval_batch_size if evaluate else args.batch_size
    model.cache.set_batch_size(batch_size)
    model.cache.init_key_and_value(key, value)
    model.cache.detach_memory()
    key_num = model.cache.renew(hidden, text, key_num)
    key, value = (model.cache._get_keys(), 
                  model.cache._get_values().transpose(1, 2))
    model.cache.set_batch_size(batch_size // len(args.devices))
    return model, key_num, key, value


def train(model, train_loader, criterion, args, epoch, optimizer, scheduler):
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
    #    mem = torch.zeros((args.nlayers+1)*args.neighbor_len, module.args.batch_size, 
    #                      args.nhid, device=device)

    for batch, data in enumerate(train_loader):
        if not args.data_from_torchtext:
            text, targets = data
            text, targets = text.t(), targets.t()
        else:
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


        module, key_num, key, value = update_cache(module, args, key, value, 
                                                   hidden, text, key_num)
        #module.cache.set_batch_size(args.batch_size)
        #module.cache.init_key_and_value(key, value)
        #module.cache.detach_memory()
        #key_num = module.cache.renew(hidden, text, key_num)
        #key, value = (module.cache._get_keys(), 
        #              module.cache._get_values().transpose(1, 2))
        #module.cache.set_batch_size(args.batch_size // len(args.devices))


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
            print('| epoch {:2d} | {:3d}/{:3d} batches | lr {:02.2e} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_loader), 
                optimizer.state_dict()["param_groups"][0]["lr"],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            writer.add_scalar("train/ppl", math.exp(cur_loss), 
                                batch + (epoch - 1) * len(train_loader))
            writer.flush()
            total_loss = 0.
            start_time = time.time()


def evaluate(model, eval_loader, criterion, args):
    model.set_batch_size(args.eval_batch_size)
    model.eval()
    total_loss = 0.
    module = model.module if args.multi_gpu else model
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[0]))
    else:
        device = torch.device("cpu")
    
    key = None
    value = None
    len_eval = len(eval_loader)

    if args.farnear:
        mem = None
    key_num = init_key_num(args, device)
    #    mem = torch.zeros((args.nlayers+1)*args.neighbor_len, args.eval_batch_size, 
    #                      args.nhid, device=device)
    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            if not args.data_from_torchtext:
                text, targets = data
                text, targets = text.t(), targets.t()
            else:
                text, targets = data.text, data.target
                if not text.size(0) == args.num_steps:
                    len_eval -= 1
                    continue
            text, targets = text.to(device), targets.to(device)
                

            if args.farnear:
                if mem is not None:
                    mem = mem.detach()
                output, hidden, mem = model(text, key, value, 
                                                   neighbor_mem=mem, 
                                                   key_num=key_num)
            else:
                output, hidden = model(text, key, value, key_num=key_num)

            module, key_num, key, value = update_cache(module, args, key, value, 
                                                       hidden, text, key_num, 
                                                       evaluate=True)
            #module.cache.set_batch_size(args.eval_batch_size)
            #module.cache.init_key_and_value(key, value)
            #module.cache.detach_memory()
            #key_num = module.cache.renew(hidden, text, key_num)
            #key, value = (module.cache._get_keys(), 
            #              module.cache._get_values().transpose(1, 2))
            #module.cache.set_batch_size(args.eval_batch_size // len(args.devices))

            #if args.farnear:
            #    if mem is not None:
            #        mem = mem.detach()
            #    output, mem, key_num, (key, value) = model(text, key, value, 
            #                                               neighbor_mem=mem, 
            #                                               key_num=key_num)
            #else:
            #    output, key_num, (key, value) = model(text, key, value, 
            #                                          key_num=key_num)

            if args.adaptive:
                loss = criterion(output.view(-1, args.nhid), targets.view(-1))
                loss = loss.mean()
            else:
                loss = criterion(output.view(-1, args.vocab_size), targets.view(-1))

            total_loss += loss

    model.set_batch_size(args.batch_size)

    return total_loss / len_eval



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    savepath = "../../../experiment/crtn/save/"

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        devices = [torch.device("cuda:" + str(i)) for i in args.devices]

        args.batch_size = len(devices) * (args.batch_size // len(devices))
        args.eval_batch_size = len(devices) * (args.eval_batch_size // len(devices))
    else:
        devices = [torch.device("cpu")]

    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        if args.dataset == "ptb":
            cutoffs = [2000, 4000, 8000]
            tie_projs += [True] * 3
        elif args.dataset == "wt103":
            cutoffs = [20000, 40000, 80000]
            tie_projs += [True] * 3

    if args.demo:
        args.batch_size = 1
        args.eval_batch_size = 1

    args.cutoffs = cutoffs
    args.tie_projs = tie_projs

    if args.load:
        checkpoint = torch.load(args.load)
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
        model_args.save = args.save
        model_args.devices = args.devices
        if args.demo:
            model_args.batch_size = 1
            model_args.eval_batch_size = 1
        args = model_args
        
    args.mem_len = args.cache_k * args.num_steps

    #Print Params
    for argk, argv in args.__dict__.items():
        print("{}: {}".format(argk, argv))
    print("")
        
    ### Load Data ###
    
    if not args.data_from_torchtext:
        print("Loading data from %s" % args.data)
        datatime_begin = time.time()

        corpus = dataloader.Corpus(args.data)
        args.vocab_size = corpus.vocabulary.num_words
        eval_batch_size = args.eval_batch_size


        train_loader = corpus.get_train_loader(batch_size=args.batch_size, 
                                               num_steps=args.num_steps)
        valid_loader = corpus.get_valid_loader(batch_size=eval_batch_size, 
                                               num_steps=args.num_steps)
        test_loader = corpus.get_test_loader(batch_size=eval_batch_size, 
                                             num_steps=args.num_steps)


        print("Data loading finished. time: {:.3f} s".format(time.time() - datatime_begin))
        print("# VOCABULARY: {} \n# train data words: {:.2e} \n# valid data words: {:.2e} \n# test data words: {:.2e} ".format(
            corpus.vocabulary.num_words, 
            len(corpus.train_data.raw_data), 
            len(corpus.valid_data.raw_data), 
            len(corpus.test_data.raw_data)))
        print("")
    else:
        train_loader, _, _ = torchtext.datasets.PennTreebank.iters(
                batch_size=args.batch_size, 
                bptt_len=args.num_steps, 
                device=devices[0])
        _, valid_loader, test_loader = torchtext.datasets.PennTreebank.iters(
                batch_size=args.eval_batch_size, 
                bptt_len=args.num_steps, 
                device=devices[0])
        vocab = train_loader.dataset.fields["text"].vocab
        args.vocab_size = len(vocab.itos)
    if args.eval:
        print("SKIP TRAINING")
    else:
        print("TRAINING......")


    if args.load:
        # clear cache
        keys = checkpoint["model_state_dict"].copy().keys()
        for key in keys:
            if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key) or re.match(r"encoder.pos_emb_bank", key):
                popitem = checkpoint["model_state_dict"].pop(key)

        if args.demo:
            model = CRTNModel(model_args, corpus=corpus)
        else:
            model = CRTNModel(model_args)

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
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


    if not args.eval:
        try:
            best_eval_loss = float('inf')
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train(model, train_loader, criterion, args, epoch, optimizer, 
                      scheduler)
                eval_loss = evaluate(model, valid_loader, criterion, args)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, 
                                                   (time.time() - epoch_start_time),
                                                   eval_loss, math.exp(eval_loss)))
                print('-' * 89)
                writer.add_scalar("valid/ppl", math.exp(eval_loss), 
                                  epoch * len(train_loader))
                writer.flush()
                if eval_loss < best_eval_loss:
                    module = model.module if args.multi_gpu else model
                    torch.save({
                        "model_args": module.args,
                        "model_state_dict": module.state_dict(),
                        "criterion": criterion.state_dict()
                        }, 
                        savepath + args.save + args.timestr + 
                        "/" + args.save + "_best" + ".pt")
                    #with open("save/" + args.save + "/" + args.save + "_best.pt", "wb") as f:
                    #    torch.save(model, f)
                    #with open("save/" + args.save + "/" + args.save + "_crit.pt", "wb") as f:
                    #    torch.save(criterion, f)
                    best_eval_loss = eval_loss

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    ### Reload the best model

    if args.eval:
        best_model = args.load
    else:
        best_model = savepath + args.save + args.timestr + "/" + args.save + "_best.pt"
    eval_checkpoint = torch.load(best_model)
    model_state_dict = eval_checkpoint["model_state_dict"]
    keys = model_state_dict.copy().keys()

    for key in keys:
        if re.match(r"cache.keys", key) or re.match(r"cache.values", key) or re.match(r"cache.words", key):
            model_state_dict.pop(key)

    model.load_state_dict(model_state_dict, strict=False)

    if args.adaptive:
        criterion.load_state_dict(eval_checkpoint["criterion"])

    if args.eval:
        best_eval_loss = evaluate(model, valid_loader, criterion, args)

    test_loss = evaluate(model, test_loader, criterion, args)
    #writer.add_embedding(model.encoder.embedding.emb_layers[0].weight, 
    #                     corpus.vocabulary.index2word.values())
    print('=' * 89)
    print('| best valid loss {:5.2f} | best valid ppl {:8.2f}'.format(
          best_eval_loss, math.exp(best_eval_loss)))
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
          test_loss, math.exp(test_loss)))
    print('=' * 89)





if __name__ == "__main__":
    args = parse_args()
    savepath = "../../../experiment/crtn/save/"
    args.timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")
    
    if not os.path.exists("./log/" + args.save + args.timestr):
        os.mkdir("./log/" + args.save + args.timestr)
    if not os.path.exists(savepath + args.save + args.timestr):
        os.mkdir(savepath + args.save + args.timestr)
    writer = SummaryWriter("./log/" + args.save + args.timestr)
    main(args)
