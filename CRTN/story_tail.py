import time
from datetime import datetime
import os
import sys
import re
import argparse
#ignore future warning from tensorboard
import warnings
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

from data.wp_loader import WPDataset
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
                        default='/home/xyz/Documents/Dataset/writingpromts/toy/',
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
                        help='attention type, 0 for vaswani; 1 for transformer-xl')
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
    parser.add_argument('--cutoffs', type=int, 
                        default=[20000, 40000, 80000], nargs="+",
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

def update_cache(model, batch_size, key, value, hidden, text, key_num):
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

    for batch, data in enumerate(train_loader):
        text, target = data.text, data.target
        if not text.size(0) == args.num_steps:
            continue

        model.zero_grad()
        
        if args.farnear:
            if mem is not None:
                mem = mem.detach()
            output, hidden, mem, key_num = model(text, key, value, 
                                                       neighbor_mem=mem, 
                                                       key_num=key_num)
        else:
            output, hidden, key_num = model(text, key, value, key_num=key_num)

        model, key_num, key, value = update_cache(model, args.batch_size, key, value, 
                                                  hidden, text, key_num)

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
    model.eval()
    module = model.module if args.multi_gpu else model
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(module.args.devices[0]))
    else:
        device = torch.device("cpu")
    
    bleu = 0.
    len_eval = 0

    vocab = eval_loader.dataset.fields["trg"].vocab
    preds = []
    trgs = []

    with torch.no_grad():
        for batch, data in enumerate(eval_loader):
            src, trg = data.src, data.trg
            eval_batch_size = src.size(1)
            len_eval += eval_batch_size
            srcs = src.split(module.args.num_steps)

            model.set_batch_size(eval_batch_size)
            key_num = init_key_num(args, device)
            key = None
            value = None
            pred = []
            numofeos = [0] * eval_batch_size
            if args.farnear:
                mem = None

            for i, block in enumerate(srcs):
                if args.farnear:
                    if mem is not None:
                        mem = mem.detach()
                    output, hidden, mem, key_num = model(block, key, value, 
                                                         neighbor_mem=mem, 
                                                         key_num=key_num)
                else:
                    output, hidden, key_num = model(block, key, value, key_num=key_num)

                if i < args.num_steps - 1:
                    model, key_num, key, value = update_cache(model, 
                                                              eval_batch_size, 
                                                              key, value, hidden, 
                                                              block, 
                                                              key_num)

            for ind in range(args.num_steps):
                if block[ind][0].item() == 1:
                    break
            while ind < args.num_steps - 1:
                out_ind = output[ind-1]
                head_prob, tails = criterion(out_ind, block[ind-1], output=True)
                word_ind = head_prob.max(1)[1]
                for i, wid in enumerate(word_ind):
                    if wid >= args.cutoffs[0]:
                        cluster = wid - args.cutoffs[0]
                        real_wid = (tails[cluster][i,:].max(0)[1] 
                                    + args.cutoffs[cluster])
                        word_ind[i] = real_wid
                    else:
                        real_wid = wid
                    word = vocab.itos[real_wid]
                    if word == "<eos>":
                        numofeos[i] = 1
                pred.append(word_ind)
                block[ind] = word_ind
                if args.farnear:
                    output, hidden, mem, key_num = model(block, key, value, 
                                                         neighbor_mem=mem, 
                                                         key_num=key_num)
                else:
                    output, hidden, key_num = model(block, key, value, key_num=key_num)
                ind += 1

            model, key_num, key, value = update_cache(model, eval_batch_size, 
                                                      key, value, hidden, block, 
                                                      key_num)
            step = 0
            block = torch.ones_like(block)
            while step < args.trgmax - args.num_steps:
                ind = step % args.num_steps

                out_ind = output[ind-1]
                head_prob, tails = criterion(out_ind, block[ind-1], output=True)
                word_ind = head_prob.max(1)[1]
                for i, wid in enumerate(word_ind):
                    if wid >= args.cutoffs[0]:
                        cluster = wid - args.cutoffs[0]
                        real_wid = (tails[cluster][i,:].max(0)[1] 
                                    + args.cutoffs[cluster])
                        word_ind[i] = real_wid
                    else:
                        real_wid = wid
                    word = vocab.itos[real_wid]
                    if word == "<eos>":
                        numofeos[i] = 1
                pred.append(word_ind)
                block[ind] = word_ind
                if sum(numofeos) >= eval_batch_size:
                    break
                if args.farnear:
                    output, hidden, mem, key_num = model(block, key, value, 
                                                         neighbor_mem=mem, 
                                                         key_num=key_num)
                else:
                    output, hidden, key_num = model(block, key, value, key_num=key_num)
                if ind == args.num_steps - 1: 
                    model, key_num, key, value = update_cache(model, 
                                                              eval_batch_size, 
                                                              key, value, hidden, 
                                                              block, 
                                                              key_num)
                    block = torch.ones_like(block)

                step += 1
            pred_tensor = torch.cat(pred, 0).view(-1, eval_batch_size)
            b_bleu, b_pred, b_trg = batch_bleu(vocab, pred_tensor, trg)
            bleu += b_bleu
            preds += b_pred
            trgs += b_trg

    model.set_batch_size(args.batch_size)
    return bleu / len_eval, preds, trgs
 

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    savepath = "../../../experiment/crtn/story/"

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

    if args.load:
        # Load Model
        checkpoint = torch.load(args.load)
        model_args = checkpoint["model_args"]

        # inject params for this time
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

        model_args.batch_size = args.batch_size
        model.eval_batch_size = args.eval_batch_size

        args = model_args

    args.mem_len = args.cache_k * args.num_steps

    #Print Params
    for argk, argv in args.__dict__.items():
        print("{}: {}".format(argk, argv))
    print("")
 
    ### Load Data ###

    print("Loading data from %s" % args.data)
    datatime_begin = time.time()

    corpus = WPDataset(args.data, args.vocab_size, args.num_steps)
    args.vocab_size = len(corpus.TRG.vocab.itos)
    
    train_loader = corpus.get_train_loader(args.batch_size, device=devices[0])
    valid_loader = corpus.get_valid_loader(args.eval_batch_size, device=devices[0])
    test_loader = corpus.get_test_loader(args.eval_batch_size, device=devices[0])


    print("Data loading finished. time: {:.3f} s".format(time.time() - datatime_begin))

    if args.eval:
        print("SKIP TRAINING")
    else:
        print("TRAINING......")

    if args.load:
        # load state_dict
        # TODO: may do some operation to clear cache

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
            best_eval_bleu = -float('inf')
            best_eval_preds, best_eval_trgs = [], []
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train(model, train_loader, criterion, args, epoch, optimizer, 
                      scheduler)
                eval_bleu, eval_preds, eval_trgs = evaluate(model, valid_loader, 
                                                            criterion, args)
                with open(savepath + args.save + args.timestr 
                          + "/eval_" + str(epoch) + ".pred", "w") as fw:
                    for p in eval_preds:
                        fw.write(p)
                        fw.write("\n")
                with open(savepath + args.save + args.timestr 
                          + "/eval_" + str(epoch) + ".trg", "w") as fw:
                    for t in eval_trgs:
                        fw.write(t)
                        fw.write("\n")
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '
                      '| valid bleu {:5.2f} |'.format(epoch, 
                                                      (time.time() - epoch_start_time),
                                                      eval_bleu * 100))
                print('-' * 89)
                writer.add_scalar("valid/bleu", eval_bleu, 
                                  epoch * len(train_loader))
                writer.flush()
                if eval_bleu > best_eval_bleu:
                    module = model.module if args.multi_gpu else model
                    torch.save({
                        "model_args": module.args,
                        "model_state_dict": module.state_dict(),
                        "criterion": criterion.state_dict()
                        }, 
                        savepath + args.save + args.timestr + 
                        "/" + args.save + "_best" + ".pt")
                    best_eval_bleu = eval_bleu
                    best_eval_preds, best_eval_trgs = eval_preds, eval_trgs
                    # save prediction
                    with open(savepath + args.save + args.timestr 
                              + "/eval_best.pred", "w") as fw:
                        for p in best_eval_preds:
                            fw.write(p)
                            fw.write("\n")
                    with open(savepath + args.save + args.timestr 
                              + "/eval_best.trg", "w") as fw:
                        for t in best_eval_trgs:
                            fw.write(t)
                            fw.write("\n")

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

    model.load_state_dict(model_state_dict, strict=False)

    if args.adaptive:
        criterion.load_state_dict(eval_checkpoint["criterion"])

    if args.eval:
        best_eval_bleu, best_eval_preds, best_eval_trgs = evaluate(model, valid_loader,
                                                                   criterion, args)

    print('=' * 89)
    print('| best valid bleu {:5.2f} |'.format(best_eval_bleu * 100))
    print('=' * 89)

    test_bleu, test_preds, test_trgs = evaluate(model, test_loader, criterion, args)

    # save prediction
    with open(savepath + args.save + args.timestr 
              + "/test_best.pred", "w") as fw:
        for p in test_preds:
            fw.write(p)
            fw.write("\n")
    with open(savepath + args.save + args.timestr 
              + "/test_best.trg", "w") as fw:
        for t in test_trgs:
            fw.write(t)
            fw.write("\n")
    print('| End of training | test bleu {:5.2f} |'.format(test_bleu * 100))
    print('=' * 89)






if __name__ == "__main__":
    args = parse_args()
    savepath = "../../../experiment/crtn/story/"
    args.timestr = "-" + datetime.now().__format__("%Y%m%d%H%M%S")
    
    if not os.path.exists("./log/" + args.save + args.timestr):
        os.mkdir("./log/" + args.save + args.timestr)
    if not os.path.exists(savepath + args.save + args.timestr):
        os.mkdir(savepath + args.save + args.timestr)
    writer = SummaryWriter("./log/" + args.save + args.timestr)
    main(args)
