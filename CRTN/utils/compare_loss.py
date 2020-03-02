import argparse
import os
import sys
import ipdb
import numpy as np
import pickle as pkl
from pprint import pprint
from tqdm import tqdm
import visdom
sys.path.append("..")
sys.path.append("../..")
from data.tail_loader import TailDataset
from CRTN.utils.visual import TargetText
vis = visdom.Visdom()
assert vis.check_connection()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_loss", type=str, help="location of baseline loss")
    parser.add_argument("--model_loss", type=str, help="location of model loss")
    parser.add_argument("--data", type=str, help="location of data",
                        default="/home/xyz/Documents/Dataset/writingprompts/medium/")
    parser.add_argument("--smooth_window", type=int, default=50, 
                        help="smooth window DEFAULT 50")
    parser.add_argument("--loss_window", type=float, default=0.2, 
                        help="loss window DEFAULT 0.2")
    parser.add_argument("--ppl_window", type=int, default=[10, 100, 1000, 10000], 
                        nargs="+", help="ppl windows")
    parser.add_argument("--func", type=int, choices=[0, 1, 2, 3], default=0, 
                        help="function, 0 for stat bag of word loss, 1 for observe variance and loss of model, 2 for observe word freq and word loss, word freq derived from data, 3 for freq window ppl evaluation")
    return parser.parse_args()

def main(args):

    if args.func == 0:
        with open(args.baseline_loss, "rb") as base_file:
            base_obj = pkl.load(base_file)
        with open(args.model_loss, "rb") as model_file:
            model_obj = pkl.load(model_file)
        opts_diff = {
                'legend': ['loss difference'],
                'showlegend': True,
               }
        opts_bar = {
                'legend': [],
                'showlegend': True,
               }
        loss_dict = dict()
        for i in range(len(base_obj)):
            if base_obj.words[i] == model_obj.words[i]:
                word = base_obj.words[i]
                lossb = base_obj.loss[i]
                lossm = model_obj.loss[i]
                if word in loss_dict.keys():
                    loss_dict[word][0].append(lossb)
                    loss_dict[word][1].append(lossm)
                    if lossm < lossb:
                        loss_dict[word][2] += 1
                else:
                    loss_dict[word] = [[lossb], [lossm]]
                    if lossm < lossb:
                        loss_dict[word].append(1)
                    else:
                        loss_dict[word].append(0)

        word_losses = loss_dict.items()
        word_losses = sorted(word_losses, key=lambda x: np.mean(x[1][0]))

        j = 0
        loss_bars = []
        loss_sum_bars = []
        x_labels = []
        for i in range(round(word_losses[-1][1][0][0] / args.loss_window) + 1):
            if j >= len(word_losses):
                break
            lossb_window = []
            lossm_window = []
            while np.mean(word_losses[j][1][0]) < (i + 1) * args.loss_window:
                lossb = np.mean(word_losses[j][1][0])
                lossb_window += word_losses[j][1][0]
                lossm_window += word_losses[j][1][1]
                j += 1
                if j >= len(word_losses):
                    break
            if len(lossb_window) > 0:
                loss_diff = np.mean(lossb_window) - np.mean(lossm_window)
                loss_bars.append(loss_diff)
                loss_diff_sum = np.sum(lossb_window) - np.sum(lossm_window)
                loss_sum_bars.append(loss_diff_sum)
            else:
                loss_bars.append(0.)
                loss_sum_bars.append(0.)
            x_labels.append(i * args.loss_window)

        vis.bar(np.array(loss_bars), np.array(x_labels), win="loss bar",
                opts={"title": "loss difference"})
        vis.bar(np.array(loss_sum_bars), np.array(x_labels), win="loss sum bar",
                opts={"title": "loss sum difference"})

        #for i in tqdm(range(len(word_losses) - args.smooth_window + 1)):
        #    lossb = np.mean(word_losses[i][1][0])
        #    lossb_window = [x[1][0] for x in word_losses[i:i+args.smooth_window]]
        #    lossm_window = [x[1][1] for x in word_losses[i:i+args.smooth_window]]

        #    loss_diff = []
        #    model_better, total = 0, 0
        #    for lbs, lms in list(zip(lossb_window, lossm_window)):
        #        for lb, lm in list(zip(lbs, lms)):
        #            loss_diff.append(lb - lm)
        #            if lb > lm:
        #                model_better += 1
        #        total += len(lbs)
        #    loss_diff = np.mean(loss_diff)
        #    prob_model_better = model_better / total
        #    vis.line(np.array([[loss_diff]]), np.array([lossb]), opts=opts_diff, win="loss diff", update="append")


            




        base_file.close()
        model_file.close()
    elif args.func == 1:
        loss_var = []
        with open(args.model_loss, "rb") as model_file:
            model_obj = pkl.load(model_file)

            loss_var = list(zip(model_obj.loss, model_obj.var))
            
            loss_var = sorted(loss_var, key=lambda x:x[0])
            for i in tqdm(range(len(loss_var) - args.smooth_window + 1)):
                loss = loss_var[i][0]
                var = np.mean([x[1] for x in loss_var[i:i+args.smooth_window]])
                vis.line(np.array([[var]]), np.array([[loss]]), 
                         win="loss-variance", update="append")
    elif args.func == 2:
        freq_loss = []
        corpus = TailDataset(args.data, 1e6, 50)
        vocab = corpus.TEXT.vocab
        with open(args.model_loss, "rb") as model_file:
            model_obj = pkl.load(model_file)

            for l, r in list(zip(model_obj.loss, model_obj.words)):
                freq_loss.append((r, vocab.freqs[r], l))
            
            freq_loss = sorted(freq_loss, key=lambda x:x[1])
            for i in tqdm(range(0, len(freq_loss) - args.smooth_window + 1)):
                freq = freq_loss[i][1]
                loss = np.mean([x[2] for x in freq_loss[i:i+args.smooth_window]])
                vis.line(np.array([[loss]]), np.array([[freq]]), 
                         win="freq-loss", update="append")
    elif args.func == 3:
        # freq window ppl
        opts = {
                "title": "freq window ppl",
                "legend": ["baseline", "model"],
                "stacked": False
                }
        ppl_windows = args.ppl_window
        ppl_windows = [0] + ppl_windows
        corpus = TailDataset(args.data, 1e6, 50)
        vocab = corpus.TEXT.vocab

        freq_loss = []
        base_freq_loss = []
        freq_labels = []
        for i in range(len(ppl_windows)):
            freq_loss.append([])
            base_freq_loss.append([])
            if i < len(ppl_windows) - 1:
                freq_labels.append("{}-{}".format(ppl_windows[i], ppl_windows[i+1]))
            else:
                freq_labels.append("{}-".format(ppl_windows[i]))

        def freq_window(freq, ppl_windows):
            for idx, ppl in enumerate(ppl_windows):
                if freq < ppl:
                    return idx - 1
            else:
                return len(ppl_windows) - 1

        def list_ppl(x):
            if len(x) > 0:
                return np.exp(np.mean(x))
            else:
                return 0

        with open(args.model_loss, "rb") as model_file:
            model_obj = pkl.load(model_file)
        with open(args.baseline_loss, "rb") as base_file:
            base_obj = pkl.load(base_file)


        for i in tqdm(range(len(base_obj))):
            if base_obj.words[i] == model_obj.words[i]:
                word = base_obj.words[i]
                lossb = base_obj.loss[i]
                lossm = model_obj.loss[i]
                freq = vocab.freqs[word]
                if freq > 0:
                    freq_idx = freq_window(freq, ppl_windows)
                    freq_loss[freq_idx].append(lossm)
                    base_freq_loss[freq_idx].append(lossb)

        freq_ppl = list(map(lambda x: list_ppl(x), freq_loss))
        base_freq_ppl = list(map(lambda x: list_ppl(x), base_freq_loss))
        
        bar_array = np.array(base_freq_ppl + freq_ppl).reshape(2, len(freq_ppl)).transpose()

        vis.bar(np.array(bar_array), np.array(freq_labels), 
                win="freq window ppl", opts=opts)


if __name__ == "__main__":
    args = parse_args()
    main(args)

