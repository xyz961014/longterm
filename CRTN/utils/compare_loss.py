import argparse
import os
import sys
import ipdb
import numpy as np
from pprint import pprint
from tqdm import tqdm
import visdom
sys.path.append("..")
from data.wp_loader import WPDataset
vis = visdom.Visdom()
assert vis.check_connection()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_loss", type=str, help="location of baseline loss")
    parser.add_argument("--model_loss", type=str, help="location of model loss")
    parser.add_argument("--data", type=str, help="location of data")
    parser.add_argument("--smooth_window", type=int, default=50, 
                        help="smooth window DEFAULT 50")
    parser.add_argument("--loss_window", type=float, default=0.2, 
                        help="loss window DEFAULT 0.2")
    parser.add_argument("--func", type=int, choices=[0, 1, 2], default=0, 
                        help="function, 0 for stat bag of word loss, 1 for observe variance and loss of model, 2 for observe word freq and word loss, word freq derived from data")
    return parser.parse_args()

def main(args):

    if args.func == 0:
        base_file = open(args.baseline_loss, "r")
        model_file = open(args.model_loss, "r")
        opts_diff = {
                'legend': ['loss difference'],
                'showlegend': True,
               }
        opts_bar = {
                'legend': [],
                'showlegend': True,
               }
        loss_dict = dict()
        while True:
            base_ref = base_file.readline()
            base_loss = base_file.readline()
            base_var = base_file.readline()
            model_ref = model_file.readline()
            model_loss = model_file.readline()
            model_var = model_file.readline()

            if base_ref.strip() == "":
                break

            # remove \n
            base_ref = base_ref.strip()
            base_loss = base_loss.strip()
            model_ref = model_ref.strip()
            model_loss = model_loss.strip()

            if base_ref == model_ref:
                base_ref = base_ref.split()
                base_loss = [float(l) for l in base_loss.split()]
                model_loss = [float(l) for l in model_loss.split()]
                for word, lossb, lossm in list(zip(base_ref, base_loss, model_loss)):
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

        vis.bar(np.array(loss_bars), np.array(x_labels), win="loss bar")
        vis.bar(np.array(loss_sum_bars), np.array(x_labels), win="loss sum bar")

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
        with open(args.model_loss, "r") as model_file:
            while True:
                model_ref = model_file.readline()
                model_loss = model_file.readline()
                model_var = model_file.readline()
                
                if model_ref.strip() == "":
                    break

                model_loss = model_loss.strip()
                model_var = model_var.strip()

                model_loss = [float(l) for l in model_loss.split()]
                model_var = [float(l) for l in model_var.split()]

                for l, v in list(zip(model_loss, model_var)):
                    loss_var.append((l, v))
            
            loss_var = sorted(loss_var, key=lambda x:x[0])
            for i in tqdm(range(len(loss_var) - args.smooth_window + 1)):
                loss = loss_var[i][0]
                var = np.mean([x[1] for x in loss_var[i:i+args.smooth_window]])
                vis.line(np.array([[var]]), np.array([[loss]]), win="loss-variance", update="append")
    elif args.func == 2:
        freq_loss = []
        corpus = WPDataset(args.data, 1e6, 50)
        vocab = corpus.TEXT.vocab
        with open(args.model_loss, "r") as model_file:
            while True:
                model_ref = model_file.readline()
                model_loss = model_file.readline()
                model_var = model_file.readline()
                
                if model_ref.strip() == "":
                    break

                model_ref = model_ref.strip()
                model_loss = model_loss.strip()

                model_ref = model_ref.split()
                model_loss = [float(l) for l in model_loss.split()]

                for l, r in list(zip(model_loss, model_ref)):
                    freq_loss.append((r, vocab.freqs[r], l))
            
            freq_loss = sorted(freq_loss, key=lambda x:x[1])
            for i in tqdm(range(0, len(freq_loss) - args.smooth_window + 1)):
                freq = freq_loss[i][1]
                loss = np.mean([x[2] for x in freq_loss[i:i+args.smooth_window]])
                vis.line(np.array([[loss]]), np.array([[freq]]), win="freq-loss", update="append")


if __name__ == "__main__":
    args = parse_args()
    main(args)

