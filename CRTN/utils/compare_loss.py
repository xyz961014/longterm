import argparse
import os
import sys
import ipdb
import numpy as np
from pprint import pprint
from tqdm import tqdm
import visdom
vis = visdom.Visdom()
assert vis.check_connection()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_loss", type=str, help="location of baseline loss")
    parser.add_argument("--model_loss", type=str, help="location of model loss")
    parser.add_argument("--smooth_window", type=int, default=50, 
                        help="smooth window")
    parser.add_argument("--func", type=int, choices=[0], default=0, 
                        help="function, 0 for stat bag of word loss")
    return parser.parse_args()

def main(args):
    base_file = open(args.baseline_loss, "r")
    model_file = open(args.model_loss, "r")

    if args.func == 0:
        opts = {
                'legend': ['Baseline', 'Model'],
                'showlegend': True
               }
        loss_dict = dict()
        while True:
            base_ref = base_file.readline()
            base_loss = base_file.readline()
            model_ref = model_file.readline()
            model_loss = model_file.readline()

            if base_ref.strip() == "":
                break

            # remove \n
            base_ref = base_ref[:-1]
            base_loss = base_loss[:-1]
            model_ref = model_ref[:-1]
            model_loss = model_loss[:-1]

            if base_ref == model_ref:
                base_ref = base_ref.split()
                base_loss = [float(l) for l in base_loss.split()]
                model_loss = [float(l) for l in model_loss.split()]
                for word, lossb, lossm in list(zip(base_ref, base_loss, model_loss)):
                    if word in loss_dict.keys():
                        loss_dict[word][0].append(lossb)
                        loss_dict[word][1].append(lossm)
                        if lossm > lossb:
                            loss_dict[word][2] -= 1
                        else:
                            loss_dict[word][2] += 1
                    else:
                        loss_dict[word] = [[lossb], [lossm]]
                        if lossm > lossb:
                            loss_dict[word].append(-1)
                        else:
                            loss_dict[word].append(1)

        word_losses = loss_dict.items()
        word_meanloss = [(x[0], np.mean(x[1][0]), np.mean(x[1][1])) for x in word_losses]
        word_meanloss = sorted(word_meanloss, key=lambda x: x[1])
        # the smaller, the better
        word_diffs = [(x[0], np.mean(x[1][1]) - np.mean(x[1][0])) for x in word_losses]
        word_diffs = sorted(word_diffs, key=lambda x:x[1])
        # the bigger, the better
        word_betters = [(x[0], x[1][2]) for x in word_losses]
        word_betters = sorted(word_betters, key=lambda x:x[1], reverse=True)
        for i in tqdm(range(len(word_meanloss))):
            lossb = np.mean([x[1] for x in word_meanloss[max(0, i-args.smooth_window):min(i+args.smooth_window, len(word_meanloss)-1)]])
            lossm = np.mean([x[2] for x in word_meanloss[max(0, i-args.smooth_window):min(i+args.smooth_window, len(word_meanloss)-1)]])
            diff_loss = lossb - lossm
            vis.line(np.array([[lossb, lossm]]), np.array([i]), opts=opts, win="loss", update="append")
            vis.line(np.array([[diff_loss]]), np.array([i]), win="loss diff", update="append")




    base_file.close()
    model_file.close()
        


if __name__ == "__main__":
    args = parse_args()
    main(args)

