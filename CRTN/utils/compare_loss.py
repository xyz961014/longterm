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
    parser.add_argument("--func", type=int, choices=[0, 1], default=0, 
                        help="function, 0 for stat bag of word loss, 1 for observe variance and loss of model")
    return parser.parse_args()

def main(args):

    if args.func == 0:
        base_file = open(args.baseline_loss, "r")
        model_file = open(args.model_loss, "r")
        opts_diff = {
                'legend': ['loss difference'],
                'showlegend': True,
               }
        opts_prob = {
                'legend': ['prob - 0.5'],
                'showlegend': True,
               }
        loss_dict = dict()
        while True:
            base_ref = base_file.readline()
            base_loss = base_file.readline()
            model_ref = model_file.readline()
            model_loss = model_file.readline()
            model_var = model_file.readline()

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
        #word_meanloss = [(x[0], np.mean(x[1][0]), np.mean(x[1][1])) for x in word_losses]
        #word_meanloss = sorted(word_meanloss, key=lambda x: x[1])

        # the smaller, the better
        #word_diffs = [(x[0], np.mean(x[1][1]) - np.mean(x[1][0])) for x in word_losses]
        #word_diffs = sorted(word_diffs, key=lambda x:x[1])
        # the bigger, the better
        #word_betters = [(x[0], x[1][2]) for x in word_losses]
        #word_betters = sorted(word_betters, key=lambda x:x[1], reverse=True)

        for i in tqdm(range(len(word_losses) - args.smooth_window + 1)):
            lossb = np.mean(word_losses[i][1][0])
            lossb_window = [x[1][0] for x in word_losses[i:i+args.smooth_window]]
            lossm_window = [x[1][1] for x in word_losses[i:i+args.smooth_window]]

            loss_diff = []
            model_better, total = 0, 0
            for lbs, lms in list(zip(lossb_window, lossm_window)):
                for lb, lm in list(zip(lbs, lms)):
                    loss_diff.append(lb - lm)
                    if lb > lm:
                        model_better += 1
                total += len(lbs)
            loss_diff = np.mean(loss_diff)
            prob_model_better = model_better / total
            vis.line(np.array([[loss_diff]]), np.array([lossb]), opts=opts_diff, win="loss diff", update="append")
            vis.line(np.array([[prob_model_better - 0.5]]), np.array([lossb]), opts=opts_prob, win="prob of model better - 0.5", update="append")




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

                model_loss = model_loss[:-1]
                model_var = model_var[:-1]

                model_loss = [float(l) for l in model_loss.split()]
                model_var = [float(l) for l in model_var.split()]

                for l, v in list(zip(model_loss, model_var)):
                    loss_var.append((l, v))
            
            loss_var = sorted(loss_var, key=lambda x:x[0])
            for i in tqdm(range(len(loss_var) - args.smooth_window + 1)):
                loss = loss_var[i][0]
                var = np.mean([x[1] for x in loss_var[i:i+args.smooth_window]])
                vis.line(np.array([[var]]), np.array([[loss]]), win="loss-variance", update="append")
        


if __name__ == "__main__":
    args = parse_args()
    main(args)

