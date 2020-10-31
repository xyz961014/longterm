from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse
import collections
import torch
import re


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--path", help="checkpoint directory")
    parser.add_argument("--output", default="average",
                        help="Output path")
    parser.add_argument("--checkpoints", default=5, type=int,
                        help="Number of checkpoints to average")

    return parser.parse_args()


def list_checkpoints(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    vals = []

    for name in names:
        counter = name.rstrip(".pt").split("_")[-1]
        if re.search("^[0-9]+$", counter):
            counter = int(counter)
            vals.append([counter, name])

    return [item[1] for item in sorted(vals)]


def main(args):
    checkpoints = list_checkpoints(args.path)

    if not checkpoints:
        raise ValueError("No checkpoint to average")

    checkpoints = checkpoints[-args.checkpoints:]
    model_values = collections.OrderedDict()
    crit_values = collections.OrderedDict()

    for checkpoint in checkpoints:
        print("Loading checkpoint: %s" % checkpoint)
        ckp = torch.load(checkpoint, map_location="cpu")
        model_state = ckp["model_state_dict"]
        crit_state = ckp["criterion"]

        for key in model_state:
            if key not in model_values:
                model_values[key] = model_state[key].float().clone()
            else:
                model_values[key].add_(model_state[key].float())

        for key in crit_state:
            if key not in crit_values:
                crit_values[key] = crit_state[key].float().clone()
            else:
                crit_values[key].add_(crit_state[key].float())

    for key in model_values:
        model_values[key].div_(len(checkpoints))
    for key in crit_values:
        crit_values[key].div_(len(checkpoints))

    state = {
                "model_args": ckp["model_args"], 
                "model_state_dict": model_values, 
                "criterion": crit_values
            }

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    torch.save(state, os.path.join(args.output, "average_0.pt"))

if __name__ == "__main__":
    main(parse_args())
