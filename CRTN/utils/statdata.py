import argparse
import os
import sys
import ipdb
sys.path.append("../..")

from CRTN.data import dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="location of dataset")
    return parser.parse_args()

def main(args):
    corpus = dataloader.Corpus(args.data)
    print("Train Words: %s" % len(corpus.train_data.data.view(-1)))
    print("Valid Words: %s" % len(corpus.valid_data.data.view(-1)))
    print("Test Words: %s" % len(corpus.test_data.data.view(-1)))
    print("Vocabs: %s" % corpus.vocabulary.num_words)


if __name__ == "__main__":
    args = parse_args()
    main(args)

