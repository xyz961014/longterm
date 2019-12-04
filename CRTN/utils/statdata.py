import argparse
import os
import sys
import ipdb
sys.path.append("../..")

from CRTN.data import dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="location of dataset")
    parser.add_argument("--func", type=int, choices=[0, 1], default=1, help="function, 0 for train,valid, test word stat of dataset; 1 for vocab stat of a txt")
    return parser.parse_args()

def main(args):
    if args.func == 0:
        corpus = dataloader.Corpus(args.data)
        print("Train Words: %s" % len(corpus.train_data.data.view(-1)))
        print("Valid Words: %s" % len(corpus.valid_data.data.view(-1)))
        print("Test Words: %s" % len(corpus.test_data.data.view(-1)))
        print("Vocabs: %s" % corpus.vocabulary.num_words)
    elif args.func == 1:
        vocab = dataloader.Vocabulary("test")
        with open(args.data, "r") as f:
            for line in f:
                vocab.addSentence(line)
        ipdb.set_trace()


if __name__ == "__main__":
    args = parse_args()
    main(args)

