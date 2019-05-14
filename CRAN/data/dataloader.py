from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import re
#import mosestokenizer

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token

#normalize = mosestokenizer.MosesPunctuationNormalizer("en")

device = torch.device("cpu")

class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<sos>", EOS_token: "<eos>", UNK_token: "<unk>"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

    def __len__(self):
        return self.num_words

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<sos>", EOS_token: "<eos>", UNK_token: "<unk>"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK
        for word in keep_words:
            self.addWord(word)

class Corpus(object):
    def __init__(self, path, name="corpus"):
        # get vocabulary
        self.vocabulary = Vocabulary(name=name)
        self._build_vocab(os.path.join(path, 'train.txt'))
        self._build_vocab(os.path.join(path, 'valid.txt'))
        self._build_vocab(os.path.join(path, 'test.txt'))

        self.train_data = textDataset(self._tokenize(os.path.join(path, 'train.txt')))
        self.valid_data = textDataset(self._tokenize(os.path.join(path, 'valid.txt')))
        self.test_data = textDataset(self._tokenize(os.path.join(path, 'test.txt')))

    def get_train_loader(self, batch_size=20, num_steps=35):
        self.train_data.batchify(batch_size)
        self.train_data.set_num_steps(num_steps)
        return DataLoader(self.train_data, batch_size, shuffle=False, drop_last=True)

    def get_valid_loader(self, batch_size=10, num_steps=35):
        self.valid_data.batchify(batch_size)
        self.valid_data.set_num_steps(num_steps)
        return DataLoader(self.valid_data, batch_size, drop_last=True)

    def get_test_loader(self, batch_size=10, num_steps=35):
        self.test_data.batchify(batch_size)
        self.test_data.set_num_steps(num_steps)
        return DataLoader(self.test_data, batch_size, drop_last=True)


    def _build_vocab(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                self.vocabulary.addSentence(normalizeString(line))

    def _tokenize(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            lines = []
            for line in f:
                line_ids = torch.LongTensor(indexesFromSentence(self.vocabulary, line)).to(device)
                lines.append(line_ids)
            corpus = torch.cat(lines)
        return corpus
        


# Lowercase and remove non-letter characters
def normalizeString(s):
    #s = s.lower()
    #s = re.sub(r"([.!?])", r" ", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Takes string sentence, returns sentence of word indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index.keys() else UNK_token for word in sentence.split()] + [EOS_token]


def _build_vocab(filename):
    with open(filename, "r") as f:
        sentences = f.readlines()
    voc = Vocabulary(name="vocab")
    for s in sentences:
        voc.addSentence(normalizeString(s))
    return voc

def _file_to_word_ids(filename, voc):
    with open(filename, "r") as f:
        sentences = f.readlines()
    file_with_word_ids = []
    for s in sentences:
        file_with_word_ids += indexesFromSentence(voc, s)
    return file_with_word_ids


def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")

    voc = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, voc)
    valid_data = _file_to_word_ids(valid_path, voc)
    test_data = _file_to_word_ids(test_path, voc)
    return train_data, valid_data, test_data, voc


class textDataset(Dataset):
    
    def __init__(self, raw_data, batch_size=20, num_steps=35, transform=None):
        self.raw_data = raw_data
        self.transform = transform
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        self.batchify(batch_size)

    def batchify(self, batch_size):
        self.batch_size = batch_size
        batch_len = len(self.raw_data) // batch_size
        self.data = self.raw_data.narrow(0, 0, batch_len * batch_size)
        self.data = self.data.view(batch_size, -1)

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def shuffle(self):
        ind = torch.randperm(self.batch_size)
        self.data = self.data[ind]

    def __len__(self):
        return self.batch_size * ((self.data.size(1) - 2) // self.num_steps)

    def __getitem__(self, idx):
        #data = self.data[idx: idx + self.num_steps]
        #target = self.data[idx + 1: idx + 1 + self.num_steps]
        #sample = data, target
        batch_num = idx % self.batch_size
        seq_num = (idx // self.batch_size) * self.num_steps
        seq_len = min(self.num_steps, self.data.size(1) - seq_num - 1)
        data = self.data[batch_num][seq_num: seq_num + seq_len]
        target = self.data[batch_num][seq_num + 1: seq_num + seq_len + 1]
        sample = data, target
        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    data_path = "/home/xyz/Documents/Dataset/ptb_sample"
    corpus = Corpus(data_path)
    train_loader = corpus.get_train_loader()
    for i, (data, target) in enumerate(train_loader):
        if i == 1:
            for b in data:
                print("")
                for w in b:
                    print(corpus.vocabulary.index2word[w.item()], end=" ")
    #print(corpus.train_data.data.shape)
    #train_data, valid_data, test_data, voc = ptb_raw_data(data_path)
    #traindataset = textDataset(train_data, 10, 5)
    #validdataset = textDataset(valid_data, 5, 3)
    #testdataset = textDataset(test_data, 10, 5)
    #train_loader = DataLoader(traindataset, batch_size=20, shuffle=False, num_workers=4)
    #valid_loader = DataLoader(validdataset, batch_size=5, shuffle=False, num_workers=4)
    ##print(len(traindataset))
    ##print(valid_data)
    #for i, (data, target) in enumerate(corpus.get_train_loader()):
    #    print(data.shape, target.shape)
        

if __name__ == "__main__":
    main()
