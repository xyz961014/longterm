from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import re
import math
#import mosestokenizer

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtext import data, datasets
from torchtext.data import Batch, Dataset
import ipdb

UNK_token = 0  # Unknown word token
PAD_token = 1  # Used for padding short sentences
EOS_token = 2  # End-of-sentence token
SOS_token = 3  # Start-of-sentence token

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


class textDataset(torch.utils.data.Dataset):
    
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

class TextDataset(object):
    def __init__(self, path, vocab_size, num_steps):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        
        self.TEXT = data.Field(sequential=True)

        self.train_dataset = datasets.LanguageModelingDataset(path + "train.txt", 
                                                              self.TEXT)
        self.valid_dataset = datasets.LanguageModelingDataset(path + "valid.txt", 
                                                              self.TEXT)
        self.test_dataset = datasets.LanguageModelingDataset(path + "test.txt", 
                                                             self.TEXT)
        self.TEXT.build_vocab(self.train_dataset, max_size=vocab_size)

    def get_train_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.train_dataset, batch_size, self.num_steps,
                                 **kwargs)

    def randomlen_train_loader(self, batch_size, **kwargs):
        return RandomLengthBPTTIterator(self.train_dataset, batch_size, self.num_steps,
                                 **kwargs)

    def get_valid_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.valid_dataset, batch_size, self.num_steps,
                                 train=False, shuffle=False, sort=False,
                                 **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.test_dataset, batch_size, self.num_steps,
                                 train=False, shuffle=False, sort=False,
                                 **kwargs)

class ExistingDataset(object):
    def __init__(self, name, num_steps):
        super().__init__()
        self.num_steps = num_steps
        
        self.TEXT = data.Field(sequential=True)
        if name == "ptb":
            self.train_set, self.valid_set, self.test_set = datasets.PennTreebank.splits(self.TEXT)
        elif name == "wt2":
            self.train_set, self.valid_set, self.test_set = datasets.WikiText2.splits(self.TEXT)
        elif name == "wt103":
            self.train_set, self.valid_set, self.test_set = datasets.WikiText103.splits(self.TEXT)

        self.TEXT.build_vocab(self.train_set)

    def get_train_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.train_set, batch_size, self.num_steps,
                                 **kwargs)

    def get_valid_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.valid_set, batch_size, self.num_steps,
                                 train=False, shuffle=False, sort=False,
                                 **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.test_set, batch_size, self.num_steps,
                                 train=False, shuffle=False, sort=False,
                                 **kwargs)

    def randomlen_train_loader(self, batch_size, **kwargs):
        return RandomLengthBPTTIterator(self.train_set, batch_size, self.num_steps,
                                 **kwargs)



class RandomLengthBPTTIterator(data.Iterator):
    def __init__(self, dataset, batch_size, bptt_len, mem_len=0, **kwargs):
        self.bptt_len = bptt_len
        self.mem_len = mem_len
        super().__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil((len(self.dataset[0].text) / self.batch_size - 1)
                         / self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        _data = TEXT.numericalize([text], device=self.device)
        _data = _data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        mem_len = self.mem_len
        while True:
            i = 0
            while i < len(_data) - 1:
                self.iterations += 1

                bptt = self.bptt_len if np.random.random() < 0.95 else self.bptt_len / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                if mem_len > 0:
                    seq_len = max(self.bptt_len - mem_len + 5, seq_len)
                seq_len = min(seq_len, len(_data) - i - 1)

                batch_text = _data[i:i + seq_len]
                batch_target = _data[i + 1:i + 1 + seq_len]

                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)

                i += seq_len
                if mem_len > 0:
                    mem_len = mem_len + seq_len - self.bptt_len
            if not self.repeat:
                return


        

if __name__ == "__main__":
    TEXT = data.Field(sequential=True)
    ptb_train, ptb_valid, ptb_test = datasets.PennTreebank.splits(TEXT)
    TEXT.build_vocab(ptb_train)
    iterator = RandomLengthBPTTIterator(ptb_train, bptt_len=10, batch_size=10)
    for d in iterator:
        print(iterator.mem_len, d.text.size(0))
        ipdb.set_trace()
