from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import re
import math
#import mosestokenizer
sys.path.append("../..")

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtext import data, datasets
from torchtext.data import Batch, Dataset
from CRTN.utils.utils import partial_shuffle

import torch.distributed as dist

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

        self.train_set = datasets.LanguageModelingDataset(path + "train.txt", 
                                                              self.TEXT)
        self.valid_set = datasets.LanguageModelingDataset(path + "valid.txt", 
                                                              self.TEXT)
        self.test_set = datasets.LanguageModelingDataset(path + "test.txt", 
                                                             self.TEXT)
        self.TEXT.build_vocab(self.train_set, max_size=vocab_size)

    def get_train_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.train_set, batch_size, self.num_steps,
                                 **kwargs)

    def randomlen_train_loader(self, batch_size, **kwargs):
        return RandomLengthBPTTIterator(self.train_set, batch_size, self.num_steps,
                                 **kwargs)

    def get_valid_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.valid_set, batch_size, self.num_steps,
                                 train=False, shuffle=False, sort=False,
                                 **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.test_set, batch_size, self.num_steps,
                                 train=False, shuffle=False, sort=False,
                                 **kwargs)

    def recl_loader(self, batch_size, target_len, context_len, **kwargs):
        return RECLIterator(self.valid_set, batch_size, target_len, context_len, 
                            **kwargs)

class ExistingDataset(object):
    def __init__(self, name, num_steps):
        super().__init__()
        self.num_steps = num_steps
        
        self.TEXT = data.Field(sequential=True)
        if name == "ptb":
            self.train_set, self.valid_set, self.test_set = datasets.PennTreebank.splits(self.TEXT, root="../.data")
        elif name == "wt2":
            self.train_set, self.valid_set, self.test_set = datasets.WikiText2.splits(self.TEXT, root="../.data")
        elif name == "wt103":
            self.train_set, self.valid_set, self.test_set = datasets.WikiText103.splits(self.TEXT, root="../.data")

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

    def sentence_train_loader(self, batch_size, sent_num, **kwargs):
        return SentenceBPTTIterator(self.train_set, batch_size, sent_num, **kwargs)

    def sentence_valid_loader(self, batch_size, sent_num, **kwargs):
        return SentenceBPTTIterator(self.valid_set, batch_size, sent_num,
                                    train=False, shuffle=False, sort=False, **kwargs)

    def sentence_test_loader(self, batch_size, sent_num, **kwargs):
        return SentenceBPTTIterator(self.test_set, batch_size, sent_num,
                                    train=False, shuffle=False, sort=False, **kwargs)

    def partial_shuffle_loader(self, batch_size, **kwargs):
        return PartialShuffleBPTTIterator(self.train_set, batch_size, self.num_steps, 
                                          **kwargs)

    def recl_loader(self, batch_size, target_len, context_len, **kwargs):
        return RECLIterator(self.valid_set, batch_size, target_len, context_len, 
                            **kwargs)


class PartialShuffleBPTTIterator(data.Iterator):
    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
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
        _data = TEXT.numericalize(
            [text], device=self.device)
        _data = _data.view(self.batch_size, -1).t().contiguous()
        _data = partial_shuffle(_data)
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(_data) - i - 1)
                batch_text = _data[i:i + seq_len]
                batch_target = _data[i + 1:i + 1 + seq_len]

                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return


class RandomLengthBPTTIterator(data.Iterator):
    def __init__(self, dataset, batch_size, bptt_len, mem_len=0, partial_shuffled=False, **kwargs):
        self.bptt_len = bptt_len
        self.mem_len = mem_len
        self.partial_shuffled = partial_shuffled
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
        if self.partial_shuffled:
            _data = partial_shuffle(_data)
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        mem_len = self.mem_len
        mem_min = round(mem_len / 2)
        max_len = self.bptt_len + 20
        while True:
            i = 0
            while i < len(_data) - 1:
                if mem_len + len(_data) - i - 1 < self.bptt_len:
                    break
                self.iterations += 1

                if dist.get_rank() == 0:
                    #bptt = self.bptt_len
                    bptt = self.bptt_len if np.random.random() < 0.95 else self.bptt_len / 2.
                    seq_len = max(5, int(np.random.normal(bptt, 5)))
                    # set max_len in case of OOM
                    seq_len = min(seq_len, max_len)
                    if mem_len > 0:
                        seq_len = max(self.bptt_len - mem_len + mem_min, seq_len)
                    seq_len = min(seq_len, len(_data) - i - 1)
                    seq_len_tensor = torch.tensor(seq_len).int()
                else:
                    seq_len_tensor = torch.zeros(1).int()

                if dist.get_world_size() > 1:
                    dist.broadcast(seq_len_tensor, 0)
                    seq_len = seq_len_tensor.item()

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

class SentenceBPTTIterator(data.Iterator):
    def __init__(self, dataset, batch_size, sent_num, partial_shuffled=False, max_sent_len=50, **kwargs):
        """
            The dataset should be already segmented by <eos> in sentence level.
            output:
                text: sent_num sentences
                target: target of text
                eos: sentence length labels, every <eos> indices, includin fake ones.
        """
        self.sent_num = sent_num
        self.max_sent_len = max_sent_len
        self.partial_shuffled = partial_shuffled
        super().__init__(dataset, batch_size, **kwargs)

        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        eos_token = TEXT.vocab.stoi["<eos>"]
        _data = TEXT.numericalize([text], device=self.device)
        eos_indices, _ = _data.eq(eos_token).nonzero(as_tuple=True)

        # insert additional fake <eos> for too long sentence ( > max_sent_len)
        fake_eoses = []
        ind = -1
        for ie in range(eos_indices.size(0)):
            curr_ind = eos_indices[ie].item()    
            while curr_ind - ind > self.max_sent_len:
                ind += self.max_sent_len
                fake_eoses.append(ind)
            ind = curr_ind
        if len(fake_eoses) > 0:
            eos_indices = torch.cat((torch.tensor(fake_eoses), eos_indices))
            eos_indices, _ = eos_indices.sort()

        self._data = _data
        self.eos_indices = eos_indices

    def __len__(self):
        return math.ceil((self.eos_indices.size(0) // self.batch_size) / self.sent_num)

    def __iter__(self):
       
        TEXT = self.dataset.fields['text']
        IND = data.Field(sequential=False, use_vocab=False)
        pad_token = TEXT.vocab.stoi["<pad>"]
        _data = self._data
        eos_indices = self.eos_indices

        # batchify sentences
        data_len = eos_indices.size(0)
        batch_len = data_len // self.batch_size
        # cut into batches
        boundry_indices = torch.arange(batch_len, data_len, batch_len)[:self.batch_size] - 1
        boundries = torch.gather(eos_indices, 0, boundry_indices)
        # cut tail of _data and batchify
        split_len = boundries - torch.cat((boundries.new_ones(1) * (-1), boundries[:-1]))
        _data = _data.squeeze(1).narrow(0, 0, split_len.sum()).split(split_len.tolist(), dim=0)

        # rebuild batch-wise eos indices
        eos_indices = eos_indices.narrow(0, 0, batch_len * self.batch_size)
        eos_indices = eos_indices.reshape(self.batch_size, batch_len)
        rel_eos_pos_bias = torch.cat((eos_indices.new_zeros(1), (eos_indices[:-1, -1] + 1))).expand(batch_len, -1).t()
        eos_indices = eos_indices - rel_eos_pos_bias
        eos_indices = torch.cat((eos_indices.new_ones(self.batch_size, 1) * (-1), eos_indices), dim=1)

        if self.partial_shuffled:
            _data = partial_shuffle(_data)
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT), ('eos', IND)])
        while True:
            for i in range(0, batch_len, self.sent_num):
                self.iterations += 1

                batch_texts, batch_targets = [], []
                max_len = 0
                for ib, batch in enumerate(_data):
                    if i + self.sent_num < batch_len:
                        start_idx, end_idx = eos_indices[ib][i] + 1, eos_indices[ib][i + self.sent_num] + 1
                    else:
                        start_idx, end_idx = eos_indices[ib][i] + 1, -2

                    text = batch[start_idx:end_idx]
                    target = batch[start_idx + 1:end_idx + 1]
                    if not text.size(0) == target.size(0):
                        text = text[:-1]
                    text_len = text.size(0)
                    if text_len > max_len:
                        max_len = text_len
                    batch_texts.append(text)
                    batch_targets.append(target)
                for ib in range(self.batch_size):
                    seq_len = batch_texts[ib].size(0)
                    batch_texts[ib] = torch.cat((batch_texts[ib], batch_texts[ib].new_ones(max_len - seq_len) \
                                      * pad_token)).unsqueeze(-1)
                    batch_targets[ib] = torch.cat((batch_targets[ib], batch_targets[ib].new_ones(max_len - seq_len) \
                                      * pad_token)).unsqueeze(-1)

                batch_text = torch.cat(batch_texts, dim=1)
                batch_target = torch.cat(batch_targets, dim=1)
                batch_eos = eos_indices[:,i:i+self.sent_num+1]
                batch_eos = batch_eos.t() - (batch_eos[:,0] + 1).expand(batch_eos.size(1), -1)
                batch_eos = batch_eos[1:]

                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                    batch_eos = batch_eos.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target,
                    eos=batch_eos)

            if not self.repeat:
                return



class RECLIterator(data.Iterator):
    def __init__(self, dataset, batch_size, target_len, context_len, end_bias=0, **kwargs):
        self.target_len = target_len
        self.context_len = context_len
        assert end_bias >= 0
        self.end_bias = end_bias
        super().__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return self.target_len

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        e = self.end_bias
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        #text = text[:len(text) // self.batch_size * self.batch_size]
        _data = TEXT.numericalize([text], device=self.device)
        _data = _data.view(self.batch_size, -1).t().contiguous()
        pad = TEXT.numericalize([[TEXT.pad_token]], device=self.device)
        #pad = pad.expand(self.context_len, self.batch_size)
        #_data = torch.cat((pad, _data), dim=0)
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            i = 0
            while i < self.target_len:
                if len(_data) - i - 3 < e:
                    break
                self.iterations += 1

                seq_len = self.context_len

                start = -3 - i - seq_len - e
                end = -2 - i - e
                if start + len(_data) < 0:
                    pad_tensor = pad.expand(-start - len(_data), -1)
                    start = 0
                    padded = True
                else:
                    padded = False
                batch_text = _data[start: end]
                batch_target = _data[start + 1: end + 1]
                if padded:
                    batch_text = torch.cat((pad_tensor, batch_text), dim=0)
                    batch_target = torch.cat((pad_tensor, batch_target), dim=0)


                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)

                i += 1
            if not self.repeat:
                return


        

if __name__ == "__main__":
    TEXT = data.Field(sequential=True)
    ptb_train, ptb_valid, ptb_test = datasets.PennTreebank.splits(TEXT)
    wt2_train, wt2_valid, wt2_test = datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(ptb_train)
    iterator = RandomLengthBPTTIterator(ptb_valid, batch_size=10, bptt_len=80, mem_len=80)
    ds = []
    for d in iterator:
        print(d.text.shape[0])
