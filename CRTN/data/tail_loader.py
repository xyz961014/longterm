import os
import time

import torch
from torchtext import data
from torchtext import datasets
import pandas as pd
import ipdb
import time
from nltk import tokenize, sent_tokenize


def process_sents(sents):
    text = ""
    for sent in sents:
        line = " ".join(tokenize.word_tokenize(sent))
        line = sent_tokenize(line)
        if not text.strip() == "":
            text += " <eos> "
        text += " ".join(line)
    return text

class TailDataset(object):
    def __init__(self, path, vocab_size, num_steps):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps

        def complete(num_steps, data, vocab):
            com_len = -len(data[0]) % num_steps
            com_list = [vocab.stoi["<pad>"]] * (com_len - 1)
            if com_len > 0:
                com_list = [vocab.stoi["<eos>"]] + com_list
            for i in range(len(data)):
                data[i] += com_list
            return data

        #def add_eos(corpus):
        #    corpus = " ".join(corpus)
        #    sentences = sent_tokenize(corpus)
        #    return " <eos> ".join(sentences).split()

        self.TEXT = data.Field(sequential=True, pad_first=True,
                               postprocessing=lambda x, y: complete(num_steps, x, y))
        self.TRG = data.Field(sequential=True, eos_token="<eos>")

        fields = [self.TEXT, self.TRG]
        exts = [".text", ".tgt"]
        
        self.train_dataset = datasets.LanguageModelingDataset(path + "train.txt", 
                                                              self.TRG)
        self.train_valid_dataset = datasets.TranslationDataset(path + "train", 
                                                               exts, fields)
        self.valid_dataset = datasets.TranslationDataset(path + "valid", exts, fields)
        self.test_dataset = datasets.TranslationDataset(path + "test", exts, fields)

        self.TRG.build_vocab(self.train_dataset, max_size=vocab_size)
        self.TEXT.vocab = self.TRG.vocab


    def get_train_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.train_dataset, batch_size, self.num_steps, 
                                 **kwargs)

    def get_train_valid_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.train_valid_dataset, batch_size, 
                                   train=False, shuffle=False, sort=True,
                                   **kwargs)

    def get_valid_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.valid_dataset, batch_size, 
                                   train=False, shuffle=False, sort=False,
                                   **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.test_dataset, batch_size, 
                                   train=False, shuffle=False, sort=False,
                                   **kwargs)

class ROCDataset(object):
    def __init__(self, path, vocab_size, num_steps):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps

        def complete(num_steps, data, vocab):
            com_len = -len(data[0]) % num_steps
            com_list = [vocab.stoi["<pad>"]] * (com_len - 1)
            if com_len > 0:
                com_list = [vocab.stoi["<eos>"]] + com_list
            for i in range(len(data)):
                data[i] += com_list
            return data

        self.TEXT = data.Field(sequential=True, pad_first=True,
                               postprocessing=lambda x, y: complete(num_steps, x, y))
        self.TRG = data.Field(sequential=True, eos_token="<eos>")
        self.LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

        fields = [self.TEXT, self.TRG]
        exts = [".text", ".tgt"]
        
        self.train_dataset = datasets.LanguageModelingDataset(path + "train.txt", 
                                                              self.TRG)
        self.valid_dataset = datasets.TranslationDataset(path + "valid", exts, fields)
        self.test_dataset = datasets.TranslationDataset(path + "test", exts, fields)

        csv_fields = [("InputText", self.TEXT), ("AnswerCandidate1", self.TRG), ("AnswerCandidate2", self.TRG), ("RightAnswer", self.LABEL)]
        #self.discriminate_data = data.TabularDataset(path + "test.csv", "csv", 
        #                                             csv_fields, skip_header=True)
        self.discriminate_data = self.build_discriminate_dataset(path, csv_fields)

        self.TRG.build_vocab(self.train_dataset, max_size=vocab_size)
        self.TEXT.vocab = self.TRG.vocab


    def get_train_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.train_dataset, batch_size, self.num_steps, 
                                 **kwargs)

    def get_valid_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.valid_dataset, batch_size, 
                                   train=False, shuffle=False, sort=False,
                                   **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.test_dataset, batch_size, 
                                   train=False, shuffle=False, sort=False,
                                   **kwargs)

    def build_discriminate_dataset(self, path, fields):
        csv_file = pd.read_csv(path + "test.csv")
        examples = []
        for idx, story in csv_file.iterrows():
            sents = story.to_list()
            inputtext = process_sents(sents[1:5])
            cand1 = process_sents([sents[5]])
            cand2 = process_sents([sents[6]])
            label = sents[-1]
            example = data.Example.fromlist([inputtext, cand1, cand2, label], fields)
            examples.append(example)

        return data.Dataset(examples, fields)
        
    def get_discriminate_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.discriminate_data, batch_size, 
                                   train=False, shuffle=False, sort=False,
                                   **kwargs)

if __name__ == "__main__":
    path = "/home/xyz/Documents/Dataset/writingprompts/"
    start_time = time.time()
    dataloader = TailDataset(path, 1e6, 20)
    print("load time: %.2f s" % (time.time() - start_time))
    ta = dataloader.get_train_loader(12)
    vocab = dataloader.TRG.vocab
    print(len(vocab.itos))
    device = torch.device("cuda:0")
    #for data in da:
    #    ipdb.set_trace()
    #    text, trg = data.text, data.target
    #    textlist = text[:,0].tolist()

    #    ipdb.set_trace()
