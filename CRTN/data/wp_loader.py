import os
import time

from torchtext import data
from torchtext import datasets
import ipdb
import time
from nltk import sent_tokenize



class WPDataset(object):
    def __init__(self, path, vocab_size, num_steps):
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

        self.TEXT = data.Field(sequential=True, pad_first=True, eos_token="<eos>",
                               postprocessing=lambda x, y: complete(num_steps, x, y))
        self.TRG = data.Field(sequential=True, eos_token="<eos>")

        fields = [self.TEXT, self.TRG]
        exts = [".text", ".tgt"]
        
        self.train_dataset = datasets.LanguageModelingDataset(path + "train.txt", 
                                                              self.TRG)
        self.valid_dataset = datasets.TranslationDataset(path + "valid", exts, fields)
        self.test_dataset = datasets.TranslationDataset(path + "test", exts, fields)

        self.TRG.build_vocab(self.train_dataset, max_size=vocab_size)
        self.TEXT.vocab = self.TRG.vocab


    def get_train_loader(self, batch_size, **kwargs):
        return data.BPTTIterator(self.train_dataset, batch_size, self.num_steps, 
                                 **kwargs)

    def get_valid_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.valid_dataset, batch_size, **kwargs)

    def get_test_loader(self, batch_size, **kwargs):
        return data.BucketIterator(self.test_dataset, batch_size, **kwargs)


if __name__ == "__main__":
    path = "/home/xyz/Documents/Dataset/writingpromts/toy/"
    start_time = time.time()
    dataloader = WPDataset(path, 10000, 200)
    print("load time: %.2f s" % (time.time() - start_time))
    ta = dataloader.get_train_loader(12, device="cuda:0")
    va = dataloader.get_valid_loader(5, device="cuda:0")
    vocab = dataloader.TRG.vocab
    for data in ta:
        text, trg = data.text, data.target
        textlist = text[:,0].tolist()
        ipdb.set_trace()
    while True:
        tait = next(ta.__iter__())
        vait = next(va.__iter__())
        src, tgt = vait.src[:,0].tolist(), vait.trg[:,0].tolist()

        ipdb.set_trace()
