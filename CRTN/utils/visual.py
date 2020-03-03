import pickle as pkl
import numpy as np
import ipdb

class TargetText(object):
    def __init__(self, words=[], 
                       loss=[], 
                       var=[], 
                       near_loss=[], 
                       batch_size=200,
                       num_steps=20):
        super().__init__()
        self.words = words
        self.loss = loss
        self.var = var
        self.near_loss = near_loss
        self.batch_size = batch_size
        self.num_steps = num_steps

    def add_words(self, words):
        for word in words:
            self.words.append(word)

    def add_losss(self, losss):
        for loss in losss:
            self.loss.append(loss)

    def add_variances(self, variances):
        for var in variances:
            self.var.append(var)

    def add_near_losss(self, near_losss):
        for near_loss in near_losss:
            self.near_loss.append(near_loss)

    def clear(self):
        self.words = []
        self.loss = []
        self.var = []
        self.near_loss = []

    def arranged_loss(self):
        loss_array = np.array(self.loss)
        loss_array = loss_array.reshape(-1, self.num_steps, self.batch_size)
        return loss_array

    def arranged_words(self):
        words_array = np.array(self.words)
        words_array = words_array.reshape(-1, self.num_steps, self.batch_size)
        return words_array

    def __len__(self):
        return len(self.words)
