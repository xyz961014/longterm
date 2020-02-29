import pickle as pkl

class TargetText(object):
    def __init__(self, words=[], loss=[], var=[], near_loss=[]):
        super().__init__()
        self.words = words
        self.loss = loss
        self.var = var
        self.near_loss = near_loss

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

    def __len__(self):
        return len(self.words)
