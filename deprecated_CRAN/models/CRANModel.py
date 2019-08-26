import torch
import torch.nn as nn
import torch.nn.functional as F

from CRAN.models.CRANUnit import CRANUnit

import ipdb

class CRANModel(nn.Module):
    def __init__(self, args, corpus=None):
        """
        Arguments of Model:
        num_steps: num of words in one step
        vocab_size: vocabulary size
        embedding_dim: dimension of embedding
        Arguments of CRANUnit:
            embedding_dim: dimension of embedding
            hidden_size: size of hidden state
            arguments of cache:
                N: size of the Cache
                dk: dimension of key
                dv: dimension of value
                L: max length of a sequence stored in one value
                k: select top k values
        """
        super().__init__()
        self.args = args
        self.demo = args.demo
        self.batch_size = args.batch_size
        self.corpus = corpus
        self.cranunit = CRANUnit(args, corpus)
        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.num_steps = args.num_steps
        self.hidden2tag = nn.Linear(args.hidden_size, args.vocab_size)

    def to(self, device):
        super().to(device)
        self.cranunit.to(device)

    def init_hidden(self):
        return torch.zeros(self.args.batch_size, self.args.hidden_size)

    def forward(self, inputs, hiddens):
        embeddings = self.embeddings(inputs)
        
        if self.demo:
            hiddens = self._build_graph(embeddings, hiddens, inputs)
        else:
            hiddens = self._build_graph(embeddings, hiddens)
        #print("modelhiddens", hiddens.shape)

        logits = self.hidden2tag(hiddens)
        #if self.demo:
        #    preds = torch.max(F.softmax(logits), 2)
        #    for pred in preds[1].view(-1):
        #        print("预测词：", self.corpus.vocabulary.index2word[pred.item()], end=" ")
        #    print("")
        return logits, hiddens[-1]

    def _build_graph(self, embeddings, hiddens, inputs=None):
        output = hiddens
        outputs = torch.tensor([], device=embeddings.device)
        for step in range(self.num_steps):
            if inputs is not None:

                logit = self.hidden2tag(output)
                pred = torch.max(torch.softmax(logit, 1), 1)
                print("\n预测词：%s (%.3f)" % (self.corpus.vocabulary.index2word[pred[1].item()], pred[0].item() * 100))
                print("p(输入词):%.3f " % (torch.softmax(logit, 1).view(-1)[inputs[step].item()].item() * 100))
                output = self.cranunit(embeddings[step], output, inputs[step])
            else:
                output = self.cranunit(embeddings[step], output)
            outputs = torch.cat((outputs, output.view([1]+list(output.size()))), 0)
        return outputs

    def set_batch_size(self, batch_size):
        self.cranunit.set_batch_size(batch_size)
