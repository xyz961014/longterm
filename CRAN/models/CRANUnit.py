import torch
import torch.nn as nn
import torch.nn.functional as F

from CRAN.layers.cache import Cache
from CRAN.layers.attention import DotProductAttention

import ipdb

def repackage_state(s):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(s, torch.Tensor):
        return s.detach()
    else:
        return list(repackage_state(v) for v in s)



class CRANUnit(nn.Module):
    def __init__(self, args, corpus=None):
        """
        Arguments of args:
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
        self.batch_size = args.batch_size
        self.demo = args.demo
        self.corpus = corpus
        self.drop = nn.Dropout(args.dropout)
        self.attention_wq = nn.Linear(args.embedding_dim + args.hidden_size, args.hidden_size)
        self.attention_wk = nn.Linear(args.hidden_size, args.hidden_size)
        if args.update == "standard":
            self.compute_hidden = nn.Linear(args.hidden_size*2, args.hidden_size)
        elif args.update == "gated":
            self.W_r = nn.Linear(args.hidden_size*2, args.hidden_size)
            self.W_z = nn.Linear(args.hidden_size*2, args.hidden_size)
            self.W_n = nn.Linear(args.hidden_size, args.hidden_size)
            self.W_i = nn.Linear(args.hidden_size, args.hidden_size)

        if args.demo:
            self.cache = Cache(args, corpus)
        else:
            self.cache = Cache(args)
        self.attn = DotProductAttention()

    def to(self, device):
        super().to(device)
        self.cache.to(device)

    def set_batch_size(self, batch_size):
        #self.args.batch_size = batch_size
        self.batch_size = batch_size
        self.cache.set_batch_size(batch_size)

    def forward(self, inputs, hiddens, words=None):
        hiddens = repackage_state(hiddens)
        if self.demo:
            print("输入词：",self.corpus.vocabulary.index2word[words.item()])
            weights, zones, zone_words, zone_indices = self.cache(inputs)
            

        else:
            weights, zones = self.cache(inputs)
        query = F.relu(self.attention_wq(torch.cat((inputs, hiddens), 1)))
        if self.demo:
            query = torch.cat(tuple(query.view([1]+list(query.size())) for _ in range(self.args.cache_N)), 0)
        else:
            query = torch.cat(tuple(query.view([1]+list(query.size())) for _ in range(self.args.cache_k)), 0)
        keys = F.relu(self.attention_wk(zones))
        zones = zones.view([-1]+list(zones.size()[2:]))
        words_weights, output = self.attn(query, keys, zones)
        if self.demo:
            weights_display = list(zip(zone_indices, zone_words, words_weights, weights.view(-1)))
            weights_display = sorted(weights_display, key=lambda x: x[0].item())
            for iz, zone in enumerate(weights_display):
                print("weight: %.3f ZONE %s:" % (zone[3].item(), zone[0].item()))
                for word, weight in list(zip(zone[1].view(-1), zone[2].view(-1))):
                    print(self.corpus.vocabulary.index2word[word.item()], end="|")
                    if zone[3].item() == 0:
                        weight = torch.tensor(0.0)
                    print("%.3f" % weight.item(), end=" ")
                print("")
            output = output.view(self.args.cache_N, self.batch_size, -1).transpose(0, 1).contiguous()
        else:
            output = output.view(self.args.cache_k, self.batch_size, -1).transpose(0, 1).contiguous()
        attention_output = torch.matmul(weights, output).view(self.batch_size, -1)
        if self.args.update == "standard":
            hidden_state = F.relu(self.compute_hidden(torch.cat((attention_output, inputs), 1)))
        elif self.args.update == "gated":
            r = F.sigmoid(self.W_r(torch.cat((attention_output, inputs), 1)))
            z = F.sigmoid(self.W_z(torch.cat((attention_output, inputs), 1)))
            n = torch.tanh(r * self.W_n(inputs) + self.W_i(attention_output))
            hidden_state = (torch.ones_like(z) - z) * n + z * attention_output
        if self.demo:
            self.cache.renew(hidden_state, words)
        else:
            self.cache.renew(hidden_state)

        return hidden_state


        
