import torch
import torch.nn as nn
import torch.nn.functional as F

from CRAN.layers.cache import Cache
from CRAN.layers.attention import DotProductAttention


class CRANUnit(nn.Module):
    def __init__(self, args):
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
        self.batch_size = args.batch_size
        self.attention_wq = nn.Linear(args.embedding_dim, args.hidden_size)
        self.attention_wk = nn.Linear(args.hidden_size, args.hidden_size)
        self.compute_hidden = nn.Linear(args.hidden_size*2, args.hidden_size)

        self.caches = nn.ModuleList([Cache(args) for _ in range(args.batch_size)])
        self.attn = DotProductAttention()

    def to(self, device):
        super().to(device)
        for cache in self.caches:
            cache.to(device)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def forward(self, inputs):
        #print(inputs.shape)
        hidden_states = torch.tensor([], device=inputs.device)
        for i in range(self.batch_size):
            weights, zones = self.caches[i](inputs[i])
            #print(weights.shape, zones.shape)
            attention_output = torch.zeros_like(zones[0][0])
            for weight, zone in list(zip(weights, zones)):
                query = self.attention_wq(inputs[i])
                key = self.attention_wk(zone)
                key, zone = key.view([1]+list(key.size())), zone.view([1]+list(zone.size()))
                #print(query.shape, key.shape, zone.shape)
                _, output = self.attn(query, key, zone)
                output = output.view(-1)
                attention_output += output * weight
                #print(attention_output.shape, inputs.shape)
            hidden_state = self.compute_hidden(torch.cat((attention_output, inputs[i]), 0))
            self.caches[i].renew(hidden_state)
            hidden_states = torch.cat((hidden_states, hidden_state.view([1]+list(hidden_state.size()))), 0)

        return hidden_states


        
