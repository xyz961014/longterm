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
        self.args = args
        self.batch_size = args.batch_size
        self.attention_wq = nn.Linear(args.embedding_dim, args.hidden_size)
        self.attention_wk = nn.Linear(args.hidden_size, args.hidden_size)
        self.compute_hidden = nn.Linear(args.hidden_size*2, args.hidden_size)

        self.cache = Cache(args)
        self.attn = DotProductAttention()

    def to(self, device):
        super().to(device)
        self.cache.to(device)

    def set_batch_size(self, batch_size):
        #self.args.batch_size = batch_size
        self.batch_size = batch_size
        self.cache.set_batch_size(batch_size)
        
    def forward(self, inputs):
        #print(inputs.shape)
        weights, zones = self.cache(inputs)
        #print(weights.shape, zones.shape)
        query = self.attention_wq(inputs)
        query = torch.cat(tuple(query.view([1]+list(query.size())) for _ in range(self.args.cache_k)), 0)
        keys = self.attention_wk(zones)
        zones = zones.view([-1]+list(zones.size()[2:]))
        _, output = self.attn(query, keys, zones)
        output = output.view(self.args.cache_k, self.batch_size, -1).transpose(0, 1).contiguous()
        attention_output = torch.matmul(weights, output).view(self.batch_size, -1)
        hidden_state = F.relu(self.compute_hidden(torch.cat((attention_output, inputs), 1)))
        self.cache.renew(hidden_state)

        return hidden_state


        
