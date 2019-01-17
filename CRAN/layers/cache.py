import torch
import torch.nn as nn
import torch.nn.functional as F

from CRAN.layers.attention import DotProductAttention

class Cache(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
        Arguments of args:
        N: size of the Cache
        dk: dimension of key
        dv: dimension of value
        L: max length of a sequence stored in one value
        k: select top k values
        """
        self.args = args
        self.keys = torch.zeros(args.cache_N, args.cache_dk, requires_grad=False)
        self.values = torch.zeros(args.cache_N, args.cache_L, args.cache_dv, requires_grad=False)
        self.renew_state = [0, 0] # current renew place
        self.attn = DotProductAttention()
        self.summary = nn.Linear(args.cache_L*args.cache_dv, args.cache_dk) # to compute the key from a Zone, may be sub with a more complicated one
        self.topk = args.cache_k

    def to(self, device):
        super().to(device)
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)

    def forward(self, query):
        #print("1",query.shape, self.keys.shape, self.values.shape)
        #print(query.type(), self.keys.type(), self.values.type())
        attention, _ = self.attn(query, self.keys, self.values.view([1]+list(self.values.size())))
        topk_weights, topk_indices = attention.topk(self.topk)
        topk_weights = topk_weights.view(-1)
        topk_indices = topk_indices.view(-1)
        #print(attention, topk_indices)
        outputs = torch.tensor([], device=self.values.device)
        #print(topk_indices.shape)
        #print(self.values.shape)
        for ind in topk_indices:
            zone = self.values[ind.item()]
            zone = zone.view([1]+list(zone.size()))
            outputs = torch.cat((outputs, zone), 0)
        #print("cacheoutput", outputs.shape)
            
        return topk_weights, outputs

    def renew(self, hidden):
        #print("hidden", hidden.shape)
        hidden = hidden.detach()
        n, l = self.renew_state
        if l < self.args.cache_L - 1:
            value_to_update = self.values[n]
            value_to_update[l] = hidden
            l += 1
            #print(self.keys[n].shape, self.summary(value_to_update.view(-1)).shape)
            self.keys[n] = torch.mean(value_to_update, 0)
            self.values[n] = value_to_update
            self.renew_state = [n, l]
        elif l == self.args.cache_L - 1:
            if n < self.args.cache_N - 1:
                self.renew_state = [n+1, 0]
                self.renew(hidden)
            elif n == self.args.cache_N - 1:
                self.eliminate_last()
                n -= 1
                l = 0
                self.renew_state = [n, l]
                self.renew(hidden)

    def eliminate_last(self):
        new_key = torch.tensor([], device=self.keys.device)
        new_value = torch.tensor([], device=self.values.device)
        for ind, (key, val) in enumerate(list(zip(self.keys,self.values))):
            if ind > 0:
                new_key = torch.cat((new_key, key.view([1]+list(key.size()))), 0)
                new_value = torch.cat((new_value, val.view([1]+list(val.size()))), 0)
        else:
            zero_key = torch.zeros(1, key.size(0), device=self.keys.device)
            zero_value = torch.zeros(1, val.size(0), val.size(1), device=self.values.device)
            new_key = torch.cat((new_key, zero_key), 0)
            new_value = torch.cat((new_value, zero_value), 0)
        self.values = new_value
        self.keys = new_key
        
