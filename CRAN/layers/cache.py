import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from CRAN.layers.attention import DotProductAttention

class Cache(nn.Module):
    def __init__(self, args, corpus=None):
        super().__init__()
        """
        Arguments of args:
        N: size of the Cache
        dk: dimension of key
        dv: dimension of value, the same as hidden_size
        L: max length of a sequence stored in one value
        k: select top k values
        """
        self.args = args
        self.corpus = corpus
        self.demo = args.demo
        self.keys = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros(args.batch_size, args.cache_dk), requires_grad=False) for i in range(args.cache_N)
            })
        self.values = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros(args.cache_L, args.batch_size, args.hidden_size), requires_grad=False) for i in range(args.cache_N)
            })
        if args.demo:
            self.words = nn.ParameterDict({
                str(i): nn.Parameter(torch.zeros(args.cache_L, args.batch_size), requires_grad=False) for i in range(args.cache_N)
                })
        self.renew_state = [0, 0] # current renew place
        self.attn = DotProductAttention()
        self.lookup = nn.Linear(args.embedding_dim, args.cache_dk)
        self.summary = nn.Linear(args.cache_L*args.hidden_size, args.cache_dk) # to compute the key from a Zone, may be sub with a more complicated one
        self.batch_size = args.batch_size
        self.L = args.cache_L
        self.N = args.cache_N
        self.dk = args.cache_dk
        self.dv = args.hidden_size
        self.topk = args.cache_k

    def to(self, device):
        super().to(device)
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)

    def _get_keys(self):
        keys = self.keys.values()
        #print("keys:",keys)
        keys = torch.cat(tuple(keys), 0)
        keys = keys.view(self.N, -1, self.dk)
        return keys

    def _get_values(self):
        values = self.values.values()
        values = torch.cat(tuple(values), 0)
        values = values.view(self.N, self.L, -1, self.dv)
        return values

    def _get_words(self):
        words = self.words.values()
        words = torch.cat(tuple(words), 0)
        words = words.view(self.N, self.L, -1)
        return words

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.keys.clear()
        self.values.clear()
        self.keys.update({
            str(i): nn.Parameter(torch.zeros(batch_size, self.dk), requires_grad=False) for i in range(self.N)
            })
        self.values.update({
            str(i): nn.Parameter(torch.zeros(self.L, batch_size, self.dv), requires_grad=False) for i in range(self.N)
            })
        if self.demo:
            self.words.update({
                str(i): nn.Parameter(torch.zeros(self.L, batch_size, dtype=torch.long), requires_grad=False) for i in range(self.N)
                })

        self.renew_state = [0, 0]


    def forward(self, query):
        #query = self.lookup(query)
        keys = self._get_keys()
        values = self._get_values()
        if self.demo:
            words = self._get_words()
            words = words.transpose(0, 2).contiguous()
            words = words.transpose(1, 2).contiguous()

        keys = keys.transpose(0, 1).contiguous()
        values = values.transpose(0, 2).contiguous()
        values = values.transpose(1, 2).contiguous()
        #print("1",query, keys, values)
        #print(query.type(), self.keys.type(), self.values.type())
        attention, _ = self.attn(query, keys, values)

        if self.demo:
            topk_weights, topk_indices = attention.topk(self.N)
            topk_indices = topk_indices.transpose(0, 2).contiguous().view(self.N, -1)
            batch = torch.cat(tuple(torch.arange(self.batch_size) for _ in range(self.N))).view(self.N, -1)
            topk_weights = F.softmax(topk_weights[0][0][:self.topk], 0).view(-1, 1, self.topk)
            topk_weights = torch.cat((topk_weights, torch.zeros(self.N - self.topk, device=topk_weights.device).view(-1, 1, self.N - self.topk)), 2)
        else:
            topk_weights, topk_indices = attention.topk(self.topk)
            topk_indices = topk_indices.transpose(0, 2).contiguous().view(self.topk, -1)
            batch = torch.cat(tuple(torch.arange(self.batch_size) for _ in range(self.topk))).view(self.topk, -1)
            topk_weights = F.softmax(topk_weights, 2)
        #outputs = torch.tensor([], device=values.device)
        #for batch in topk_indices:
        #    zones = torch.tensor([], device=values.device)
        #    for ib, ind in enumerate(batch):
        #        zone = self.values[str(ind.item()+max(0, self.renew_state[0]-self.N+1))][:, ib, :]
        #        zone = zone.view([1]+list(zone.size()))
        #        zones = torch.cat((zones, zone), 0)
        #    zones = zones.view([1]+list(zones.size()))
        #    #zones = self._get_values()[batch]
        #    outputs = torch.cat((outputs, zones), 0)
        ##print("cacheoutput", outputs.shape)
        
        outputs = values[batch, topk_indices]
        if self.demo:
            word_output = words[batch, topk_indices]
            return topk_weights, outputs, word_output, topk_indices
            #for iz, zone in enumerate(word_output):
            #    zone = zone.view(-1)
            #    print("weight: %.3f ZONE %s:" % (topk_weights.view(-1)[iz].item(), topk_indices.view(-1)[iz].item()))
            #    for w in zone:
            #        print(self.corpus.vocabulary.index2word[w.item()], end=" ")
            #    print("")
        return topk_weights, outputs

    def renew(self, hidden, word=None):
        hidden = hidden.detach()
        n, l = self.renew_state
        if l < self.L:
            try:
                value_to_update = self.values[str(n)]
            except:
                print(self)
            if word is not None: 
                word_to_update = self.words[str(n)]
                word_to_update[l] = word
                self.words[str(n)] = word_to_update
            value_to_update[l] = hidden
            l += 1
            #print(self.keys[n].shape, self.summary(value_to_update.view(-1)).shape)
            self.keys.update({str(n): nn.Parameter(F.relu(self.summary(value_to_update.view(-1, self.L*self.dv))), requires_grad=False)})
            self.values[str(n)] = value_to_update
            self.renew_state = [n, l]
        elif l == self.L:
            if n >= self.N - 1:
                self.eliminate_last()
            self.renew_state = [n+1, 0]
            self.renew(hidden, word)

    def eliminate_last(self):
        #print(self.keys.keys(), self.values.keys())
        keys_keys = list(self.keys.keys())
        keys_values = list(self.values.keys())

        keys_keys = sorted(list(map(int, keys_keys)))
        keys_values = sorted(list(map(int, keys_values)))

        eli_key = self.keys.pop(str(keys_keys[0]))
        eli_value = self.values.pop(str(keys_values[0]))
        self.keys.update({
            str(keys_keys[-1]+1): nn.Parameter(torch.zeros(self.batch_size, self.dk), requires_grad=False)
            })
        self.values.update({
            str(keys_values[-1]+1): nn.Parameter(torch.zeros(self.L, self.batch_size, self.dv), requires_grad=False)
            })
        if self.demo:
            keys_words = list(self.words.keys())
            keys_words = sorted(list(map(int, keys_words)))
            eli_word = self.words.pop(str(keys_words[0]))
            self.words.update({
                str(keys_words[-1]+1): nn.Parameter(torch.zeros(self.L, self.batch_size, dtype=torch.long), requires_grad=False)
                })
        device = eli_key.device
        self.to(device)
                
