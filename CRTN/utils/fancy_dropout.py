from torch.nn import Parameter
import torch.nn.functional as F

import numpy as np
import torch
import ipdb


def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, Parameter(w))

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)

def _weight_drop_linear(module, dropout):
    w = module.weight
    module.weight = Parameter()
    module.register_parameter('weight_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        raw_w = module.weight_raw
        w = F.dropout(raw_w, p=dropout, training=module.training)
        module.weight.data = w

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)
    

class WeightDropLinear(torch.nn.Linear):
    """
    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)

        #weights = ['weight']
        #_weight_drop(self, weights, weight_dropout)
        #_weight_drop_linear(self, weight_dropout)
        self.dropout = weight_dropout

    def forward(self, inputs):
        if not self.training:
            return super().forward(inputs)
        drop_w = F.dropout(self.weight, p=self.dropout, training=self.training)
        output = F.linear(inputs, Parameter(drop_w), self.bias)
        return output



def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout > 0:
        emb_size = embed.weight.size(0)
        mask = embed.weight.new_empty(emb_size, 1)
        mask = mask.bernoulli(1 - dropout)
        mask = mask.expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        scale = scale.expand_as(masked_embed_weight)
        masked_embed_weight = scale * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    word_emb = F.embedding(words, 
                           masked_embed_weight,
                           padding_idx, 
                           embed.max_norm, 
                           embed.norm_type,
                           embed.scale_grad_by_freq, 
                           embed.sparse)
    return word_emb


if __name__ == '__main__':
    V = 50
    h = 5
    bptt = 10
    batch_size = 2

    lin = WeightDropLinear(h, h, weight_dropout=0.5)
    origY = torch.randn(5,5)
    ipdb.set_trace()
    Y = lin(origY)

    embed = torch.nn.Embedding(V, h)

    words = np.random.randint(low=0, high=V-1, size=(batch_size, bptt))
    words = torch.LongTensor(words)

    origX = embed(words)
    X = embedded_dropout(embed, words)

    #print(origX)
    #print(X)
