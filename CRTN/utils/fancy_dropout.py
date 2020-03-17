from torch.nn import Parameter

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

class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)



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

    word_emb = torch.nn.functional.embedding(words, 
                                             masked_embed_weight,
                                             padding_idx, 
                                             embed.max_norm, 
                                             embed.norm_type,
                                             embed.scale_grad_by_freq, 
                                             embed.sparse)
    return word_emb


if __name__ == '__main__':
  V = 50
  h = 6
  bptt = 10
  batch_size = 2

  embed = torch.nn.Embedding(V, h)

  words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
  words = torch.LongTensor(words)

  origX = embed(words)
  X = embedded_dropout(embed, words)

  print(origX)
  print(X)
