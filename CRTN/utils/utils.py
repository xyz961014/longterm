import torch
import numpy as np

import torch.distributed as dist

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def partial_shuffle(data):
    # Method from "Partially Shuffling the Training Data to Improve Language Models" by Ofir Press
    # https://arxiv.org/abs/1903.04167

    # data size (batch_size, batch_len)

    #N = data.size(0)  # batch size
    #M = data.shape[1]  # length of each batch element (or equivalently, row)

    shifted = []
    for i, row in enumerate(data):
        M = row.size(0)
        split = torch.from_numpy(np.random.randint(M, size=1))
        shifted.append(
            torch.cat((row[split:], row[:split])) #partial shuffle of a single row
        )
    print('The training data has been partially shuffled!')
    if isinstance(data, torch.Tensor):
        return torch.stack(shifted)
    else:
        return shifted

def init_cache_info(args, evaluate=False):
    """
    store relative position of cahce chunk and its recall times
    [length, distance, recall times, all query times]
    postions including each cache slots and neighbor mem if exsists, final shape is 
    [cache_N ( + nei_len), batch_size, 4] neighbor_mem info at last
    """
    batch_size = args.eval_batch_size if evaluate else args.batch_size
    if args.distributed:
        batch_size = batch_division(batch_size, args.rank, single_value=True)

    #pos = torch.arange(args.cache_N, 0, -1, dtype=torch.float).cuda()
    #recall_query = torch.zeros((args.cache_N, 2), dtype=torch.float).cuda()
    #cache_info = torch.cat((pos.unsqueeze(-1), recall_query), dim=-1).unsqueeze(1)
    #cache_info = cache_info.expand(-1, batch_size, -1).contiguous()

    if args.farnear:
        if args.sentence_cache:
            cache_info = torch.zeros(args.cache_N + args.neighbor_len, batch_size, 4).cuda()
        else:
            cache_info = torch.zeros(args.cache_N + 1, batch_size, 4).cuda()
    else:
        cache_info = torch.zeros(args.cache_N, batch_size, 4).cuda()

    return cache_info

def batch_division(batch_size, rank=0, world_size=None, single_value=False):
    if world_size is None:
        world_size = dist.get_world_size()
    batch_div = batch_size // world_size
    if rank < world_size - 1:
        if not single_value:
            return batch_div * rank, batch_div * (rank + 1)
        else:
            return batch_div
    elif rank == world_size - 1:
        if not single_value:
            return batch_div * rank, batch_size
        else:
            return batch_size - batch_div * rank

def param_in(p, params):
    for param in params:
        if p.equal(param):
            return True
    else:
        return False
 
def padding_hidden(tensor_list, length=-1, before=True):
    """
    return size: (batch_size, nalyers, nei_len, nhid)
    """
    max_len = max([t.size(1) for t in tensor_list])
    if length > 0:
        assert length >= max_len
        max_len = length

    tensors = []
    for tensor in tensor_list:
        if before:
            tensor = torch.cat((tensor.new_zeros(tensor.size(0),
                                                 max_len - tensor.size(1),
                                                 tensor.size(2)), 
                                tensor),
                               dim=1)
        else:
            tensor = torch.cat((tensor,
                                tensor.new_zeros(tensor.size(0),
                                                 max_len - tensor.size(1),
                                                 tensor.size(2))), 
                               dim=1)
        tensors.append(tensor.unsqueeze(0))

    return torch.cat(tensors, dim=0)


def padding_cache(tensor_list, cache_eos, cache_L):
    """
    return size: (batch_size, sentence_num, nalyers, cache_L, nhid)
    """
    sentence_len = cache_eos - torch.cat((cache_eos.new_zeros(1, cache_eos.size(1)), cache_eos[:-1]))

    assert cache_L >= sentence_len.max()

    cache_batches = []
    for it, tensor in enumerate(tensor_list):
        splits = sentence_len[:,it]
        sentences = tensor.split(splits.tolist(), dim=1)
        padded_sentences = padding_hidden(sentences, length=cache_L, before=False)
        cache_batches.append(padded_sentences.unsqueeze(0))

    return torch.cat(cache_batches, dim=0)

class Logger(object):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def log(self, string):
        with open(self.filename, "a") as f:
            f.write(string)
            f.write("\n")
        print(string)
