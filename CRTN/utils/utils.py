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
    data = data.t()  
    N = data.shape[0]  # batch size
    M = data.shape[1]  # length of each batch element (or equivalently, row)

    splits = torch.from_numpy(np.random.randint(M, size=N))
    shifted = []
    for i, row in enumerate(data):
        shifted.append(
            torch.cat((row[splits[i]:], row[:splits[i]])) #partial shuffle of a single row
        )
    #print('The training data has been partially shuffled!')
    return torch.stack(shifted).t()

def init_cache_info(args, evaluate=False):
    """
    store relative position of cahce chunk and its recall times
    [pos, recall times, all query times]
    """
    batch_size = args.eval_batch_size if evaluate else args.batch_size
    if args.distributed:
        batch_size = batch_division(batch_size, args.rank, single_value=True)
    pos = torch.arange(args.cache_N, 0, -1, dtype=torch.float).cuda()
    recall_query = torch.zeros((args.cache_N, 2), dtype=torch.float).cuda()
    cache_info = torch.cat((pos.unsqueeze(-1), recall_query), dim=-1).unsqueeze(1)
    cache_info = cache_info.expand(-1, batch_size, -1).contiguous()

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
 
