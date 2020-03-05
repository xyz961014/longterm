import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../baseline")
sys.path.append("../../baseline/pytorch")
from copy import copy
import ipdb
from story_tail import main as tail_main
from main import main as lm_main
from pytorch.xl_lm import main as base_lm_main
from pytorch.story_tail import main as base_tail_main

def process_tail_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    tail_main(local_args)


def process_lm_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    lm_main(local_args)

def process_base_tail_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    base_tail_main(local_args)

def process_base_lm_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    base_lm_main(local_args)
