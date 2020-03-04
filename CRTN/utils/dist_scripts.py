import sys
sys.path.append("..")
sys.path.append("../..")
from copy import copy
import ipdb
from story_tail import main

def process_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    main(local_args)


