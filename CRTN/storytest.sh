#!/bin/bash

python story_tail.py \
    --adam \
    --lr 0.00025 \
    --nlayers 3 \
    --dropout 0.1 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --batch_size 50 \
    --eval_batch_size 12 \
    --num_steps 20 \
    --tied \
    --attn_type 1 \
    --cache_k 3 \
    --cache_N 5 \
    --seed 1111 \
    --epochs 10 \
    --log-interval 50 \
    --save story_demo \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --no_summary \
    --wise_summary \
    --farnear \
    ${@:1}

