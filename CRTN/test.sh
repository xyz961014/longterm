#!/bin/bash

python main.py \
    --adam \
    --datasets ptb \
    --epochs 100 \
    --lr 0.00025 \
    --dropout 0.1 \
    --nlayers 3 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --batch_size 200 \
    --eval_batch_size 200 \
    --num_steps 20 \
    --neighbor_len 20 \
    --tied \
    --attn_type 1 \
    --cache_k 2 \
    --cache_N 5 \
    --seed 1111 \
    --log-interval 50 \
    --adaptive \
    --no_summary \
    --wise_summary \
    --farnear \
    --query_method single_sum \
    --multi_gpu \
    --save demo \
    ${@:1}

