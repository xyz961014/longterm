#!/bin/bash

kernprof -lv main.py \
    --data_from_torchtext \
    --adam \
    --lr 0.00025 \
    --nlayers 3 \
    --dropout 0.1 \
    --nhead 4 \
    --emsize 50 \
    --nhid 50 \
    --batch_size 10 \
    --num_steps 40 \
    --tied \
    --attn_type 1 \
    --cache_k 3 \
    --cache_N 5 \
    --seed 1111 \
    --epochs 1 \
    --log-interval 100 \
    --save demo \
    --adaptive \
    --no_summary \
    --wise_summary \
    ${@:1}

