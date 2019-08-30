#!/bin/bash

python main.py \
    --data /data/private/xyz/datasets/ptb_sample \
    --adam \
    --lr 0.00025 \
    --nlayers 6 \
    --dropout 0.1 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --batch_size 30 \
    --num_steps 40 \
    --tied \
    --attn_type 1 \
    --cache_k 3 \
    --cache_N 5 \
    --seed 1111 \
    --epochs 100 \
    --log-interval 100 \
    --save demo \
    --adaptive \
    --no_summary \
    --wise_summary \
    ${@:1}
