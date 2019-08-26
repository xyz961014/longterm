#!/bin/bash

python main.py \
    --data /data/disk4/private/xyz/datasets/ptb_sample \
    --adam \
    --lr 27e-5 \
    --nlayers 15 \
    --dropout 0.45 \
    --nhead 8 \
    --emsize 240 \
    --nhid 240 \
    --batch_size 50 \
    --num_steps 70 \
    --d_ff 1300 \
    --tied \
    --attn_type 1 \
    --cache_k 3 \
    --cache_N 5 \
    --seed 1111 \
    --epochs 300 \
    --log-interval 50 \
    --save run1 \
    --adaptive \
    --no_summary \
    --wise_summary \
    ${@:1}

