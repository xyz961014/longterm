#!/bin/bash

python main.py \
    --adam \
    --lr 0.00025 \
    --nlayers 3 \
    --dropout 0.1 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --batch_size 60 \
    --num_steps 40 \
    --tied \
    --attn_type 1 \
    --cache_k 1 \
    --cache_N 2 \
    --adaptive \
    --seed 1111 \
    --epochs 100 \
    --log-interval 100 \
    --save testmodel2 \
    ${@:1}

