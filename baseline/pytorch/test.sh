#!/bin/bash

python xl_lm.py \
    --adam \
    --datasets ptb \
    --epochs 50 \
    --lr 0.00025 \
    --dropout 0.1 \
    --nlayers 3 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --batch_size 50 \
    --eval_batch_size 50 \
    --num_steps 20 \
    --mem_len 60 \
    --tied \
    --attn_type 1 \
    --seed 1111 \
    --adaptive \
    --log-interval 50 \
    --multi_gpu \
    --save base_demo \
    ${@:1}
