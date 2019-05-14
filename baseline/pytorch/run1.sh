#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/ptb_sample"
elif [[ $1 == "local" ]]; then
    data="/home/xyz/Documents/Dataset/ptb_sample"
fi
python lm.py \
    --data "${data}" \
    --adam \
    --lr 0.00025 \
    --tied \
    --emsize 400 \
    --nhid 400 \
    --nhead 8 \
    --nlayers 12 \
    --d_ff 1000 \
    --num_steps 70 \
    --mem_len 70 \
    --epochs 300 \
    --dropout 0.4 \
    --attn_type 1 \
    --seed 1111 \
    --save att1ada \
    --adaptive
