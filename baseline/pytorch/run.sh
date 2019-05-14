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
    --emsize 200 \
    --nhid 200 \
    --nhead 4 \
    --nlayers 2 \
    --d_ff 1000 \
    --num_steps 70 \
    --mem_len 210 \
    --epochs 100 \
    --batch_size 60 \
    --dropout 0.2 \
    --attn_type 1 \
    --adaptive \
    --div_val 2 \
    --seed 1111 \
    ${@:2}
