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
    --emsize 100 \
    --nhid 100 \
    --nhead 4 \
    --nlayers 3 \
    --d_ff 1000 \
    --num_steps 40 \
    --mem_len 40 \
    --epochs 50 \
    --batch_size 60 \
    --dropout 0.1 \
    --attn_type 1 \
    --adaptive \
    --div_val 2 \
    --seed 1111 \
    --log-interval 100 \
    ${@:2}
