#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/ptb_sample"
elif [[ $1 == "242" ]]; then
    data="/data/disk5/private/xyz/datasets/ptb_sample"
elif [[ $1 == "245" ]]; then
    data="/data/private/xyz/datasets/ptb_sample"
elif [[ $1 == "local" ]]; then
    data="/home/xyz/Documents/Dataset/ptb_sample"
fi
python lm.py \
    --data "${data}" \
    --adam \
    --lr 25e-5 \
    --tied \
    --emsize 100 \
    --nhid 100 \
    --nhead 4 \
    --nlayers 6 \
    --d_ff 1000 \
    --num_steps 40 \
    --mem_len 120 \
    --epochs 100 \
    --batch_size 30 \
    --dropout 0.1 \
    --attn_type 1 \
    --adaptive \
    --seed 1111 \
    --multi_gpu \
    ${@:2}
