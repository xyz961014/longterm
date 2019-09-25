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
    --lr 27e-5 \
    --tied \
    --emsize 240 \
    --nhid 240 \
    --nhead 8 \
    --nlayers 15 \
    --d_ff 1300 \
    --num_steps 70 \
    --mem_len 210 \
    --epochs 300 \
    --batch_size 50 \
    --dropout 0.45 \
    --attn_type 1 \
    --adaptive \
    --seed 1111 \
    ${@:2}
