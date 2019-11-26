#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/ptb_sample"
elif [[ $1 == "242" ]]; then
    data="/data/disk5/private/xyz/datasets/ptb_sample"
elif [[ $1 == "243" ]]; then
    data="/data/disk5/private/xyz/datasets/ptb_sample"
elif [[ $1 == "245" ]]; then
    data="/data/private/xyz/datasets/ptb_sample"
elif [[ $1 == "102" ]]; then
    data="/data/private/xyz/datasets/ptb_sample"
elif [[ $1 == "local" ]]; then
    data="/home/xyz/Documents/Dataset/ptb_sample"
fi

python main.py \
    --data "${data}" \
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
    --cache_k 2 \
    --cache_N 5 \
    --seed 1111 \
    --epochs 200 \
    --log-interval 50 \
    --adaptive \
    --no_summary \
    --wise_summary \
    --farnear \
    --neighbor_len 70 \
    ${@:2}

