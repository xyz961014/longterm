#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/ptb_sample"
elif [[ $1 == "242" ]]; then
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
    ${@:2}

