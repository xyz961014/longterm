#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/ptb_sample/"
elif [[ $1 == "242" ]]; then
    data="/data/disk5/private/xyz/datasets/ptb_sample/"
elif [[ $1 == "243" ]]; then
    data="/data/disk5/private/xyz/datasets/ptb_sample/"
elif [[ $1 == "245" ]]; then
    data="/data/private/xyz/datasets/ptb_sample/"
elif [[ $1 == "102" ]]; then
    data="/data/private/xyz/datasets/ptb_sample/"
elif [[ $1 == "local" ]]; then
    data="/home/xyz/Documents/Dataset/ptb_sample/"
fi

python main.py \
    --data "${data}" \
    --datasets ptb \
    --adam \
    --epochs 100 \
    --lr 25e-5 \
    --dropout 0.4 \
    --nlayers 12 \
    --d_ff 1024 \
    --nhead 8 \
    --emsize 256 \
    --nhid 256 \
    --batch_size 100 \
    --eval_batch_size 100 \
    --num_steps 20 \
    --neighbor_len 20 \
    --cache_k 2 \
    --cache_N 5 \
    --tied \
    --attn_type 1 \
    --seed 1111 \
    --adaptive \
    --no_summary \
    --wise_summary \
    --farnear \
    --query_method single_sum \
    --eval_steps 3000 \
    --log-interval 50 \
    --multi_gpu \
    ${@:2}

