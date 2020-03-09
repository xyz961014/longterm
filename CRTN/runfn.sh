#!/bin/bash


python main.py \
    --datasets ptb \
    --adam \
    --epochs 200 \
    --lr 5e-4 \
    --dropout 0.4 \
    --dropatt 0.4 \
    --weight_decay 1e-5 \
    --nlayers 12 \
    --d_ff 1024 \
    --nhead 8 \
    --emsize 256 \
    --nhid 256 \
    --batch_size 200 \
    --eval_batch_size 200 \
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
    --distributed \
    ${@:1}

