#!/bin/bash

python main.py \
    --datasets ptb \
    --adam \
    --epochs 200 \
    --lr 8e-4 \
    --warmup_steps 5000 \
    --dropout 0.6 \
    --dropatt 0.3 \
    --dropemb 0.15 \
    --dropinp 0.65 \
    --dropwei 0.05 \
    --drophid 0.25 \
    --weight_decay 1e-5 \
    --nlayers 12 \
    --d_ff 1024 \
    --nhead 10 \
    --emsize 400 \
    --nhid 400 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --num_steps 30 \
    --neighbor_len 30 \
    --cache_k 2 \
    --cache_N 5 \
    --tied \
    --attn_type 1 \
    --seed 1111 \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --no_summary \
    --wise_summary \
    --farnear \
    --query_method single_sum \
    --eval_steps 3000 \
    --log-interval 100 \
    --distributed \
    ${@:1}

