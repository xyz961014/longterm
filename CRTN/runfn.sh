#!/bin/bash

python main.py \
    --datasets ptb \
    --adam \
    --std_epochs 200 \
    --ema_epochs 0 \
    --lr 8e-4 \
    --eta_min 1e-6 \
    --warmup_steps 20000 \
    --dropout 0.6 \
    --dropatt 0.3 \
    --dropemb 0.1 \
    --dropinp 0.65 \
    --dropwei 0.05 \
    --drophid 0.25 \
    --weight_decay 1e-5 \
    --nlayers 12 \
    --d_ff 1024 \
    --nhead 10 \
    --emsize 400 \
    --nhid 400 \
    --batch_size 30 \
    --eval_batch_size 10 \
    --num_steps 80 \
    --neighbor_len 80 \
    --cache_k 2 \
    --cache_N 5 \
    --tied \
    --seed 1111 \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --farnear \
    --summary_method sum \
    --query_method single \
    --eval_steps 3000 \
    --log-interval 100 \
    --distributed \
    ${@:1}

