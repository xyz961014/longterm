#!/bin/bash

python main.py \
    --adam \
    --datasets ptb \
    --std_epochs 50 \
    --lr 25e-5 \
    --dropout 0.2 \
    --dropatt 0.2 \
    --nlayers 2 \
    --nhead 4 \
    --emsize 100 \
    --nhid 80 \
    --d_ff 400 \
    --batch_size 200 \
    --eval_batch_size 200 \
    --num_steps 20 \
    --neighbor_len 20 \
    --tied \
    --cache_k 2 \
    --cache_N 5 \
    --seed 1111 \
    --log-interval 30 \
    --adaptive \
    --farnear \
    --summary_method sum \
    --query_method single \
    --distributed \
    --save demo \
    ${@:1}

