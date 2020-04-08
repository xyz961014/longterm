#!/bin/bash

python xl_lm.py \
    --adam \
    --datasets ptb \
    --lr 25e-5 \
    --std_epochs 50 \
    --ema_epochs 10 \
    --lr 0.00025 \
    --dropout 0.2 \
    --dropatt 0.2 \
    --nlayers 3 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --d_ff 1000 \
    --batch_size 200 \
    --eval_batch_size 200 \
    --num_steps 20 \
    --mem_len 60 \
    --tied \
    --seed 1111 \
    --adaptive \
    --log-interval 30 \
    --distributed \
    --save base_demo \
    ${@:1}
