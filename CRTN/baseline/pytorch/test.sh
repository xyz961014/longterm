#!/bin/bash

python xl_lm.py \
    --adam \
    --datasets ptb \
    --lr 25e-5 \
    --std_epochs 50 \
    --ema_epochs 10 \
    --lr 0.00025 \
    --warmup_steps 0 \
    --dropout 0.2 \
    --dropatt 0.2 \
    --nlayers 2 \
    --nhead 4 \
    --emsize 80 \
    --nhid 80 \
    --d_ff 400 \
    --batch_size 200 \
    --eval_batch_size 200 \
    --num_steps 20 \
    --mem_len 60 \
    --tied \
    --seed 1111 \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --log-interval 30 \
    --distributed \
    --save base_demo \
    ${@:1}
