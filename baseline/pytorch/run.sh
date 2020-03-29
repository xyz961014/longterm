#!/bin/bash

python xl_lm.py \
    --datasets ptb \
    --adam \
    --epochs 200 \
    --lr 1e-3 \
    --warmup_steps 20000 \
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
    --batch_size 40 \
    --eval_batch_size 40 \
    --num_steps 80 \
    --mem_len 240 \
    --tied \
    --seed 1111 \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --eval_steps 3000 \
    --log-interval 100 \
    --distributed \
    --save base \
    ${@:1}
