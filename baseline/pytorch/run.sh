#!/bin/bash

python xl_lm.py \
    --datasets ptb \
    --adam \
    --std_epochs 150 \
    --ema_epochs 150 \
    --lr 3e-4 \
    --warmup_steps 3000 \
    --dropout 0.5 \
    --dropatt 0.2 \
    --dropemb 0.2 \
    --dropinp 0.6 \
    --dropwei 0.1 \
    --dropfor 0.2 \
    --weight_decay 12e-7 \
    --nlayers 16 \
    --d_ff 900 \
    --nhead 10 \
    --emsize 380 \
    --nhid 380 \
    --batch_size 10 \
    --eval_batch_size 10 \
    --num_steps 70 \
    --mem_len 70 \
    --tied \
    --seed 1111 \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --eval_steps 3000 \
    --log-interval 200 \
    --distributed \
    --save base \
    ${@:1}
