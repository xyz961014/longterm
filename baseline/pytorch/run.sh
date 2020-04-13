#!/bin/bash

python xl_lm.py \
    --datasets ptb \
    --adam \
    --std_epochs 200 \
    --ema_epochs 50 \
    --lr 8e-4 \
    --eta_min 1e-4 \
    --emb_mult 1 \
    --ema_lr_mult 1 \
    --alpha 0 \
    --beta 0 \
    --warmup_steps 20000 \
    --dropout 0.6 \
    --dropatt 0.3 \
    --dropemb 0.15 \
    --dropinp 0.65 \
    --dropwei 0.05 \
    --dropfor 0.25 \
    --weight_decay 1e-5 \
    --nlayers 12 \
    --d_ff 1024 \
    --nhead 10 \
    --emsize 400 \
    --nhid 400 \
    --batch_size 40 \
    --eval_batch_size 10 \
    --num_steps 80 \
    --mem_len 80 \
    --tied \
    --seed 1111 \
    --adaptive \
    --cutoffs 2000 4000 8000 \
    --eval_steps 3000 \
    --log-interval 200 \
    --distributed \
    --save base \
    ${@:1}
