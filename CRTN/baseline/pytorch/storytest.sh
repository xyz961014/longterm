#!/bin/bash

python story_tail.py \
    --adam \
    --lr 0.00025 \
    --nlayers 3 \
    --dropout 0.3 \
    --nhead 4 \
    --emsize 100 \
    --nhid 100 \
    --batch_size 50 \
    --eval_batch_size 50 \
    --num_steps 20 \
    --mem_len 80 \
    --tied \
    --seed 1111 \
    --epochs 100 \
    --log-interval 50 \
    --save baseline_demo \
    --adaptive \
    --vocab_size 25000 \
    --cutoffs 5000 10000 15000 \
    --eval_steps 5000 \
    --distributed \
    ${@:1}

