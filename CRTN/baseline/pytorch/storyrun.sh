#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/writingprompts/medium/"
elif [[ $1 == "242" ]]; then
    data="/data/disk5/private/xyz/datasets/writingprompts/medium/"
elif [[ $1 == "243" ]]; then
    data="/data/disk5/private/xyz/datasets/writingprompts/medium/"
elif [[ $1 == "245" ]]; then
    data="/data/private/xyz/datasets/writingprompts/medium/"
elif [[ $1 == "102" ]]; then
    data="/data/private/xyz/datasets/writingprompts/medium/"
fi

python story_tail.py \
    --data "${data}" \
    --adam \
    --lr 25e-5 \
    --nlayers 12 \
    --dropout 0.2 \
    --d_ff 1024 \
    --nhead 8 \
    --emsize 256 \
    --nhid 256 \
    --batch_size 150 \
    --eval_batch_size 200 \
    --num_steps 20 \
    --mem_len 60 \
    --tied \
    --seed 1111 \
    --epochs 50 \
    --log-interval 50 \
    --save baseline_run \
    --adaptive \
    --vocab_size 50000 \
    --cutoffs 10000 20000 30000 \
    --eval_steps 3000 \
    --log-interval 100 \
    --distributed \
    ${@:2}

