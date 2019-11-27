#!/bin/bash

if [[ $1 == "234" ]]; then
    data="/data/disk4/private/xyz/datasets/writingpromts/"
elif [[ $1 == "242" ]]; then
    data="/data/disk5/private/xyz/datasets/writingpromts/"
elif [[ $1 == "243" ]]; then
    data="/data/disk5/private/xyz/datasets/writingpromts/"
elif [[ $1 == "245" ]]; then
    data="/data/private/xyz/datasets/writingpromts/"
elif [[ $1 == "102" ]]; then
    data="/data/private/xyz/datasets/writingpromts/"
fi

python story_tail.py \
    --data "${data}" \
    --adam \
    --lr 25e-5 \
    --nlayers 12 \
    --dropout 0.1 \
    --nhead 8 \
    --emsize 240 \
    --nhid 240 \
    --batch_size 200 \
    --eval_batch_size 10 \
    --num_steps 20 \
    --tied \
    --attn_type 1 \
    --cache_k 2 \
    --cache_N 5 \
    --seed 1111 \
    --epochs 20 \
    --log-interval 50 \
    --save story_run \
    --adaptive \
    --vocab_size 100000 \
    --cutoffs 20000 40000 80000 \
    --no_summary \
    --wise_summary \
    --farnear \
    --neighbor_len 20 \
    --multi_gpu \
    ${@:2}
