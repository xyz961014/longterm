#!/bin/bash

python recl.py \
    --data /home/xyz/Documents/Dataset/ptb_sample \
    --model_names CRTN \
    --model_paths \
    ../../../experiment/crtn/merge7shift_best.pt \
    --initc 70 \
    --initr 0.5 \
    --delta 70 \
    ${@:1}

