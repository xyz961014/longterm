#!/bin/bash

python recl.py \
    --data /home/xyz/Documents/Dataset/ptb_sample \
    --model_names LSTM XL CRTN \
    --model_paths ../baseline/pytorch/save/lstm_best.pt \
    ../baseline/pytorch/save/xl_best.pt \
    ./save/crtn/crtn_best.pt \
    --initc 80 \
    --initr 1.0 \
    --delta 40 \
    ${@:1}

