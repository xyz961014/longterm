#!/bin/bash

python recl.py \
    --data /home/xyz/Documents/Dataset/ptb_sample \
    --model_names LSTM XL CRTN \
    --model_paths \
    ./save/lstm_best.pt \
    ./save/xl_best.pt \
    ./save/base_best.pt \
    --initc 70 \
    --initr 0.5 \
    --delta 70 \
    ${@:1}

