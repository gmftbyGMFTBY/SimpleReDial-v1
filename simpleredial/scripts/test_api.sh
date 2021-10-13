#!/bin/bash

mode=$1
dataset=$2
prefix_name=$3
python test_api.py \
    --size 100 \
    --url 9.91.66.241 \
    --port 22335 \
    --mode $mode \
    --dataset $dataset \
    --topk 10 \
    --seed 0 \
    --block_size 1 \
    --prefix_name $prefix_name
