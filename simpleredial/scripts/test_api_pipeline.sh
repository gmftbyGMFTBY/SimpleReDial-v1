#!/bin/bash

# size is 1000, because the test size is 1000 in restoration-200k corpus test set
dataset=$1
prefix_name=$2
python test_api.py \
    --size 1000 \
    --url 9.91.66.241 \
    --port 22335 \
    --mode pipeline \
    --dataset $dataset \
    --topk 100 \
    --seed 0 \
    --block_size 1 \
    --prefix_name $prefix_name
