#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
file_tag=$3     # 22335,22336
cuda=$4
# ========== metadata ========== #

CUDA_VISIBLE_DEVICES=$cuda python test.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --file_tags $file_tag \
    --mode compare
