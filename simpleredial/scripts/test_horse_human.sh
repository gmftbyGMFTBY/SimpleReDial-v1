#!/bin/bash

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

CUDA_VISIBLE_DEVICES=$cuda python test.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --mode horse_human
