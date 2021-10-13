#!/bin/bash

# ========== metadata ========== #
dataset=$1
recall_mode=$2
cuda=$3
# ========== metadata ========== #


# NOTE: Make sure the ./build_es_index.sh has been done (q-q/q-r)

CUDA_VISIBLE_DEVICES=$cuda python test.py \
    --dataset $dataset \
    --model dual-bert \
    --multi_gpu $cuda \
    --mode es_recall \
    --recall_mode $recall_mode \
    --log
