#!/bin/bash

dataset=$1
english_datasets=(ubuntu)
chinese_datasets=(douban ecommerce)
if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    lang=zh
else
    lang=en
fi

python extraction_keywords.py \
    --size 50000 \
    --lang $lang \
    --dataset $dataset

