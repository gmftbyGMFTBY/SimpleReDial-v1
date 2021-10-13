#!/bin/bash

# ./create_post_train_dataset.sh <dataset_name>
dataset=$1
chinese_datasets=(douban ecommerce)
if [[ ${chinese_datasets[@]} =~ $dataset ]]; then
    ckpt=bert-base-chinese
else
    ckpt=bert-base-uncased
fi
python data/create_post_training_data.py \
    --input_file ./data/$dataset/train_post.txt \
    --output_file ./data/$dataset/train_post.hdf5 \
    --bert_pretrained $ckpt \
    --dataset $dataset \
    --dupe_factor 10
