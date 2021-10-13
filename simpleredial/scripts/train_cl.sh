#!/bin/bash
# curriculumn training

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)
version=$(cat config/base.yaml | shyaml get-value version)

# backup
echo "find root_dir: $root_dir"
echo "find version: $version"
mv $root_dir/ckpt/$dataset/$model/*_$version.pt $root_dir/bak/$dataset/$model
# delete the previous tensorboard file
rm $root_dir/rest/$dataset/$model/$version/* 
rm -rf $root_dir/rest/$dataset/$model/$version 

gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29424 train_curriculum_learning.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda
