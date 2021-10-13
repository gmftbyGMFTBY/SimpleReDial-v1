#!/bin/bash

# init mode: create the train dataset and save the candidate responses into elasticsearch index
# retrieval mode: build the test dataset and search the hard negative samples for each test sample, which is the same as the previous corpus, such as Douban, Ecommerce, Ubuntu
mode=$1    # init / retrieval
python process.py --seed 50 --name lccc --train_size 500000 --test_size 1000 --database_size 1000000 --mode $mode --samples 10
