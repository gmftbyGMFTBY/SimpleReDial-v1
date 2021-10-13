#!/bin/bash

mode=$1    # init / retrieval
python process.py --seed 50 --name lccc-large --database_size 5000000 --mode $mode --samples 10
