#!/bin/bash

# only use ong GPU device
cuda=$1
CUDA_VISIBLE_DEVICES=$cuda python deploy.py
