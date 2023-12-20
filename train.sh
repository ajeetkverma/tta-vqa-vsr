#!/usr/bin/env bash

# This script is used to train a model. More specific setttings can
# be found and modified in a train.yml file under the experiment dir


# basic settings
degradation=SO
model=TecoGAN
exp_id=texture04x_1
gpu_id=0


exp_dir=./experiments_${degradation}/${model}/${exp_id}


# run
CUDA_VISIBLE_DEVICES=0 python -W ignore ./codes/main.py \
--exp_dir ${exp_dir} \
--mode train \
--model ${model} \
--opt train.yml \
--gpu_id ${gpu_id}