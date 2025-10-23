#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# last change 12 Sep 2025

printf "\n"
printf "\n TRAINING START \n"
printf "\n"

DATA_DIR=./build/dataset/
WEIGHTS=./build/float
DATASET=vcor
GPU_ID=0
export PYTHONPATH=${PWD}:${PYTHONPATH}

python3 code/train.py --batch-size 512  --epochs 36 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}

#python code/train.py  --batch-size 50 --test-batch-size 5 --epochs 10 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}
# mv ./build/float ./build/float_1.5Mpix
# mkdir ./build/float
# python code/train.py --resume ./build/float_1.5Mpix/color_resnet18.pt --batch-size 50 --test-batch-size 5 --epochs 10 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}


printf "\n"
printf "\n TRAINING END \n"
printf "\n"
