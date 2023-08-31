#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo " "

echo "Running Training..."
echo " "

DATA_DIR=./build/data/
#WEIGHTS=./pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/float
WEIGHTS=./build/float
DATASET=vcor
GPU_ID=0
export PYTHONPATH=${PWD}:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train.py --batch-size 512                     --epochs 30 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}

# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train.py                                                 --batch-size 50 --test-batch-size 5 --epochs 10 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}
# mv ./build/float ./build/float_1.5Mpix
# mkdir ./build/float
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train.py --resume ./build/float_1.5Mpix/color_resnet18.pt --batch-size 50 --test-batch-size 5 --epochs 10 --backbone resnet18 --save-model --data_root ${DATA_DIR}${DATASET} --save_dir=${WEIGHTS}
