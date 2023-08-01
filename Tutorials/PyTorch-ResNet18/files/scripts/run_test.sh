#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

DATA_DIR=./build/data
#WEIGHTS=./pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/float
WEIGHTS=./build/float
DATASET=vcor
GPU_ID=0

export PYTHONPATH=${PWD}:${PYTHONPATH}


echo " "
echo "Conducting testing with floating point CNN"
echo " "
# float test
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --backbone resnet18 --resume ${WEIGHTS}/color_last_resnet18.pt --data_root ${DATA_DIR}/${DATASET}
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py  --batch-size 50 --test-batch-size 5--backbone resnet18 --resume ${WEIGHTS}/color_last_resnet18.pt --data_root ${DATA_DIR}/${DATASET}
