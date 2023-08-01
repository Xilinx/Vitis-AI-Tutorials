#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Activate environment..."
# conda activate color_cls

DATA_DIR=./build/data/
#WEIGHTS=./pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/float
WEIGHTS=./build/float
DATASET=vcor
GPU_ID=0
#QUANT_DIR=${QUANT_DIR:-quantized}
QUANT_DIR=./build/quantized
export PYTHONPATH=${PWD}:${PYTHONPATH}


echo "Conducting Quantization"
# fix calib
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --backbone resnet18 --resume ${WEIGHTS}/color_last_resnet18.pt --data_root ${DATA_DIR}/${DATASET} --quant_mode calib --quant_dir=${QUANT_DIR}

# fix test
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --backbone resnet18 --resume ${WEIGHTS}/color_last_resnet18.pt --data_root ${DATA_DIR}/${DATASET} --quant_mode test  --quant_dir=${QUANT_DIR}

# deploy
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py --backbone resnet18 --resume ${WEIGHTS}/color_last_resnet18.pt --data_root ${DATA_DIR}/${DATASET} --quant_mode test  --quant_dir=${QUANT_DIR} --deploy --device=cpu
