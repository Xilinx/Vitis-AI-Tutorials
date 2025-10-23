#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# last change: 12 Sep 2025

DATA_DIR=./build/dataset
#WEIGHTS=./pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/float
WEIGHTS=./build/float
DATASET=vcor
GPU_ID=0

export PYTHONPATH=${PWD}:${PYTHONPATH}


printf "\n"
printf "TEST ON FLOATING POINT MODEL: START\n"
printf "\n"

# float test
python3 code/test.py --backbone resnet18 --resume ${WEIGHTS}/color_resnet18.pt --data_root ${DATA_DIR}/${DATASET} --device cpu
python code/test.py  --batch-size 50 --test-batch-size 5--backbone resnet18 --resume ${WEIGHTS}/color_last_resnet18.pt --data_root ${DATA_DIR}/${DATASET}


printf "\n"
printf "TEST ON FLOATING POINT MODEL: END\n"
printf "\n"