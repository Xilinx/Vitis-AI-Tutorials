#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, AMD
# date:  28 July 2023

# Build script for baseline and pruning flows

## enable TensorFlow2 environment
#conda activate vitis-ai-tensorflow2


echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO]  CREATING CIFAR10 DATASET OF IMAGES"
echo "----------------------------------------------------------------------------------"
echo " "
# organize CIFAR10  data
#mkdir -p dataset
#mkdir -p dataset/cifar10
#python  ./cifar10_generate_images.py


echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO]  BASELINE FLOW"
echo "----------------------------------------------------------------------------------"
echo " "
source ./cifar10_run_baseline.sh main $1

: '
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO]  PRUNING FLOW"
echo "----------------------------------------------------------------------------------"
echo " "
source ./cifar10_run_pruning.sh main
'
