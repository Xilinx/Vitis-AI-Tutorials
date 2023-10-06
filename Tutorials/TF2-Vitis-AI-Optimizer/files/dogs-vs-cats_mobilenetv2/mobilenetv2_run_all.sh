#!/bin/sh

## Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Authors: Daniele Bagni and Mark Harvey
# date:  28 July 2023

# Build script for baseline and pruning flows

## enable TensorFlow2 environment
#conda activate vitis-ai-tensorflow2


# ****************************************************************


echo "-----------------------------------------"
echo "[DB INFO] CONVERT DATASET TO TFRECORDS"
echo "-----------------------------------------"
python -u images_to_tfrec.py 2>&1 | tee ../log/tfrec.log


echo " "
echo "----------------------------------------------------------------------------------"
echo "  BASELINE FLOW"
echo "----------------------------------------------------------------------------------"
echo " "
source ./run_baseline.sh main


echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO]  PRUNING FLOW"
echo "----------------------------------------------------------------------------------"
echo " "
source ./run_pruning.sh main
