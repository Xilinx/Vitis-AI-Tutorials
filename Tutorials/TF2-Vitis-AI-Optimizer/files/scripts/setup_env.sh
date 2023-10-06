#!/usr/bin/env bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


# Author: Daniele Bagni, AMD
# date:  27 July 2023


export WRK_DIR=/workspace/tutorials/TF2-Vitis-AI-Optimizer

echo " "
echo "STEP0: SET LICENSE FILE"
echo " "
export XILINXD_LICENSE_FILE=${WRK_DIR}/vai_optimizer.lic


# to suppress too much TF2 verbosity
# see also https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
export  TF_CPP_MIN_LOG_LEVEL=3


# needed by ResNet18
pip3 install image-classifiers
