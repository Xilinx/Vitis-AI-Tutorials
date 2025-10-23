#!/bin/bash

# ===========================================================
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License
# ===========================================================

# Date 29 Sep. 2025

export WRK_DIR=~/Public/VAI5.1/bash
export VITIS_AI_REPO=${WRK_DIR}/Vitis-AI

CUR_DIR=${PWD}
echo "from"
echo "PWD         = " ${PWD}
echo "CURRENT DIR = " ${CUR_DIR}
SDK_DIR=${WRK_DIR}/sdk_vai5.1

cd ${SDK_DIR}
echo "to "
echo "SDK DIR     = " ${SDK_DIR}
echo "PWD         = " ${PWD}
unset LD_LIBRARY_PATH
source ./environment-setup-cortexa72-cortexa53-amd-linux
echo "source ./environment-setup-cortexa72-cortexa53-amd-linux"

cd ${CUR_DIR}
echo "back to"
echo "CUR_DIR     = " ${CUR_DIR}
echo "PWD         = " ${PWD}
