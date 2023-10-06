#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 July  2023

#cd ./TF2-VAI-Optimizer/dogs-vs-cats_mobilenetv2

xdputil query

# =======================================================================
# run python app
# =======================================================================

python3 app_mt.py -t 1
python3 app_mt.py -t 6
python3 app_mt.py -t 8

# =======================================================================
# build and run C++ app
# =======================================================================

cd ./application/code/

# build the app to measure fps performance
bash -x ./build_get_dpu_fps.sh
mv code get_dpu_fps

# =======================================================================
# run CNN

# get its fps performance
./get_dpu_fps ../../mobilenetv2.xmodel 1 5000
./get_dpu_fps ../../mobilenetv2.xmodel 3 5000
./get_dpu_fps ../../mobilenetv2.xmodel 6 5000
