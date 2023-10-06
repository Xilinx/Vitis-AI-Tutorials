#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 July  2023

#cd ./TF2-VAI-Optimizer/cifar10_customcnn/

xdputil query

# =======================================================================
# run python app
# =======================================================================

python3 cifar10_app_mt.py -t 1 -m *.xmodel
python3 cifar10_app_mt.py -t 6 -m *.xmodel
python3 cifar10_app_mt.py -t 8 -m *.xmodel

# =======================================================================
# build and run C++ app
# =======================================================================

cd ./code/

# build the app to measure prediction accuracy
bash -x ./build_app.sh
mv code run_cnn

# build the app to measure fps performance
bash -x ./build_get_dpu_fps.sh
mv code get_dpu_fps

# =======================================================================
# run CNN

# get its prediction accuracy
./run_cnn ../*.xmodel ../test/ ./cifar10_labels.dat  | tee logfile_minicnn.txt
python3 ./src/check_runtime_top5_cifar10.py -i ./logfile_minicnn.txt | tee logfile_predictions_minicnn.txt
# get its fps performance
./get_dpu_fps ../*.xmodel 1 10000
./get_dpu_fps ../*.xmodel 3 10000
./get_dpu_fps ../*.xmodel 6 10000
