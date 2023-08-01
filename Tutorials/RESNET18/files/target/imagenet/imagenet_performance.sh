#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

TARGET=$1

# check DPU prediction top1_accuracy
echo " "
echo " IMAGENET RESNET50 TOP1 ACCURACY ON DPU"
echo " "
python3 ./code_resnet50/src/check_runtime_top1_imagenet.py \
	-i ./rpt/predictions_resnet50_imagenet.log
echo " "
echo " "
echo " IMAGENET RESNET18 TOP1 ACCURACY ON DPU"
echo " "
python3 ./code_resnet50/src/check_runtime_top1_imagenet.py \
	-i ./rpt/predictions_resnet18_imagenet.log
echo " "


echo " "
echo " IMAGENET RESNET18 PERFORMANCE (fps)"
echo " "
./get_dpu_fps ./${TARGET}_resnet18_imagenet.xmodel  1 1000  | tee  ./rpt/log1.txt  # 1 thread
./get_dpu_fps ./${TARGET}_resnet18_imagenet.xmodel  2 1000  | tee  ./rpt/log2.txt  # 2 threads
./get_dpu_fps ./${TARGET}_resnet18_imagenet.xmodel  3 1000  | tee  ./rpt/log3.txt  # 3 threads
cat ./rpt/log1.txt ./rpt/log2.txt ./rpt/log3.txt >  ./rpt/${TARGET}_resnet18_imagenet_results_fps.log
rm -f ./rpt/log?.txt

echo " "
echo " IMAGENET RESNET50 PERFORMANCE (fps)"
echo " "
./get_dpu_fps ./${TARGET}_resnet50_imagenet.xmodel  1 1000  | tee  ./rpt/log1.txt  # 1 thread
./get_dpu_fps ./${TARGET}_resnet50_imagenet.xmodel  2 1000  | tee  ./rpt/log2.txt  # 2 threads
./get_dpu_fps ./${TARGET}_resnet50_imagenet.xmodel  3 1000  | tee  ./rpt/log3.txt  # 3 threads
cat ./rpt/log1.txt ./rpt/log2.txt ./rpt/log3.txt >  ./rpt/${TARGET}_resnet50_imagenet_results_fps.log
rm -f ./rpt/log?.txt

echo " "
