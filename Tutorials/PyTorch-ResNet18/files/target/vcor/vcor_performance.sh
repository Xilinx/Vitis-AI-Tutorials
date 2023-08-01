#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

TARGET=$1

# check DPU prediction top1_accuracy
echo " "
echo " VCOR RESNET18 TOP5 ACCURACY"
echo " "
python3 ./code/src/check_runtime_top5_vcor.py -i ./rpt/predictions_vcor_resnet18.log | tee ./rpt/results_predictions.log


echo " "
echo " CIFAR10 RESNET18 PERFORMANCE (fps)"
echo " "
./get_dpu_fps ./${TARGET}_train_resnet18_vcor.xmodel  1 10000  | tee  ./rpt/log1.txt  # 1 thread
./get_dpu_fps ./${TARGET}_train_resnet18_vcor.xmodel  2 10000  | tee  ./rpt/log2.txt  # 2 threads
./get_dpu_fps ./${TARGET}_train_resnet18_vcor.xmodel  3 10000  | tee  ./rpt/log3.txt  # 3 threads
cat ./rpt/log1.txt ./rpt/log2.txt ./rpt/log3.txt >  ./rpt/${TARGET}_train_resnet18_vcor_results_fps.log
rm -f ./rpt/log?.txt

echo " "
