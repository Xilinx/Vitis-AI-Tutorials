#!/bin/bash

#Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# date 28 Apr 2023

# check DPU prediction top1_accuracy
python3 ./code/src/check_runtime_top5_cifar10.py -i ./rpt/logfile_cifar10_LeNet.txt
python3 ./code/src/check_runtime_top5_cifar10.py -i ./rpt/logfile_cifar10_miniVggNet.txt
python3 ./code/src/check_runtime_top5_cifar10.py -i ./rpt/logfile_cifar10_miniGoogleNet.txt
python3 ./code/src/check_runtime_top5_cifar10.py -i ./rpt/logfile_cifar10_miniResNet.txt
