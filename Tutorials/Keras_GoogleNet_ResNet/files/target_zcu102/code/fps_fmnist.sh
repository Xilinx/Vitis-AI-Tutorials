#!/bin/bash

#Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# date 28 Apr 2023


echo " "
echo " FMNIST LEnet fps"
echo " "
./get_dpu_fps ./fmnist/LeNet/LeNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/LeNet/LeNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/LeNet/LeNet.xmodel  6 10000  # 6 threads

echo " "
echo " FMNIST miniVGGnet fps"
echo " "
./get_dpu_fps ./fmnist/miniVggNet/miniVggNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/miniVggNet/miniVggNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/miniVggNet/miniVggNet.xmodel  6 10000  # 6 threads


echo " "
echo " FMNIST miniGOOGLEnet fps"
echo " "
./get_dpu_fps ./fmnist/miniGoogleNet/miniGoogleNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/miniGoogleNet/miniGoogleNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/miniGoogleNet/miniGoogleNet.xmodel  6 10000  # 6 threads

echo " "
echo " FMNIST miniRESnet fps"
echo " "
./get_dpu_fps ./fmnist/miniResNet/miniResNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/miniResNet/miniResNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/miniResNet/miniResNet.xmodel  6 10000  # 6 threads
