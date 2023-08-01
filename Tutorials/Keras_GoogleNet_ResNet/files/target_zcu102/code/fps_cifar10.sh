#!/bin/bash

#Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# date 28 Apr 2023


echo " "
echo " CIFAR10 LEnet fps"
echo " "
./get_dpu_fps ./cifar10/LeNet/LeNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./cifar10/LeNet/LeNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./cifar10/LeNet/LeNet.xmodel  6 10000  # 6 threads

echo " "
echo " CIFAR10 miniVGGnet fps"
echo " "
./get_dpu_fps ./cifar10/miniVggNet/miniVggNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./cifar10/miniVggNet/miniVggNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./cifar10/miniVggNet/miniVggNet.xmodel  6 10000  # 6 threads


echo " "
echo " CIFAR10 miniGOOGLEnet fps"
echo " "
./get_dpu_fps ./cifar10/miniGoogleNet/miniGoogleNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./cifar10/miniGoogleNet/miniGoogleNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./cifar10/miniGoogleNet/miniGoogleNet.xmodel  6 10000  # 6 threads

echo " "
echo " CIFAR10 miniRESnet fps"
echo " "
./get_dpu_fps ./cifar10/miniResNet/miniResNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./cifar10/miniResNet/miniResNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./cifar10/miniResNet/miniResNet.xmodel  6 10000  # 6 threads
