#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  10 Aug. 2023


source ./cifar10/run_all_cifar10_target.sh   main $1
source ./imagenet/run_all_imagenet_target.sh main $1
