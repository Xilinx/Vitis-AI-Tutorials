#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

echo " "
echo "COMPILING HOST APPS"
echo " "
cd host_apps
#make clean
make all

echo " "
echo "COMPILING IP CORES"
echo " "
cd ../ip
#make clean
make all
cd ..
