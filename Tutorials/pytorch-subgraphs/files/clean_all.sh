#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


# clean folders
echo " "
echo "-----------------------------------------"
echo "CLEANING FOLDERS."
echo "-----------------------------------------"

: '
rm -rf  ./build/quantized_model
rm  -f  ./build/data/*.pt
rm  -f  ./build/data/*.tar.gz
'


rm -rf   ./build/log*
mkdir -p ./build/log

rm -rf  ./build/compiled_model

rm -rf   ./build/target/*
rm -rf   ./build/target
mkdir -p ./build/target
mkdir -p ./build/target/test_images
