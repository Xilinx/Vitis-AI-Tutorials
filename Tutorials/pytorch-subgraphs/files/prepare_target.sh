#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


echo " "
echo "-----------------------------------------"
echo "PREPARE TARGET FOLDER."
echo "-----------------------------------------"

#prepare test images
python -c "from common import *; generate_images('./build/target/test_images')" 2>&1 | tee ./build/log/generate_test_images.log

#copy xmodel
cp -r ./application ./build/target
cp ./build/compiled_model/CNN_int_*.xmodel ./build/target/application

# launch functional debug
python -c "import common; dbg_model=common.RUN_CNN_DEBUG('./build/target/application/test')" 2>&1 | tee ./build/log/functional_debug_on_host.log
mv *.bin.npy ./build/target/application/

# make a tar file
cd ./build
tar -cvf target_$1.tar ./target &> /dev/null
cd ..
