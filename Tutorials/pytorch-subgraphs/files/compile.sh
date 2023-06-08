#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102..."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104..."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      TARGET=vck190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VCK190..."
      echo "-----------------------------------------"
elif [ $1 = vek280 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCV2DX8G/VEK280/arch.json
      TARGET=vek280
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VEK280..."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu104, zcu102, vck190, vek280 ... exiting"
      exit 1
fi


compile() {
  vai_c_xir \
  --xmodel      ./build/quantized_model/CNN_int.xmodel \
  --arch        $ARCH \
  --net_name    CNN_int_${TARGET}\
  --output_dir  ./build/compiled_model
}

compile    2>&1 | tee ./build/log/compile_${TARGET}.log

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
