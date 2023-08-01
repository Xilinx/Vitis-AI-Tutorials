#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


if [ $1 = zcu102 ]; then
    TARGET=zcu102
elif [ $1 = u50 ]; then
    TARGET=u50
elif [ $1 = vck190 ]; then
    TARGET=vck190
elif [ $1 = vek280 ]; then
    TARGET=vek280
else
    echo  "Target not found. Valid choices are: zcu102, u50, vck190...exiting"
    exit 1
fi


# analyze DPU/CPU subgraphs
xir    graph ./build/compiled_model/CNN_int_${TARGET}.xmodel 2>&1 | tee ./build/log/cnn_int8_graph_info.txt
xir subgraph ./build/compiled_model/CNN_int_${TARGET}.xmodel 2>&1 | tee ./build/log/cnn_int8_subgraph_tree.txt
xir dump_txt ./build/compiled_model/CNN_int_${TARGET}.xmodel            ./build/log/cnn_int8_dump_xmodel.txt
xir png      ./build/compiled_model/CNN_int_${TARGET}.xmodel            ./build/log/cnn_int8_xmodel.png
xir svg      ./build/compiled_model/CNN_int_${TARGET}.xmodel            ./build/log/cnn_int8_xmodel.svg
