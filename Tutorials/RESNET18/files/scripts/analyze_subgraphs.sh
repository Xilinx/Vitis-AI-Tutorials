#!/bin/bash

# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


if [ $1 == zcu102 ]; then
    TARGET=zcu102
elif [ $1 == u50 ]; then
    TARGET=u50
elif [ $1 == vck190 ]; then
    TARGET=vck190
elif [ $1 == vek280 ]; then
    TARGET=vek280
else
    echo  "Target not found. Valid choices are: zcu102, u50, vck190...exiting"
    exit 1
fi

PRENAME=$2

# analyze DPU/CPU subgraphs
xir    graph ./build/compiled_${TARGET}/${TARGET}_${PRENAME}.xmodel 2>&1 | tee ./build/log/${TARGET}_${PRENAME}_graph_info.txt
xir subgraph ./build/compiled_${TARGET}/${TARGET}_${PRENAME}.xmodel 2>&1 | tee ./build/log/${TARGET}_${PRENAME}_subgraph_tree.txt
xir dump_txt ./build/compiled_${TARGET}/${TARGET}_${PRENAME}.xmodel 2>&1 | tee ./build/log/${TARGET}_${PRENAME}_dump_xmodel.txt
xir svg      ./build/compiled_${TARGET}/${TARGET}_${PRENAME}.xmodel            ./build/log/${TARGET}_${PRENAME}_cnn2d_xmodel.svg
xir png      ./build/compiled_${TARGET}/${TARGET}_${PRENAME}.xmodel            ./build/log/${TARGET}_${PRENAME}_cnn2d_xmodel.png
