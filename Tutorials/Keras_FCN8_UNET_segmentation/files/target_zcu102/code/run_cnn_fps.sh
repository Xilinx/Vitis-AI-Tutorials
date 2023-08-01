#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


# now run semantic segmentation with 3  multithreads
echo " "
echo " FCN8 fps"
echo " "
./get_dpu_fps ./fcn8/model/fcn8.xmodel 1 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 2 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 3 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 4 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 5 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 6 1000

echo " "
echo " FCN8ups fps"
echo " "
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 1 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 2 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 3 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 4 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 5 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 6 1000

echo " "
echo " UNET  fps"
echo " "
./get_dpu_fps ./unet/v2/model/unet2.xmodel 1 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 2 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 3 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 4 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 5 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 6 1000
'
