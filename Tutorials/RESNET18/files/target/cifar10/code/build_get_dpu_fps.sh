#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}
os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
arch=`uname -p`
target_info=${os}.${os_version}.${arch}
install_prefix_default=$HOME/.local/${target_info}
$CXX --version

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

name=$(basename $PWD)
if [[ "$CXX"  == *"sysroot"* ]];then
$CXX -O2 -fno-inline -I. \
     -I=/usr/include/opencv4 \
     -I=/install/Debug/include \
     -I=/install/Release/include \
     -L=/install/Debug/lib \
     -L=/install/Release/lib \
     -I$PWD/../../common  -o $name -std=c++17 \
     $PWD/src/get_dpu_fps.cc \
     $PWD/../../common/common.cpp  \
     -Wl,-rpath=$PWD/lib \
     -lvart-runner \
     ${OPENCV_FLAGS} \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread
else
$CXX -O2 -fno-inline -I. \
     -I${install_prefix_default}.Debug/include \
     -I${install_prefix_default}.Release/include \
     -L${install_prefix_default}.Debug/lib \
     -L${install_prefix_default}.Release/lib \
     -Wl,-rpath=${install_prefix_default}.Debug/lib \
     -Wl,-rpath=${install_prefix_default}.Release/lib \
     -I$PWD/../../common  -o $name -std=c++17 \
     $PWD/src/get_dpu_fps.cc \
     $PWD/../../common/common.cpp  \
     -Wl,-rpath=$PWD/lib \
     -lvart-runner \
     ${OPENCV_FLAGS} \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread
fi
