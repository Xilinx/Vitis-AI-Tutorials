#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# clean everything
clean() {
rm -rf ./build ./rpt ./png*
rm -f ./run_cnn get_dpu_fps *.png
mkdir ./png_fcn8
mkdir ./png_fcn8ups
mkdir ./png_unet
mkdir ./rpt
##remove images
rm -rf dataset1
}

# build  test images
dataset() {
tar -xvf test.tar.gz >& /dev/null
mv ./build/dataset1/ .
}

# compile CNN application
compile() {
cd code
bash -x ./build_app.sh
mv code ../run_cnn # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ..
}

# now run semantic segmentation with 3 CNNs using VART C++ APIs with single thread
run_models() {
./run_cnn ./fcn8/model/fcn8.xmodel        ./dataset1/img_test/ 1 1  2> /dev/null | tee ./rpt/logfile_cpp_fcn8.txt
mv *.png ./png_fcn8/
./run_cnn ./fcn8ups/model/fcn8ups.xmodel  ./dataset1/img_test/ 1 1  2> /dev/mull | tee ./rpt/logfile_cpp_fcn8ups.txt
mv *.png ./png_fcn8ups/
./run_cnn ./unet/v2/model/unet2.xmodel    ./dataset1/img_test/ 1 1  2> /dev/null | tee ./rpt/logfile_cpp_unet2.txt
mv *.png ./png_unet/
}

run_fps() {
# get the fps performance  with multithreads
bash -x ./code/run_cnn_fps.sh 2> /dev/null | tee ./rpt/logfile_fps.txt
}

main()
{
  clean
  dataset
  compile
  run_models
  run_fps
}


"$@"
