#!/bin/bash

#Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# date 28 Apr 2023


# build cifar10 test images
bash ./code/build_cifar10_test.sh


# compile CNN application
cd code
bash -x ./build_app.sh
mv code ../run_cnn # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ..

# now run the cifar10 classification with 4 CNNs uing VART C++ APIs
./run_cnn ./cifar10/LeNet/LeNet.xmodel                 ./cifar10_test/ ./code/cifar10_labels.txt | tee ./rpt/logfile_cifar10_LeNet.txt
./run_cnn ./cifar10/miniVggNet/miniVggNet.xmodel       ./cifar10_test/ ./code/cifar10_labels.txt | tee ./rpt/logfile_cifar10_miniVggNet.txt
./run_cnn ./cifar10/miniGoogleNet/miniGoogleNet.xmodel ./cifar10_test/ ./code/cifar10_labels.txt | tee ./rpt/logfile_cifar10_miniGoogleNet.txt
./run_cnn ./cifar10/miniResNet/miniResNet.xmodel       ./cifar10_test/ ./code/cifar10_labels.txt | tee ./rpt/logfile_cifar10_miniResNet.txt

# check DPU prediction accuracy
bash -x ./code/check_cifar10_accuracy.sh | tee ./rpt/summary_cifar10_prediction_accuracy_on_dpu.txt


## run fps measurements
bash -x ./code/fps_cifar10.sh | tee ./rpt/logfile_cifar10_fps.txt


#remove images
rm -r cifar10_test
