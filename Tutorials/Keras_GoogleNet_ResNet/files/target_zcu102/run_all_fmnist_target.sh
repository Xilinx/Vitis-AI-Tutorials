#!/bin/bash

#Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# date 28 Apr 2023

# build fmnist test images
bash ./code/build_fmnist_test.sh

: '
## compile CNN application
#cd code
#bash -x ./build_app.sh
#mv code ../run_cnn # change name of the application
#bash -x ./build_get_dpu_fps.sh
#mv code ../get_dpu_fps
#cd ..
'

# now run the fmnist classification with 4 CNNs uing VART C++ APIs
./run_cnn ./fmnist/LeNet/LeNet.xmodel                 ./fmnist_test/ ./code/fmnist_labels.txt | tee ./rpt/logfile_fmnist_LeNet.txt
./run_cnn ./fmnist/miniVggNet/miniVggNet.xmodel       ./fmnist_test/ ./code/fmnist_labels.txt | tee ./rpt/logfile_fmnist_miniVggNet.txt
./run_cnn ./fmnist/miniGoogleNet/miniGoogleNet.xmodel ./fmnist_test/ ./code/fmnist_labels.txt | tee ./rpt/logfile_fmnist_miniGoogleNet.txt
./run_cnn ./fmnist/miniResNet/miniResNet.xmodel       ./fmnist_test/ ./code/fmnist_labels.txt | tee ./rpt/logfile_fmnist_miniResNet.txt
# check DPU prediction accuracy
bash -x ./code/check_fmnist_accuracy.sh | tee ./rpt/summary_fmnist_prediction_accuracy_on_dpu.txt

# run multithreading Python VART APIs to get fps
bash -x ./code/fps_fmnist.sh | tee ./rpt/logfile_fmnist_fps.txt

#remove images
rm -r fmnist_test
