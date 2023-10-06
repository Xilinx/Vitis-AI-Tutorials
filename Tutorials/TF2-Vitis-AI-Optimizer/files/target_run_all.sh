#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: daniele.bagni@amd.com
# Date:   03 Aug. 2023

BOARD=$1

cd ~


tar -xvf ${BOARD}.tar 

# unpack everything
cd cifar10_resnet18/
tar -xvf ./${BOARD}_build_np.tar  > /dev/null
tar -xvf ./${BOARD}_build_pr.tar  > /dev/null
cd ..
cd cifar10_minivggnet/
tar -xvf ./${BOARD}_build_pr.tar  > /dev/null
tar -xvf ./${BOARD}_build_np.tar  > /dev/null
cd ../dogs-vs-cats_mobilenetv2/
tar -xvf ./${BOARD}_build_np.tar  > /dev/null
tar -xvf ./${BOARD}_build_pr.tar  > /dev/null

# run miniVggNet
cd ../cifar10_minivggnet/
cd build_np/target_${BOARD}/
source ./run_all_cifar10_target.sh 2>&1 | tee ~/${BOARD}_cifar10_minivggnet_np.log
cd ../../build_pr/target_${BOARD}/
source ./run_all_cifar10_target.sh 2>&1 | tee ~/${BOARD}_cifar10_minivggnet_pr.log
cd


# run ResNet18
cd cifar10_resnet18/
cd build_np/target_${BOARD}/
source ./run_all_cifar10_target.sh 2>&1 | tee ~/${BOARD}_cifar10_resnet18_np.log
cd ../../build_pr/target_${BOARD}/
source ./run_all_cifar10_target.sh 2>&1 | tee ~/${BOARD}_cifar10_resnet18_pr.log
cd 

# run MobileNetV2
cd dogs-vs-cats_mobilenetv2/
cd build_np/target_${BOARD}/
source ./run_all_mobilenetv2_target.sh 2>&1 | tee ~/${BOARD}_mobilenetv2_np.log
cd ../../build_pr/target_${BOARD}/
source ./run_all_mobilenetv2_target.sh 2>&1 | tee ~/${BOARD}_mobilenetv2_pr.log
cd


tar -cvf ${BOARD}_log.tar ${BOARD}_*.log


