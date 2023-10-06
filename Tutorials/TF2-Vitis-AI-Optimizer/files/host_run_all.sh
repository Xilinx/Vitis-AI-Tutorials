#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: daniele.bagni@amd.com
# Date:   04 Aug. 2023

: '

# run the patch
tar -xvf patch.tar.gz
cd ./patch
source ./run_patch.sh
cd ..
'

# set the environment
source ./scripts/setup_env.sh


#===========================================================================================

echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO] RESNET18 TRAINED ON CIFAR10 DATASET OF IMAGES (32x32x3)"
echo "----------------------------------------------------------------------------------"
echo " "

# Custom CNN: ResNet18
echo " "
echo " ResNet18 trained on CIFAR10"
echo " "
## first start with a soft link
ln -nsf ./cifar10_customcnn ./cifar10_resnet18
cd ${WRK_DIR}/files/cifar10_resnet18
## clean folders
rm -rf .vai
rm -rf ./log
mkdir -p ./log
rm -rf build_*
#rm -rf build_np/compiled_model_*
#rm -rf build_np/quant_model
#rm -rf build_np/target_*
#rm -rf build_np/logs
#rm -rf build_pr/compiled_model_*
#rm -rf build_pr/quant_model
#rm -rf build_pr/pruned_model
#rm -rf build_pr/transform_model
#rm -rf build_pr/target_*
#rm -rf build_pr/logs

# organize CIFAR10  data
mkdir -p dataset
mkdir -p dataset/cifar10
python3  ./cifar10_generate_images.py

echo "----------------------------------"
echo "[DB INFO] RESNET18   BASELINE FLOW"
echo "----------------------------------"
echo " "
source ./cifar10_run_baseline.sh main ResNet18 2>&1 | tee ./log/logfile_cifar10_resnet18_baseline.txt


echo " "
echo "---------------------------------"
echo "[DB INFO] RESNET18 PRUNING FLOW"
echo "---------------------------------"
echo " "
source ./cifar10_run_pruning.sh main ResNet18 2>&1 | tee ./log/logfile_cifar10_resnet18_pruned.txt


## now  make an archive and remove softlink
cd ${WRK_DIR}/files
tar -cvf cifar10_resnet18.tar ./cifar10_resnet18/*  > /dev/null
rm cifar10_resnet18

#===========================================================================================

echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO] CUSTOM VGGNET TRAINED ON CIFAR10 DATASET OF IMAGES (32x32x3)"
echo "----------------------------------------------------------------------------------"
echo " "

## Custom CNN: miniVggNet
echo " "
echo " miniVggNet trained on CIFAR10"
echo " "
## first start with a soft link
ln -nsf ./cifar10_customcnn ./cifar10_minivggnet
cd ${WRK_DIR}/files/cifar10_minivggnet
## clean folders
rm -rf .vai
rm -rf ./log
mkdir ./log
rm -rf build_*
#rm -rf build_np/compiled_model_*
#rm -rf build_np/quant_model
#rm -rf build_np/target_*
#rm -rf build_np/logs
#rm -rf build_pr/compiled_model_*
#rm -rf build_pr/quant_model
#rm -rf build_pr/pruned_model
#rm -rf build_pr/transform_model
#rm -rf build_pr/target_*
#rm -rf build_pr/logs

## organize CIFAR10  data
#mkdir -p dataset
#mkdir -p dataset/cifar10
#python3  ./cifar10_generate_images.py

echo "----------------------------------"
echo "[DB INFO] MINIVGGNET BASELINE FLOW"
echo "----------------------------------"
echo " "
source ./cifar10_run_baseline.sh main miniVggNet 2>&1 | tee ./log/logfile_cifar10_minivggnet_baseline.txt

echo " "
echo "---------------------------------"
echo "[DB INFO] MINIVGGNET PRUNING FLOW"
echo "---------------------------------"
echo " "
source ./cifar10_run_pruning.sh main miniVggNet 2>&1 | tee ./log/logfile_cifar10_minivggnet_pruned.txt

## now  make an archive and remove softlink
cd ${WRK_DIR}/files
tar -cvf cifar10_minivggnet.tar ./cifar10_minivggnet/* > /dev/null
rm cifar10_minivggnet


#===========================================================================================

echo " "
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO] MOBILENETV2 TRAINED ON DOGS vs. CATS DATASET OF IMAGES (224x224x3)"
echo "----------------------------------------------------------------------------------"
echo " "
echo " "
cd ${WRK_DIR}/files/dogs-vs-cats_mobilenetv2
rm -rf .vai
source ./mobilenetv2_run_all.sh 2>&1 | tee ./log/logfile_mobilenetv2_run_all.txt


#===========================================================================================
# group the applications per target board
#===========================================================================================
cd ${WRK_DIR}/files
tar -xvf cifar10_minivggnet.tar > /dev/null
tar -xvf cifar10_resnet18.tar   > /dev/null
# vck190
echo "preparing vck190 archives"
tar -cvf vck190.tar ./cifar10_minivggnet/vck190_build_*.tar ./cifar10_resnet18/vck190_build_*.tar ./dogs-vs-cats_mobilenetv2/vck190_build_*.tar > /dev/null
# zcu102
echo "preparing zcu102 archives"
tar -cvf zcu102.tar ./cifar10_minivggnet/zcu102_build_*.tar ./cifar10_resnet18/zcu102_build_*.tar ./dogs-vs-cats_mobilenetv2/zcu102_build_*.tar > /dev/null
# vek280
echo "preparing vek280 archives"
tar -cvf vek280.tar ./cifar10_minivggnet/vek280_build_*.tar ./cifar10_resnet18/vek280_build_*.tar ./dogs-vs-cats_mobilenetv2/vek280_build_*.tar > /dev/null

# clean directories
rm -r cifar10_minivggnet
rm -r cifar10_resnet18
