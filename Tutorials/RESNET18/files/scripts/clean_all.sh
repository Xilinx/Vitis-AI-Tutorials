#!/bin/sh

## Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
## SPDX-License-Identifier: MIT

## Author: Daniele Bagni, AMD/Xilinx Inc

## date 08 Aug 2023


# clean folders
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP1A] CLEANING FOLDERS"
echo "----------------------------------------------------------------------------------"
for file in $(find . -name "*.fuse_hidden*"); do
    echo ${file}
    rm -f ${file}
done
for file in $(find . -name "*.*~"); do
    echo ${file}
    rm -f ${file}
done
rm -r ./build/compiled_v70
rm -r ./build/compiled_vck190
rm -r ./build/compiled_zcu102
rm -r ./build/compiled_vek280
rm -r ./build/compiled_vck5000
rm -r ./build/log
rm -r ./build/quantized
rm -r ./build/target*
rm -rf ./modelzoo/*resnet50*
rm -rf ./modelzoo/ImageNet/val_dataset
#rm -rf ./modelzoo/ImageNet/ILSVRC2012_img_val.tar
rm -f  ./modelzoo/ImageNet/val_dataset.zip
rm -f  ./target/imagenet/val_dataset.zip
### if you want to rebuild the CIFAR10 dataset uncomment next line
#rm -r ./build/dataset
# clean target folder
cd ./target/cifar10
rm -f *.xmodel
rm -f ./rpt/*
rm -f ./cnn_resnet18_cifar10
rm -f ./get_dpu_fps
cd ../imagenet
rm -f *.xmodel
rm -f ./rpt/*
rm -f ./resnet*imagenet
rm -f ./get_dpu_fps
rm -rf val_dataset
cd ../../

### if you uncomment next three lines, also the trained models will be deleted!
rm -r ./build/float
mkdir -p ./build/float
mkdir -p ./build/dataset

mkdir -p ./build/quantized
mkdir -p ./build/log
mkdir -p ./build/compiled_v70
mkdir -p ./build/compiled_vck190
mkdir -p ./build/compiled_zcu102
mkdir -p ./build/compiled_vek280
mkdir -p ./build/compiled_vck5000


: '
# dos2unix to all txt files
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP1B] DOS2UNIX"
echo "----------------------------------------------------------------------------------"

pip install dos2unix

for file in $(find . -name "*.sh"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name "*.py"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name "*.c*"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name "*.h"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
'
