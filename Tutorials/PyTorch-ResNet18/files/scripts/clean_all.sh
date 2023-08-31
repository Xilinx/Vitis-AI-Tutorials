#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

## Author: Daniele Bagni, AMD/Xilinx Inc
## date 11 Aug 2023


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


rm -r ./build/compiled_vck190
rm -r ./build/compiled_zcu102
rm -r ./build/compiled_vek280
rm -r ./build/compiled_vck5000
rm -r ./build/compiled_v70
rm -r ./build/quantized
rm -r ./build/target*
# clean target folder
cd ./target/vcor
rm -f *.xmodel
rm -f ./rpt/*
rm -f ./cnn_resnet18_vcor
rm -f ./get_dpu_fps
cd ../../

# if you remove it also the trained models will be deleted!
rm -r ./build/log
rm -r ./build/float
mkdir -p ./build/float
rm -r ./build/data
mkdir -p ./build/data
mkdir -p ./build/data/vcor
mkdir -p ./build/log

mkdir -p ./build/quantized
mkdir -p ./build/compiled_vck190
mkdir -p ./build/compiled_zcu102
mkdir -p ./build/compiled_vek280
mkdir -p ./build/compiled_vck5000
mkdir -p ./build/compiled_v70


# dos2unix to all txt files
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP1B] DOS2UNIX"
echo "----------------------------------------------------------------------------------"

#pip install dos2unix
for file in $(find . -name "*.sh"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name "*.py"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name "*.c*"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name "*.h"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
for file in $(find . -name ".fuse_hidden*"); do
    python /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.8/site-packages/dos2unix.py ${file} ${file}_tmp > /dev/null
    mv ${file}_tmp ${file}
done
