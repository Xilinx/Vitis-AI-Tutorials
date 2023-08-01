#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023



TARG=vek280
#zcu102
#vck190
#zcu104

# -------------------------------------------------------------------------------
# make sure text file scripts are in "unix" format
# -------------------------------------------------------------------------------


for file in $(find $PWD -name "*.sh"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done
for file in $(find $PWD -name "*.py"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done
for file in $(find $PWD -name "*.c*"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done
for file in $(find $PWD -name "*.h*"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done
for file in $(find $PWD -name "*.m"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done
for file in $(find $PWD -name "*.txt"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done


# -------------------------------------------------------------------------------
# setup
# -------------------------------------------------------------------------------

#conda activate vitis-ai-pytorch
# clean folders
bash ./clean_all.sh

## install missing package in the docker image
#pip install torchsummary

: '
# -------------------------------------------------------------------------------
# run training
# -------------------------------------------------------------------------------
echo " "
echo "-----------------------------------------"
echo "RUN TRAINING."
echo "-----------------------------------------"
python -u train.py --epochs 300  2>&1 | tee ./build/log/train.log
'

# -------------------------------------------------------------------------------
# quantize & export quantized model
# -------------------------------------------------------------------------------
echo " "
echo "-----------------------------------------"
echo "RUN QUANTIZATION."
echo "-----------------------------------------"
python -u quantize.py --quant_mode calib 2>&1 | tee ./build/log/quant_calib.log
python -u quantize.py --quant_mode test  2>&1 | tee ./build/log/quant_test.log


# -------------------------------------------------------------------------------
# compile for target boards
# -------------------------------------------------------------------------------
source compile.sh $TARG

# -------------------------------------------------------------------------------
# move log files
# -------------------------------------------------------------------------------

mv ./build/*.log ./build/log/

: '
# add Xilinx headers
cd ./build/log/
for file in $(find . -name "*.log"); do
  echo ${file}
  cat ../../header.txt ${file} > tmp.txt
  mv tmp.txt ${file}
done
cd ../..
cat ./header.txt ./doc/image/cnn_int8_subgraph_tree.txt > tmp.txt
mv ./tmp.txt ./doc/image/cnn_int8_subgraph_tree.txt
'

# -------------------------------------------------------------------------------
# make target and run functional debug on host
# -------------------------------------------------------------------------------
bash ./prepare_target.sh $TARG


# -------------------------------------------------------------------------------
# analyze DPU/CPU subgraphs
# -------------------------------------------------------------------------------
echo " "
echo "-----------------------------------------"
echo "ANALYZE SUBGRAPHS."
echo "-----------------------------------------"
bash ./analyze_subgraphs.sh $TARG &> /dev/null

# -------------------------------------------------------------------------------
# save log files
# -------------------------------------------------------------------------------
tar -cvf logfiles_${TARGET}.tar ./build/log/
