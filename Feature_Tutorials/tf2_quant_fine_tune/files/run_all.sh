#!/bin/bash

# Copyright 2021 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

#-----------------------------------------------
# Run this script from within the Vitis-AI docker
#-----------------------------------------------


# activate the conda Python virtual environment
conda activate vitis-ai-tensorflow2

# list of GPUs to use
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="1"


# set number of classes
export CLASSES=10

# set build folder
export BUILD_DIR=build

# set log folder
export LOGS=${BUILD_DIR}/logs


# make folder for log files
rm -rf __pycache__
rm -rf ${LOGS}
mkdir -p ${LOGS}


# make ImageNet validation set TFRecords
python -u make_val_tfrec.py -mc ${CLASSES} -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/make_val_tfrec.log

# make ImageNet training set TFRecords
python -u make_train_tfrec.py -mc ${CLASSES} -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/make_train_tfrec.log

# fine-tune MobileNet model
python -u train.py -mc ${CLASSES} -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/train.log

# Quantization without fine-tuning
python -u quant.py -bd ${BUILD_DIR} -e 2>&1 | tee ${LOGS}/quant.log

# Quantization with fine-tuning
python -u quant_ft.py -mc ${CLASSES} -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/quant_ft.log

# compile for chosen targets
source compile.sh zcu102 ${BUILD_DIR} ${LOGS}
source compile.sh zcu104 ${BUILD_DIR} ${LOGS}
source compile.sh vck190 ${BUILD_DIR} ${LOGS}

# create target folders for chosen targets
python -u make_target.py -t zcu102 -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/target_zcu102.log
python -u make_target.py -t zcu104 -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/target_zcu104.log
python -u make_target.py -t vck190 -bd ${BUILD_DIR} 2>&1 | tee ${LOGS}/target_vck190.log

