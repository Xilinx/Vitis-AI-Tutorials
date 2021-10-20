#!/bin/sh

# Copyright 2020 Xilinx Inc.
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

# activate the python virtual environment
conda activate vitis-ai-tensorflow2

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}


# list of GPUs to use - modify as required for your system
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0"


# convert dataset to TFRecords - this only needs to be run once
python -u images_to_tfrec.py 2>&1 | tee ${LOG}/tfrec.log


# training
python -u train.py 2>&1 | tee ${LOG}/train.log


# quantize & evaluate
python -u quantize.py --evaluate 2>&1 | tee ${LOG}/quantize.log


# compile for selected target board
source compile.sh zcu102
source compile.sh zcu104
source compile.sh vck190
source compile.sh u50


# make target folder
python -u target.py -m ${BUILD}/compiled_zcu102/customcnn.xmodel -t ${BUILD}/target_zcu102 2>&1 | tee ${LOG}/target_zcu102.log
python -u target.py -m ${BUILD}/compiled_zcu104/customcnn.xmodel -t ${BUILD}/target_zcu104 2>&1 | tee ${LOG}/target_zcu104.log
python -u target.py -m ${BUILD}/compiled_vck190/customcnn.xmodel -t ${BUILD}/target_vck190 2>&1 | tee ${LOG}/target_vck190.log
python -u target.py -m ${BUILD}/compiled_u50/customcnn.xmodel -t ${BUILD}/target_u50 2>&1 | tee ${LOG}/target_u50.log


