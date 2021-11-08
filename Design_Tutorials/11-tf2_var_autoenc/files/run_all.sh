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

# clean up everything
rm -rf build
rm -rf __pycache__



# activate TF2 conda environment
conda activate vitis-ai-tensorflow2

# list of GPUs to use - modify as required for your system
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="1"

# make logs folder
mkdir -p build/logs

# train with predictions enabled
python -u train.py -p 2>&1 | tee build/logs/train.log

# quantize with predictions enabled
python -u quantize.py -p 2>&1 | tee build/logs/quant.log

# compile
source compile.sh zcu102
source compile.sh u50
source compile.sh vck190

# make PNG image of subgraphs
xir png build/compiled_model_zcu102/autoenc.xmodel build/autoenc_zcu102.png
xir png build/compiled_model_vck190/autoenc.xmodel build/autoenc_vck190.png
xir png build/compiled_model_u50/autoenc.xmodel build/autoenc_u50.png


# make target folders
python -u make_target.py -m build/compiled_model_zcu102/autoenc.xmodel -td build/target_zcu102 2>&1 | tee build/logs/target_zcu102.log
python -u make_target.py -m build/compiled_model_u50/autoenc.xmodel -td build/target_u50 2>&1 | tee build/logs/target_u50.log
python -u make_target.py -m build/compiled_model_vck190/autoenc.xmodel -td build/target_vck190 2>&1 | tee build/logs/target_vck190.log

