#!/bin/bash

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

#--------------------------
# U50 setup
#--------------------------
wget https://www.xilinx.com/bin/public/openDownload?filename=U50_xclbin-v2.tar.gz -O U50_xclbin.tar.gz
tar -xzvf U50_xclbin.tar.gz
cd U50_xclbin/6E250M
sudo cp -f dpu.xclbin hbm_address_assignment.txt /usr/lib
cd /workspace
rm -rf U50_xclbin.tar.gz
rm -rf U50_xclbin


#--------------------------
# VART setup
#--------------------------
sudo dpkg -i ./libvart-1.1.0-Linux-build48.deb

#wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.1.2.tar.gz -O vitis-ai-runtime-1.1.2.tar.gz
#tar -xzvf vitis-ai-runtime-1.1.2.tar.gz
#sudo dpkg -i ./vitis-ai-runtime-1.1.2/VART/X86_64/libvart-1.1.0-Linux-build*.deb
#rm -rf vitis-ai-runtime-1.1.2*

export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/
cd /workspace
source /opt/xilinx/xrt/setup.sh



