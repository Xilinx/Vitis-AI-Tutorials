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

ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../.. && pwd )"
export ML_DIR
dcf_dir='../../xilinx_dnndk_v3.1/host_x86/dcf' 
net=skin_ld
model_dir=quantize_results
output_dir=dnnc_output

home_dir=$ML_DIR

echo "Compiling network: ${net}"

dnnc --frozen_pb=${model_dir}/deploy_model.pb     \
     --parser=tensorflow \
     --output_dir=${output_dir} \
     --net_name=${net}                           \
     --dcf=${dcf_dir}/ZCU102.dcf               \
     --cpu_arch=arm64                            \
     --mode=debug                                \
     --save_kernel \
     --dump all


echo " copying dpu elf file into /../zcu102/baseline/model/arm64_4096 "
cp ${output_dir}/dpu_${net}\_*.elf  ../zcu102/baseline/model/arm64_4096



