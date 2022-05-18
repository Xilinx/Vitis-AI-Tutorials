#!/bin/sh

# Copyright 2022 Xilinx Inc.
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



ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
echo "-----------------------------------------"
echo "COMPILING MODEL FOR VCK190.."
echo "-----------------------------------------"

compile() {
      vai_c_tensorflow2 \
            --model           quantized_model.h5 \
            --arch            $ARCH \
            --output_dir      compiled_model \
            --net_name        customcnn   
}


compile 2>&1 | tee compile.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
