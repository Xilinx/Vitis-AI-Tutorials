#!/bin/sh
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

vai_c_tensorflow2 -m tf2_resnet50/vai_q_output/quantized.h5 \
		  -a /opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json \
		  -o tf2_resnet50/u250 \
		  -n resnet50_tf2 \
		  --options '{"input_shape": "4,100,100,3"}'
