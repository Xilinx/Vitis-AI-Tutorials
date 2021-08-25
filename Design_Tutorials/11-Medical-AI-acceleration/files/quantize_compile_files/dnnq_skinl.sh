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

TF_NETWORK_PATH="/workspace/mpsoc/skinl"
FROZEN_MODEL="model_incv3_2_6.pb"
INPUT_NODES="input_1"
OUTPUT_NODES="dense_3/Softmax "
INPUT_FN="skinl_input_fn.calib_input"


vai_q_tensorflow quantize \
			  --input_frozen_graph ${TF_NETWORK_PATH}/${FROZEN_MODEL}  \
			  --input_fn  ${INPUT_FN} \
	          --input_nodes ${INPUT_NODES} \
			  --output_nodes ${OUTPUT_NODES}\
			  --input_shapes ?,224,224,3 \
			  --calib_iter 10 \
			  --method 1 \
			  --gpu 0 \
			  --output_dir ${TF_NETWORK_PATH}/qoutput
