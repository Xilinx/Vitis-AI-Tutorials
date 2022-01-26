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

#!/bin/bash

export XILINXD_LICENSE_FILE=./license.lic
DATASET=data
WEIGHTS=float

echo "                                                         "
echo "#########################################################"
echo "# Quantizing pruned model                               #"
echo "# this requires that the dense checkpoint is            #"
echo "# provided with the --pruned_weights option             #"
echo "# and the pruned model python be provided e.g.          #"
echo "#--prune_model_py pruned_model_defs.pruned_fpn_resnet_0 #"
echo "#########################################################"
echo "                                                         "    

export PYTHONPATH=${PWD}:${PYTHONPATH} 
export W_QUANT=1
PRUNED_WEIGHTS=checkpoint/citys_reduced/fpn/epoch_1_prune_iter_1_dense_ckpt.pth.tar
PRUNED_MODEL_DEF=fpn_model_defs.pruned_1

GPU_ID=0
echo "====> Quantize and Test with input_size = 512x1024..."
echo "Performing quantization with fast finetuning and calibration"
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/prune/test.py --eval --dataset citys_reduced --crop-size 512 --num-classes 6 \
--data-folder ${DATASET} --prune_model_py ${PRUNED_MODEL_DEF} --pruned_weights ${PRUNED_WEIGHTS} \
--quant_mode calib --quant_dir quantize_result_fpn_iter0 --fast_finetune

echo "Testing quantized model accuracy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/prune/test.py --eval --dataset citys_reduced --crop-size 512 --num-classes 6 \
--data-folder ${DATASET} --prune_model_py ${PRUNED_MODEL_DEF} --pruned_weights ${PRUNED_WEIGHTS}  \
--quant_mode test --quant_dir quantize_result_fpn_iter0 --fast_finetune

echo "Dumping Xmodel into --quant_dir..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/prune/test.py --dump_xmodel --eval --dataset citys_reduced --num-classes 6 \
--crop-size 512 --data-folder ${DATASET} --prune_model_py ${PRUNED_MODEL_DEF} --pruned_weights ${PRUNED_WEIGHTS} \
--quant_mode test --quant_dir quantize_result_fpn_iter0 --fast_finetune

echo "Done!"
