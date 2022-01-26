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

export XILINXD_LICENSE_FILE=./license.lic
DATASET=data
WEIGHTS=checkpoint/citys_reduced/fpn/epoch_1_prune_iter_0_sparse_ckpt.pth.tar
GPU_ID=0

export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "                                                  "
echo "###################################################"
echo "# Resuming pruning reduced class FPN model        #"
echo "#                                                 #"
echo "# Resuming pruning requires use of the original   #"
echo "# model definition and uses the sparse weights    #"
echo "# checkpoint                                      #"
echo "###################################################"
echo "                                                   "

echo "====> Resuming Pruning SemanticFPN(ResNet18) with input_size = 512x1024..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/prune/train.py --lr 1e-4 --dataset citys_reduced --prune True \
--model fpn --num-classes 6 --backbone resnet18 --epochs 200 --crop-size 512 --data-folder ${DATASET} --prune_ratio 0.1 \
--pruned_weights ${WEIGHTS} 2>&1 | tee resumed_pruning.log
