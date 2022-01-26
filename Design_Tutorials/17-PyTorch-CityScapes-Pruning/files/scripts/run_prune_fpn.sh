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
WEIGHTS=float
GPU_ID=0

export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "                                                    "
echo "####################################################"
echo "# Initial Pruning of the reduced class FPN model   #"
echo "# this only uses 6 classes vs. the original 19     #"
echo "#                                                  #"
echo "# By default this will run 5 iterations of pruning #"
echo "# The first step is the analysis which typically   #"
echo "# requires at least several hours                  #"
echo "####################################################"
echo "                                                    "

echo "====> Pruning SemanticFPN(ResNet18) with input_size = 512x1024..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/prune/train.py --lr 1e-4 --dataset citys_reduced --prune True --model fpn \
 --num-classes 6 --backbone resnet18 --epochs 200 --crop-size 512 --data-folder ${DATASET} --prune_iterations 5 \
 --prune_ratio 0.1 --weight checkpoint/citys_reduced/fpn/model_best.pth.tar 2>&1 | tee fpn-pruning.log
