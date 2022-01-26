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


DATASET=data/
WEIGHTS=float
GPU_ID=0

echo "                                                   "
echo "###################################################"
echo "# Training reduced class FPN model on Cityscapes  #"
echo "# this only uses 6 classes vs. the original 19    #"
echo "# Inspect the cityscapesscripts/helpers/labels.py #"
echo "# and reducedclasscityscapes.py for more info     #"
echo "###################################################"
echo "                                                   " 

echo "Conducting teraining with: "
echo " SemanticFPN(ResNet18) with input_size: 512x1024. "
export PYTHONPATH=${PWD}:${PYTHONPATH}

echo "====> perform SemanticFPN(ResNet18) with input_size = 512x1024..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train/train.py --lr 1e-4 --dataset citys_reduced \
--model fpn --num-classes 6 --backbone resnet18 --epochs 200 --crop-size 512 \
--data-folder ${DATASET} 2>&1 | tee train_6_cls.log
