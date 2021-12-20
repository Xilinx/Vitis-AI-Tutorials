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

PATH_TO_TRAIN_IMAGES=/data2/datasets/coco/coco2017/images/train2017/ 
PATH_TO_VAL_IMAGES=/data2/datasets/coco/coco2017/images/val2017/ 

find ${PATH_TO_VAL_IMAGES} -maxdepth 1 -type f -not -path '*.txt' | sort 2>&1 | tee val2017.txt
find ${PATH_TO_TRAIN_IMAGES} -maxdepth 1 -type f -not -path '*.txt' | sort 2>&1 | tee train2017.txt
