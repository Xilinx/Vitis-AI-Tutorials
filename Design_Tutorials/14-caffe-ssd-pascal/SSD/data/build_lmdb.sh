#!/bin/bash
# Copyright 2019 Xilinx Inc.
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

rm -rf /workspace/SSD/data/VOCdevkit/VOC0712
mkdir /workspace/SSD/data/VOCdevkit/VOC0712/
mkdir /workspace/SSD/data/VOCdevkit/VOC0712/lmdb
convert_annoset --anno_type=detection --label_type=xml --label_map_file=/workspace/SSD/data/labelmap_voc.prototxt --check_label=False --min_dim=0 --max_dim=0 --resize_height=0 --resize_width=0 --backend=lmdb --shuffle=False --check_size=False --encode_type=jpg --encoded=True --gray=False /workspace/SSD/data/VOCdevkit/ /workspace/SSD/data/test.txt /workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
convert_annoset --anno_type=detection --label_type=xml --label_map_file=/workspace/SSD/data/labelmap_voc.prototxt --check_label=False --min_dim=0 --max_dim=0 --resize_height=0 --resize_width=0 --backend=lmdb --shuffle=False --check_size=False --encode_type=jpg --encoded=True --gray=False /workspace/SSD/data/VOCdevkit/ /workspace/SSD/data/trainval.txt /workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
