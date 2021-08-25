#!/usr/bin/env bash

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


ssd_detect \
-file_type="image" \
-mean_value="104,117,123" \
-confidence_threshold=0.3 \
/workspace/SSD/workspace/VGG16-SSD/deploy.prototxt \
/workspace/SSD/workspace/VGG16-SSD/snapshots/pretrained.caffemodel detect.list 2>&1 | tee detections.log


