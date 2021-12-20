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
/opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/yolo_detect yolov4_quantized/quantize_train_test.prototxt \
                                                 yolov4_quantized/quantize_train_test.caffemodel \
                                                 voc/2007_test.txt \
                                                 -out_file yolov4_quantized/caffe_result_quant.txt \
                                                 -confidence_threshold 0.005 \
                                                 -classes 20 \
                                                 -anchorCnt 3 \
												 -model_type yolov3 \
												 -biases "12,16,19,36,40,28, 36,75,76,55,72,146, 142,110,192,243,459,401"
