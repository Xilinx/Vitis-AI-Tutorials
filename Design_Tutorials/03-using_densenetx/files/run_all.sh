#!/bin/bash

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


source 0_setenv.sh

# training
source 1_train.sh

# convert Keras model to Tensorflow frozen graph
source 2_keras2tf.sh

# Evaluate frozen graph
source 3_eval_frozen.sh

# Quantize
source 4_quant.sh

# Evaluate quantized model
source 5_eval_quant.sh

# compile for target
source 6_compile.sh zcu102
source 6_compile.sh vck190
source 6_compile.sh u50

# make target folders
source 7_make_target.sh zcu102
source 7_make_target.sh vck190
source 7_make_target.sh u50


