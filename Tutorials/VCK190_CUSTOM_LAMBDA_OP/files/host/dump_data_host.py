# Copyright 2022 Xilinx Inc.
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

'''
The following file will generate random data as an input stimulus and run inference on the model.  
The output files can be used for cross checking results vs a DPU target and developing CPU custom op layers.
'''

import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./quantized_model.h5',
                        help="Path to the dir where the quantized model is saved")
    args = parser.parse_args()
    my_generator = np.random.default_rng(seed=1)
    img = my_generator.integers(0, high=255, size=(512,512,6))/255.0
    input = np.expand_dims(img,axis=0)
    #################################
    ##### load and quantize model
    #################################
    quant_model = tf.keras.models.load_model(args.model,custom_objects={'Lambda': Lambda})

    #################################
    ##### dump data
    #################################
    vitis_quantize.VitisQuantizer.dump_model(quant_model, input,"./dump_results", dump_float=True)


if __name__ == '__main__':
    main()


