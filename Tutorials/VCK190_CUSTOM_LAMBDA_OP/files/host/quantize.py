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
The following file will generate random data as an input stimulus for quantization calibration and
will quantize the pre-trained checkpoint from the Deepwatermap repository
'''

import os,sys
import argparse
import tensorflow as tf
import deepwatermap
from metrics import running_precision, running_recall, running_f1
from metrics import adaptive_maxpool_loss
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import numpy as np

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/cp.135.ckpt',
                        help="Path to the dir where the checkpoints are saved")
    args = parser.parse_args()

    rand_data_generator = np.random.default_rng(seed=1)
    X_train = rand_data_generator.integers(0, high=255, size=(100,512,512,6))/255.0

    model = deepwatermap.model()
    if os.path.exists(args.checkpoint_path+".index"):
        model.load_weights(args.checkpoint_path)
    else:
        print("Error, checkpoint not found: ", args.checkpoint_path)
        print("download checkpoint from: https://utexas.app.box.com/s/j9ymvdkaq36tk04be680mbmlaju08zkq/file/565662752887")
        print("and extract it to ./checkpoints")
        sys.exit(1)
    print("By default, quantization will use randomly generated integers.  For higher accuracy, use the real Deepwatermap dataset which is available here: ")
    print("https://utexas.app.box.com/s/j9ymvdkaq36tk04be680mbmlaju08zkq/folder/94459511962 ")    
    quantizer = vitis_quantize.VitisQuantizer(model, custom_objects={'Lambda': Lambda, 'adaptive_maxpool_loss': adaptive_maxpool_loss, 'running_precision':running_precision, 'running_recall':running_recall, 'running_f1':running_f1}) 
    quantized_model = quantizer.quantize_model(calib_dataset=X_train, calib_batch_size=5, calib_steps=20, replace_sigmoid=True, add_shape_info=True)
    quantized_model.save("quantized_model.h5")
                    
if __name__ == '__main__':
    main()
