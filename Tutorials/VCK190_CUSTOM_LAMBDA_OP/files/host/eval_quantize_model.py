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

''' Trains a DeepWaterMap model. We provide a copy of the trained checkpoints.
You should not need this script unless you want to re-train the model.
'''

import os
import argparse
import sys
import tensorflow as tf
from metrics import running_precision, running_recall, running_f1
from metrics import adaptive_maxpool_loss
from tensorflow.keras.layers import Lambda
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from trainer import TFModelTrainer 

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./quantized_model.h5',
                        help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--data_path', type=str,default='/data3/datasets/DeepWater',
                        help="Path to the tfrecord files")
    args = parser.parse_args()
    if os.path.exists(args.checkpoint_path):
        trainer = TFModelTrainer(args.checkpoint_path, args.data_path)
    else:
        print("Error, checkpoint not found: ", args.checkpoint_path)
        print("make sure to quantize the model first using 'python quantize.py'")
        sys.exit(1)
    
    quantized_model = tf.keras.models.load_model(args.checkpoint_path)
    quantized_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9),
                    loss=adaptive_maxpool_loss,
                    metrics=[tf.keras.metrics.binary_accuracy,
                            running_precision, running_recall, running_f1])
    quantized_model.evaluate(trainer.dataset_val,batch_size=trainer.batch_size, steps=trainer.validation_steps)
        
if __name__ == '__main__':
    main()
