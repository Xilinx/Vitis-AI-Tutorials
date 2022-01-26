'''
 Copyright 2021 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Author: Mark Harvey
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model


from dataset_utils import input_fn_test
from mobilenetv2 import mobilenetv2

input_height = 224
input_width = 224
input_chan = 3
tfrec_dir = 'data/tfrecords'
batchsize = 50


# simple generator to provide incrementing integer
def step_number(num=0):
  while True:
    num += 1
    yield num

step_num = step_number()


# model to be evaluated
eval_model = mobilenetv2(input_shape=(input_height,input_width,input_chan),classes=2,alpha=1.0,incl_softmax=False)

eval_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# test dataset
test_dataset = input_fn_test(tfrec_dir,batchsize)

# eval function
def evaluate(checkpoint_path=''):
  eval_model.load_weights(checkpoint_path)
  scores = eval_model.evaluate(test_dataset)  
  eval_metric_ops = {'accuracy': scores[1]}
  print('*** Accuracy after',next(step_num),'steps:',scores[-1],'***')
  return eval_metric_ops


if __name__ == '__main__':
  path = './tf_ckpt/tf_float.ckpt'
  evaluate(path)

