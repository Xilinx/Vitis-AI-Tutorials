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
 load Keras saved model in HDF5 format and create
 TensorFlow1 compatible checkpoint
'''

'''
Author: Mark Harvey
'''

import os
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.models import model_from_json

from mobilenetv2 import mobilenetv2


def keras_convert(float_model,tf_ckpt,pruning):
    '''
    Convert Keras model to TF checkpoint
    '''

    # set learning phase for no training
    tf.compat.v1.keras.backend.set_learning_phase(0)


    # restore model architecture and weights
    json_file = open(os.path.join(os.path.dirname(os.path.abspath(float_model)),'baseline_arch.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(float_model)


    if (pruning):
      # in pruning mode
      print('Pruning mode...')

      # save weights in TF format- only used in pruning flow
      loaded_model.save_weights(tf_ckpt, save_format='tf')
    else:
      # in baseline mode
      print('Baseline mode...')

      # fetch underlying TF session
      tf_session = tf.compat.v1.keras.backend.get_session()

      # write out complete tensorflow checkpoint & meta graph
      saver = tf.compat.v1.train.Saver()
      save_path = saver.save(tf_session,tf_ckpt)
      print (' Checkpoint created :',tf_ckpt)

    return



def run_main():

    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--float_model', type=str,  default='build/float_model/fmodel.h5', help='Full path of trained floating-point model.')   
    ap.add_argument('-tc','--tf_ckpt',     type=str,  default='build/tf_ckpt/tf_float.ckpt', help='TensorFlow checkpoint path. Default is build/tf_ckpt/tf_float.ckpt.')     
    ap.add_argument('-w', '--pruning',     action='store_true', help='Pruning or baseline mode. Default is baseline mode.')  
    args = ap.parse_args()

    print('-------------------------------------')
    print('keras_2_tf command line arguments:')
    print(' --float_model   :', args.float_model)
    print(' --tf_ckpt       :', args.tf_ckpt)
    print(' --pruning       :', args.pruning)
    print('-------------------------------------')

    keras_convert(args.float_model,args.tf_ckpt,args.pruning)


if __name__ == '__main__':
    run_main()
