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
Make, save and evaluate quantized model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import sys


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from dataset_utils import input_fn

DIVIDER = '-----------------------------------------'



def quant_model(build_dir,batchsize,evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    float_dir = build_dir + '/float_model'
    quant_dir = build_dir + '/quant_model'
    tfrec_dir = build_dir + '/tfrec_val'

    print(DIVIDER)
    print('Make & Save quantized model..')

    # load the floating point trained model
    float_model = load_model(float_dir+'/float_model.h5',compile=False)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]
    chans = float_model.input_shape[3]
    print(' Input dimensions: height:',height,' width:',width, 'channels:',chans)


    # make TFRecord dataset and image processing pipeline
    test_dataset = input_fn(tfrec_dir, batchsize, False)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quant_model = quantizer.quantize_model(calib_dataset=test_dataset)

    # saved quantized model
    os.makedirs(quant_dir ,exist_ok=True)
    quant_model.save(quant_dir+'/quant_model.h5')
    print(' Saved quantized model to',quant_dir+'/quant_model.h5')

    if (evaluate):
        '''
        Evaluate the quantized model
        '''
        print(DIVIDER)
        print ('Evaluating quantized model..')

        quant_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['sparse_categorical_accuracy'])

        scores = quant_model.evaluate(test_dataset,
                                      steps=None,
                                      verbose=0)

    print(' Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
    print(DIVIDER)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-b',  '--batchsize', type=int, default=50,            help='Batchsize for quantization. Default is 50')
    ap.add_argument('-bd', '--build_dir',  type=str, default='build',help='Path to build folder. Default is build')
    ap.add_argument('-e',  '--evaluate',  action='store_true', help='Evaluate quantized model if set.')
    args = ap.parse_args()  

    print('\n'+DIVIDER)
    print('TensorFlow version :',tf.__version__)
    print('Keras version      :',tf.keras.__version__)
    print('Python             :',sys.version)
    print(DIVIDER)
    print ('Command line options:')
    print (' --build_dir   :', args.build_dir)
    print (' --batchsize   :', args.batchsize)
    print (' --evaluate    :', args.evaluate)
    print(DIVIDER+'\n')


    quant_model(args.build_dir, args.batchsize, args.evaluate)


if __name__ ==  "__main__":
    main()
