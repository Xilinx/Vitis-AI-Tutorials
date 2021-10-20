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
Quantize the floating-point model
'''

'''
Author: Mark Harvey, Xilinx Inc
'''


import argparse
import os
import shutil
import sys
import cv2

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope

from utils import input_fn, mnist_download
from vae import Sampling

DIVIDER = '-----------------------------------------'




def quant_model(float_model,quant_model,batchsize,predict,pred_dir):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # make dataset and image processing pipeline
    _, x_test, _, x_test_noisy = mnist_download()
    calib_dataset = input_fn((x_test_noisy,x_test), batchsize, False)

    with custom_object_scope({'Sampling': Sampling}):
      # load trained floating-point model    
      float_model = load_model(float_model, compile=False, custom_objects={'Sampling': Sampling} )

      # quantizer
      quantizer = vitis_quantize.VitisQuantizer(float_model)
      quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    '''
    Predictions
    '''
    if (predict):
      print('\n'+DIVIDER)
      print ('Predicting with quantized model..')
      print(DIVIDER+'\n')

      # remake predictions folder
      shutil.rmtree(pred_dir, ignore_errors=True)
      os.makedirs(pred_dir)

      predict_dataset = input_fn((x_test_noisy), batchsize, False)
      predictions = quantized_model.predict(predict_dataset, verbose=0)

      # scale pixel values back up to range 0:255 then save as PNG
      for i in range(20):
        cv2.imwrite(pred_dir+'/pred_'+str(i)+'.png', predictions[i] * 255.0)
      print('Predictions saved as images in ./' + pred_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='build/float_model/f_model.h5', help='Full path of floating-point model. Default is build/float_model/k_model.h5')
    ap.add_argument('-q', '--quant_model',  type=str, default='build/quant_model/q_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=100,                      help='Batchsize for quantization. Default is 100')
    ap.add_argument('-p', '--predict',      action='store_true', help='Run predictions if set. Default is no predictions.')
    ap.add_argument('-pd','--pred_dir',     type=str, default='build/quant_predict', help='Full path of folder for saving predictions. Default is build/quant_predict')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --predict      : ', args.predict)
    print (' --pred_dir     : ', args.pred_dir)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.predict, args.pred_dir)


if __name__ ==  "__main__":
    main()
