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
Variational autoencoder training and evaluation
'''

'''
Author: Mark Harvey, Xilinx Inc
'''


import os
import sys
import shutil
import cv2
import argparse


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.utils import custom_object_scope

from vae import encoder, decoder, Sampling
from utils import input_fn, loss_func, mnist_download



_DIVIDER = '---------------------------'


def train(float_model,predict,pred_dir,tblogs_dir,batchsize,learnrate,epochs):

    '''
    Variational encoder model
    '''

    image_dim = 28
    image_chan = 1
    input_layer = Input(shape=(image_dim,image_dim,image_chan))
    encoder_mu, encoder_log_variance, encoder_z = encoder.call(input_layer)

    dec_out = decoder.call(encoder_z)
    model = Model(inputs=input_layer, outputs=dec_out)



    '''
    Prepare MNIST dataset
    '''
    x_train, x_test, x_train_noisy, x_test_noisy = mnist_download()
    train_dataset = input_fn((x_train_noisy,x_train), batchsize, True)
    test_dataset = input_fn((x_test_noisy,x_test), batchsize, False)
    predict_dataset = input_fn((x_test_noisy), batchsize, False)


    '''
    Call backs
    '''
    tb_call = TensorBoard(log_dir=tblogs_dir)
    chkpt_call = ModelCheckpoint(filepath=float_model, 
                                 monitor='val_mse',
                                 mode='min',
                                 verbose=1,
                                 save_weights_only=False,
                                 save_best_only=True)

    callbacks_list = [tb_call, chkpt_call]


    '''
    Compile
    '''
    model.compile(optimizer=Adam(lr=learnrate),
                  loss=lambda y_true,y_predict: loss_func(y_true,y_predict,encoder_mu,encoder_log_variance),
                  metrics=['mse'])


    '''
    Training
    '''
    print(_DIVIDER)
    print('Training...')
    print(_DIVIDER)
    # make folder for saving trained model checkpoint
    os.makedirs(os.path.dirname(float_model), exist_ok=True)

    # remake Tensorboard logs folder
    shutil.rmtree(tblogs_dir, ignore_errors=True)
    os.makedirs(tblogs_dir)

    train_history = model.fit(train_dataset,
                              epochs=epochs,
                              steps_per_epoch=len(x_train)//batchsize,
                              validation_data=test_dataset,
                              callbacks=callbacks_list,
                              verbose=1)

    '''
    Predictions
    '''
    if (predict):
      print(_DIVIDER)
      print('Making predictions...')
      print(_DIVIDER)
      # remake predictions folder
      shutil.rmtree(pred_dir, ignore_errors=True)
      os.makedirs(pred_dir)

      with custom_object_scope({'Sampling': Sampling}):
        model = load_model(float_model, compile=False, custom_objects={'Sampling': Sampling})
      model.compile(loss=lambda y_true,y_predict: loss_func(y_true,y_predict,encoder_mu,encoder_log_variance))
      predictions = model.predict(predict_dataset, verbose=1)

      # scale pixel values back up to range 0:255 then save as PNG
      for i in range(20):
        cv2.imwrite(pred_dir+'/pred_'+str(i)+'.png', predictions[i] * 255.0)
        cv2.imwrite(pred_dir+'/input_'+str(i)+'.png', x_test_noisy[i] * 255.0)
      print('Inputs and Predictions saved as images in ./' + pred_dir)

    print("\nTensorBoard can be opened with the command: tensorboard --logdir=./tb_logs --host localhost --port 6006")
       
    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m',  '--float_model', type=str,   default='build/float_model/f_model.h5', help='Full path of floating-point model. Default is build/float_model/f_model.h5')
    ap.add_argument('-p',  '--predict',     action='store_true', help='Run predictions if set. Default is no predictions.')
    ap.add_argument('-pd', '--pred_dir',    type=str,   default='build/float_predict', help='Full path of folder for saving predictions. Default is build/float_predict')
    ap.add_argument('-tb', '--tblogs_dir',  type=str,   default='build/tb_logs', help='Full path of folder for saving TensorBoard logs. Default is build/tb_logs')
    ap.add_argument('-b',  '--batchsize',   type=int,   default=100,    help='Batchsize for training. Default is 100')
    ap.add_argument('-lr', '--learnrate',   type=float, default=0.0001, help='Initial learning rate. Default is 0.0001')
    ap.add_argument('-e',  '--epochs',      type=int,   default=40,     help='Number of training epochs. Default is 40')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print('Keras version      : ',tf.keras.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --predict      : ', args.predict)
    print (' --pred_dir     : ', args.pred_dir)
    print (' --tblogs_dir   : ', args.tblogs_dir)
    print (' --batchsize    : ', args.batchsize)
    print (' --learnrate    : ', args.learnrate)
    print (' --epochs       : ', args.epochs)
    print('------------------------------------\n')


    train(args.float_model,args.predict,args.pred_dir,args.tblogs_dir,args.batchsize,args.learnrate,args.epochs)


if __name__ ==  "__main__":
    main()

