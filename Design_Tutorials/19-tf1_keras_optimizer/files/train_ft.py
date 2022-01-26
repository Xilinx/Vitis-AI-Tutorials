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
Training and pruning fine-tune script.
'''

'''
Author: Mark Harvey
'''

import os
import sys
import argparse


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from dataset_utils import input_fn_trn, input_fn_test
from mobilenetv2 import mobilenetv2
import numpy as np

DIVIDER = '-----------------------------------------'



class EarlyStoponAcc(tf.keras.callbacks.Callback):
  '''
  Early stop on reaching target accuracy 
  '''
  def __init__(self, target_acc):
    super(EarlyStoponAcc, self).__init__()
    self.target_acc=target_acc

  def on_epoch_end(self, epoch, logs=None):
    accuracy=logs['val_acc']
    if accuracy >= self.target_acc:
      self.model.stop_training=True
      print('Reached target accuracy of',self.target_acc,'..exiting.')


def train(input_ckpt,output_ckpt,tfrec_dir,tboard_dir,input_height,input_width, \
          input_chan,batchsize,epochs,learnrate,target_acc):

      
    '''
    tf.data pipelines
    '''
    # train and test folders
    train_dataset = input_fn_trn(tfrec_dir,batchsize)
    test_dataset = input_fn_test(tfrec_dir,batchsize)


    '''
    Call backs
    '''
    tb_call = TensorBoard(log_dir=tboard_dir)

    chkpt_call = ModelCheckpoint(filepath=output_ckpt,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True
                                 )

    early_stop_call = EarlyStoponAcc(target_acc)

    callbacks_list = [tb_call, chkpt_call, early_stop_call]


    # if required, tf.set_pruning_mode must be set before defining the model
    if (input_ckpt!=''):
      tf.set_pruning_mode()

    '''
    Define the model
    '''
    model = mobilenetv2(input_shape=(input_height,input_width,input_chan),classes=2,alpha=1.0,incl_softmax=False)


    '''
    Compile model
    Adam optimizer to change weights & biases
    Loss function is categorical crossentropy
    '''
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learnrate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    '''
    If an input checkpoint is specified then assume we are fine-tuning a pruned model,
    so load the weights into the model, otherwise we are training from scratch
    '''
    if (input_ckpt!=''):
      print('Loading checkpoint - fine-tuning from',input_ckpt)
      model.load_weights(input_ckpt)
    else:
      print('Training from scratch..')
          
      print('\n'+DIVIDER)
      print(' Model Summary')
      print(DIVIDER)
      print(model.summary())
      print("Model Inputs: {ips}".format(ips=(model.inputs)))
      print("Model Outputs: {ops}".format(ops=(model.outputs)))



    '''
    Training
    '''
    print('\n'+DIVIDER)
    print(' Training model with training set..')
    print(DIVIDER)

    # make folder for saving trained model checkpoint
    os.makedirs(os.path.dirname(output_ckpt), exist_ok = True)


    # run training
    train_history=model.fit(train_dataset,
                            epochs=epochs,
                            steps_per_epoch=20000//batchsize,
                            validation_data=test_dataset,
                            validation_steps=5000//batchsize,
                            callbacks=callbacks_list,
                            verbose=1)

    '''
    save just the model architecture (no weights) to a JSON file
    '''
    with open(os.path.join(os.path.dirname(output_ckpt),'baseline_arch.json'), 'w') as f:
      f.write(model.to_json())



    print("\nTensorBoard can be opened with the command: tensorboard --logdir={dir} --host localhost --port 6006".format(dir=tboard_dir))


    return




def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-i', '--input_ckpt',  type=str,   default='',      help='Path of input checkpoint. Default is no checkpoint')
  ap.add_argument('-o', '--output_ckpt', type=str,   default='build/float_model/f_model.h5', help='Folder for saving output floating-point model.')
  ap.add_argument('-tf','--tfrec_dir',   type=str,   default='data/tfrecords', help='TFRecords folder path. Default is data/tfrecords.')
  ap.add_argument('-t', '--tboard_dir',  type=str,   default='build/tb_logs', help='TensorBoard logs folder path. Default is build/tb_logs.')
  ap.add_argument('-ih','--input_height',type=int,   default=224,     help='Input image height in pixels.')
  ap.add_argument('-iw','--input_width', type=int,   default=224,     help='Input image width in pixels.')
  ap.add_argument('-ic','--input_chan',  type=int,   default=3,       help='Number of input image channels.')
  ap.add_argument('-b', '--batchsize',   type=int,   default=50,      help='Training batchsize. Must be an integer. Default is 50.')
  ap.add_argument('-e', '--epochs',      type=int,   default=30,      help='number of training epochs. Must be an integer. Default is 30.')
  ap.add_argument('-lr','--learnrate',   type=float, default=0.001,   help='optimizer learning rate. Must be floating-point value. Default is 0.001')
  ap.add_argument('-a', '--target_acc',  type=float, default=1.0,     help='Target accuracy. Default is 1.0 (100%)')
  ap.add_argument('-g', '--gpu',         type=str,   default='0',     help='String value to select which CUDA GPU devices to use')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('Keras version      : ',tf.keras.__version__)
  print('TensorFlow version : ',tf.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--input_ckpt   : ',args.input_ckpt)
  print ('--output_ckpt  : ',args.output_ckpt)
  print ('--tfrec_dir    : ',args.tfrec_dir)
  print ('--tboard_dir   : ',args.tboard_dir)
  print ('--input_height : ',args.input_height)
  print ('--input_width  : ',args.input_width)
  print ('--input_chan   : ',args.input_chan)
  print ('--batchsize    : ',args.batchsize)
  print ('--epochs       : ',args.epochs)
  print ('--learnrate    : ',args.learnrate)
  print ('--target_acc   : ',args.target_acc)
  print ('--gpu          : ',args.gpu)
  print(DIVIDER)


  # indicate which GPU to use
  os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


  train(args.input_ckpt,args.output_ckpt,args.tfrec_dir,args.tboard_dir,args.input_height, \
        args.input_width,args.input_chan,args.batchsize,args.epochs,args.learnrate,args.target_acc)


if __name__ == '__main__':
    run_main()
