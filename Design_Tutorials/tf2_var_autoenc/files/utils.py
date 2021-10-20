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
Custom loss function
Utility functions for tf.data pipeline
'''

'''
Author: Mark Harvey, Xilinx Inc
'''


import os
import numpy as np

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
 


def loss_func(y_true, y_predict, encoder_mu, encoder_log_variance):
  '''    
  Loss function: Kulback-Leibler Divergence + Reconstruction loss
  Reconstruction loss is mean squared error
  '''
  reconstruction_loss_factor = 784
  reconstruction_loss = K.mean(K.square(y_true - y_predict), axis=[1, 2, 3])
  reconstruction_loss = reconstruction_loss_factor * reconstruction_loss

  kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=-1)
  
  return reconstruction_loss + kl_loss


def mnist_download():
  '''
  MNIST dataset download and pre-processing
  Pixels are scaled to range 0.0 to 1.0 then rounded up or down to
  be exactly 0.0 or 1.0
  Adds random noise to create noisy train and test images
  Returns:
    - 'clean' train & test data as numpy arrays - labels not returned
    - 'noisy' train & test data as numpy arrays - labels not returned  
  '''
  (x_train, _), (x_test, _) = mnist.load_data()
  # scale to (0,1)
  x_train = (x_train/255.0).astype(np.float32)
  x_test = (x_test/255.0).astype(np.float32)
  # add channel dimension
  x_train = x_train.reshape(x_train.shape[0],28,28,1)
  x_test = x_test.reshape(x_test.shape[0],28,28,1)
  # add noise
  noise = np.random.normal(loc=0.2, scale=0.3, size=x_train.shape)
  x_train_noisy = np.clip(x_train + noise, 0, 1)
  noise = np.random.normal(loc=0.2, scale=0.3, size=x_test.shape)
  x_test_noisy = np.clip(x_test + noise, 0, 1)
  return x_train, x_test, x_train_noisy, x_test_noisy


def input_fn(input_data,batchsize,is_training):
  '''
  Dataset creation and augmentation for training
  '''
  dataset = tf.data.Dataset.from_tensor_slices(input_data)
  if is_training:
    dataset = dataset.shuffle(buffer_size=1000,seed=42)
  dataset = dataset.batch(batchsize, drop_remainder=False)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  if is_training:
      dataset = dataset.repeat()
  return dataset

