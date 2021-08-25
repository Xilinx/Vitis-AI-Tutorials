'''
 Copyright 2020 Xilinx Inc.

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


import tensorflow as tf
import numpy as np


# Silence TensorFlow messages
tf.logging.set_verbosity(tf.logging.INFO)


def datadownload():
    
    # MNIST dataset has 70k images. Training set is 60k, test set is 10k.
    # Each image is 28x28x8bits
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Scale image data from range 0:255 to range 0:1.0
    # Also converts train & test data to float from uint8
    x_train = (x_train/255.0).astype(np.float32)
    x_test = (x_test/255.0).astype(np.float32)

    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    # take 5000 images & labels from the train dataset to create a validation set
    x_valid = x_train[55000:]
    y_valid = y_train[55000:]
    
    # train dataset reduced to 55000 images
    x_train = x_train[:55000]
    y_train = y_train[:55000]

    # reshape
    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)
    x_valid = x_valid.reshape(x_valid.shape[0],28,28,1)
    
        
    return (x_train,y_train), (x_test,y_test), (x_valid,y_valid)
