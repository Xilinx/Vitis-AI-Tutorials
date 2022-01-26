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


'''
MobileNetv2: https://arxiv.org/abs/1801.04381
'''

'''
Author: Mark Harvey
'''

'''
Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dropout,Add,DepthwiseConv2D,Dense
from tensorflow.keras.layers import GlobalAveragePooling2D,BatchNormalization,Activation,ReLU



def cbr(inputs, filters, kernel, strides):
    '''
    Convolution - BatchNorm - ReLU6
    '''
    net = Conv2D(filters, kernel, padding='same', strides=strides, kernel_initializer='he_uniform', use_bias=False)(inputs)
    net = BatchNormalization()(net)
    net = ReLU(6.)(net)
    return net



def residuals(inputs, filters, kernel, t=1, alpha=1.0, strides=1, use_residual=False):
    '''
    Bottleneck block
    '''

    # Depth
    tchannel = K.int_shape(inputs)[-1] * t
    # Width
    cchannel = int(filters * alpha)

    net = cbr(inputs, tchannel, 1, 1)

    net = DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding='same', use_bias=False)(net)
    net = BatchNormalization()(net)
    net = ReLU(6.)(net)

    net = Conv2D(cchannel, 1, strides=1, padding='same', use_bias=False)(net)
    net = BatchNormalization()(net)

    if use_residual:
        net = Add()([net, inputs])

    return net



def bottleneck(inputs, filters, kernel, t, alpha, strides, n):
    '''
    Inverted residual block
    '''
    net = residuals(inputs, filters, kernel, t, alpha, strides, False)
    for i in range(1, n):
        net = residuals(net, filters, kernel, t, alpha, 1, True)

    return net


# https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v




def mobilenetv2(input_shape=(None, None, None),classes=None,alpha=1.0,incl_softmax=False):
  '''
  MobileNetV2
  Arguments:
      input_shape  : tuple of integers indicating height, width, channels
      classes      : integer to set number of classes
      alpha        : floating-point alpha value
      incl_softmax : Boolean, True adds softmax activation after final dense/logits layer
  '''

  input_layer = Input(shape=input_shape)

  first_filters = _make_divisible(32 * alpha, 8)
  net = cbr(input_layer, first_filters, 3, strides=2)

  net = bottleneck(net, 16,  3, t=1, alpha=alpha, strides=1, n=1)
  net = bottleneck(net, 24,  3, t=6, alpha=alpha, strides=2, n=2)
  net = bottleneck(net, 32,  3, t=6, alpha=alpha, strides=2, n=3)
  net = bottleneck(net, 64,  3, t=6, alpha=alpha, strides=2, n=4)
  net = bottleneck(net, 96,  3, t=6, alpha=alpha, strides=1, n=3)
  net = bottleneck(net, 160, 3, t=6, alpha=alpha, strides=2, n=3)
  net = bottleneck(net, 320, 3, t=6, alpha=alpha, strides=1, n=1)

  if alpha > 1.0:
      last_filters = _make_divisible(1280 * alpha, 8)
  else:
      last_filters = 1280

  net = cbr(net, last_filters, 1, strides=1)
  net = GlobalAveragePooling2D()(net)

  # if the softmax layer is included in the model, then 
  # from_logits must be set to False in the compile method 
  # during training (see train.py)
  if (incl_softmax):
    net = Dense(classes)(net)
    output_layer = Activation('softmax')(net)
  else:
    output_layer = Dense(classes)(net)

  return Model(inputs=input_layer, outputs=output_layer)

