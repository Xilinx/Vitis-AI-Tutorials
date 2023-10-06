#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Mark Harvey
# Modified by: Daniele Bagni, AMD/Xilinx
# date:  14 July 2023


'''
tf.data image processing pipeline
'''



import os
# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

from tensorflow.keras.layers import RandomRotation, Rescaling, RandomFlip, RandomZoom
from tensorflow.keras.preprocessing.image import smart_resize

from config import config as cfg

input_shape = cfg.input_shape
h = cfg.input_shape[0]
w = cfg.input_shape[1]


def parser(data_record):
    ''' TFRecord parser '''

    feature_dict = {
      'label' : tf.io.FixedLenFeature([], tf.int64),
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width' : tf.io.FixedLenFeature([], tf.int64),
      'chans' : tf.io.FixedLenFeature([], tf.int64),
      'image' : tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(data_record, feature_dict)
    label = tf.cast(sample['label'], tf.int32)
    h = tf.cast(sample['height'], tf.int32)
    w = tf.cast(sample['width'], tf.int32)
    c = tf.cast(sample['chans'], tf.int32)
    image = tf.io.decode_image(sample['image'], channels=3)
    image = tf.reshape(image,[h,w,3])

    return image, label


def randomcrop(x,y):
    '''
    Image random cropping
    Args:     Image and label
    Returns:  cropped image and unchanged label
    '''
    x = tf.image.random_crop(x, input_shape, seed=42)
    return x, y


def augment(x,y):
  '''
  Image augmentation
  Args:     Image and label
  Returns:  augmented image and unchanged label
  '''
  x = tf.image.random_brightness(x, 0.1, seed=42)
  x = tf.image.random_contrast(x, 0.9, 1.1, seed=42)
  x = tf.image.random_saturation(x, 0.9, 1.1, seed=42)
  return x, y


AUTOTUNE = tf.data.AUTOTUNE

random_rotation = RandomRotation(factor=(-0.1, 0.1),fill_mode='constant',
                                 interpolation='bilinear',seed=42,fill_value=0.0)
random_flip = RandomFlip(mode='horizontal', seed=42)
rescaling = Rescaling(scale=1./255)
random_zoom = RandomZoom(height_factor=(-0.2, 0.2), fill_mode='constant', interpolation='bilinear', seed=2, fill_value=0.0)


def input_fn_train(tfrec_dir, batchsize):
    '''
    Dataset pipeline for training
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/train_*.tfrecord'.format(tfrec_dir), shuffle=True)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=20000, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x, y: (smart_resize(x, (int(h*1.14),int(w*1.14)), interpolation='bicubic'), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x,y: randomcrop(x,y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(lambda x, y: (random_flip(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (random_zoom(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (random_rotation(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (rescaling(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.repeat()
    return dataset



def input_fn_test(tfrec_dir, batchsize):
    '''
    Dataset pipeline for testing
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/test_*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (smart_resize(x, (h,w), interpolation='bicubic'), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(lambda x, y: (rescaling(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def input_fn_image(tfrec_dir, batchsize):
    '''
    Dataset pipeline for image creation
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/test_*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (smart_resize(x, (h,w), interpolation='bicubic'), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    return dataset
