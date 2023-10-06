#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Mark Harvey
# Modified by: Daniele Bagni, AMD/Xilinx
# date:  28 July 2023

'''
Configuration
'''

import os
#https://bobbyhadz.com/blog/disable-suppress-tensorflow-warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
#https://github.com/tensorflow/tensorflow/issues/8340
logging.getLogger("tensorflow").setLevel(logging.WARNING)




DIVIDER = '-----------------------------------------'


# GPU
gpu_list='0'

# Dataset preparation and TFRecord creation
data_dir = 'data'
tfrec_dir = data_dir + '/tfrecords'
img_shard = 500


# MobileNet build parameters
input_shape=(224,224,3)
classes=2
alpha=1.0


# Training
batchsize=150
train_init_lr=0.001
train_epochs=85
train_target_acc=1.0
train_output_ckpt='/float_model/f_model.h5'

# Pruning & Fine-tuning
prune_output_ckpt='/pruned_model/p_model'
init_prune_ratio=0.1
incr_prune_ratio=0.1
prune_steps=6
finetune_init_lr=0.0007


# Transform
transform_output_ckpt='/transform_model/t_model.h5'

# Quantization
quant_output_ckpt='/quant_model/q_model.h5'

# Compile
compile_dir='/compiled_model_'
model_name='mobilenetv2'

# Target
target_dir='/target_'

# Application code
app_dir='application'
