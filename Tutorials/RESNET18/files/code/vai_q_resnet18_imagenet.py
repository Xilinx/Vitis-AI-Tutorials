#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

"""
Vitis AI quantization of ResNet-18 trained on ImageNet Dataset

Returns:

"""


# ==========================================================================================
# import dependencies
# ==========================================================================================


from config import imagenet_config    as cfg #DB
#from config import imagenet_input_fn  as preproc #DB


import argparse
import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow as tf
# placeholders are not executable immediately so we need to disable eager exicution in TF 2 not in 1
tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD

from classification_models.keras import Classifiers



# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI TF2 Quantization of ResNet18 pre-trained on ImageNet")

    # model config
    parser.add_argument("--float_file", type=str, default="./build/float/float_resnet18_imagenet.h5",
                        help="h5 floating point file full path name")
    # others
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu devices.")
    # quantization config
    parser.add_argument("--quant_file", type=str, default="./build/quantized/q_resnet18_imagenet.h5",
                        help="quantized model file full path ename ")

    return parser.parse_args()


#def main():
args = get_arguments()

# ==========================================================================================
# Global Variables
# ==========================================================================================
print(cfg.SCRIPT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

FLOAT_HDF5_FILE = os.path.join(cfg.SCRIPT_DIR,  args.float_file)
QUANT_HDF5_FILE = os.path.join(cfg.SCRIPT_DIR,  args.quant_file)


# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[DB INFO] Preparing Data ...\n")

DATAS_DIR = cfg.DATASET_DIR
TEST_DIR  = cfg.VALID_DIR

#Reading image paths
test_img_paths = [img_path for img_path in glob.glob(TEST_DIR+"/*.JPEG")]
NUMEL = len(test_img_paths)
assert (NUMEL  > 0 )

y_test= np.zeros((NUMEL,1),      dtype="uint16")
x_test= np.zeros((NUMEL,cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH,3),dtype="uint8")

i = 0
for img_path in test_img_paths:
    # B G R format
    img2 = cv2.imread(img_path)
    B, G, R = cv2.split(img2)     #resnet18
    img = cv2.merge([R, G, B])    #resnet18
    #img = img2                   #resnet50
    height, width = img.shape[0], img.shape[1]
    img = img.astype(float)

    # aspect_preserving_resize
    smaller_dim = np.min([height, width])
    _RESIZE_MIN = 256
    scale_ratio = _RESIZE_MIN*1.0 / (smaller_dim*1.0)
    new_height = int(height * scale_ratio)
    new_width = int(width * scale_ratio)
    #resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR )

    resized_img = img
    # central_crop
    crop_height = 224
    crop_width = 224
    amount_to_be_cropped_h = (new_height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (new_width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    cropped_img = resized_img[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width, :]

    # sub mean
    _R_MEAN =      0 #resnet18
    _G_MEAN =      0 #resnet18
    _B_MEAN =      0 #resnet18
    #_R_MEAN = 123.68 #resnet50
    #_G_MEAN = 116.78 #resnet50
    #_B_MEAN = 103.94 #resnet50
    _CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
    means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
    meaned_img = cropped_img - means

    img_array  = img_to_array(meaned_img, data_format=None)
    filename   = os.path.basename(img_path)
    class_name = filename.split(".")[0]
    label      = cfg.labelNames_dict[class_name] +0# class index
    print("filename:  ", img_path)
    print("classname: ", class_name)
    print("label    : ", label)
    x_test[i] = img_array
    y_test[i] = np.uint16(label)
    i = i + 1

X_test  = np.asarray(x_test)
Y_test  = np.asarray(y_test)
print("X Y shapes: ", X_test.shape, Y_test.shape)

# ==========================================================================================
# Get ResNet18 pre-trained model on ImageNet
# ==========================================================================================
loss = keras.losses.SparseCategoricalCrossentropy()
metric_top_5 = keras.metrics.SparseTopKCategoricalAccuracy()
accuracy = keras.metrics.SparseCategoricalAccuracy()


print("\n[DB INFO] Get ResNet18 pretrained model...\n")
# original imagenet-based ResNet18 model
ResNet18, preprocess_input = Classifiers.get("resnet18")
print("ResNet18 preprocess_input ", preprocess_input)
model18 = ResNet18((224, 224, 3), weights="imagenet")
#print(model18.summary())
model18.compile(optimizer="adam", loss=loss, metrics=[accuracy, metric_top_5])
#print("\n[DB INFO] Saving ResNet-18 Imagenet floating point Model...\n")
#model18.save(FLOAT_HDF5_FILE)

# ==========================================================================================
# Prediction
# ==========================================================================================
print("\n[DB INFO] Make Predictions with Float Model...\n")

### Evaluation on Test Dataset
num_validation_images = 500
eval_batch_size=50
num_steps=num_validation_images/eval_batch_size

#t_ModelLoss, t_Model_top1, t_Model_top5 = model50.evaluate(X_test, Y_test, steps=num_steps)
t_ModelLoss, t_Model_top1, t_Model_top5 = model18.evaluate(X_test, Y_test, steps=num_steps)
print("X_test Model Loss is {}".format(t_ModelLoss))
print("X_test Model top1 is {}".format(t_Model_top1))
print("X_test Model top5 is {}".format(t_Model_top5))


# ==========================================================================================
# Vitis AI Quantization
# ==========================================================================================
print("\n[DB INFO] Vitis AI Quantization...\n")

from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model18)
q_model = quantizer.quantize_model(calib_dataset=X_test[0:100])

print("\n[DB INFO] Evaluation of Quantized Model...\n")
with vitis_quantize.quantize_scope():
    #q_model = tf.keras.models.load_model("./quantized.h5", custom_objects=custom_objects)
    #q_model = tf.keras.models.load_model(QUANT_HDF5_FILE)
    q_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    q_eval_results = q_model.evaluate(X_test, X_test)
    print("\n***************** Summary *****************")
    print("X_Test Quantized model accuracy: ", q_eval_results[1])

print("\n[DB INFO] Saving Quantized Model...\n")
q_model.save(QUANT_HDF5_FILE)
loaded_model = keras.models.load_model(QUANT_HDF5_FILE)
eval_results = loaded_model.evaluate(X_test, Y_test)
print("\n***************** Summary *****************")
print("X_Test Quantized model accuracy: ", eval_results[1])
"""
