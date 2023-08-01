#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# ==========================================================================================
# import dependencies
# ==========================================================================================


from config import cifar10_config as cfg #DB

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.utils import np_utils

import tensorflow as tf
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
    parser = argparse.ArgumentParser(description="Vitis AI TF2 Quantization of ResNet18 trained on CIFAR10")

    # model config
    parser.add_argument("--float_file", type=str, default="./build/float/train2_resnet18_cifar10.h5",
                        help="h5 floating point file full path name")
    # others
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu devices.")
    # quantization config
    parser.add_argument("--quant_file", type=str, default="./build/quantized/q_train2_resnet18_cifar10.h5",
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

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train.shape, X_test.shape, np.unique(Y_train).shape[0]
# one-hot encoding
n_classes = 10

# Normalize the data
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = cfg.Normalize(X_train)
X_test  = cfg.Normalize(X_test)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)

encoder = OneHotEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train).toarray()
Y_test = encoder.transform(Y_test).toarray()
Y_val =  encoder.transform(Y_val).toarray()

# ==========================================================================================
# Get the trained floating point model
# ==========================================================================================

model = keras.models.load_model(FLOAT_HDF5_FILE)

# ==========================================================================================
# Prediction
# ==========================================================================================
print("\n[DB INFO] Make Predictions with Float Model...\n")

## Evaluation on Training Dataset
ModelLoss, ModelAccuracy = model.evaluate(X_train, Y_train)
print("X_Train Model Loss     is {}".format(ModelLoss))
print("X_Train Model Accuracy is {}".format(ModelAccuracy))

## Evaluation on Test Dataset
t_ModelLoss, t_ModelAccuracy = model.evaluate(X_test, Y_test)
print("X_Test Model Loss     is {}".format(t_ModelLoss))
print("X_Test Model Accuracy is {}".format(t_ModelAccuracy))


# ==========================================================================================
# Vitis AI Quantization
# ==========================================================================================
print("\n[DB INFO] Vitis AI Quantization...\n")

from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
q_model = quantizer.quantize_model(calib_dataset=X_train[0:100])
#q_model.save(QUANT_HDF5_FILE)

print("\n[DB INFO] Evaluation of Quantized Model...\n")
with vitis_quantize.quantize_scope():
    #q_model = tf.keras.models.load_model("./quantized.h5", custom_objects=custom_objects)
    #q_model = tf.keras.models.load_model(QUANT_HDF5_FILE)
    q_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    q_eval_results = q_model.evaluate(X_test, Y_test)
    print("\n***************** Summary *****************")
    print("X_Test Quantized model accuracy: ", q_eval_results[1])

print("\n[DB INFO] Saving Quantized Model...\n")
q_model.save(QUANT_HDF5_FILE)
loaded_model = keras.models.load_model(QUANT_HDF5_FILE)
#eval_results = loaded_model.evaluate(X_test, Y_test)
#print("\n***************** Summary *****************")
#print("X_Test Quantized model accuracy: ", eval_results[1])
eval_results = loaded_model.evaluate(X_train, Y_train)
print("\n***************** Summary *****************")
print("X_Train Quantized model accuracy: ", eval_results[1])
