#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# Train ResNet-18 on CIFAR10 data loaded into memory

# based on "Implementing ResNet-18 Using Keras" from
# https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras/notebook

# ==========================================================================================
# References
# ==========================================================================================

#https://colab.research.google.com/github/bhgtankita/MYWORK/blob/master/Grad_CAM_RESNET18_Transfer_Learning_on_CIFAR10.ipynb#scrollTo=vhO24OrY0ckv

# https://github.com/songrise/CNN_Keras

#https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/tensorflow/saving_and_serializing.ipynb#scrollTo=yKikmbdC3O_i

#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

#https://www.kaggle.com/code/parasjindal96/basic-deep-learning-tutorial-using-keras/notebook

# ==========================================================================================
# import dependencies
# ==========================================================================================

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cifar10_config as cfg #DB
print(cfg.SCRIPT_DIR)

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.utils import plot_model,  to_categorical
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
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="TF2 ResNet18 Training on Cifar10 Dataset stored as files")
    ap.add_argument("-w",  "--weights", default="build/float",      help="path to best model h5 weights file")
    #ap.add_argument("-n",  "--network", default="ResNet18_cifar10", help="input CNN")
    #ap.add_argument("-d",  "--dropout",   type=int, default=-1,     help="whether or not Dropout should be used")
    #ap.add_argument("-bn", "--BN",        type=int, default=-1,     help="whether or not BN should be used")
    ap.add_argument("-e",  "--epochs",    type=int, default=50,     help="# of epochs")
    ap.add_argument("-bs", "--batch_size",type=int, default=256,    help="size of mini-batches passed to network")
    ap.add_argument("-g",  "--gpus",      type=str, default="0",    help="choose gpu devices.")
    #ap.add_argument("-l",  "--init_lr",   type=float, default=0.01, help="initial Learning Rate")
    return ap.parse_args()

args = vars(get_arguments())
args2 = get_arguments()

for key, val in args2._get_kwargs():
    print(key+" : "+str(val))


# ==========================================================================================
# Global Variables
# ==========================================================================================

print(cfg.SCRIPT_DIR)

## Silence TensorFlow messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS  = args["weights"]
#NETWORK = args["network"]

NUM_EPOCHS = args["epochs"]     #25
#INIT_LR    = args["init_lr"]    #1e-2
BATCH_SIZE = args["batch_size"] #32

# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[DB INFO] Loading Data for Training and Test...\n")

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train.shape, X_test.shape, np.unique(Y_train).shape[0]
# one-hot encoding
n_classes = cfg.NUM_CLASSES

# Pre-processing & Normalize the data
X_train = cfg.Normalize(X_train)
X_test  = cfg.Normalize(X_test)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)

encoder = OneHotEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train).toarray()
Y_test = encoder.transform(Y_test).toarray()
Y_val =  encoder.transform(Y_val).toarray()

# data augmentation
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05,
                         height_shift_range=0.05)
aug.fit(X_train)


# ==========================================================================================
"""
ResNet-18
Reference:
[1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In
ICCV, 2015.
"""

# WARNING
# This Subclassed Model cannot be save in HDF5 format, which is what Vitis AI Quantizer requires
# see Table 17 from UG1414

"""
class ResnetBlock(Model):
    # A standard resnet block.

    def __init__(self, channels: int, down_sample=False):
        #channels: same as number of convolution kernels
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a "glorot_uniform" in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        #num_classes: number of classes in specific classification task.

        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

"""
# ==========================================================================================


# ==========================================================================================
# Get ResNet18 pre-trained model
# ==========================================================================================
print("\n[DB INFO] Get ResNet18 pretrained model...\n")

# original imagenet-based ResNet18 model
ResNet18, preprocess_input = Classifiers.get("resnet18")
#orig_model = ResNet18((224, 224, 3), weights="imagenet")
#print(orig_model.summary())


# build new model for CIFAR10
base_model = ResNet18(input_shape=(32,32,3), weights="imagenet", include_top=False)
#next to lines commented: the training would become awful
##for layer in base_model.layers:
##    layer.trainable = False
"""
dict_keys(["loss", "accuracy", "val_loss", "val_accuracy"])
X_Train Model Loss is     2.185528039932251
X_Train Model Accuracy is 0.2323250025510788
X_Test Model Loss is      2.185682535171509
X_Test Model Accuracy is  0.23180000483989716
"""
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
#model.summary()

# ==========================================================================================
# CallBack Functions
# ==========================================================================================
print("\n[DB INFO] CallBack Functions ...\n")
es = EarlyStopping(patience= 8, restore_best_weights=True, monitor="val_accuracy")


# ==========================================================================================
# Training for 50 epochs on Cifar-10
# ==========================================================================================
print("\n[DB INFO] Training the Model...\n")

#use categorical_crossentropy since the label is one-hot encoded
# opt = SGD(learning_rate=0.1,momentum=0.9,decay = 1e-04) #parameters suggested by He [1]
model.compile(optimizer = "adam",loss="categorical_crossentropy", metrics=["accuracy"])


#I did not use cross validation, so the validate performance is not accurate.
STEPS = len(X_train) // BATCH_SIZE
startTime1 = datetime.now() #DB
history = model.fit(aug.flow(X_train,Y_train,batch_size = BATCH_SIZE),
                    steps_per_epoch=STEPS, batch_size = BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_data=(X_train, Y_train),callbacks=[es])
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

print("\n[DB INFO] saving HDF5 model...\n")
##model.save("resnet18_cifar10_float", save_format="tf") #TF2 saved model directory
fname1 = os.path.sep.join([WEIGHTS, "train2_resnet18_cifar10_float.h5"])
model.save(fname1) #HDF5 Keras saved model file
# once saved the model can be load with following commands #DB
#from keras.models import load_model
#print("[INFO] loading pre-trained network...") #DB
#model = load_model(fname) #DB

print("\n[DB INFO] plot model...\n")
model_filename = os.path.join(cfg.SCRIPT_DIR, "build/log/train2_float_model.png")
plot_model(model, to_file=model_filename, show_shapes=True)

# ==========================================================================================
# Prediction
# ==========================================================================================
print("\n[DB INFO] Make Predictions with Float Model on CIFAR10...\n")

## Evaluation on Training Dataset
ModelLoss, ModelAccuracy = model.evaluate(X_train, Y_train)
print("X_Train Model Loss is {}".format(ModelLoss))
print("X_Train Model Accuracy is {}".format(ModelAccuracy))
"""
# expected results
X_Train Model Loss is 0.057520072907209396
X_Train Model Accuracy is 0.9808750152587891
"""

## Evaluation on Test Dataset
t_ModelLoss, t_ModelAccuracy = model.evaluate(X_test, Y_test)
print("X_Test Model Loss is {}".format(t_ModelLoss))
print("X_Test Model Accuracy is {}".format(t_ModelAccuracy))
"""
# expected results
X_Test Model Loss is 0.7710350751876831
X_Test Model Accuracy is 0.8434000015258789
"""


# make predictions on the test set
preds = model.predict(X_test)
# show a nicely formatted classification report
print(classification_report(Y_test.argmax(axis=1), preds.argmax(axis=1), target_names=cfg.labelNames_list))


# ==========================================================================================
# Training curves
# ==========================================================================================
print("\n[DB INFO] Generate Training Curves File...\n")

def plotmodelhistory(history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(history.history["accuracy"])
    axs[0].plot(history.history["val_accuracy"])
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")

    axs[0].legend(["train", "validate"], loc="upper left")
    # summarize history for loss
    axs[1].plot(history.history["loss"])
    axs[1].plot(history.history["val_loss"])
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["train", "validate"], loc="upper left")
    plt.show()

# list all data in history
print(history.history.keys())
plotmodelhistory(history)
plot_filename = os.path.join(cfg.SCRIPT_DIR, "build/log/train2_history.png")
plt.savefig(plot_filename)


# ==========================================================================================
print("\n[DB INFO] End of ResNet18 Training2 on CIFAR10...\n")
