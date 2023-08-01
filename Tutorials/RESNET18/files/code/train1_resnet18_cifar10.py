#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# Train ResNet-18 on CIFAR10 data stored as file of images

# ==========================================================================================
# import dependencies
# ==========================================================================================

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cifar10_config as cfg #DB
print(cfg.SCRIPT_DIR)

# import the necessary packages
from sklearn.metrics import classification_report
import numpy as np
import cv2
from datetime import datetime

import os
import argparse
from random import seed
from random import random
from random import shuffle
import glob


## Import usual libraries
import tensorflow as tf
from tensorflow                             import keras
from tensorflow.keras                       import backend as K
from tensorflow.keras.utils                 import plot_model, to_categorical #DB
from tensorflow.keras.preprocessing.image   import ImageDataGenerator #DB
from tensorflow.keras                       import optimizers
from tensorflow.keras.optimizers            import SGD
from tensorflow.keras.callbacks             import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.datasets              import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

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
    ap.add_argument("-w",  "--weights", default="build/float",      help="path to best model HDF5 weights file")
    #ap.add_argument("-n",  "--network", default="ResNet18_cifar10_train1", help="input CNN")
    #ap.add_argument("-d",  "--dropout",   type=int, default=-1,     help="whether or not Dropout should be used")
    #ap.add_argument("-bn", "--BN",        type=int, default=-1,     help="whether or not BN should be used")
    ap.add_argument("-e",  "--epochs",    type=int, default=50,     help="# of epochs")
    ap.add_argument("-bs", "--batch_size",type=int, default=256,    help="size of mini-batches passed to network")
    ap.add_argument("-g",  "--gpus",      type=str, default="0",    help="choose gpu devices.")
    ap.add_argument("-l",  "--init_lr",   type=float, default=0.01, help="initial Learning Rate")
    return ap.parse_args()

args  = vars(get_arguments())
args2 = get_arguments()

for key, val in args2._get_kwargs():
    print(key+" : "+str(val))

# ==========================================================================================
# Global Variables
# ==========================================================================================

print(cfg.SCRIPT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = args["gpus"]
## Silence TensorFlow messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS = args["weights"]
#NETWORK = args["network"]

NUM_EPOCHS = args["epochs"]     #25
INIT_LR    = args["init_lr"]    #1e-2
BATCH_SIZE = args["batch_size"] #32

# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[DB INFO] Creating lists of images ...\n")

# make a list of all files currently in the TRAIN folder
imagesList = [img for img in glob.glob(cfg.TRAIN_DIR + "/*/*.png")]

seed(42)
shuffle(imagesList)
x_train, y_train = list(), list()
for img in imagesList:
        filename = os.path.basename(img)
        classname = filename.split("_")[0]
        #print(img)
        # read image with OpenCV
        img_orig = cv2.imread(img)
        #rs_img = cv2.resize(img_orig, (256,256))
        #cv2.imshow(classname, rs_img)
        #cv2.waitKey(0)
        img_array = img_to_array(img_orig, data_format=None)
        x_train.append(img_array)
        y_train.append(cfg.labelNames_dict[classname])

print("[DB INFO] x_train: done...")
# make a list of all files currently in the VALID folder
imagesList = [img for img in glob.glob(cfg.VALID_DIR + "/*/*.png")]
shuffle(imagesList)
x_valid, y_valid = list(), list()
for img in imagesList:
        filename = os.path.basename(img)
        classname = filename.split("_")[0]
        #print(img)
        # read image with OpenCV
        img_orig = cv2.imread(img)
        img_array = img_to_array(img_orig, data_format=None)
        x_valid.append(img_array)
        y_valid.append(cfg.labelNames_dict[classname])

print("[DB INFO] x_valid: done...")
# make a list of all files currently in the VALID folder
imagesList = [img for img in glob.glob(cfg.TEST_DIR + "/*/*.png")]
shuffle(imagesList)
x_test, y_test = list(), list()
for img in imagesList:
    filename = os.path.basename(img)
    classname = filename.split("_")[0]
    #print(img)
    # read image with OpenCV
    img_orig = cv2.imread(img)
    img_array = img_to_array(img_orig, data_format=None)
    x_test.append(img_array)
    y_test.append(cfg.labelNames_dict[classname])

print("[DB INFO] x_test: done...")
# one-hot encode the training and testing labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)
y_valid = to_categorical(y_valid, 10)

# check settings #DB
assert True, ( len(x_train) > cfg.NUM_TRAIN_IMAGES)
assert True, ( len(x_test) >= (cfg.NUM_TRAIN_IMAGES+cfg.NUM_VAL_IMAGES))
assert True, ( cfg.NUM_TRAIN_IMAGES==cfg.NUM_VAL_IMAGES )

# ==========================================================================================
# Pre-processing data
# ==========================================================================================
print("\n[DB INFO] Preprocessing images ...\n")

x_test  = np.asarray(x_test)
x_train = np.asarray(x_train)
x_valid = np.asarray(x_valid)

#Normalize and convert from BGR to RGB
x_train = cfg.Normalize(x_train)
x_test  = cfg.Normalize(x_test)
x_valid = cfg.Normalize(x_valid)


# ==========================================================================================
# Data Generators
# ==========================================================================================
print("\n[DB INFO] Data Generators ...\n")
test_datagen  = ImageDataGenerator()
train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()
aug_datagen   = ImageDataGenerator(
        #rescale=1/255,
        rotation_range=5,
        horizontal_flip=True,
        height_shift_range=0.05,
        width_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.2)

aug_generator = aug_datagen.flow(
        x_train, y_train,
        batch_size=BATCH_SIZE)

train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=BATCH_SIZE)

validation_generator = valid_datagen.flow(
        x_valid, y_valid,
        batch_size=BATCH_SIZE)

pred_generator = test_datagen.flow(
        x_test, y_test,
        batch_size=1)


# ==========================================================================================
# CallBack Functions
# ==========================================================================================
print("\n[DB INFO] CallBack Functions ...\n")

# construct the callback to save only the *best* model to disk
# based on the validation accuray
fname = os.path.sep.join([WEIGHTS, "train1_best_chkpt.h5"])
checkpoint = ModelCheckpoint(fname,
		monitor="val_accuracy", mode="max",
		save_best_only=True, verbose=1)

callbacks_list = [checkpoint]

def poly_decay(epoch):
                # initialize the maximum number of epochs, base learning rate, and power of the polynomial
                maxEpochs = NUM_EPOCHS
                baseLR = INIT_LR
                power = 1.0
                # compute the new learning rate based on polynomial decay
                alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
                # return the new learning rate
                return alpha

callbacks_list = [checkpoint, LearningRateScheduler(poly_decay)]

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
x_layer = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax")(x_layer)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
#model.summary()

# ==========================================================================================
# Training for 50 epochs on Cifar-10
# ==========================================================================================
print("\n[DB INFO] Training the Model...\n")

opt = SGD(learning_rate=INIT_LR, momentum=0.9)
#opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

startTime1 = datetime.now() #DB
# run training/media/danieleb/DATA$
H = model.fit(aug_generator,
            steps_per_epoch=len(x_train)//BATCH_SIZE,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=len(x_valid)//BATCH_SIZE,
            callbacks=callbacks_list,
            shuffle=True,verbose=2)

endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

# save CNN complete model on HDF5 file #DB
fname1 = os.path.sep.join([WEIGHTS, "train1_final.h5"])
model.save(fname1)
# once saved the model can be load with following commands #DB
#from keras.models import load_model
#print("[INFO] loading pre-trained network...") #DB
#model = load_model(fname) #DB

# plot the CNN model #DB
print("\n[DB INFO] plot model...\n")
model_filename = os.path.join(cfg.SCRIPT_DIR, "build/log/train1_float_model.png")
plot_model(model, to_file=model_filename, show_shapes=True)


# ==========================================================================================
# Prediction
# ==========================================================================================
print("\n[DB INFO] evaluating network on Test and Validation datasets...\n")
# Evaluate model accuracy with test set
scores = model.evaluate(x_valid, y_valid, batch_size=BATCH_SIZE) #MH
print('Validation Loss: %.3f'     % scores[0]) #MH
print('validation Accuracy: %.3f' % scores[1]) #MH
scores = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE) #MH
print('Test Loss: %.3f'     % scores[0]) #MH
print('Test Accuracy: %.3f' % scores[1]) #MH

# make predictions on the test set
preds = model.predict(x_test)
# show a nicely formatted classification report
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=cfg.labelNames_list))

# ==========================================================================================
# Plot files
# ==========================================================================================
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
print(H.history.keys())
plotmodelhistory(H)
plot_filename = os.path.join(cfg.SCRIPT_DIR, "build/log/train1_history.png")
plt.savefig(plot_filename)

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("./doc/images/" + "train1_resnet18_cifar10_network" + "_plot.png")


# ==========================================================================================
print("\n[DB INFO] End of ResNet18 Training1 on CIFAR10...\n")
