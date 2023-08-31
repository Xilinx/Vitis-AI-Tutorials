#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# ==========================================================================================
# import dependencies
# ==========================================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
keras = tf.keras
from keras.utils import Sequence

from config import imagenet_config as cfg
print(cfg.SCRIPT_DIR)

eval_batch_size = 50
EVAL_NUM = cfg.NUM_VAL_IMAGES
NUMEL    = cfg.NUM_CLASSES

# ==========================================================================================
# get file names
# ==========================================================================================

def get_images_infor_from_file(image_dir, image_list, label_offset):
  with open(image_list, 'r') as fr:
    lines = fr.readlines()
  imgs = []
  labels = []
  for line in lines:
    img_name, label = line.strip().split(" ")
    #print("img_name ", img_name)
    #print("orig label ", label)
    img_path = os.path.join(image_dir, img_name)
    label = int(label) + 1 - label_offset
    #print("new  label ", label)
    imgs.append(img_path)
    labels.append(label)
    ##cross check
    filename   = os.path.basename(img_path)
    class_name = filename.split(".")[0]
    label2      = cfg.labelNames_dict[class_name] +0# class index
    #print("filename:  ", img_path)
    #print("classname: ", class_name)
    #print("label2   : ", label2)
    assert label2==label, "found a mismatch in labels"

  return imgs, labels

# ==========================================================================================
# Resnet50 Images Sequencer
# ==========================================================================================

class ImagenetSequence_ResNet50(Sequence):
  def __init__(self, filenames, labels, batch_size):
    self.filenames, self.labels = filenames, labels
    self.batch_size = batch_size

  def __len__(self):
    return int(np.ceil(len(self.filenames) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

    processed_imgs = []

    for filename in batch_x:
      # B G R format
      img2 = cv2.imread(filename)
      #B, G, R = cv2.split(img2)     #resnet18
      #img = cv2.merge([R, G, B])    #resnet18
      img = img2                     #resnet50
      height, width = img.shape[0], img.shape[1]
      img = img.astype(float)

      # aspect_preserving_resize
      smaller_dim = np.min([height, width])
      _RESIZE_MIN = 256
      scale_ratio = _RESIZE_MIN*1.0 / (smaller_dim*1.0)
      new_height = int(height * scale_ratio)
      new_width = int(width * scale_ratio)
      resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR )
      #resized_img = img

      # central_crop
      crop_height = 224
      crop_width = 224
      amount_to_be_cropped_h = (new_height - crop_height)
      crop_top = amount_to_be_cropped_h // 2
      amount_to_be_cropped_w = (new_width - crop_width)
      crop_left = amount_to_be_cropped_w // 2
      cropped_img = resized_img[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width, :]

      ##very simplified and brutal way to rescale the image
      #cropped_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)

      # sub mean
      #_R_MEAN =     0 #resnet18
      #_G_MEAN =     0 #resnet18
      #_B_MEAN =     0 #resnet18
      _R_MEAN = 123.68 #resnet50
      _G_MEAN = 116.78 #resnet50
      _B_MEAN = 103.94 #resnet50
      _CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
      means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
      meaned_img = cropped_img - means

      # model.predict(np.expand_dims(meaned_img, 0))
      # model.evaluate(np.expand_dims(meaned_img, 0), np.expand_dims(labels[0], 0))
      processed_imgs.append(meaned_img)

    return np.array(processed_imgs), np.array(batch_y)


# ==========================================================================================
# Evaluate Floating Point Resnet50 CNN
# ==========================================================================================

# get input data with proper preprocessing
print("\n[DB INFO] Get Input Data with Proper Pre-processing...\n")
dataset_dir   = cfg.VALID_DIR
val_list_file = os.path.sep.join([cfg.DATASET_DIR, "val.txt"])
labels_offset = 1
img_paths, labels = get_images_infor_from_file(dataset_dir, val_list_file, labels_offset)
imagenet_seq50    = ImagenetSequence_ResNet50(img_paths, labels, 50)

# get CNN pre-trained on ImageNet
print("\n[DB INFO] Get ResNet50 CNN pre-trained on ImageNet...\n")
from tensorflow.keras.applications.resnet50 import ResNet50
modelzoo_cnn=os.path.sep.join([cfg.SCRIPT_DIR,"modelzoo/tf2_resnet50_imagenet_224_224_7.76G_3.0/float/resnet_50.h5"])
#next two lines are perfectly equivalent
model50 = ResNet50(weights="imagenet")
#model50 = ResNet50(weights=modelzoo_cnn)
model50.summary()
# save CNN complete model on HDF5 file #DB
cnn_filename = os.path.sep.join([cfg.SCRIPT_DIR, "build/float/float_resnet50_imagenet.h5"])
model50.save(cnn_filename)

print("\n[DB INFO] Compile ResNet50 CNN...\n")
# define accuracy metrics
loss = keras.losses.SparseCategoricalCrossentropy()
metric_top_5 = keras.metrics.SparseTopKCategoricalAccuracy()
accuracy = keras.metrics.SparseCategoricalAccuracy()
# compile CNN model
model50.compile(optimizer="adam", loss=loss, metrics=[accuracy, metric_top_5])

# evaluate CNN floating point model
print("\n[DB INFO] Evaluate Average Prediction Accuracy of ResNet50 CNN...\n")
res50 = model50.evaluate(imagenet_seq50, steps=EVAL_NUM/eval_batch_size, verbose=1)
print("Original  ResNet50 top1, top5: ", res50[1], res50[2])


# ==========================================================================================
# Quantize Resnet50 CNN
# ==========================================================================================
print("\n[DB INFO] Vitis AI PT Quantization of ResNet50 CNN...\n")
# get Vitis AI Quantizer
from tensorflow_model_optimization.quantization.keras import vitis_quantize

quantizer = vitis_quantize.VitisQuantizer(model50)
q_model50 = quantizer.quantize_model(calib_dataset=imagenet_seq50.__getitem__(0)[0])
QUANT_HDF5_FILE = os.path.sep.join([cfg.SCRIPT_DIR, "build/quantized/q_resnet50_imagenet.h5"])
q_model50.save(QUANT_HDF5_FILE)

# ==========================================================================================
# Evaluate Quantized Int8 Resnet50 CNN
# ==========================================================================================
print("\n[DB INFO] Evaluation of ResNet50 Quantized Model...\n")

#modelzoo_q_cnn=os.path.sep.join([cfg.SCRIPT_DIR,"modelzoo/tf2_resnet50_imagenet_224_224_7.76G_3.0/quantized/quantized.h5"])
#q_model = keras.models.load_model(modelzoo_q_cnn)
q_model = keras.models.load_model(QUANT_HDF5_FILE)

# build plain arrays of data and labels from Images Sequencer
X_test= np.zeros((EVAL_NUM,cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH,3),dtype="float32")
Y_test= np.zeros((EVAL_NUM,1)                                 ,dtype="float32")
start = 0
step  = eval_batch_size
stop  = EVAL_NUM
for i in range(start, stop, step):
    X_test[i:i+step, :, :, :]= np.asarray(imagenet_seq50.__getitem__(i//step)[0])
    Y_test[i:i+step, 0]      = np.asarray(imagenet_seq50.__getitem__(i//step)[1])
    #print(i)

X_test  = np.asarray(X_test)
Y_test  = np.asarray(Y_test)

with vitis_quantize.quantize_scope():
    #q_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    q_model.compile(optimizer="adam", loss=loss, metrics=[accuracy, metric_top_5])

    q_res = q_model.evaluate(X_test, Y_test)
    print("Quantized ResNet50 top1, top5: ", q_res[1] , q_res[2])
