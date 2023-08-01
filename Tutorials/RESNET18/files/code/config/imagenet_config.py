#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023



import os
import numpy as np
import cv2

###############################################################################
# project folders
###############################################################################

def get_script_directory():
    path = os.getcwd()
    return path

# get current directory
SCRIPT_DIR = get_script_directory()

# dataset top level folder
DATASET_DIR = os.path.join(SCRIPT_DIR, "target/imagenet")#DB

# validation folder
VALID_DIR  = os.path.join(DATASET_DIR, "val_dataset")


###############################################################################
# global variables
###############################################################################

# since we do not have validation data or access to the testing labels we need
# to take a number of images from the training data and use them instead
NUM_CLASSES      =  1000
NUM_VAL_IMAGES   =   500

#Size of images
IMAGE_WIDTH  = 224
IMAGE_HEIGHT = 224

#normalization factor to scale image 0-255 values to 0-1 #DB
NORM_FACTOR = 255.0 # could be also 256.0

# As labels (which are indeed the "class index") for the ImageNet Validation dataset you take the
# numbers of "val.txt" which are related to the class names of "words.txt"
labelNames_dict = {}
VAL_TXT = os.path.join(DATASET_DIR, "val.txt")
with open(VAL_TXT, 'r') as f:
    lines = f.readlines()
val_num = 0
for k in range(NUM_VAL_IMAGES):
    line = lines[k]
    val_num += 1
    #if val_num % 1000 == 0:
    #    print('preprocess %d / %d'%(val_num, len(lines)))
    (filename, val) = line.strip().split(" ") #line.rstrip().split(" ")
    filename = os.path.basename(filename)
    key = filename.split(".")[0]
    labelNames_dict[key]= int(val)
labelNames_list = list(labelNames_dict.values())
"""
d = labelNames_dict
print(d)
print(d.keys())
print(d.values())
"""

#lists with the class name and its class index, from 0 to 999
class_list = []
index_list = []
WORDS_TXT = os.path.join(DATASET_DIR, "words.txt")
with open(WORDS_TXT, 'r') as f1:
    lines = f1.readlines()
val_num = -1
for k in range(NUM_CLASSES):
    line = lines[k]
    val_num += 1
    #if val_num % 1000 == 0:
    #    print('preprocess %d / %d'%(val_num, len(lines)))
    key = line.strip()
    class_list.append(key)
    index_list.append(int(val_num))

"""
print(val_num, k)
for kk in range(NUM_CLASSES):
    print(index_list[kk], class_list[kk])
"""
