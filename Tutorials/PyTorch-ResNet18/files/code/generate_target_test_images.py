#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
'''

# last change: 26 May 2023


##############################################################################################

import glob
import numpy as np

import os
import sys
import shutil
import cv2

from random import seed
from random import random
from random import shuffle #DB

##############################################################################################
# global variables
##############################################################################################

## ImageNet size
IMG_W = np.short(224)
IMG_H = np.short(224)

## 3 MPixel size (/2 just to try)
#IMG_W = np.short(1920/2)
#IMG_H = np.short(1536/2)

## 8 MPixel size
#IMG_W = np.short(3840)
#IMG_H = np.short(2048)

MAX_NUM_TEST_IMAGES = 300

labelNames_dict = {
"beige" : 0,   "black" : 1, "blue"   : 2, "brown" :  3, "gold"   :  4,
"green" : 5,    "grey" : 6, "orange" : 7, "pink"  :  8, "purple" :  9,
 "red" : 10, "silver" : 11, "tan"   : 12, "white" : 13, "yellow" : 14}

labelNames_list = [ "beige", "black",  "blue",   "brown", "gold",
                    "green", "grey",   "orange", "pink",  "purple",
                    "red",   "silver", "tan",    "white", "yellow"]

def get_script_directory():
    path = os.getcwd()
    return path

# get current directory
SCRIPT_DIR = get_script_directory()


##############################################################################################
# make the required folders
##############################################################################################
# dataset top level
TEST_DIR = os.path.join(SCRIPT_DIR, "./build/target/vcor/test")

# make a list of all files currently in the test folder
imagesList = [img_file for img_file in glob.glob(TEST_DIR + "/*/*.jpg")]

# seed random number generator
seed(1)
# randomly shuffle the list of images
shuffle(imagesList)

counter = np.array([0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0], dtype="uint32")
num_test = 0

# move the images to their class folders inside train (50000), valid (5000), test (5000)
# we want to be sure that all the folders contain same number of images per each class
for img_file in imagesList:
    filename_and_ext  = img_file.split("/")[-1]
    filename = filename_and_ext.split(".")[0]
    classname = img_file.split("/")[-2]
    print(" ")
    print("pathname     ", img_file)
    print("filename.jpg ", filename_and_ext)
    print("classname    ", classname)

    # read image with OpenCV
    img_orig = cv2.imread(img_file)
    label = labelNames_dict[classname]
    new_img_file = img_file.split(".jpg")[0] + "_" + classname + ".png"
    img_resiz = cv2.resize(img_orig, (IMG_W,IMG_H))
    cv2.imwrite(new_img_file, img_resiz)
    print("filename.png ", new_img_file)
    if os.path.isfile(img_file):
        os.remove(img_file)
        print("removing     ", img_file)
    num_test = num_test+1
    counter[ label ] = counter[ label ] +1;
    if (num_test == MAX_NUM_TEST_IMAGES) :
        break


    #out_filename = os.path.join(dst_dir, labelNames_list[label]+"/"+filename)
    #os.rename(img, out_filename)

print("classes histogram in test folder: ", counter)
print("num images in test folder  = ", num_test)
