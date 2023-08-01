#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


# ##################################################################################################

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

from datetime import datetime
import os
import argparse

from config import imagenet_config as cfg


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--file",  required=True, help="input logfile")
ap.add_argument("-n", "--numel", default="500", help="number of test images")
args = vars(ap.parse_args())

logfile = args["file"] # root path name of dataset

try:
    f = open(logfile, "r")
except IOError:
    print("cannot open ", logfile)
else:
    lines = f.readlines()
    tot_lines = len(lines)
    print(logfile, " has ", tot_lines, " lines")

#f.seek(0)
f.close()

# ##################################################################################################

NUMEL = int(args["numel"]) #500

 ##################################################################################################

top1_true  = 0
top1_false = 0
top5_true  = 0
top5_false = 0
img_count  = 0
false_pred = 0

test_ids = np.zeros(([NUMEL,1]))
preds    = np.zeros(([NUMEL, 1]))
idx = 0

for ln in range(0, tot_lines):
    if "Image" in lines[ln]:
        top5_lines = lines[ln:ln+6]
        filename= top5_lines[0].split("Image :")[1]
        class_name   = filename.split(".JPEG ")[0].strip()
        predicted    = int(filename.split("out index = ")[1].strip())
        ground_truth = int(cfg.labelNames_dict[class_name])
        if predicted == ground_truth :
            top1_true += 1
            #top5_true += 1
        else:
            #top5_false += 1
            top1_false +=1
        #check top5
        for i in range (1, 5):
            filename= top5_lines[0].split("Image :")[1]
            class_name   = filename.split(".JPEG ")[0].strip()
            predicted    = int(filename.split("out index = ")[1].strip())
            ground_truth = int(cfg.labelNames_dict[class_name])

        img_count +=1
        idx += 1
        if ( idx == (NUMEL-1) ):
            break
    else:
        continue

assert (top1_true+top1_false)  == img_count, "ERROR: top1 true+false not equal to the number of images"
#assert (top5_true+top5_false)  == img_count, "ERROR: top5 true+false not equal to the number of images"

print("number of total images predicted ", img_count)
print("number of top1 false predictions ", top1_false)
print("number of top1 right predictions ", top1_true)
#print("number of top5 false predictions ", top5_false)
#print("number of top5 right predictions ", top5_true)

top1_accuracy = float(top1_true)/(top1_true+top1_false)
#top5_accuracy = float(top5_true)/(top5_true+top5_false)

print("top1 accuracy = %.2f" % top1_accuracy)
#print("top5 accuracy = %.2f" % top5_accuracy)
