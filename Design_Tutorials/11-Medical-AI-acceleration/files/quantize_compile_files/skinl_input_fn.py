#!/usr/bin/env python
# coding: utf-8
# 
# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2

INPUT_SHAPE = (224,224)
IMAGE_DIR = './data/calib/'

calib_image_list="./data/calibration.txt"
calib_batch_size = 10

def calib_input(iter):
    images =[]
    line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        cline = line[iter * calib_batch_size+index]
        calib_image_name = cline.strip()

        #open imagea as BGR
        print('processing images: {}\n'.format(IMAGE_DIR + calib_image_name))
        image = cv2.imread(IMAGE_DIR + calib_image_name)
        #convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #resize to (224, 224) cv2 image shape is (width/column, height/row)
        image = cv2.resize(image, INPUT_SHAPE)
        image = image/255.0
        #append in the list
        images.append(image)
    
    images = np.asarray(images)
    print('input shape of iter {0} is {1}\n'.format(iter, images.shape))
    return { "input_1": images}



