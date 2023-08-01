#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
'''

# date 28 Apr 2023


import cv2
import os
import numpy as np

from config import fashion_mnist_config as cfg


calib_image_dir  = cfg.SCRIPT_DIR + "/../dataset/fashion-mnist/calib/"
calib_image_list = calib_image_dir +  "calib_list.txt"
print("script running on folder ", cfg.SCRIPT_DIR)
print("CALIB DIR ", calib_image_dir)

calib_batch_size = 50

_R_MEAN = 0
_G_MEAN = 0
_B_MEAN = 0

MEANS = np.array([_B_MEAN,_G_MEAN,_R_MEAN],np.dtype(np.int32))

def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image

def central_crop(image, crop_height, crop_width):
  image_height = image.shape[0]
  image_width = image.shape[1]

  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2

  return image[offset_height:offset_height + crop_height, offset_width:
               offset_width + crop_width]

'''
def normalize(image):
  image=image/cfg.NORM_FACTOR
  image=image-0.5
  image=image*2
  return image
'''

def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  #print(line)
  for index in range(0, calib_batch_size):
      curline = line[iter * calib_batch_size + index]
      #print(curline)
      calib_image_name = curline.strip()

      # read image as grayscale, returns numpy array (28,28)
      #image = cv2.imread(calib_image_dir + calib_image_name, cv2.IMREAD_GRAYSCALE)

      # read image as rgb, returns numpy array (28,28, 3)
      image = cv2.imread(calib_image_dir + calib_image_name)

      # scale the pixel values to range 0 to 1.0
      #image = image/255.0
      image2 = cfg.Normalize(image)
      #image = central_crop(image, 28, 28) #DB
      #image = mean_image_subtraction(image, MEANS) #DB
      #image = cfg.ScaleTo1(image) #DB

      # reshape numpy array to be (28,28,3)
      image2 = image2.reshape((image2.shape[0], image2.shape[1], 3))
      images.append(image2)

  return {"conv2d_1_input": images}


#######################################################

def main():
  calib_input(20)


if __name__ == "__main__":
    main()
