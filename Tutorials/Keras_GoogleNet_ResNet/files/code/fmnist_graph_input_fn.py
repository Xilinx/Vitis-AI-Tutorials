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


calib_image_dir  = cfg.CALIB_DIR
calib_image_list = os.path.join(calib_image_dir,"calib_list.txt")
print("script running on folder ", cfg.SCRIPT_DIR)
print("CALIB DIR ", calib_image_dir)

calib_batch_size = 50
line = open(calib_image_list).readlines()
tot_num_images = len(line)

def calib_input(iter):
  assert(int(iter)<=int(tot_num_images/calib_batch_size)),"number of iterations must be <=20"
  images = []
  #print(line)
  for index in range(0, calib_batch_size):

      curline = line[(iter-1) * calib_batch_size + index]
      #print(curline)
      calib_image_name = curline.strip()

      # read image as grayscale, returns numpy array (28,28)
      #image = cv2.imread(calib_image_dir + calib_image_name, cv2.IMREAD_GRAYSCALE)

      # read image as rgb, returns numpy array (28,28, 3)
      filename = os.path.join(calib_image_dir, calib_image_name)
      image = cv2.imread(filename)

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
