#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

import cv2

# sub mean
#_R_MEAN =     0 #resnet18
#_G_MEAN =     0 #resnet18
#_B_MEAN =     0 #resnet18
_R_MEAN = 123.68 #resnet50
_G_MEAN = 116.78 #resnet50
_B_MEAN = 103.94 #resnet50

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]

def resize_shortest_edge(image, size):
  H, W = image.shape[:2]
  if H >= W:
    nW = size
    nH = int(float(H)/W * size)
  else:
    nH = size
    nW = int(float(W)/H * size)
  return cv2.resize(image,(nW,nH))

def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image

def BGR2RGB(image):
  B, G, R = cv2.split(image)
  image = cv2.merge([R, G, B])
  return image

def central_crop(image, crop_height, crop_width):
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2
  return image[offset_height:offset_height + crop_height, offset_width:
               offset_width + crop_width, :]

def normalize(image):
  image=image/255.0
  image=image-0.5
  image=image*2
  return image


def crop_and_resize(img, w, h):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h

    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv2.resize(img, (im_w_r , im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv2.resize(img, (im_w_r , im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv2.resize(img, (w, h))
    return img



def preprocess_fn(image_path, crop_height = 224, crop_width = 224):
    if "JPEG" in image_path:
      print(image_path)
      bgr_image = cv2.imread(image_path)
      #image = BGR2RGB(bgr_image)
      #image = central_crop(image, crop_height, crop_width)
      #image = resize_shortest_edge(image, crop_height)
      bgr_image = crop_and_resize(bgr_image,crop_width,  crop_height)
      #image = normalize(image)
      rgb_image = mean_image_subtraction(bgr_image, MEANS)
      return rgb_image
