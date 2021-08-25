#!/usr/bin/env python
# © Copyright (C) 2016-2017 Xilinx, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import caffe
import cv2
from PIL import Image

def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        }
    
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--input_image', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images should be stored')
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    caffe.set_mode_cpu()
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv_encoder6_0_0'].data.shape
    input_image = cv2.imread(args.input_image, 1).astype(np.float32)
    input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    b,g,r = cv2.split(input_image)
    h = input_image.shape[0]
    w = input_image.shape[1]
    for y in range (0, h):
        for x in range (0, w):
            r[y,x] = r[y,x] * 0.022 - 0.287
            g[y,x] = g[y,x] * 0.022 - 0.325
            b[y,x] = b[y,x] * 0.022 - 0.284

    input_image=cv2.merge((b,g,r))   
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])
    out = net.forward_all(**{net.inputs[0]: input_image})
    prediction = net.blobs['deconv_encoder6_0_0'].data[0].argmax(axis=0)
    prediction_rgb = label_img_to_color(prediction)
    
    if args.out_dir is not None:
        input_path_ext = args.input_image.split(".")[-1]
        input_image_name = args.input_image.split("/")[-1:][0].replace('.' + input_path_ext, '')
        out_path_im = args.out_dir + input_image_name + '_enet_encoder' + '.' + input_path_ext
        out_path_gt = args.out_dir + input_image_name + '_enet_gt' + '.' + input_path_ext
        cv2.imwrite(out_path_im, prediction_rgb)






