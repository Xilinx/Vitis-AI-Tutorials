#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
'''

import os, random, shutil

def make_dir(source, target):
    dir_names = os.listdir(source)
    print("dir names: ", dir_names)
    for names in dir_names:
        for i in ['train','test']:
            path = target + '/' + i + '/' + names
            print("target ", target)
            print("names ",  names)
            print("path ",  path)
            if not os.path.exists(path):
                os.makedirs(path)

def divideTrainValiTest(source, target):
    pic_name = os.listdir(source)
    print("pic_name ", pic_name)
    for classes in pic_name:
        pic_classes_name = os.listdir(os.path.join(source, classes))
        print("pic_classes_name ", pic_classes_name)
        random.shuffle(pic_classes_name)
        train_list = pic_classes_name[0 : int(0.8*len(pic_classes_name))]
        test_list  = pic_classes_name[int(0.8*len(pic_classes_name)) :  ]
        for train_pic in train_list:
            shutil.copyfile(source + '/' + classes + '/' + train_pic, target + '/train/' + classes + '/' + train_pic)
        for test_pic in test_list:
            shutil.copyfile(source + '/' + classes + '/' + test_pic, target + '/test/' + classes + '/' + test_pic)

#if __name__ == '__main__':
filepath = r'./data/cropped_vcor/test'
dist = r'./data/cropped_vcor_split'
make_dir(filepath, dist)
divideTrainValiTest(filepath, dist)
