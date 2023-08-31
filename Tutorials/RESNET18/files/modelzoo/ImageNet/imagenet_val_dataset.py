#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  09 Aug 2023

import zipfile
import glob
import os

NUM_VAL_IMAGES=500
filenames_list = []
f1 = open("./val.txt", "r");
f1_lines = f1.readlines();
for k in range(NUM_VAL_IMAGES):
    line = f1_lines[k]
    (filename, val) = line.strip().split(" ") #line.rstrip().split(" ")
    filename = os.path.basename(filename)
    filenames_list.append(filename)
#print(filenames_list)

dst_dir="./val_dataset"
N = 0
with zipfile.ZipFile('val_dataset.zip','w') as f2:
    for filename in filenames_list:
        path_filename=os.path.join(dst_dir, filename)
        f2.write(path_filename)
        N = N+1
    f2.write("./val.txt")
    f2.write("./words.txt")
f2.close()
print("val_dataset.zip archive is ready with ", N, "images")
