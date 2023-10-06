#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Mark Harvey
# Modified by: Daniele Bagni, AMD/Xilinx
# date:  01 Aug 2023


'''
Make the target folder
Creates images, copies application code and compiled xmodel in a single folder
'''



import argparse
import os
import shutil
import sys
from tqdm import tqdm


from config import config as cfg

# indicate which GPU to use
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu_list

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


import tensorflow as tf

from dataset_utils import input_fn_image

DIVIDER=cfg.DIVIDER



def make_target(build_dir,target):

  # configuration
  tfrec_dir=cfg.tfrec_dir
  input_dir=build_dir+cfg.compile_dir+target
  output_dir = build_dir+cfg.target_dir+target
  model_name=cfg.model_name
  model_path = input_dir+'/'+model_name+'.xmodel'
  app_dir=cfg.app_dir

  # remove any previous data
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(output_dir)

  # make the dataset
  target_dataset = input_fn_image(tfrec_dir,1)

  '''
  # extract images & labels from TFRecords
  # save as JPEG image files
  # the label will be built into the JPEG filename
  '''
  i = 0
  for tfr in tqdm(target_dataset):

    label = tfr[1][0].numpy()

    # reshape image to remove batch dimension
    img = tf.reshape(tfr[0], [tfr[0].shape[1],tfr[0].shape[2],tfr[0].shape[3]] )

    # JPEG encode
    img = tf.cast(img, tf.uint8)
    img = tf.io.encode_jpeg(img)

    # save as file
    filepath =  os.path.join(output_dir,'images',str(label)+'_image'+str(i)+'.jpg')
    tf.io.write_file(filepath, img)

    i += 1


  # copy application code
  print('Copying application python code from',app_dir,'...')
  shutil.copy(os.path.join(app_dir, 'app_mt.py'), output_dir)

  # copy script
  print('Copying application shell script from',app_dir,'...')
  shutil.copy(os.path.join(app_dir, 'run_all_mobilenetv2_target.sh'), output_dir)


  # copy compiled model
  print('Copying compiled xmodel from',model_path,'...')
  shutil.copy(model_path, output_dir)

  return



def main():

   # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir', type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-t' , '--target',    type=str, default='zcu102',help='Target platform. Default is zcu102')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('Keras version      : ',tf.keras.__version__)
  print('TensorFlow version : ',tf.__version__)
  print(sys.version)
  print(DIVIDER)


  make_target(args.build_dir,args.target)

  return


if __name__ ==  "__main__":
    main()
