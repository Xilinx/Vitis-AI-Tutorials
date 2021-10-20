'''
 Copyright 2021 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Make the target folder
Creates images, copies application code and compiled xmodel to 'target'
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys
import cv2
from tqdm import tqdm

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

from utils import mnist_download

DIVIDER = '-----------------------------------------'




def make_target(target_dir,image_format,num_images,app_dir,model):

  # remove any previous data
  shutil.rmtree(target_dir, ignore_errors=True)    
  os.makedirs(target_dir)
  os.makedirs(target_dir+'/images')


  # download MNIST test dataset
  _, _, _, x_test_noisy = mnist_download()


  # Convert numpy arrays of training dataset into image files.
  for i in range(len(x_test_noisy[:num_images])):
    img_file=os.path.join(target_dir,'images','input_'+str(i)+'.'+image_format)
    cv2.imwrite(img_file, x_test_noisy[i]*255.0)

  # copy application code
  print('Copying application code from',app_dir,'...')
  shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

  # copy compiled model
  print('Copying compiled model from',model,'...')
  shutil.copy(model, target_dir)


  return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-td','--target_dir',   type=str, default='build/target', help='Full path of target folder. Default is build/target')
    ap.add_argument('-f', '--image_format', type=str, default='png', choices=['png','jpg','bmp'], help='Image file format - valid choices are png, jpg, bmp. Default is png')  
    ap.add_argument('-n', '--num_images',   type=int, default=2000, help='Number of images to create. Default is 2000')
    ap.add_argument('-a', '--app_dir',      type=str, default='application', help='Full path of application code folder. Default is application')
    ap.add_argument('-m', '--model',        type=str, default='build/compiled_model/autoenc.xmodel', help='Full path of compiled model.Default is build/compiled_model/autoenc.xmodel')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --target_dir   : ', args.target_dir)
    print (' --image_format : ', args.image_format)
    print (' --num_images   : ', args.num_images)
    print (' --app_dir      : ', args.app_dir)
    print (' --model        : ', args.model)
    print('------------------------------------\n')


    make_target(args.target_dir,args.image_format,args.num_images,args.app_dir,args.model)


if __name__ ==  "__main__":
    main()
  
