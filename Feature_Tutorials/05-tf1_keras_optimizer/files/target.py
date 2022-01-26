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
Creates images, copies application code and compiled xmodel in a single folder
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



DIVIDER = '-----------------------------------------'



def make_target(build_dir,data_dir,target,app_dir,model_name):

    target_dir = os.path.join(build_dir, 'target_' + target)
    test_images_dir = os.path.join(data_dir,'test_images')
    model = os.path.join(build_dir, 'compiled_model_' + target,model_name+'.xmodel')

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)

    # copy images with preprocessing
    print('Copying test images from',test_images_dir,'...')
    shutil.copytree(test_images_dir, target_dir+'/images')

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
    ap.add_argument('-bd','--build_dir',   type=str,  default='build',  help='Build folder path. Default is build.')
    ap.add_argument('-d', '--data_dir',    type=str,  default='data',   help='path to folder containing test images')
    ap.add_argument('-t', '--target',      type=str,  default='zcu102', help='Name of target device or board. Default is zcu102.')
    ap.add_argument('-a', '--app_dir',     type=str,  default='application', help='Full path of application code folder. Default is application')
    ap.add_argument('-m', '--model_name',  type=str,  default='mobilenetv2', help='Model name. Default is mobilenetv2')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir    : ', args.build_dir)
    print (' --data_dir     : ', args.data_dir)
    print (' --target       : ', args.target)
    print (' --app_dir      : ', args.app_dir)
    print (' --model_name   : ', args.model_name)
    print('------------------------------------\n')


    make_target(args.build_dir,args.data_dir,args.target,args.app_dir,args.model_name)


if __name__ ==  "__main__":
    main()
  