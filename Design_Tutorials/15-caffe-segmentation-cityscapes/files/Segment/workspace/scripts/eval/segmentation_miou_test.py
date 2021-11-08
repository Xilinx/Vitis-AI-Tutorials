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
import argparse
import numpy as np
from PIL import Image
import os
import sys
import cv2
import time
import caffe

IMG_MEAN = np.array((73,82,72))  # mean_values for B, G,R
SCALE = 0.022
INPUT_W, INPUT_H = 1024, 512 # W, H 
TARGET_W, TARGET_H = 2048, 1024
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Segmentation Network")
    parser.add_argument("--imgpath", type=str, default='/workspace/Segment/Cityscapes/leftImg8bit/val',
                        help="Path to the directory containing the cityscapes validation dataset.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict.")
    parser.add_argument("--modelpath", type=str, default='./',
                        help="Path to the directory containing the deploy.prototxt and caffemodel.")
    parser.add_argument("--mode", type=str, default='float',
                        help="use float or fix version.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--mean-values", type=str, default=IMG_MEAN,
                        help="Comma-separated string with BGR mean values.")
    parser.add_argument("--scale-value", type=str, default=SCALE,
                        help="Comma-separated string with scale value.") 
    parser.add_argument("--savepath", type=str, default='./results',
                        help="where to save the vis results.")
    parser.add_argument("--colorFormat", type=bool, default=False,
                        help="add corlors on results.")
    return parser.parse_args()

    args = get_arguments()

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

def segment(net, img_file, name):
    im_ = cv2.imread(img_file)
    w, h = TARGET_W, TARGET_H
    in_ = cv2.resize(im_, (INPUT_W, INPUT_H))
    in_ = in_ * 1.0
    in_ -= IMG_MEAN
    in_ = in_ * SCALE
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    #caffe
    net.forward()
    t_s = time.time()
    out = net.blobs['score'].data[0].argmax(axis=0)
    t = time.time() - t_s
    print('it took {:.3f}s'.format(t))
    #save color_output
    gray_to_save = cv2.resize(out, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    pred_label_color = label_img_to_color(out)
    color_to_save = Image.fromarray(pred_label_color.astype(np.uint8))
    color_to_save = color_to_save.resize((w,h))
    return gray_to_save, color_to_save
    
def main():  
    args = get_arguments()
    img_path = args.imgpath
    save_path = args.savepath
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = 0
    caffe.set_device(int(args.gpu))
    if args.mode=='float':
        net = caffe.Net(os.path.join(args.modelpath, 'deploy.prototxt'),os.path.join(args.modelpath, 'deploy.caffemodel'),caffe.TEST) 
    elif args.mode=='fix':
        net = caffe.Net(os.path.join(args.modelpath, 'Fix/fix_train_test.prototxt'),os.path.join(args.modelpath, 'Fix/fix_train_test.caffemodel'),caffe.TEST)
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            image_name = it 
            count += 1
            image_file = os.path.join(img_path, c, it + '_leftImg8bit.png')
   
            print(str(count) + ', ' + image_name)
            assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
            gray_to_save, color_to_save = segment(net, image_file, image_name)
            cv2.imwrite(os.path.join(save_path, image_name + '_leftImg8bit.png'), gray_to_save)
            if args.colorFormat:
                color_to_save.save(os.path.join(save_path, image_name + '_leftImg8bit_with_color.png'))
        
    print('Finished!!')


if __name__ == '__main__':
    main()
