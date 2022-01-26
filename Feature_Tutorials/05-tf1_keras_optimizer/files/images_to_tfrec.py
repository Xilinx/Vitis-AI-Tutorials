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
This script will convert the images from the dogs-vs-cats dataset into TFRecords.

Each TFRecord contains 5 fields:

- label
- height
- width
- channels
- image - JPEG encoded

The dataset must be downloaded from https://www.kaggle.com/c/dogs-vs-cats/data
 - this will require a Kaggle account.
The downloaded 'dogs-vs-cats.zip' archive should be placed in the same folder 
as this script, then this script should be run.
'''

'''
Author: Mark Harvey
'''


import os
import argparse
import zipfile
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

DIVIDER = '-----------------------------------------'



def _bytes_feature(value):
  '''Returns a bytes_list from a string / byte'''
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  '''Returns a float_list from a float / double'''
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  ''' Returns an int64_list from a bool / enum / int / uint '''
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _calc_num_shards(img_list, img_shard):
  ''' calculate number of shards'''
  last_shard =  len(img_list) % img_shard
  if last_shard != 0:
    num_shards =  (len(img_list) // img_shard) + 1
  else:
    num_shards =  (len(img_list) // img_shard)
  return last_shard, num_shards


def resize_maintain_aspect(image,target_h,target_w):
  '''
  Resize image but maintain aspect ratio
  '''
  image_height, image_width = image.shape[:2]
  if image_height > image_width:
    new_height = target_h
    new_width = int(image_width*(target_h/image_height))
  else:
    new_width = target_w
    new_height = int(image_height*(target_w/image_width))

  image = cv2.resize(image,(new_width,new_height),interpolation=cv2.INTER_CUBIC)
  return image



def pad_image(image,target_h,target_w):

  # make a black canvas
  color = (0,0,0)
  canvas = np.full((target_h,target_w, 3), color, dtype=np.uint8)

  # center resized image onto canvas
  x_start = (target_w - image.shape[1]) // 2
  y_start = (target_h - image.shape[0]) // 2

  # copy img image into center of canvas
  canvas[y_start:y_start + image.shape[0], x_start:x_start + image.shape[1] ] = image
  return canvas


def write_tfrec(tfrec_filename,img_list,target_h,target_w):
  ''' write TFRecord file '''

  with tf.io.TFRecordWriter(tfrec_filename) as writer:
   
    for img in img_list:
      filename = os.path.basename(img)
      class_name,_ = filename.split('.',1)
      if class_name == 'dog':
        label = 0
      else:
        label = 1

      # read the JPEG source file to numpy array
      image = cv2.imread(img)

      # resize
      image = resize_maintain_aspect(image,target_h,target_w)

      # pad
      image = pad_image(image,target_h,target_w)

      # convert to RGB from BGR
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # re-encode to JPEG
      _,im_buf_arr = cv2.imencode('.jpg', image)

      # features dictionary
      feature_dict = {
        'label' : _int64_feature(label),
        'height': _int64_feature(image.shape[0]),
        'width' : _int64_feature(image.shape[1]),
        'chans' : _int64_feature(image.shape[2]),
        'image' : _bytes_feature(im_buf_arr.tostring())
      }

      # Create Features object
      features = tf.train.Features(feature = feature_dict)

      # create Example object
      tf_example = tf.train.Example(features=features)

      # serialize Example object into TFRecord file
      writer.write(tf_example.SerializeToString())

  return






def img_to_npz(data_dir,listImages,image_height,image_width,output_file,save_image):
  '''
  Converts a list of images to a single compressed numpy file.
  Images are resized and padded to image_height x image_width, then 
  are normalized so that all pixels are floating-point numbers.
  Labels are derived from the image filenames and packed into the numpy file.
  '''

  if save_image == True:
    os.makedirs(os.path.join(data_dir,'test_images'), exist_ok = True)

  # make data array for images
  x = np.ndarray(shape=(len(listImages),image_height,image_width,3), dtype=np.float32, order='C')

  # make labels array for labels
  y = np.ndarray(shape=(len(listImages)), dtype=np.uint8, order='C')

  for i in tqdm(range(len(listImages))):

    # open image to numpy array
    img = cv2.imread(listImages[i])

    # resize
    img = resize_maintain_aspect(img,image_height,image_width)

    # pad
    canvas = pad_image(img,image_height,image_width)

    filename = os.path.basename(listImages[i])

    if save_image == True:
      cv2.imwrite(os.path.join(data_dir,'test_images',filename), canvas)

    # switch to RGB from BGR
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # normalize then write into data array
    x[i] = ((canvas/127.5)-1).astype(np.float32)

    # make labels and write into labels array
    class_name,_ = filename.split('.',1)
    if class_name == 'dog':
      label = 0
    else:
      label = 1

    y[i] = int(label)

 
  print(' Saving numpy file, this may take some time...',flush=True)
  np.savez_compressed(output_file, x=x, y=y)
  print(' Saved to',output_file+'.npz',flush=True)

  return



def make_tfrec(data_dir,tfrec_dir,input_height,input_width,img_shard):

  dataset_dir = os.path.join(data_dir,'dataset')

  print('TFRecord files will be written to',tfrec_dir)

  # remove any previous data
  shutil.rmtree(dataset_dir, ignore_errors=True)    
  shutil.rmtree(tfrec_dir, ignore_errors=True)
  os.makedirs(dataset_dir)   
  os.makedirs(tfrec_dir)

  # unzip the dogs-vs-cats archive that was downloaded from Kaggle
  zip_ref = zipfile.ZipFile('dogs-vs-cats.zip', 'r')
  zip_ref.extractall(dataset_dir)
  zip_ref.close()

  # unzip train archive (inside the dogs-vs-cats archive)
  zip_ref = zipfile.ZipFile(os.path.join(dataset_dir, 'train.zip'), 'r')
  zip_ref.extractall(dataset_dir)
  zip_ref.close()
  print('Unzipped dataset.')
  
  # remove un-needed files
  os.remove(os.path.join(dataset_dir, 'sampleSubmission.csv'))
  os.remove(os.path.join(dataset_dir, 'test1.zip'))
  os.remove(os.path.join(dataset_dir, 'train.zip'))  
  
  # make a list of all images
  imageList = []
  for (root, name, files) in os.walk(os.path.join(dataset_dir, 'train')):
      imageList += [os.path.join(root, file) for file in files]
 
  # make lists of images according to their class
  catImages=[]
  dogImages=[]
  for img in imageList:
    filename = os.path.basename(img)
    class_name,_ = filename.split('.',1)
    if class_name == 'cat':
        catImages.append(img)
    else:
        dogImages.append(img)
  assert(len(catImages)==len(dogImages)), 'Number of images in each class do not match'

  # define train/test split as 80:20
  split = int(len(dogImages) * 0.2)

  testImages = dogImages[:split] + catImages[:split]
  trainImages = dogImages[split:] + catImages[split:]
  random.shuffle(testImages)
  random.shuffle(trainImages)
  print(len(trainImages),'training images and',len(testImages),'test images.')


  ''' Test TFRecords '''
  print('Creating test TFRecord files...')

  # how many TFRecord files?
  last_shard, num_shards = _calc_num_shards(testImages, img_shard)
  print (num_shards,'TFRecord files will be created.')

  if (last_shard>0):
    print ('Last TFRecord file will have',last_shard,'images.')
  
  # create TFRecord files (shards)
  start = 0
  for i in tqdm(range(num_shards)):    
    tfrec_filename = 'test_'+str(i)+'.tfrecord'
    write_path = os.path.join(tfrec_dir, tfrec_filename)
    if (i == num_shards-1):
      write_tfrec(write_path, testImages[start:],input_height,input_width)
    else:
      end = start + img_shard
      write_tfrec(write_path, testImages[start:end],input_height,input_width)
      start = end

  ''' Make test images and labels into numpy file'''
  img_to_npz(data_dir,testImages,input_height,input_width, os.path.join(data_dir,'testData'),  True)

  ''' Training TFRecords '''
  print('Creating training TFRecord files...')

  # how many TFRecord files?
  last_shard, num_shards = _calc_num_shards(trainImages, img_shard)
  print (num_shards,'TFRecord files will be created.')
  if (last_shard>0):
    print ('Last TFRecord file will have',last_shard,'images.')
  
  # create TFRecord files (shards)
  start = 0
  for i in tqdm(range(num_shards)):    
    tfrec_filename = 'train_'+str(i)+'.tfrecord'
    write_path = os.path.join(tfrec_dir, tfrec_filename)
    if (i == num_shards-1):
      write_tfrec(write_path, trainImages[start:],input_height,input_width)
    else:
      end = start + img_shard
      write_tfrec(write_path, trainImages[start:end],input_height,input_width)
      start = end

  
  # delete temp images to save disk space
  shutil.rmtree(os.path.join(dataset_dir))  

  print('\nDATASET PREPARATION COMPLETED')
  print(DIVIDER,'\n')

  return
    


def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--data_dir',    type=str, default='data', help='Data folder path. Default is data.')
  ap.add_argument('-tf','--tfrec_dir',   type=str, default='data/tfrecords', help='TFRecords folder path. Default is data/tfrecords.')
  ap.add_argument('-ih','--input_height',type=int, default=224,  help='Input image height in pixels.')
  ap.add_argument('-iw','--input_width', type=int, default=224,  help='Input image width in pixels.')
  ap.add_argument('-s', '--img_shard',   type=int, default=2000, help='Number of images per shard. Default is 2000') 
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('Command line options:')
  print (' --data_dir     : ',args.data_dir)
  print (' --tfrec_dir    : ',args.tfrec_dir)
  print (' --input_height : ',args.input_height)
  print (' --input_width  : ',args.input_width)
  print (' --img_shard    : ',args.img_shard)
  print(DIVIDER)

  make_tfrec(args.data_dir,args.tfrec_dir,args.input_height,args.input_width,args.img_shard)

if __name__ == '__main__':
    run_main()
