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
Application code for running variational autoencoder
'''


'''
Author: Mark Harvey, Xilinx Inc
'''


from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import re


divider = '------------------------------------'

def preprocess_fn(image_path):
    '''
    Image pre-processing.
    Opens image as grayscale then normalizes to range 0:1
      input arg: path of image file
      return: numpy array
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(image, [image.shape[0],image.shape[1],1] )
    image = (image/255.0).astype(np.float32)
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    '''
    Create a list of DPU subgraphs
    '''
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]



def sampling_layer(encoder_mu, encoder_log_variance):
  '''
  Sampling layer
  '''
  batch = encoder_mu.shape[0]
  dim = encoder_mu.shape[1]
  epsilon = np.random.normal(size=(batch, dim))
  sample = encoder_mu + (np.exp(0.5 * encoder_log_variance) * epsilon)
  return sample


def sigmoid(x):
  '''
  calculate sigmoid
  '''
  pos = x >= 0
  neg = np.invert(pos)
  result = np.empty_like(x)
  result[pos] = 1 / (1 + np.exp(-x[pos]))
  result[neg] = np.exp(x[neg]) / (np.exp(x[neg]) + 1)
  return result


def execute_async(dpu, tensor_buffers_dict):
  '''
  launch dpu runner and wait for execution to finish
  '''
  input_tensor_buffers = [ tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()  ]
  output_tensor_buffers = [ tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()  ]
  jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
  return dpu.wait(jid)


def init_dpu_runner(dpu_runner):
  '''
  Setup DPU runner in/out buffers and dictionary
  '''

  io_dict = {}
  inbuffer = []
  outbuffer = []

  # create input buffer, one member for each DPU runner input
  # add inputs to dictionary
  dpu_inputs = dpu_runner.get_input_tensors()
  i=0
  for dpu_input in dpu_inputs:
    #print('DPU runner input:',dpu_input.name,' Shape:',dpu_input.dims)
    inbuffer.append(np.empty(dpu_input.dims, dtype=np.float32, order="C"))
    io_dict[dpu_input.name] = inbuffer[i]
    i += 1

  # create output buffer, one member for each DPU runner output
  # add outputs to dictionary
  dpu_outputs = dpu_runner.get_output_tensors()
  i=0
  for dpu_output in dpu_outputs:
    #print('DPU runner output:',dpu_output.name,' Shape:',dpu_output.dims)
    outbuffer.append(np.empty(dpu_output.dims, dtype=np.float32, order="C"))
    io_dict[dpu_output.name] = outbuffer[i]
    i += 1

  return io_dict, inbuffer, outbuffer


def runThread(id, start, encoder_dpu_runner, decoder_dpu_runner, img):
  '''
  Thread worker function
  '''

  #  Set up encoder DPU runner buffers & I/O mapping dictionary
  encoder_dict, encoder_inbuffer, encoder_outbuffer = init_dpu_runner(encoder_dpu_runner)

  # batchsize
  batchSize = encoder_dict['quant_input_3'].shape[0]

  # Set up decoder DPU runner buffers
  decoder_dict, decoder_inbuffer, decoder_outbuffer = init_dpu_runner(decoder_dpu_runner)


  # set runSize
  n_of_images = len(img)
  count = 0
  write_index = start

  # loop over image list
  while count < n_of_images:
    if (count+batchSize<=n_of_images):
        runSize = batchSize
    else:
        runSize=n_of_images-count

    '''
    initialise encoder input and execute DPU runner
    '''
    # init input image to input buffer
    for j in range(runSize):
      imageRun = encoder_dict['quant_input_3']
      imageRun[j, ...] = img[(count + j) % n_of_images].reshape(tuple(encoder_dict['quant_input_3'].shape[1:]))
  
    execute_async(encoder_dpu_runner, encoder_dict)

    '''
    run sampling layer
    quant_dense_1_fix = encoder_log_variance
    quant_dense_fix = encoder_mu
    '''
    sample_z = sampling_layer(encoder_dict['quant_dense_fix'], encoder_dict['quant_dense_1_fix'])

    '''
    initialise decoder input and execute DPU runner
    '''
    # init sample_z to input buffer
    for j in range(runSize):
      sampleRun = decoder_dict['quant_sampling_reshaped_inserted_fix_7']
      sampleRun[j, ...] = sample_z[j].reshape(tuple(decoder_dict['quant_sampling_reshaped_inserted_fix_7'].shape[1:]))

    execute_async(decoder_dpu_runner, decoder_dict)


    # write results to global predictions buffer
    for j in range(runSize):
      predictions_buffer[write_index] = sigmoid(decoder_dict['quant_conv2d_transpose_3_fix'][j])
      write_index += 1
    count = count + runSize



def app(image_dir, pred_dir, threads, model):  
  '''
  MAIN APPLICATION FUNCTION
  '''

  '''
  image pre-processing
  '''
  # make a list of all images
  listimage=os.listdir(image_dir)
  listimage.sort(key=lambda f: int(re.sub('\D', '', f)))
  runTotal = len(listimage)

  # preprocess images
  print('Pre-processing',runTotal,'images...')
  img = []
  for i in range(runTotal):
      path = os.path.join(image_dir,listimage[i])
      img.append(preprocess_fn(path))
  print(divider)

  '''
  set up global buffer for all threads to write to
  '''
  global predictions_buffer
  predictions_buffer = [None] * runTotal


  '''
  set up DPU runners
  subgraphs[0] = encoder, subgraphs[1] = decoder
  '''
  # get a list of all DPU subgraphs
  g = xir.Graph.deserialize(model)
  subgraphs = get_child_subgraph_dpu(g)
  print('Found',len(subgraphs),'DPU subgraphs')

  
  all_dpu_runners = []
  for i in range(threads):
    all_dpu_runners.append( [vart.Runner.create_runner(subgraphs[0], "run"),
                             vart.Runner.create_runner(subgraphs[1], "run")]  )
  

  '''
  run threads
  '''
  print('Starting',threads,'threads...')
  threadAll = []
  start=0
  for i in range(threads):
      if (i==threads-1):
          end = len(img)
      else:
          end = start+(len(img)//threads)
      in_q = img[start:end]
      t1 = threading.Thread(target=runThread, args=(i,start,all_dpu_runners[i][0],all_dpu_runners[i][1], in_q))
      threadAll.append(t1)
      start=end

  time1 = time.time()
  for x in threadAll:
      x.start()
  for x in threadAll:
      x.join()
  time2 = time.time()
  timetotal = time2 - time1

  fps = float(runTotal / timetotal)
  print (divider)
  print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


  '''
  post-processing - save output images
  '''
  # make folder for saving predictions
  os.makedirs(pred_dir, exist_ok=True)
  
  for i in range(len(predictions_buffer)):
    cv2.imwrite(os.path.join(pred_dir,'pred_'+str(i)+'.png'), predictions_buffer[i]*255.0)

  print('Predicted images saved to','./'+pred_dir)

  return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images',         help='Path to folder of images. Default is images')  
  ap.add_argument('-pd','--pred_dir',  type=str, default='predictions',    help='Path to folder of predictions. Default is predictions')  
  ap.add_argument('-t', '--threads',   type=int, default=1,                help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='autoenc.xmodel', help='Path of xmodel. Default is autoenc.xmodel')

  args = ap.parse_args()

  print(divider)  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --pred_dir  : ', args.pred_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print(divider) 

  app(args.image_dir, args.pred_dir, args.threads, args.model)

if __name__ == '__main__':
  main()
