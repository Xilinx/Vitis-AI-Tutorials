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
Author: Mark Harvey
'''

import os
import sys
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from mobilenetv2 import mobilenetv2


DIVIDER = '-----------------------------------------'

def export(build_dir,output_file,output_node,input_height,input_width,input_chan):
  
  tf.keras.backend.set_learning_phase(0)
  model = mobilenetv2(input_shape=(input_height,input_width,input_chan),classes=2,alpha=1.0,incl_softmax=False)
  model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
  
  graph_def = tf.compat.v1.keras.backend.get_session().graph.as_graph_def() 
  graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def, [output_node])

  tf.io.write_graph(graph_def,build_dir,output_file,as_text=True)

  return


def run_main():
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd','--build_dir',    type=str, default='build_pr', help='Build folder path. Default is build_pr.')
    ap.add_argument('-o', '--output_file',  type=str, default='inference_graph.pbtxt', help='Name of graph file. Default is inference_graph.pbtxt')
    ap.add_argument('-on','--output_node',  type=str, default='dense/BiasAdd',         help='Name of output node. Default is dense/BiasAdd')
    ap.add_argument('-ih','--input_height', type=int, default=224,   help='Input image height in pixels.')
    ap.add_argument('-iw','--input_width',  type=int, default=224,   help='Input image width in pixels.')
    ap.add_argument('-ic','--input_chan',   type=int, default=3,     help='Input image channels.')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('Keras version      : ',tf.keras.__version__)
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--output_file  : ',args.output_file)
    print ('--output_node  : ',args.output_node)
    print ('--input_height : ',args.input_height)
    print ('--input_width  : ',args.input_width)
    print ('--input_chan   : ',args.input_chan)
    print(DIVIDER)
 
    export(args.build_dir,args.output_file,args.output_node,args.input_height,args.input_width,args.input_chan)

if __name__ == '__main__':
    run_main()

