# Copyright 2022 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
from typing import List
import numpy as np
import vart
import os
import pathlib
import xir
import sys
import argparse
import re
import vitis_ai_library


divider = "------------------------------"

def preprocess_lambda_custom_op(input_tensor_buffers,cross_check_input):
  print("preprocess begin.")

  if cross_check_input is not None:
    #use the binary file from the host cpu dump process to cross check
    img = np.fromfile(cross_check_input, dtype='uint8')
    img = img.reshape(512,512,6)
  else:
      #generate random input data:
      my_generator = np.random.default_rng(seed=1)
      img = my_generator.integers(0, high=255, size=(512,512,6))/255.0
      fix_pos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")
      scale = 2**fix_pos
      img = (img*scale).astype(np.int8)
  input_data = np.asarray(input_tensor_buffers[0])
  input_data[0] = img

  print("preprocess end.")

def postprocess_lambda_custom_op(output_tensor_buffers):
  print("postprocess begin.")
  output_data = np.asarray(output_tensor_buffers[0])

  batchsize = output_tensor_buffers[0].get_tensor().dims[0]
  size = int(output_tensor_buffers[0].get_tensor().get_element_num()/batchsize)
  filename = output_tensor_buffers[0].get_tensor().name + ".bin"

  with open(filename, "wb") as f:
    for data in output_data[0]:
      f.write(data)
  print("data dump out to file ", filename)

  print("postprocess end.")

def app(model,cross_check_input):
  g = xir.Graph.deserialize(model)
  runner = vitis_ai_library.GraphRunner.create_graph_runner(g)

  input_tensor_buffers = runner.get_inputs()
  output_tensor_buffers = runner.get_outputs()

  preprocess_lambda_custom_op(input_tensor_buffers,cross_check_input)

  v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
  runner.wait(v)

  postprocess_lambda_custom_op(output_tensor_buffers)

  del runner

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('-m', '--model', type=str, default='./compile_model/customcnn.xmodel', help='Path of xmodel. Default is autoenc.xmodel')
  ap.add_argument('-c', '--cross_check_input', type=str, default=None, help='Path to the quant_input_1.bin generated on the host machine, if unspecified, will generate random input data')

  args = ap.parse_args()

  print(divider)
  print ('Command line options:')
  print (' --model     : ', args.model)
  print (' --image_dir : ', args.cross_check_input)
  print(divider)

  app(args.model, args.cross_check_input)

if __name__ == '__main__':
  main()
