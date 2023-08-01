#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


MODEL=$1 #./build/freeze/fcn8/frozen_graph.pb

PYSCRIPT=/workspace/tools/Vitis-AI-Quantizer/vai_q_tensorflow1.x/tensorflow/python/tools/import_pb_to_tensorboard.py

python $PYSCRIPT --model_dir=$MODEL --log_dir=./tb_log/

tensorboard --logdir=./tb_log/
#TensorBoard 1.12.2 at http://Prec5820Tow:6006 (Press CTRL+C to quit)
