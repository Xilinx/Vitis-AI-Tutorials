#!/bin/bash

# Copyright 2020 Xilinx Inc.
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



# conda environment
conda activate vitis-ai-tensorflow


# folders
export BUILD=./build
export TARGET=./target
export LOG=${BUILD}/logs
export TB_LOG=${BUILD}/tb_logs
export CHKPT_DIR=${BUILD}/chkpts
export FREEZE=${BUILD}/freeze
export COMPILE=${BUILD}/compile
export QUANT=${BUILD}/quantize

# checkpoints & graphs filenames
export CHKPT_FILENAME=float_model.ckpt
export INFER_GRAPH_FILENAME=inference_graph.pb
export FROZEN_GRAPH=frozen_graph.pb

# logs & results files
export TRAIN_LOG=train.log
export FREEZE_LOG=freeze.log
export EVAL_FR_LOG=eval_frozen_graph.log
export QUANT_LOG=quant.log
export EVAL_Q_LOG=eval_quant_graph.log
export COMP_LOG=compile.log

# training parameters
export EPOCHS=50
export LEARNRATE=0.0001
export BATCHSIZE=100

# network parameters
export INPUT_HEIGHT=28
export INPUT_WIDTH=28
export INPUT_CHAN=1
export INPUT_SHAPE=?,${INPUT_HEIGHT},${INPUT_WIDTH},${INPUT_CHAN}
export INPUT_NODE=images_in
export OUTPUT_NODE=dense_1/BiasAdd
export NET_NAME=customcnn

# DPU mode - best performance with DPU_MODE = normal
export DPU_MODE=normal
#export DPU_MODE=debug


# target board
export BOARD=ZCU102
export ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${BOARD}/${BOARD}.json

# number of images used in calibration
export CALIB_IMAGES=1000

# number of images copied to SD card
export SDCARD_IMAGES=5000
