#!/bin/sh

# Copyright 2021 Xilinx Inc.
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

# Author: Mark Harvey, Xilinx Inc


# This shell script sets the environment for running the complete flow WITH pruning.



# folders

export BUILD=./build_pr
export LOG=${BUILD}/logs

export DATA=./data
export TFREC=${DATA}/tfrecords

export TRAIN_DIR=trained_model

export TF_BOARD=${TRAIN_DIR}/tb_logs
export TF_DIR=${BUILD}/tf_ckpt
export FROZEN_DIR=${BUILD}/frozen_model
export QUANT_DIR=${BUILD}/quant_model
export APP_DIR=application

export PRUNE_ANA=${BUILD}/prune_ana
export PRUNE_DIR=${BUILD}/prune
export FT_DIR=${BUILD}/ft
export TRSF_DIR=${BUILD}/transform


# checkpoints
export FLOAT_MODEL=${TRAIN_DIR}/f_model.h5
export TF_CKPT=tf_float.ckpt
export TF_META=${TF_CKPT}.meta
export PR_CKPT=pruned.ckpt
export FT_CKPT=fine_tuned.ckpt
export TRSF_CKPT=transform.ckpt
export FROZEN_MODEL=frozen_graph.pb



# AI Optimizer license
export XILINXD_LICENSE_FILE=./AI_OPTIMIZER.lic


# make the necessary folders
mkdir -p ${LOG}

# network parameters
export INPUT_HEIGHT=224
export INPUT_WIDTH=224
export INPUT_CHAN=3
export INPUT_SHAPE=1,${INPUT_HEIGHT},${INPUT_WIDTH},${INPUT_CHAN}
export INPUT_SHAPE_Q=?,${INPUT_HEIGHT},${INPUT_WIDTH},${INPUT_CHAN}
export INPUT_NODE=input_1
export OUTPUT_NODE=dense/BiasAdd
export NET_NAME=mobilenetv2_pr

# Training parameters
# Batchsize set for a GPU with 8GB of memory - modify if necessary
export BATCHSIZE=50
export EPOCHS=30
export LEARNRATE=0.0008

# list of GPUs to use
export GPU_LIST="0"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_LIST}

# TensorFlow
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true

