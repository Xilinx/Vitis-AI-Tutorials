#!/bin/bash

# Copyright 2019 Xilinx Inc.
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


# train, evaluate & save trained model
run_train() {  
  python train.py \
    --input_height       ${INPUT_HEIGHT} \
    --input_width        ${INPUT_WIDTH} \
    --input_chan         ${INPUT_CHAN} \
    --epochs             ${EPOCHS} \
    --learnrate          ${LEARNRATE} \
    --batchsize          ${BATCHSIZE} \
    --output_ckpt_path   ${CHKPT_DIR}/${CHKPT_FILENAME} \
    --infer_graph_path   ${CHKPT_DIR}/${INFER_GRAPH_FILENAME} \
    --tboard_path        ${TB_LOG}
}


train() {
  echo "-----------------------------------------"
  echo "TRAINING STARTED.."
  echo "-----------------------------------------"

  # delete previous results
  rm -rf ${BUILD}

  # make folders
  mkdir ${BUILD}
  mkdir ${LOG}
  mkdir -p ${TB_LOG}
  mkdir -p ${CHKPT_DIR}

  # training
  run_train 2>&1 | tee ${LOG}/${TRAIN_LOG}  

  echo "-----------------------------------------"
  echo "TRAINING COMPLETED"
  echo "-----------------------------------------"
}

train
