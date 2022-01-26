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


# Build script for baseline (non-pruning) flow


# enable TensorFlow1 environment
conda activate vitis-ai-tensorflow


echo "-----------------------------------------"
echo " STEP #0: SET UP ENVIRONMENT VARIABLES"
echo "-----------------------------------------"

source ./0_setenv_np.sh


echo "-----------------------------------------"
echo " STEP #1: CONVERT DATASET TO TFRECORDS"
echo "-----------------------------------------"

python -u images_to_tfrec.py \
  --data_dir       ${DATA} \
  --tfrec_dir      ${TFREC} \
  --input_height   ${INPUT_HEIGHT} \
  --input_width    ${INPUT_WIDTH} \
  2>&1 | tee ${LOG}/tfrec.log


echo "-----------------------------------------"
echo " STEP #2: TRAINING"
echo "-----------------------------------------"

#  --learnrate      ${LEARNRATE} \

python -u train_ft.py \
  --output_ckpt    ${FLOAT_MODEL} \
  --tfrec_dir      ${TFREC} \
  --tboard_dir     ${TF_BOARD} \
  --input_height   ${INPUT_HEIGHT} \
  --input_width    ${INPUT_WIDTH} \
  --input_chan     ${INPUT_CHAN} \
  --batchsize      ${BATCHSIZE} \
  --epochs         ${EPOCHS} \
  --learnrate      ${LEARNRATE} \
  --target_acc     0.80 \
  --gpu            ${GPU_LIST} \
  2>&1 | tee ${LOG}/train.log


echo "-----------------------------------------"
echo " STEP #3: CONVERT KERAS CHECKPOINT TO TF "
echo "-----------------------------------------"

python -u keras_to_tf.py \
  --float_model    ${FLOAT_MODEL} \
  --tf_ckpt        ${TF_DIR}/${TF_CKPT} \
  2>&1 | tee ${LOG}/keras_to_tf.log



echo "-----------------------------------------"
echo " NOTE: STEPS 4 TO 7 NOT USED IN BASELINE FLOW"
echo "-----------------------------------------"



echo "-----------------------------------------"
echo " STEP #8: FREEZE THE GRAPH "
echo "-----------------------------------------"

# command definition
run_freeze_graph() {
  freeze_graph \
    --input_meta_graph  ${TF_DIR}/${TF_META} \
    --input_checkpoint  ${TF_DIR}/${TF_CKPT} \
    --output_graph      ${FROZEN_DIR}/${FROZEN_MODEL} \
    --output_node_names ${OUTPUT_NODE} \
    --input_binary      true
}

mkdir -p ${FROZEN_DIR}
run_freeze_graph 2>&1 | tee ${LOG}/freeze.log



echo "-----------------------------------------"
echo " STEP #9: QUANTIZE FROZEN GRAPH "
echo "-----------------------------------------"

# quantize command definition
quantize() {
  vai_q_tensorflow quantize \
    --input_frozen_graph ${FROZEN_DIR}/${FROZEN_MODEL} \
    --input_fn           image_input_fn.calib_input \
    --output_dir         ${QUANT_DIR} \
    --input_nodes        ${INPUT_NODE} \
    --output_nodes       ${OUTPUT_NODE}  \
    --input_shapes       ${INPUT_SHAPE_Q}  \
    --calib_iter         10 \
    --gpu                ${GPU_LIST}
}

quantize 2>&1 | tee ${LOG}/quantize.log


echo "-----------------------------------------"
echo " STEP #10: EVALUATE QUANTIZED GRAPH "
echo "-----------------------------------------"

python -u eval_graph.py \
  --data_dir    ${DATA} \
  --graph       ${QUANT_DIR}/quantize_eval_model.pb \
  --input_node  ${INPUT_NODE} \
  --output_node ${OUTPUT_NODE} \
  --batchsize   ${BATCHSIZE} \
  --gpu         ${GPU_LIST} \
  2>&1 | tee ${LOG}/eval_quant.log



echo "-----------------------------------------"
echo " STEP #11: COMPILE FOR TARGET "
echo "-----------------------------------------"

# compile command definition
compile() {
  TARGET=$1
  if [ $TARGET = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      echo "COMPILING MODEL FOR ZCU102.."
  elif [ $TARGET = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      echo "COMPILING MODEL FOR ZCU104.."
  elif [ $TARGET = kv260 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
      echo "COMPILING MODEL FOR KV260.."
  elif [ $TARGET = u280 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8L/U280/arch.json
      echo "COMPILING MODEL FOR ALVEO U280.."
  elif [ $TARGET = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      echo "COMPILING MODEL FOR VERSAL VCK190.."
  else
      echo  "Target not found. Valid choices are: zcu102, zcu104, kv260, u280, vck190 ..exiting"
      exit 1
  fi

  vai_c_tensorflow \
    --frozen_pb       ${BUILD}/quant_model/quantize_eval_model.pb \
    --arch            $ARCH \
    --output_dir      ${BUILD}/compiled_model_${TARGET} \
    --net_name        ${NET_NAME}

}

# compile for target boards
for targetname in zcu102 zcu104 kv260 u280 vck190; do
  compile $targetname 2>&1 | tee ${LOG}/compile_${targetname}.log
done 



echo "-----------------------------------------"
echo " STEP #12: MAKE TARGET FOLDER"
echo "-----------------------------------------"

for targetname in zcu102 zcu104 kv260 u280 vck190; do
  python -u target.py \
         --build_dir ${BUILD} \
         --data_dir   ${DATA} \
         --target     $targetname \
         --app_dir    ${APP_DIR} \
         --model_name ${NET_NAME} \
         2>&1 | tee ${LOG}/target_${targetname}.log
done

echo "-----------------------------------------"
echo "BASELINE FLOW COMPLETED.."
echo "-----------------------------------------"

