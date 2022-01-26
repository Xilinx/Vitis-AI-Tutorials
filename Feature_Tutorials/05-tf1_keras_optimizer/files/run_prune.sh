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

source ./0_setenv_pr.sh


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
  --gpu            ${GPU_LIST} \
  2>&1 | tee ${LOG}/train.log



echo "-----------------------------------------"
echo " STEP #3: CONVERT KERAS CHECKPOINT TO TF "
echo "-----------------------------------------"

python -u keras_to_tf.py \
  --float_model    ${FLOAT_MODEL} \
  --tf_ckpt        ${TF_DIR}/${TF_CKPT} \
  --pruning  \
  2>&1 | tee ${LOG}/keras_to_tf.log


echo "-----------------------------------------"
echo " STEP #4: EXPORT INFERENCE GRAPH"
echo "-----------------------------------------"

python -u export_inf_graph.py \
  --build_dir      ${BUILD} \
  --output_file    inference_graph.pbtxt \
  --output_node    ${OUTPUT_NODE} \
  --input_height   ${INPUT_HEIGHT} \
  --input_width    ${INPUT_WIDTH} \
  --input_chan     ${INPUT_CHAN} \
  2>&1 | tee ${LOG}/export_inf_graph.log


echo "-----------------------------------------"
echo " STEP #5: PRUNING ANALYSIS"
echo "-----------------------------------------"

# activate the TF pruning conda environment
conda activate vitis-ai-optimizer_tensorflow

prune_analysis() {
  vai_p_tensorflow \
   --action             ana \
   --input_graph        ${BUILD}/inference_graph.pbtxt \
   --input_ckpt         ${TF_DIR}/${TF_CKPT} \
   --eval_fn_path       eval_model.py \
   --target             "accuracy" \
   --workspace          ${PRUNE_ANA} \
   --input_nodes        ${INPUT_NODE} \
   --input_node_shapes  ${INPUT_SHAPE} \
   --output_nodes       ${OUTPUT_NODE} \
   --gpu                ${GPU_LIST}
}
prune_analysis 2>&1 | tee ${LOG}/prune_analysis.log


echo "-----------------------------------------"
echo " STEP #6: PRUNING & FINE-TUNING LOOP"
echo "-----------------------------------------"

# pruning command definition
# --workspace folder must be same as pruning analysis
# as pruning uses the .ana file created by pruning analysis
prune() {
  sparsity=$1
  vai_p_tensorflow \
    --action                prune \
    --input_graph           ${BUILD}/inference_graph.pbtxt \
    --input_ckpt            ${FT_DIR}/${FT_CKPT} \
    --output_graph          ${PRUNE_DIR}/pruned_graph.pbtxt \
    --output_ckpt           ${PRUNE_DIR}/${PR_CKPT} \
    --input_nodes           ${INPUT_NODE} \
    --input_node_shapes     ${INPUT_SHAPE} \
    --workspace             ${PRUNE_ANA} \
    --output_nodes          ${OUTPUT_NODE} \
    --sparsity              $sparsity \
    --gpu                   ${GPU_LIST}
}


# fine-tuning command definition
finetune_pruned_model() {
  target_acc=$1
  python -u train_ft.py \
    --input_ckpt     ${PRUNE_DIR}/${PR_CKPT} \
    --output_ckpt    ${FT_DIR}/${FT_CKPT} \
    --target_acc     $target_acc \
    --tfrec_dir      ${TFREC} \
    --tboard_dir     ${TF_BOARD} \
    --input_height   ${INPUT_HEIGHT} \
    --input_width    ${INPUT_WIDTH} \
    --input_chan     ${INPUT_CHAN} \
    --batchsize      ${BATCHSIZE} \
    --epochs         ${EPOCHS} \
    --learnrate      ${LEARNRATE} \
    --gpu            ${GPU_LIST}
}


TF_CPP_MIN_LOG_LEVEL=0

# clear any previous results
rm -rf    ${PRUNE_DIR}
rm -rf    ${FT_DIR}
mkdir -p  ${PRUNE_DIR}
mkdir -p  ${FT_DIR}

# copy trained checkpoint to fine-tuned checkpoint folder
# fine-tuned checkpoint is input for pruning
cp -f ${TF_DIR}/${TF_CKPT}*   ${FT_DIR}/.
mv ${FT_DIR}/${TF_CKPT}.data-00000-of-00002  ${FT_DIR}/${FT_CKPT}.data-00000-of-00002
mv ${FT_DIR}/${TF_CKPT}.data-00001-of-00002  ${FT_DIR}/${FT_CKPT}.data-00001-of-00002
mv ${FT_DIR}/${TF_CKPT}.index ${FT_DIR}/${FT_CKPT}.index


# pruning loop 
# first iterations in this loop are done with a lower target_acc
# last iteration of fine-tuning uses the final accuracy
for i in {1..5}
do
   echo "----------------------------------"
   echo " PRUNING STEP" $i
   echo "----------------------------------"

   # sparsity value, increments by 0.1 at each step
   SPARSITY=$(printf %.1f "$(($i))"e-1)
   echo " ** Sparsity value:" $SPARSITY

   # prune
   prune ${SPARSITY} 2>&1 | tee ${LOG}/prune_step_$i.log
   
   # fine-tuning
   echo " ** Running fine-tune.."
   rm ${FT_DIR}/*

   if [[ $i -eq 5 ]]
   then
    TARGET_ACC=0.94
   else
    TARGET_ACC=0.91
   fi

   finetune_pruned_model ${TARGET_ACC} 2>&1 | tee ${LOG}/ft_step_$i.log

done


echo "-----------------------------------------"
echo " STEP #7: CREATE DENSE CHECKPOINT (TRANSFORM)"
echo "-----------------------------------------"

# command definition
generate_dense_ckpt() {
  input_checkpoint=$1
  output_checkpoint=$2
  vai_p_tensorflow \
    --action      transform \
    --input_ckpt  $input_checkpoint \
    --output_ckpt $output_checkpoint \
    --gpu         ${GPU_LIST}
}

generate_dense_ckpt ${FT_DIR}/${FT_CKPT} ${TRSF_DIR}/${TRSF_CKPT} 2>&1 | tee ${LOG}/transform.log


echo "-----------------------------------------"
echo " STEP #8: FREEZE THE GRAPH "
echo "-----------------------------------------"

# command definition
run_freeze_graph() {
  freeze_graph \
    --input_graph       ${PRUNE_DIR}/pruned_graph.pbtxt \
    --input_checkpoint  ${TRSF_DIR}/${TRSF_CKPT} \
    --input_binary      false  \
    --output_graph      ${FROZEN_DIR}/${FROZEN_MODEL} \
    --output_node_names ${OUTPUT_NODE}
}

mkdir -p ${FROZEN_DIR}
run_freeze_graph 2>&1 | tee ${LOG}/freeze.log



echo "-----------------------------------------"
echo " STEP #9: QUANTIZE FROZEN GRAPH "
echo "-----------------------------------------"

# enable TensorFlow1 environment
conda activate vitis-ai-tensorflow

# quantize command definition
quantize() {
  vai_q_tensorflow quantize \
    --input_frozen_graph ${FROZEN_DIR}/${FROZEN_MODEL}  \
		--input_fn           image_input_fn.calib_input \
		--output_dir         ${QUANT_DIR} \
	  --input_nodes        ${INPUT_NODE} \
		--output_nodes       ${OUTPUT_NODE} \
		--input_shapes       ${INPUT_SHAPE_Q} \
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
         --build_dir  ${BUILD} \
         --data_dir   ${DATA} \
         --target     $targetname \
         --app_dir    ${APP_DIR} \
         --model_name ${NET_NAME} \
         2>&1 | tee ${LOG}/target_${targetname}.log
done

echo "-----------------------------------------"
echo "PRUNING FLOW COMPLETED.."
echo "-----------------------------------------"

