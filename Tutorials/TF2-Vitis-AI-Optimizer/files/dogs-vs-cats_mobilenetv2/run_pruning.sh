#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Authors: Daniele Bagni and Mark Harvey
# date:  28 July 2023


# Build script for pruning flow


## enable TensorFlow2 environment
#conda activate vitis-ai-tensorflow2



echo "-----------------------------------------"
echo " STEP #0: SET UP ENVIRONMENT VARIABLES"
echo "-----------------------------------------"

# path to Optimizer license
#export WRK_DIR=/workspace/tutorials/TF2-Vitis-AI-Optimizer/
#export XILINXD_LICENSE_FILE=${WRK_DIR}/vai_optimizer.lic
ls -l  ${XILINXD_LICENSE_FILE}

# make build folder
export BUILD=build_pr
export LOG=${BUILD}/logs
mkdir -p ${LOG}


make_training(){
echo "-----------------------------------------"
echo " STEP #2: TRAINING"
echo "-----------------------------------------"
# train from scratch
#python -u implement.py --mode train --build_dir ${BUILD} 2>&1 | tee ${LOG}/train.log

# reuse baseline training
cp -rf build_np/float_model ./build_pr/
cp -f  build_np/trained_accuracy.txt ./build_pr/
cmp build_np/float_model/f_model.h5 build_pr/float_model/f_model.h5
diff build_np/trained_accuracy.txt  build_pr/trained_accuracy.txt
}

make_pruning(){
echo "-----------------------------------------"
echo " STEP #3: PRUNING"
echo "-----------------------------------------"
# enable TensorFlow2 AI Optimizer environment
#conda activate vitis-ai-optimizer_tensorflow2
python -u implement.py --mode prune --build_dir ${BUILD} 2>&1 | tee ${LOG}/prune.log
grep -ise          "Pruning iter" ${LOG}/prune.log
grep -ise "Pruned model accuracy" ${LOG}/prune.log

echo "-----------------------------------------"
echo " STEP #4: TRANSFORM"
echo "-----------------------------------------"
python -u implement.py --mode transform --build_dir ${BUILD} 2>&1 | tee ${LOG}/transform.log
}

make_quantize(){
echo "-----------------------------------------"
echo " STEP #5: QUANTIZATION"
echo "-----------------------------------------"
python -u implement.py --mode quantize --build_dir ${BUILD} 2>&1 | tee ${LOG}/quantize.log
}

make_compile(){
echo "-----------------------------------------"
echo " STEP #6: COMPILE FOR TARGET"
echo "-----------------------------------------"
# modify the list of targets as required
#for targetname in zcu102 zcu104 kv260 u280 vck190; do
for targetname in zcu102 vck190 vek280; do
  python -u implement.py --mode compile --build_dir ${BUILD} --target ${targetname} 2>&1 | tee ${LOG}/compile_${targetname}.log
done
}

make_target(){
echo "-----------------------------------------"
echo " STEP #7: MAKE TARGET FOLDER"
echo "-----------------------------------------"
# modify the list of targets as required
#for targetname in zcu102 zcu104 kv260 u280 vck190; do
for targetname in zcu102 vck190 vek280; do
  python -u target.py --build_dir  ${BUILD} --target ${targetname} 2>&1 | tee ${LOG}/target_${targetname}.log
done
# prepare TAR archives
for targetname in zcu102 vck190 vek280; do
  cp -r ../cifar10_customcnn/application      ${BUILD}/target_${targetname}/
  rm -f ${BUILD}/target_${targetname}/application/*cifar*
  rm -f ${BUILD}/target_${targetname}/application/code/src/*.py
  rm -f ${BUILD}/target_${targetname}/application/code/src/main*.cc
  rm -f ${BUILD}/target_${targetname}/application/code/*.dat
  rm -f ${BUILD}/target_${targetname}/application/code/logfile*
  rm -f ${BUILD}/target_${targetname}/application/code/*cifar*
  rm -f ${BUILD}/target_${targetname}/application/code/build_app.sh
  tar -cvf ${targetname}_${BUILD}.tar ${BUILD}/target_${targetname}/ >/dev/null 2>&1
done
}

main()
{
    make_training
    make_pruning
    make_quantize
    make_compile
    make_target
}


echo "-----------------------------------------"
echo "PRUNING FLOW COMPLETED..."
echo "-----------------------------------------"


# ****************************************************************
# DO NOT REMOVE FOLLOWING LINE

"$@"
