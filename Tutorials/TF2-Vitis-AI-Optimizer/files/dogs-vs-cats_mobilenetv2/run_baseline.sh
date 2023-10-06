#!/bin/sh

## Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Authors: Daniele Bagni and Mark Harvey
# date:  28 July 2023

# Build script for baseline (non-pruning) flow


## enable TensorFlow2 environment
#conda activate vitis-ai-tensorflow2


echo "-----------------------------------------"
echo " STEP #0: SET UP ENVIRONMENT VARIABLES"
echo "-----------------------------------------"

# make build folder
export BUILD=build_np
export LOG=${BUILD}/logs

mkdir -p ${LOG}

# ****************************************************************

make_tfrec(){
echo "-----------------------------------------"
echo " STEP #1: CONVERT DATASET TO TFRECORDS"
echo "-----------------------------------------"
python -u images_to_tfrec.py 2>&1 | tee ${LOG}/tfrec.log
}

make_training(){
echo "-----------------------------------------"
echo " STEP #2: TRAINING"
echo "-----------------------------------------"
python -u implement.py --mode train --build_dir ${BUILD} 2>&1 | tee ${LOG}/train.log
}

make_quantize(){
echo "-----------------------------------------"
echo " STEP #3: QUANTIZATION"
echo "-----------------------------------------"
python -u implement.py --mode quantize --build_dir ${BUILD} 2>&1 | tee ${LOG}/quantize.log
}

make_compile(){
echo "-----------------------------------------"
echo " STEP #4: COMPILE FOR TARGET"
echo "-----------------------------------------"
# modify the list of targets as required
#for targetname in zcu102 zcu104 kv260 u280 vck190;
for targetname in zcu102 vck190 vek280; do
  python -u implement.py --mode compile --build_dir ${BUILD} --target ${targetname} 2>&1 | tee ${LOG}/compile_${targetname}.log
done
}

make_target(){
echo "-----------------------------------------"
echo " STEP #5: MAKE TARGET FOLDER"
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
    make_quantize
    make_compile
    make_target
}

echo "-----------------------------------------"
echo "BASELINE FLOW COMPLETED.."
echo "-----------------------------------------"


# ****************************************************************
# DO NOT REMOVE FOLLOWING LINE

"$@"
