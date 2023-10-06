#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Mark Harvey
# Modified by: Daniele Bagni, AMD/Xilinx
# date:  01 Aug 2023

# Build script for pruning flow


## enable TensorFlow2 environment
#conda activate vitis-ai-tensorflow2



echo "-----------------------------------------"
echo " STEP #1: SET UP ENVIRONMENT VARIABLES"
echo "-----------------------------------------"

## path to Optimizer license
#export WRK_DIR=/workspace/tutorials/TF2-Vitis-AI-Optimizer/
#export XILINXD_LICENSE_FILE=${WRK_DIR}/vai_optimizer.lic
ls -l  ${XILINXD_LICENSE_FILE}


# make build folder
export BUILD=build_pr
export LOG=${BUILD}/logs
mkdir -p ${LOG}

CNN=$2
echo " "
echo "CNN= " ${CNN}
echo " "

# ****************************************************************

make_training(){
echo "-----------------------------------------"
echo " STEP #2: TRAINING"
echo "-----------------------------------------"
# train from scratch
#python -u cifar10_implement.py --mode train --build_dir ${BUILD} 2>&1 | tee ${LOG}/train.log

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
python -u cifar10_implement.py --mode prune --build_dir ${BUILD}  -n ${CNN} 2>&1 | tee ${LOG}/prune.log
grep -ise          "Pruning iter" ${LOG}/prune.log
grep -ise "Pruned model accuracy" ${LOG}/prune.log

echo "-----------------------------------------"
echo " STEP #4: TRANSFORM"
echo "-----------------------------------------"
python -u cifar10_implement.py --mode transform --build_dir ${BUILD}  -n ${CNN} 2>&1 | tee ${LOG}/transform.log
}

make_quantize(){
echo "-----------------------------------------"
echo " STEP #5: QUANTIZATION"
echo "-----------------------------------------"
python -u cifar10_implement.py --mode quantize --build_dir ${BUILD}  -n ${CNN} 2>&1 | tee ${LOG}/quantize.log
}

make_compile(){
echo "-----------------------------------------"
echo " STEP #6: COMPILE FOR TARGET"
echo "-----------------------------------------"
# modify the list of targets as required
#for targetname in zcu102 zcu104 kv260 u280 vck190; do
for targetname in zcu102 vck190 vek280; do
  python -u cifar10_implement.py --mode compile --build_dir ${BUILD} --target ${targetname}  -n ${CNN} 2>&1 | tee ${LOG}/compile_${targetname}.log
done
}


make_target(){
  echo "-----------------------------------------"
  echo " STEP #7: MAKE TARGET FOLDER"
  echo "-----------------------------------------"

 CUR_DIR=$PWD
 # prepare test images once
 cp -r dataset/cifar10/test .  >/dev/null 2>&1 # temporary folder
 cd ./test
 for targetdir in airplane automobile bird cat dog deer frog horse ship truck; do
   echo "WORKING ON " ${targetdir}
   mv ${targetdir}/*.png .
   rm -r ${targetdir}
 done
 cd ${CUR_DIR}

 # prepare TAR archives
 for targetname in zcu102 vck190 vek280; do
   mkdir -p ${BUILD}/target_${targetname}
   cp application/cifar10_app_mt.py         ${BUILD}/target_${targetname}/
   cp application/run_all_cifar10_target.sh ${BUILD}/target_${targetname}/
   cp -r application/code                   ${BUILD}/target_${targetname}/
   cp ${BUILD}/compiled_model_${targetname}/*.xmodel ${BUILD}/target_${targetname}/
   cp -r ./test ${BUILD}/target_${targetname}/ >/dev/null 2>&1
   tar -cvf ${targetname}_${BUILD}.tar ${BUILD}/target_${targetname}/ >/dev/null 2>&1
 done
 #remove temporary folder
 rm -r ./test
}


main()
{
    make_training
    make_pruning
    make_quantize
    make_compile
    make_target 2>&1 |  tee ${LOG}/target_${targetname}.log
}


echo "-----------------------------------------"
echo "PRUNING FLOW COMPLETED.."
echo "-----------------------------------------"


# ****************************************************************
# DO NOT REMOVE FOLLOWING LINE

"$@"
