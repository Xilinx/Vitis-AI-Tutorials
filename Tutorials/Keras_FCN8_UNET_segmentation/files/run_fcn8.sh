#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


## clean up previous log files
#rm -f ./log/*.log

CNN=fcn8


# folders
WORK_DIR=./build
LOG_DIR=${WORK_DIR}/../log
RPT_DIR=${WORK_DIR}/../rpt
TARGET_102=${WORK_DIR}/../target_zcu102
TARGET_104=${WORK_DIR}/../target_zcu104
TARGET_190=${WORK_DIR}/../target_vck190
TARGET_280=${WORK_DIR}/../target_vek280
KERAS_MODEL_DIR=${WORK_DIR}/../keras_model
DATASET_DIR=${WORK_DIR}/dataset1

TB_LOG_DIR=${WORK_DIR}/tb_log
CHKPT_DIR=${WORK_DIR}/tf_chkpts
FREEZE_DIR=${WORK_DIR}/freeze
COMPILE_DIR=${WORK_DIR}/compile
QUANT_DIR=${WORK_DIR}/quantize_results

# checkpoints & graphs filenames
CHKPT_FILENAME=float_model.ckpt
META_GRAPH_FILENAME=${CHKPT_FILENAME}.meta
FROZEN_GRAPH_FILENAME=frozen_graph.pb
QUANTIZED_FILENAME=quantize_eval_model.pb

# logs & results files
PREPARE_DATA_LOG=${CNN}_prepare_data.log
TRAIN_LOG=${CNN}_training.log
FREEZE_LOG=${CNN}_freeze_graph.log
EVAL_FR_LOG=${CNN}_evaluate_frozen_graph.log
QUANT_LOG=${CNN}_quantize.log
EVAL_Q_LOG=${CNN}_evaluate_quantized_graph.log
COMP_LOG=${CNN}_compile.log

# CNN parameters
INPUT_NODE="input_1"
OUTPUT_NODE="activation/truediv" # output node of floating point CNN
Q_OUTPUT_NODE="conv2d_transpose_2/conv2d_transpose" # output node of quantized CNN

##################################################################################




a_clean_and_make_directories() {
    echo " "
    echo "##################################################################################"
    echo "A) CLEAN PREVIOUS DIRECTORIES"
    echo " ##################################################################################"
    echo " "
    # clean up previous results
    rm -rf ${WORK_DIR}; mkdir ${WORK_DIR}
    rm -rf ${LOG_DIR}; mkdir ${LOG_DIR}
    rm -rf ${RPT_DIR}; mkdir ${RPT_DIR}
    rm -rf ${CHKPT_DIR}; mkdir ${CHKPT_DIR}
    rm -rf ${DATASET_DIR}; mkdir ${DATASET_DIR}
    mkdir  ${DATASET_DIR}/img_calib ${DATASET_DIR}/img_test ${DATASET_DIR}/img_train ${DATASET_DIR}/img_valid
    mkdir  ${DATASET_DIR}/seg_calib ${DATASET_DIR}/seg_test ${DATASET_DIR}/seg_train ${DATASET_DIR}/seg_valid
    rm -rf ${FREEZE_DIR}; mkdir ${FREEZE_DIR}
    rm -rf ${QUANT_DIR}; mkdir ${QUANT_DIR}
    rm -rf ${COMPILE_DIR}; mkdir ${COMPILE_DIR}
    mkdir ${LOG_DIR}/${CNN} ${CHKPT_DIR}/${CNN} ${FREEZE_DIR}/${CNN} ${QUANT_DIR}/${CNN} ${COMPILE_DIR}/${CNN}
}


##################################################################################

1_generate_images() {
    echo " "
    echo "##################################################################################"
    echo "Step1: CREATE DATA AND FOLDERS"
    echo "##################################################################################"
    echo " "
    # clean files in pre-built sub-directories
    rm -f ${DATASET_DIR}/img_*/* ${DATASET_DIR}/seg_*/*
    # unzip the original dataset
    unzip ${WORK_DIR}/../dataset1.zip -d ${WORK_DIR} >& /dev/null
    cd code
    # put the data into proper folders
    python prepare_data.py
    cd ..
    # clean previous directories
    rm -r ${DATASET_DIR}/annotations_* ${DATASET_DIR}/images_*

    # YOU MUST HAVE THE the HDF5 weights file for VGG encoder subnet of FCN8
    cd $KERAS_MODEL_DIR
    rm -f vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5*  #remove any previous file, if any
    wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 >& /dev/null
    cd ..

}

##################################################################################
# effective training
2_fcn8_train() {

    # effective training and predictions
    cd code
    echo "current dir is "
    pwd
    echo " "
    echo "##################################################################################"
    echo "Step2a: FCN8 TRAINING"
    echo "##################################################################################"
    echo " "
    python fcn8_training.py         --upscale "False"
    echo " "
    cd ../code
    echo "##################################################################################"
    echo "Step2b: FCN8 MAKING PREDICTIONS"
    echo "##################################################################################"
    echo " "
    python fcn8_make_predictions.py --upscale "False"
    cd ..
}

##################################################################################
# Keras to TF chkpt files
3_fcn8_Keras2TF() {
    echo " "
    echo "#######################################################################################"
    echo "Step3: FCN8 KERAS to TENSORFLOW GRAPH CONVERSION"
    echo "#######################################################################################"
    echo " "
    # clean TF Check Point files
    #rm ${CHKPT_DIR}/${CNN}/*
    # from Keras to TF
    cd code
    python Keras2TF.py --model "fcn8"
    cd ..
}



##################################################################################
# freeze the inference graph
4a_fcn8_freeze() {
    echo " "
    echo "##############################################################################"
    echo "Step4a: FCN8 FREEZE TF GRAPHS"
    echo "##############################################################################"
    echo " "
    # freeze the TF graph
    freeze_graph \
	--input_meta_graph  ${CHKPT_DIR}/${CNN}/${META_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}/${FROZEN_GRAPH_FILENAME} \
	--output_node_names ${OUTPUT_NODE}

    echo " "
    echo "##############################################################################"
    echo "Step4a: FCN8 INSPECT FROZEN GRAPH"
    echo "##############################################################################"
    echo " "
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}/${FROZEN_GRAPH_FILENAME}
}


##################################################################################
# evaluate the original graph
4b_eval_graph() {
    echo " "
    echo "##############################################################################"
    echo "Step4b: FCN8 EVALUATING THE ORIGINAL GRAPH"
    echo "##############################################################################"
    echo " "
    cd code
    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0
    cd ..
}


##################################################################################
5a_fcn8_quantize() {
    echo " "
    echo "##########################################################################"
    echo "Step5a: FCN8 QUANTIZATION"
    echo "##########################################################################"
    echo " "
    # log the decent_q version being used
    echo " "
    vai_q_tensorflow --version
    echo " "
    #quantize
    cd code
    vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        ${Q_OUTPUT_NODE} \
	 --output_dir          ../${QUANT_DIR}/${CNN}/ \
	 --method              1 \
	 --input_fn            graph_input_fn.calib_input \
	 --calib_iter          10 \
	 --gpu 0

    cd ..
}

##################################################################################
# make predictions with quantized graph

5b_eval_quantized_graph() {
    echo " "
    echo "##############################################################################"
    echo "Step5b: FCN8 EVALUATE QUANTIZED GRAPH"
    echo "##############################################################################"
    echo " "
    cd code
    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${Q_OUTPUT_NODE} \
	   --gpu=0
    cd ..
}


##################################################################################
# Compile xmodel file for VCK190 board with Vitis AI Compiler
6_compile_vai_vck190() {
  echo " "
  echo "##########################################################################"
  echo "COMPILE FCN8 XMODEL FILE WITH Vitis AI for VCK190 TARGET"
  echo "##########################################################################"
  echo " "

  vai_c_tensorflow \
      --frozen_pb ${QUANT_DIR}/${CNN}/quantize_eval_model.pb \
      --arch /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
      --output_dir ${COMPILE_DIR}/${CNN} \
      --options    "{'mode':'normal'}" \
      --net_name ${CNN}
 }



 ##################################################################################
 # Compile xmodel file for VEK280 board with Vitis AI Compiler
 6_compile_vai_vek280() {
   echo " "
   echo "##########################################################################"
   echo "COMPILE FCN8 XMODEL FILE WITH Vitis AI for VEK280 TARGET"
   echo "##########################################################################"
   echo " "

   vai_c_tensorflow \
       --frozen_pb ${QUANT_DIR}/${CNN}/quantize_eval_model.pb \
       --arch /opt/vitis_ai/compiler/arch/DPUCV2DX8G/VEK280/arch.json \
       --output_dir ${COMPILE_DIR}/${CNN} \
       --options    "{'mode':'normal'}" \
       --net_name ${CNN}
  }

##################################################################################
# Compile xmodel file for ZCU102 board with Vitis AI Compiler
6_compile_vai_zcu102() {
  echo " "
  echo "##########################################################################"
  echo "COMPILE FCN8 XMODEL FILE WITH Vitis AI for ZCU102"
  echo "##########################################################################"
  echo " "
  # for Vitis AI == 1.0
  #python /opt/vitis_ai/compiler/vai_c_tensorflow \

  # for Vitis AI >= 1.1
  vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
	 --output_dir ${COMPILE_DIR}/${CNN} \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}
   #	 --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU102/ZCU102.json \

 }

 ##################################################################################
 # Compile xmodel file for ZCU104 board with Vitis AI Compiler
 6_compile_vai_zcu104() {
   echo " "
   echo "##########################################################################"
   echo "COMPILE FCN8 MODEL FILE WITH Vitis AI for ZCU104"
   echo "##########################################################################"
   echo " "
   # for Vitis AI == 1.0
   #python /opt/vitis_ai/compiler/vai_c_tensorflow \

   # for Vitis AI >= 1.1
   vai_c_tensorflow \
 	 --frozen_pb ${QUANT_DIR}/${CNN}/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
 	 --output_dir ${COMPILE_DIR}/${CNN} \
 	 --options    "{'mode':'normal'}" \
 	 --net_name ${CNN}
  }


##################################################################################
##################################################################################

main() {

    a_clean_and_make_directories

    # create the proper folders and images from the original dataset
    1_generate_images #2>&1 | tee ${LOG_DIR}/${CNN}/${PREPARE_DATA_LOG}

    # do the training and make predictions
    2_fcn8_train     #2>&1 | tee ${LOG_DIR}/${CNN}/${TRAIN_LOG}


    # from Keras to TF
    3_fcn8_Keras2TF  #2>&1 | tee ${LOG_DIR}/${CNN}/fcn8_keras2tf.log

    # freeze the graph and inspect it
    4a_fcn8_freeze   #2>&1 | tee ${LOG_DIR}/${CNN}/${FREEZE_LOG}

    # evaluate the frozen graph performance
    4b_eval_graph #2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_FR_LOG}

    # quantize
    5a_fcn8_quantize #2>&1 | tee ${LOG_DIR}/${CNN}/${QUANT_LOG}

    # evaluate post-quantization model
    5b_eval_quantized_graph #2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_Q_LOG}

    6_compile_vai_zcu104 #2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel file to target board directory
    mv  ${COMPILE_DIR}/${CNN}/*.xmodel   ${TARGET_104}/${CNN}/model/
    cp ${COMPILE_DIR}/${CNN}/*.json      ${TARGET_104}/${CNN}/model/

    # compile to generate xmodel file for target board
    6_compile_vai_vck190 #2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel file to target board directory
    mv  ${COMPILE_DIR}/${CNN}/*.xmodel   ${TARGET_190}/${CNN}/model/
    cp ${COMPILE_DIR}/${CNN}/*.json      ${TARGET_190}/${CNN}/model/

    6_compile_vai_vek280 #2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel file to target board directory
    mv  ${COMPILE_DIR}/${CNN}/*.xmodel   ${TARGET_280}/${CNN}/model/
    cp ${COMPILE_DIR}/${CNN}/*.json      ${TARGET_280}/${CNN}/model/

    6_compile_vai_zcu102 #2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel file to target board directory
    mv  ${COMPILE_DIR}/${CNN}/*.xmodel   ${TARGET_102}/${CNN}/model/
    cp ${COMPILE_DIR}/${CNN}/*.json      ${TARGET_102}/${CNN}/model/

    # copy test images into target board
    tar -cvf "test.tar" ${DATASET_DIR}/img_test ${DATASET_DIR}/seg_test >& /dev/null
    gzip -f test.tar
    cp -f test.tar.gz ${TARGET_190}/
    cp -f test.tar.gz ${TARGET_280}/
    cp -f test.tar.gz ${TARGET_102}/
    #cp -f test.tar.gz ${TARGET_104}/

: '
    tar -cvf target_zcu102.tar ${TARGET_102}/  >& /dev/null
    #tar -cvf target_zcu104.tar ${TARGET_104}/  >& /dev/null
    tar -cvf target_vck190.tar ${TARGET_190}/  >& /dev/null
    tar -cvf target_vek280.tar ${TARGET_280}/  >& /dev/null
'

    echo "#####################################"
    echo "MAIN FCN8 FLOW COMPLETED"
    echo "#####################################"


}

main
