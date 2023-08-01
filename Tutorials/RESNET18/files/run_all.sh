#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  24 May 2023


LOG_FILENAME=$2


# ===========================================================================
# analyze DPU graphs for TARGET ZCU102 CIFAR10
# ===========================================================================
analyze_graphs(){
echo "----------------------------------------------------------------------------------"
echo "ANALYZING GRAPHS FOR ZCU102"
echo "----------------------------------------------------------------------------------"
source ./scripts/analyze_subgraphs.sh zcu102 q_train1_resnet18_cifar10_final.h5
}

# ===========================================================================
# remove redundant information from host logfile
# ===========================================================================
prepare_logfile(){
  #cat  logfile_resnet18_cifar10.txt logfile3_resnet18_cifar10.txt > logfile0_resnet18_cifar10.txt
  #mv logfile0_resnet18_cifar10.txt logfile_resnet18_cifar10.txt
  dos2unix -f ${LOG_FILENAME} #logfile_run_all_7apr2023.txt
  cat ${LOG_FILENAME}  | grep -v "loss: " | tee prova1.txt
  cat prova1.txt | grep -v "100%|" | tee prova2.txt
  cat prova2.txt | grep -v "ETA: " | tee prova3.txt
  cat ./doc/header.txt prova3.txt > logfile_host.txt
  rm -f prova*.txt
}

# ===========================================================================
# STEP1: clean and dos2unix
# ===========================================================================
run_clean_dos2unix(){
source ./scripts/clean_all.sh
}

# ===========================================================================
# STEP2: Prepare CIFAR10 Dataset
# ===========================================================================
cifar10_dataset(){
#rm -rf build/dataset
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP2]  CREATING CIFAR10 DATASET OF IMAGES"
echo "----------------------------------------------------------------------------------"
echo " "
# organize CIFAR10  data
python code/cifar10_generate_images.py   | tee build/log/cifar10_generate_images.log
}

# ===========================================================================
# STEP3: Train ResNet18 CNNs on CIFAR10
# ===========================================================================
run_cifar10_training(){
# floating point model training
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3A] CIFAR10 TRAINING (way 1)"
echo "----------------------------------------------------------------------------------"
echo " "
python ./code/train1_resnet18_cifar10.py --epochs 50 | tee ./build/log/train1_resnet18_cifar10.log
mv ./build/float/train1_best_chkpt.h5 ./build/float/train1_resnet18_cifar10_best.h5
mv ./build/float/train1_final.h5      ./build/float/train1_resnet18_cifar10_final.h5

echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3B] CIFAR10 TRAINING (way 2)"
echo "----------------------------------------------------------------------------------"
echo " "
python ./code/train2_resnet18_cifar10.py --epochs 50 | tee ./build/log/train2_resnet18_cifar10.log

}

# ===========================================================================
# STEP4: Vitis AI Quantization of ResNet18 on CIFAR10
# ===========================================================================
quantize_resnet18_cifar10(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP4A] QUANTIZE CIFAR10 TRAINED CNN1: MODEL INSPECTION"
echo "----------------------------------------------------------------------------------"
echo " "
python  ./code/inspect_resnet18_cifar10.py --float_file ./build/float/train1_resnet18_cifar10_final.h5
mv build/log/inspect_results.txt build/log/inspect_results_train1_resnet18_cifar10_final.txt
mv build/log/model.svg build/log/model_train1_resnet18_cifar10_final.svg
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP4B] QUANTIZE CIFAR10 TRAINED CNNs1"
echo "----------------------------------------------------------------------------------"
echo " "
echo "[DB INFO STEP4B-1] CIFAR10 TRAINED CNN1 FINAL"
echo " "
python  ./code/vai_q_resnet18_cifar10.py   --float_file ./build/float/train1_resnet18_cifar10_final.h5 --quant_file ./build/quantized/q_train1_resnet18_cifar10_final.h5
echo " "
echo "[DB INFO STEP4B-2] CIFAR10 TRAINED CNN1 BEST"
echo " "
python  ./code/vai_q_resnet18_cifar10.py   --float_file ./build/float/train1_resnet18_cifar10_best.h5  --quant_file ./build/quantized/q_train1_resnet18_cifar10_best.h5
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP4C] QUANTIZE CIFAR10 TRAINED CNN2"
echo "----------------------------------------------------------------------------------"
echo " "
echo "[DB INFO STEP4C-1] MODEL INSPECTION"
echo " "
python  ./code/inspect_resnet18_cifar10.py --float_file ./build/float/train2_resnet18_cifar10_float.h5
mv build/log/inspect_results.txt build/log/inspect_results_train2_resnet18_cifar10_float.txt
mv build/log/model.svg           build/log/model_train2_resnet18_cifar10_float.svg
echo " "
echo "[DB INFO STEP4C-2] EFFECTIVE QUANTIZATION"
echo " "
python  ./code/vai_q_resnet18_cifar10.py   --float_file ./build/float/train2_resnet18_cifar10_float.h5 --quant_file ./build/quantized/q_train2_resnet18_cifar10.h5

}

# ===========================================================================
# STEP5: Vitis AI Compile ResNet18 CIFAR10 for Target Board
# ===========================================================================
compile_resnet18_cifar10(){
#train1
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP5A] COMPILE CIFAR10 QUANTIZED CNN1"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_train1_resnet18_cifar10_final.h5
source ./scripts/run_compile.sh vck190  q_train1_resnet18_cifar10_final.h5
source ./scripts/run_compile.sh vek280  q_train1_resnet18_cifar10_final.h5
source ./scripts/run_compile.sh vck5000 q_train1_resnet18_cifar10_final.h5
source ./scripts/run_compile.sh v70     q_train1_resnet18_cifar10_final.h5

mv   ./build/compiled_zcu102/zcu102_q_train1_resnet18_cifar10_final.h5.xmodel  ./target/cifar10/zcu102_train1_resnet18_cifar10.xmodel
mv   ./build/compiled_vck190/vck190_q_train1_resnet18_cifar10_final.h5.xmodel  ./target/cifar10/vck190_train1_resnet18_cifar10.xmodel
mv   ./build/compiled_vek280/vek280_q_train1_resnet18_cifar10_final.h5.xmodel  ./target/cifar10/vek280_train1_resnet18_cifar10.xmodel
mv   ./build/compiled_v70/v70_q_train1_resnet18_cifar10_final.h5.xmodel        ./target/cifar10/v70_train1_resnet18_cifar10.xmodel
mv ./build/compiled_vck5000/vck5000_q_train1_resnet18_cifar10_final.h5.xmodel  ./target/cifar10/vck5000_train1_resnet18_cifar10.xmodel

#train2
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP5B] COMPILE CIFAR10 QUANTIZED CNN2"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_train2_resnet18_cifar10.h5
source ./scripts/run_compile.sh vck190  q_train2_resnet18_cifar10.h5
source ./scripts/run_compile.sh vek280  q_train2_resnet18_cifar10.h5
source ./scripts/run_compile.sh vck5000 q_train2_resnet18_cifar10.h5
source ./scripts/run_compile.sh v70     q_train2_resnet18_cifar10.h5

mv   ./build/compiled_zcu102/zcu102_q_train2_resnet18_cifar10.h5.xmodel  ./target/cifar10/zcu102_train2_resnet18_cifar10.xmodel
mv   ./build/compiled_vck190/vck190_q_train2_resnet18_cifar10.h5.xmodel  ./target/cifar10/vck190_train2_resnet18_cifar10.xmodel
mv   ./build/compiled_vek280/vek280_q_train2_resnet18_cifar10.h5.xmodel  ./target/cifar10/vek280_train2_resnet18_cifar10.xmodel
mv ./build/compiled_vck5000/vck5000_q_train2_resnet18_cifar10.h5.xmodel  ./target/cifar10/vck5000_train2_resnet18_cifar10.xmodel
mv         ./build/compiled_v70/v70_q_train2_resnet18_cifar10.h5.xmodel  ./target/cifar10/v70_train2_resnet18_cifar10.xmodel

}

# ===========================================================================
# STEP6: prepare archive for TARGET ZCU102 runtime application for CIFAR10
# ===========================================================================
prepare_cifar10_archives() {
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP6] PREPARING CIFAR10 ARCHIVE FOR TARGET BOARDS"
echo "----------------------------------------------------------------------------------"
echo " "
cp -r target       ./build
cd ./build/dataset/cifar10
tar -cvf test.tar ./test > /dev/null
cp test.tar       ../../../build/target/cifar10/
rm test.tar
cd ../../../
rm -rf ./build/target/imagenet #unuseful at the moment
# zcu102
cp -r ./build/target/  ./build/target_zcu102  > /dev/null
rm -f ./build/target_zcu102/cifar10/vck*_cifar10.xmodel
rm -f ./build/target_zcu102/cifar10/vek*_cifar10.xmodel
rm -f ./build/target_zcu102/cifar10/v70*_cifar10.xmodel
# vck190
cp -r ./build/target/  ./build/target_vck190  > /dev/null
rm -f ./build/target_vck190/cifar10/zcu1*_cifar10.xmodel
rm -f ./build/target_vck190/cifar10/vek2*_cifar10.xmodel
rm -f ./build/target_vck190/cifar10/vck5*_cifar10.xmodel
rm -f ./build/target_vck190/cifar10/v70*_cifar10.xmodel
# vek280
cp -r ./build/target   ./build/target_vek280  > /dev/null
rm -f ./build/target_vek280/cifar10/zcu*_cifar10.xmodel
rm -f ./build/target_vek280/cifar10/vck*_cifar10.xmodel
rm -f ./build/target_vek280/cifar10/v70*_cifar10.xmodel
# vck5000
cp -r ./build/target/  ./build/target_vck5000  > /dev/null
rm -f ./build/target_vck5000/cifar10/zcu1*_cifar10.xmodel
rm -f ./build/target_vck5000/cifar10/vek2*_cifar10.xmodel
rm -f ./build/target_vck5000/cifar10/vck1*_cifar10.xmodel
rm -f ./build/target_vck5000/cifar10/v70*_cifar10.xmodel
# v70
cp -r ./build/target/  ./build/target_v70  > /dev/null
rm -f ./build/target_v70/cifar10/zcu1*_cifar10.xmodel
rm -f ./build/target_v70/cifar10/vek2*_cifar10.xmodel
rm -f ./build/target_v70/cifar10/vck*_cifar10.xmodel
}

# ===========================================================================
# main for CIFAR10
# ===========================================================================
# do not change the order of the following commands

main_cifar10(){
  echo " "
  echo " "
  pip install image-classifiers
  run_clean_dos2unix          # 1
  #cifar10_dataset             # 2
  #run_cifar10_training       # 3
  quantize_resnet18_cifar10   # 4
  compile_resnet18_cifar10    # 5
  ###cross compile the application on target
  ##cd target
  ##source ./cifar10/run_all_cifar10_target.sh compile_cif10
  ##cd ..
  prepare_cifar10_archives    # 6
  echo " "
  echo " "
}

# ======================================================================================================================================================
# ======================================================================================================================================================
# IMAGENET
# ======================================================================================================================================================
# ======================================================================================================================================================


# ===========================================================================
# STEP7: Vitis AI Quantization of ResNet50 on ImageNet
# ===========================================================================
quantize_resnet50_imagenet(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP7] IMAGENET RESNET50: EVALUATE & QUANTIZE"
echo "----------------------------------------------------------------------------------"
echo " "
python  ./code/eval_resnet50.py
}

# ===========================================================================
# STEP8: Vitis AI Quantization of ResNet18 on ImageNet
# ===========================================================================
quantize_resnet18_imagenet(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP8] IMAGENET RESNET18: EVALUATE & QUANTIZE"
echo "----------------------------------------------------------------------------------"
echo " "
python  ./code/eval_resnet18.py
}

# ===========================================================================
# STEP9: Vitis AI Compile ResNet50 Imagenet Target Board
# ===========================================================================
compile_resnet50_imagenet(){
#train1
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP9] COMPILE IMAGENET QUANTIZED RESNET50"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_resnet50_imagenet.h5
source ./scripts/run_compile.sh vck190  q_resnet50_imagenet.h5
source ./scripts/run_compile.sh vek280  q_resnet50_imagenet.h5
source ./scripts/run_compile.sh vck5000 q_resnet50_imagenet.h5
source ./scripts/run_compile.sh v70     q_resnet50_imagenet.h5
mv   ./build/compiled_zcu102/zcu102_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/zcu102_resnet50_imagenet.xmodel
mv   ./build/compiled_vck190/vck190_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/vck190_resnet50_imagenet.xmodel
mv   ./build/compiled_vek280/vek280_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/vek280_resnet50_imagenet.xmodel
mv ./build/compiled_vck5000/vck5000_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/vck5000_resnet50_imagenet.xmodel
mv     ./build/compiled_v70/v70_q_resnet50_imagenet.h5.xmodel      ./target/imagenet/v70_resnet50_imagenet.xmodel
}

# ===========================================================================
# STEP10: Vitis AI Compile ResNet18 Imagenet Target Board
# ===========================================================================
compile_resnet18_imagenet(){
#train1
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP10] COMPILE IMAGENET QUANTIZED RESNET18"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_resnet18_imagenet.h5
source ./scripts/run_compile.sh vck190  q_resnet18_imagenet.h5
source ./scripts/run_compile.sh vek280  q_resnet18_imagenet.h5
source ./scripts/run_compile.sh vck5000 q_resnet18_imagenet.h5
source ./scripts/run_compile.sh v70     q_resnet18_imagenet.h5
mv   ./build/compiled_zcu102/zcu102_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/zcu102_resnet18_imagenet.xmodel
mv   ./build/compiled_vck190/vck190_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/vck190_resnet18_imagenet.xmodel
mv   ./build/compiled_vek280/vek280_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/vek280_resnet18_imagenet.xmodel
mv ./build/compiled_vck5000/vck5000_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/vck5000_resnet18_imagenet.xmodel
mv         ./build/compiled_v70/v70_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/v70_resnet18_imagenet.xmodel
}


# ===========================================================================
# STEP11: prepare archive for TARGET ZCU102 runtime application for ImageNet
# ===========================================================================
prepare_imagenet_archives() {
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP11] PREPARING IMAGENET ARCHIVE FOR TARGET BOARDS"
echo "----------------------------------------------------------------------------------"
echo " "
if [ -d "./build/target" ]; then
  ### CIFAR10 was running before this and you have to do nothing
  echo "./build/target exists already ..."
else
  ### CIFAR10 was not called before this and you have to build some folders
  echo "./build/target does not exists ..."
  mkdir -p ./build/target
  mkdir -p ./build/target_vck190
  mkdir -p ./build/target_vck5000
  mkdir -p ./build/target_vek280
  mkdir -p ./build/target_zcu102
  mkdir -p ./build/target_v70  
fi
cp -r ./target/imagenet ./build/target/
# zcu102
cp -r ./build/target/imagenet ./build/target_zcu102/
rm -f ./build/target_zcu102/imagenet/vck*_imagenet.xmodel
rm -f ./build/target_zcu102/imagenet/vek*_imagenet.xmodel
# vck190
cp -r ./build/target/imagenet ./build/target_vck190/
rm -f ./build/target_vck190/imagenet/zcu1*_imagenet.xmodel
rm -f ./build/target_vck190/imagenet/vek2*_imagenet.xmodel
rm -f ./build/target_vck190/imagenet/vck5*_imagenet.xmodel
# vek280
cp -r ./build/target/imagenet ./build/target_vek280/
rm -f ./build/target_vek280/imagenet/zcu*_imagenet.xmodel
rm -f ./build/target_vek280/imagenet/vck*_imagenet.xmodel
# vck5000
cp -r ./build/target/imagenet ./build/target_vck5000/
rm -f ./build/target_vck5000/imagenet/zcu1*_imagenet.xmodel
rm -f ./build/target_vck5000/imagenet/vek2*_imagenet.xmodel
rm -f ./build/target_vck5000/imagenet/vck1*_imagenet.xmodel
# v700
cp -r ./build/target/imagenet ./build/target_v70/
rm -f ./build/target_v70/imagenet/zcu1*_imagenet.xmodel
rm -f ./build/target_v70/imagenet/vek2*_imagenet.xmodel
rm -f ./build/target_v70/imagenet/vck*_imagenet.xmodel

# prerare tar files
cd ./build
tar -cvf  target_zcu102.tar  ./target_zcu102 > /dev/null
tar -cvf  target_vck190.tar  ./target_vck190 > /dev/null
tar -cvf  target_vek280.tar  ./target_vek280 > /dev/null
tar -cvf  target_vck5000.tar ./target_vck5000 > /dev/null
tar -cvf  target_v70.tar     ./target_v70 > /dev/null
#rm -r target
cd ..
}

# ===========================================================================
# prepare imagenet test images
# ===========================================================================
prepare_imagenet_test_images(){
  cp ./modelzoo/ImageNet/val_dataset.zip ./target/imagenet
  cd ./target/imagenet/
  rm ./words.txt
  rm ./val.txt
  unzip -o -q val_dataset.zip #unzip forcing overwrite in quiet mode
  cd ../../
}

# ===========================================================================
# remove imagenet test images
# ===========================================================================
remove_imagenet_test_images(){
  cd ./target/imagenet/
  rm -r ./val_dataset
  rm ./words.txt
  rm ./val.txt
  cd ../../
}

# ===========================================================================
# main for ImageNet
# ===========================================================================
# do not change the order of the following commands

main_imagenet(){
  echo " "
  echo " "
  #run_clean_dos2unix            # 1
  prepare_imagenet_test_images
  quantize_resnet50_imagenet    # 7
  quantize_resnet18_imagenet    # 8
  compile_resnet50_imagenet     # 9
  compile_resnet18_imagenet     #10
  ###cross compile the application on target
  ##cd target
  ##source ./imagenet/run_all_imagenet_target.sh compile_resnet
  ##cd ..
  remove_imagenet_test_images
  prepare_imagenet_archives
  echo " "
  echo " "
}


# ===========================================================================
# main for all
# ===========================================================================

# do not change the order of the following commands
main_all(){
  main_cifar10    # 1 to  6
  main_imagenet   # 7 to 11
}


# ===========================================================================
# DO NOT REMOVE THE FOLLOWING LINE
# ===========================================================================

"$@"
