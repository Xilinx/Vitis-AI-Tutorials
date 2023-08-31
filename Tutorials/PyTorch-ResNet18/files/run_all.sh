#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

## Author: Daniele Bagni, AMD/Xilinx Inc

## date 11 Aug 2023

# REMEMBER THAT $1 is the "main" routine
LOG_FILENAME=$2
MODEL_NAME=$3


echo " "
echo "==========================================================================="
echo "WARNING: "
echo "  'run_all.sh' MUST ALWAYS BE LAUNCHED BELOW THE 'files' FOLDER LEVEL "
echo "  (SAME LEVEL OF 'scripts' AND 'target' FOLDER)                       "
echo "  AS IT APPLIES RELATIVE PATH AND NOT ABSOLUTE PATHS                  "
echo "==========================================================================="
echo " "




# ===========================================================================
# STEP1: clean and dos2unix
# ===========================================================================
clean_dos2unix(){
source ./scripts/clean_all.sh
}

# ===========================================================================
# STEP2: Prepare VCoR Dataset
# ===========================================================================
vcor_dataset(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP2]  CREATING VCoR DATASET OF IMAGES"
echo "----------------------------------------------------------------------------------"
echo " "
# unzip VoCR  dataset archive
if [ -f "./archive.zip" ]
then
echo "VoCR Dataset zip archive file is found"
unzip ./archive.zip -d ./build/data/vcor &> /dev/null
else
echo "ERROR: VCoR Dataset zip archive file is NOT found"
fi

}

# ===========================================================================
# STEP3: Train and Test ResNet18 CNNs on VCoR
# ===========================================================================
vcor_training(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3] Unzip ResNet18 CNN"
echo "----------------------------------------------------------------------------------"
echo " "
# unzip ResNet18 model zoo archive
if [ -f "./pt_vehicle-color-classification_3.5.zip" ]
then
  echo "pt_vehicle-color-classification_3.5.zip file is found"
  if [ ! -d "./pt_vehicle-color-classification_3.5" ]
  then
    unzip ./pt_vehicle-color-classification_3.5.zip
  fi
  cd pt_vehicle-color-classification_3.5
  # clean some files/folders
  rm -rf code data *.md *.txt *.sh
  cd ..
else
  echo "ERROR: pt_vehicle-color-classification_3.5.zip file is NOT found"
fi
# floating point model training
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3A] VCoR TRAINING"
echo "----------------------------------------------------------------------------------"
echo " "
bash -x ./scripts/run_train.sh
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3B] VCoR TESTING"
echo "----------------------------------------------------------------------------------"
echo " "
bash -x ./scripts/run_test.sh
}


# ===========================================================================
# STEP4: Vitis AI Quantization of ResNet18 on VCoR
# ===========================================================================
vcor_quantize_resnet18(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP4] QUANTIZE VCoR TRAINED CNN"
echo "----------------------------------------------------------------------------------"
bash -x ./scripts/run_quant.sh
}

# ===========================================================================
# STEP5: Vitis AI Compile ResNet18 VCoR for Target Board
# ===========================================================================
vcor_compile_resnet18(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP5] COMPILE VCoR QUANTIZED CNN"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  ResNet_0_int.xmodel
source ./scripts/run_compile.sh vck190  ResNet_0_int.xmodel
source ./scripts/run_compile.sh vek280  ResNet_0_int.xmodel
source ./scripts/run_compile.sh vck5000 ResNet_0_int.xmodel
source ./scripts/run_compile.sh v70     ResNet_0_int.xmodel
mv   ./build/compiled_zcu102/zcu102_ResNet_0_int.xmodel.xmodel  ./target/vcor/zcu102_train_resnet18_vcor.xmodel
mv   ./build/compiled_vck190/vck190_ResNet_0_int.xmodel.xmodel  ./target/vcor/vck190_train_resnet18_vcor.xmodel
mv   ./build/compiled_vek280/vek280_ResNet_0_int.xmodel.xmodel  ./target/vcor/vek280_train_resnet18_vcor.xmodel
mv ./build/compiled_vck5000/vck5000_ResNet_0_int.xmodel.xmodel  ./target/vcor/vck5000_train_resnet18_vcor.xmodel
mv ./build/compiled_v70/v70_ResNet_0_int.xmodel.xmodel          ./target/vcor/v70_train_resnet18_vcor.xmodel
}

# ===========================================================================
# STEP6: prepare archive for TARGET ZCU102 runtime application for VCoR
# ===========================================================================
vcor_prepare_archives() {
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP6] PREPARING VCoR ARCHIVE FOR TARGET BOARDS"
echo "----------------------------------------------------------------------------------"
echo " "
cp -r target       ./build
cd ./build/data/vcor
tar -cvf test_jpg.tar ./test > /dev/null
cp test_jpg.tar       ../../../build/target/vcor/
rm test_jpg.tar
cd ../../../
#prepare test images from jpg to png
cd ./build/target/vcor/
tar -xvf test_jpg.tar
cd ../../../
python ./code/generate_target_test_images.py
cd ./build/target/vcor/
tar -cvf test.tar ./test > /dev/null
rm -f  ./test_jpg.tar
rm -rf ./test
cd ../../../
#rm -rf ./build/target/imagenet #unuseful at the moment
# zcu102
cp -r ./build/target/  ./build/target_zcu102  > /dev/null
rm -f ./build/target_zcu102/vcor/vck*_vcor.xmodel
rm -f ./build/target_zcu102/vcor/vek*_vcor.xmodel
rm -f ./build/target_zcu102/vcor/v70*_vcor.xmodel
# vck190
cp -r ./build/target/  ./build/target_vck190  > /dev/null
rm -f ./build/target_vck190/vcor/zcu1*_vcor.xmodel
rm -f ./build/target_vck190/vcor/vek2*_vcor.xmodel
rm -f ./build/target_vck190/vcor/vck5*_vcor.xmodel
rm -f ./build/target_vck190/vcor/v70*_vcor.xmodel
# vek280
cp -r ./build/target   ./build/target_vek280  > /dev/null
rm -f ./build/target_vek280/vcor/zcu*_vcor.xmodel
rm -f ./build/target_vek280/vcor/vck*_vcor.xmodel
rm -f ./build/target_vek280/vcor/v70*_vcor.xmodel
# vck5000
cp -r ./build/target/  ./build/target_vck5000  > /dev/null
rm -f ./build/target_vck5000/vcor/zcu1*_vcor.xmodel
rm -f ./build/target_vck5000/vcor/vek2*_vcor.xmodel
rm -f ./build/target_vck5000/vcor/vck1*_vcor.xmodel
rm -f ./build/target_vck5000/vcor/v70*_vcor.xmodel
# v70
cp -r ./build/target/  ./build/target_v70  > /dev/null
rm -f ./build/target_v70/vcor/zcu1*_vcor.xmodel
rm -f ./build/target_v70/vcor/vek2*_vcor.xmodel
rm -f ./build/target_v70/vcor/vck*_vcor.xmodel
# built tar files
cd ./build
tar -cvf ./target_vek280.tar  ./target_vek280
tar -cvf ./target_vck190.tar  ./target_vck190
tar -cvf ./target_zcu102.tar  ./target_zcu102
tar -cvf ./target_v70.tar     ./target_v70
tar -cvf ./target_vck5000.tar ./target_vck5000
cd ..
}

# ===========================================================================
# main for VCoR
# ===========================================================================
# do not change the order of the following commands

main_vcor(){
  echo " "
  echo " "
  vcor_dataset            # 2
  vcor_training           # 3
  vcor_quantize_resnet18  # 4
  vcor_compile_resnet18   # 5
  ###cross compile the application on target
  ##cd target
  ##source ./vcor/run_all_vcor_target.sh compile_cif10
  ##cd ..
  vcor_prepare_archives > /dev/null  # 6
  echo " "
  echo " "
}


# ===========================================================================
# main for all
# ===========================================================================

# do not change the order of the following commands
main_all(){
    pip install randaugment
    pip install torchsummary
    clean_dos2unix
    main_vcor
}


# ===========================================================================
# DO NOT REMOVE THE FOLLOWING LINE
# ===========================================================================

"$@"
