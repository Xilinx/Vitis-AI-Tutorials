#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

TARGET=$2
#vek280


#clean
clean_vcor(){
echo " "
echo "[TARGET INFO] clean vcor folders"
echo " "
cd vcor
rm -rf test
rm -f *~
rm -f  run_cnn cnn* get_dpu_fps *.txt
rm -rf rpt
rm -f  *.txt
rm -f  *.log
mkdir -p rpt
cd ..
}

# compile CNN application
compile_vcor(){
  echo " "
  echo "[TARGET INFO] compile vcor app"
  echo " "
cd vcor/code
echo "PWD1 = " $PWD
bash -x ./build_app.sh
mv code ../cnn_resnet18_vcor # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build vcor test images
test_images_vcor(){
  echo " "
  echo "[TARGET INFO] build test images"
  echo " "

cd vcor
bash ./build_vcor_test.sh
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the vcor classification using VART C++ APIs
run_cnn_vcor(){
  echo " "
  echo "[TARGET INFO] run CNN "
  echo " "

cd vcor
./cnn_resnet18_vcor ./${TARGET}_train_resnet18_vcor.xmodel ./test/ ./vcor_labels.dat | tee ./rpt/predictions_vcor_resnet18.log
# check DPU prediction accuracy
bash -x ./vcor_performance.sh ${TARGET}
echo "PWD4 = " $PWD
cd ..
}

#remove images
end_vcor(){
  echo " "
  echo "[TARGET INFO] end of vcor"
  echo " "

cd vcor
rm -rf test
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


main()
{
    clean_vcor
    compile_vcor
    test_images_vcor
    run_cnn_vcor
    end_vcor
}




"$@"
