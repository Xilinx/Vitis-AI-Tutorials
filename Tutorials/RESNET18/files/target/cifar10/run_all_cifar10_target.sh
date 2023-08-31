#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  10 Aug. 2023


TARGET=$2
#vek280


#clean
clean_cif10(){
echo " "
echo "clean cifar10"
echo " "
cd cifar10
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
compile_cif10(){
echo " "
echo "compile cifar10"
echo " "
cd cifar10/code
echo "PWD1 = " $PWD
bash -x ./build_app.sh
mv code ../cnn_resnet18_cifar10 # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build cifar10 test images
test_images_cif10(){
echo " "
echo "build test images for cifar10"
echo " "
cd cifar10
bash ./build_cifar10_test.sh
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the cifar10 classification with 4 CNNs using VART C++ APIs
run_cnn_cif10(){
echo " "
echo " run cifar10 CNN"
echo " "
cd cifar10
./cnn_resnet18_cifar10 ./${TARGET}_train1_resnet18_cifar10.xmodel ./test/ ./cifar10_labels.dat | tee ./rpt/predictions_cifar10_resnet18.log
# check DPU prediction accuracy
bash -x ./cifar10_performance.sh ${TARGET}
echo "PWD4 = " $PWD
cd ..
}

#remove images
end_cif10(){
echo " "
echo "end of cifar10"
echo " "
cd cifar10
rm -rf test
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


main()
{
    clean_cif10
    compile_cif10
    test_images_cif10
    run_cnn_cif10
    end_cif10
}




"$@"
