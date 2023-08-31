#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

#usage: source imagenet/run_all_imagenet_target.sh main <TARGET_BOARD (i.e. vck190)>

TARGET=$2

#clean
clean_imagenet(){
echo " "
echo "clean imagenet"
echo " "
cd imagenet
rm -rf val_dataset
rm -f  get_dpu_fps
rm -f  resnet_imagenet
rm -rf rpt
rm -f  *.txt
rm -f  *.log
mkdir -p rpt
cd ..
cp ./cifar10/get_dpu_fps ./imagenet
}

# compile CNN application
compile_resnet(){
echo " "
echo "compile imagenet"
echo " "
cd ./imagenet/code_resnet50
echo "PWD1 = " $PWD
bash -x ./build_resnet50.sh
mv ./code_resnet50 ../resnet_imagenet      # change name of the application
#bash -x ./build_get_dpu_fps.sh
#mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build imagenet test images
build_images_imagenet(){
echo " "
echo "build imagenet test images"
echo " "
cd imagenet
unzip val_dataset.zip > /dev/null
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the resnet50 classification using VART C++ APIs
run_cnn_resnet50(){
echo " "
echo "run resnet50 CNN"
echo " "
cd imagenet
./resnet_imagenet ./${TARGET}_resnet50_imagenet.xmodel ./val_dataset/ ./words.txt 1 2>&1 | tee ./rpt/predictions_resnet50_imagenet.log
# check DPU prediction accuracy
bash -x ./imagenet_performance.sh ${TARGET}
echo "PWD4 = " $PWD
cd ..
}

# now run the resnet18 classification using VART C++ APIs
run_cnn_resnet18(){
echo " "
echo "run resnet18 CNN"
echo " "
cd imagenet
./resnet_imagenet ./${TARGET}_resnet18_imagenet.xmodel ./val_dataset/ ./words.txt 0 2>&1 | tee ./rpt/predictions_resnet18_imagenet.log
# check DPU prediction accuracy
bash -x ./imagenet_performance.sh ${TARGET}
echo "PWD4 = " $PWD
cd ..
}


# profile the resnet18 classification
profile_resnet18(){
    cd imagenet
    vaitrace --fine_grained --txt --va ./resnet_imagenet ${TARGET}_resnet18_imagenet.xmodel ./val_dataset/ ./words.txt 0
    cd ..
}


#remove images
end_imagenet(){
echo " "
echo "end of imagenet"
echo " "
cd imagenet
rm -rf val_dataset
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


main()
{
    clean_imagenet
    compile_resnet
    build_images_imagenet
    run_cnn_resnet50
    run_cnn_resnet18
    #profile_resnet18
    end_imagenet
}



"$@"
