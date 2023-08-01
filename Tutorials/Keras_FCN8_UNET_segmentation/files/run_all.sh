#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


#dos2unix conversion
dos2unixconv(){

for file in $(find $PWD -name "*.h"); do
      sed -i 's/\r//g' ${file}
      echo  ${file}
done
for file in $(find $PWD -name "*.c*"); do
      sed -i 's/\r//g' ${file}
      echo  ${file}
done
for file in $(find $PWD -name "*.sh"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done
}

#clean all
cleanall(){
# clean rpt directory
rm -rf ./rpt
mkdir rpt

#clean ZCU102 xmodel
rm target_zcu102/fcn8/model/*.xmodel target_zcu102/fcn8/model/*.json
rm target_zcu102/fcn8ups/model/*.xmodel target_zcu102/fcn8ups/model/*.json
rm target_zcu102/unet/v2/model/*.xmodel target_zcu102/unet/v2/model/*.json
rm target_zcu102/rpt/*
rm -r target_vck190
rm -r target_zcu104
rm -r target_vek280
rm target_*.tar
mkdir target_zcu102/fcn8
mkdir target_zcu102/fcn8/model
mkdir target_zcu102/fcn8ups
mkdir target_zcu102/fcn8ups/model
mkdir target_zcu102/unet/
mkdir target_zcu102/unet/v2
mkdir target_zcu102/unet/v2/model
}

# run the CNNs

main(){

#missing packages
pip install seaborn

#clean
dos2unixconv
cleanall

#target boards
cp -rf target_zcu102 target_vck190
cp -rf target_zcu102 target_vek280
#cp -rf target_zcu102 target_zcu104

# launch CNNs
source ./run_fcn8.sh
source ./run_fcn8ups.sh
source ./run_unet.sh
}


"$@"
