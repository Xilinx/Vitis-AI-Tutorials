#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023




# ===================================================================
# dos2unix to all files
cd $WRK_DIR/$TUTORIAL/files
for file in $(find . -name "*.sh" ); do dos2unix ${file}; done
for file in $(find . -name "*.tcl"); do dos2unix ${file}; done
for file in $(find . -name "*.h"  ); do dos2unix ${file}; done
for file in $(find . -name "*.c*" ); do dos2unix ${file}; done
for file in $(find . -name "*.cfg" );    do dos2unix ${file}; done
for file in $(find . -name "*akefile" ); do dos2unix ${file}; done
for file in $(find . -name ".fuse_hidden*" ); do rm -f ${file}; done


# ===================================================================
# set the project environmental variables
source ./sh_scripts/set_proj_env_2022v2.sh


# ===================================================================
# run standalone HLS projects
bash  -x ./sh_scripts/run_hls_projects.sh

# ===================================================================
# run makefile-base flow
cd makefile_flow
bash -x ./run_makefile_flow.sh
cd ..


# ===================================================================
# build the MPSOC DPU TRD
cd $WRK_DIR/$TUTORIAL/files/dpu_trd
DIR=$WRK_DIR/$TUTORIAL/files/dpu_trd/ip/dpu_ip/DPUCZDX8G_v4_1_0
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "running makefile now ..."
  make all
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo " "
  echo "Error: ${DIR} not found!"
  echo "Now manually installing DPUCZDX8G_v4_1_0!"
  echo " "
  make prep
fi
cd $WRK_DIR/$TUTORIAL/files
# ======================================================================
