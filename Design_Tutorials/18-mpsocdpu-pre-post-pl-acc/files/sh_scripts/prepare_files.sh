#!/bin/bash
#/*******************************************************************************
#
# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*******************************************************************************/

: '
#cd VDPU-PRE-POST-PLACC/files # you are supposed to be here

cd ${MPSOCDPU_PRE_POST_PL_ACC}/postproc/hls
unzip data_post.zip
cd ${MPSOCDPU_PRE_POST_PL_ACC}/preproc/hls
unzip data_pre.zip
cd ${MPSOCDPU_PRE_POST_PL_ACC}
'

# **********************************************************************************************
# CLEAN EVERYTHING
# **********************************************************************************************

#echo "CLEANING FILES"
#source ${MPSOCDPU_PRE_POST_PL_ACC}/sh_scripts/clean_files.sh

# **********************************************************************************************
# PRE-PROCESSING PL ACCELERATOR
# **********************************************************************************************

echo "PREPARE PRE_PROC FILES FOR HLS PROJECT with soft links"
ln -nsf ../../common_src/ap_bmp.cpp          ./preproc/hls/src/ap_bmp.cpp
ln -nsf ../../common_src/ap_bmp.h            ./preproc/hls/src/ap_bmp.h
ln -nsf ../../common_src/dpupreproc_main.cpp ./preproc/hls/src/dpupreproc_main.cpp
ln -nsf ../../common_src/dpupreproc_ref.cpp  ./preproc/hls/src/dpupreproc_ref.cpp
ln -nsf ../../common_src/dpupreproc_tb.cpp   ./preproc/hls/src/dpupreproc_tb.cpp
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/hls/src/dpupreproc_tb.h

echo "PREPARE PRE_PROC FILES FOR VITIS HOST with soft links"
ln -nsf ../../common_src/ap_bmp.cpp          ./preproc/vitis/host/ap_bmp.cpp
ln -nsf ../../common_src/ap_bmp.h            ./preproc/vitis/host/ap_bmp.h
ln -nsf ../../common_src/dpupreproc_ref.cpp  ./preproc/vitis/host/dpupreproc_ref.cpp
ln -nsf ../../common_src/dpupreproc_tb.cpp   ./preproc/vitis/host/dpupreproc_tb.cpp
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/vitis/host/dpupreproc_tb.h
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/vitis/host/dpupreproc_tb.h

echo "PREPARE PRE_PROC FILES FOR VITIS KERNELS with soft links"
ln -nsf ../../hls/src/dpupreproc_defines.h   ./preproc/vitis/kernels/dpupreproc_defines.h
ln -nsf ../../hls/src/dpupreproc_vhls.cpp    ./preproc/vitis/kernels/dpupreproc_vhls.cpp


# **********************************************************************************************
# POST-PROCESSING PL ACCELERATOR
# **********************************************************************************************

echo "PREPARE POST_PROC FILES FOR HLS PROJECT with soft links"
ln -nsf ../../common_src/dpupostproc_main.cpp ./postproc/hls/src/dpupostproc_main.cpp
ln -nsf ../../common_src/dpupostproc_ref.cpp  ./postproc/hls/src/dpupostproc_ref.cpp
ln -nsf ../../common_src/dpupostproc_tb.cpp   ./postproc/hls/src/dpupostproc_tb.cpp
ln -nsf ../../common_src/dpupostproc_tb.h     ./postproc/hls/src/dpupostproc_tb.h
ln -nsf ../../common_src/luts.h               ./postproc/hls/src/luts.h

echo "PREPARE POST_PROC FILES FOR VITIS HOST with soft links"
ln -nsf ../../common_src/dpupostproc_ref.cpp  ./postproc/vitis/host/dpupostproc_ref.cpp
ln -nsf ../../common_src/dpupostproc_tb.cpp   ./postproc/vitis/host/dpupostproc_tb.cpp
ln -nsf ../../common_src/dpupostproc_tb.h     ./postproc/vitis/host/dpupostproc_tb.h
ln -nsf ../../common_src/luts.h               ./postproc/vitis/host/luts.h

echo "PREPARE POST_PROC FILES FOR VITIS KERNELS with soft links"
ln -nsf ../../hls/src/dpupostproc_defines.h   ./postproc/vitis/kernels/dpupostproc_defines.h
ln -nsf ../../hls/src/dpupostproc_vhls.cpp    ./postproc/vitis/kernels/dpupostproc_vhls.cpp
ln -nsf ../../common_src/luts.h               ./postproc/vitis/kernels/luts.h


# **********************************************************************************************
# DPU TRD PLUS PL-ACCELERATORS: HOST APPS
# **********************************************************************************************
#cd ${MPSOCDPU_PRE_POST_PL_ACC}/makefile_flow/host_apps
#make clean
#cd ${MPSOCDPU_PRE_POST_PL_ACC}/makefile_flow/ip
#make clean
cd ${MPSOCDPU_PRE_POST_PL_ACC}/makefile_flow/host_apps
make prepare_files
cd ${MPSOCDPU_PRE_POST_PL_ACC}

