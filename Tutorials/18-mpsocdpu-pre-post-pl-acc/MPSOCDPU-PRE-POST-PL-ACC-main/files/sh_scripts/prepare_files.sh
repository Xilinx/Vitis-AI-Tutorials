#!/usr/bin/env python
# coding: utf-8

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


: '
cd VDPU-PRE-POST-PLACC/files # you are supposed to be here

cd ${MPSOCDPU_PRE_POST_PL_ACC}/postproc/hls
unzip data_post.zip
cd ${MPSOCDPU_PRE_POST_PL_ACC}/preproc/hls
unzip data_pre.zip
cd ${MPSOCDPU_PRE_POST_PL_ACC}
'

# **********************************************************************************************
# PRE-PROCESSING PL ACCELERATOR
# **********************************************************************************************
echo " "
echo "PREPARE PRE_PROC FILES FOR HLS PROJECT with soft links"
echo " "
ln -nsf ../../common_src/ap_bmp.cpp          ./preproc/hls/src/ap_bmp.cpp
ln -nsf ../../common_src/ap_bmp.h            ./preproc/hls/src/ap_bmp.h
ln -nsf ../../common_src/dpupreproc_main.cpp ./preproc/hls/src/dpupreproc_main.cpp
ln -nsf ../../common_src/dpupreproc_ref.cpp  ./preproc/hls/src/dpupreproc_ref.cpp
ln -nsf ../../common_src/dpupreproc_tb.cpp   ./preproc/hls/src/dpupreproc_tb.cpp
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/hls/src/dpupreproc_tb.h

echo " "
echo "PREPARE PRE_PROC FILES FOR VITIS HOST with soft links"
echo " "
ln -nsf ../../common_src/ap_bmp.cpp          ./preproc/vitis/host/ap_bmp.cpp
ln -nsf ../../common_src/ap_bmp.h            ./preproc/vitis/host/ap_bmp.h
ln -nsf ../../common_src/dpupreproc_ref.cpp  ./preproc/vitis/host/dpupreproc_ref.cpp
ln -nsf ../../common_src/dpupreproc_tb.cpp   ./preproc/vitis/host/dpupreproc_tb.cpp
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/vitis/host/dpupreproc_tb.h
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/vitis/host/dpupreproc_tb.h

echo " "
echo "PREPARE PRE_PROC FILES FOR VITIS KERNELS with soft links"
echo " "
ln -nsf ../../hls/src/dpupreproc_defines.h   ./preproc/vitis/kernels/dpupreproc_defines.h
ln -nsf ../../hls/src/dpupreproc_vhls.cpp    ./preproc/vitis/kernels/dpupreproc_vhls.cpp

# **********************************************************************************************
# POST-PROCESSING PL ACCELERATOR
# **********************************************************************************************
echo " "
echo "PREPARE POST_PROC FILES FOR HLS PROJECT with soft links"
echo " "
ln -nsf ../../common_src/dpupostproc_main.cpp ./postproc/hls/src/dpupostproc_main.cpp
ln -nsf ../../common_src/dpupostproc_ref.cpp  ./postproc/hls/src/dpupostproc_ref.cpp
ln -nsf ../../common_src/dpupostproc_tb.cpp   ./postproc/hls/src/dpupostproc_tb.cpp
ln -nsf ../../common_src/dpupostproc_tb.h     ./postproc/hls/src/dpupostproc_tb.h
ln -nsf ../../common_src/luts.h               ./postproc/hls/src/luts.h

echo " "
echo "PREPARE POST_PROC FILES FOR VITIS HOST with soft links"
echo " "
ln -nsf ../../common_src/dpupostproc_ref.cpp  ./postproc/vitis/host/dpupostproc_ref.cpp
ln -nsf ../../common_src/dpupostproc_tb.cpp   ./postproc/vitis/host/dpupostproc_tb.cpp
ln -nsf ../../common_src/dpupostproc_tb.h     ./postproc/vitis/host/dpupostproc_tb.h
ln -nsf ../../common_src/luts.h               ./postproc/vitis/host/luts.h

echo " "
echo "PREPARE POST_PROC FILES FOR VITIS KERNELS with soft links"
echo " "
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
cd ${WRK_DIR}/${TUTORIAL}/files/makefile_flow/host_apps
make prepare_files
cd ${WRK_DIR}/${TUTORIAL}/files
