#!/usr/bin/env python
# coding: utf-8

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


# **********************************************************************************************
# PRE-PROCESSING PL ACCELERATOR
# **********************************************************************************************

# clean everything
echo "CLEANING HLS PREPROC files"
unlink     ./preproc/hls/src/ap_bmp.cpp
unlink     ./preproc/hls/src/ap_bmp.h
unlink     ./preproc/hls/src/dpupreproc_main.cpp
unlink     ./preproc/hls/src/dpupreproc_ref.cpp
unlink     ./preproc/hls/src/dpupreproc_tb.cpp
unlink     ./preproc/hls/src/dpupreproc_tb.h
rm         ./preproc/hls/src/*~
rm         ./preproc/hls/*~
rm         ./preproc/hls/*.log
rm -f      ./preproc/hls/hls_*m_axi.xo
rm -f      ./preproc/hls/hls_*m_axi.zip
rm -rf     ./preproc/hls/vhls_*_prj
rm -f      ./preproc/hls/data_pre/*.txt


echo "CLEANING VITIS PREPROC files"
unlink       ./preproc/vitis/host/ap_bmp.cpp
unlink       ./preproc/vitis/host/ap_bmp.h
unlink       ./preproc/vitis/host/dpupreproc_ref.cpp
unlink       ./preproc/vitis/host/dpupreproc_tb.cpp
unlink       ./preproc/vitis/host/dpupreproc_tb.h
unlink       ./preproc/vitis/kernels/dpupreproc_vhls.cpp
unlink       ./preproc/vitis/kernels/dpupreproc_defines.h
rm           ./preproc/vitis/*~
rm           ./preproc/vitis/host/*~
rm           ./preproc/vitis/kernels/*~

# **********************************************************************************************
# POST-PROCESSING PL ACCELERATOR
# **********************************************************************************************

# clean everything
echo "CLEANING HLS POSTPROC files"
unlink     ./postproc/hls/src/dpupostproc_main.cpp
unlink     ./postproc/hls/src/dpupostproc_ref.cpp
unlink     ./postproc/hls/src/dpupostproc_tb.cpp
unlink     ./postproc/hls/src/dpupostproc_tb.h
unlink     ./postproc/hls/src/luts.h
rm         ./postproc/hls/src/*~
rm         ./postproc/hls/*~
rm         ./postproc/hls/*.log
rm         ./postproc/hls/data_post/hls_*.bin
rm         ./postproc/hls/data_post/ref_*.bin
rm -rf     ./postproc/hls/vhls_*_prj
rm -rf     ./postproc/hls/hls_*m_axi.xo
rm -rf     ./postproc/hls/hls_*m_axi.zip


echo "CLEANING VITIS POSTPROC files"
unlink       ./postproc/vitis/host/dpupostproc_ref.cpp
unlink       ./postproc/vitis/host/dpupostproc_tb.cpp
unlink       ./postproc/vitis/host/dpupostproc_tb.h
unlink       ./postproc/vitis/host/luts.h
unlink       ./postproc/vitis/kernels/dpupostproc_vhls.cpp
unlink       ./postproc/vitis/kernels/dpupostproc_defines.h
unlink       ./postproc/vitis/kernels/luts.h
rm           ./postproc/vitis/*~
rm           ./postproc/vitis/host/*~
rm           ./postproc/vitis/kernels/*~

# **********************************************************************************************


# **********************************************************************************************
# DPU TRD PLUS PL-ACCELERATORS: HOST APPS
# **********************************************************************************************
echo "CLEANING MAKEFILE FLOW HOST APPS files"
cd ${MPSOCDPU_PRE_POST_PL_ACC}/makefile_flow/host_apps
make clean
echo "CLEANING MAKEFILE FLOW IP files"
cd ${MPSOCDPU_PRE_POST_PL_ACC}/makefile_flow/ip
make clean
echo "CLEANING DPU_TRD files"
cd ${MPSOCDPU_PRE_POST_PL_ACC}/dpu_trd/
make clean
cd ${MPSOCDPU_PRE_POST_PL_ACC}
