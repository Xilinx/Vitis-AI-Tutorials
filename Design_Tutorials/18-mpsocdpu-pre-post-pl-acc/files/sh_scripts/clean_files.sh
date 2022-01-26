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
cd ${MPSOCDPU_PRE_POST_PL_ACC}
