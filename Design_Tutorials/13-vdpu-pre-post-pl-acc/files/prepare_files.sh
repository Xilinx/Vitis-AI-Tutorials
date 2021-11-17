#!/bin/bash

#*************************************************************************************
# Copyright 2020 Xilinx Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************/


# **********************************************************************************************
cd preproc/hls/data_pre
ln -nsf im2_1920x832.bmp ./testing_0_1920x832.bmp
cd ../../../

# **********************************************************************************************
# PRE-PROCESSING PL ACCELERATOR
# **********************************************************************************************

# clean everything
echo "CLEANING HLS files"
unlink     ./preproc/hls/src/ap_bmp.cpp
unlink     ./preproc/hls/src/ap_bmp.h
unlink     ./preproc/hls/src/dpupreproc_main.cpp
unlink     ./preproc/hls/src/dpupreproc_ref.cpp
unlink     ./preproc/hls/src/dpupreproc_tb.cpp
unlink     ./preproc/hls/src/dpupreproc_tb.h
unlink     ./preproc/hls/src/dpupreproc_vhls.cpp
rm         ./preproc/hls/src/*~
rm         ./preproc/hls/*~
rm         ./preproc/hls/*.log
echo "PREPARE PRE_PROC FILES FOR HLS PROJECT with soft links"
ln -nsf ../../common_src/ap_bmp.cpp          ./preproc/hls/src/ap_bmp.cpp
ln -nsf ../../common_src/ap_bmp.h            ./preproc/hls/src/ap_bmp.h
ln -nsf ../../common_src/dpupreproc_main.cpp ./preproc/hls/src/dpupreproc_main.cpp
ln -nsf ../../common_src/dpupreproc_ref.cpp  ./preproc/hls/src/dpupreproc_ref.cpp
ln -nsf ../../common_src/dpupreproc_tb.cpp   ./preproc/hls/src/dpupreproc_tb.cpp
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/hls/src/dpupreproc_tb.h
ln -nsf ../../common_src/dpupreproc_vhls.cpp ./preproc/hls/src/dpupreproc_vhls.cpp


echo "CLEANING VITIS files"
unlink       ./preproc/vitis/host/ap_bmp.cpp
unlink       ./preproc/vitis/host/ap_bmp.h
unlink       ./preproc/vitis/host/dpupreproc_ref.cpp
unlink       ./preproc/vitis/host/dpupreproc_tb.cpp
unlink       ./preproc/vitis/host/dpupreproc_tb.h
unlink       ./preproc/vitis/kernels/dpupreproc_vhls.cpp
rm           ./preproc/vitis/*~
rm           ./preproc/vitis/host/*~
rm           ./preproc/vitis/kernels/*~
echo "PREPARE PRE_PROC FILES FOR VITIS HOST with soft links"
ln -nsf ../../common_src/ap_bmp.cpp          ./preproc/vitis/host/ap_bmp.cpp
ln -nsf ../../common_src/ap_bmp.h            ./preproc/vitis/host/ap_bmp.h
ln -nsf ../../common_src/dpupreproc_ref.cpp  ./preproc/vitis/host/dpupreproc_ref.cpp
ln -nsf ../../common_src/dpupreproc_tb.cpp   ./preproc/vitis/host/dpupreproc_tb.cpp
ln -nsf ../../common_src/dpupreproc_tb.h     ./preproc/vitis/host/dpupreproc_tb.h
echo "PREPARE PRE_PROC FILES FOR VITIS KERNELS with soft links"
ln -nsf ../../common_src/dpupreproc_vhls.cpp ./preproc/vitis/kernels/dpupreproc_vhls.cpp



# **********************************************************************************************
# POST-PROCESSING PL ACCELERATOR
# **********************************************************************************************

# clean everything
echo "CLEANING HLS files"
unlink     ./postproc/hls/src/dpupostproc_main.cpp
unlink     ./postproc/hls/src/dpupostproc_ref.cpp
unlink     ./postproc/hls/src/dpupostproc_tb.cpp
unlink     ./postproc/hls/src/dpupostproc_tb.h
unlink     ./postproc/hls/src/dpupostproc_vhls.cpp
rm         ./postproc/hls/src/*~
rm         ./postproc/hls/*~
rm         ./postproc/hls/*.log
echo "PREPARE PRE_PROC FILES FOR HLS PROJECT with soft links"
ln -nsf ../../common_src/dpupostproc_main.cpp ./postproc/hls/src/dpupostproc_main.cpp
ln -nsf ../../common_src/dpupostproc_ref.cpp  ./postproc/hls/src/dpupostproc_ref.cpp
ln -nsf ../../common_src/dpupostproc_tb.cpp   ./postproc/hls/src/dpupostproc_tb.cpp
ln -nsf ../../common_src/dpupostproc_tb.h     ./postproc/hls/src/dpupostproc_tb.h
ln -nsf ../../common_src/dpupostproc_vhls.cpp ./postproc/hls/src/dpupostproc_vhls.cpp
ln -nsf ../../common_src/lut_exp.h            ./postproc/hls/src/lut_exp.h

echo "CLEANING VITIS files"
unlink       ./postproc/vitis/host/dpupostproc_ref.cpp
unlink       ./postproc/vitis/host/dpupostproc_tb.cpp
unlink       ./postproc/vitis/host/dpupostproc_tb.h
unlink       ./postproc/vitis/kernels/dpupostproc_vhls.cpp
unlink       ./postproc/vitis/kernels/lut_exp.h
rm           ./postproc/vitis/*~
rm           ./postproc/vitis/host/*~
rm           ./postproc/vitis/kernels/*~
echo "PREPARE PRE_PROC FILES FOR VITIS HOST with soft links"
ln -nsf ../../common_src/dpupostproc_ref.cpp  ./postproc/vitis/host/dpupostproc_ref.cpp
ln -nsf ../../common_src/dpupostproc_tb.cpp   ./postproc/vitis/host/dpupostproc_tb.cpp
ln -nsf ../../common_src/dpupostproc_tb.h     ./postproc/vitis/host/dpupostproc_tb.h
echo "PREPARE PRE_PROC FILES FOR VITIS KERNELS with soft links"
ln -nsf ../../common_src/dpupostproc_vhls.cpp ./postproc/vitis/kernels/dpupostproc_vhls.cpp
ln -nsf ../../common_src/lut_exp.h            ./postproc/vitis/kernels/lut_exp.h


# **********************************************************************************************
