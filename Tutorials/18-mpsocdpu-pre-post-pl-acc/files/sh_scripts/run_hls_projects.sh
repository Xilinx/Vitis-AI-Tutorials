#!/usr/bin/env python
# coding: utf-8

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


# ===================================================================
# clean project files
bash  ${WRK_DIR}/${TUTORIAL}/files/sh_scripts/clean_files.sh
rm -r ${WRK_DIR}/${TUTORIAL}/files/preproc/hls/vhls*
rm -r ${WRK_DIR}/${TUTORIAL}/files/postproc/hls/vhls*

# ===================================================================
# prepare files with soflinks
bash ${WRK_DIR}/${TUTORIAL}/files/sh_scripts/prepare_files.sh

# ===================================================================
# run HLS pre-processor TB
cd ${WRK_DIR}/${TUTORIAL}/files/preproc/hls
vitis_hls -f hls_script.tcl

# ===================================================================
# run HLS post-processor TB
cd ${WRK_DIR}/${TUTORIAL}/files/postproc/hls
vitis_hls -f hls_script.tcl

# ===================================================================
# go back to main directory
cd ${WRK_DIR}/${TUTORIAL}/files
