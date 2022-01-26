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

# set vitis environment
#source ./set_proj_env.sh

# clean project files
bash  ${MPSOCDPU_PRE_POST_PL_ACC}/sh_scripts/clean_files.sh
rm -r ${MPSOCDPU_PRE_POST_PL_ACC}/preproc/hls/vhls*
rm -r ${MPSOCDPU_PRE_POST_PL_ACC}/postproc/hls/vhls*

# prepare files with soflinks
bash ${MPSOCDPU_PRE_POST_PL_ACC}/sh_scripts/prepare_files.sh

# run HLS pre-processor TB
cd ${MPSOCDPU_PRE_POST_PL_ACC}/preproc/hls
vitis_hls -f hls_script.tcl

# run HLS post-processor TB
cd ${MPSOCDPU_PRE_POST_PL_ACC}/postproc/hls
vitis_hls -f hls_script.tcl


# go back to main directory
cd ${MPSOCDPU_PRE_POST_PL_ACC}

