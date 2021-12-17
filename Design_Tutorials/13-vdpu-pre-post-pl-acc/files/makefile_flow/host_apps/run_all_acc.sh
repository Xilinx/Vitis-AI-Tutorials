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


#echo " "
#echo "PREPARING FOLDERS"
#echo " "
#mkdir host_apps
#mv preproc postproc pre2post ./host_apps
#cd host_apps
echo " "
echo "RUN STANDALONE PL PREPROC"
echo " "
cd preproc
./host_preproc_xrt  /mnt/sd-mmcblk0p1/dpu.xclbin
#check results
cmp ./data_pre/testing_0_1920x832_ref.bmp ./data_pre/testing_0_1920x832_out.bmp
rm -r src Makefile # clean folders
echo " "
echo "RUN STANDALONE PL POSTPROC"
echo " "
cd ../postproc
./host_postproc_xrt /mnt/sd-mmcblk0p1/dpu.xclbin
#check results
cmp ./data_post/arm_ref_index.bin ./data_post/pl_hls_index.bin
rm -r src Makefile # clean folders
echo " "
echo "RUN KERNELS CHAIN OF PRE + DPU + POST"
echo " "
cd ../pre2post
./pre2post ./model/fcn8.xmodel ./data_pre2post/dataset1/img_test/ 1 1 1
mv *.png  ./data_pre2post/
rm -r src Makefile # clean folders
#check results
cmp ./data_pre2post/hw_out_000.png ./data_pre2post/ref_out_000.png
cmp ./data_pre2post/hw_out_001.png ./data_pre2post/ref_out_001.png
cmp ./data_pre2post/hw_out_002.png ./data_pre2post/ref_out_002.png
cmp ./data_pre2post/hw_preproc_out_000.png ./data_pre2post/sw_preproc_out_000.png
cmp ./data_pre2post/hw_preproc_out_001.png ./data_pre2post/sw_preproc_out_001.png
cmp ./data_pre2post/hw_preproc_out_002.png ./data_pre2post/sw_preproc_out_002.png
cmp ./data_pre2post/post_uint8_out_idx.bin ./data_pre2post/post_uint8_ref_idx.bin
cd ..
