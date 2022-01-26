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

# move host_apps files as they are in the original sd-card
mv postproc/data_post              .
mv preproc/data_pre                .
mv pre2dpu2post/data_pre2dpu2post  .
mv postproc/host_postproc_xrt      .
mv preproc/host_preproc_xrt        .
mv pre2dpu2post/host_pre2dpu2post_xrt  .
rm -r postproc pre*

echo " "
echo "PREPARING ENVIRONMENTAL VARIABLES"
echo " "
#cd /mnt/sd-mmcblk0p1/
export LD_LIBRARY_PATH=/mnt/sd-mmcblk0p1/app/samples/lib
export XLNX_VART_FIRMWARE=/mnt/sd-mmcblk0p1/dpu.xclbin
ls -l


echo " "
echo "RUN STANDALONE PL PREPROC"
echo " "
./host_preproc_xrt /media/sd-mmcblk0p1/dpu.xclbin
#check results
cmp ./data_pre/inp_000_ref.bmp ./data_pre/inp_000_out.bmp

echo " "
echo "RUN STANDALONE PL POSTPROC"
echo " "
./host_postproc_xrt /media/sd-mmcblk0p1/dpu.xclbin
#check results
cmp ./data_post/arm_ref_index.bin ./data_post/pl_hls_index.bin

echo " "
echo "RUN CHAIN OF PRE+DPU+POST"
echo " "
./host_pre2dpu2post_xrt ./model/zcu102_unet2.xmodel ./data_pre2dpu2post/dataset1/img_test/ 1 1 1
mv *.png  ./data_pre2dpu2post/
# check results
cmp ./data_pre2dpu2post/hw_out_000.png          ./data_pre2dpu2post/ref_out_000.png
cmp ./data_pre2dpu2post/hw_out_001.png          ./data_pre2dpu2post/ref_out_001.png
cmp ./data_pre2dpu2post/hw_out_002.png          ./data_pre2dpu2post/ref_out_002.png
cmp ./data_pre2dpu2post/hw_preproc_out_000.png  ./data_pre2dpu2post/sw_preproc_out_000.png
cmp ./data_pre2dpu2post/hw_preproc_out_001.png  ./data_pre2dpu2post/sw_preproc_out_001.png
cmp ./data_pre2dpu2post/hw_preproc_out_002.png  ./data_pre2dpu2post/sw_preproc_out_002.png
cmp ./data_pre2dpu2post/post_uint8_out_idx.bin  ./data_pre2dpu2post/post_uint8_ref_idx.bin
