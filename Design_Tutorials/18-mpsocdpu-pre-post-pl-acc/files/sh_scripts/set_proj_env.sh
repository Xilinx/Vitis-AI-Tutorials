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
#
# Authors: Peter Schillinger,Daniele Bagni  Xilinx Inc

# set the Vitis/Vivado 2021.1 release
source /tools/Xilinx/Vitis/2021.1/settings64.sh

# working directory
export WRK_DIR=/media/danieleb/DATA

# for Vitis AI
export VITIS_AI_PATH=/media/danieleb/DATA/Vitis-AI-1.4.1

# for the HLS kernels and host applications
export MPSOCDPU_PRE_POST_PL_ACC=${WRK_DIR}/MPSOCDPU-PRE-POST-PL-ACC/files

# for the necessary included files and libraries of host_apps
export TRD_HOME=/media/danieleb/DATA/MPSOCDPU-PRE-POST-PL-ACC/files/dpu_trd

# for ZCU102 common sw package
export VITIS_SYSROOTS=${TRD_HOME}/xilinx-zynqmp-common-v2021.1/sdk/sysroots/cortexa72-cortexa53-xilinx-linux

# for common zcu102 platform
export VITIS_PLATFORM=xilinx_zcu102_base_202110_1
export VITIS_PLATFORM_DIR=${TRD_HOME}/${VITIS_PLATFORM} #xilinx_zcu102_base_202110_1
export VITIS_PLATFORM_PATH=${VITIS_PLATFORM_DIR}/${VITIS_PLATFORM}.xpfm #xilinx_zcu102_base_202110_1.xpfm
export SDX_PLATFORM=${VITIS_PLATFORM_PATH}


#some basic checking
error=0
if [ -d "$TRD_HOME" ]; then
  echo "TRD_HOME=$TRD_HOME exists"
else
  echo "TRD_HOME=$TRD_HOME does not exist"
  error=$((error+1))
fi

if [ -d "$VITIS_AI_PATH" ]; then
  echo "VITIS_AI_PATH=$VITIS_AI_PATH exists"
else
  echo "VITIS_AI_PATH=$VITIS_AI_PATH does not exist"
  error=$((error+1))
fi

if [ -d "$MPSOCDPU_PRE_POST_PL_ACC" ]; then
echo "MPSOCDPU_PRE_POST_PL_ACC=$MPSOCDPU_PRE_POST_PL_ACC exists"
else
  echo "MPSOCDPU_PRE_POST_PL_ACC=$MPSOCDPU_PRE_POST_PL_ACC does not exist"
  error=$((error+1))
fi

if [ -d "$VITIS_SYSROOTS" ]; then
echo "VITIS_SYSROOTS=$VITIS_SYSROOTS exists"
else
  echo "VITIS_SYSROOTS=$VITIS_SYSROOTS does not exist"
  error=$((error+1))
fi

if [ -f "$VITIS_PLATFORM_PATH" ]; then
echo "VITIS_PLATFORM_PATH=$VITIS_PLATFORM_PATH exists"
else
  echo "VITIS_PLATFORM_PATH=$VITIS_PLATFORM_PATH does not exist"
  error=$((error+1))
fi
if [ $error > 0 ];
then
    echo "$error variables not set correctly";
else
    echo "All Variables checked OK";
fi;

rm -rf 0
