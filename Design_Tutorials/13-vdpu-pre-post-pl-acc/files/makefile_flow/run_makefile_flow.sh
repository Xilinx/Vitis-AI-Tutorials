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

#change the following two directories according to your needs
export VDPU_PRE_POST_PL_ACC=/media/danieleb/DATA/ZF/new_VDPU-PRE-POST-PL-ACC/files
export DB_FATHER_PATH=/media/danieleb/DATA/ZF/ZF_ProAI-main/NEW_ZF_PACKAGE_FINAL/


cd host_apps
make clean
make all
cd ../ip
make clean
make all
