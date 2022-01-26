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



# set the project environmental variables
source ./sh_scripts/set_proj_env.sh

# run standalone HLS projects
bash  -x ./sh_scripts/run_hls_projects.sh

# run makefile-base flow
cd makefile_flow
bash -x ./run_makefile_flow.sh
cd ..


# build the MPSOC DPU TRD
cd dpu_trd
make all

