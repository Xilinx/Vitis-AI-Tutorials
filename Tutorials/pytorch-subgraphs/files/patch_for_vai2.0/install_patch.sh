#!/bin/bash
# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#enter into the Vitis AI 2.0 GPU docker image

sudo env PATH=/opt/vitis_ai/conda/bin:$PATH 
CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-pytorch conda install ./xcompiler-2.0.1-py36hea4fdf2_32.tar.bz2
CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-pytorch conda install ./xcompiler-2.0.1-py37hea4fdf2_32.tar.bz2


: '
# once installed the patches, remember to save the docker image to make the change permanent, see the following:

$ sudo docker ps -l
CONTAINER ID   IMAGE                            
17573557f30b   xilinx/vitis-ai-gpu:2.0.0.1103   
$ sudo docker commit -m"patch" 17573557f30b   xilinx/vitis-ai-gpu:2.0.0.1103 
'
