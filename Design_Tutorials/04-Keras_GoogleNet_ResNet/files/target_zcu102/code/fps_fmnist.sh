#!/bin/bash

## © Copyright (C) 2016-2020 Xilinx, Inc
##
## Licensed under the Apache License, Version 2.0 (the "License"). You may
## not use this file except in compliance with the License. A copy of the
## License is located at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.

echo " "
echo " FMNIST LEnet fps"
echo " "
./get_dpu_fps ./fmnist/LeNet/LeNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/LeNet/LeNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/LeNet/LeNet.xmodel  6 10000  # 6 threads

echo " "
echo " FMNIST miniVGGnet fps"
echo " "
./get_dpu_fps ./fmnist/miniVggNet/miniVggNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/miniVggNet/miniVggNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/miniVggNet/miniVggNet.xmodel  6 10000  # 6 threads


echo " "
echo " FMNIST miniGOOGLEnet fps"
echo " "
./get_dpu_fps ./fmnist/miniGoogleNet/miniGoogleNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/miniGoogleNet/miniGoogleNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/miniGoogleNet/miniGoogleNet.xmodel  6 10000  # 6 threads

echo " "
echo " FMNIST miniRESnet fps"
echo " "
./get_dpu_fps ./fmnist/miniResNet/miniResNet.xmodel  1 10000  # 1 thread
./get_dpu_fps ./fmnist/miniResNet/miniResNet.xmodel  5 10000  # 5 threads
./get_dpu_fps ./fmnist/miniResNet/miniResNet.xmodel  6 10000  # 6 threads
