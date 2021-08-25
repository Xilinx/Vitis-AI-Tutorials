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


CNN=LeNet

cp ./src/fps_tf_main.cc ./tf_main.cc
make clean
make
mv LeNet fps_LeNet

echo " "
echo "./LeNet 1"
./fps_LeNet 1
echo " "
echo "./LeNet 2"
./fps_LeNet 2
echo " "
echo "./LeNet 3"
./fps_LeNet 3
echo " "
echo "./LeNet 4"
./fps_LeNet 4
echo " "
echo "./LeNet 5"
./fps_LeNet 5
echo " "
echo "./LeNet 6"
./fps_LeNet 6
echo " "
echo "./LeNet 7"
./fps_LeNet 7
echo " "
echo "./LeNet 8"
./fps_LeNet 8
echo " "
