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


# run VART with multi-threading on baseline CNN
echo " "
echo "BASELINE CNN                     MULTITHREADING"
echo " "
./get_dpu_fps ./baseline/model/arm64_4096/alexnetBNnoLRN.xmodel 1 1000
./get_dpu_fps ./baseline/model/arm64_4096/alexnetBNnoLRN.xmodel 2 1000
./get_dpu_fps ./baseline/model/arm64_4096/alexnetBNnoLRN.xmodel 3 1000
./get_dpu_fps ./baseline/model/arm64_4096/alexnetBNnoLRN.xmodel 4 1000
./get_dpu_fps ./baseline/model/arm64_4096/alexnetBNnoLRN.xmodel 5 1000
./get_dpu_fps ./baseline/model/arm64_4096/alexnetBNnoLRN.xmodel 6 1000


# run VART ith multi-threading on pruned CNN
echo " "
echo "PRUNED CNN WITH PYTHON APP AND MULTITHREADING"
echo " "
./get_dpu_fps ./pruned/model/arm64_4096/alexnetBNnoLRN.xmodel 1 1000
./get_dpu_fps ./pruned/model/arm64_4096/alexnetBNnoLRN.xmodel 2 1000
./get_dpu_fps ./pruned/model/arm64_4096/alexnetBNnoLRN.xmodel 3 1000
./get_dpu_fps ./pruned/model/arm64_4096/alexnetBNnoLRN.xmodel 4 1000
./get_dpu_fps ./pruned/model/arm64_4096/alexnetBNnoLRN.xmodel 5 1000
./get_dpu_fps ./pruned/model/arm64_4096/alexnetBNnoLRN.xmodel 6 1000









