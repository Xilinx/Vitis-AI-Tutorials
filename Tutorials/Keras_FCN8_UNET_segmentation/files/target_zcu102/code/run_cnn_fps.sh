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

# now run semantic segmentation with 3  multithreads
echo " "
echo " FCN8 fps"
echo " "
./get_dpu_fps ./fcn8/model/fcn8.xmodel 1 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 2 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 3 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 4 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 5 1000
./get_dpu_fps ./fcn8/model/fcn8.xmodel 6 1000

echo " "
echo " FCN8ups fps"
echo " "
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 1 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 2 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 3 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 4 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 5 1000
./get_dpu_fps ./fcn8ups/model/fcn8ups.xmodel 6 1000

echo " "
echo " UNET  fps"
echo " "
./get_dpu_fps ./unet/v2/model/unet2.xmodel 1 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 2 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 3 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 4 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 5 1000
./get_dpu_fps ./unet/v2/model/unet2.xmodel 6 1000
