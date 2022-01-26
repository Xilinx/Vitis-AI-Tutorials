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


echo "                                                         "
echo "#########################################################"
echo "# Compiling model file                                  #"
echo "#                                                       #"
echo "# Edit this script to point to the desired snapshot     #"
echo "#########################################################"
echo "                                                         "

#uncomment for original floating point model
#vai_c_xir -x quantize_result_float_unet_512/UNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json  -n compiled_result_float_unet_512/UNet

#uncomment for iter0 model
vai_c_xir -x quantize_result_unet_iter0/UNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_unet_pruned_iter0/Unet

#uncomment for iter1 model
#vai_c_xir -x quantize_result_unet_iter1/UNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_unet_pruned_iter1/UNet

#uncomment for iter2 model
#vai_c_xir -x quantize_result_unet_iter2/UNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_unet_pruned_iter2/Unet

#uncomment for iter3 model
#vai_c_xir -x quantize_result_unet_iter3/UNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_unet_pruned_iter3/Unet

#uncomment for iter4 model
#vai_c_xir -x quantize_result_unet_iter4/UNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_unet_pruned_iter4/Unet