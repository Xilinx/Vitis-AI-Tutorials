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
#vai_c_xir -x quantize_result_fpn_res18_12/FPN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json  -n compiled_result_fpn_res18_256_512/FPN

#uncomment for iter0 model
vai_c_xir -x quantize_result_fpn_iter0/FPN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_fpn_pruned_iter0/FPN

#uncomment for iter1 model
#vai_c_xir -x quantize_result_fpn_iter1/FPN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_fpn_pruned_iter1/FPN

#uncomment for iter2 model
#vai_c_xir -x quantize_result_fpn_iter2/FPN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_fpn_pruned_iter2/FPN

#uncomment for iter3 model
#vai_c_xir -x quantize_result_fpn_iter3/FPN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_fpn_pruned_iter3/FPN

#uncomment for iter4 model
#vai_c_xir -x quantize_result_fpn_iter4/FPN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json -n compiled_fpn_pruned_iter4/FPN