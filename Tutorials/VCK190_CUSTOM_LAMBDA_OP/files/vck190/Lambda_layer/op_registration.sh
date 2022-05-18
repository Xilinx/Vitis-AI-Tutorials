# Copyright 2022 Xilinx Inc.
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

ln -sf /usr/lib/libvart_op_imp_python-cpu-op.so /usr/lib/libvart_op_imp_Lambda.so
rm -rf /usr/lib/python3.8/site-packages/vart_op_imp
rm -rf /usr/lib/python3.8/vart_op_imp
cp -r vart_op_imp/ /usr/lib/python3.8/site-packages/
