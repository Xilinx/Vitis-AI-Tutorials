#!/bin/bash

: '
/**

* © Copyright (C) 2016-2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
'
# modified by daniele.bagni@xilinx.com
# date 13 June 2022


#dos2unix conversion
for file in $(find $PWD -name "*.sh"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
done


# clean rpt directory
rm -rf ./rpt
mkdir rpt

#clean ZCU102 xmodel
rm target_zcu102/fcn8/model/*.xmodel target_zcu102/fcn8/model/*.json
rm target_zcu102/fcn8ups/model/*.xmodel target_zcu102/fcn8ups/model/*.json
rm target_zcu102/unet/v2/model/*.xmodel target_zcu102/unet/v2/model/*.json
rm target_zcu102/rpt/*
rm -r target_vck190
rm -r target_zcu104
rm target_*.tar

#missing packages
pip install seaborn

cp -rf target_zcu102 target_vck190
cp -rf target_zcu102 target_zcu104


# launch CNNs
source ./run_fcn8.sh
source ./run_fcn8ups.sh
source ./run_unet.sh
