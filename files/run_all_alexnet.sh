#!/bin/bash

##
##* © Copyright (C) 2016-2020 Xilinx, Inc
##*
##* Licensed under the Apache License, Version 2.0 (the "License"). You may
##* not use this file except in compliance with the License. A copy of the
##* License is located at
##*
##*     http://www.apache.org/licenses/LICENSE-2.0
##*
##* Unless required by applicable law or agreed to in writing, software
##* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
##* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
##* License for the specific language governing permissions and limitations
##* under the License.
##*/

# *******************************************************************************


#dos2unix conversion
for file in $(find $PWD -name "*.sh"); do
    sed -i 's/\r//g' ${file}
    echo  ${file}
    # make all *.sh scripts to be executable
    chmod +x ${file}
done


CNN=alexnetBNnoLRN

# set the project environmental variables
source caffe/set_prj_env_variables.sh

# set the proejct directories
#python set_the_CATSvsDOGS_prj.py -i $ML_DIR

# train the CNN and make predictions
source caffe/caffe_flow_AlexNet.sh             2>&1 | tee logfile_caffe_AlexNet.txt

# quantize the CNN
source deploy/${CNN}/quantiz/vaiq_${CNN}.sh    2>&1 | tee logfile_vaiq_${CNN}.txt

# generate ELF file for ZCU102 board
source deploy/${CNN}/quantiz/vaic_${CNN}.sh    2>&1 | tee logfile_vaic_${CNN}.txt

# create test images for ZCU102 board
cd input/jpg/
tar -cvf test_images.tar ./test
mv test_images.tar ../../deploy/${CNN}/zcu102/
cd ../../


##create unified log file
#cat caffe/models/alexnetBNnoLRN/m2/rpt/logfile_2_alexnetBNnoLRN.log caffe/models/alexnetBNnoLRN/m2/rpt/predictions_2_alexnetBNnoLRN.log logfile_vaiq_${CNN}.txt logfile_vaic_${CNN}.txt > logfile_run_all_alexnet.txt
