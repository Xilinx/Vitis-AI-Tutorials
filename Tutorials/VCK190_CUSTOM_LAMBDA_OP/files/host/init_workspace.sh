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

echo "##################################################"
echo "# DeepWaterMap Custom Op Tutorial                   #"
echo "##################################################"

echo "                                                  "
echo "##################################################"
echo "# Downloading Deepwatermap Repository            #"
echo "# from https://github.com/isikdogan/deepwatermap #"
echo "##################################################"
echo "                                                  " 

git clone https://github.com/isikdogan/deepwatermap
cd deepwatermap

echo "                                                  "
echo "##################################################"
echo "# Modifying deepwatermap.py to add fixed input   #"
echo "# shape for DPU compatibility                    #"
echo "##################################################"
echo "                                                  "

sed -i 's/None, None/512, 512/g' deepwatermap.py

echo "                                                  "
echo "##################################################"
echo "# Moving files into workspace                   #"
echo "##################################################"
echo "                                                  "

mv ../*.py ./
mv ../compile.sh ./

echo "                                                  "
echo "##################################################"
echo "# Done Setting up Workspace                      #"
echo "##################################################"
echo "                                                  "

echo "                                                  "
echo "####################################################################################"
echo "# Users should download and extract the following pretrained checkpoint to         #"
echo "# the ./checkpoints directory:                                                     #"
echo "# https://utexas.app.box.com/s/j9ymvdkaq36tk04be680mbmlaju08zkq/file/565662752887  #"
echo "#                                                                                  #"
echo "# If you wish to train the model, download the training data from here (~200GB)    #"
echo "# https://utexas.app.box.com/s/j9ymvdkaq36tk04be680mbmlaju08zkq/folder/94459511962 #"                                                                             #"
echo "####################################################################################"
echo "                                                  "
