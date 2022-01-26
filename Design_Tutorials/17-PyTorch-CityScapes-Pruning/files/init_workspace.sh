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

echo "##################################################"
echo "# Cityscapes Pruning Tutorial Preparation        #"
echo "##################################################"

echo "                                                  "
echo "##################################################"
echo "# Downloading PyTorch FPN Resnet18 Model         #"
echo "##################################################"
echo "                                                  " 

wget https://www.xilinx.com/bin/public/openDownload?filename=pt_SemanticFPN_cityscapes_256_512_10G_1.4.zip -O pt_SemanticFPN_cityscapes_256_512_10G_1.zip
unzip pt_SemanticFPN_cityscapes_256_512_10G_1.zip && cd pt_SemanticFPN_cityscapes_256_512_10G_1.4

echo "                                                  "
echo "##################################################"
echo "# Downloading Cityscapes Dataset                 #"
echo "#                                                #"
echo "# if not already done, you must register here:   #"
echo "# https://www.cityscapes-dataset.com/register/   #"
echo "#                                                #"
echo "# This step takes some time and requres ~25 GB   #"
echo "# Hard drive space after extracting              #"
echo "##################################################"
echo "                                                  "

git clone https://github.com/mcordts/cityscapesScripts.git
python3 -m pip install cityscapesscripts
csDownload gtFine_trainvaltest.zip -d data
csDownload leftImg8bit_trainvaltest.zip -d data
cd data
unzip gtFine_trainvaltest.zip
rm -rf README
rm -rf license.txt
unzip leftImg8bit_trainvaltest.zip

echo "                                                  "
echo "##################################################"
echo "# making subdirectories and copying pruning      #"
echo "# python scripts (train.py and test.py)          #"
echo "# as well as reduced class dataset python script #"
echo "# reducedclasscityscapes.py and __init__.py      #"
echo "##################################################"
echo "                                                  "

cd ../
mkdir code/prune
mkdir fpn_model_defs
mkdir unet_model_defs
cp ../scripts/prune/*.py code/prune/
cp ../scripts/train/train.py code/train/
cp ../scripts/train/test.py code/test/
cp ../scripts/dataset/* code/datasets/
cp ../scripts/*.sh ./
cp ../scripts/config/* code/configs/

echo "                                                  "
echo "##################################################"
echo "# Done Setting up Workspace                      #"
echo "##################################################"
echo "                                                  "

echo "                                                  "
echo "##################################################"
echo "# importing unet model from:                     #"
echo "# https://github.com/milesial/Pytorch-UNet       #"
echo "##################################################"
echo "                                                  "

git clone https://github.com/milesial/Pytorch-UNet
cp Pytorch-UNet/unet/unet_parts.py code/models/
cp Pytorch-UNet/unet/unet_model.py code/models/
cat code/models/__init__.py Pytorch-UNet/unet/__init__.py >> code/models/__init__2.py 
rm -rf code/models/__init__.py
mv code/models/__init__2.py code/models/__init__.py


echo "                                                  "
echo "##################################################"
echo "# Modifying UNet Padding in unet_parts.py        #"
echo "# for DPU compatibiliy                           #"
echo "##################################################"
echo "                                                  "

sed -i 's/x1 = F.pad/#x1 = F.pad/g' code/models/unet_parts.py
sed -i 's/diffY/#diffY/g' code/models/unet_parts.py
sed -i 's/diffX/#diffX/g' code/models/unet_parts.py

echo "                                                  "
echo "##################################################"
echo "# Downloading Model Prototxt File for running    #"
echo "# with Vitis AI Libraries                        #"
echo "##################################################"
echo "                                                  "

wget https://www.xilinx.com/bin/public/openDownload?filename=SemanticFPN_cityscapes_pt-zcu102_zcu104_kv260-r1.4.1.tar.gz
tar -xzvf openDownload?filename=SemanticFPN_cityscapes_pt-zcu102_zcu104_kv260-r1.4.1.tar.gz
mkdir ZCU102
mkdir ZCU102/FPN
mkdir ZCU102/UNet
cp SemanticFPN_cityscapes_pt/SemanticFPN_cityscapes_pt.prototxt ZCU102/FPN/FPN.prototxt
cp SemanticFPN_cityscapes_pt/SemanticFPN_cityscapes_pt.prototxt ZCU102/UNet/UNet.prototxt
rm -rf SemanticFPN_cityscapes_pt
rm -rf openDownload?filename=SemanticFPN_cityscapes_pt-zcu102_zcu104_kv260-r1.4.1.tar.gz
