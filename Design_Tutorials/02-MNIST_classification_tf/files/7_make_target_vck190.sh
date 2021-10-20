#!/bin/bash

# Copyright 2020 Xilinx Inc.
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

# Author: Mark Harvey


echo "-----------------------------------------"
echo "MAKE TARGET VCK190 STARTED.."
echo "-----------------------------------------"

# remove previous results
rm -rf ${TARGET_VCK190}
mkdir -p ${TARGET_VCK190}/model_dir

# copy application to TARGET_VCK190 folder
cp ${APP}/*.py ${TARGET_VCK190}
echo "  Copied application to TARGET_VCK190 folder"


# copy xmodel to TARGET_VCK190 folder
cp ${COMPILE_VCK190}/${NET_NAME}.xmodel ${TARGET_VCK190}/model_dir/.
echo "  Copied xmodel file(s) to TARGET_VCK190 folder"

# create image files and copy to target folder
mkdir -p ${TARGET_VCK190}/images

python generate_images.py  \
    --dataset=mnist \
    --image_dir=${TARGET_VCK190}/images \
    --image_format=jpg \
    --max_images=10000

echo "  Copied images to TARGET_VCK190 folder"

echo "-----------------------------------------"
echo "MAKE TARGET VCK190 COMPLETED"
echo "-----------------------------------------"




