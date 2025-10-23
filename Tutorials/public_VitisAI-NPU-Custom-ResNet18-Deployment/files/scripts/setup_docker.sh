#!/bin/bash

# ===========================================================
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License
# ===========================================================

# Date 29 Sep. 2025

# install packages 

cd /workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files

sudo chown -R danieleb /workspace
sudo chgrp -R danieleb /workspace

: '
python3 -m pip install jupyter
python3 -m pip install randaugment 
python3 -m pip install torchsummary
#python3 -m pip install torchinfo
#python3 -m pip install utils
python3 -m pip install onnx_tool
'

export PATH=${PATH}:${HOME}/.local/bin


















