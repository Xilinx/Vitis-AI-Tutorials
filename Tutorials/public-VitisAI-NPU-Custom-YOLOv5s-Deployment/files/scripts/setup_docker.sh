#!/bin/bash

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 12 Sep 2025

# install packages 

cd /workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/yolov5

python3 -m pip install jupyter
python3 -m pip install -qr requirements.txt comet_ml
python3 -m pip install torchinfo
python3 -m pip install utils
python3 -m pip install onnx_tool
export PATH=${PATH}:${HOME}/.local/bin


# solve the issue <module 'PIL.Image' has no attribute 'ANTIALIAS'> 
# <REPLACE THE PATH NAME BELOW WITH CORRECT ONE>
# sudo cp ../code/summary.py /usr/local_*/lib/python3.10/site-packages/torch/utils/tensorboard/summary.py 

export YOLOV5S_TUTORIAL=/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files
















