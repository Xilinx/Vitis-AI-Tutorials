#!/bin/bash 

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# date: 29 Sep 2025 

: '
# you need to install these packages only once:
parted /dev/mmcblk0 resizepart 2 100%
resize2fs /dev/mmcblk0p2
'

export VAISW_INSTALL_DIR=/etc/vai
export PYTHONPATH=$VAISW_INSTALL_DIR/lib/python
export HOME=/home/root/

#enable Zebra statistics
#export VAISW_RUNSESSION_SUMMARY=all

# this is the currently available snapshot
export RESNET50_SNAPSHOT=/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.resnet50.TF

: '
# you need to install these packages only once:
python3 -m pip install Pillow
python3 -m pip install torch
python3 -m pip install torchvision
python3 -m pip install onnx
python3 -m pip install onnxruntime #==1.20.1
'

cd /home/root/



                         


