#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025 

import torch
import utils
import yaml
import os
import glob

config = {
    "path" : "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/datasets/vehicles_open_image",
    "train": "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/datasets/vehicles_open_image/train",
    "val"  : "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/datasets/vehicles_open_image/valid",
    "nc": 5,
    "names": ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
}

with open("data.yaml", "w") as file:
    yaml.dump(config, file, default_flow_style=False)


