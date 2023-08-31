#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# ==========================================================================================
# import dependencies
# ==========================================================================================


from config import cifar10_config as cfg #DB

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model


# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI TF2 Quantization of ResNet18")

    # model config
    parser.add_argument("--float_file", type=str, default="./build/float/train2_resnet18_cifar10.h5",
                        help="h5 floating point file full path name")

    return parser.parse_args()

args = get_arguments()

# ==========================================================================================
# Global Variables
# ==========================================================================================
print(cfg.SCRIPT_DIR)

FLOAT_HDF5_FILE = os.path.join(cfg.SCRIPT_DIR,  args.float_file)

# ==========================================================================================
# Get the trained floating point model
# ==========================================================================================

model = keras.models.load_model(FLOAT_HDF5_FILE)

# ==========================================================================================
# Vitis AI Model Inspector
# ==========================================================================================
print("\n[DB INFO] Vitis AI Model Inspector...\n")

from tensorflow_model_optimization.quantization.keras import vitis_inspect
#inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G")
inspector = vitis_inspect.VitisInspector(target="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json")
filename_dump = os.path.join(cfg.SCRIPT_DIR,  "build/log/inspect_results.txt")
filename_svg  = os.path.join(cfg.SCRIPT_DIR,  "build/log/model.svg")
inspector.inspect_model(model,
                        input_shape=[1,  32, 32, 3],
                        plot=True,
                        plot_file=filename_svg,
                        dump_results=True,
                        dump_results_file=filename_dump,
                        verbose=0)
