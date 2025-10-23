#!/usr/bin/env python3

# Copyright © 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

# This script runs a YOLOv5 PyTorch snapshot on an NPU using the Vitis AI Runtime (VART).
# It currently supports processing a single batch.
# The script outputs the raw results from the YOLOv5 model to /tmp/yolov5_output.raw.

# Usage:
#  VAISW_USE_RAW_OUTPUTS=1 python3 yolov5_npu_runner.py --snapshot <snapshot_dir> --image <image_path> [--use_unquantize] [--dump_input] [--dump_output]

# Note that postprocessing of the output is not included in this script.

import numpy as np
import os
import time
import argparse
from PIL import Image
import VART

WIDTH = 640             # Model required width
HEIGHT = 640            # Model required height
COEFF = 128             # Quantize factor for input tensor
MEAN = 0.0
SCALE = 0.0039215

def quantize_to_int8(image, coeff):
    image = np.clip(np.round(image * coeff), -128, 127).astype(np.int8)
    return image

def preprocess_image(image_path, use_unquantize, dump_input):
    image = Image.open(image_path)

    # PyTorch of yolov5 model need RGB as input
    image = image.convert('RGB')

    # resize the image to model supported resolution
    image = image.resize((WIDTH, HEIGHT))

    image = np.array(image)

    image = (image - MEAN) * SCALE

    if use_unquantize:
        image = image.astype(np.float32)
        if dump_input:
            # Dump input data to file after quantization
            input_file = f"/tmp/input_unquantize_{HEIGHT}x{WIDTH}.raw"
            image.tofile(input_file)
            print(f"Input dumped to {input_file}")
    else:
        image = quantize_to_int8(image, COEFF)
        if dump_input:
            # Dump input data to file after quantization
            input_file = f"/tmp/input_quantize_{HEIGHT}x{WIDTH}.raw"
            image.tofile(input_file)
            print(f"Input dumped to {input_file}")

    # Model input is NCHW but the JPEG is in HWC
    image = np.transpose(image, (2, 0, 1))  # Change shape to 3x640x640

    # Add batch as N
    image = np.expand_dims(image, axis=0)   # Change shape to 1x3x640x640
    return image

def run(snapshot_dir, image_path, use_unquantize, dump_input, dump_output, num_inferences):
    model = VART.Runner(snapshot_dir=snapshot_dir)

    # Preprocess the input image once
    preprocessed_image = preprocess_image(image_path, use_unquantize, dump_input)

    total_time = 0

    # Run model multiple times using the same preprocessed image
    for i in range(num_inferences):
        print(f"Frame {i+1}")
        before = time.time()
        outp = model([preprocessed_image])
        after = time.time()
        inference_time = (after - before) * 1000  # Convert to milliseconds
        total_time += inference_time
        print(f"Total Inference (NPU + ONNX Graph) took {inference_time:.2f} ms\n")

        if dump_output:
            # Dump output in raw format
            for j, output in enumerate(outp):
                output_file = f"/tmp/yolov5_output{j}_{i}.raw"
                output.tofile(output_file)
                print(f"Output {j} of inference {i+1} dumped to {output_file}")

    average_time = total_time / num_inferences
    print(f" ")
    print(f"Average Total Inference Time for {i+1} Frames: {average_time:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Options for yolov5_npu_runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--snapshot", type=str, default='', help="Path to the snapshot folder")
    parser.add_argument("--image", type=str, required=True, help="Path to the input JPEG image")
    parser.add_argument("--use_unquantize", action="store_true", help="Use unquantize float instead of quantize int for model input")
    parser.add_argument("--dump_input", action="store_true", help="Dump input data of model")
    parser.add_argument("--dump_output", action="store_true", help="Dump output data of model")
    parser.add_argument("--num_inferences", type=int, default=1, help="Number of inferences to run")
    args = parser.parse_args()
    snapshot_dir = args.snapshot
    image_path = args.image
    use_unquantize = args.use_unquantize
    dump_input = args.dump_input
    dump_output = args.dump_output
    num_inferences = args.num_inferences
    run(snapshot_dir, image_path, use_unquantize, dump_input, dump_output, num_inferences)
