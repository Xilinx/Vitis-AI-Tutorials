#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

# This script runs a YOLOv5 PyTorch snapshot on an NPU using the Vitis AI Runtime (VART).
# It currently supports processing a single batch.
# The script outputs the raw results from the YOLOv5 model to /tmp/yolov5_output.raw.

# Usage:
#  python3 target_yolov5_vehicles_inference.py --images_dir yolov5/datasets/coco/images/val2017 --json host_coco_10images_detections.json --NbImages 10
#  python3 target_yolov5_vehicles_inference.py --images_dir files/app/images                    --json host_zidane_bus_coco_detections.json --NbImages 4


"""
cd app
python3 ./target_yolov5_vehicles_inference.py --images_dir ~/datasets/vehicles_open_image/valid/images --json ./target_res/json_vehicles/snap_1Q_vehicles_vehicles_250images_detections_conf025_iou045.json  --NbImages 250 
--dump_out False --snap ../snapshots/snapshot.bs1.vehicles_1Q_yolov5s
"""

import numpy as np
import os
import time
import argparse
from PIL import Image
import VART
import torch
import cv2
from torchvision.ops import nms
from pathlib import Path
import json

# ****************************************************************************************
# pre processing
# ****************************************************************************************

WIDTH = 640             # Model required width
HEIGHT = 640            # Model required height
COEFF = 128             # Quantize factor for input tensor
MEAN = 0.0
SCALE = 0.0039215

PADDING_SIZE = 672
MAX_NUM_OF_IMAGES = 250
CONF_THRESHOLD = 0.25 #0.001 #0.25
IOU_THRESHOLD =  0.45 #0.75 #0.45

WRK_DIR="/home/root/YOLOv5s"

# Clear the screen based on the operating system
os.system('cls' if os.name == 'nt' else 'clear')

def pad_image(inp_image) :    
    # Calculate padding
    target_size = PADDING_SIZE
    current_size = inp_image.size  # (width, height)
    # Calculate padding for width and height
    pad_width  = (target_size - current_size[0]) // 2
    pad_height = (target_size - current_size[1]) // 2
    # Create a new image with the target size and a white background (255,255,255) or black (0,0,0)
    out_image = Image.new("RGB", (target_size, target_size), (0,0,0))
    # Paste the original image onto the center of the new image
    out_image.paste(inp_image, (pad_width, pad_height))
    return out_image


def quantize_to_int8(image, coeff):
    image = np.clip(np.round(image * coeff), -128, 127).astype(np.int8)
    return image

def preprocess_image(image_path, use_unquantize, dump_input):
    image = Image.open(image_path)
    # PyTorch of yolov5 model need RGB as input
    image = image.convert('RGB')
    # get image shape
    image3 = np.array(image)
    im_h, im_w, _ = image3.shape
    image_shape = (im_h, im_w)
    # resize the image to model supported resolution
    rsz_image = image.resize((WIDTH, HEIGHT))
    padded_image = rsz_image #pad_image(rsz_image)
    image = np.array(padded_image)
    image = (image - MEAN) * SCALE # 1/255
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
    return image_shape, image

# ****************************************************************************************
# post processing
# ****************************************************************************************

# Replace COCO_CLASSES with categories specific to Vehicles-OpenImage dataset
VEHICLES_CLASSES =  ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

# Category mapping as a dictionary for Vehicles-OpenImage dataset
category_map = {
    "Ambulance": 1,
    "Bus": 2,
    "Car": 3,
    "Motorcycle": 4,
    "Truck": 5
}

# Function to read predictions from a binary file
def read_predictions(bin_file, input_size=640):
    # Read raw prediction data
    pred_data = np.fromfile(bin_file, dtype=np.float32)
    # Shape should be (1, 25200, 85): 25200 predictions with 85 values (classes + bbox + confidence)
    pred_data = pred_data.reshape(1, 25200, 85)
    return torch.tensor(pred_data)


# Post-processing function to handle YOLOv5 ONNX/PL outputs
def post_process(image_id, pred, input_size, original_size, conf_thres=0.5, iou_thres=0.4):
    # Extract components
    boxes = pred[..., :4]
    scores = pred[..., 4]
    class_probs = pred[..., 5:]
    # Convert center-based format to corner-based format
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2 = x1 + w
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2 = y1 + h
    # Scale to original image size
    scale_x = original_size[1] / input_size[1]
    scale_y = original_size[0] / input_size[0]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    # Clamp boxes to ensure they are not negative
    boxes[:, 0] = boxes[:, 0].clamp(min=0)  # Ensure X1 is non-negative
    boxes[:, 1] = boxes[:, 1].clamp(min=0)  # Ensure Y1 is non-negative    # Apply NMS
    keep = nms(boxes.clone().detach(), scores.clone().detach(), iou_thres)
    keep = keep[scores[keep] > conf_thres]
    # Extract class predictions
    final_boxes = boxes.clone().detach()[keep]
    final_scores = scores.clone().detach()[keep]
    final_classes = torch.argmax(class_probs.clone().detach()[keep], dim=1)
    #int_boxes = final_boxes.to(torch.int32)
    # prepare the detections
    x1y1x2y2_detections = []
    for i in range(len(final_boxes)):
        x1y1x2y2_detections.append({
            'image_id': image_id,
            'bbox': final_boxes[i].tolist(),
            'score': final_scores[i].item(),
            'category_id': final_classes[i].item(), # class_name
            'class_name': VEHICLES_CLASSES[final_classes[i].item()]
        })
    print(f"Number of detections after nms: {len(x1y1x2y2_detections)}")
    return x1y1x2y2_detections

def post_process_without_nms(pred, input_size, original_size, conf_thres=0.5) :
    # Assuming results is the output tensor from the model
    results = pred
    # Scale to original image size
    scale_x = original_size[1] / input_size[1]
    scale_y = original_size[0] / input_size[0]
    # Set a confidence threshold
    conf_threshold = conf_thres # you can adjust this value
    # Filtering to get detections above the confidence threshold
    xcyxwh_detections = results[results[:, 4] > conf_threshold]  # Index 4 is the confidence score
    print(f"Number of detections above confidence threshold: {len(xcyxwh_detections)}")
    # Process detections
    for detection in xcyxwh_detections:
        # Unpack the detection
        x_center, y_center, width, height, conf, *class_probs = detection.tolist()
        # Get the class index (highest probability class)
        class_index = class_probs.index(max(class_probs))
        # Calculate bounding box coordinates from center x, center y, width, height
        x1 = int( (x_center - width  / 2) * scale_x )
        y1 = int( (y_center - height / 2) * scale_y )
        x2 = int( (x_center + width  / 2) * scale_x )
        y2 = int( (y_center + height / 2) * scale_y )
        print(f'Box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {conf:.2f}, Class: {class_index}')
    return xcyxwh_detections

# ****************************************************************************************
# MAKE PREDICTIONS
# ****************************************************************************************

def detect_and_annotate(inp_predictions, image_path, original_size,
                        input_size=(WIDTH, HEIGHT), 
                        conf_thres=0.5, iou_thres=0.4, 
                        dump_output=False) :
    
    image = cv2.imread(image_path)
    image_id = os.path.basename(image_path).split('.jpg')[0]
    print(f" image_id = {image_id}, image_path={image_path}")
    predictions = inp_predictions.squeeze(0)
    # xcyxwh_detections = post_process_without_nms(predictions, input_size, (im_h, im_w), conf_thres)
    x1y1x2y2_detections = post_process(image_id, predictions, input_size, original_size, conf_thres, iou_thres)
    # Check if detections is empty
    if not x1y1x2y2_detections:
        print("No detections found.")
        return []    
    # do your own annotations by drawing the bounding box
    for det in x1y1x2y2_detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class_name']} {det['score']:.2f}"
        color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if dump_output: #DB
        output_image_path = Path(image_path).stem + "_detection_results.jpg"
        cv2.imwrite(output_image_path, image)
        print(f"Annotated image saved to: {output_image_path}")
    '''
    for det in x1y1x2y2_detections:
        print(f"Image ID: {image_id}, Class: {det['category_id']} {det['class_name']}, Score: {det['score']:.2f}, Box: {det['bbox']}")
    '''
    
    return x1y1x2y2_detections


# ****************************************************************************************
# Main functions
# ****************************************************************************************

def str_to_bool(value):
    """Convert a string to a boolean."""
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected for {value}")

def main_inf(inp_snapshot_dir, inp_images_dir, inp_detections_file,  NbImages, dump_output) :
    dump_input     = False
    #dump_output    = False
    
    use_unquantize = True #args.use_unquantize
    snapshot_dir    = inp_snapshot_dir    #"/home/root/snapshots/snapshot.bs1.coco_yolov5s" 
    folder_path     = inp_images_dir      #"/home/root/demo/yolov5/data/images" #args.folder   
    detections_file = inp_detections_file #"npu_coco_detections.json" 
    model = VART.Runner(snapshot_dir=snapshot_dir)

    # tuple of image file extensions to consider
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    # files of images
    image_files = [ file for file in os.listdir(folder_path) if file.lower().endswith((image_extensions)) ]
    # sort the images
    image_files.sort() 
    images_count = len(image_files)
    print(f"Number of available images: {images_count}")
    print(f"Number of input images    : {NbImages}")
    total_detections = []
    total_inference_time = 0
    total_postproc_time = 0
    num_images = 0
    for index, image_file in enumerate(image_files, start=1) :
        if index >= NbImages :
            break
        image_path = os.path.join(folder_path, image_file)
        #print(f"now preprocessing {image_path}")
        print(f"\rProcessing image {index}/{NbImages} ", end='')  # Print on the same line #DB
        # Preprocess the image
        original_size, prepr_image = preprocess_image(image_path, use_unquantize, dump_input)
        before = time.time()
        # Perform inference: output is a list
        out_list = model([prepr_image])
        after = time.time()
        inference_time = (after - before) * 1000 # time in milliseconds (1s=1000 ms)
        #print(f"Inference on {image_path} took {inference_time:.2f} ms")
        '''
        print("Entire out_list:")
        print(out_list)
        print("\nLoop thorugh the elements in out_list:")
        # If out_list contains arrays, loop through them to see more detail
        for index, item in enumerate(out_list):
            print(f"Item {index} (Type: {type(item)}, Shape: {getattr(item, 'shape', 'N/A')}):")
            print(item)
        '''

        '''
        print("\nLengths and shapes of elements in out_list:")
        for index, item in enumerate(out_list):
            length = len(item)  # This works if item is a list or array
            shape = getattr(item, 'shape', 'N/A')  # Use shape if it has this attribute
            print(f"Item {index}: Length = {length}, Shape = {shape}")            
        print()
        '''

        '''
        # very strangely this is the:
        #Lengths and shapes of elements in out_list:
        Item 0: Length = 1, Shape = (1, 25200, 10)
        Item 1: Length = 1, Shape = (1, 3, 80, 80, 10)
        Item 2: Length = 1, Shape = (1, 3, 40, 40, 10)
        Item 3: Length = 1, Shape = (1, 3, 20, 20, 10)
        '''

        # Convert the list of NumPy arrays to a single NumPy array
        #out_array = np.array(out_list)  # This worked fine for COCO dataset, but VEHICLES requires the line below...
        out_array = np.array(out_list[0])  # patch for VEHICLES dataset
        pred_data = torch.tensor(out_array)

        before = time.time()
        # post-process
        x1y1x2y2_detections = detect_and_annotate(pred_data[0], image_path, original_size,
                                            conf_thres=CONF_THRESHOLD,
                                            iou_thres=IOU_THRESHOLD,
                                            dump_output=dump_output)                                                  
        # Collect detections into the list                                     
        total_detections.extend(x1y1x2y2_detections)  # Append detections for the current image
        after = time.time()
        detections_time = (after - before) * 1000
        total_inference_time = inference_time + total_inference_time
        total_postproc_time  = total_postproc_time + detections_time
        num_images = index
        #print(f"Inference on {image_path} took {inference_time:.2f} ms")
    print(f"Total Inference Time on {num_images} images = {total_inference_time:.2f} ms")
    print(f"Total PostProc  Time on {num_images} images = {total_postproc_time:.2f} ms")
    
    # Save detections to a JSON file
    with open(detections_file, 'w') as f:
        json.dump(total_detections, f, indent=4)  
    print(f"\nDetections saved to {detections_file}")         
        
'''
def main():
    # Set up the main argument parser
    parser = argparse.ArgumentParser(description="Choose a main function to run with parameters.") 
    # Add --mode argument
    parser.add_argument('--mode', choices=['run', 'post', 'inf'], required=True, help='Specify which function to run.')
    # Add arguments specific to each function
    parser.add_argument("--snapshot", type=str, 
                            default='/home/root/snapshots/snapshot.bs1.coco_yolov5s', 
                            help="Path to the snapshot folder")
    parser.add_argument("--image_path", type=str, 
                            default='/home/root/demo/yolov5/data/images/bus.jpg', 
                            help='Path to the input image')
    parser.add_argument("--images_dir", type=str, 
                            #default='/home/root/demo/yolov5/data/images', 
                            default = '/home/root/demo/yolov5/datasets/coco/images/val2017',
                            help='Path to the input images folder')
    parser.add_argument("--json", type=str, 
                            default='target_bus_zidane_coco_detections.json', 
                            help='Path to the output json file')
    parser.add_argument("--pred_data", type=str, 
                        default = '/tmp/yolov5_output0_0.raw', 
                        help="Path to the binary prediction file")
    # Parse the arguments
    args = parser.parse_args()
    # Call the appropriate main function based on the mode
    if args.mode == 'run':
        if args.snapshot is None or args.image_path is None:
            parser.error('--snapshot and --image_path are required for mode "run".')
        main_run(args.snapshot, args.image_path)    
    elif args.mode == 'post':
        if args.pred_data is None or args.image_path is None :
            parser.error('--pred_data and --image_path are required for mode "post".')
        main_post(args.pred_data, args.image_path)
    elif args.mode == 'inf':
        if args.snapshot is None or args.images_dir is None or args.json is None:
            parser.error('--snapshot and --images_dir and --json are required for mode "inf".')
        main_inf(args.snapshot, args.images_dir, args.json)
'''

def main():
    # Set up the main argument parser
    parser = argparse.ArgumentParser(description="YOLOv5 Detection TARGET Script") 
    parser.add_argument("--snapshot", type=str, 
                            default=os.path.join(WRK_DIR,"snapshots/snapshot.bs1.vehicles_250Q_yolov5s"), 
                            help="Path to the snapshot folder")    
    parser.add_argument("--images_dir", type=str, 
                            default=os.path.join(WRK_DIR,"vehicles_open_image/valid/images"),                         
                            help='Path to the input images folder')
    parser.add_argument("--json", type=str, 
                            default=os.path.join(WRK_DIR,"snap100Q_vehicles_250images_detections_conf025_iou045.json"),
                            help='Path to the output JSON file with detections')
    parser.add_argument("--NbImages", type=int, default = MAX_NUM_OF_IMAGES, 
                        help='max Number of Images to be evaluated')
    parser.add_argument('--dump_out', type=str_to_bool, default=False, help='Enable dumping output (true/false)')
    # Parse the arguments
    args = parser.parse_args()
        
    # run main inference
    main_inf(args.snapshot, args.images_dir, args.json, args.NbImages, args.dump_out)


  
if __name__ == "__main__":
    main()
