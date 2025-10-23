#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

# This script runs a YOLOv5 PyTorch NN on the host PC. 
# It currently supports processing a single batch.


"""
Usage:
cd files/yolov5 
python3 ../app/host_vehicles_yolov5_inference.py \
        --images_dir ./vehicles_open_image/valid/images \
        --json ../app/host_res/vehicles_10images_detections.json \
        --NbImages 10 --dump_out True \
        --weights <your weights.pt file>
"""


import numpy as np
import os
import sys
import time
import argparse
from PIL import Image

import torch
import cv2
from torchvision.ops import nms
from pathlib import Path
import json

# ****************************************************************************************
# PRE PROCESSING
# ****************************************************************************************

WIDTH  = 640            # Model required width
HEIGHT = 640            # Model required height
MEAN   = 0.0
SCALE  = 0.0039215

MAX_NUM_OF_IMAGES = 250
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45 

#IOU_THRESHOLD  = 0.75
#CONF_THRESHOLD = 0.001 


# Clear the screen based on the operating system
os.system('cls' if os.name == 'nt' else 'clear')

#Add the yolov5 directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5')))
from models.experimental import attempt_load

def preprocess_image(image_path):
    image2 = Image.open(image_path)
    # Convert to RGB
    image2 = image2.convert('RGB')
    # get image shape
    image3 = np.array(image2)
    im_h, im_w, _ = image3.shape
    image_shape = (im_h, im_w)
    # Resize
    image = image2.resize((WIDTH, HEIGHT))
    image = np.array(image)
    image = (image - MEAN) * SCALE # 1/255
    # Change shape to N CHW format
    # Model input is NCHW but the JPEG is in HWC
    image = np.transpose(image, (2, 0, 1))  # Change shape to 3x640x640
    image = np.expand_dims(image, axis=0)   # Change shape to 1x3x640x640
    # return torch.tensor(image, dtype=torch.float32)
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

# Post-processing function to handle YOLOv5 outputs
def post_process(image_id, pred, input_size, original_size, conf_thres=0.5, iou_thres=0.4):
   # Extract components
   boxes = pred[..., :4].clone()
   scores = pred[..., 4].clone()
   class_probs = pred[..., 5:].clone()
   # Convert from center-based (Xc Yc W H) format to corner-based (X1 Y1 X2 Y2) format
   boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # X1
   boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # Y1
   boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # X2=X1+W
   boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # Y2=Y1+H
   # Scale to original image size
   scale_x = original_size[1] / input_size[1]
   scale_y = original_size[0] / input_size[0]
   boxes[:, [0, 2]] *= scale_x
   boxes[:, [1, 3]] *= scale_y
   # Clamp boxes to ensure they are not negative
   boxes[:, 0] = boxes[:, 0].clamp(min=0)  # Ensure X1 is non-negative
   boxes[:, 1] = boxes[:, 1].clamp(min=0)  # Ensure Y1 is non-negative
   # Apply NMS
   keep = nms(boxes.clone().detach(), scores.clone().detach(), iou_thres)
   keep = keep[scores[keep] > conf_thres]
   # Extract class predictions
   final_boxes = boxes.clone().detach()[keep]
   final_scores = scores.clone().detach()[keep]
   final_classes = torch.argmax(class_probs.clone().detach()[keep], dim=1) 
   # Debug: Output the class ihost_quick_test.pyndices and probabilities for inspection
   # print(f"Final Classes Indices: {final_classes.tolist()}")
   # print(f"Scores: {final_scores.tolist()}")
   # int_boxes = final_boxes.to(torch.int32)
   # Prepare the detections
   X1Y1X2Y2_detections = []
   for i in range(len(final_boxes)) :
        det = {
            'image_id': image_id, 
            'bbox': final_boxes[i].tolist(), 
            'score': final_scores[i].item(), 
            'category_id': final_classes[i].item(), # class name
            'class_name': VEHICLES_CLASSES[final_classes[i].item()] 
        }
        X1Y1X2Y2_detections.append(det)
   print(f"Number of detections after nms: {len(X1Y1X2Y2_detections)}")        
   return X1Y1X2Y2_detections

   
   
def post_process_without_nms(pred, input_size, original_size, conf_thres=0.5) :
    # Assuming results is the output tensor from the model
    results = pred
    # Scale to original image size
    scale_x = original_size[1] / input_size[1]
    scale_y = original_size[0] / input_size[0]
    # Set a confidence threshold
    conf_threshold = conf_thres # you can adjust this value
    # Filtering to get detections above the confidence threshold
    XcYcWH_detections = results[results[:, 4] > conf_threshold]  # Index 4 is the confidence score
    #print(f"Number of detections above confidence threshold: {len(XcYcWH_detections)}")
    # Get the detections without NMS
    for detection in XcYcWH_detections:
        # Unpack the detection
        x_center, y_center, width, height, conf, *class_probs = detection.tolist()
        # Get the class index (highest probability class)
        class_index = class_probs.index(max(class_probs))
        # Calculate bounding box coordinates from center x, center y, width, height
        x1 = int( (x_center - width  / 2) * scale_x )
        y1 = int( (y_center - height / 2) * scale_y )
        x2 = int( (x_center + width  / 2) * scale_x )
        y2 = int( (y_center + height / 2) * scale_y )
        #print(f'Box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {conf:.2f}, Class: {class_index}')
    return XcYcWH_detections

   
# ****************************************************************************************
# COMPUTE DETECTIONS (aka PREDICTIONS)
# ****************************************************************************************

def detect_and_annotate(inp_predictions, image_path, original_size, input_size=(WIDTH, HEIGHT),
                            conf_thres=0.5, iou_thres=0.4, 
                            dump_output=False):
    
    image = cv2.imread(image_path)
    image_id = os.path.basename(image_path).split(".jpg")[0]   
    print(f"detecting image_id = {image_id}, image_path={image_path}")          
    predictions = inp_predictions[0] #inp_predictions.squeeze(0)
    #XcYcWH_detections   = post_process_without_nms(predictions, input_size, original_size, conf_thres)    
    X1Y1X2Y2_detections = post_process(image_id, predictions, input_size, original_size, conf_thres, iou_thres)
    # Check if X1Y1X2Y2_detections are empty
    if not X1Y1X2Y2_detections:
        print("No detections found.")
        return []
    # do your own annotations by drawing the bounding box    
    for det in X1Y1X2Y2_detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class_name']} {det['score']:.2f}"
        color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if dump_output:
        output_image_path = f"{Path(image_path).stem}_detection_results.jpg"
        cv2.imwrite(output_image_path, image)
        print(f"Annotated image saved to: {output_image_path}") 
    '''
    for det in X1Y1X2Y2_detections:
        print(f"Image ID: {image_id}, Class: {det['category_id']} {det['class_name']}, Score: {det['score']:.2f}, Box: {det['bbox']}")
    '''
    return X1Y1X2Y2_detections


def check_yolov5_output(yolov5_tensor):
    # Check the results
    results = yolov5_tensor
    # Print the type of 'results'
    print(f"Type of results: {type(results)}")
    print(f"Number of elements in the tuple: {len(results)}")
    # Iterate through each element in the tuple to print its type and shape
    for i, result in enumerate(results):
        print(f"Element {i}: Type = {type(result)}")
        if isinstance(result, torch.Tensor):
            print(f"Element {i} shape: {result.shape}")
        else:
            print(f"Element {i} is not a tensor.")        
    # Extract predictions
    predictions = results[0]
    # Check the number of predictions
    num_predictions = predictions.shape[1]  # Second dimension for number of predictions
    print(f"Number of predictions: {num_predictions}")
    '''
    # If there are predictions (num_predictions > 0), inspect the predictions
    if num_predictions > 0:
        # Print the first few predictions
        print("First few predictions:")
        print(predictions[0, :5])  # Print first 5 predictions
    else:
        print("No predictions were made.")
    # Check the contents of Element 1
    additional_info = results[1]
    print(f"Additional information (Element 1) type: {type(additional_info)}")
    print(f"Number of extra information entries: {len(additional_info)}")
    ## Inspect the additional information
    #for i, info in enumerate(additional_info):
    #    print(f"Entry {i}: {info}")
    '''

'''
we can summarize the results from the YOLOv5 inference as follows:

    Predictions (Element 0):
        The predictions tensor has the shape [1, 25200, 10]. This indicates that for a single input image, you received 25,200 potential predictions, each consisting of 10 values. These values generally include:
            Bounding box coordinates (usually in the format [x1, y1, x2, y2] or [center_x, center_y, width, height]).
            The objectness score.
            Class probabilities for each detection.

    Additional Information (Element 1):
        This element is a list containing 3 tensors, each having a shape of [1, N, H, W], where N, H, and W can vary depending on the output features of the model. This often corresponds to feature maps (for instance, intermediate outputs from the backbone or detectors). They may provide additional insights into the detections, which could be useful for visualizations or further processing.

'''

# ****************************************************************************************
# MAIN FUNCTIONS
# ****************************************************************************************

# FROM STRING TO BOOLEAN 
def str_to_bool(value):
   """Convert a string to a boolean."""
   if value.lower() in ('true', '1', 'yes'):
       return True
   elif value.lower() in ('false', '0', 'no'):
       return False
   else:
       raise argparse.ArgumentTypeError(f"Boolean value expected for {value}")


# INFERENCE MAIN FUNCTION
def main_inf(model, inp_images_dir, inp_detections_file, NbImages, dump_output) :
    dump_input  = False
    # tuple of image file extensions to consider
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    # files of images
    image_files = [ file for file in os.listdir(inp_images_dir) if file.lower().endswith((image_extensions)) ]
    # sort the images files
    image_files.sort() 
    images_count = len(image_files)
    print(f"Number of images: {images_count}")
    # Initialize a list to accumulate all detections
    all_detections = []
    total_inference_time = 0
    total_postproc_time = 0  
    num_images=0  
    for index, image_file in enumerate(image_files, start=1) :
        if index > NbImages :
            break        
        image_path = os.path.join(inp_images_dir, image_file)
        print()
        print(f"Processing image {index}/{images_count}: {image_path}", end='')  # "\r" Print on the same line #DB  
        print()
        # Preprocess the image
        original_size, prepr_image = preprocess_image(image_path)
        im_tensor = torch.tensor(prepr_image).float() 
        before = time.time()
        # Perform inference
        with torch.no_grad():
            results = model(im_tensor)  # Run inference
        check_yolov5_output(results) #for debug purpose
        after = time.time()
        inference_time = (after - before) * 1000
        total_inference_time = inference_time + total_inference_time
        # Loop through each tensor in the list
        #for idx, tensor in enumerate(results):
        # Assuming results contains the two elements as described
        for idx, element in enumerate(results):
            # Check if the element is a predictions tensor
            if isinstance(element, torch.Tensor) and element.dim() == 3:  # Check for predictions tensor shape
                print(f"Processing predictions tensor (Entry {idx}): Type = {type(element)}, Shape = {element.shape}")
                # Perform operations on each tensor
                before = time.time()
                X1Y1X2Y2_detections = detect_and_annotate(element, image_path, original_size,
                                             conf_thres=CONF_THRESHOLD,
                                             iou_thres=IOU_THRESHOLD,
                                             dump_output=dump_output)
                # Collect detections into the list
                all_detections.extend(X1Y1X2Y2_detections)  # Append detections for the current image
                after = time.time()
                detections_time = (after - before) * 1000   
                total_postproc_time  = total_postproc_time + detections_time 
            elif isinstance(element, list):
                print(f"Skipping additional information (Entry {idx}): Type = {type(element)}, Number of entries = {len(element)}")
            # You might want to log or process the additional information but not as predictions.
            else:
                print(f"Unexpected element type at Entry {idx}: {type(element)}")
        num_images = index        
    print(f"Total Inference Time on {num_images} images = {total_inference_time:.2f} ms")
    print(f"Total PostProc  Time on {num_images} images = {total_postproc_time:.2f} ms") 

    # Save detections to a JSON file
    with open(inp_detections_file, 'w') as f:
        json.dump(all_detections, f, indent=4)
    print(f"\nDetections saved to {inp_detections_file}")
        

def main():
    # Set up the main argument parser
    parser = argparse.ArgumentParser(description="Vehicles Custom YOLOv5 Detection HOST Script") 
    parser.add_argument("--images_dir", type=str, #required=True, 
                        default = "./vehicles_open_image/valid/images",
                        help='Path to the input images folder')
    parser.add_argument("--json", type=str, #required=True, 
                        default="../app/host_res/vehicles_250images_detections.json",
                        help='Path to the output JSON file with detections')
    parser.add_argument("--NbImages", type=int, default = 1000, #MAX_NUM_OF_IMAGES, 
                        help='max Number of Images to be evaluated')
    parser.add_argument('--dump_out', type=str_to_bool, default=False, help='Enable dumping output (true/false)')
    parser.add_argument("--weights", type=str, #required=True, 
                        default="../weights/best.pt",
                        help='Path to the NN weights file')
    # Parse the arguments
    args = parser.parse_args()
 
    # Load the Custom YOLOv5 model once
    local_model_path=args.weights
    full_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), local_model_path))
    print(f"local    model path: {local_model_path}")
    print(f"absolute model path: {full_model_path}")
    model = attempt_load(full_model_path)
    model.eval() 
    
    # run main inference
    main_inf(model, args.images_dir, args.json, args.NbImages, args.dump_out)



if __name__ == "__main__":
    main()





