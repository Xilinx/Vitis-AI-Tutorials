#!/usr/bin/env python3

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

"""
Usage:
cd files/app
cd files/app
python3 vehicles_mean_ap.py  --inp_xyxy pred_x1y1x2y2.json 
python3 vehicles_mean_ap.py  --inp_ann annotations.json --inp_xyxy pred_x1y1x2y2.json 
"""

# Import the necessary packages
import os
import sys
import json
import argparse
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


# Clear the screen based on the operating system
#os.system('cls' if os.name == 'nt' else 'clear')

'''
# check json file, for debug:
with open(COCO_VAL_PRED_X1Y1X2Y2, 'r') as f:
    data = json.load(f)
    print(data)  # Print the content to debug its structure
'''

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


# Convert YOLOv5 detection format (X1 Y1 X2 Y2) to COCO format (X1 Y1 W H)
def convert_detections(input_json_path, output_json_path):
    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        detections = json.load(f)

    # Prepare list for converted detections
    coco_annotations = []

    # Process each detection
    for detection in detections:
        image_id = detection['image_id']
        bbox = detection['bbox']
        yolov5_category_id = detection['category_id']  # from 0 to 80
        coco_category_id =  category_map.get(VEHICLES_CLASSES[yolov5_category_id])
        print(f"category_id YOLOV5: {yolov5_category_id } of class {VEHICLES_CLASSES[yolov5_category_id]} remapped as COCO: {coco_category_id}")         
        # Extract X1, Y1, X2, Y2 from the bbox
        x1, y1, x2, y2 = bbox
        # Convert to X1, Y1, W, H format
        w = x2 - x1
        h = y2 - y1
        # Append new annotation
        coco_annotation = {
            "image_id": image_id,
            #"category_id": detection['category_id'],  # Keeping category_id
            "category_id": coco_category_id,                         
            "bbox": [x1, y1, w, h],  # X1 Y1 W H format
            "score": detection['score']
        }
        coco_annotations.append(coco_annotation)

    # Write the converted detections to the output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_annotations, f, indent=4)

# Compute mean Average Precision (mAP) using COCO evaluation and predictions 
# files, both in X1 Y1 W H format
def compute_map(annotations_file, predictions_file):
    
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        #check_requirements("pycocotools>=2.0.6")
        
        # use the COCO class to load and read the ground-truth annotations
        coco_gt = COCO(annotations_file)

        # Load the detections made by the model from the JSON file
        coco_dt = coco_gt.loadRes(predictions_file)
    
        # Initialize the COCO evaluation object with ground truth and detection results
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        # Run the evaluation on the predictions
        coco_eval.evaluate()
        coco_eval.accumulate()  # Accumulate metrics across images
        coco_eval.summarize()   # Display the evaluation metrics in a summarized form
        print()
    except Exception as e:
        print(f"pycocotools unable to run: {e}")

    # Extracting per-class AP from COCOeval
    # Get category IDs
    cat_ids = coco_gt.getCatIds()  # Get category IDs from GT
    category_names = [coco_gt.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]  # Get category names
    num_classes = len(category_names)

    # The mAP results are stored in the "precisions" array of COCOeval
    # Shape of precisions: [num_iou_thrs, num_classes, max_detections]
    # Use IoU thresholds [0.5:0.95] and [0.50]
    print(f"Average Precision (AP) for Class mAP50 mAP50-0.95")
    ap_data = coco_eval.eval['precision']  # Get the precision values from COCOeval
    print("Shape of ap_data:", ap_data.shape)

    # Print the header
    print(f"{'Class':<15} {'mAP50':<10} {'mAP50-0.95':<15}")
    print("=" * 40)
    for i, cat_id in enumerate(cat_ids):
        ap_mAP50 = ap_data[0, i, :, :, :].mean()  # AP at IoU=0.50 across all max_dets and area ranges
        ap_mAP50_95 = ap_data[1:, i, :, :, :].mean()  # AP for IoU=0.5:0.95 across all max_dets and area range      
        print(f"{category_names[i]:<15}: {ap_mAP50:<10.5f} {ap_mAP50_95:<15.5f}")
    print() 
    print(f"Average Recall (AR) for Class mAP50 mAP50-0.95")
    ar_data = coco_eval.eval['recall']  # Get the recall values from COCOeval
    print("Shape of ar_data:", ar_data.shape)
    # Print the header
    print(f"{'Class':<15} {'mAP50':<10} {'mAP50-0.95':<15}")
    print("=" * 40)
    for i, cat_id in enumerate(cat_ids):
       # Corrected mAP calculations
        ar_mAP50 = ar_data[0, i, :, :].mean()  # AP at IoU=0.50 across all max_dets and area ranges
        ar_mAP50_95 = ar_data[1:, i, :, :].mean()  # AP for IoU=0.5:0.95 across all max_dets and area range      
        print(f"{category_names[i]:<15}: {ar_mAP50:<10.5f} {ar_mAP50_95:<15.5f}")
    print()          


              

def main():
    # Set up the main argument parser
    parser = argparse.ArgumentParser(description="YOLOv5 COCO mAP Computation") 
    parser.add_argument("--inp_ann" , type=str, default="datasets/vehicles_open_image/gt_vehicles_val.json", help="annotations file");
    parser.add_argument("--inp_pred", type=str, default="app/host_res/json_vehicles/vehicles_250images_detections_conf025_iou045.json", help="input preds (x1y1x2y2)");    
                
    # Parse known arguments
    args, unknown_args = parser.parse_known_args()

    # Check if any unknown arguments were supplied
    if unknown_args :
        print(f"WARNING: Only the '--mode' argument is allowed.")
        print(f"These Unknown Arguments will be ignored: {unknown_args}.")

    BASE_PATH          = "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files"          # starting folder
    COCO_GT_ANNOTATION = os.path.join(BASE_PATH, args.inp_ann) # COCO Ground Truth path

    COCO_VAL_PRED_X1Y1X2Y2 = os.path.join(BASE_PATH, args.inp_pred) # detections in X1 Y1 X2 Y2 format

    COCO_VAL_PRED_X1Y1WH   = os.path.join(BASE_PATH, args.inp_pred.split('.json')[0] + "_xywh.json" )

    print("GT annotations       file: ", COCO_GT_ANNOTATION)
    print("input preds (x1y1wh) file: ", COCO_VAL_PRED_X1Y1WH)
    print("tmp preds (x1y1x2y2) file: ", COCO_VAL_PRED_X1Y1X2Y2)

    convert_detections(COCO_VAL_PRED_X1Y1X2Y2, COCO_VAL_PRED_X1Y1WH)
    compute_map(COCO_GT_ANNOTATION, COCO_VAL_PRED_X1Y1WH)



# Define the main function to execute the evaluation
if __name__ == "__main__" :
    main()
 
 
     
