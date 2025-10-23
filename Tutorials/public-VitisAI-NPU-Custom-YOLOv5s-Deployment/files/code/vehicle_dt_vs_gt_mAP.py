#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

"""
Usage
cd yolov5
python3 ../code/vehicle_dt_vs_gt_mAP.py --inp_json best_predictions.json
python3 ../code/vehicle_dt_vs_gt_mAP.py --inp_json ep20_bs32_best_predictions.json
"""

'''
This script 

1)  takes the "best_predictions.json", created by "val.py" at runtime,  and convert them  
    in the more convenient format of "new_best_predictions.json", 
    without remapping the category_id as it seems the correct one already
    (but this should be further checked with "check_coco_json.py", not yet done).
2)  then it computes the mAP against the "gt_vehicles_val.json" GT file, created by "convert2_text_gt_into_coco_json.py"
'''


# Import the necessary packages
import os
import sys
import json
import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# Replace COCO_CLASSES with categories specific to Vehicles-OpenImage dataset
VEHICLE_CLASSES =  ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

# Category mapping as a dictionary for Vehicles-OpenImage dataset
category_map = {
    "Ambulance": 1,
    "Bus": 2,
    "Car": 3,
    "Motorcycle": 4,
    "Truck": 5
}


# Convert YOLOv5 detection 
def convert_detections(input_json_path, output_json_path):
    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        detections = json.load(f)
    # Prepare a list for converted detections
    coco_annotations = []
    # Process each detection
    for detection in detections:
        image_id = detection['image_id']
        bbox = detection['bbox']
        yolo_category_id = detection['category_id']
        if yolo_category_id ==0 :
            print(f"found category_id = {yolo_category_id} in image_id {image_id}")
        coco_category_id = category_map[VEHICLE_CLASSES[yolo_category_id]]
        score = detection['score']
        #coco_category_id = yolo_category_id  
        #print(f"category_id VEHICLES: {yolo_category_id} of class {VEHICLE_CLASSES[yolov5_category_id]} remapped as COCO: {coco_category_id}")       
        # Append new annotation
        coco_annotation = {
            "image_id": image_id,
            "category_id": coco_category_id,                         
            #"category_id": yolo_category_id,                         
            "bbox": bbox, 
            "score": score
        }
        coco_annotations.append(coco_annotation)
        
    # Write the converted detections to the output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_annotations, f, indent=2)

# Compute mean Average Precision (mAP) using COCO evaluation and predictions 
# files, both in X1 Y1 W H format
def compute_map(annotations_file, predictions_file):
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
    # Print the header
    print(f"{'Class':<15} {'mAP50':<10} {'mAP50-0.95':<15}")
    print("=" * 40)
    for i, cat_id in enumerate(cat_ids):
        # AP at IoU=0.50:0.95
        mAP50_95 = ap_data[:, i, :].mean()  # mean across all max_dets
        # AP at IoU=0.50
        mAP50 = ap_data[0, i, :].mean()  # only for IoU=0.50 and all max_dets
        print(f"{category_names[i]:<15}: {mAP50:<10.3f} {mAP50_95:<15.3f}")
    print()

'''
# Print the results in a consolidated format
print(f"{'Class':<15} {'Images':<10} {'Instances':<12} {'P':<8} {'R':<8} {'mAP50':<10} {'mAP50-95':<10}")
print("=" * 80)

# Print the results for each class
for class_name in ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]:
    print(f"{class_name:<15} {image_counts[class_name]:<10} {instance_counts[class_name]:<12} "
          f"{precision[class_name]:<8.3f} {recall[class_name]:<8.3f} "
          f"{mAP_results[class_name]['mAP50']:<10.3f} {mAP_results[class_name]['mAP50_95']:<10.3f}")
'''


def main():

   # Set up the main argument parser
   parser = argparse.ArgumentParser(description="Computing Vehicles YOLOv5 mAP from Ultralytics with pyCOCOtools") 
   parser.add_argument("--inp_json", type=str, 
                       default = "best_predictions.json",
                       help='predictions input json file')
   # Parse the arguments
   args = parser.parse_args()
   filename  = args.inp_json
   json_name1 = os.path.join("datasets/vehicles_open_image", filename) 
   json_name2 = os.path.join("datasets/vehicles_open_image", "new_"+filename) 
   
   base_path = "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files"  # starting folder
   coco_gt_annotation   = os.path.join(base_path, "datasets/vehicles_open_image/gt_vehicles_val.json") 
   coco_val_pred_x1y1wh = os.path.join(base_path, json_name1)  
   out_json_file        = os.path.join(base_path, json_name2)
   
   # Convert detections from YOLO format to COCO format
   convert_detections(coco_val_pred_x1y1wh, out_json_file)
   # Compute mean Average Precision (mAP) using COCO evaluations
   compute_map(coco_gt_annotation, out_json_file)


# Define the main function to execute the evaluation
if __name__ == "__main__":
    main()
