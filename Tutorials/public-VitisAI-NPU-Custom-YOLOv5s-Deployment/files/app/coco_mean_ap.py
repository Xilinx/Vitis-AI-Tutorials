#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

"""
Usage:
cd files/app
python3 coco_mean_ap.py  --inp_xyxy pred_x1y1x2y2.json 
python3 coco_mean_ap.py  --inp_ann annotations.json --inp_xyxy pred_x1y1x2y2.json 
"""

# Import the necessary packages
import os
import sys
import json
import argparse
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

# COCO_CLASSES from YOLOv5 model
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Category mapping as a dictionary
category_mean_ap = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90
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
        coco_category_id =  category_mean_ap.get(COCO_CLASSES[yolov5_category_id])
        print(f"category_id YOLOV5: {yolov5_category_id } of class {COCO_CLASSES[yolov5_category_id]} remapped as COCO: {coco_category_id}")         
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
    print("Shape of ap_data:", ap_data.shape)

    # Print the header
    print(f"{'Class':<15} {'mAP50':<10} {'mAP50-0.95':<15}")
    print("=" * 40)
    for i, cat_id in enumerate(cat_ids):
        ap_mAP50 = ap_data[0, i, :, :, :].mean()  # AP at IoU=0.50 across all max_dets and area ranges
        ap_mAP50_95 = ap_data[1:, i, :, :, :].mean()  # AP for IoU=0.5:0.95 across all max_dets and area range            
        print(f"{category_names[i]:<15}: {ap_mAP50:<10.5f} {ap_mAP50_95:<15.5f}")
        if i > 20:
            break
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
        if i> 20 :
            break
    print()                 


def main():
    # Set up the main argument parser
    # --inp_ann annotations.json --inp_xyxy pred_x1y1x2y2.json --out_xywh pred_xywh.json
    parser = argparse.ArgumentParser(description="YOLOv5 COCO mAP Computation") 
    parser.add_argument("--inp_ann" , type=str, default="datasets/coco/annotations/instances_val2017.json", help="annotations file");
    parser.add_argument("--inp_pred", type=str, default="app/host_res/json_coco/coco_5000images_detections_conf025_iou045.json", help="input preds (x1y1x2y2)");    
        
    # Parse known arguments
    args, unknown_args = parser.parse_known_args()

    # Check if any unknown arguments were supplied
    if unknown_args :
        print(f"WARNING: Only the '--mode' argument is allowed.")
        print(f"These Unknown Arguments will be ignored: {unknown_args}.")

    BASE_PATH          = "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files"          # starting folder
    COCO_GT_ANNOTATION = os.path.join(BASE_PATH, args.inp_ann) # COCO Ground Truth path

    COCO_VAL_PRED_X1Y1X2Y2 = os.path.join(BASE_PATH, args.inp_pred) # detections in X1 Y1 X2 Y2 format

    COCO_VAL_PRED_X1Y1WH   = os.path.join(BASE_PATH, args.inp_pred.split('.json')[0] + "_x1y1x2y2.json" )

    
    convert_detections(COCO_VAL_PRED_X1Y1X2Y2, COCO_VAL_PRED_X1Y1WH)
    compute_map(COCO_GT_ANNOTATION, COCO_VAL_PRED_X1Y1WH)

    print("GT annotations       file: ", COCO_GT_ANNOTATION)
    print("input preds (x1y1wh) file: ", COCO_VAL_PRED_X1Y1WH)
    print("tmp preds (x1y1x2y2) file: ", COCO_VAL_PRED_X1Y1X2Y2)


# Define the main function to execute the evaluation
if __name__ == "__main__" :
    main()
 
 
 
 
