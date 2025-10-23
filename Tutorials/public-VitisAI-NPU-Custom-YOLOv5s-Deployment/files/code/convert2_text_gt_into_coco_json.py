#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# updated on 02 Sep 2025 
 
import os
import json
import cv2

# Define the class names
class_names = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

# Category mapping as a dictionary for Vehicles-OpenImage dataset
category_map = {
    "Ambulance": 1,
    "Bus": 2,
    "Car": 3,
    "Motorcycle": 4,
    "Truck": 5
}


# Prepare the COCO structure
coco_data = {
    "info": {
        "description": "Vehicles Dataset",
        "url": "https://public.roboflow.com/object-detection/vehicles-openimages",
        "version": "1.0",
        "year": 2025,
        "contributor": "SpiderMan",
        "date_created": "2025/02/01"
    },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial License"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": []
}

def create_coco_json(images_path, labels_path, output_file) :
    # Create categories
    for i, class_name in enumerate(class_names):
        category_id = category_map[class_name]
        coco_data["categories"].append({
            "supercategory": "vehicle",
            "id": category_id,
            "name": class_name
        })

    # Iterate through images and their labels to fill COCO format
    annotation_id = 1  # Unique ID for annotations
    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") :
            # Gather image information
            #image_id = len(coco_data["images"]) + 1  # Image IDs should start from 1
            image_id = filename.split(".jpg")[0]
            image_file = os.path.join(images_path, filename)
            img = cv2.imread(image_file)   
            if img is not None :
                # Get the width and height
                image_height, image_width, _ = img.shape  # shape gives (height, width, channels)
                #print(f"Width: {image_width}, Height: {image_height}")
            else:
                print(f"Error: Image {image_file} could not be loaded.")                          
    
            # Append image data
            coco_data["images"].append({
                "id": image_id,
                #"license": 1,  # Assuming license id 1
                "width": image_width,                
                "height": image_height,
                "file_name": filename,
                #"coco_url": f"http://dummy_coco_url_{filename}",
                #"flickr_url": "http://dummy_flicker_url_{filename}",
                "date_captured": "2025-01-31 10:02:30", 
            })
        
            # Load corresponding label file
            label_file = os.path.join(labels_path, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        #if class_id == 0 :
                        #    print(f"found class_is=${class_id} in file name {label_file}")
                        remapped_class_id = category_map[class_names[class_id]]
                        center_x = float(parts[1]) * image_width  # Scale to pixel coordinates
                        center_y = float(parts[2]) * image_height  # Scale to pixel coordinates
                        width    = float(parts[3]) * image_width      # Scale to pixel coordinates
                        height   = float(parts[4]) * image_height    # Scale to pixel coordinates                        
                        # Calculate bounding box
                        x_min = center_x - width / 2
                        y_min = center_y - height / 2

                        # Append annotation data
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "category_id": remapped_class_id,
                            "iscrowd": 0,  # Set to 1 if the object is a crowd
                            "image_id": image_id,
                            "area": width * height,  # Area of the bounding box
                            "bbox": [x_min, y_min, width, height],  # [x, y, width, height]
                        })
                        annotation_id += 1
            else:
                print(f"Warning: No label file found for image {image_file}")
    # Save COCO JSON to file
    with open(output_file, 'w') as json_file:
        json.dump(coco_data, json_file, indent=2)
    print(f"COCO JSON file has been created as {output_file}")


# Main function to run the script
if __name__ == "__main__":

    BASE_PATH = "/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files"       # starting folder
    labels_dir   = os.path.join(BASE_PATH, "datasets/vehicles_open_image/valid/labels")  # Directory
    images_dir   = os.path.join(BASE_PATH, "datasets/vehicles_open_image/valid/images")  # Directory
    output_json  = os.path.join(BASE_PATH, "datasets/vehicles_open_image/gt_vehicles_val.json") # Output file for     
    
    create_coco_json(images_dir, labels_dir, output_json)
