#!/usr/bin/env python3

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

import sys 
import os
import cv2
import torch
from PIL import Image
import numpy as np
import torch
from torchvision.ops import nms

WIDTH = 640             # Model required width
HEIGHT = 640            # Model required height
MEAN = 0.0
SCALE = 0.0039215
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

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

# *********************************************************************************************************
# Images
im2_path = "../app/images/zidane.jpg"
im1_path = "../app/images/bus.jpg"
#im1 = Image.open("../app/images/zidane.jpg")  # PIL image
im1 = cv2.imread(im1_path)[..., ::-1]  # OpenCV image (BGR to RGB)
im2 = cv2.imread(im2_path)[..., ::-1]  # OpenCV image (BGR to RGB)

# *********************************************************************************************************

print("\nUSING TORCH HUB LOAD AS REFERENCE\n")
# Model
#model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape='False')  # Change to your desired model
model.eval()
# Inference
results = model([im1, im2], size=640)  # batch of images
# Results
results.print()
results.save() # to Saved images to "runs/detect/expXX"
results.show() # open the image with Imagik
# show results as 
#      xmin    ymin    xmax   ymax  confidence  class    name
#print(f"results.xyxy[0]:\n${results.xyxy[0]}")  # im1 predictions (tensor)
print(f"results.pandas().xyxy[0]:\n${results.pandas().xyxy[0]}")  # im1 predictions (pandas)
print(f"results.pandas().xywh[0]:\n${results.pandas().xywh[0]}")  # im1 predictions (pandas)
print(f"results.pandas().xyxy[1]:\n${results.pandas().xyxy[1]}")  # im2 predictions (pandas)
print(f"results.pandas().xywh[1]:\n${results.pandas().xywh[1]}")  # im2 predictions (pandas)


# *********************************************************************************************************
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
    image= np.array(image)
    # Normalize
    image = (image - MEAN) * SCALE # 1/255
    # Change shape to N CHW format
    # Model input is NCHW but the JPEG is in HWC
    image = np.transpose(image, (2, 0, 1))  # Change shape to 3x640x640
    image = np.expand_dims(image, axis=0)   # Change shape to 1x3x640x640
    #return torch.tensor(image, dtype=torch.float32)
    return image_shape, image

# Post-processing function to handle YOLOv5 outputs from Host
def host_post_process(image_id, pred, input_size, original_size, conf_thres=0.5, iou_thres=0.4):
   # Extract components
   boxes = pred[..., :4].clone()
   scores = pred[..., 4].clone()
   class_probs = pred[..., 5:].clone()
   # Convert center-based format (xc yc w h) to corner-based format (X1 Y1 W H)
   # Calculate x1, y1
   boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
   boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
   # Calculate x2, y2 directly from x1, y1
   boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
   boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
   # Scale to original image size (if needed)
   scale_x = original_size[1] / input_size[1]
   scale_y = original_size[0] / input_size[0]
   boxes[:, [0, 2]] *= scale_x  # Scale x1 and x2
   boxes[:, [1, 3]] *= scale_y  # Scale y1 and y2
   # Apply NMS which needs X1 Y1 X2 Y2
   keep = nms(boxes.clone().detach(), scores.clone().detach(), iou_thres)
   keep = keep[scores[keep] > conf_thres]
   # Extract class predictions
   final_boxes = boxes.clone().detach()[keep]
   final_scores = scores.clone().detach()[keep]
   final_classes = torch.argmax(class_probs.clone().detach()[keep], dim=1)
   int_boxes = final_boxes #final_boxes.to(torch.int32)
   detections = []
   for i in range(len(final_boxes)) : 
    detections.append({ 
            #'image_id': image_id,
            'box': int_boxes[i].tolist(), # X1 Y1 X2 Y2 format
            'score': final_scores[i].item(),
            'class': final_classes[i].item(),
            'class_name': COCO_CLASSES[final_classes[i].item()] 
    })
   print(f"Number of detections after nms: {len(detections)}")
   return detections

# Convert from [x1, y1, x2, y2] to [xc, yc, w, h]
def convert_xy_to_cwh(boxes):
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1          # Height
    #return torch.stack((xc, yc, w, h), dim=1)
    return [xc, yc, w, h]

print("\nUSING EXPERIMENTAL LOAD\n")

#Add the yolov5 directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5')))
from models.experimental import attempt_load

# Model
model2 = attempt_load('../yolov5/yolov5s.pt')  # Change to your desired model
model2.eval() 

# Inference

# tuple of image file extensions to consider
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
# files of images
inp_images_dir = "../app/images/"
image_files = [ file for file in os.listdir(inp_images_dir) if file.lower().endswith((image_extensions)) ]
## sort the images files
#image_files.sort() 
images_count = len(image_files)
print(f"Number of images: {images_count}")
total_detections = []
input_size=(WIDTH, HEIGHT)
# Define the headers
header_x1y1x2y2 = f"{'$':<4} {'xmin':<10} {'ymin':<10} {'xmax':<10} {'ymax':<10} {'confidence':<12} {'class':<6} {'name'}"
header_xcycwh   = f"{'$':<4} {'xcenter':<10} {'ycenter':<10} {'width':<9} {'height':<8} {'confidence':<12} {'class':<6} {'name'}"

for index, image_file in enumerate(image_files, start=1) :
    
    image_path = os.path.join(inp_images_dir, image_file)
    print(f"\rProcessing image {index}/{images_count}", end='')  # Print on the same line #DB  
    # Preprocess the image
    img_shape, prepr_image = preprocess_image(image_path)
    im_tensor = torch.tensor(prepr_image).float() 
    # Perform inference
    with torch.no_grad():
        results = model(im_tensor)  # Run inference: results are in [Xc Yc W H] format
        im_h, im_w = img_shape
        possible_id = os.path.basename(image_path).split('.')[0]   
        if possible_id.isdigit() : 
            image_id= int(possible_id)  # Use the image name as ID
        else :
            image_id = -1
        print(f" image_id = {image_id}, image_path={image_path}") 
        # Loop through each tensor in the list
        for idx, tensor in enumerate(results):
            # post process with NMS which needs [X1 Y1 X2 Y2] format
            detections = host_post_process(image_id, tensor.squeeze(0), input_size, img_shape, CONF_THRESHOLD, IOU_THRESHOLD)
            #print(f"tensor[${idx}]=\n${res}")
            print(header_x1y1x2y2)
            for i, det in enumerate(detections):
                xyxy = det['box'] # x1 y1 x2 y2
                x1 = xyxy[0]
                y1 = xyxy[1]
                x2 = xyxy[2]
                y2 = xyxy[3]
                score = float(det['score'])
                category_id = int(det['class'])
                category_name = det['class_name']
                print(f"{i:2d} {x1:12.6f} {y1:12.6f} {x2:12.6f} {y2:12.6f} {score:12.6f} {category_id:2d} {category_name}")            
            print(header_xcycwh)                
            for i, det in enumerate(detections):
                xyxy = det['box'] # x1 y1 x2 y2
                boxes_xy = xyxy #torch.tensor(xyxy, dtype=torch.float32)
                [xc, yc, w, h] =convert_xy_to_cwh(boxes_xy)
                score = float(det['score'])
                category_id = int(det['class'])
                category_name = det['class_name']
                print(f"{i:2d} {xc:12.6f} {yc:12.6f} {w:12.6f} {h:12.6f} {score:12.6f} {category_id:2d} {category_name}")       

print("\nEND\n")                     













