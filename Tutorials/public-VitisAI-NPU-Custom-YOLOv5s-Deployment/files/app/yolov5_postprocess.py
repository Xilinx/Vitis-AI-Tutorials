#!/usr/bin/env python3

# Copyright © 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# updated on 02 Sep 2025

# This script is intended for reference purposes only and is not suitable for production use.
# It reads the output of a YOLOv5 PyTorch ONNX model from a binary file, processes the output to extract
# bounding boxes, objectness scores, and class probabilities, applies non-maximum suppression (NMS)
# to filter out overlapping boxes, and formats the final detections with class names from the COCO
# dataset. The script then annotates the input image with the detected objects and
# saves the annotated image.

# Usage:
#  python3 postprocess_yolov5.py --pred_data <raw model output> --image <jpeg input>

import torch
import numpy as np
import cv2
from torchvision.ops import nms
from pathlib import Path
import argparse

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

# Function to read predictions from a binary file
def read_predictions(bin_file, input_size=640):
    # Read raw prediction data
    pred_data = np.fromfile(bin_file, dtype=np.float32)
    # Shape should be (1, 25200, 85): 25200 predictions with 85 values (classes + bbox + confidence)
    pred_data = pred_data.reshape(1, 25200, 85)
    return torch.tensor(pred_data)

# Post-processing function to handle YOLOv5 ONNX/PL outputs
def post_process(pred, input_size, original_size, conf_thres=0.5, iou_thres=0.4):
    # Extract components
    boxes = pred[..., :4]
    scores = pred[..., 4]
    class_probs = pred[..., 5:]

    # Convert center-based format to corner-based format
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2

    # Scale to original image size
    scale_x = original_size[1] / input_size
    scale_y = original_size[0] / input_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # Apply NMS
    keep = nms(boxes.clone().detach(), scores.clone().detach(), iou_thres)
    keep = keep[scores[keep] > conf_thres]

    # Extract class predictions
    final_boxes = boxes.clone().detach()[keep]
    final_scores = scores.clone().detach()[keep]
    final_classes = torch.argmax(class_probs.clone().detach()[keep], dim=1)

    detections = []
    for i in range(len(final_boxes)):
        detections.append({
            'box': final_boxes[i].tolist(),
            'score': final_scores[i].item(),
            'class': final_classes[i].item(),
            'class_name': COCO_CLASSES[final_classes[i].item()]
        })

    return detections

def detect_and_annotate(bin_file, image_path, input_size=640, conf_thres=0.5, iou_thres=0.4):
    image = cv2.imread(image_path)
    im_h, im_w, _ = image.shape  # Store original image dimensions

    # Read raw predictions from the binary file
    predictions = read_predictions(bin_file, input_size)
    predictions = predictions.squeeze(0)  # Remove batch dimension

    detections = post_process(predictions, input_size, (im_h, im_w), conf_thres, iou_thres)

    # Annotate the image
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        label = f"{det['class_name']} {det['score']:.2f}"
        color = (0, 0, 255)  # Red color in BGR

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        # Draw the label inside the box at the top-left corner
        cv2.putText(image, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Generate output file name
    output_image_path = Path(image_path).stem + "_detection_results.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved to: {output_image_path}")

    # Print detections
    for det in detections:
        print(f"Class: {det['class_name']}, Score: {det['score']:.2f}, Box: {det['box']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolov5 Object Detection and Annotation")
    parser.add_argument("--pred_data", type=str, required=True, help="Path to the binary prediction file")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Run detection and annotation
    detect_and_annotate(args.pred_data, args.image)
