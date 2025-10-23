#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright © 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
MIT License
'''

# last change: 02 Sep 2025


import argparse
import os
import sys
import torch
from torchvision import models, transforms
from PIL import Image

# Set the MAX  number of images to process per class: 50 on ImageNet validation Dataset
NB_IMAGES = 1

# this script must start from this folder
# REQUIRED_DIR="/home/jetson/danieleb/vai2024.1/resnet18"
REQUIRED_DIR="/workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18"

# get the current directory "cur_dir" and its absolute path
cur_dir = os.getcwd()
absolute_path=os.path.abspath(cur_dir)
print(f"current directory: {cur_dir}")
print(f"Absolute Path of current directory: {absolute_path}")

# check if the current dir matches the required dir
if absolute_path != REQUIRED_DIR : 
    print(f"ERROR: this script must be run from {REQUIRED_DIR}")
    sys.exit(1)


# Define the path to your ImageNet validation dataset
#validation_dir = "/workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/imagenet/val" 
validation_dir = os.path.join(cur_dir, "../imagenet/val") 

# define the file with the words of classes related to the image file names:
#words_file = "/workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/imagenet/synset_words.txt"
words_file = os.path.join(cur_dir, "../imagenet/synset_words.txt")

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run inference on ImageNet validation dataset with ResNet models.')
parser.add_argument('--model', type=str, choices=['resnet18', 'resnet50'], default='resnet18',
                    help='Model to use for inference: resnet18 or resnet50.')
parser.add_argument('--nb_images', type=int, default=NB_IMAGES,
                    help='Number of images to process per class.')
args = parser.parse_args() 

nb_images = args.nb_images

# Load the selected pre-trained model
if args.model == 'resnet18':
    model = models.resnet18()
    model_path = os.path.join(cur_dir, "build/float/pretrained/resnet18-f37072fd.pth")
    model.load_state_dict(torch.load(model_path))
else:  # default is resnet50
    model = models.resnet50(pretrained=True)


model.eval()  # Set the model to evaluation mode

# Define image transformations for image pre-processing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labels (Classes) for ImageNet
labels_folder_name = []
labels_class_name = []
with open(words_file) as f: 
    for line in f.readlines() :
        # print(f"{line}")
        parts = line.split(" ", 1) # Split on the first space only
        # Check if there are at least two parts
        if len(parts) == 2:
            first_part = parts[0]
            second_part = parts[1]
        else:
            first_part = parts[0]  # If no space is found, the whole string is the first part
            second_part = ''        # No second part
        labels_folder_name.append(first_part)
        labels_class_name.append(second_part)

# Inference function
def infer_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():  # Disable gradient computation
        output = model(image_tensor)  # Get model predictions

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item(), output


# Accuracy calculation variables
correct_predictions = 0
total_images = 1

# Iterate through images in the validation directory and perform inference
for class_dir in os.listdir(validation_dir):
    class_path = os.path.join(validation_dir, class_dir)
    
    if os.path.isdir(class_path):
        processed_images_count = 1  # Counter for images processed in the current class
        for image_file in os.listdir(class_path):
            if processed_images_count > nb_images:  # Check if the limit is reached
                break  # Stop processing more images in this class
            image_path = os.path.join(class_path, image_file)
            #print(f"\rProcessing: {image_path}", end='') # Print on the same line #DB
            print(f"\rProcessing: {processed_images_count}/{total_images} images from folder {os.path.basename(class_path)}", end='')

            predicted_idx, output = infer_image(image_path)
            predicted_label = labels_class_name[predicted_idx]
            predicted_confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx].item()
            #print(f'Predicted Label: {predicted_label}, Confidence: {predicted_confidence:.4f}')

            # Check if the prediction is correct
            true_label_idx = labels_folder_name.index(class_dir)  # Assuming the directory name corresponds to the class.
            if predicted_idx == true_label_idx:
                correct_predictions += 1
            total_images += 1
            processed_images_count += 1  # Increment the processed images counter
        #print()
        #print(f"found {processed_images_count} images into folder {class_path}")

# Calculate top-1 average accuracy
print()
if total_images > 0 :
    top_1_accuracy = correct_predictions / total_images * 100  # Convert to percentage
    print(f'Top-1 Average Accuracy: {top_1_accuracy:.2f}%')
else :
    print("No images processed.")

print("THE END")





