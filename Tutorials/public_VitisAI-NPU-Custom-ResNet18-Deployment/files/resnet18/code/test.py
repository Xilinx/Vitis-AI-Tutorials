#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright © 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
MIT License
'''

# last change: 02 Sep 2025

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms

import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
from resnet import resnet18, resnet34, resnet50



## ImageNet size
IMG_W = np.short(224)
IMG_H = np.short(224)

## 3 MPixel size (/2 just to try)
#IMG_W = np.short(1920/2)
#IMG_H = np.short(1536/2)

## 8 MPixel size
#IMG_W = np.short(3840)
#IMG_H = np.short(2048)


def test(model, device, test_loader, deploy=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #print("TARGET =", target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if deploy:
                return

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#def main():

print("\nimage size is ", IMG_W, "cols x ", IMG_H, " rows\n" )
# Training settings
parser = argparse.ArgumentParser(description='PyTorch ResNet18 Example')
parser.add_argument('--batch_size', type=int, default=19, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--backbone', type=str, default='resnet18', help='backbone from resnet18,resnet34,resnet50')
#parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)') #DB
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str, default='', help='For resume model')
parser.add_argument('--data_root', type=str, default='./build/data', help='dataset')
# old VitisAI 3.5 quantization, not compatible with VitisAI 2024.1
''' 
# DB
parser.add_argument("--device", default="gpu", type=str)
if args.deploy:
   args.test_batch_size = 1
'''
parser.add_argument("--deploy", action='store_true')
parser.add_argument("--device", default="cpu", type=str) #DB

args = parser.parse_args()

torch.manual_seed(args.seed)
if (args.device == 'gpu') and (torch.cuda.is_available()) :
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Testing on {device} device.")

data_root = args.data_root


val_transform = transforms.Compose(
             [transforms.Resize([IMG_H, IMG_W]),
             transforms.ToTensor(),
             ])

test_set = datasets.ImageFolder(
                os.path.join(data_root, 'test'), #DB: note the 'test' and not 'val'
                #os.path.join(data_root, 'val'),
                transform=val_transform)

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=4) #DB
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4) #DB
num_classes = len(test_set.classes)
print('test num:', len(test_set))
print('classes:', test_set.classes)

if args.backbone == 'resnet18':
    model = resnet18(num_classes=num_classes).to(device)
elif args.backbone == 'resnet34':
    model = resnet34(num_classes=num_classes).to(device)
elif args.backbone == 'resnet50':
    model = resnet50(num_classes=num_classes).to(device)
else:
    print('error')

#return

if args.resume != '':
    # model.load_state_dict(torch.load(args.resume), strict=True) # DB: original
    # Load the state dict with map_location set to 'cpu' # DB: added
    model.load_state_dict(torch.load(args.resume, map_location=torch.device('cpu')), strict=True)

#from torchsummary import summary
#summary(model, (3, IMG_H, IMG_W))


test(model, device, test_loader, args.deploy)


#return



#if __name__ == '__main__':
#    main()
