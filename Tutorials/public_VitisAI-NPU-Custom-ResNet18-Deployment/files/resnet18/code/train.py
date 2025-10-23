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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


import numpy as np
import cv2
from randaugment import RandAugment
import scipy.io as scio
from PIL import Image, ImageFile

## ImageNet size
IMG_W = np.short(224)
IMG_H = np.short(224)

## 3 MPixel size (/2 just to try)
#IMG_W = np.short(1920/2)
#IMG_H = np.short(1536/2)

## 8 MPixel size
#IMG_W = np.short(3840)
#IMG_H = np.short(2048)



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'}

def load_url(url, model_dir='./build/float/pretrained/', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    model_dict = torch.load(cached_file, map_location=map_location)
    del model_dict['fc.weight']
    del model_dict['fc.bias']
    return model_dict

def resnet18(pretrained=True, num_classes=196):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = models.resnet18(pretrained=False, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    return model

def resnet34(pretrained=True, num_classes=196):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = models.resnet34(pretrained=False, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, num_classes=196):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)



#def main():
    # Training settings

print("\nimage size is ", IMG_W, "cols x ", IMG_H, " rows\n" )
parser = argparse.ArgumentParser(description='PyTorch ResNet18 Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--backbone', type=str, default='resnet18', help='backbone from resnet18,resnet34,resnet50')
parser.add_argument('--data_root', type=str, default='./build/data', help='dataset')
parser.add_argument('--save_dir', type=str, default='./build/float', help='save model path')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--resume', type=str, default='', help='For resume model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_cuda else "cpu")
print(f"Training on {device} device.")


data_root = args.data_root

train_transform = transforms.Compose(
        [
         transforms.Resize([IMG_H,IMG_W]),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         ])

val_transform = transforms.Compose(
        [transforms.Resize([IMG_H,IMG_W]),
         transforms.ToTensor(),
         ])

train_set = datasets.ImageFolder(
            os.path.join(data_root,'train'),
            transform=train_transform)

test_set = datasets.ImageFolder(
            os.path.join(data_root, 'test'),
            transform=val_transform)

val_set = datasets.ImageFolder(
            os.path.join(data_root, 'val'), 
            transform=val_transform)

num_classes = len(train_set.classes)
print('train num:', len(train_set))
print('classes:', train_set.classes)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=4) #DB
val_loader   = torch.utils.data.DataLoader(val_set,  batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=4) #DB

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
    model.load_state_dict(torch.load(args.resume), strict=True)

from torchsummary import summary
summary(model, (3, IMG_H, IMG_W))


optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma) # {step_size} epoch decay *{gamma})
best = 0.0
early_stop = 0
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    #cur = test(model, device, test_loader) #DB
    cur  = test(model, device, val_loader) #DB
    scheduler.step()
    if cur > best and args.save_model:
        print('cur acc:', cur)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"color_{args.backbone}.pt"))
        best = cur
        early_stop = 0
    else:
        early_stop += 1

    if early_stop > 10:  #DB: #if early_stop > 20: 
        break

if args.save_model:
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"color_last_{args.backbone}.pt"))


#if __name__ == '__main__':
#    main()
