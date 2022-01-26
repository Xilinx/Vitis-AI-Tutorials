# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import torch.nn.functional as F
from PIL import Image
import code.utils as utils
from code.utils.misc import save_checkpoint
from code.utils.metrics import batch_pix_accuracy, pixel_accuracy, batch_intersection_union
from code.utils.lr_scheduler import LR_Scheduler
from code.utils.metrics import *
from code.utils.parallel import DataParallelModel, DataParallelCriterion
from code.datasets import get_segmentation_dataset

from code.configs.model_config import Options
import logging
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
#torch.backends.cudnn.benchmark = True
from pytorch_nndct import Pruner



def data_transform(img, im_size, mean, std):
    from torchvision.transforms import functional as FT
    img = img.resize(im_size, Image.BILINEAR)
    tensor = FT.to_tensor(img)  # convert to tensor (values between 0 and 1)
    tensor = FT.normalize(tensor, mean, std)  # normalize the tensor
    return tensor

def evaluate(val_loader, model, criterion):
    def eval_batch(model, image, target):
        outputs = model(image)

        if isinstance(outputs, tuple):# for aux
            outputs = outputs[0]
        h, w = target.size(1), target.size(2)
        outputs = F.upsample(input=outputs, size=(h, w), mode='bilinear', align_corners=True)
        target = target.cuda()
        correct, labeled = batch_pix_accuracy(outputs.data, target)
        inter, union = batch_intersection_union(outputs.data, target, args.num_classes)
        return correct, labeled, inter, union

    is_best = False
    model.eval()
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    tbar = tqdm(val_loader, desc='\r')
    for i, (image, target) in enumerate(tbar):
        if torch_ver == "0.3":
            image = Variable(image, volatile=True).cuda()
            correct, labeled, inter, union = eval_batch(model, image, target)
            
        else:
            with torch.no_grad():
                image = image.cuda()
                correct, labeled, inter, union = eval_batch(model, image, target)

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        tbar.set_description(
            'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
    return mIoU

def ana_eval_fn(model, val_loader, loss_fn):
    return evaluate(val_loader, model, loss_fn)

class Trainer():
    def __init__(self, args):
        self.prune_done = False
        self.initial_mio=0.0
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', root=args.data_folder, **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode ='val', root=args.data_folder, **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} 
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, \
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, \
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = args.num_classes
        self.best_pred = 0.0 
        self.best_filename=None
        if (args.prune_model_py is not None):
            import importlib
            pruned_model_py = importlib.import_module(args.prune_model_py)

            if args.model == "unet":
                model = pruned_model_py.UNET()
            else:     
                model = pruned_model_py.FPN()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            if args.model == "unet":
                from code.models import UNet
                model =  UNet(n_channels=3, n_classes=args.num_classes, bilinear=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:     
                from code.models import fpn  
                model = fpn.get_fpn(nclass=args.num_classes, backbone=args.backbone, pretrained=False)

                # optimizer using different LR
                params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
                if hasattr(model, 'head'):
                    params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
                optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.weight_decay)
        # criterions
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.model, self.optimizer = model, optimizer
    
        # using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        # resuming checkpoint
        if args.pruned_weights is not None:
            model.load_state_dict(torch.load(args.pruned_weights))

        if args.weight is not None:
            if not os.path.isfile(args.weight):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cuda:0')
            checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.initial_mio=self.best_pred
            print("=> loaded checkpoint '{}' (epoch {})" \
                  .format(args.weight, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, \
                                            args.epochs, len(self.trainloader), warmup_epochs=5)
        inputs = torch.randn([1, 3, args.crop_size, args.base_size], dtype=torch.float32)
        self.pruner = Pruner(self.model, inputs.to('cuda:0'))

    def analyze(self):
        from code.datasets import get_segmentation_dataset
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])

        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,'crop_size': args.crop_size}
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval', root=args.data_folder,**data_kwargs)
        loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        test_data = data.DataLoader(testset, batch_size=args.batch_size,drop_last=False, shuffle=False)
        self.pruner.ana(ana_eval_fn, args=(test_data, self.criterion))

    def prune(self,iteration,weights_file=None,last_snapshot_name=None):
        if (weights_file is not None and iteration==0):    
            #give the snapshots a new name if we are resuming pruning
            snapshot_name = './'+args.model+'_model_defs/pruned_'+str(iteration)+'resume.py'
            last_snapshot = args.model+'_model_defs.pruned_'+str(iteration)+'resume'
        else:
            snapshot_name = './'+args.model+'_model_defs/pruned_'+str(iteration)+'.py'
            last_snapshot = args.model+'_model_defs.pruned_'+str(iteration)
        model = self.pruner.prune(ratio=iteration*args.prune_ratio+args.prune_ratio, output_script=snapshot_name)
        self.prune_done = True
        return model,last_snapshot

    def training(self, epoch, model=None):
        train_loss = 0.0
        if model==None:
            #initial model
            self.model=self.model
        else:
            self.model=model

        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            image = image.cuda()
            target =  target.cuda()
            outputs = self.model(image)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))


    def validation(self, epoch,iteration):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            if isinstance(outputs, tuple):# for aux
                outputs = outputs[0]
            target = target.cuda()
            correct, labeled = batch_pix_accuracy(outputs.data, target)
            inter, union = batch_intersection_union(outputs.data, target, self.nclass)
            return correct, labeled, inter, union
        if epoch==0:
            #initialize the best snapshot at the start of each pruning phase
            self.best_pred = 0.0
            self.best_filename = './checkpoint/citys_reduced/'+args.model+'/epoch_'+str(epoch+1)+'_prune_iter_'+str(iteration)+'_sparse_ckpt.pth.tar'
            
        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True).cuda()
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    image = image.cuda()
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
        new_pred = mIoU
        if (self.initial_mio - mIoU)>0.02:
            print("Starting MIoU was: ", self.initial_mio )
            print("continue training, IoU loss is: ",(self.initial_mio - mIoU))
            continue_training = True
        else:
            self.best_pred=new_pred
            print("Starting MIoU was: ", self.initial_mio )
            print("training stopped, IoU loss is: ",(self.initial_mio - mIoU))
            continue_training = False

        if new_pred >= self.best_pred:
            is_best = False
            self.best_filename = './checkpoint/citys_reduced/'+args.model+'/epoch_'+str(epoch+1)+'_prune_iter_'+str(iteration)+'_sparse_ckpt.pth.tar'
            self.best_pred = new_pred
            torch.save(self.model.state_dict(), './checkpoint/citys_reduced/'+args.model+'/epoch_'+str(epoch+1)+'_prune_iter_'+str(iteration)+'_sparse_ckpt.pth.tar')
            #don't try to save the pruned_state_dict unless you've just run a round of pruning
            print("self.prune_done: ", self.prune_done)
            if (self.prune_done == True):
                torch.save(self.model.pruned_state_dict(), './checkpoint/citys_reduced/'+args.model+'/epoch_'+str(epoch+1)+'_prune_iter_'+str(iteration)+'_dense_ckpt.pth.tar')
        return self.best_filename, continue_training

if __name__ == "__main__":
    args = Options().parse()
    for key, val in args._get_kwargs():
        logging.info(key+' : '+str(val))
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    last_snapshot_name=None
    best_snapshot=None
    if (args.eval_pruned == True):
        from code.datasets import get_segmentation_dataset
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,'crop_size': args.crop_size}
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval', root=args.data_folder,**data_kwargs)
        loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        test_data = data.DataLoader(testset, batch_size=args.batch_size,drop_last=False, shuffle=False)
        evaluate(test_data,trainer.model,trainer.criterion)
        
    
    if (args.prune == True):
        if (args.prune_model_py is None and args.pruned_weights is None):
            trainer.analyze()
        for iteration in range(0,args.prune_iterations):
            epoch=0
            #if we are not passing in a pruned model and this is the first iteration of pruning, then just prune the current model
            if (args.prune_model_py is None and iteration==0):
                model,last_snapshot_name = trainer.prune(iteration)
                trainer.training(0,model)
            #else we are passing in a pruned model
            elif epoch>0:
                model,last_snapshot_name = trainer.prune(iteration,best_snapshot,last_snapshot_name)
                trainer.training(0,model)
            elif (epoch==0 and iteration==0):
                #in this case we are loading pruned weights and want to prune further from those:
                if args.pruned_weights is not None:
                    model,last_snapshot_name = trainer.prune(iteration,args.pruned_weights,args.prune_model_py)
                    trainer.training(0, model) 
            else:
                model,last_snapshot_name = trainer.prune(iteration,best_snapshot,last_snapshot_name)
                trainer.training(0,model) 
            best_snapshot, continue_training = trainer.validation(0,iteration)
            epoch=epoch+1
            while (continue_training and epoch < args.epochs):
                trainer.training(epoch, None)
                best_snapshot,continue_training = trainer.validation(epoch,iteration)
                epoch=epoch+1
                if epoch>=args.epochs:
                    print("training stopped as IoU loss could not be met within epoch limit")
