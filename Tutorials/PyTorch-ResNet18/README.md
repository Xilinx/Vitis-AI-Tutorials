<!--

Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

Author: Daniele Bagni, Xilinx Inc
-->


<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis™ AI Tutorials</h1>
    </td>
 </tr>
</table>

#  ResNet18 in PyTorch from Vitis AI Library

- Version:      Vitis AI 3.0 with Pytorch 1.12.1

- Support:      ZCU102, ZCU102, VCK190, VEK280, V70

- Last update:  05 July 2023


## Table of Contents

[1 Introduction](#1-introduction)

[2 Prerequisites](#2-prerequisites)

[3 The Docker Tools Image](#3-the-docker-tools-image)

[4 The VCoR Dataset](#4-the-vcor-dataset)

[5 Vehicle Color Classification with ResNet18](#5-vehicle-color-classification-with-resnet18)

[License](#license)



## 1 Introduction

### 1.1 Rationale

In this Deep Learning (DL) tutorial you will take the **ResNet18** CNN, from the [Vitis AI 3.0 Library](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk),   and use it to classify the different colors of the "car object" inside images by running the inference application on FPGA devices.

Although ResNEt18 was already trained on the [ImageNet](https://www.image-net.org/) dataset in the **PyTorch** framework, you will re-train it using the
**Kaggle' Vehicle Color Recognition** dataset, shortened as  [VCoR](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset). 	


### 1.2 The Vitis AI Flow

The [Vitis AI 3.0](https://github.com/Xilinx/Vitis-AI) deployment process requires to quantize the floating point CNN model into INT8 (signed 8-bit) fixed point and then compile the application and run it on the embedded system composed by the Deep Processor Unit (DPU) and the ARM CPU of the target board.
Different target boards (for example VCK190 and ZCU102) apply different architectures (respectively DPUCVDX8G and .... for this example) for the DPU IP soft core.

Assuming you have already trained your CNN and you own its original pytorch model, typically a [PTH](https://fileinfo.com/extension/pth) file with extension ``.pt``, you will deploy such CNN on the FPGA target boards by following these steps:

  1. (optional) Run the **Vitis AI Inspector** to check if the original model is compatible with the AMD/Xilinx **Deep Processor Unit (DPU)** architecture available on the target board (if not you have to modify your CNN and retrain it).

  2. Run the **Vitis AI Quantizer** to generate a 8-bit (named **Int8**) fixed point model of the CNN from the original 32-bit floating point model. If you apply the so called *Post-Training Quantization* (PTQ), this will be a single step, otherwise you would need to re-train - or more properly said "fine-tune" - the CNN with the *Quantization-Aware-Training* (QAT),

  3. (optional) Run inference with the quantized INT8 model on the Vitis AI environment (running on the host desktop) to check the prediction accuracy: if the difference is not negligible (for example it is larger than 5%, you have to re-do the quantization by replacing PTQ with QAT).  

  4. Run the **Vitis AI Compiler**  on the INT8 model to generate the ``.xmodel`` microcode for the DPU IP soft-core on your target board.

  5. Compile the application running on the ARM CPU - tightly coupled with the DPU - of the target board by using either C++ or Python code with the **Vitis AI RunTime** (VART) APIs.  

Based on the above mentioned process, you will be able to measure the inference performance both in terms of average prediction accuracy and frames-per-second (fps) throughput on your target board.

All the commands reported in this document are also collected into the [run_all.sh](files/run_all.sh) script.


### 1.3 Starting from the Model Zoo

From the [Model Zoo list](https://github.com/Xilinx/Vitis-AI/tree/3.0/model_zoo/model-list) select  the [pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0](https://github.com/Xilinx/Vitis-AI/tree/3.0/model_zoo/model-list/pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0) hyperlink and get the related [model.yaml](https://github.com/Xilinx/Vitis-AI/tree/3.0/model_zoo/model-list/pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/model.yaml) file which shows from where you can download the floating point pre-trained CNN model and its pre-quantized  versions, as well as pre-compiled models for different target boards:

```shell
# get the floating point model
wget  https://www.xilinx.com/bin/public/openDownload?filename=pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0.zip

# get the quantized model for ZCU102
wget https://www.xilinx.com/bin/public/openDownload?filename=chen_color_resnet18_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz

# get the quantized model for VCK190
wget https://www.xilinx.com/bin/public/openDownload?filename=chen_color_resnet18_pt-vck190-r3.0.0.tar.gz
```





## 2 Prerequisites

- Accurate reading of this [README.md](README.md) file from the top to the bottom, before running any script.

- Host PC with Ubuntu >= 18.04.5 (and possibly with GPU support to run the CNN training).

- The entire repository of [Vitis AI 3.0](https://github.com/Xilinx/Vitis-AI) stack from [www.github.com/Xilinx](https://www.github.com/Xilinx) web site.

-  Accurate reading of [Vitis AI Stack User Guide 1414](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Overview) (shortly UG1414).

-  Accurate reading of [Vitis AI Library User Guide 1354](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk) (shortly UG1354).

- Accurate reading of [online Vitis AI](https://xilinx.github.io/Vitis-AI) documentation from [xilinx.github.io](https://xilinx.github.io) web site. In particular, pay attention to the installation and setup instructions for both host PC and target board, it is recommended you build a GPU-based docker image with PyTorch.

- A Vitis AI target board such as either:
    - [ZCU102](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html), or
    - [ZCU104](https://www.xilinx.com/products/boards-and-kits/zcu104.html), or
    - [VCK190](https://www.xilinx.com/products/boards-and-kits/vck190.html), or
    - the newest Versal EDGE AI board **VEK280**, available only on [Versal AI Early Access Lounge](https://www.xilinx.com/member/vitis-ai-vek280.html) (you need to register to it first);
    - the newest **Alveo V70** card, available only on [Versal AI Early Access Lounge](https://www.xilinx.com/member/vitis-ai-vek280.html) (you need to register to it first).


- Familiarity with Deep Learning principles.


### 2.1 Working Directory

In the following of this document it is assumed you have installed Vitis AI somewhere in your file system and this will be your working directory ``<WRK_DIR>``, for example in my case ``<WRK_DIR>`` is set to
``/media/danieleb/DATA/VAI3.0``.  You have also created a folder named ``tutorials`` under such ``<WRK_DIR>`` and you have copied this tutorial there and renamed it ``PyTorch-ResNet18``:

```text
VAI3.0  # your WRK_DIR
├── board_setup
├── demos
├── docker
├── docs
├── docsrc
├── dpu
├── examples
│   ├── custom_operator
│   ├── ofa
│   ├── vai_library
│   ├── vai_optimizer
│   ├── vai_profiler
│   ├── vai_quantizer
│   ├── vai_runtime
│   ├── waa
│   └── wego
├── model_zoo
├── src
├── third_party
└── tutorials #folder created by you
    ├── PyTorch-ResNet18    # this tutorial
            ├── README.md   # this file
            ├── files       # project files
                ├── build
                ├── code
                ├── doc
                ├── scripts
                ├── target

```

### 2.2 Dos-to-Unix Conversion

In case you might get some strange errors during the execution of the scripts, you have to process -just once- all the``*.sh`` shell and the python ``*.py`` scripts with the [dos2unix](http://archive.ubuntu.com/ubuntu/pool/universe/d/dos2unix/dos2unix_6.0.4.orig.tar.gz) utility.
In that case run the following commands from your Ubuntu host PC (out of the Vitis AI docker images):

```text
sudo apt-get install dos2unix
cd <WRK_DIR>/tutorials/RESNET18 #your repo directory
for file in $(find . -name "*.sh"); do
  dos2unix ${file}
done
for file in $(find . -name "*.py"); do
  dos2unix ${file}
done
for file in $(find . -name "*.c*"); do
  dos2unix ${file}
done
for file in $(find . -name "*.h*"); do
  dos2unix ${file}
done
```

These operations are already included in the script [clean_all.sh](files/scripts/clean_all.sh).


## 3 The Docker Tools Image

You have to know few things about [Docker](https://docs.docker.com/) in order to run the Vitis AI smoothly on your host PC environment.

### 3.1 Build the Image

From the Vitis AI 3.0 repository, run the following commands:

```text
cd <WRK_DIR>
cd docker
./docker_build.sh -t gpu -f pytoch
```

Once the process is finished, with the command ``docker images`` you should see something like this:

```text
REPOSITORY                        TAG         IMAGE ID       CREATED         SIZE
xilinx/vitis-ai-pytorch-gpu   3.0.0.001   1b99612d429a   27 hours ago    21.4GB
```


Note that docker does not have an automatic garbage collection system as of now. You can use this command to do a manual garbage collection:
```
docker rmi -f $(docker images -f "dangling=true" -q)
```

### 3.2 Launch the Docker Image

To launch the docker container with Vitis AI tools, execute the following commands from the ``<WRK_DIR>`` folder:

```text
cd <WRK_DIR> # you are now in Vitis_AI subfolder
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest
conda activate vitis-ai-pytorch
cd /workspace/tutorials/
cd PyTorch-ResNet18 # your current directory
```

Note that the container maps the shared folder ``/workspace`` with the file system of the Host PC from where you launch the above command.
This shared folder enables you to transfer files from the Host PC to the docker container and vice versa.

The docker container does not have any graphic editor, so it is recommended that you work with two terminals and you point to the same folder, in one terminal you use the docker container commands and in the other terminal you open any graphic editor you like.

If you need to add some further package, for example ``randaugment`` and ``torchsummary``:
```shell
sudo su
conda activate vitis-ai-pytorch
pip install randaugment
pip install torchsummary
#exit
```

then remember to permanently save the modified docker image from a different terminal (a second one, besides the first one in which you are running the docker image),
by launching the following commands:

```shell
$ sudo docker ps -l
$ sudo docker commit -m"COMMENT" CONTAINER_ID DOCKER_IMAGE
```

you should see something like this:

```shell
$ sudo docker ps -l
CONTAINER ID   IMAGE                                       COMMAND                  CREATED       
8626279e926e   xilinx/vitis-ai-pytorch-gpu:3.0.0.001   "/opt/nvidia/nvidia_…"   6 hours ago  

$ sudo docker commit -m"pyt new_package" 8626279e926e   xilinx/vitis-ai-pytorch-gpu:3.0.0.001
```


## 4 The VCoR Dataset

The dataset adopted in this tutorial is the **Kaggle' Vehicle Color Recognition**, shortened as  [VCoR](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset). 	

You can find its hyperlink into the [Vitis AI Model Zoo Table online](https://xilinx.github.io/Vitis-AI/docs/reference/ModelZoo_VAI3.0_Github_web.htm).

This dataset is composed of 15 classes of colors (for the cars) to be classified. It contains labeled RGB images that are 224x224x3 in size and it was developed for the paper

  - Panetta, Karen, Landry Kezebou, Victor Oludare, James Intriligator, and Sos Agaian. 2021. "Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos" AI 2, no. 4: 684-704. [https://doi.org/10.3390/ai2040041](https://doi.org/10.3390/ai2040041)

  - open access: [https://www.mdpi.com/2673-2688/2/4/41](https://www.mdpi.com/2673-2688/2/4/41)

While being out of the docker container, download the ~602MB ``archive.zip`` file from the [VCoR](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset) website and then unzip it to the``build/dataset/vcor`` folder  
(as also shown in the [run_all.sh](files/run_all.sh) script):

```shell
cd <WRK_DIR>/tutorials/PyTorch-ResNet18/files
# you must have already downloaded the zip archive
unzip ./archive.zip -d ./build/dataset/vcor/
```



## 5 Vehicle Color Classification with ResNet18

From the docker image, unzip the archive ``pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0.zip`` in the ``files`` folder
and clean some files/folders, doing the following actions (as also shown in the [run_all.sh](files/run_all.sh) script):

```shell
cd <WRK_DIR> # you are now in Vitis_AI subfolder
# enter in the docker image
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest
# activate the environment
conda activate vitis-ai-pytorch
# go to the tutorial directory
cd /workspace/tutorials/
cd PyTorch-ResNet18/files # your current directory
# you must have already downloaded the archive
unzip pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0.zip
# clean some files/folders
cd pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0
rm -rf code data *.md *.txt *.sh
cd ..
```

You will get the ``files/pt_vehicle-color-classification_VCoR_224_224_3.64G_3.0/`` folder where you can find the pre-trained floating point model and the quantized model
respectively in the sub-folder ``float`` and ``quant``. You can ignore and remove all the other sub-folders.


The ResNet18 CNN applied in this tutorial aims to recognize the color of the car vehicle in the input image.

In practical applications, the input image often contains multiple vehicles, or there are many areas as the background,
so it is usually used together with an object detection CNN, which means firstly
use the object detection network to detect the vehicle area and cut the original image according to the bounding box which is the output of the object detection network, then send the cropped image to the network for classification. In the ``pt_vehicle-color-classification_VCoR_224_224_3.64G``
you could use the YoloV3 CNN to detect the cars in the VCoR dataset and use cropped images to build a new dataset to train and test the model.
If your input image contains little background, or your CNN is not used in conjunction with an object detection CNN, then you can skip this step
(which is what done indeed in this tutorial).

This vehicle color model falls under the [Vitis AI Library “classification” examples](https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai_library/samples/classification/readme):

- The model name is ``chen_color_resnet18_pt``, which makes it not obvious that it is actually a vehicle color classification.

- Here is the [list of the car colors](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_library/xnnpp/src/classification/car_color_chen.txt). Since there are 15 colors, there are also 15 classes to be classified.

- The DPU output will be a [data structure of classification results](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/vitis-ai-Classification) with 15 classes. Such output tensor will then be used by the ARM CPU to compute the functions ``SoftMax`` and related ``Top-5`` prediction accuracy.

- The prototxt file that comes with the [compiled model download](https://www.xilinx.com/bin/public/openDownload?filename=chen_color_resnet18_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz) specifies the number of classification results returned by the Vitis AI Library.  In this case it is set to return the top-5 prediction accuracy results.




### 5.1 Training

If you want to train the ResNet18 CNN on the VCoR dataset from scratch, just launch the script [run_train.sh](files/scripts/run_train.sh) (which is already done from the [run_all.sh](files/scripts/run_all.sh) script):

```shell
cd <WRK_DIR> # you are now in Vitis_AI subfolder
# enter in the docker image
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest
# activate the environment
conda activate vitis-ai-pytorch
# go to the tutorial directory
cd /workspace/tutorials/
cd PyTorch-ResNet18/files # your current directory
bash -x ./scripts/run_train.sh main_vcor
```

You should see something like this:

```text
. . .

Train Epoch: 29 [0/7267 (0%)]	Loss: 0.004103
Train Epoch: 29 [5120/7267 (71%)]	Loss: 0.006058
Test set: Average loss: 0.4320, Accuracy: 1380/1550 (89.032%)

. . .

classes: ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

Test set: Average loss: 0.4320, Accuracy: 1380/1550 (89.032%)
```

Note that when you use ``ToTensor()`` class in the [train.py](files/code/train.py) and [test.py](files/code/test.py) files, PyTorch [automatically converts all images into ``[0,1]`` range](https://discuss.pytorch.org/t/does-pytorch-automatically-normalizes-image-to-0-1/40022).

The images are supposed to be in RGB format and not in BGR (usually adopted by OpenCV library)


### 5.2 Quantization

If you want to quantize the floating point ResNet18 CNN from scratch, just launch the script [run_quant.sh](files/scripts/run_quant.sh) (which is already done from the [run_all.sh](files/scripts/run_all.sh) script).

You should see something like this:

```text
. . .

[VAIQ_NOTE]: =>Doing weights equalization...
[VAIQ_NOTE]: =>Quantizable module is generated.(quantized/ResNet.py)
[VAIQ_NOTE]: =>Get module with quantization.

Test set: Average loss: 0.4234, Accuracy: 1376/1550 (88.774%)

. . .

[VAIQ_NOTE]: =>Successfully convert 'ResNet_0' to xmodel.(quantized/ResNet_0_int.xmodel)
[VAIQ_NOTE]: ResNet_int.pt is generated.(quantized/ResNet_int.pt)
[VAIQ_NOTE]: ResNet_int.onnx is generated.(quantized/ResNet_int.onnx)
```


### 5.3 Compile the Target DPU  

The quantized CNN has then to be compiled for the DPU architecture of your target board, with the script [run_compile.sh](files/scripts/run_compile.sh) (which is already done from the [run_all.sh](files/scripts/run_all.sh) script).

You should see something like this:

```text
-----------------------------------------
COMPILING MODEL FOR VCK190..
-----------------------------------------
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCVDX8G_ISA3_C32B6
[UNILOG][INFO] Graph name: ResNet_0, with op num: 171
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/tutorials/PyTorch-ResNet18/files/./build/compiled_vck190/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/tutorials/PyTorch-ResNet18/files/./build/compiled_vck190/vck190_ResNet_0_int.xmodel.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is df3f607e0b632e90d4c8eafdd174bef8, and has been saved to "/workspace/tutorials/PyTorch-ResNet18/files/./build/compiled_vck190/md5sum.txt"
**************************************************
```



### 5.4 Embedded C++ Application with VART APIs

The C++ application running on the embedded ARM CPU of your target board is written in the [main_int8.cc](files/target/vcor/code/src/main_int8.cc) file. Note that the input images are pre-processed - before entering into the DPU - exactly in the same way they were pre-processed during the training, that is:

  - RGB image format (and not BGR);

  - the pixel range [0, 255] is normalized into data range [0,1]

Here is the related fragment of C++ code:

```text
Mat image = imread(baseImagePath + images[n + i]);

/*image pre-process*/
Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
resize(image, image2, Size(inHeight, inWidth), 0, 0, INTER_NEAREST);
for (int h = 0; h < inHeight; h++)
{
  for (int w = 0; w < inWidth; w++)
  {
    for (int c = 0; c < 3; c++)
    {
      //in RGB mode
      imageInputs[i*inSize+h*inWidth*3+w*3+2-c] = (int8_t)( (image2.at<Vec3b>(h, w)[c]/255.0f)*input_scale );
    }
  }
}
```

Once you launch it into the target board, you should see something like this:

```text

...

number of total images predicted  1499
number of top1 false predictions  202
number of top1 right predictions  1297
number of top5 false predictions  11
number of top5 right predictions  1488
top1 accuracy = 0.87
top5 accuracy = 0.99

...

./get_dpu_fps ./vck190_train_resnet18_vcor.xmodel 1 10000
[DPU avg FPS  ] 4141.89

./get_dpu_fps ./vck190_train_resnet18_vcor.xmodel 2 10000
[DPU avg FPS  ] 6650.38

./get_dpu_fps ./vck190_train_resnet18_vcor.xmodel 3 10000
[DPU avg FPS  ] 6654.61

```

As you can see the average prediction accuracy of the CNN running on the DPU is 87%
and the throughput in frames-per-second (fps) is from 4141 fps (1 thread only) to 6654 (3 threads).


Note that the test images were properly prepared with the [generate_target_test_images.py](files/code/generate_target_test_images.py) script
in order to append the class name to the image file name, thus enabling the usage of [check_runtime_top5_vcor.py](files/target/vcor/code/src/check_runtime_top5_vcor.py)
to check the prediction accuracy.






<div style="page-break-after: always;"></div>


## License

The MIT License (MIT)

Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


<p align="center"><sup>XD106 | © Copyright 2022 Xilinx, Inc.</sup></p>
