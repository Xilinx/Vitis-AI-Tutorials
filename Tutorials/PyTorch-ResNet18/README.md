<!--

Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

Author: Daniele Bagni, AMD/Xilinx Inc
-->


<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis™ AI Tutorials</h1>
    </td>
 </tr>
</table>

#  ResNet18 in PyTorch from Vitis AI Library

- Version:      Vitis AI 3.5 with Pytorch 1.13.1

- Support:      ZCU102, ZCU102, VCK190, VEK280, Alveo V70

- Last update:  11 Aug 2023


## Table of Contents

[1 Introduction](#1-introduction)

[2 Prerequisites](#2-prerequisites)

[3 The Docker Tools Image](#3-the-docker-tools-image)

[4 The VCoR Dataset](#4-the-vcor-dataset)

[5 Vehicle Color Classification](#5-vehicle-color-classification)

[License](#license)



## 1 Introduction

### 1.1 Rationale

In this Deep Learning (DL) tutorial you will take a public domain Convolutional Neural Network (CNN) like [ResNet18](https://github.com/songrise/CNN_Keras/blob/main/src/ResNet-18.py) and pass it through the [Vitis AI 3.5](https://github.com/Xilinx/Vitis-AI) stack to run DL inference on FPGA devices; the application is classifying the different colors of the "car object" inside images.

Although ResNet18 was already trained on the [ImageNet](https://www.image-net.org/) dataset in the **PyTorch** framework, you will re-train it using the
**Kaggle' Vehicle Color Recognition** dataset, shortened as  [VCoR](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset). 	


### 1.2 The Vitis AI Flow

Assuming you have already trained your CNN and you own its original model, typically a [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file with extension ``.h5``, you will deploy such CNN on the FPGA target boards by following these steps:

  1. (optional) Run the [Model Inspector](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html?highlight=inspector#model-inspector) to check if the original model is compatible with the AMD [Deep Processor Unit (DPU)](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-system-integration.html)  architecture available on the target board (if not, you have to modify your CNN and retrain it).

  2. Run the [Model Quantization](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html?highlight=quantizer#model-quantization) process to generate a 8-bit fixed point (shortly "int8") model from the original 32-bit floating point CNN. If you apply the so called *Post-Training Quantization* (PTQ), this will be a single step, otherwise you would need to re-train - or more properly said "fine-tune" - the CNN with the *Quantization-Aware-Training* (QAT).

  3. (optional) Run inference with the int8 model on the Vitis AI environment (running on the host desktop) to check the prediction accuracy: if the difference is not negligible (for example it is larger than 5%, you can re-do the quantization by replacing PTQ with QAT).  

  4. Run the [Model Compilation](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html?highlight=quantizer#model-compilation) process  on the int8 model to generate the ``.xmodel`` microcode for the DPU IP soft-core of your target board.

  5. Compile the application running on the ARM CPU - tightly coupled with the DPU - of the target board, by using either C++ or Python code with the [Vitis AI RunTime (VART)](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-deployment.html#vitis-ai-runtime) APIs.  

Based on that you will be able to measure the inference performance both in terms of average prediction accuracy and frames-per-second (fps) throughput on your target board.

All the commands reported in this document are also collected into the [run_all.sh](files/run_all.sh) script.



## 2 Prerequisites

Here is what you need to have and do before starting with the real content of this tutorial.

- Familiarity with DL principles.

- Accurate reading of this [README.md](README.md) file from the top to the bottom, before running any script.

- Host PC with Ubuntu >= 18.04.5 (and possibly with GPU support to run the CNN training).

- Clone the entire repository of [Vitis AI 3.5](https://github.com/Xilinx/Vitis-AI) stack from [www.github.com/Xilinx](https://www.github.com/Xilinx) web site.

-  Accurate reading of [Vitis AI User 3.5 Guide 1414](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai) (shortly UG1414).

- Accurate reading of [Vitis AI 3.5 Online Documentation](https://xilinx.github.io/Vitis-AI/3.5/html/index.html). In particular, pay attention to the installation and setup instructions for both host PC and target board, it is recommended you build a GPU-based docker image with TF2.

- An AMD target board such as either:
    - the Zynq® UltraScale+™ MPSoC [ZCU102](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html), or
    - the Zynq® UltraScale+™ MPSoC [ZCU104](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu104-g.html), or
    - the Versal AI Core series [VCK190](https://www.xilinx.com/products/boards-and-kits/vck190.html), or
    - the Versal AI Edge series [VEK280](https://www.xilinx.com/products/boards-and-kits/vek280.html), or
    - the [Alveo V70 AI Accelerator](https://www.xilinx.com/applications/data-center/v70.html).

- The ``archive.zip`` file with the [Kaggle](www.kaggle.com) dataset of images, as explained in Section [4 The VCoR Dataset](#4-the-vcor-dataset).

- The [Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo) ``pt_vehicle-color-classification_3.5.zip`` archive of size 179507695 bytes, as explained in Section [5.1  Get ResNet18 from Vitis AI Model Zoo](#51-get-resnet18 -from-vitis-ai-model-zoo).



### 2.1 Working Directory

In the following of this document it is assumed you have installed Vitis AI 3.5 (shortly ``VAI3.5``) somewhere in your file system and this will be your working directory ``${WRK_DIR}``, for example ``export WRK_DIR=/media/danieleb/DATA/VAI3.5``.  You have also created a folder named ``tutorials`` under such ``${WRK_DIR}`` and you have copied this tutorial there and renamed it ``PyTorch-ResNet18``. Using the command ``tree -d -L 2`` you should see the following directories:

```
${WRK_DIR} # your Vitis AI 3.5 working directory
.
├── bck
├── board_setup
│   ├── v70
│   └── vek280
├── demos
├── docker
│   ├── common
│   ├── conda
│   └── dockerfiles
├── docs
│   ├── docs
│   ├── _downloads
│   ├── doxygen
│   ├── _images
│   ├── _sources
│   └── _static
├── docsrc
│   ├── build
│   └── source
├── dpu
├── examples
│   ├── custom_operator
│   ├── ofa
│   ├── OnBoard
│   ├── vai_library
│   ├── vai_optimizer
│   ├── vai_profiler
│   ├── vai_quantizer
│   ├── vai_runtime
│   ├── waa
│   └── wego
├── model_zoo
│   ├── images
│   └── model-list
├── src
│   ├── AKS
│   ├── vai_library
│   ├── vai_optimizer
│   ├── vai_petalinux_recipes
│   ├── vai_quantizer
│   └── vai_runtime
├── third_party
│   ├── tflite
│   └── tvm
└── tutorials # created by you
    ├── PyTorch-ResNet18 # this tutorial
    ├── TF2-Vitis-AI-Optimizer
```

### 2.2 Dos-to-Unix Conversion

In case you might get some strange errors during the execution of the scripts, you have to process -just once- all the``*.sh`` shell and the python ``*.py`` scripts with the [dos2unix](http://archive.ubuntu.com/ubuntu/pool/universe/d/dos2unix/dos2unix_6.0.4.orig.tar.gz) utility.
In that case run the following commands from your Ubuntu host PC (out of the Vitis AI docker images):

```
sudo apt-get install dos2unix
cd ${WRK_DIR}/tutorials/PyTorch-ResNet18 #your repo directory
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

These operations are already included in the script [clean_all.sh](files/scripts/clean_all.sh), launched by the [run_all.sh](files/run_all.sh) script, which collects all the commands shown in the rest of this document.

It is strongly recommended that you familiarize with the [run_all.sh](files/run_all.sh) script in order to understand all what it does, ultimately the entire Vitis AI flow on the host computer.



## 3 The Docker Tools Image

You have to know few things about [Docker](https://docs.docker.com/) in order to run the Vitis AI smoothly on your host PC environment.

### 3.1 Build the Image

From the Vitis AI 3.5 repository, run the following commands:

```
cd ${WRK_DIR}
cd docker
./docker_build.sh -t gpu -f pytoch
```

Once the process is finished, with the command ``docker images`` you should see something like this:

```
REPOSITORY                        TAG               IMAGE ID       CREATED         SIZE
xilinx/vitis-ai-pytorch-gpu   3.5.0.001-b56bcce50   3c5d174a1807   27 hours ago    21.4GB
```


### 3.2 Launch the Docker Image

To launch the docker container with Vitis AI tools, execute the following commands from the ``${WRK_DIR}`` folder:

```
cd ${WRK_DIR} # you are now in Vitis_AI subfolder

./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest

conda activate vitis-ai-pytorch

cd /workspace/tutorials/

cd PyTorch-ResNet18 # your current directory
```

Note that the container maps the shared folder ``/workspace`` with the file system of the Host PC from where you launch the above command.
This shared folder enables you to transfer files from the Host PC to the docker container and vice versa.

The docker container does not have any graphic editor, so it is recommended that you work with two terminals and you point to the same folder, in one terminal you use the docker container commands and in the other terminal you open any graphic editor you like.

If you need to add some further package, for example ``randaugment`` and ``torchsummary``:

```
sudo su
conda activate vitis-ai-pytorch
pip install randaugment
pip install torchsummary
#exit
```

then remember to permanently save the modified docker image from a different terminal (a second one, besides the first one in which you are running the docker image),
by launching the following commands:

```
$ sudo docker ps -l
$ sudo docker commit -m"COMMENT" CONTAINER_ID DOCKER_IMAGE
```

you should see something like this:

```
$ sudo docker ps -l
CONTAINER ID   IMAGE                                       COMMAND                  CREATED       
8626279e926e   xilinx/vitis-ai-pytorch-gpu:3.5.0.001-b56bcce50     "/opt/nvidia/nvidia_…"   6 hours ago  

$ sudo docker commit -m"pyt new_package" 8626279e926e   xilinx/vitis-ai-pytorch-gpu:3.5.0.001-b56bcce50  
```


### 3.3 Things to Know

1. In case you "[Cannot connect to the Docker daemon at unix:/var/d9f942cdf7de   xilinx/vitis-ai-tensorflow2-gpu:3.5.0.001-b56bcce50 run/docker.sock. Is the docker daemon running?](https://stackoverflow.com/questions/44678725/cannot-connect-to-the-docker-daemon-at-unix-var-run-docker-sock-is-the-docker)" just launch the following command:

  ```
  sudo systemctl restart docker
  ```

2. Note that docker does not have an automatic garbage collection system as of now. You can use this command to do a manual garbage collection:

  ```
  docker rmi -f $(docker images -f "dangling=true" -q)
  ```

3. In order to clean the (usually huge amount of) space consumed by Docker have a look at this post: [Docker Overlay2 Cleanup](https://bobcares.com/blog/docker-overlay2-cleanup/). The next commands are of great effect (especially the last one):

  ```
  docker system df
  docker image prune --all
  docker system prune --all
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
cd ${WRK_DIR}/tutorials/PyTorch-ResNet18/files
# you must have already downloaded the zip archive
unzip ./archive.zip -d ./build/dataset/vcor/
```



## 5 Vehicle Color Classification with ResNet18

All the commands shown in the next subsections are available [run_all.sh](files/run_all.sh) script, normally called with the command:

```
source ./run_all main_vocr
```

### 5.1  Get ResNet18 from Vitis AI Model Zoo

You have to download the ``pt_vehicle-color-classification_3.5.zip`` archive of ResNet18 reported in this [model.yaml](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/model-list/pt_vehicle-color-classification_3.5/model.yaml) file.
As the file name says, such CNN has been trained RGB images of input size 224x224
and it requires a computation of 3.64GOPs per image.

From the docker image, unzip the archive ``pt_vehicle-color-classification_3.5.zip`` in the ``files`` folder
and clean some files/folders, doing the following actions (already available in the [run_all.sh](files/run_all.sh) script):

```shell
cd ${WRK_DIR} # you are now in Vitis_AI subfolder
# enter in the docker image
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest
# activate the environment
conda activate vitis-ai-pytorch
# go to the tutorial directory
cd /workspace/tutorials/
cd PyTorch-ResNet18/files # your current directory
# you must have already downloaded the archive
unzip pt_vehicle-color-classification_3.5.zip
# clean some files/folders
cd pt_vehicle-color-classification_3.5
rm -rf code data *.md *.txt *.sh
cd ..
```

You will get the ``files/pt_vehicle-color-classification_3.5/`` folder where you can find the pre-trained floating point model and the quantized model
respectively in the sub-folder ``float`` and ``quant``. You can ignore and remove all the other sub-folders.


The ResNet18 CNN applied in this tutorial aims to recognize the color of the car vehicle in the input image.

In practical applications, the input image often contains multiple vehicles, or there are many areas as the background,
so it is usually used together with an object detection CNN, which means firstly
use the object detection network to detect the vehicle area and cut the original image according to the bounding box which is the output of the object detection network, then send the cropped image to the network for classification. In the ``pt_vehicle-color-classification_3.5``
you could use the YoloV3 CNN to detect the cars in the VCoR dataset and use cropped images to build a new dataset to train and test the model.
If your input image contains little background, or your CNN is not used in conjunction with an object detection CNN, then you can skip this step
(which is what done indeed in this tutorial).

This vehicle color model falls under the [Vitis AI Library “classification” examples](https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai_library/samples/classification/readme):

- The model name is ``chen_color_resnet18_pt``, which makes it not obvious that it is actually a vehicle color classification.

- Here is the [list of the car colors](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_library/xnnpp/src/classification/car_color_chen.txt). Since there are 15 colors, there are also 15 classes to be classified.

- The DPU output will be a [data structure of classification results](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/vitis-ai-Classification) with 15 classes. Such output tensor will then be used by the ARM CPU to compute the functions ``SoftMax`` and related ``Top-5`` prediction accuracy.



### 5.2 Training

If you want to train the ResNet18 CNN on the VCoR dataset from scratch, just launch the script [run_train.sh](files/scripts/run_train.sh) (which is already done from the [run_all.sh](files/scripts/run_all.sh) script):

```shell
cd ${WRK_DIR} # you are now in Vitis_AI subfolder
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


### 5.3 Quantization

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


### 5.4 Compile the Target DPU  

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


### 5.5 Run on the Target Board

The throughput measured in fps is shown only for the VEK280 board, just as a reference. For the other boards the results can be pretty different depending on the different DPU architectures and related batch size (BS).

All the commands illustrated in the following subsections are inside the script [run_all_vcor_target.sh](files/target/vcor/run_all_vcor_target.sh), they are applied directly in the target board ``xxxyyy``
(i.e. zcu102, vck190, v70, vek280, etc) by launching the command ``run_all_target.sh xxxyyy``, which involves  the [run_all_target.sh](files/target/run_all_target.sh) higher level script.


#### 5.5.1 Multithreading C++ Application Code

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


Note that the DPU API apply [OpenCV](https://opencv.org/) functions to read an image file (either ``png`` or ``jpg`` or whatever format) therefore the images are seen as BGR and not as native RGB. All the training and inference steps done in this tutorial treat images as RGB, which is true also for the above C++ normalization routine.


#### 5.5.2 Run-Time Execution

It is possible and straight-forward to compile the application directly on the target (besides compiling it into the host computer environment).
In fact this is what the script [run_all_vcor_target.sh](files/target/vcor/run_all_vcor_target.sh)  does, when launched on the target.  

Turn on your target board and establish a serial communication with a ``putty`` terminal from Ubuntu or with a ``TeraTerm`` terminal from your Windows host PC.

Ensure that you have an Ethernet point-to-point cable connection with the correct IP addresses to enable ``ssh`` communication in order to quickly transfer files to the target board with ``scp`` from Ubuntu or ``pscp.exe`` from Windows host PC. For example, you can set the IP addresses of the target board to be ``192.168.1.217`` while the host PC is  ``192.168.1.140``.

Once a ``tar`` file of the ``build/target_vek280`` (or ``build/target_vck190``, etc) folder has been created, copy it from the host PC to the target board. For example, in case of an Ubuntu PC, use the following command:
```
scp target_vek280.tar root@192.168.1.217:~/
```

From the target board terminal, run the following commands (in case of VEK280):
```
tar -xvf target_vek280.tar
cd target_vek280
bash -x ./run_all_target.sh vek280
```


The application based on VART C++ APIs is built with the [build_app.sh](files/target/vcor/code/build_app.sh) script and finally launched for each CNN, the effective top-5 classification accuracy is checked by a python script [check_runtime_top5_vcor.py](files/target/code/src/check_runtime_top5_vcor.py) which is launched from within
the [vcor_performance.sh](files/target/vcor/vcor_performance.sh) script.

Note that the test images were properly prepared with the [generate_target_test_images.py](files/code/generate_target_test_images.py) script
in order to append the class name to the image file name, thus enabling the usage of [check_runtime_top5_vcor.py](files/target/vcor/code/src/check_runtime_top5_vcor.py)
to check the prediction accuracy.



#### 5.5.3 DPU Performance

On the VEK280 board, the purely DPU performance (not counting the CPU tasks) measured in fps is:

-  ~4804 fps with 1 thread,  

- ~10099 fps fps with 3 threads.


The prediction accuracy is:

```
...
number of total images predicted  300
number of top1 false predictions  41
number of top1 right predictions  259
number of top5 false predictions  4
number of top5 right predictions  296
top1 accuracy = 0.86
top5 accuracy = 0.99
...
```




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
