<!--
Copyright 2020-2022 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Mark Harvey, Xilinx Inc
-->

<table>
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI Tutorials</h1>
   </td>
 </tr>
 <tr>
 <td align="center"><h3>MNIST Classification using Vitis AI and TensorFlow 1.15</h3>
 </td>
 </tr>
</table>


### Current status

 + Tested with Vitis AI&trade; 1.4 and TensorFlow 1.15
 + Tested on the following platforms: ZCU102, Alveo U50


## Introduction

This tutorial introduces the user to the Vitis AI TensorFlow design process and shows how to go from a python description of the network model to running a compiled model on a Xilinx&reg; evaluation board.

The application code in this example design is written in Python and uses the VART runtime.

We will run the following steps:

  + Training and evaluation of a small custom convolutional neural network (CNN) using TensorFlow 1.15
  + Removal of the training nodes and conversion of the graph variables to constants (often referred to as *freezing the graph*).
  + Evaluation of the floating-point frozen model using the MNIST test dataset.
  + Quantization of the floating-point frozen model.
  + Evaluation of the quantized 8bit model using the MNIST test dataset.
  + Compilation of the quantized model to create the .xmodel file ready for execution on the DPU accelerator IP.
  + Downloading and running the application on the ZCU102 and/or Alveo&trade; U50 evaluation board.


## The MNIST Dataset

The MNIST handwritten digits dataset is a publicly available dataset containing 70 thousand 8-bit grayscale images each of which is 28x28 pixels. The complete dataset of 70 thousand images is normally divided into 60 thousand images for training and 10 thousand images for validation. The dataset is considered to be the 'hello world' of machine learning and makes a simple introduction to learn the complete Xilinx Vitis AI flow.

![mnist](./files/img/mnist.png?raw=true "Example MNIST images")


## The Convolution Neural Network

The convolutional neural network (CNN) in this design has deliberately been kept as simple as possible. It consists of just four layers of 2D convolution interspersed with batch normalization and ReLU activation. The network is described in the customcnn.py python script.

![cnn](./files/img/cnn.png?raw=true "Custom CNN")

## Preparing the Host Machine and Target Boards

The host machine has several requirements that need to be met before we begin. You will need:

  + An x86 host machine with a supported OS and either the CPU or GPU versions of the Vitis AI docker installed. For more information, see [System Requirements](https://github.com/Xilinx/Vitis-AI/blob/master/docs/learn/system_requirements.md).

  + The host machine requires the docker to be installed and the Vitis AI CPU or GPU docker image to be built. For more information, see [Getting Started](https://github.com/Xilinx/Vitis-AI#getting-started).

  + A GPU card suitable for training is recommended, but the training in this tutorial is quite simple, and a CPU can be used.

  + If you plan to use the ZCU102 evaluation board, it should be prepared with the board image as per [Step2: Setup the Target](https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc/VART#step2-setup-the-target) instructions. Hints on how to connect the various cables to the ZCU102 are also available [here](https://www.xilinx.com/html_docs/vitis_ai/1_4/installation.html#yjf1570690235238).

  + For the Alveo&trade; U50, follow the [Setup Alveo Accelerator Card](https://github.com/Xilinx/Vitis-AI/tree/master/setup/alveo) instructions.


For more details, refer to the latest version of the *Vitis AI User Guide* ([UG1414](https://www.xilinx.com/html_docs/vitis_ai/1_4/zmw1606771874842.html)).

This tutorial assumes the user is familiar with Python3, TensorFlow and has some knowledge of machine learning principles.


## Shell Scripts in this Tutorial

  + ``0_setenv.sh`` : Sets all the necessary environment variables used by the other scripts. You can edit most variables to configure the environment for their own requirements.
**Note:** The <code>image_input_fn.py</code> Python script contains a reference to the list of calibration images as an absolute path:

 ```python
 calib_image_list = './build/quantize/images/calib_list.txt'
 ```

 If either of the ``BUILD`` or ``QUANT`` variables are modified in ``0_setenv.sh``, the absolute path in ``image_input_fn.py`` also needs to be modified.

 It is highly recommended to leave the ``CALIB_IMAGES`` variable set to 1000 because it is the minimum recommended number of images for calibration of the quantization.


  + ``1_train.sh``: Runs training and evaluation of the network. It saves the trained model as an inference graph and floating-point checkpoint.

  + ``2_freeze.sh``: Converts the inference graph and checkpoint into a single binary protobuf file (.pb). The output .pb file is generally known as a 'frozen graph' since all variables are converted into constants, and graph nodes associated with training, such as the optimizer and loss functions, are stripped out.

  + ``3_eval_frozen_graph.sh``: This is an optional step that tests the accuracy of the frozen graph. The accuracy results should be very similar to the results obtained after training.

  + ``4_quant.sh``: This script first creates a set of image files to be used in the calibration phase of quantization and then launches the ``vai_q_tensorflow quantize`` command to convert the floating-point frozen graph to a fixed-point integer model.

  + ``5_eval_quant_graph.sh``: This step is optional but highly recommended. It runs the same evaluation function that was used to evaluate the frozen graph on the quantized model. Users should confirm that the accuracy reported by the evaluation of the quantized model is sufficient for their requirements and similar to the results for the floating-point models.

  + ``6_compile_u50.sh``: Launches the ``vai_c_tensorflow`` command to compile the quantized model into a .xmodel file for the Alveo U50 accelerator card.
  + ``6_compile_zcu102.sh``: Launches the ``vai_c_tensorflow`` command to compile the quantized model into an .xmodel file for the ZCU102 evaluation board.

  + ``7_make target_u50.sh``: Copies the .xmodel and images to the ``./build/target_u50`` folder ready for use with the Alveo U50 accelerator card.
  + ``7_make target_zcu102.sh``: Copies the .xmodel and images to the ``./build/target_zcu102`` folder ready to be copied to the ZCU102 evaluation board's SD card.


## Image Pre-Processing

All images undergo simple pre-processing before being used for training, evaluation, and quantization calibration. The images are normalized to bring all pixel values into the range of 0 to 1 by dividing them by 255.


## Implementation

Clone or download this GitHub repository to your local machine where you have installed the necessary tools. Open a Linux terminal, cd into the ``repository`` folder, then into the ``files`` folder. Start the Vitis AI docker. If you have a GPU in the host system, it is recommended that you use the GPU version of the docker container. However, if a GPU is not available, then the CPU docker container is sufficient:


```shell
# navigate to tutorial folder
cd <path_to_tutorial>/files

# to start GPU docker
./docker_run.sh xilinx/vitis-ai-gpu:latest

# ..or to start CPU docker
./docker_run.sh xilinx/vitis-ai-cpu:latest
```


The docker container will start and you should see something like this in the terminal:


```shell
==========================================

__      ___ _   _                   _____
\ \    / (_) | (_)            /\   |_   _|
 \ \  / / _| |_ _ ___ ______ /  \    | |
  \ \/ / | | __| / __|______/ /\ \   | |
   \  /  | | |_| \__ \     / ____ \ _| |_
    \/   |_|\__|_|___/    /_/    \_\_____|

==========================================

Docker Image Version:  1.4
Build Date: 2021-12-20
VAI_ROOT: /opt/vitis_ai

For TensorFlow Workflows do:
     conda activate vitis-ai-tensorflow
For Caffe Workflows do:
     conda activate vitis-ai-caffe
For Neptune Workflows do:
     conda activate vitis-ai-neptune
For PyTorch Workflows do:
     conda activate vitis-ai-pytorch
For TensorFlow 2.3 Workflows do:
     conda activate vitis-ai-tensorflow2
For Darknet Optimizer Workflows do:
     conda activate vitis-ai-optimizer_darknet
For Caffe Optimizer Workflows do:
     conda activate vitis-ai-optimizer_caffe
For TensorFlow 1.15 Workflows do:
     conda activate vitis-ai-optimizer_tensorflow
For LSTM Workflows do:
     conda activate vitis-ai-lstm
Vitis-AI /workspace >
```

>:bulb: If you get a "Permission Denied" error message when starting the Docker container, it is almost certainly because the `docker_run.sh` script is not set as being executable. You can fix this by running the command:
>
>```shell
>$ chmod +x ./docker_run.sh
>```

The complete tools flow can be run just by executing the `source run_all.sh` command, or by running each of the steps in order (from `0_setenv.sh` to `7_make_target_<board_name>.sh`):

```shell
$ source ./0_setenv.sh
$ source ./1_train.sh
.
.
$ source ./7_make_target_u50.sh
```

## Running the Application on the ZCU102 Board

All of the required files for running on the ZCU102 board are copied into the `files/build/target_zcu102` folder by the `7_make_target_zcu102.sh` script. The `7_make_target_zcu102.sh` script also copies the test set images to `files/build/target_zcu102/images` - the application code will preprocess and classify these images. The entire target_zcu102 folder neeeds to be copied to the ZCU102 SD card.

You can copy the complete `files/build/target_zcu102` folder to the `/home/root folder` of the flashed SD card in one of several ways:

1. Direct copy to SD Card:

  + If your host machine has an SD card slot, insert the flashed SD card, and when it is recognized you will see two volumes, BOOT and ROOTFS. Navigate into the ROOTFS and then into the /home folder.  Make the ./root folder writeable by issuing the command ``sudo chmod -R 777 root`` and then copy the entire `files/build/target_zcu102` folder from the host machine into the `/home/root` folder of the SD card.

  + Unmount both the BOOT and ROOTFS volumes from the host machine and then eject the SD Card from the host machine.

2. With scp command:

  + If the ZCU102 is connected to a network and reachable by the host machine, the target folder can be copied using `scp`. If you connect directly from your host machine to the ZCU102 using ethernet, you may need to set up static IP addresses.

  + The command is similar to ``scp -r build/target_zcu102 root@192.168.1.227:~/.`` assuming that the ZCU102 IP address is 192.168.1.227.  Adjust this and the path to the target folder as appropriate for your system.

  + If a password is asked, insert 'root'.


With the target folder copied to the SD Card, and the ZCU102 booted, you can issue the command for launching the application.
**Note:** It is done on the ZCU102 board and not the host machine. So, it requires a connection to the ZCU102 such as a serial connection to the UART or an SSH connection through Ethernet.

You can start the application by navigating into the target_zcu102 folder (`cd target_zcu102`) and then issuing the command ``python3 app_mt.py``. The application starts, and after a few seconds, shows the throughput (in frames/sec) and the accuracy:


```shell
root@xilinx-zcu102-2021_1:~ cd target_zcu102
root@xilinx-zcu102-2021_1:~/target_zcu102# python3 app_mt.py
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 1 threads...
Throughput=3644.82 fps, total frames = 10000, time=2.7436 seconds
Correct:9857, Wrong:143, Accuracy:0.9857
```

For better throughput, the number of threads can be increased by using the `-t` option. For example, to execute with four threads:

```shell
root@xilinx-zcu102-2021_1:~/target_zcu102# python3 app_mt.py -t 4
Command line options:
 --image_dir :  images
 --threads   :  4
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 4 threads...
Throughput=5224.95 fps, total frames = 10000, time=1.9139 seconds
Correct:9857, Wrong:143, Accuracy:0.9857
```

## Running the application on the Alveo U50

This final step copies all the required files for running on the U50 into the `./build/target_u50` folder. The `7_make_target_u50.sh` script also copies the test set images to `/build/target_u50/images`. The application code preprocesses and classifies these images.

**Note**: The U50 needs to have been flashed with the correct deployment shell. It is done as during the *Preparing the host machine and target boards* step.

Run the followings steps from inside the Vitis AI Docker container:

  + Ensure that Vitis-AI TensorFlow1 conda environment is enabled (if not, the run `conda activate vitis-ai-tensorflow`).

  + Run `source setup.sh DPUCAHX8H`, which sets environment variables to point to the correct overlay for the U50. The complete steps to run are as follows:


```shell
source setup.sh DPUCAHX8H
cd build/target_u50
/usr/bin/python3 app_mt.py
```

You should see something like this:


```shell
(vitis-ai-tensorflow) Vitis-AI /workspace/build/target_u50 > /usr/bin/python3 app_mt.py
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 1 threads...
Throughput=8006.58 fps, total frames = 10000, time=1.2490 seconds
Correct:9870, Wrong:130, Accuracy:0.9870
```

Similar to the ZCU102, the number of threads can be increased for higher throughput:


```shell
(vitis-ai-tensorflow) Vitis-AI /workspace/build/target_u50 > /usr/bin/python3 app_mt.py -t 6
Command line options:
 --image_dir :  images
 --threads   :  6
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 6 threads...
Throughput=22105.08 fps, total frames = 10000, time=0.4524 seconds
Correct:9870, Wrong:130, Accuracy:0.9870
```

</hr>
<p align="center"><sup>Copyright&copy; 2020-2022 Xilinx</sup></p>
