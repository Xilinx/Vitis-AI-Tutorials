<!--
Copyright 2021-2023 Xilinx Inc.

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
<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI Tutorials</h1>
  </td>
 </tr>
 <tr>
 <td align="center"><h1>Fine-Tuning TensorFlow2 Quantized Models</h1>
 </td>
 </tr>
</table>


## Current Status

+ Tested with Vitis-AI 1.4, TensorFlow 2.3 and ZCU102 evaluation board
+ Dataset: ImageNet ILSVRC2012
+ Network: MobileNet


# Introduction

Machine learning models such as Convolutional Neural Networks (CNN) are generally trained and saved using floating-point formats. However, the Xilinx&reg; family of DPU accelerators operate on fixed point integer values (usually 8-bit), and so this requires our original floating-point model to be quantized. The process of converting a model whose values have a vast dynamic range to a functionally equivalent model but with a much more limited dynamic range can lead to a small loss in accuracy.

Some models are affected more than others by this change in dynamic range, and when this happens, it may be necessary to use quantization fine-tuning to recover accuracy.

This tutorial shows how to implement the Vitis AI&trade; quantization fine-tuning for TensorFlow 2.3.  The following are the steps we run:

 + Download the Imagenet training and validation datasets and convert them to TFRecord format
 + Train, evaluate and save the floating-point MobileNet model
 + Quantization of the trained model *without* fine-tuning to demonstrate accuracy loss
 + Quantization *with* fine-tuning to recover the lost accuracy and bring the quantized model back up to the same accuracy as the floating-point model
 + Compiling the fine-tuned model for the target board
 + Execution of the compiled model on the target board

This tutorial assumes you are familiar with Python3, TensorFlow, and have some knowledge of machine learning principles.

The complete flow is shown in the following figures:

<p align="center">
 <img src="img/steps0_3.png" width="100%" height="100%">
</p>
<p align="center">
 <img src="img/steps4_5.png" width="60%" height="50%">
</p>


# The MobileNet CNN

We will use a MobileNet that has been pre-trained on ImageNet as our starting point. The Keras MobileNet model is instantiated as a base model  without including the final classification layers. We then add a new classification head that consists of a dropout layer, a 10-output dense/fully-connected layer, and a softmax layer.

The MobileNet base model is set to be 'frozen' and will not be trained at all, and only the new 10-output classification head will be trained.

The complete code for the MobileNet model is contained in the <code>xfer_model.py</code> file. The base model is instantiated with the ImageNet weights but no classifier head layers:

```python
base_model = MobileNet(input_shape=input_shape,
                       alpha=1.0,
                       depth_multiplier=1,
                       include_top=False,
                       weights='imagenet',
                       pooling='avg')
```

It is set to be non-trainable:

```python
base_model.trainable = False
```

The output of the base model is then fed into a new classifier head which is parameterized for the number of classes we are using:

```python
net = base_model.output

# add output classifier head
net = Dropout(0.01)(net)
output_layer = Dense(classes)(net)
```


# Before You Begin

The host machine has several requirements to be met before we begin. You will need:

  + An x86 host machine with a supported OS and either the CPU or GPU versions of the Vitis-AI docker installed - see [System Requirements](https://github.com/Xilinx/Vitis-AI/blob/master/docs/learn/system_requirements.md).

  + The host machine will require Docker to be installed and the Vitis-AI CPU or GPU docker image to be built - see [Getting Started](https://github.com/Xilinx/Vitis-AI#getting-started).

  + A GPU card suitable for training is recommended, but the training in this tutorial is quite simple, and a CPU can be used.

  + If you plan to use the ZCU102 evaluation board, it should be prepared with the board image per [Step2: Setup the Target](https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc/VART#step2-setup-the-target) instructions. Hints on how to connect the various cables to the ZCU102 are also available [here](https://www.xilinx.com/html_docs/vitis_ai/1_4/installation.html#yjf1570690235238).

For more details, refer to the latest version of the *Vitis AI User Guide* ([UG1414](https://www.xilinx.com/html_docs/vitis_ai/1_4/zmw1606771874842.html)).

This tutorial assumes the user is familiar with Python3, TensorFlow, and has some knowledge of machine learning principles.


# Setting up the workspace

1. Copy this repository by doing either of the following:

    + Download the repository as a ZIP file to the host machine, and then unzip the archive.
    + From a terminal, use the `git clone` command.

2. Open a Linux terminal, `cd` to the repository folder, and then `cd` to the `files` folder.

4. Start either the Vitis AI GPU or CPU docker (we recommend using the GPU docker if possible):

     ```shell
     # navigate to tutorial folder
     cd <path_to_tutorial>/files

     # to start GPU docker container
     ./docker_run.sh xilinx/vitis-ai-gpu:latest

     # to start CPU docker container
     ./docker_run.sh xilinx/vitis-ai-cpu:latest
     ```

  The docker container starts, and after accepting the license agreement, you  see something like this in the terminal:

     ```shell
     ==========================================

     __      ___ _   _                   _____
     \ \    / (_) | (_)            /\   |_   _|
      \ \  / / _| |_ _ ___ ______ /  \    | |
       \ \/ / | | __| / __|______/ /\ \   | |
        \  /  | | |_| \__ \     / ____ \ _| |_
         \/   |_|\__|_|___/    /_/    \_\_____|

     ==========================================

     Docker Image Version:  1.4.776
     Build Date: 2021-06-22
     VAI_ROOT: /opt/vitis_ai

     For TensorFlow 1.15 Workflows do:
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

>:bulb: *If you get a "Permission Denied" error when starting the docker container, it is almost certainly because the docker_run.sh script is not set to be executable. You can fix this by running the following command:*
>    
>    ```shell
>     chmod +x docker_run.sh
>    ```


Activate the Tensorflow2 python virtual environment with `conda activate vitis-ai-tensorflow2`, and you should see the prompt change to indicate that the environment is active:


```shell
Vitis-AI /workspace > conda activate vitis-ai-tensorflow2
(vitis-ai-tensorflow2) Vitis-AI /workspace >
```

# Implementing the Design

The remainder of this tutorial describes the steps to implement the tutorial. Each command needs to be run from within the Vitis-AI Docker container, which was started in the previous section.

A shell script called <code>run_all.sh </code> is also provided - this contains the commands needed to run the complete flow:

```shell
source run_all.sh
```

## Step 0 - Download the Datasets and Convert to TFRecords

  + Download <Code>ILSVRC2012_img_val.tar</code> from [Academic Torrents](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) or other sources. Place <Code>ILSVRC2012_img_val.tar</code> in the <path_to_repo>/files folder.



  + Convert the images in the <Code>ILSVRC2012_img_val.tar</code> archive to TFRecords (requires an internet connection). The `-mc 10` option will ensure that only the first ten classes are used instead of all 1000 classes:

```shell
python -u make_val_tfrec.py -mc 10 2>&1 | tee make_val_tfrec.log
```

It creates a folder called tfrec_val that contains a single TFRecord file. If you use more classes, more than one TFRecord file may be generated. The <code>ILSVRC2012_img_val.tar</code> archive can be deleted now to save disk space.


  + Convert the images in the <code>ILSVRC2012_img_val.tar</code> archive to TFRecords (requires internet connection). The `-mc 10` option will ensure that only the first ten classes are used. These are the same ten classes used when converting the validation set:

```shell
python -u make_train_tfrec.py -mc 10 2>&1 | tee make_train_tfrec.log
```

It creates a folder called tfrec_train that contains 10 TFRecord files. If required, the <code>ILSVRC2012_img_val.tar</code> archive can be deleted to save disk space.



## Step 1 - Training

Train the floating-point MobileNet model. The `-mc 10` option will ensure that the classifer output vector is of the correct shape and matches the number of classes used to create the TFRecords:

```shell
python -u train.py -mc 10 2>&1 | tee train.log
```

After the training is completed, the trained floating-point model is saved as float_model/float.h5. Evaluation is also  run and should give an accuracy of approximately 85%.


## Step 2 - Quantize without Fine-Tuning

Next, we will run quantization *without* fine-tuning followed by evaluation. This step is just to demonstrate that normal quantization does not give sufficient accuracy:

```shell
python -u quant.py -e 2>&1 | tee quant.log
```

Evaluation gives an accuracy of approximately 65% which is much lower than the floating-point accuracy.


## Step 3 - Quantize with Fine-Tuning

Now, we can run quantization *with* fine-tuning:


```shell
python -u quant_ft.py 2>&1 | tee quant_ft.log
```

This is effectively a form of retraining of the quantized model. First, we load the floating-point model:

```python
float_model = load_model(float_dir+'/float_model.h5', compile=False)
```

Then, we create a quantized model that can be fine-tuned:

```python
# Quantization-aware training model
quantizer = vitis_quantize.VitisQuantizer(float_model)
ft_model = quantizer.get_qat_model()
```

After that, training is run using the <code>.compile()</code> and <code>.fit()</code> methods of the <code>tf.keras</code> API.

Once the quantization-aware training is complete, evaluation is run and should give an accuracy of approximately 85%, which should be almost identical to the original floating-point model accuracy.


## Step 4 - Compile the Fine-Tuned Model

To run step 3, run the `compile.sh` with one of the target boards as a command line argument, for example:

```shell
source compile.sh zcu102
```

The `compile.sh` script also supports `zcu104` and `vck190` as command-line arguments. The `compile.sh` shell script compiles the quantized model and create an .xmodel file that contains the instructions and data to be executed by the DPU in the compiled_model_<target> folder.


## Step 5 - Make the Target Folder to be Copied to the SD Card

Run `make_target.py` with the appropriate command lines options to create a folder ready to be copied to the target board, for example:

```shell
python -u make_target.py -m compiled_model_zcu102/mobilenet.xmodel -td target_zcu102 2>&1 | tee logs/target_zcu102.log
```

## Step 6 - Run the Application on the Target

The entire `target_<xxx>` folder needs to be copied to the /home/root folder of the flashed SD card. It can be done in one of several ways:

1. Direct copy to SD Card:

  + If the host machine has an SD card slot, insert the flashed SD card, and when it is recognized, you will see two volumes, BOOT and ROOTFS. Navigate into the ROOTFS and then into the /home folder.  Make the ./root folder writeable by issuing the command ``sudo chmod -R 777 root``. Copy the entire `target` folder from the host machine into the /home/root folder of the SD card.

  + Unmount both the BOOT and ROOTFS volumes from the host machine and then eject the SD Card from the host machine.

2. With <code>scp</code> command:

  + If the target evaluation board is connected to the same network as the host machine, the `target_<xxx>` folder can be copied using scp.

  + The command is similar to ``scp -r ./build/target_zcu102 root@192.168.1.227:~/.``  assuming that the target board IP address is 192.168.1.227. Adjust this as appropriate for your system.

  + If the password is asked for, insert 'root'.


After the `target_<xxx>` folder is copied to the SD Card and the evaluation board is booted, you can issue the command for launching the application.
**Note**: It is done on the target evaluation board and not the host machine, so it requires a connection to the board, such as a serial connection to the UART or an SSH connection via Ethernet.

You can start the application by navigating into the `target_<xxx>` folder on the evaluation board and then issuing the command ``python3 app_mt.py``. The application starts and after a few seconds shows the accuracy and throughput in frames/sec, like this:


```shell
root@xilinx-zcu102-2020_2:~/target# python3 app_mt.py
-----------------------------------------------
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  mobilenet.xmodel
-----------------------------------------------
Found 500 images - processing 500 of them
Found 1 subgraphs in mobilenet.xmodel
Pre-processing 500 images...
Starting 1 threads...
-----------------------------------------------
Correct:416, Wrong:84, Accuracy:0.8320
-----------------------------------------------
FPS: 194.12, total frames: 500, total time: 2.576 seconds
-----------------------------------------------
```


The throughput can be improved by increasing the number of threads with the `--threads` option:


```shell
root@xilinx-zcu102-2020_2:~/target# python3 app_mt.py --threads 6
-----------------------------------------------
Command line options:
 --image_dir :  images
 --threads   :  6
 --model     :  mobilenet.xmodel
-----------------------------------------------
Found 500 images - processing 500 of them
Found 1 subgraphs in mobilenet.xmodel
Pre-processing 500 images...
Starting 6 threads...
-----------------------------------------------
Correct:416, Wrong:84, Accuracy:0.8320
-----------------------------------------------
FPS: 731.21, total frames: 500, total time: 0.684 seconds
-----------------------------------------------
```



## References

1. [Xilinx Vitis-AI User Guide ver 1.4](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_4/ug1414-vitis-ai.pdf)

2. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
