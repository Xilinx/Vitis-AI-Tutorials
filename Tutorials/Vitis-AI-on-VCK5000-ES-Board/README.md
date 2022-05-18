
<div style="page-break-after: always;"></div>
<table style="width:100%">
  <tr>
    <th width="100%" colspan="6"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI on VCK5000 ES Board</h1>
</th>
  </tr>
</table>
</div>

### Current status

1. Tested with Vitis AI 1.4

2. Tested with Vitis 2021.1

3. Tested in hardware on VCK5000 ES Board

**Date: 06 Dec 2021**

# Introduction

Vitis AI, understood as the Xilinx’s development stack for AI inference on unified Xilinx hardware platforms has minimize the differentiation for model deployment flow on Versal devices. This tutorial will start from card installation and go through the workflow step by step to run the first Vitis AI sample on **VCK5000 ES card**. If you are using a **VCK5000 prod cards**. Please visit the [vck5000 lounge](https://www.xilinx.com/member/vck5000-aie.html#vitis-ai) .

Generally speaking, users can easily get the inference work on VCK5000 with following three steps.

 + Setup the host environments with the scripts Xilinx has provided.
 + Enter the docker environment to enable Vitis AI tools.
 + Inference on board with the pre-compiled models.  

# Inplementation

This section will lead you through the steps necessary to complete the setup.
.
## Prerequisites

 + An x86 host machine with that meets the [sytem requirements](https://github.com/Xilinx/Vitis-AI/blob/master/doc/system_requirements.md) and internet access to download files.
 + Host machine with Xilinx design tools pre-installed (No earlier than version 2021.1).
 
## Step 1 - Set up the host

Xilinx has provided some scripts to help to automatically finish the Alveo card setup process which user could refer to and understand the required steps.

First of all, please make sure the card has been placed proerly and securely in your server or workstation. For detailed info, please refer to the VCK5000 installation guide [UG1531](https://www.xilinx.com/content/dam/xilinx/member/vck5000-aie/ug1531-vck5000_WtMkX.pdf).

Please run the ***lspci*** command to check that the VCK5000 Card has been installed.

```shell
$ lspci -vd 10ee:
```

An output similar to the following example is seen. 

```
02:00.0 Processing accelerators: Xilinx Corporation Device 5044

        Subsystem: Xilinx Corporation Device 000e

        Flags: bus master, fast devsel, latency 0, IRQ 16, NUMA node 0

        Memory at 380030000000 (64-bit, prefetchable) [size=128M]

        Memory at 380038020000 (64-bit, prefetchable) [size=128K]

        Capabilities: <access denied>

        Kernel driver in use: xclmgmt

        Kernel modules: xclmgmt

02:00.1 Processing accelerators: Xilinx Corporation Device 5045

        Subsystem: Xilinx Corporation Device 000e

        Flags: bus master, fast devsel, latency 0, IRQ 17, NUMA node 0

        Memory at 380038000000 (64-bit, prefetchable) [size=128K]

        Memory at 380028000000 (64-bit, prefetchable) [size=128M]

        Memory at 380038040000 (64-bit, prefetchable) [size=64K]

        Capabilities: <access denied>

        Kernel driver in use: xocl

        Kernel modules: xocl
```

Then, clone the Vitis-AI repository from Github and enter the vck5000 folder to start the AI journey on VCK5000.

```shell
$ git clone https://github.com/Xilinx/Vitis-AI.git -b 1.4
$ cd <path_to_Vitis-AI>/setup/vck5000
```

```bash
Cloning into 'Vitis-AI'...
remote: Enumerating objects: 67660, done.
remote: Counting objects: 100% (18112/18112), done.
remote: Compressing objects: 100% (8223/8223), done.
remote: Total 67660 (delta 9172), reused 17922 (delta 9142), pack-reused 49548
Receiving objects: 100% (67660/67660), 1.79 GiB | 34.94 MiB/s, done.
Resolving deltas: 100% (28954/28954), done.
Checking out files: 100% (45090/45090), done.
```

User can easily install the dependent tools including XRT, XRM and DPU xclbin file with a single script as the following command.

```shell
$ source ./install.sh
```

## Step 2 - Install the platform

The VCK5000 platform is OS dependent, please download platform based on your OS from the following links and install it on your X86 host machine.

|            OS               | RedHat / CentOS                                                |
  |-----------------------------|----------------------------------------------------------------|
  | Deployment Target Platform  | [xilinx-vck5000-es1-gen3x16-platform-2-1.noarch.rpm.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-platform-2-1.noarch.rpm.tar.gz)      | 
  | Development Target Platform | [xilinx-vck5000-es1-gen3x16-2-202020-1-dev-1-3123623.noarch.rpm](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vck5000-es1-gen3x16-2-202020-1-dev-1-3123623.noarch.rpm) |

  |            OS               | Ubuntu                                                         |
  |-----------------------------|----------------------------------------------------------------|
  | Deployment Target Platform  | [xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz)      | 
  | Development Target Platform | [xilinx-vck5000-es1-gen3x16-2-202020-1-dev_1-3123623_all.deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vck5000-es1-gen3x16-2-202020-1-dev_1-3123623_all.deb) |
  
User can simply use the following command to check your OS version. 

```shell
$ lsb_release -a
```

```
LSB Version:	core-9.20170808ubuntu1-noarch:printing-9.20170808ubuntu1-noarch:security-9.20170808ubuntu1-noarch
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.1 LTS
Release:	18.04
Codename:	bionic
```

After download the platform, please extract the file and install them in the following order.

***Noted:*** The command in following section is writen based on the Ubuntu 18.04 system.

```bash
$ tar -xzvf xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz
$ sudo apt install xilinx-sc-fw-vck5000_4.4.6-2.e1f5e26_all.deb
$ sudo apt install xilinx-vck5000-es1-gen3x16-validate_2-3123623_all.deb
$ sudo apt install xilinx-vck5000-es1-gen3x16-base_2-3123623_all.deb
```

After installation, user can use the following command to check if the platform has been installed properply. The ***xbmgmt*** is the platform manangement tools which is integreted with XRT. The ***sudo roles*** is required to execute the command.

```bash
$ sudo /opt/xilinx/xrt/bin/xbmgmt flash --scan
```

An output similar to the following example is seen which shows the platform currently running on the VCK5000 card with all the platforms have been installed.

```bash
Card [0000:02:00.0]

    Card type:         vck5000-es1

    Flash type:        OSPI_VERSAL

    Flashable partition running on FPGA:

        xilinx_vck5000-es1_g3x16_201921_1,[ID=0x5e51824d],[SC=4.4]

    Flashable partitions installed in system:

        xilinx_vck5000-es1_gen3x16_base_2,[ID=0xb376430f2629b15d],[SC=4.4.6]
```

Program the platform with the following command.

```bash
$ sudo /opt/xilinx/xrt/bin/xbmgmt flash --update
```

Type ***y*** to proceed the programing and you will see the output like below.

```bash
Card [0000:02:00.0]:

         Status: SC needs updating

         Current SC: 4.4

         SC to be flashed: 4.4.6

         Status: shell needs updating

         Current shell: xilinx_vck5000-es1_g3x16_201921_1

         Shell to be flashed: xilinx_vck5000-es1_gen3x16_base_2

Are you sure you wish to proceed? [y/n]: y

Updating SC firmware on card[0000:02:00.0]

Stopping user function...

....................................

Updating shell on card[0000:02:00.0]

PDI dsabin supports only primary bitstream: /opt/xilinx/firmware/vck5000-es1/gen3x16/base/partition.xsabin

INFO: ***PDI has 29148592 bytes

Successfully flashed Card[0000:02:00.0]

1 Card(s) flashed successfully.

Cold reboot machine to load the new image on card(s).
```

***Noted:*** Cold reboot is compulsury to let the new platform take effect.

## Step 3 - Enable Docker environment

Vitis AI tools has been released in the format of Docker container to avoid the libraries mismatch among different OS versions. Please all the users refer to [official Docker Documentation](https://docs.docker.com/engine/install/) to install the docker. Then start the docker container with the following command.

```bash
$ cd <path_to_Vitis-AI>
$ ./docker_run.sh xilinx/vitis-ai-cpu:1.4.916
```

The docker container will start and you should see something like this in the terminal:


```bash
==========================================

__      ___ _   _                   _____
\ \    / (_) | (_)            /\   |_   _|
 \ \  / / _| |_ _ ___ ______ /  \    | |
  \ \/ / | | __| / __|______/ /\ \   | |
   \  /  | | |_| \__ \     / ____ \ _| |_
    \/   |_|\__|_|___/    /_/    \_\_____|

==========================================

Docker Image Version:  1.4.916
Build Date: 2021-07-20
VAI_ROOT: /opt/vitis_ai

For TensorFlow 1.15 Workflows do:
     conda activate vitis-ai-tensorflow 
For Caffe Workflows do:
     conda activate vitis-ai-caffe 
For PyTorch Workflows do:
     conda activate vitis-ai-pytorch 
For TensorFlow 2.3 Workflows do:
     conda activate vitis-ai-tensorflow2 
Vitis-AI /workspace > 
```

Then, you can use the following command to set environment variables. It should be noted that the xclbin file should be in the /opt/xilinx/overlaybins directory.

```bash
$ cd /workspace/setup/vck5000
$ source ./setup.sh
```

## Step 4 - Inference with pre-compiled model
In this section, it will demonstrate the inference progress with pre-compiled models. Users who is using customized model or want to attempt the quantization and compilation workflow can refer to another [Vitis-AI tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Introduction/03-Basic/Module_5) 

1) Download the pre-compiled model from model zoo
  For each model, there will be a yaml file which is used for describe all the details about the model. 
	In the yaml, you will find the model's download links for different platforms. Please choose the corresponding model and download it. Click [Xilinx AI Model Zoo](../../models/AI-Model-Zoo/model-list) to view all the models. For VCK5000 card. Please download the model with DPUCVDX8H, taking `Enet` as an example here.

	* Download the model according to the model's yaml file.
	```
	$ wget https://www.xilinx.com/bin/public/openDownload?filename=ENet_cityscapes_pt-vck5000-DPUCVDX8H-r1.4.0.tar.gz -O ENet_cityscapes_pt-vck5000-DPUCVDX8H-r1.4.0.tar.gz
	```

	* Install the model package.  
	If the `/usr/share/vitis_ai_library/models` folder does not exist, create it first.
	```
	  $ sudo mkdir /usr/share/vitis_ai_library/models
	  $ tar -xzvf ENet_cityscapes_pt-vck5000-DPUCVDX8H-r1.4.0.tar.gz
	  $ sudo cp ENet_cityscapes_pt /usr/share/vitis_ai_library/models -r
	```	

**Note that different alveo cards correspond to different model files, which cannot be used alternately.** 

2) Prepare the images or videos for inference

   Download the image and video packages and untar them with the following commands.
  
```
cd /workspace
$ wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.4.0_images.tar.gz -O vitis_ai_library_r1.4.0_images.tar.gz
$ wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.4.0_video.tar.gz -O vitis_ai_library_r1.4.0_video.tar.gz
$ tar -xzvf vitis_ai_library_r1.4.0_images.tar.gz -C demo/Vitis-AI-Library/
$ tar -xzvf vitis_ai_library_r1.4.0_video.tar.gz -C demo/Vitis-AI-Library/
```

3) Compile and inference

Please compile the example and run the inference with the following image.

<p align="left">
  <img width="512" height="256" src="images/sample_segmentation.png">
  </p>

```
$ cd /workspace/demo/Vitis-AI-Library/samples/segmentation
$ bash -x build.sh
$ ./test_jpeg_segmentation ENet_cityscapes_pt sample_segmentation.jpg
```

The output file will be generated as below.

```
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0103 22:37:46.185174   197 demo.hpp:1183] batch: 0     image: sample_segmentation.jpg

I0103 22:37:46.344715   197 demo.hpp:1183] batch: 1     image: sample_segmentation.jpg

I0103 22:37:46.509322   197 demo.hpp:1183] batch: 2     image: sample_segmentation.jpg

I0103 22:37:46.676263   197 demo.hpp:1183] batch: 3     image: sample_segmentation.jpg

I0103 22:37:46.844214   197 demo.hpp:1183] batch: 4     image: sample_segmentation.jpg

I0103 22:37:47.005530   197 demo.hpp:1183] batch: 5     image: sample_segmentation.jpg

I0103 22:37:47.173975   197 demo.hpp:1183] batch: 6     image: sample_segmentation.jpg

I0103 22:37:47.340173   197 demo.hpp:1183] batch: 7     image: sample_segmentation.jpg
```
<p align="left">
<img src="images/result.png">
</p>

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
