<!--
Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
MIT License 
-->

<table class="sphinxhide" style="width:100%;">
  <tr>
    <td align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/logo-white-text.png">
        <img alt="AMD logo" src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%">
      </picture>
      <h1>Vitis AI Tutorials</h1>
    </td>
  </tr>
</table>

# Deploying a Custom ResNet18 on Vitis AI NPU

#### Current Status

- Release:      `Vitis AI 5.1`
- Host OS:      Ubuntu 22.04.5 LTS (Jammy JellyFish)
- Target Board: VEK280 revB3

- Last update:  23 Oct. 2025


## Table of Contents

[1 Introduction](#1-introduction)


[2 Getting Started with Vitis AI](#2-getting-started-with-vitis-ai)
- [2.1 Directories Organization](#21-directories-organization)
- [2.2 Setup the Ubuntu Host PC](#22-setup-the-ubuntu-host-pc)
- [2.3 Setup the VEK280 Target Board](#23-setup-the-vek280-target-board)
- [2.4 Before Running Any Applications](#24-before-running-any-applications)


[3 How to Use VART Cpp API](#3-how-to-use-vart-cpp-api)
- [3.1 Input Parameters](#31-input-parameters)
- [3.2 RGB or BGR Preprocessing](#32-rgb-or-bgr-preprocessing)
- [3.3 Cross Compile the Demo Applications](#33-cross-compile-the-demo-applications)
- [3.4 Launch the Just Compiled vart_ml_demo](#34-launch-the-just-compiled-vart_ml_demo)


[4 Run Classification with ImageNet Dataset](#4-run-classification-with-imagenet-dataset)
- [4.1 ResNet18 RGB vs. BGR](#41-resnet18-rgb-vs-bgr)
- [4.2 ResNet50 RGB vs. BGR](#42-resnet50-rgb-vs-bgr)


[5 Deploy Custom ResNet18 from Host PC to Target Board](#5-deploy-custom-resnet18-from-host-pc-to-target-board)
- [5.1 Overview](#51-overview)
- [5.2 Train ResNet18 with VCoR Dataset](#52-train-resnet18-with-vcor-dataset)
- [5.3 Generating a Snapshot for VCoR ResNet18](#53-generating-a-snapshot-for-vcor-resnet18)
- [5.4 Cpp Application for the Target](#54-cpp-application-for-the-target)


[6 FPS Performance](#6-fps-performance)
- [6.1 Standard ResNet50](#61-standard-resnet50)
- [6.2 Custom ResNet18](#62-custom-resnet18)


[Appendix A](#appendix-a)
- [A1 How to Download the ImageNet Dataset](#a1-how-to-download-the-imagenet-dataset)


[References](#references)


## 1 Introduction

This tutorial shows you how to deploy a Convolutional Neural Network (CNN) model for **Image Classification** with the **Vitis AI 5.1** development flow, from the host PC to the Versal VEK280, which contains the NPU (**Neural Processing Unit**).

The NN is a ResNet18, originally trained on the [ImageNet](https://www.image-net.org/) dataset (1000 classes) in the PyTorch framework. It is then re-trained (or fine-tuned) using the **Kaggle Vehicle Color Recognition** dataset (15 classes), shortened as [VCoR](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset). This allows the model to classify the color of cars, and other vehicles.

The VEK280 NPU is an inference accelerator from AMD. Its architecture is a combination of FPGA Programmable Logic (PL), and [AI Engine](https://www.xilinx.com/products/technology/ai-engine.html#overview) (AIE) technology. In the following document, you might see the Versal NPU referred to as the "Vitis AI NPU." Both terms are equivalent, as you cannot work on the Versal NPU without the Vitis AI flow.

The target board running the Versal NPU for this tutorial is the [Versal AI Edge VEK280](https://www.xilinx.com/products/boards-and-kits/vek280.html).

### Warning

The C++ applications do not perform any sophisticated checks on the input parameters. If something goes wrong, you might encounter unexpected `segmentation violation` errors. For example:

1. You use a `GroundTruth` file with fewer images than those stored in the `imgPath` folder.
2. Your `nbImages` value is greater than the number of images actually stored in the `imgPath` folder.
3. Your `imgPath` folder does not contain images, but only subfolders (even if those subfolders contain images).
4. You load a snapshot of a CNN that does not perform image classification (such as YOLOv5).


## 2 Getting Started with Vitis AI

The following document assumes you have installed Vitis AI 5.1 using the `vitis-ai-5.1.tar` archive somewhere in your file system, one level below a designated working directory `{WRK_DIR}` on the Ubuntu host PC. For example, this document uses a structure like this:

```bash
# your working directory
export WRK_DIR=~/Public/VAI5.1/release
# VITIS-ai is one level below the working dir
export VITIS_AI_REPO=${WRK_DIR}/Vitis-AI
```

**Note:** The `${VITIS_AI_REPO}` variable points to the folder that stores the current Vitis AI release.


### 2.1 Directories Organization


After expanding the `vitis-ai-5.1.tar` archive in your `${WRK_DIR}` directory, go to its `${VITIS_AI_REPO}` sub-directory. You should see the following folders:

```
${WRK_DIR} # ~/Public/VAI5.1/release
.
|
├── ${WRK_DIR}/Vitis-AI # ${VITIS_AI_REPO}

              ├── bin
              ├── docker
              ├── examples
              ├── npu_ip
              ├── src
              └── tools
```


Create a folder named `tutorials` under the `${WRK_DIR}` directory. Copy this tutorial into that folder and rename it as `${WRK_DIR}/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment`.


### 2.2 Setup the Ubuntu Host PC

If you are working on an Ubuntu PC, set it up using the following steps:

1. Install the Docker image.
   Note that without the `--gpus all` flag, the Docker image runs on your host CPU without any GPU support.
   Also note that the `--ipc=host` flag increases the shared memory size, as described in the [Using pre-built images](https://github.com/pytorch/pytorch?tab=readme-ov-file#using-pre-built-images) section of the [PyTorch](https://github.com/pytorch/pytorch) official GitHub page.


   ```shell
   cd ${VITIS_AI_REPO}

    source npu_ip/settings.sh VE2802_NPU_IP_O00_A304_M3 

   ./docker/run.bash --gpus all --ipc=host -v ${WRK_DIR}:/workspace
   # real example:
   #./docker/run.bash --gpus all --ipc=host -v /home/danieleb/Public/VAI5.1/release:/workspace
   ```

   **WARNING:** AT THE END OF THESE COMMANDS YOU ARE INSIDE THE DOCKER IMAGE (BE CAREFUL: NO PROMPT SIGNALs IT).

2. If you need a package that is not included in the Docker image, add it by manually editing the `$VITIS_AI_REPO/docker/vaisw.dockerfile` file. Insert your changes immediately after the `ENV MISPO_DOCKER=1` line, as shown in the following example:

   ```shell
    # line 596
    ENV MISPO_DOCKER=1
    # add my own python package for TF2
    RUN python3 -m pip install --no-cache-dir install image-classifiers
    # add my own python package for PyTorch
    RUN python3 -m pip install --no-cache-dir install tqdm
    RUN python3 -m pip install --no-cache-dir install onnxruntime
    RUN python3 -m pip install --no-cache-dir install torch
    RUN python3 -m pip install --no-cache-dir install torchvision
    RUN python3 -m pip install --no-cache-dir install torchsummary
    RUN python3 -m pip install --no-cache-dir install torchinfo
    RUN python3 -m pip install --no-cache-dir install randaugment
    RUN python3 -m pip install --no-cache-dir install utils
    #RUN python3 -m pip install --no-cache-dir install onnx_tool
    #RUN python3 -m pip install --no-cache-dir install jupyter
   ```
3. Install the Cross Compiler or "VitisAI SDK"  with these commands:

   ```shell
    cd ${WRK_DIR}/bck
    # make it executable
    chmod 777 ./sdk.sh
    # install the VitisAI SDK into folder
    bash -x ./sdk.sh -d ../sdk_vai5.1
   ```
   
   **Note**: Use [setup_sdk.sh](files/scripts/setup_sdk.sh) to setup the VitisAI SDK when needed.

After you complete these actions, your Vitis AI setup on the Host Computer should look like this:

  ```
  # this is the ${WRK_DIR}
  .
  ├── bck
  |
  ├── sdk_vai5.1
  |
  ├── tutorials     # where this repository goes
  |
  └── Vitis-AI      # this is the ${VITIS_AI_REPO}
      ├── bin
      ├── docker
      ├── examples
      ├── npu_ip
      ├── src
      └── tools
  ```

### 2.3 Setup the VEK280 Target Board


After obtaining the IP address of the target board (for example, `192.168.1.217`) using the `ifconfig` command, establish a remote connection between the host and the target with a command such as `ssh -X root@192.168.1.217`.

To transfer files from the host to the target, use a command similar to the following:

    ```
    scp myfile root@192.168.1.217:/home/root
    ```


To transfer files from the target to the host, use a command similar to the following:

    ```
    scp root@192.168.1.217:/home/root/myfile
    ```

If you do not have a `/home/root` folder, create it with the following commands:

   ```shell
   cd /
   mkdir /home
   mkdir /home/root
   export HOME=/home/root
   ```


### 2.4 Before Running Any Applications

Before launching any application on the target VEK280 board, execute the following procedure exactly in this order:

1. Copy the subfolder ``vart_ml`` from ``${VITIS_AI_REPO}/src`` to this repository, with the following commands:

    ```shell
     cd ${WRK_DIR}/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files
     cp -r ${VITIS_AI_REPO}/src/vart_ml .
    ```

    Then overwrite its original content with the newest files placed from the [cpp_code](files/cpp_code) folder:
    ```shell
    cd ${WRK_DIR}/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/vart_ml
    cp -r ../cpp_code/* .
    ```

2.  Copy the entire ``$VITIS_AI_REPO`` from the Ubuntu Host PC to the target board with this command:

    ```shell
    cd ${VITIS_AI_REPO}
    cd ..
    scp -r ${VITIS_AI_REPO} root@192.168.1.217:/home/root # renamed as Vitis-AI for ease of use
    ```

3. Take the [target_setup_vai.sh](files/scripts/target_setup_vai.sh) bash script, and copy it
(via ``scp``)  into the ``/home/root`` folder of the target board. Note the following lines:

    ```shell
    export VAISW_INSTALL_DIR=/etc/vai
    export PYTHONPATH=$VAISW_INSTALL_DIR/lib/python
    export HOME=/home/root

    # enable statistics
    #export VAISW_RUNSESSION_SUMMARY=all

    # this is the currently available snapshot
    export RESNET50_SNAPSHOT=/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.resnet50.TF
    ```

4. Create these folders below the  ``/home/root`` in the target board:

    ```shell
    cd /home/root/
    # make new folders
    mkdir snapshots
    mkdir datasets
    mkdir datasets/imagenet
    mkdir datasets/imagenet/val
    mkdir datasets/vcor
    mkdir datasets/vcor/test
    ```

    You should see the following files and folders below `/home/root` level:

    ```
    . # /home/root
    |
    ├── snapshots
    ├── Vitis-AI
    ├── datasets
    ├── target_setup_vai.sh
    ```

    Now execute the following commands:
    ```shell
    # check if you need to reboot the board a second time
    source /etc/vai.sh
    # then go here:
    cd /home/root/
    # setup the board environment
    source ./target_setup_vai.sh
    # add once forever these packages
    python3 -m pip install tqdm
    python3 -m pip install onnxruntime
    # clean files
    cd Vitis_AI
    rm -rf docker # you do not need docker on the target
    rm -rf bin    # you do not need this folder

    # ResNet50 classification
    vart_ml_runner.py --snapshot ${RESNET50_SNAPSHOT}
    ```

    You should see a text similar to this:

    ```
    root@xilinx-vek280-xsct-20242:/home/root# vart_ml_runner.py --snapshot ${RESNET50_SNAPSHOT}
    VART: parsing snapshot /run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.resnet50.TF
    XAIEFAL: INFO: Resource group Avail is created.
    XAIEFAL: INFO: Resource group Static is created.
    XAIEFAL: INFO: Resource group Generic is created.
    VART: Allocated         config area in DDR: 	Addr = [ 0x880000000, 0x50000000000, 0x60000000000 ] 	Size = [ 0xd5d400, 0x9a1400, 0xd5d400]
    VART: Effective aligned config area in DDR: 	Addr = [ 0x880000000, 0x50000000000, 0x60000000000 ] 	Size = [ 0xd5d000, 0x9a1000, 0xd5d000]
    VART: Allocated snapshot   tmp area in DDR: 	Addr = [ 0x880d5e000, 0x500009a2000, 0x60000d5e000 ] 	Size = [ 0x159001, 0x127c01, 0x127c01]
    VART: Effective aligned    tmp area in DDR: 	Addr = [ 0x880d5e000, 0x500009a2000, 0x60000d5e000 ] 	Size = [ 0x158c01, 0x127801, 0x127801]
    Found snapshot for IP VE2802_NPU_IP_O00_A304_M3 matching running device
    [========================= 100% =========================]
    Inference took 29 ms
    Inference took 25 ms
    Inference took 21 ms
    Inference took 22 ms
    Inference took 22 ms
    Inference took 20 ms
    Inference took 21 ms
    Inference took 25 ms
    Inference took 22 ms
    Inference took 22 ms
    OK: no error found
    ```

5. Obtain the 50,000 images of the ImageNet validation set (`val` folder) by following the instructions in Appendix [A1 How to Download the ImageNet Dataset](#a1-how-to-download-the-imagenet-dataset). Then, use `scp` to copy them from the host computer to the target board:

   ```shell
   # start from imagenet folder
   cd ${WRK_DIR}/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/imagenet
   # copy the val folder
   scp -r ./val/* root@192.168.1.217:/home/root/datasets/imagenet/val/
   ```


#### 2.4.1 Target Board

Power on the board and start it by following these steps:

1. Check that the boot process completed correctly with the command `source /etc/vai.sh`. If it did not, run `reboot`.

2. Launch the setup script (this only needs to be done once per session, until you shut down the board): `source /home/root/target_setup.sh`


## 3 How to Use VART Cpp API

The VART (Vitis AI Run Time) C++ APIs are described in UG1703.

The simplest way to getting started with C++ API, is by re-compiling the [vart_ml_demo.cpp](files/cpp_code/vart_mnl/demo/vart_ml_demo.cpp)
file, which is the source code of the NPU runner executable.

Re-compile the NPU SW components is explained in UG1703 and will be addressed here in the next section [3.3 Cross Compile the Demo Applications](#33-cross-compile-the-demo-applications).


### 3.1 Input Parameters

All the C++ files in the `$VITIS_AI_REPO/src/vart_ml/demo` folder use the same common functions from the [common.cpp](files/cpp_code/vart_ml/demo/common.cpp) module. This ensures that the common interface remains consistent across these files.


Here are some arguments:

```
--imgPath  PATH   either a directory or a list of images
--snapshot PATH   path to the snapshot directory
--batchSize BATCHSIZE  size of a batch of images to process, defaults to 1
--goldFile  PATH       path to the file containing the gold results
--labels    PATH       path to the file containing labels of results, defaults to labels'
--mean      MEAN       mean of a pixel (depends on the framework), defaults to 0
--nbImages  NBIMAGES   number of images to process, defaults to 1
--std       STD        standard deviation (depends on the framework)
--colorFmt  COLORFMT   Color Format, either 1 (RGB) or 0 (BGR)
```

The last one was added for preprocessing, as explained in [section 3.2](#32-rgb-or-bgr-preprocessing).


### 3.2 RGB or BGR PreProcessing


Note that the input images must be pre-processed—before entering the NPU—in exactly the same way as during the training of the machine learning model. This process can differ for each CNN.

For example, both ResNet50 and ResNet18 in the PyTorch framework require:
  - RGB image format, not BGR;
  - Pixel values in the range [0, 255] normalized to the range [0, 1].

OpenCV reads images in BGR format, not RGB. If RGB images are required, swap the color channels as shown in the commented line.

The following C++ code fragment illustrates this concept:


```cpp
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
      imageInputs[i*inSize+h*inWidth*3+w*3+2-c] = (int8_t)( (image2.at<Vec3b>(h, w)[c]/255.0f) );
      // in BGR model
    //imageInputs[i*inSize+h*inWidth*3+w*3+  c] = (int8_t)( (image2.at<Vec3b>(h, w)[c]/255.0f) );
    }
  }
}
```

For this reason, code related to `ColorFormat`—which can be either `RGB` or `BGR`—has been added, as shown in the following fragments from 
[common.h](files/cpp_code/vart_ml/demo/common.h) 
and 
[common.cpp](files/cpp_code/vart_ml/demo/common.cpp):


```cpp
// from common.h
namespace ColorFormat
{
    constexpr int rgb = 1;
    constexpr int RGB = 1;
    constexpr int bgr = 0;
    constexpr int BGR = 0;
}

// ************************************************
// from common.cpp

static int COLORFMT = ColorFormat::BGR; //defaul value (0)
// some code
void usage(std::string cmd, std::string reason)
{
  // some code
  std::cout << "  --colorFmt COLORFMT   Color Format, either 1 (RGB) or 0 (BGR, the default)" << std::endl;
   // some code
}
// some code
int read_options(int argc, char* argv[], struct options& options,
std::vector<std::string>& images_paths)
{
  // some code
  valid_options.insert("colorFmt");    //DB
	valid_options.insert("dataFormat");	 //DB
  // some code
  //DB: added colorFmt (RGB=1, BGR=0)
	if (!options_map["colorFmt"].empty())
		set_colorFmt(strtoul(options_map["colorFmt"].c_str(), NULL, 0));
	std::cout << "preprocessing will apply " << ((COLORFMT == ColorFormat::RGB) ? "RGB" : "BGR")
      << " color format" << std::endl;
  // some code
}

// some code
void preprocess(float* const buf,
				std::string image_path,
				size_t height,
				size_t width,
				size_t channel)
{
	cv::Mat preprocessed = cv::Mat(height, width, CV_8UC3);
	cv::Mat image        = cv::imread(image_path);
	cv::resize(image, preprocessed, cv::Size(height, width), 0, 0);
	for (size_t h = 0; h < height; h++)
	{
		for (size_t w = 0; w < width; w++)
		{
			for (size_t c = 0; c < channel; c++)
				if (COLORFMT == ColorFormat::BGR)
					buf[width*channel*h + channel*w +   c] = (preprocessed.at<cv::Vec3b>(h, w)[c] - MEAN) / STD; //DB: BGR
				else //(COLORFMT == ColorFormat::RGB)
			  	buf[width*channel*h + channel*w + 2-c] = (preprocessed.at<cv::Vec3b>(h, w)[c] - MEAN) / STD; //DB: RGB
		}
	}
}
// some code
void set_colorFmt (int new_colFmt) //DB
{
	if (new_colFmt > 1)
	{
		std::cout << "ERROR: colorFmt can be only either 1 (RGB) or 0 (BGR)! Forcing it to BGR" << std::endl;
		new_colFmt = ColorFormat::BGR;
	}
	COLORFMT = (new_colFmt == 1) ? ColorFormat::RGB : ColorFormat::BGR;
}
```

### 3.3 Cross Compile the Demo Applications

When you recompile the software applications that run on the ARM CPU of the VEK280 target board from the x86 host PC, you perform a **cross-compilation** process.

Here are the steps to cross-compile the `vart_ml_demo` application. Execute these commands outside the Docker container (note that `make debug` can be used to enable debug mode):


```bash
# source SDK sysroot path if not yet done
source $WRK_DIR/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/scripts/setup_sdk.sh

# go to source dir
cd $WRK_DIR/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/vart_ml

# build process
make clean
# make debug # to compile in debug mode
make         # to compile in release mode

# generate the vart_ml_install.tar.gz archive
make install-sdk
make install-tar

# copy the  tar.gz archive from host to target
scp ./vart_ml_install.tar.gz root@192.168.1.111:/home/root/
```

Now you need to copy also the executable files on the board:

```shell

# copy the executables from host to target (I copy the whole folder, being quicker)
scp -r ./demo/* root@192.168.1.217:/home/root/Vitis-AI/src/vart_ml/demo/

# connect to target board
ssh -X root@192.168.1.217
```

From the terminal connected to the target, execute the following commands:

```bash
cd /home/root
tar -xvf ./vart_ml_install.tar.gz -C /
```
This installs all files into the /usr folder of the target.


### 3.4 Launch the Just Compiled vart_ml_demo

To check the new `vart_ml_demo` that you have just compiled on the target board, launch it with the following parameters:

```shell
# setup the env
source ~/target_setup_vai.sh

# change dir
cd Vitis-AI/src/vart_ml/demo

# launch just compiled new executable
vart_ml_demo --snapshot ${RESNET50_SNAPSHOT} \
  --imgPath ~/datasets/imagenet/ILSVRC2012_img_val \
  --labels /etc/vai/labels/labels  \
  --goldFile ~/imagenet/datasets/ILSVRC_2012_val_GroundTruth_10p.txt \
  --batchSize 19 --nbImages 38 --colorFmt 1
```



## 4 Run Classification with ImageNet Dataset

In this section, you work with two neural network models available in this Vitis AI release: ResNet18 and ResNet50. Both models are developed in the PyTorch framework and trained with the [ImageNet](https://www.image-net.org) dataset (1000 classes).

Launch the standard commands to enter the Docker container environment from the host computer, then run the following commands to generate a snapshot:


```bash
# create new folder and change directory
cd ${VITIS_AI_REPO}
source npu_ip/settings.sh
mkdir snapshots
cd examples/python_examples/batcher

# generate snapshot for ResNet18
VAISW_QUANTIZATION_NBIMAGES=190 VAISW_SNAPSHOT_DIRECTORY=./snapshot.py.Q190_resnet18.bs19 ./run_classification.sh -f pytorch -n resnet18 -b 19 -d /workspace/datasets/imagenet/ILSVRC2012_img_val --nbImages 190

# generate snapshot for ResNet50
VAISW_SNAPSHOT_DIRECTORY=./snapshots/snapshot.py.resnet50.bs19 ./run_classification.sh -f pytorch -n resnet50 -b 19 -d /workspace/datasets/imagenet/ILSVRC2012_img_val/ILSVRC2012_img_val --nbImages 190

# move the snapshots
mkdir /workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/snapshots
mv snapshot* /workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/snapshots/

# copy the snapshots to the target board with IP Address 192.168.1.111
cd /workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/snapshots
scp -r ./snapshots root@192.168.1.217:/home/root/snapshots

# connect to the target board
ssh -X root@192.168.1.217
```

From the terminal just ``ssh`` connected to the target board, execute the following commands:

```bash
# set up target board env
source /home/root/target_setup_vai.sh

# change dir
cd ~/Vitis-AI/src/vart_ml/demo

# run inference with ResNet50 on ImageNet
vart_ml_demo --batchSize 19 \
  --goldFile ~/datasets/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt \
  --imgPath ~/datasets/imagenet/ILSVRC2012_img_val \
  --snapshot ~/snapshots/snapshot.py.resnet50.bs19 \
  --labels /etc/vai/labels/labels \
  --nbImages 1349 --dataFormat native --colorFmt 1

# run inference with ResNet18 on ImageNet
vart_ml_demo --batchSize 19 \
  --goldFile ~/datasets/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt  \
  --imgPath ~/datasets/imagenet/ILSVRC2012_img_val \
  --labels /etc/vai/labels/labels \
  --snapshot ~/snapshots/snapshot.py.resnet18.bs19 \
  --nbImages 1349   --dataFormat native --colorFmt 1
```

You should see something like this for ResNet50:

```
Accuracy Summary:
[AMD] [resnet50 TEST top1] 71.68% passed.
[AMD] [resnet50 TEST top5] 88.44% passed.
[AMD] [resnet50 ALL TESTS] 71.68% passed.
[AMD] VART ML runner data format was set to NATIVE.
[AMD] 6312.83 imgs/s (1349 images)
```

and you should see something like this for ResNet18:

```
Accuracy Summary:
[AMD] [resnet18 TEST top1] 49.52% passed.
[AMD] [resnet18 TEST top5] 72.35% passed.
[AMD] [resnet18 ALL TESTS] 49.52% passed.
[AMD] VART ML runner data format was set to NATIVE.
[AMD] 12669.70 imgs/s (1349 images)
```


### 4.1 ResNet18 RGB vs. BGR

On your target board, run the ``vart_ml_demo`` with the following parameters:

```shell
cd /home/root/Vitis-AI/src/vart_ml/demo

# BGR
./vart_ml_demo  --snapshot ~/snapshots/snapshot.py.resnet18.bs19 --imgPath ~/imagenet/ILSVRC2012_img_val \
  --labels /etc/vai/labels/labels --batchSize 19 --nbImages 1349 --goldFile ~/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt \
  --colorFmt 0 > logfile_imagenet_resnet18_rgb_vek280.txt 2>&1

# RGB
./vart_ml_demo  --snapshot ~/snapshots/snapshot.py.resnet18.bs19 --imgPath ~/imagenet/ILSVRC2012_img_val \
  --labels /etc/vai/labels/labels --batchSize 19 --nbImages 1349 --goldFile ~/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt
  --colorFmt 1 > logfile_imagenet_resnet18_bgr_vek280.txt 2>&1
```

You see lines similar to the following near the end of the output text:

```
============================================================
Accuracy Summary (BGR):
[AMD] [resnet18 TEST top1] 35.36% passed.
[AMD] [resnet18 TEST top5] 58.93% passed.
[AMD] [resnet18 ALL TESTS] 35.36% passed.
============================================================
Accuracy Summary (RGB):
[AMD] [resnet18 TEST top1] 49.52% passed.
[AMD] [resnet18 TEST top5] 72.35% passed.
[AMD] [resnet18 ALL TESTS] 49.52% passed.
```

**Note**: The ResNet18 average prediction top1 accuracy drops from 49.52 (in RGB mode) down to 35.36 (in BGR mode) for the images taken from the ImageNet test dataset.

Overall, this average prediction accuracy is much lower than the value you get by running the following commands on the host computer:
```shell
cd files/resnet18

# Do a softlink to where your imagenet dataset is placed. In this tutorial, it is in /workspace/datasets/imagenet
ln -ns /workspace/datasets/imagenet ../imagenet
# launch the test
python3 ./code/test_imagenet.py --model resnet18 --nb_images 50
```
You should see the following:
```
Processing: 50/50000 images from folder n04273569
Top-1 Average Accuracy: 69.76%
```
It is aligned what is described in the
[PyTorch ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
web page.

**Note**: If you process only 1 (`--nb_images 1`) or 5 (`--nb_images 5`) or 10 (`--nb_images 10`)
images per class, you get `Top-1 Average Accuracy` respectively of `68.53%` (over 1000 images), `69.97%` (over 5000 images), `69.77%` (over 10000 images).


This difference in the results might be because of several reasons:

1. Different preprocessing applied during the training process (by the developer of the neural network) and during this inference process. The `vart_ml_demo` application is provided as an example to follow. You need to determine the best way to compute accuracy for your use case.
2. The `MEAN` and `STD` constant parameters are significantly different.



### 4.2 ResNet50 RGB vs. BGR

On your target board, run the ``vart_ml_demo`` with the following parameters:

```shell
cd ~/Vitis-AI/src/vart_ml/demo/

# RGB
./vart_ml_demo  --snapshot ~/snapshots/snapshot.py.resnet50.bs19  --imgPath ~/datasets/imagenet/ILSVRC2012_img_val --labels /etc/vai/labels/labels --batchSize 19 --nbImages 1349 --goldFile ~/datasets/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt --colorFmt 1

# BGR
./vart_ml_demo  --snapshot ~/snapshots/snapshot.py.resnet50.bs19  --imgPath ~/datasets/imagenet/ILSVRC2012_img_val --labels /etc/vai/labels/labels --batchSize 19 --nbImages 1349 --goldFile ~/datasets/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt --colorFmt 0
```

You see the following lines (or something similar) around the end of the output text:

```
============================================================
Accuracy Summary (RGB):
[AMD] [resnet50 TEST top1] 71.68% passed.
[AMD] [resnet50 TEST top5] 88.44% passed.
[AMD] [resnet50 ALL TESTS] 71.68% passed.

============================================================
Accuracy Summary (BGR):
[AMD] [resnet50 TEST top1] 56.04% passed.
[AMD] [resnet50 TEST top5] 75.54% passed.
[AMD] [resnet50 ALL TESTS] 56.04% passed.
```


**Note**: The ResNet50 average prediction top1 accuracy drops from 71.68 (in RGB mode) down to 56.04 (in BGR mode) for the
images taken from the  ImageNet test dataset.


On the Host computer, launch:

```shell
cd files/resnet18
python3 ./code/test_imagenet.py --model resnet50 --nb_images 50
```

You see the following:

```
Processing: 50/50000 images from folder n04273569
Top-1 Average Accuracy: 76.14%
```

It is aligned with what described in the
[PyTorch ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
web page. It refers to the original ResNet architecture introduced in the paper "Deep Residual Learning for Image Recognition" by Kaiming He et al. This is typically referred to as `ResNet50 v1`, which includes the basic building blocks of residual networks, to differentiate it from the more recent [ResNet50 V1.5](https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50) NN.


Finally, if you process only 1 (`--nb_images 1`), 5 (`--nb_images 5`), or 10 (`--nb_images 10`) images per class, you obtain a `Top-1 Average Accuracy` of 75.42% (over 1,000 images), 76.56% (over 5,000 images), and 76.21% (over 10,000 images), respectively.


## 5 Deploy Custom ResNet18 from Host PC to Target Board

### 5.1 Overview

To run a neural network model on the Versal NPU, complete the **Model Preparation** step, which requires Vitis AI to create a **snapshot** from the Ubuntu x86 host PC.

During the snapshot generation process, the Vitis AI flow tries to convert the model into an ONNX format. If this conversion fails, the models are not supported by the NPU. There is nothing specific for you to do. This requirement defines the scope of supported models and is one of the reasons compilation might fail.

#### 5.1.1 Snapshots

Snapshots describe NPU kernel configurations for specific machine learning networks. These files contain all the necessary information to replay inference execution, including the neural network's structure, weights, calibration results, and any fine-tuning execution. When using a snapshot, the original weight file is not needed. Additionally, the snapshot includes the input sizes, and the batch size, indicating that it is tailored for batch size execution. You run inference and generate a snapshot. The flow assumes that a Python inference script is ready, as you need this to verify the trained model. For now, the snapshot generation process only supports Python.

Assume you have a Python script, `test.py`, that runs inference on your floating point model (for example, a `resnet50` neural network) with a command like `python3 test.py`. This script must include the correct pre-processing and post-processing routines and must point to the original floating point model and the test dataset used during its original training to achieve the best average prediction accuracy.

Set up the Vitis AI environment by running the command `source npu_ip/settings.sh`. To create a snapshot in Vitis AI, add the prefix `VAISW_SNAPSHOT_DIRECTORY=path_name_of_your_snapshot` to your inference command. The complete command looks like this:

```shell
VAISW_SNAPSHOT_DIRECTORY=path_name_of_your_snapshot python3 test.py
```
The inference process starts, and Vitis AI generates the snapshot of your model. This approach does not require any changes to your original test.py code. In previous Vitis AI releases (<=3.5), you had to add many APIs to the original inference code to generate the snapshot.

### 5.2 Train ResNet18 with VCoR Dataset

The ResNet18 CNN used in this tutorial recognizes the color of the car vehicle in the input image. In practical applications, the input image often contains multiple vehicles or large background areas. It is usually used together with an object detection CNN. Typically, the object detection network detects the vehicle area and crops the original image according to the bounding box output, then sends the cropped image to the network for classification. If your input image contains little background, or your CNN is not used with an object detection CNN, you can skip this step, as done in this tutorial.

#### 5.2.1 The VCoR Dataset
The dataset used in this tutorial is the VCoR dataset, which consists of 15 vehicle color classes to be classified. It contains labeled RGB images sized 224x224x3 and was developed for the paper:

  - Panetta, Karen, Landry Kezebou, Victor Oludare, James Intriligator, and Sos Agaian. 2021. "Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos" AI 2, no. 4: 684-704. https://doi.org/10.3390/ai2040041

  - Open access: https://www.mdpi.com/2673-2688/2/4/41

While outside the Docker container, download the ~602MB archive.zip file from the VCoR website and unzip it to the resnet18/build/dataset/vcor folder:

```shell
# go to tutorial main folder folder
cd ${WRK_DIR}/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files
# you must have already downloaded the zip archive
unzip ./archive.zip -d ./build/dataset/vcor/
```



#### 5.2.2 Training

If you want to train the ResNet18 CNN on the VCoR dataset from scratch, use the scripts [train.py](files/resnet18/code/train.py) and
[run_train.sh](files/resnet18/scripts/run_train.sh), enter the Docker image and then launch these commands:

```shell
# go to the tutorial directory
cd /workspace/tutorials/
# your current directory
cd VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18
# launch training
bash -x ./scripts/run_train.sh main_vcor
```

At the end of the training process, you should see something like this:

```text
python3 code/train.py --batch-size 512 --epochs 36 --backbone resnet18 --save-model --data_root ./build/dataset/vcor --save_dir=./build/float

image size is  224 cols x  224  rows

train num: 7267
classes: ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']
----------------------------------------------------------------

        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,864
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
       BasicBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,864
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,864
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
       BasicBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,728
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,456
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,192
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
       BasicBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,456
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,456
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
       BasicBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         294,912
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         589,824
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          32,768
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
       BasicBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         589,824
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
       BasicBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,072
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 76, 7]               0
       BasicBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
       BasicBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                   [-1, 15]           7,695
================================================================
Total params: 11,184,207
Trainable params: 11,184,207
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 62.79
Params size (MB): 42.66
Estimated Total Size (MB): 106.02
----------------------------------------------------------------
Train Epoch: 1 [0/7267 (0%)]	Loss: 2.920296
Train Epoch: 1 [5120/7267 (71%)]	Loss: 2.229534
Test set: Average loss: 2.7214, Accuracy: 436/1550 (28.129%)
...
Train Epoch: 25 [   0/7267 (0%)]	Loss: 0.001452
Train Epoch: 25 [5120/7267 (71%)]	Loss: 0.005146
Test set: Average loss: 0.4195, Accuracy: 1371/1550 (88.452%)

Train Epoch: 32 [0/7267 (0%)]	Loss: 0.001645
Train Epoch: 32 [5120/7267 (71%)]	Loss: 0.001076
Test set: Average loss: 0.4353, Accuracy: 1381/1550 (89.097%)
```

**Note**: When you use ``ToTensor()`` class in the [train.py](files/resnet18/code/train.py) and [test.py](files/resnet18/code/test.py) files, PyTorch [automatically converts all images into ``[0,1]`` range](https://discuss.pytorch.org/t/does-pytorch-automatically-normalizes-image-to-0-1/40022).

The images must be in RGB format and not in BGR, usually adopted by OpenCV library.


### 5.3 Generating a Snapshot for VCor ResNet18

The script [test.py](files/resnet18/code/test.py) computes inference, or makes predictions, on the test images of the VCoR dataset using the 32-bit floating point neural network model.

Here is the sequence of commands to generate the snapshot from the host computer, after entering the Docker container:

```bash
# go to the tutorial directory
cd /workspace/tutorials/
# your current directory
cd VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18

# launch inference on the Host PC
python3 ./code/test.py --batch_size 19 --data_root ./build/dataset/vcor --resume ./build/float/color_resnet18.pt
# alternatively to above command line you can use this:
#bash -x ./scripts/run_test.sh
```

At the end of the inference process, you should see the following output text (or something similar):

```
image size is  224 cols x  224  rows

test num: 1556
classes: ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

Test set: Average loss: 0.4648, Accuracy: 1368/1556 (87.918%)
```

The average top-1 prediction accuracy of the 32-bit floating point model is 87.918%, measured on 1556 test images.

Now launch these commands to generate the snapshots:

```shell
# set the NPU env to generate snapshots
cd /workspace/Vitis-AI
source npu_ip/settings.sh VE2802_NPU_IP_O00_A304_M3   --nofallback

# go to ResNet18 directory
cd /workspace/tutorials/VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18
# re-launch inference to generate the snapshots
VAISW_SNAPSHOT_DIRECTORY=/workspace/Vitis-AI/snapshots/snapshot.py.vcor_resnet18.bs19 python3 ./code/test.py --batch_size 19 --data_root ./build/dataset/vcor --resume ./build/float/color_resnet18.pt
VAISW_SNAPSHOT_DIRECTORY=/workspace/Vitis-AI/snapshots/snapshot.py.vcor_resnet18.bs8  python3 ./code/test.py --batch_size  8 --data_root ./build/dataset/vcor --resume ./build/float/color_resnet18.pt
VAISW_SNAPSHOT_DIRECTORY=/workspace/Vitis-AI/snapshots/snapshot.py.vcor_resnet18.bs1  python3 ./code/test.py --batch_size  1 --data_root ./build/dataset/vcor --resume ./build/float/color_resnet18.pt
```

You should see some output text like this:

```
Test set: Average loss: 0.4558, Accuracy: 1363/1556 (87.596%)

[VAISW] 	 82 batches of 19 samples (the first batch is not used to compute the detailed times)
[VAISW] 	 1 input per batch (19x3x224x224) 
[VAISW] 	 1 output per batch (19x15) 
[VAISW] 	 1 total subgraph:
[VAISW] 		 1 VAISW (FPGA) subgraph: 100.00% of total MACs (35.70 G) 
[VAISW] 		 0 Framework (CPU) subgraph
[VAISW] 	 [INFO]:  snapshot directory dumped in ./snapshot.py.vcor_resnet18.bs19
[VAISW] 	 [INFO]:  snapshot dumped for VE2802_NPU_IP_O00_A304_M3
[VAISW] 	 1558 samples
```


### 5.4 Cpp Application for the Target

The [test_vcor_resnet18.cpp](files/cpp_code/vart_ml/demo/test_vcor_resnet18.cpp) file is derived from the original [vart_ml_demo.cpp](files/cpp_code/vart_ml/demo/vart_ml_demo.cpp) file. It differs in the postprocessing section, as the adopted dataset is VCoR (with 15 classes) instead of ImageNet (with 1,000 classes).

This file is not present in the original content of the `Vitis AI/src/vart_ml/demo` folder, as it is new. Therefore, the `demo/Makefiles` file also needs to be updated by adding `test_vcor_resnet18` to the `TARGETS`, as shown here:

```
TARGETS:= connect simple_demo simple_cpp_api_demo pipelining_demo threads_demo multi_models_demo vart_ml_demo test_vcor_resnet18
```

**Note**: The [vcor_test_GroundTruth.txt](files/resnet18/vcor_test_GroundTruth.txt) text file has only one blank (' ') character to separate the field in the same line, as for example:
```
6bdb5f9cbd.jpg purple
```
This is important during the post-processing phase (routine ``compare_gold()`` from [common.cpp](files/cpp_code/demo/common.cpp) file)
to compute the average prediction accuracy.


#### 5.4.1 Code Changes

If you launch these commands in the host PC you can see the modified ``*.cpp`` and ``*.h`` files (by searching for the ``//DB`` string):

```shell
cd vart_ml
grep -rn "//DB" *
```

Basically they are [common.cpp](files/cpp_code/vart_ml/demo/common.cpp) and [vart_ml_demo.cpp](files/cpp_code/vart_ml/demo/vart_ml_demo.cpp), 
besides the new [test_vcor_resnet18.cpp](files/cpp_code/vart_ml/demo/test_vcor_resnet18.cpp).

The first file was changed due to the preprocessing, as explained in section
[3.2 RGB or BGR PreProcessing](#32-rgb-or-bgr-preprocessing).

The second file was changed to print out some more information at run time:

```cpp
std::cout << " input  tensors = " <<  inputCnt[m] << std::endl;
std::cout << "output  tensors = " << outputCnt[m] << std::endl;
for (size_t i = 0; i < inputCnt[m]; i++)
{
	std::cout << "  input tensor["<<m<<"] height = " <<  inputTensors[m][i]->shape[0] << std::endl;
	std::cout << "  input tensor["<<m<<"] width  = " <<  inputTensors[m][i]->shape[1] << std::endl;
	std::cout << "  input tensor["<<m<<"] chan   = " <<  inputTensors[m][i]->shape[2] << std::endl;
}
for (size_t i = 0; i < outputCnt[m]; i++)
{
	std::cout << "output[" <<m<< "]tensor height = " << outputTensors[m][i]->shape[0] << std::endl;
	std::cout << "output[" <<m<< "]tensor width  = " << outputTensors[m][i]->shape[1] << std::endl;
	std::cout << "output[" <<m<< "]tensor chan   = " << outputTensors[m][i]->shape[2] << std::endl;  
}
```

#### 5.4.2 Steps to Run the Application

Copy the [test_vcor_resnet18.cpp](files/cpp_code/vart_ml/demo/test_vcor_resnet18.cpp) into  the ``files/vart_ml/demo`` local folder.


##### Step-1

Compile the code application in the ``vart_ml/demo`` subfolder following what described in section [3.3 Cross Compile the Demo Applications](#33-cross-compile-the-demo-applications).

Then copy the compiled executables from the ``demo`` folder of the host to the target board:

```shell
cd VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18/vart_ml
# copy the executables from host to target:
scp -r ./demo/* root@192.168.1.111:/home/root/Vitis-AI/src/vart_ml/demo/
```

##### Step-2

After that is done, copy (with ``scp``) the ResNet18 snapshot from host to the target board:

```shell
# copy the snapshots to the target board with IP Address 192.168.1.111
cd ${VITIS_AI_REPO}
scp -r ./snapshots root@192.168.1.111:/home/root/
```

##### Step-3

Copy the VCoR test images from host to the target board:

```shell
cd /workspace/tutorials/
cd VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18/build/dataset/vcor/

# copy the VCoR test images to the target board
scp -r   ./test/* root@192.168.1.217:/home/root/datasets/vcor/test/
```

##### Step-4

Copy the VCoR text files (`vcor_*.txt`) from host to the target board:

```shell
cd /workspace/tutorials/
cd VitisAI-NPU-Custom-ResNet18-Deployment/files/resnet18

# copy the VCoR test images to the target board
scp vcor_*.txt  root@192.168.1.217:/home/root/datasets/vcor/

# Connect to target board via ``ssh``
ssh root@192.168.1.217
```

##### Step-5

After connecting to the target boar, run the following commands:

```shell
cd
source ~/target_setup_vai.sh

cd Vitis-AI/src/vart_ml/demo

# inference in BGR
./test_vcor_resnet18 --snapshot ~/RESNET18/snapshots/snapshot.py.vcor_resnet18.bs19 --imgPath ~/datasets/vcor/test --labels ~/datasets/vcor/vcor_labels.txt --batchSize 19 --nbImages 1539 --goldFile ~/datasets/vcor/vcor_test_GroundTruth.txt  --colorFmt 0 > logfile_vcor_resnet18_bgr_vek280.txt 2>&1
#inference in RGB
./test_vcor_resnet18 --snapshot ~/RESNET18/snapshots/snapshot.py.vcor_resnet18.bs19 --imgPath ~/datasets/vcor/test --labels ~/datasets/vcor/vcor_labels.txt --batchSize 19 --nbImages 1539
--goldFile ~/datasets/vcor/vcor_test_GroundTruth.txt  --colorFmt 1 > logfile_vcor_resnet18_rgb_vek280.txt 2>&1
```

#### 5.4.3 Prediction Accuracy Results


You should get something similar to the following top-1 average prediction accuracy results:

```
============================================================

Accuracy Summary (RGB)
[AMD] [wrp_network TEST top1] 82.69% passed.
[AMD] [wrp_network TEST top5] 98.69% passed.
[AMD] [wrp_network ALL TESTS] 82.69% passed.
[AMD] 981.00 imgs/s (1539 images)

============================================================
Accuracy Summary (BGR)
[AMD] [wrp_network TEST top1] 26.39% passed.
[AMD] [wrp_network TEST top5] 71.06% passed.
[AMD] [wrp_network ALL TESTS] 26.39% passed.
[AMD] 985.68 imgs/s (1539 images)
```

**Note**: The average prediction top1 accuracy drops from 82.69 (in RGB mode) down to 26.39 (in BGR mode).




## 6 FPS Performance

This section covers measuring the inference time on the target board to process a certain amount of images, which results in a "Frames-Per-Second" (shorty "fps" or "FPS") performance value.

You must set the `VAISW_RUNSESSION_SUMMARY=all` environment variable, which enables the dumping of performance statistics at the end of the application’s execution.


### 6.1 Standard ResNet50

The performance for the Pytorch ResNet50 can be measured by running these commands on the target board:
```shell
export VAISW_RUNSESSION_SUMMARY=all
source ~/target_setup.sh
cd ~/Vitis-AI/src/vart_ml/demo

# Batch Size = 1
./vart_ml_demo --batchSize 1   --goldFile ~/datasets/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt \
  --imgPath /home/root/datasets/imagenet/ILSVRC2012_img_val --snapshot ~/snapshots/snapshot.py.resnet50.bs1 \
  --labels /etc/vai/labels/resnet50_labels.txt   --nbImages 1349 --colorFmt 1 --dataFormat native

# Batch Size = 19
./vart_ml_demo --batchSize 19   --goldFile ~/datasets/imagenet/ILSVRC_2012_val_GroundTruth_10p.txt \
  --imgPath /home/root/datasets/imagenet/ILSVRC2012_img_val --snapshot ~/snapshots/snapshot.py.resnet50.bs19 \
  --labels /etc/vai/labels/resnet50_labels.txt   --nbImages 1349 --colorFmt 1 --dataFormat native
```

You should see the following output text for BS=1:

```
============================================================
Accuracy Summary:
[AMD] [resnet50 TEST top1] 71.68% passed.
[AMD] [resnet50 TEST top5] 88.44% passed.
[AMD] [resnet50 ALL TESTS] 71.68% passed.
[AMD] VART ML runner data format was set to NATIVE.
[AMD] 364.31 imgs/s (1349 images)
[VART]
[VART]		 board XIL_VEK280_REVB3 (AIE: 304 = 38x8)
[VART]		 1349 inferences of batch size 1 (the first inference is not used to compute the detailed times)
[VART]		 1 input layer. Tensor shape: 1x224x224x3 (INT8)
[VART]		 1 output layer. Tensor shape: 1x1000 (INT8)
[VART]		 1 total subgraph:
[VART]			 1 VART (AIE) subgraph
[VART]			 0 Framework (CPU) subgraph
[VART]		 1349 samples
[VART]
[VART] "resnet50" run summary:
[VART]		 detailed times in ms
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Performance Summary               |  ms/batch  |  ms/batch  |  ms/batch  |   sample/s |
[VART] |                                   |    min     |    max     |   median   |   median   |
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Whole Graph total                 |       2.66 |       2.87 |       2.70 |     369.96 |
[VART] |   VART total (   1 sub-graph)     |       2.65 |       2.85 |       2.67 |     373.83 |
[VART] |     AI acceleration (*)           |       2.49 |       2.51 |       2.50 |     400.00 |
[VART] |     CPU processing                |       0.15 |       0.34 |       0.17 |            |
[VART] |       Others                      |            |            |       0.17 |            |
[VART] |   Others                          |            |            |       0.03 |            |
[VART] +-----------------------------------+------------+------------+------------+------------+
```


You should see the following output text for BS=19:

```
============================================================
Accuracy Summary:
[AMD] [resnet50 TEST top1] 71.68% passed.
[AMD] [resnet50 TEST top5] 88.44% passed.
[AMD] [resnet50 ALL TESTS] 71.68% passed.
[AMD] VART ML runner data format was set to NATIVE.
[AMD] 6111.05 imgs/s (1349 images)
[VART]
[VART]		 board XIL_VEK280_REVB3 (AIE: 304 = 38x8)
[VART]		 71 inferences of batch size 19 (the first inference is not used to compute the detailed times)
[VART]		 1 input layer. Tensor shape: 19x224x224x3 (INT8)
[VART]		 1 output layer. Tensor shape: 19x1000 (INT8)
[VART]		 1 total subgraph:
[VART]			 1 VART (AIE) subgraph
[VART]			 0 Framework (CPU) subgraph
[VART]		 1349 samples
[VART]
[VART] "resnet50" run summary:
[VART]		 detailed times in ms
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Performance Summary               |  ms/batch  |  ms/batch  |  ms/batch  |   sample/s |
[VART] |                                   |    min     |    max     |   median   |   median   |
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Whole Graph total                 |       3.05 |       3.12 |       3.06 |    6213.21 |
[VART] |   VART total (   1 sub-graph)     |       3.02 |       3.06 |       3.03 |    6272.70 |
[VART] |     AI acceleration (*)           |       2.55 |       2.57 |       2.56 |    7436.40 |
[VART] |     CPU processing                |       0.46 |       0.50 |       0.47 |            |
[VART] |       Others                      |            |            |       0.47 |            |
[VART] |   Others                          |            |            |       0.03 |            |
[VART] +-----------------------------------+------------+------------+------------+------------+
```

### 6.2 Custom ResNet18

The performance for the PyTorch custom ResNet18 is measured on the target board at different batch sizes: 1, 8 and 19.


Executing the following commands  on the target board:

```shell
source /etc/vai.sh
source ~/scripts/target_setup_vai.sh
export VAISW_RUNSESSION_SUMMARY=all
cd Vitis-AI/src/vart_ml/demo

# inference with BS=1
./test_vcor_resnet18  --batchSize 1   --goldFile ~/datasets/vcor/vcor_test_GroundTruth.txt 
--imgPath /home/root/datasets/vcor/test --snapshot ~/snapshots/snapshot.py.vcor_resnet18.
bs1   --labels ~/datasets/vcor/vcor_labels.txt   --nbImages 1539 --colorFmt 1 --dataFormat 
native

# inference with BS=19
./test_vcor_resnet18  --batchSize 19   --goldFile ~/datasets/vcor/vcor_test_GroundTruth.txt --imgPath /home/root/datasets/vcor/test --snapshot ~/snapshots/snapshot.py.vcor_resnet18.bs19   --labels ~/datasets/vcor/vcor_labels.txt   --nbImages 1539 --colorFmt 1 --dataFormat native
```

You should see the following output text related to BS=1:

```
ccuracy Summary:
[AMD] [wrp_network TEST top1] 82.69% passed.
[AMD] [wrp_network TEST top5] 98.69% passed.
[AMD] [wrp_network ALL TESTS] 82.69% passed.
[AMD] VART ML runner data format was set to NATIVE.
[AMD] 832.94 imgs/s (1539 images)
[VART]
[VART]		 board XIL_VEK280_REVB3 (AIE: 304 = 38x8)
[VART]		 1539 inferences of batch size 1 (the first inference is not used to compute the detailed times)
[VART]		 1 input layer. Tensor shape: 1x224x224x3 (INT8)
[VART]		 1 output layer. Tensor shape: 1x15 (INT8)
[VART]		 1 total subgraph:
[VART]			 1 VART (AIE) subgraph
[VART]			 0 Framework (CPU) subgraph
[VART]		 1539 samples
[VART]
[VART] "wrp_network" run summary:
[VART]		 detailed times in ms
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Performance Summary               |  ms/batch  |  ms/batch  |  ms/batch  |   sample/s |
[VART] |                                   |    min     |    max     |   median   |   median   |
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Whole Graph total                 |       1.13 |       1.33 |       1.16 |     862.81 |
[VART] |   VART total (   1 sub-graph)     |       1.12 |       1.30 |       1.13 |     884.17 |
[VART] |     AI acceleration (*)           |       0.99 |       1.00 |       0.99 |    1007.05 |
[VART] |     CPU processing                |       0.13 |       0.31 |       0.14 |            |
[VART] |       Others                      |            |            |       0.14 |            |
[VART] |   Others                          |            |            |       0.03 |            |
[VART] +-----------------------------------+------------+------------+------------+------------+
```


You should see the following output text related to BS=19:

```
============================================================
Accuracy Summary:
[AMD] [wrp_network TEST top1] 82.69% passed.
[AMD] [wrp_network TEST top5] 98.69% passed.
[AMD] [wrp_network ALL TESTS] 82.69% passed.
[AMD] VART ML runner data format was set to NATIVE.
[AMD] 12379.66 imgs/s (1539 images)
[VART]
[VART]		 board XIL_VEK280_REVB3 (AIE: 304 = 38x8)
[VART]		 81 inferences of batch size 19 (the first inference is not used to compute the detailed times)
[VART]		 1 input layer. Tensor shape: 19x224x224x3 (INT8)
[VART]		 1 output layer. Tensor shape: 19x15 (INT8)
[VART]		 1 total subgraph:
[VART]			 1 VART (AIE) subgraph
[VART]			 0 Framework (CPU) subgraph
[VART]		 1539 samples
[VART]
[VART] "wrp_network" run summary:
[VART]		 detailed times in ms
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Performance Summary               |  ms/batch  |  ms/batch  |  ms/batch  |   sample/s |
[VART] |                                   |    min     |    max     |   median   |   median   |
[VART] +-----------------------------------+------------+------------+------------+------------+
[VART] | Whole Graph total                 |       1.46 |       1.54 |       1.49 |   12794.61 |
[VART] |   VART total (   1 sub-graph)     |       1.45 |       1.50 |       1.46 |   13049.45 |
[VART] |     AI acceleration (*)           |       1.05 |       1.06 |       1.05 |   18009.48 |
[VART] |     CPU processing                |       0.40 |       0.44 |       0.40 |            |
[VART] |       Others                      |            |            |       0.40 |            |
[VART] |   Others                          |            |            |       0.03 |            |
[VART] +-----------------------------------+------------+------------+------------+------------+
```

Comments:

0. Larger the BatchSize and more the FPS, which is intuitively expected.

1. The best performance is for the snapshot with BatchSize = 19, as expected.

2. The "ideal" NPU computation time (the line with `AIE`) is `1.05ms` for the 19 images in the batch, which means `18009 FPS`.

3. The entire NN graph (`Whole Graph`) requires `1.46ms`, which means `12794 FPS`.

4. This is the time for standalone inference; the pre-processing and post-processing tasks executed on the ARM CPU are not included.


## Appendix A


### A1 How to Download the ImageNet Dataset


You can download the tar files for training and validation sets of [ImageNet](https://www.image-net.org/) dataset, by executing the following commands (see comments):

```shell
# create a local directory
mkdir imagenet
cd imagenet
mkdir val    # validation dataset
mkdir train  # training   dataset

# get training set
cd train
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate

# get validation set
cd ../val
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
cd ..
```

After you have two tar files, in the same directory, download
[official Pytorch example script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) and read it.

#### A 1.1 Training Set

At this stage, `imagenet/train` contains 1000 compressed `.tar` files, one for each category.
Execute the following commands:

``` shell
cd imagenet
cd train

# For each .tar file:
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file

find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

cd ..

# This results in a training directory like so:
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
```

#### A 1.2 Validation Set

Now execute the following commands, by still remaining in the `imagenet/val` folder:

```shell

cd imagenet
cd val

# extract validation .tar;
tar -xvf ILSVRC2012_img_val.tar

# get script from soumith
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
# the script creates all class directories and moves images into corresponding directories
bash -x ./valprep.sh
```

You can check the amount of images (50000 expected) with the following command:

```shell
find .  -name "*.JPEG" | wc -l
```


#### A 1.3 Conclusion


Your `imagenet/` directory has now a proper folders hierarchy. It will be recognized by Pytorch `ImageFolder dataloader` with code
similar to this:

```python
from torchvision import datasets

train_dataset = datasets.ImageFolder('imagenet/train', transform=transform)
val_dataset = datasets.ImageFolder('imagenet/val',transform=transform)
```

<br>



# References

- [Vitis AI Early Access Release Secure Site](https://account.amd.com/en/member/vitis-ai-ea-release.html)

- [Valid URL for downloading Imagenet dataset?](https://stackoverflow.com/questions/64714119/valid-url-for-downloading-imagenet-dataset)

- [ResNet50 V1.5](https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/)


### License 

The MIT License (MIT)

Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

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

