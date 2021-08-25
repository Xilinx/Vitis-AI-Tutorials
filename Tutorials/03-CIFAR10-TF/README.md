<table style="width:100%">
  <tr>
    <th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>CIFAR10 Classification using Vitis AI and TensorFlow</h2>
</th>
  </tr>

</table>

### Current status

1. Tested with Vitis AI 1.0
2. Tested in hardware on ZCU102

## 1. Introduction

This tutorial introduces the user to the Vitis AI TensorFlow design process and will illustrate how to go from a python description of the network model to running a compiled model on the Xilinx DPU accelerator.

The application code in this example design is written in Python and uses the Unified APIs that were introduced in Vitis AI 1.0.

We will run the following steps:

  + Training and evaluation of small custom convolutional neural network using TensorFlow.
  + Removal of the training nodes and conversion of the graph variables to constants (..often referred to as 'freezing the graph').
  + Evaluation of the frozen model using the CIFAR-10 test dataset.
  + Quantization of the floating-point frozen model.
  + Evaluation of the quantized model using the CIFAR-10 test dataset.
  + Compilation of the quantized model to create the .elf file ready for execution on the DPU accelerator IP.
  + Conversion of the .elf file to a Shared Library/Static Object file (.so) for use with the Python APIs.
  + Download and run the application on an evaluation board.


## 2. The CIFAR-10 dataset

CIFAR-10 is a publically available dataset that contains a total of 60k RGB images each of which are 32pixels x 32pixels x8bits per color channel. The small image size of 32 x 32 means that it is not very useful for real-world applications, but the CIFAR-10 dataset makes a good starting point for studying machine learning. The complete dataset of 60k images is normally divided into 50k images for training and 10k images for validation.

There are a total of 10 mutually exclusive classes (or labels):


| Class index | Class name |
| :---: | --- |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |



## 3. The convolution neural network

The convolutional neural network in this design is extremely simple and consists of multiple layers of 2D convolution, batch normalization and ReLU activation. The network is described in the customCNN.py python script.


## 4. Prerequistes

  + An Ubuntu host machine configured as per the requirements defined in Table 1 of the <a href="https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf">Vitis AI User Guide (UG1414)</a>.

  + The host machine should also have:
    + The vitis-ai-docker-runtime which is the runtime docker image for DPU-v2 development.
    + Either the vitis-ai-docker-tools GPU image or the vitis-ai-docker-tools CPU image.
    + A terminal emulator and/or SSH Client such as Putty.

See the <a href="https://github.com/Xilinx/Vitis-AI">Vitis AI GitHub README.md</a> for details.

  + The host machine will need access to the internet for downloading files.
  + A ZCU102 or other supported evaluation board.
  + This tutorial assumes the user is familiar with Vitis AI, Python3, TensorFlow and has some knowledge of machine learning principles.


## 5. Shell scripts

  + ``0_setenv.sh`` : Sets all necessary environment variables used by the other scripts. Mosts variables can be edited by the user to configure the environment for their own requirements, for example the targetted evaluation board type can be changed by editing the ``export BOARD=ZCU102`` variable.

  Note that the image_input_fn.py Python script contains a reference to the list of calibration images as an absolute path:

 ```python
 calib_image_list = './build/quantize/images/calib_list.txt'
 ```

 ..so if either of the ``BUILD`` or ``QUANT`` variables are modified in ``0_setenv.sh`` then the absolute path in image_input_fn.py will also need to be modified.

 It is highly recommended to leave the ``NUM_IMAGES`` variable set to 1000 as this is the minimum recommnded number of images for calibration of the quantization.


  + ``1_train.sh``  : Runs training and evaluation of the network. Will save the trained model as a graph and checkpoint.

  + ``2_freeze.sh`` : Converts the inference graph and checkpoint into a single binary protobuf file (.pb). The output .pb file is generally known as a 'frozen graph' since all variables are converted into constants and gpraph nodes associated with training such as the optimizer and loss functions are stripped out.

  + ``3_eval_frozen_graph.sh`` : This is an optional step which tests the accuracy of the frozen graph. The accuracy results should be very simialr to the results obtained after training.

  + ``4_quant.sh`` : This script first creates a set of image files to be used in the calibration phase of quantization and then launches the ``vai_q_tensorflow quantize`` command to convert the floating-point frozen graph to a fixed-point integer model.

  + ``5_eval_quant_graph.sh`` : This step is optional but highly recommended - it will run the same evaluation function that was used to evaluate the frozen graph on the quantized model. Users should confirm that the accuracy reported by the evaluation of the quantized model is sufficient for their requirements and similar to the results for the floating-point models.

  + ``6_compile.sh`` : Launches the ``vai_c_tensorflow`` command to compile the quantized model into an .elf file.

  + ``7_make target.sh`` : Copies the .elf and images to the target folder ready to be copied to the evaluation board's SD card. The .elf file will be converted to a Shared Library/Static Object .so file using the ARM compiler on the evaulation board.



## 6. Image pre-processing

All images are undergo simple pre-processing before being used for training, evaluation and quantization calibration. The images are normalized to bring all pixel values into the range 0 to 1 by dividing them by 255.


## 7. Generating the application

1. Clone or download this GitHub repository to your local machine where you have installed the necessary tools.

2. Open a terminal and ``cd`` into the repository folder. Start the Vitis AI tools docker - depending on where you have installed it, the command will look something like this:

 ``/home/username/Vitis-AI/docker_run.sh xilinx/vitis-ai:tools-1.0.0-gpu``

 or like this if you are using the CPU-only tools docker:

  ``/home/username/Vitis-AI/docker_run.sh xilinx/vitis-ai:tools-1.0.0-cpu``.

If you are running the training step, it is highly recommnded to use the GPU version of the Vitis Tools Docker container.

3. Activate the TensorFlow virtual environment with ``conda activate vitis-ai-tensorflow``.


![Alt text](./files/img/tools_docker.png?raw=true "tools_docker")



4. The complete tools flow can be run just by executing the ``source ./run_all_tools.sh`` command, or by running each of the steps in order (from 0_xx to 7_xx):


```shell
$ source ./0_setenv.sh
$ source ./1_train.sh
.
.
$ source ./7_make_target.sh
```

5. Ensure that you have a .elf file in the `target` folder

6. Exit from the Tools docker container (type 'exit' at the prompt.)


When the flow completes, the `target` folder will contain everything needed to run the application on the target platform. This `target` folder will need to be copied to the SD card (see steps below).It should contain the following files:

  + a dpuv2_rundir folder that contains a meta.json file.
  + the cifar10_app.py script which is the Python script to execute on the target board.
  + the runner.py Python script.
  + The images folder that contains JPEG images from the dataset.
  + The .elf file.
  + A shell script called compile_so.sh that will generate the .so file.



## 8. Running the application on the ZCU102 board

The SD card for the ZCU102 will need to be flashed with the appropriate board image which can be found <a href="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2019.2.img.gz">here</a>.

Follow the instructions in Chapter 2 of <a href="https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf">UG1414</a> to flash the SD card. It is highly recommended to test that the ZCU102 boots correctly from the SD card before going further.


The complete ``target`` folder needs to be copied to the ``/home/root`` folder of the flashed SD card, this can be done in one of several ways:

1. Direct copy to SD Card:

  + If the host machine has an SD card slot, insert the flashed SD card and when it is recognised you will see two volumes, BOOT and ROOTFS. Navigate into the ROOTFS and then into the ``/home`` folder.  Make the ``root`` folder writeable by issuing the command ``sudo chmod -R 777 root`` and then copy the entire ``target`` folder from the host machine into the ``root`` folder of the SD card.

  + Unmount both the BOOT and ROOTFS volumes from the host machine and then eject the SD Card from the host machine.

2. With scp command:

  + If the ZCU102 is connected to the same network as the host machine, the ``target`` folder can be copied using scp.

  + The command will be something like ``scp -r ./target root@192.168.1.227:~/``  assuming that the ZCU102 IP address is 192.168.1.227 - adjust this and the path to the ``target`` folder as appropriate for your system.

  + If the password is asked for, insert 'root'.



With the ``target`` folder copied to the SD Card and the ZCU102 booted, you can issue the command for launching the application - note that this done on the ZCU102 board, not the host machine, so it requires a connection to the ZCU102 such as a serial connection to the UART or an SSH connection via Ethernet.


The commands to start the application are:

```shell
cd target
source ./compile_so.sh
python3 cifar10_app.py -t 3 -b 6 -j /home/root/target/dpuv2_rundir/
```

..where -t defines the number of threads and -b defines the input batchsize. The -j option gives the path of the folder that contains the meta.json file.

The application will start and after a few seconds will show the throughput in frames/sec.

## 9 References

+ CIFAR-10 dataset developed by Alex Krizhevsky for <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">Learning Multiple Layers of Features from Tiny Images</a>.
