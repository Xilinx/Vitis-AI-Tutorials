<table>
 <tr>
   <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Tutorials</h1>
   </td>
 </tr>
 <tr>
 <td align="center"><h1>Medical AI Application Acceleration with Xilinx AI Stack</h1>
 </td>
 </tr>
</table>

# Introduction
Medical AI is becoming an essential part of modern healthcare. Hardware implementation of the algorithm part of these applications offer much better efficiency and lower latency. This tutorial demonstrates how a medical application developer can take a medical dataset, develop and evaluate an end-to-end AI accelerated application using Xilinx&reg; Vitis&trade; AI, without writing any lower level RTL code.
Here we take the open-source Skin Lesions dataset, create a machine learning network and train it on an x86 platform. Then we use the Xilinx Vitis AI toolset to quantize, compile and evaluate the model in one of the Xilinx AI inference platforms, the ZCU102 board.

# Prerequisites
## Host Machine Prep
+ A CentOS or an Ubuntu host machine configured as described in the Table 1 of the <a href="https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf">Vitis AI User Guide (UG1414)</a>.
+ Docker-ce installed. (see instructions <a href="https://docs.docker.com/engine/install/">Install Docker Engine</a>).
+ A terminal emulator and/or SSH Client (i.e. Putty) installed.
+ See the <a href="https://github.com/Xilinx/Vitis-AI">Vitis AI GitHub README.md</a> for details.

## Target Platform: ZCU102 Evaluation Board Prep
+ Configure the ZCU102 board so that it boots from an SD card ( for instructions refer to chapter 2 of <a href="https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf">UG1414</a>)
+ Follow the instructions in the Guide_to_create_boot image_zcu102.pdf (look for it in the cloned folder)
and create a bootable SD card image.
+ Connect it to a DHCP enabled network via its ethernet port.

# Dataset Download and Preprocessing
+ In the Files folder, copy the "notebooks_for_training" folder to your host machine
+ Follow the steps described in the 'dataaug.ipynb' Jupyter notebook

# ML Network Training and Model Creation
+ Follow the steps described in the ''training.ipynb' Jupyter notebook

# Porting the Model to Target Platform (ZCU102 Eval Board)
## 1. Vitis AI Tools Docker Bring Up
+ Follow the instructions described in the Phase-2 part of the article (look for it in the cloned folder)
## 2. Model Quantization
+ Follow the instructions described in the Phase-2 part of the article (look for it in the cloned folder)
## 3. Network Compilation
+ Follow the instructions described in the Phase-2 part of the article (look for it in the cloned folder)

# Application Evaluation on the Target ZCU102 Board
Once the zcu102 board is up and running it automatically launches a Jupyter notebook server. Jupyter notebooks mentioned in the article are already included in the board image and should be accessible on a browser using http://<board-ip>:8888 address. For Instructions on how to gather the board's ip address by accessing it over JTAG or via SSH refer to the chapter 2 of <a href="https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf">UG1414</a>)
+ Copy all subdirectories from ‘/test_dir’ (created during ‘Dataset download and preprocessing’ step) to the ‘/workspace/skinl/data’ directory in the board file system. Make sure to maintain the directory structure.
+ Follow the steps described in the Jupyter notebook 'skinl_app.ipynb' to evaluate the application.

# License

Licensed under the Apache License, Version 2.0 - see the [LICENSE.txt](files/LICENSE.txt) file for details

# Acknowledgments
This tutorial uses the following third-party material:
Tschandl, Philipp, 2018, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", https://doi.org/10.7910/DVN/DBW86T, Harvard Dataverse, V1, UNF:6:IQTf5Cb+3EzwZ95U5r0hnQ
