# Vitis AI (on Ultra96V2) Custom Platform Tutorial
This repository contains sources for topics related to porting Vitis&trade; AI 1.0 to a custom platform. For the purposes of these tutorials, we'll follow the workflow with the Ultra96-V2 board from Avnet, and will add input capability from the MIPI CSI2 RX input source @ 640x480p60 with inline CSC to BGR for a low-latency ML platform. The tutorial has three parts:
* [Building a custom Vitis Ultra96V2 + MIPI Platform](/files/tutorials/1-Building-a-Vitis-Ultra96V2-and-MIPI-platform.md): Generating a custom hardware and software platform with the Xilinx&reg; Vivado&reg; design suite and the Vitis unified software platform, using a MIPI input stream.
* [Working with the Model Zoo](/files/tutorials/2-Working-with-the-Model-Zoo.md): Compiling pre-trained and pruned models for a new DPU configuration, using the output of our custom Ultra96-V2 Platform.
* [Designing with Vitis AI for a custom Edge Platform](/files/tutorials/3-Designing-with-Vitis-AI-for-a-custom-Edge-Platform.md): Building a custom application and deploy Vitis AI and high level libraries to a custom edge platform, and building a compound image processing pipeline with both FaceDetection and FaceLandmark high-level libraries (and models) provided by the Vitis AI Library.

## Requirements for the Design
In order to complete the tutorials in this repository, you need the following software and hardware tools:

 - Xilinx Design Tools (Vivado design suite and Vitis unified software platform), version 2019.2
 - Board files for Ultra96-V2 installed
 - The Ultra96-V2 board
 - Power supply for Ultra96-V2 board
 - MicroUSB to USB-A cable (x2)
 - AES-ACC-USB-JTAG board
 - A blank, FAT32 formatted microSD card
 - Xilinx CSI2 RX MIPI IP License
 - D3 Engineering Designcore Camera Mezzanine Board OV5640 (MIPI Input Source)

Optional additional tools:
-   DisplayPort monitor
-   Mini-display port cable suitable for the chosen monitor
-   USB Webcam

## Preparing your workspace
Clone this repository to your local machine.
