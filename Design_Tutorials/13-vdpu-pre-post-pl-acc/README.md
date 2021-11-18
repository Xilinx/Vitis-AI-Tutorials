<!--
Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

<table class="sphinxhide">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis-AI™ 1.4 - Machine Learning Tutorials</h1>
    <a href="https://www.xilinx.com/products/design-tools/vitis.html">See Vitis™ Development Environment on xilinx.com</br></a>
    <a href="https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html">See Vitis-AI™ Development Environment on xilinx.com</a>
    </td>
 </tr>
</table>


# Pre- and Post-processing PL Accelerators for ML with Versal DPU

## Current status


1. Designed with Vitis 2020.2 environment

2. Tested in hardware ON VCK190PP with ``XVDPU-PCIE TRD`` platform.


#### Authors

- [Daniele Bagni](mailto:danieleb@xilinx.com):
- [Peter Schillinger](mailto:peters@xilin.com):
- [Herve Ratigner](mailto:herver@xilinx.com):
- [Kay Migge](mailto:kaym@xilinx.com)

#### Acknowledgements
- [Srikanth Erusalagandi](mailto:serusal@xilinx.com)
- [Tony McDowell](mailto:tmcdowe@xilinx.com)
- [Florent Werbrouck](mailt@florentw@xilinx.com)


#### Last update  

18 November 2021



# 1 Introduction

This repository contains the **Pre- and Post-processing** kernels to be used in Machine Learning (**ML**) jointly to the Deep learning Processor Unit (shortly **DPU**) to accelerate in the Programmable Logic (shortly **PL**)  the same tasks that otherwise would be executed by the ARM host CPU of the FPGA target device. Off-loading those two tasks from the ARM CPU improves the overall system performance in terms of frames-per-second (**fps**).

The two accelerators were tested using data coming from the Semantic Segmentation CNN of this tutorial:
[VAI-KERAS-FCN8-SEMSEG](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/05-Keras_FCN8_UNET_segmentation), where the CNN was retrained with larger image sizes as 1920x832, but the accelerators are general enough to be used or easily adapted with few changes also to other Deep Learning applications, such as Object Detection or Image Classification.

At the moment we are targeting the VCK190 Pre-Production (PP) board, with the so called ``XVDPU TRD`` platform, which contains a DPU designed with 96 AI Engine cores (over the 400 available) besides other PL resources (BRAMs, URAMs, FFs, LUTs, DSPs).

The two accelerators do not use any core from the AI Engine array of the Versal ACAP, to be more portable later also on MPSoC devices. Their design is done with **Vitis High Level Synthesis** (shortly **HLS** in the following of this document) within the Vitis suite.

The application running on the host ARM CPU applies **XRT APIs**.

This tutorial can also be seen as a complete example of how using the [WAA](https://github.com/Xilinx/Vitis-AI/tree/master/demo/Whole-App-Acceleration) flow with **Vitis 2020.2** targeting the VCK190 PP board.


## WARNING

Everything shown in this project was designed on an Ubuntu 18.04.5 Desktop with related Vitis 2020.2 suite.
This project was never tried on a Windows OS PC.


## Dos-to-Unix Conversion

In case you might get some strange errors during the execution of the scripts, you have to pre-process -just once- all the``*.sh``, ``*.cpp``, ``*.h`` files with the [dos2unix](http://archive.ubuntu.com/ubuntu/pool/universe/d/dos2unix/dos2unix_6.0.4.orig.tar.gz) utility.
In that case run the following commands from your Ubuntu host PC (out of the Vitis AI docker images):
```bash
#sudo apt-get install dos2unix
cd <WRK_DIR> #your working directory
for file in $(find . -name "*.sh" ); do dos2unix ${file}; done
for file in $(find . -name "*.tcl"); do dos2unix ${file}; done
for file in $(find . -name "*.h"  ); do dos2unix ${file}; done
for file in $(find . -name "*.c*" ); do dos2unix ${file}; done
```


# 2 Design Flow with HLS

For each accelerator there are two project folders named [hls](files/preproc/hls) and [vitis](files/preproc/vitis), respectively with the source files adopted in the standalone HLS design and in the final Vitis system design.

For each accelerator the files are the same among the two subfolders, the only difference being that the [vitis](files/preproc/vitis) folder requires also the ARM host code with XRT APIs, which is not needed  
by the [vitis_hls](files/preproc/vitis_hls). Therefore, the file [dpupreproc_defines.h](files/preproc/vitis/kernels/dpupreproc_defines.h) must have the line ``#define ARM_HOST`` commented when used in the kernels subproject, but it must have such line not commented when used in the host code, as shown in the [dpupreproc_defines.h](files/preproc/vitis/host/dpupreproc_defines.h) (this is the only difference between these two files that have the same name and are placed in different folders).

The same concept is valid also for the post-processing kernel and its related folders [hls](files/postproc/hls) and [vitis](files/postproc/vitis), respectively for the source files adopted in the standalone HLS design and in the final Vitis system design.


## WARNING

In order to avoid proliferation of files with the same name, we used soft-links for the files that are in common between either the standalone HLS or the Vitis project.
Run the following command before reading the rest of this document:

```shell
cd VDPU-PRE-POST-PLACC/files
bash -x ./prepare_files
```

## 2.1 Pre-processing Kernel


### 2.1.1 Kernel Functionality

In ML, the preprocessing job has to change the statistics on the data to be used for training the CNN in order to facilitate such training.
There many ways to do that preprocessing, the most popular ways are the following two explained with Python code fragments, respectively the "Caffe" and "TensorFlow" method (this is my terminology to explain it in a simple way):

```python
. . .
if (TensorFlow_preproc): #TensorFLow mode
  _B_MEAN = 127.5
  _G_MEAN = 127.5
  _R_MEAN = 127.5
  MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
  SCALES = [0.007843137, 0.007843137, 0.007843137] # 1.0/127.5
else: #Caffe mode
  _B_MEAN = 104.0
  _G_MEAN = 117.0
  _R_MEAN = 123.0
  MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
  SCALES = [1.0, 1.0, 1.0]
. . .
def preprocess_one_image_fn(image_path, pre_fix_scale, width, height):
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    B, G, R = cv2.split(image)
    B = (B - means[0]) * scales[0] * pre_fix_scale
    G = (G - means[1]) * scales[1] * pre_fix_scale
    R = (R - means[2]) * scales[2] * pre_fix_scale
    image = cv2.merge([R, G, B])
    image = image.astype(np.int8)
    return image
```

From one hand, in Caffe normally the input image R G B pixels are manipulated by subtracting the R G B mean values (``MEANS``) of all the training dataset images and so the output data is of type ``signed char`` (in C/C++) or ``int8`` (python numpy), with a possible range from -128 to +127, being 8-bit.
From another hand,  in TensorFlow normally the pixels are manipulated by normalizing them in the interval from -1.0 to 1.0 or something similar.

During the CNN training phase the pre-processimg works on floating point data, but in real life the DPU works with ``int8`` after quantization with [Vitis AI tools](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Quantizer) and so in the application running on the target device in real time, you have to scale the data with the ``fix_scale`` parameter that comes from a query to the DPU before starting the ML prediction (inference) task itself, with Python code similar to this:

```python
input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
pre_fix_scale = 2**input_fixpos
```

In conclusion, before starting its job,  the image pre-processing module requires six floating point input parameters:
```
float MEANS[3];
float SCALES[3];
```
and the scaling factor that could be either
``float pre_fix_scale;``
or alternatively
``int input_fixpos;``
this last one being a value from 1 to 7 because it represents the exponent ``i`` of a power of ``2,`` that is ``2^i``.    

In the HLS TestBench (TB) all those parameters are fixed in the [dpupreproc_defines.h](files/preproc/vitis/kernels/dpupreproc_defines.h) file, to test the functionality of the core.

The input image used in the self-checking TB was taken from the test dataset of the
[VAI-KERAS-FCN8HDTV-SEMSEG](https://gitenterprise.xilinx.com/danieleb/VAI-KERAS-FCN8HDTV-SEMSEG) CNN.

### 2.1.2 HLS Design

Just launch the command
```shell
vitis_hls -f hls_script.tcl
```
and the whole HLS flow will run in its steps: CSIM, SYN, coSIM and IMP. See the related screenshots of Figures 1, 2, 3 and 4.


![figure1](files/preproc/img/pre_csim.png)

*Figure 1. Pre-processing CSIM step with Vitis HLS*


![figure2](files/preproc/img/pre_syn.png)

*Figure 2. Pre-processing SYN step with Vitis HLS*

![figure3](files/preproc/img/pre_cosim.png)

*Figure 3. Pre-processing coSIM step with Vitis HLS*

![figure4](files/preproc/img/pre_imp.png)

*Figure 4. Pre-processing IMP step with Vitis HLS*


Note that the file [dpupreproc_defines.h](files/preproc/hls/src/dpupreproc_defines.h) must have the line ``#define ARM_HOST`` commented.

As you see from figure 4, after Place-And-Route, the accelerator consumes the following resources: 4294 LUT, 7042 FF, 2 BRAM and 13 DSP from the  Versal 1902 device  with a minimum clock period of 2.8ns, which corresponds to  356MHz maximum clock frequency.

Figure 3 reports the cycle accurate simulation (coSIM step), considering the amount of clock cycles to process the whole image before sending it back to DDR memory, the latency of this kernel is  given by 1198260 (cycles) x 2.8ns (clock period) = 3.359ms.

Even assuming a longer clock period of 5ns (corresponding to 200MHz clock frequency) the latency would become 5.99ms.

Note that this latency is the time to process the entire frame (1920x832x3) of pixels because this is the way Vitis HLS works if you want to do a functional cycle accurate simulation (acknowledged as "coSIM") of the accelerator. But in itself this core has a real latency of few dozens of clock cycles. Such effective latency could be exploited either by using AXI4 Streaming interfaces (which are not accepted by the DPU core, which is unable to work in a streaming mode) instead of full MAXI4 interfaces or by adding a ping-pong buffer of few image lines among the Pre-processing accelerator and the external DDR memory.  




## 2.2 Post-processing Kernel


###2.2.1 Kernel Functionality

In ML, the post-processing job has to present the "features map" generated by the CNN in a form that can be understood by human beings; in case of Semantic Segmentation this require to understand which pixel of the image belongs to which class.

In this application case there are 28 classes per each pixel, so the output tensor generated by the DPU is a 3D volume with half the horizontal and vertical size of the input images -that is 1920/2 and 832/2 respectively- and 28 channels.

For each set of 28 values related to one pixel, the post-processing task computes first the Softmax classifier and then search for its maximum value and related index: the index of this max value represent the object class (coded with a number from 0 to 27) with the highest probability to be predicted by the CNN. This can be illustrated by looking at the C/C++ code of the file [dpupostproc_ref.cpp](files/postproc/common_src/dpupostproc_ref.cpp):

```cpp
void ref_SoftMax(signed char  *inp_data, float *out_data, float post_scale_factor, unsigned char size)
{
  float result[MAX_NUM_OF_CLASSES];
  float sum = 0.0f;
  for (int i=0; i<size; i++) {
	  int addr = 128+inp_data[i];
	  assert( (addr>=0) & (addr<=255) );
    float x = addr*post_scale_factor;
    result[i]= expf(x);
    sum += result[i];
  }
  float div = 1.0f / sum;
  for (int i=0; i<size; i++)
    out_data[i]=result[i] * div;
}

void ref_ArgMax(float *inp_data, unsigned char *out_max, unsigned char *out_index, unsigned char size)
{
  unsigned char  max=0, index=0;
  for (int i=0; i<size; i++) {
    float val = inp_data[i];
    val = val * 255.0f;
    int i_val = (int) val;
    assert( (i_val<=255) & (i_val>=0) );
    unsigned char u_val = i_val;
    if (u_val > max) {
    	max = u_val;
    	index = i;
    }
  }
  *out_index = index;
  *out_max = max;
}
```

```cpp
void ref_dpupostproc(signed char *inp_data, unsigned char *out_max,
     unsigned char *out_index, float post_scale_factor, unsigned short int height, unsigned short int width)
{
  unsigned short int rows = height;
  unsigned short int cols = width;
  unsigned short int size = MAX_NUM_OF_CLASSES;

  float softmax[MAX_NUM_OF_CLASSES];
  signed char ch_vect[MAX_NUM_OF_CLASSES];
  unsigned char index, max;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      for(int cl=0; cl<size; cl++) {
    	  signed char  tmp_data  = inp_data[r*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES + c*MAX_NUM_OF_CLASSES + cl];
    	  ch_vect[cl] =  tmp_data;
      }
      ref_SoftMax(ch_vect, softmax, post_scale_factor, size);
      ref_ArgMax(softmax, &max, &index, size);
      out_max[  r*POST_MAX_WIDTH + c] = (unsigned char) max;
      out_index[r*POST_MAX_WIDTH + c] = index;
    }
  }
}
```

As already done for the pre-processing, also in this case there is the need to scale the data generated by the DPU before inputting them into the SoftMax classifier and this is done with the ``post_scale_factor`` parameter that comes from a query to the DPU at run time, with Python code similar to this:

```python
output_fixpos = outputTensors[0].get_attr("fix_point")
post_scale_fact = 1 / (2**output_fixpos)
```

Note that ``output_fixpos`` is value from 1 to 7 because it represents the exponent ``i`` of a power of ``2,`` that is ``2^i``.    

The SoftMax function is computed by a Look Up Table (LUT), since there are 7 possible ``output_fixpos`` values the file [luts.h](files/postproc/common_src/luts.h) contains basically 7 different LUTs, one for each value.

In the HLS TB this parameter is fixed in the [dpupostproc_defines.h](files/postproc/vitis/kernels/dpupostproc_defines.h) file, to test the functionality of the core.

The input data used in the HLS self-checking TB were taken by running the CNN xmodel generated in the [VAI-KERAS-FCN8HDTV-SEMSEG](https://gitenterprise.xilinx.com/danieleb/VAI-KERAS-FCN8HDTV-SEMSEG) tutorial directly on the VCK190 board at run time, they were saved as ``npy`` (python numpy) files, then converted in ``mat`` (MATLAB) files and finally in ``.txt`` text files.


Note: the ARM CPU could compute all the Look Up Table and send it to the post-processor as an alternative architectural choice to save BRAMs


### 2.2.2 HLS Design



Just launch the command
```shell
vitis_hls -f hls_script.tcl
```
and the whole HLS flow will run in its steps: CSIM, SYN, coSIM and IMP. See the related screenshots of Figures 5, 6, 7 and 8.


![figure5](files/postproc/img/post_csim.png)

*Figure 5. Post-processing CSIM step with Vitis HLS*


![figure6](files/postproc/img/post_syn.png)

*Figure 6. Post-processing SYN step with Vitis HLS*

![figure7](files/postproc/img/post_cosim.png)

*Figure 7. Post-processing coSIM step with Vitis HLS*

![figure8](files/postproc/img/post_impl.png)

*Figure 8. Post-processing IMP step with Vitis HLS*


Note that the file [dpupostproc_defines.h](files/postproc/hls/src/dpupostproc_defines.h) must have the line ``#define ARM_HOST`` commented.

As you see from Figure 8, after Place-And-Route, the accelerator consumes the following resources: 14347 LUT, 17395 FF, 38 BRAM and 58 DSP from the  Versal 1902 device  with a minimum clock period of 2.891ns, which corresponds to  345MHz maximum clock frequency.

Figure 7 reports the cycle accurate simulation (coSIM step), considering the amount of clock cycles to process the whole image before sending it back to DDR memory, the latency of this kernel is  given by 1722479 (cycles) x 2.981ns (clock period) = 5.134ms.

Even assuming a longer clock period of 5ns (corresponding to 200MHz clock frequency) the latency would become 8.61ms.

Note that this latency is the time to process the entire frame (860x416x28) of data because this is the way Vitis HLS works if you want to do a functional cycle accurate simulation (acknowledged as "coSIM") of the accelerator. But in itself this core has a real latency of few dozens of clock cycles. Such effective latency could be exploited either by using AXI4 Streaming interfaces (which are not accepted by the DPU core, which is unable to work in a streaming mode) instead of full MAXI4 interfaces or by adding a ping-pong buffer of few image lines among the Post-processing accelerator and the external DDR memory.  



# 3 Vitis GUI-based Design Flow

This section explains how to build the embedded system project with the Vitis GUI, now that you have developed the two accelerator kernels as standalone HLS projects. You must have available the following ``platform`` and ``petalinux`` folders/files related to the ``XVDPU TRD`` platform design:

```text
# TRD platform file
VDPU_TRD/platform/vck190_dpu_pcie/vck190_dpu_pcie.xpfm
# Sysroot path
VDPU_TRD/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux/
# Root FS file
VDPU_TRD/petalinux/xilinx-vck190-base-trd/images/linux/rootfs.ext4
# Linux Kernel Image file
VDPU_TRD/petalinux/xilinx-vck190-base-trd/images/linux/Image
```

Since the DPU core is not yet in this design, the two PL accelerators work with pre-defined scaling factors. In the real application the information about such scaling factors should arrive by searching for the  ``fix_point`` attributes of the input and output tensors of the CNN subgraph running in the DPU.


## 3.1 Pre- and Post-processing Kernels Together

This section contains the instructions to create an embedded system design in which both the Pre- and Post-processing kernels are working in parallel (of course on different data).

This step was done after having created an embedded system with only one kernel at a time and then functionally tested such standalone kernel.
Then the host code was written in a way to encapsulate the code related to each kernel so that they could work in parallel without any interference.  
If you look at the [host_preproc_xrt.cpp](files/preproc/vitis/host/host_preproc_xrt.cpp) and
[host_postproc_xrt.cpp](files/postproc/vitis/host/host_postproc_xrt.cpp) files you will note that the ``main()`` routine is embedded by ``#ifndef TWO_KERNELS``.

Since the instructions to create the standalone projects of "one-kernel-at-a-time" are basically the same of "two-kernels-in-parallel", I prefer to describe here the steps of the more general case with the two kernels in parallel,  for the sake of conciseness.

1. From Vitis GUI create a new **application project** and select the ``vck190_dpu_pcie.xpfm`` file associated to the ``XVDPU-PCIE TRD`` **platform** design, as illustrated in Figure 9.

2. Select the ARM Cortex A72 as **application domain** and fill the appropriate **Sysroot path** , **Root FS** and Linux **Kernel Image** tags, with the above mentioned files, see also Figure 10.


3. Select the "AI Engine System Design Examples -> Empty Application" as design template, see Figure 11. Then set the **Active build configuration** to **Hardware**.


4. Delete the subproject "two_kernels_kernels" as illustrated in Figure 12.

5. With the mouse, click on "File" menu and select "New->Hw Kernel project" and give it a name as "postproc", this will be the subsystem of the post-processing accelerator (or kernel). Make sure to have selected the "two_kernels" as "system project name". See Figure 13.

6. Now import the following three files of source code for such accelerator:  [dpupostproc_vhls.cpp](files/postproc/vitis/kernels/dpupostproc_vhls.cpp), [dpupostproc_defines.h](files/postproc/vitis/kernels/dpupostproc_defines.h) (this one with  ``#define ARM_HOST`` commented) and
[lut_exp.h](files/postproc/vitis/kernels/lut_exp.h). See Figure 14.

7. With the mouse click on the file ``postproc.prj`` and select the top level function ``hls_dpupostproc_m_axi``, which is the name of the accelerator in this Vitis flow.  With the mouse right click on the "postproc" kernel in the project Explorer (on the left) and select "Build". Make sure that you have first put the **Active build configuration** to **Hardware**. See Figure 15.

8. Similarly to what done in the steps 5 and 6, you now create the Pre-processing kernel. Again, with the mouse, click on "File" menu and select "New->Hw Kernel project" and give it a name as "preproc", this will be the subsystem of the pre-processing accelerator. Now add the source code files [dpupreproc_vhls.cpp](files/preproc/vitis/kernels/dpupreproc_vhls.cpp) and [dpupreproc_defines.h](files/preproc/vitis/kernels/dpupreproc_defines.h) (this last one with  ``#define ARM_HOST`` commented).

9. Similarly to what done in the step 7, with the mouse click on the file ``preproc.prj`` and select the top level function ``hls_dpupreproc_m_axi``, which is the name of the accelerator in this Vitis flow. With the mouse right click on the "preproc" kernel in the project Explorer (on the left) and select "Build". Make sure that you have first put the **Active build configuration** to **Hardware**. See Figure 16.

11. Now you have to import all the necessary files for the host application from [preproc/vitis/host](files/preproc/vitis/host) and [postproc/vitis/host](files/postproc/vitis/host). At the end of the process you will have what illustrated in Figure 17. You need also to add the file [host_main.cpp](files/two_kernels/host_main.cpp).

12. Now you have to set the **C/C++ Build Settings** for the host application. With the mouse right-click on the "two_kernels[linux on psv_cortexa72]" in the project Explorer and select "C/C++ Build -> Settings -> Dialect" and choose **ISO C++1y**. See Figure 18.

13. Still in the **C/C++ Build Settings**, you have to remove the OpenCL library and  add the XRT ``xrt_coreutil`` library. See Figure 19.

14. Now right click with the mouse on the "two_kernel_system[vck190_dpu_pcie]" and launch the "Build" action. You have to wait now for several minutes, depending on your host PC. The ``sd_card`` to boot the Linux OS on the VCK190 PP board together with the ``binary_container_1.xclbin`` bitstream to program the device will be created at the end of this process.

15. Prepare a new SD card to boot the VCK190PP by writing the file ``sd_card.img`` with an utility like ``Win32DiskImager`` (on my Windows10 OS laptop). See Figure 20.

**NOTE**
Most of the the above actions could be skipped by opening the Vitis GUI and importing the vitis archive [two_kernels_system.ide.zip](files/two_kernels/two_kernels_system.ide.zip).
The only problem is that you have to manually adapt the *TRD platform file*, the *Sysroot path*, the *Root FS file* and the *Linux Kernel Image file* to have it correctly working.



![figure9](files/two_kernels/img/two_k_Screenshot_from_2021-07-29_18-27-15.png)

*Figure 9. Vitis GUI flow: selecting the platform file*

![figure10](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-03-42.png)

*Figure 10. Vitis GUI flow: setting the domain specific tags*


![figure11](files/two_kernels/img/two_k_Screenshot_from_2021-07-29_18-30-01.png)

*Figure 11. Vitis GUI flow: design template selection*


![figure12](files/two_kernels/img/two_k_Screenshot_from_2021-07-29_18-31-56.png)

*Figure 12. Vitis GUI flow: remove the "two_kernels_kernels" subproject*


![figure13](files/two_kernels/img/two_k_Screenshot_from_2021-07-29_18-32-59.png)

*Figure 13. Vitis GUI flow: add the "postproc" kernel subproject*

![figure14](files/two_kernels/img/two_k_Screenshot_from_2021-07-29_18-34-09.png)

*Figure 14. Vitis GUI flow: add the source code files to the "postproc" kernel*


![figure15](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-12-58.png)

*Figure 15. Vitis GUI flow: build the "postproc" kernel*


![figure16](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-16-01.png)

*Figure 16. Vitis GUI flow: build the "preproc" kernel*


![figure17](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-29-53.png)

*Figure 17. Vitis GUI flow: import the source code for the host application*

![figure18](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-30-14.png)

*Figure 18. Vitis GUI flow: C/C++ dialect set to compile the host application*


![figure19A](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-30-38.png)

![figure19B](files/two_kernels/img/two_k_Screenshot_from_2021-07-30_17-31-10.png)

*Figure 19. Vitis GUI flow: adjust the libraries to compile the host application*

![figure20](files/two_kernels/img/Win32DiskImager.png)

*Figure 20. Write the sd card to boot the VCK190PP target board*


## 3.2 System Debug of the Kernels Together


1. Now turn on and boot your VCK190PP target board, open a **PuTTY** terminal from your host PC to communicate in UART directly with you target board. As illustrated in the right part of Figure 21, set the board IP Address (for example 190.1268.1.200) with the following command running it on the PuTTY terminal:
```shell
ifconfig eth0 192.168.1.200
```

2. To test the two PL kernel at runtime on the target board, you have to transfer their input data. Use a file-transfer (``scp`` based) utility like FileZilla and copy the data folders [data_post](files/postproc/hls/data_post) and [data_pre](files/preproc/hls/data_pre) from your host PC to the folder ``/mnt/    `` of the target board. See the left part of Figure 21.  

3. With the mouse right click on the project Explorer "two kernels system[vck190-dpu-pcie]" and select "Debug Configurations" and the click twice on "System Project Debug", as shown in Figure 22.

4. Then you need to set the "Linux Agent" for the debug server using the same IP address of item 1 above, as illustrated in Figure 23.

5. Run the debugger. You should see the positive results reported in Figure 24 and also stored in the
[vck190pp_logfile.txt](files/two_kernels/vck190pp_logfile.txt) file.


![figure21](files/two_kernels/img/two_k_filezilla_Screenshot_from_2021-08-02_09-46-00.png)

*Figure 21. Debug flow: file transfer of accelerator I/O data between host PC and target board*

![figure22](files/two_kernels/img/two_k_dbg1_Screenshot_from_2021-08-02_09-47-46.png)

*Figure 22. Debug flow: setting the Debug Configurations*

![figure23](files/two_kernels/img/two_k_dbg2_Screenshot_from_2021-08-02_09-49-23.png)

*Figure 23. Debug flow: set the Linux agent between host and target*

![figure24](files/two_kernels/img/two_k_dbg3_Screenshot_from_2021-08-02_09-58-20.png)

*Figure 24. Debug flow: test ended successfully*


<hr/>
<p align="center"><sup>Copyright&copy; 2021 Xilinx</sup></p>
