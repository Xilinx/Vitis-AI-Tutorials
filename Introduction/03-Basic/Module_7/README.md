# 3.7 Acceleration with HLS kernel
---

Version: Vitis&trade; 2020.2 and Vitis AI&trade; 1.3

**Note**: You need to use the new image you generated in Module 7 of Section 3. After the OS boot is successful, you can refer to [quick_start](https://github.com/Xilinx/Vitis-AI/tree/master/demo/Vitis-AI-Library#quick-start-for-edge) guide to learn how to prepare the development environment.
This example suite for the Vitis AI Library and Vitis Accelerated Kernel shows how the Vitis AI Library runs neural networks on DPUs and how to use the HLS kernel to speed up Pre and Postprocessing. To learn how to immigrate from OpenCV to HLS, refer to [app/README.md](app/README.md)

Some system-level functions:

- Working with DRM on Zynq&reg; MPSoC

  - Using the DRM to display

- V4L2 initialization and control
  - Streaming video capture from V4L2 such as USB camera.

The directory structure looks as follows:

```bash
├── app
│   ├── include                    #headers directory of work pipeline
│   │   ├── decodethread.hpp       #header file of decode thread
│   │   ├── dputhread.hpp          #header file of dpu threadd
│   │   ├── filter.hpp             #dpu filer header
│   │   ├── frameinfo.hpp          #important data structure
│   │   ├── guithread.hpp          #display thread headerfile
│   │   ├── mythread.hpp           #thread pool
│   │   └── sortthread.hpp         #sort thread header
│   ├── README.md                  #instruction of how to migrate from OpenCV to HLS
│   ├── src                        #implementation of the work threads
│   │   ├── decodethread.cpp
│   │   ├── dputhread.cpp
│   │   ├── guithread.cpp
│   │   ├── mythread.cpp
│   │   └── sortthread.cpp
│   └── test
│       └── usb_input_multi_threads_xcl_refinedet.cpp #main application
├── kernel                            #HLS kernel work directory
│   ├── build                         #directory of config file
│   │   ├── dpu_conf.vh               #dpu IP config file
│   │   ├── Makefile                  #The Makefile used to replace the file in ${DPU_TRD}
│   │   ├── preprocessor_config.ini   #config file of pre_processor
│   │   └── prj_config_104_2dpu       #Config file required be Makefile for IP integration
│   ├── flash_sd_card.sh              #Script to generate the OS image
│   └── src                           #HLS kernel source codes directory
│       ├── pre_processor.cpp   
│       └── pre_processor.h
├── my_V4l2s                          # V4l2 libraries work directory to get image data from USB camera
│   ├── CMakeLists.txt                # cmake file to build libs
│   ├── include                       
│   │   ├── dpdrm.hpp                 #Encapsulate drm library
│   │   ├── line_exception.hpp        #Encapsulate throw error
│   │   ├── V4l2Access.hpp            #Interface to access the camera device
│   │   ├── V4l2Capture.hpp           #Methods to get the image
│   │   ├── V4l2Device.hpp            #Device capablity
│   │   ├── V4l2MmapDevice.hpp        #Device type
│   │   ├── V4l2Output.hpp            #asistant class
│   │   ├── V4l2ReadWriteDevice.hpp   #Device type
│   │   └── xcl2.hpp                  #Encapsulate the kernel usage
│   ├── src
│   │   ├── V4l2Access.cpp
│   │   ├── V4l2Capture.cpp
│   │   ├── V4l2Device.cpp
│   │   ├── V4l2MmapDevice.cpp
│   │   ├── V4l2Output.cpp
│   │   ├── V4l2ReadWriteDevice.cpp
│   │   └── xcl2.cpp
│   └── test
│       ├── build.sh                #compile script
│       ├── test_hls_kernel.cpp     #test case for hls kernel
│       └── test.yuv                #test data
└── README.md
└── CMakeLists.txt
└── build_app_arm.sh                # Build application with hls kernel
└── build_app_hls.sh                # Build application with arm preprocessing

```

## Pre-Requisites
- Hardware required:
  - ZCU104 evaluation board
  - Micro-USB cable to connect to the laptop for the terminal emulator
  - SD card to burn the OS image.
- Software required:
- Vitis 2020.2 [Vitis Core Development Kit](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2020-2.html) installed in the server.
- [Silicon Labs quad CP210x USB-to-UART bridge driver](https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers) installed in the laptop.
- Serial terminal emulator. Examples: Tera Term, MobaXterm installed in the laptop
- [XRT 2020.2](https://github.com/Xilinx/XRT/tree/2020.2) installed in the laptop
- [ZCU104 base platform](https://github.com/Xilinx/Vitis_Embedded_Platform_Source/tree/master/Xilinx_Official_Platforms/xilinx_zcu104_base) installed in the server
- [Vitis AI runtime package](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.3.0.tar.gz) based on VAI1.3
- [dpu_sw_optimize.tar.gz](https://github.com/Xilinx/Vitis-AI/blob/master/dsa/DPU-TRD/app/dpu_sw_optimize.tar.gz) Running `zynqmp_dpu_optimize.sh` to optimize the board settings
---

## Workflow Overview
- Step 1: Create base platform
- Step 2: DPU and HLS kernel integration
- Step 3: Set up cross-compiler environment and build application
- Step 4: Board development setting
- Step 5: Run the application

## Step 1: Create Base Platform
Build  platform from scratch. For more information, see [Vitis_Platform_Creation](../../../../Vitis_Platform_Creation/README.md).

The other way is to use the platform code of [Xilinx_Official_Platform](https://github.com/Xilinx/Vitis_Embedded_Platform_Source), in which the platform build process is entirely scripted. **Note:** This platform build process involves cross-compiling Linux. The building of the platform is supported on Linux environments only (although it is possible to build inside a VM or Docker container).

Also, note that the default PetaLinux configuration uses local scratchpad areas, which does not work if you are building on a networked file system. Yocto will throw an error. Update PetaLinux to change the build area to a locally-mounted hard drive (most Xilinx internal network servers have a /scratch or /tmp area for this purpose).

After cloning the platform source, and with both Vivado&trade; and PetaLinux set up, run `make` from the top-level platform directory. In this case, we choose ZCU104_Base platform.

### Step 2:  DPU and HLS Kernel Integration

1. Clone the Vitis AI repository and vitis_library. The preprocessor kernel implements based on vitis_library. Before building the hardware design, you need to clone the vitis_library in the `${DPU-TRD}/prj/Vitis` directory.
  ``
$ git clone https://github.com/Xilinx/Vitis-AI.git
$ cd Vitis-AI/dsa/DPU-TRD/prj/Vitis/
$ git clone https://github.com/Xilinx/Vitis_Libraries.git``

2. Download the [!mpsoc common system](https://  www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zynqmp-common-v2020.2.tar.gz), and unzip the files.

  ```
$ tar -xvzf  xilinx-zynqmp-common-v2020.2.tar.gz
$ cd xilinx-zynqmp-common-v2020.2
$ gzip -d rootfs.ext4.gz
```

3. Copy the kernel source code and the config files to the DPU-TRD work directory

  ```
$ cp ${Module_7}/kernel/build/Makefile Vitis-AI/dsa/DPU-TRD/prj/Vitis
$ cp ${Module_7}/kernel/build/preprocessor_config.ini Vitis-AI/dsa/DPU-TRD/prj/Vitis
$ cp ${Module_7}/kernel/build/dpu_conf.vh Vitis-AI/dsa/DPU-TRD/prj/Vitis
$ cp ${Module_7}/kernel/src/pre_processor.cpp Vitis-AI/dsa/DPU-TRD/prj/Vitis
$ cp ${Module_7}/kernel/src/pre_processor.h Vitis-AI/dsa/DPU-TRD/prj/Vitis
$ cp ${Module_7}/kernel/config_file/prj_config_104_2dpu Vitis-AI/dsa/DPU-TRD/prj/Vitis/config_file
```

4. Run the following commands to start IP integration.

  ```
export SDX_PLATFORM=<path-to-platform>
export EDGE_COMMON_SW=<path-to-edge_common_sw>
make KERNEL=DPU DEVICE=ZCU104
```
5. Follow one of these methods to set up the target environment
  - Replace the BOOT.BIN and the dpu.xclbin on target.

  ```
$ scp Vitis-AI/dsa/DPU-TRD/prj/Vitis/binary_container_1/BOOT.BIN root@<zcu104 board ip>:/mnt/sd-mmcblk0p1/
$ scp Vitis-AI/dsa/DPU-TRD/prj/Vitis/binary_container_1/dpu.xclbin root@<zcu104 board ip>:/mnt/sd-mmcblk0p1/
```
  - Burn the SD card with the image generated in `${Vitis-AI}/dsa/DPU-TRD/prj/Vitis/binary_container_1/sd_card.img`

## Step 3: Cross Compiler Environment Set up and Build application
---
- Install the SDK package and the Vitis AI cross compile package

  ```
$sh sdk-2020.2.0.0.sh
$tar -xvzf vitis_ai_2020.2-r1.3.0.tar.gz -C <path-to-sysroot>/
```
- After installing the SDK package, use the script to set up the cross-compile environment by using the following commands:
```
$ source <full_path_of_sdk>/environment-setup-aarch64-xilinx-linux
$ cd ${Section_3-Basic}/Module_7/
$ sh build_app_hls.sh
```
- Copy the test YUV file to target.

  ```
$ scp test.yuv root@<zcu104 ip>:/home/root
```
- Copy the library and the applications to the target board
```
$ scp ${Module_7}/install/libhls_v4l2s.so root@Board_IP:/usr/lib
$ scp ${Module_7}/test/test_hls_kernel  root@Board_IP:/home/root
$ scp ${Module_7}/test/usb_input_multi_threads_refinedet_hls_drm  root@Board_IP:/home/root



### Step 4: Run the Application on Target

- Test the HLS kernel to see if it works properly with the test example. When the test example works, you notice that two PNG files are generated in your execution directory.
```
# chmod +x test_hls_kernel
# ./test_hls_kernel
```
- In this case we use RefineDet model to show the body detection.
```
  ./usb_input_multi_threads_refinedet_hls_drm refinedet_pruned_0_8 0 -t 3
```
- DESCRIPTION
```
    - refinedet_pruned_0_8: The model used;
    - 0                   :  Camera input;
    - -t 3                :  thread number 3;
```

- Optional

  If you are interested in the gap between HLS kernel's and ARM's processing performance, recompile the program to see the performance

- Build the application

  ```
  $ build_app_arm.sh

  ```

  - Copy the library and the applications to the target board
  ```  
  $ scp ${Module_7}/install/libhls_v4l2s.so root@Board_IP:/usr/lib
  $ scp ${Module_7}/test/test_hls_kernel  root@Board_IP:/home/root
  $ scp ${Module_7}/test/usb_input_multi_threads_refinedet_hls_drm  root@Board_IP:/home/root
  ```

Copyright&copy; 2020-2022 Xilinx
