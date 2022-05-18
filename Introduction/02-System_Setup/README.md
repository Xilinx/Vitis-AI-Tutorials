# Section 2: System Setup

In this tutorial, we use Vitis 2020.2 and Vitis AI v1.3 for demonstration. The tutorial will be updated when Vitis tools have updates.

| Type                    | Version                       |
| ----------------------- | ----------------------------- |
| Vitis Software Platform | 2020.2                        |
| Vitis-AI                | v1.3                          |
| PetaLinux               | 2020.2                        |

he host OS can be a natively installed OS or a virtual machine. Please check [Table 1 in Vitis Release Notes][1] for supported host operating systems. It needs to have at least a 100 GB hard disk to install all the tools, models, and datasets. 16GB DDR memory is preferred for development flow.

[1]: https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/acceleration_installation.html#ariaid-title2


## Vitis Software Platform Setup

<!--Vitis Logo on the right-->
<img src = "./images/xilinx-vitis.png" align = "right" >

### Vitis Core Tools

1. Download the Vitis Software Platform from [Xilinx Download Center](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html.

2. Install the Vitis Software Platform

   - Run `./xsetup` in extracted Vitis installer tar.
   - Ensure **Devices -> Install devices for Alveo and Xilinx Edge acceleration platforms** is selected during installation.
   - Refer to [detailed Vitis User Guide](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/acceleration_installation.html#dhg1543555360045__ae364401))

3. Install Vitis dependent packages

   ```bash
   sudo <install_dir>/Vitis/<release>/scripts/installLibs.sh
   ```

   The **installLibs.sh** script from Vitis installation directory can detect the current OS release and install required dependencies. Please check the script for details about what will be installed.

### PetaLinux

**Note:** PetaLinux is only needed for platform development in customization flow.

1. Download PetaLinux from [Xilinx Download Center](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools.html)

2. Install PetaLinux dependencies according to [UG1144](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2020_2/ug1144-petalinux-tools-reference-guide.pdf) Table 2.

   - Ubuntu

      ```
      sudo apt install iproute gcc g++ net-tools libncurses5-dev zlib1g:i386 libssl-dev flex bison libselinux1 xterm autoconf libtool texinfo zlib1g-dev gcc-multilib build-essential screen pax gawk python
      ```

   - CentOS

      ```
      sudo yum install iproute gcc gcc-++ net-tools libncurses5-devel zlib-devel openssl-devel flex bison libselinux xterm autoconf libtool texinfo SDL-devel glibc-devel glibc glib2-devel automake screen pax libstdc++ gawk python
      ```

3. Install PetaLinux

   ```
   ./petalinux-v2020.2-final-installer.run
   ```

4. Setup PetaLinux requirements

   - Ubuntu: change shell from dash to bash

      ```
      sudo dpkg-reconfigure dash
      ```




### Load Vitis, XRT, and PetaLinux environment

1. Launch a bash terminal window and run the following commands to initialize Vitis and PetaLinux environment.

   ```bash
   # Load Vitis environment
   source /tools/Xilinx/Vitis/2020.2/settings64.sh
   # Load PetaLinux environment
   source /opt/petalinux-v2020.2/settings.sh
   ```



---

## Vitis AI Setup

<img src = "./images/vitis-ai.png" align = "right" >

The Vitis™ AI development environment is Xilinx’s development platform for AI inference on Xilinx hardware platforms, including both edge devices and Alveo&trade; cards. It consists of optimized IP, tools, libraries, models, and example designs. It is designed with high efficiency and ease of use in mind, unleashing the full potential of AI acceleration on Xilinx FPGA and ACAP.

Please visit <https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html> for more information about Vitis AI.

Vitis AI is released as [GitHub repository](https://github.com/Xilinx/Vitis-AI) and Docker.

Follow the instructions below to setup the Vitis AI environment step by step.

### Host Requirements

Use X86 processor based Ubuntu or CentOS as the host operating system. If you have a nVidia graphics card with CUDA support, it helps to reduce the running time.

Git package is needed at the host machine. For Ubuntu, use the following command to install:
~~~
sudo apt update
sudo apt install git
~~~

For CentOS, use following the command to install:
~~~
sudo yum install git
~~~

Docker environment is needed at host side. For docker installation without GPU support, please refer to following links:
* Ubuntu: <https://docs.docker.com/engine/install/ubuntu>
* CentOS: <https://docs.docker.com/engine/install/centos>

For docker installation with nVidia GPU support,  refer to following links:
* Ubuntu and CentOS: <https://github.com/NVIDIA/nvidia-docker>

### Get Vitis AI Git Repository

Go to your working directory, say ~/workspace, then use following command to get the Vitis AI repository.
~~~
git clone https://github.com/Xilinx/Vitis-AI.git
~~~
After that, you get to the Vitis AI directory *~/workspace/Vitis-AI*.

### Prepare Docker

Users can choose from two dockers: CPU docker is for users without an nVidia graphics card, and GPU docker is for those with one. The CPU docker can be pulled [Docker Hub](https://hub.docker.com/r/xilinx/vitis-ai/tags), or you can build it with the Dockerfile. For GPU docker, build it yourself with the Dockerfile.

#### CPU Docker
To get the CPU docker from Docker Hub, use the following command:
~~~
docker pull xilinx/vitis-ai:latest
~~~

To build it by yourself, use following the command:
~~~
cd ~/workspace/Vitis-AI/setup/docker
./docker_build_cpu.sh
~~~

Using any option, you finally get the docker image *vitis-ai:latest* settled on your host.

#### GPU Docker
Build the docker using following command:
~~~
cd ~/workspace/Vitis-AI/setup/docker
./docker_build_gpu.sh
~~~

After that, you could get the docker image *vitis-ai-gpu:latest* on your host.

Copyright&copy; 2020-2022 Xilinx
