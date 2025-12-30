<!--
Copyright © 2025 Advanced Micro Devices, Inc.
All rights reserved.
MIT License
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/logo-white-text.png">
    <img alt="AMD logo" src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="35%">
  </picture>
  <h1>Vitis AI Tutorial 1: YOLOX Mixed Precision</h1>
</p>


---

## 📘 Table of Contents

1. [Introduction](#1-introduction)  
   - [Vitis AI Overview](#vitis-ai-overview)  & [Mixed Precision Overview](#mixed-precision-overview)
2. [Prerequisites](#2-prerequisites)  
   - [Hardware & Software Requirements](#hardware--software-requirements)
3. [Environment Setup](#3-environment-setup)  
   - [Installing Vitis AI Tools](#31-installing-vitis-ai-tools)  
   - [Initialize Environment](#32-initialize-environment)  
   - [Host Static IP Setup](#33-host-static-ip-setup)  
   - [Setting Up Cross Compiler](#34-setting-up-cross-compiler)  
   - [Configure NPU IP](#35-configure-npu-ip)  
   - [Launch Docker](#36-launch-docker)
4. [Model Preparation](#4-model-preparation)  
   - [Selecting / Training Models](#41-selecting--training-models)  
   - [Preparing for Quantization](#42-preparing-for-quantization)
5. [Mixed Precision Quantization Workflow](#5-mixed-precision-quantization-workflow)  
   - [Precision Configuration](#51-precision-configuration)  
   - [Mixed Precision Control](#52-mixed-precision-control)  
   - [Configure Environment Variables](#53-configure-environment-variables)
6. [Snapshot Generation](#6-snapshot-generation)  
   - [To set Snapshot Directory](#61-set-snapshot-directory)  
   - [Generate Snapshot](#62-generate-snapshot)  
7. [Board / Target Setup](#7-board--target-setup)  
   - [Setup the Board Hardware](#71-setup-the-board-hardware)  
   - [Setup the SD Card](#72-setup-the-sd-card)  
   - [Setup Serial Terminal](#73-setup-serial-terminal)  
   - [Power On the Board](#74-power-on-the-board)  
   - [Login to the Board](#75-login-to-the-board)  
   - [Configure Network (Static IP)](#76-configure-network-static-ip)  
   - [Verify Connection](#761-verify-connection)
8. [Deployment](#8-deployment)  
   - [Initialize Runtime Environment](#81-initialize-vitis-ai-runtime-environment)  
   - [Copy Model Snapshot to Target](#82-copy-model-snapshot-from-host-to-target)  
   - [Run Inference](#83-run-inference-on-the-target)  
   - [Evaluate Performance](#84-evaluate-performance)
9. [References](#9-references)

---

## 1. Introduction

### Vitis AI Overview
**Vitis AI** is AMD’s unified AI inference stack for adaptive computing platforms, enabling deployment of deep learning models from edge to cloud.

Mixed precision quantization allows using different numerical precisions (like INT8 and BF16) across model layers — balancing performance, power, and accuracy.
With Vitis AI 5.1, users can dynamically control precision settings during snapshot generation, making it possible to selectively apply higher precision where it most impacts accuracy.

In this tutorial, you will:

- Learn how to apply **mixed precision quantization** to a **YOLOX** model.  
- Generate and compare results from:
  - An **INT8-only** snapshot (baseline)  
  - A **Mixed Precision** snapshot (INT8 + BF16)  
- Analyze the **accuracy** and **performance** differences between both runs to understand precision trade-offs.


---

## 2. Prerequisites

### Hardware & Software Requirements

| Component        | Version / Requirement                            |
|------------------|--------------------------------------------------|
| **Release**      | 5.1                                              |
| **Host OS**      | Ubuntu 22.04.5 LTS (Jammy Jellyfish)             |
| **Target Board** | VEK280 RevB3                                     |
| **Petalinux**    | 2025.1                                           |
| **Vivado**       | 2025.1.1                                         |
| **Vitis**        | 2025.1.1                                         |
| **Python**       | 3.12.6                                           |
| **Docker**       | 24.0.5 or later                                  |
| **FFmpeg**       | 6.0 or later                                     |
| **GStreamer**    | 1.24.4 or later                                  |

---

## 3. Environment Setup

### 3.1 Installing Vitis AI Tools

```bash
# Navigate to the directory containing the downloaded Vitis AI package
cd <path-to-downloaded-vitis-ai-5.1.tar>

# Extract the package
tar xf vitis-ai-5.1.tar
```

### 3.2 Initialize Environment

```bash
# Move into the extracted directory
cd Vitis-AI

```bash
# Make snapshot directory to store all the snapshots
mkdir snapshots

# Export the repository path as an environment variable
export VITIS_AI_REPO=$PWD
```

### 3.3 Setting up the static IP of the host

Set this host IP address as: 192.168.1.1
Note: this will be used later stage to stablis connection with the board

### 3.4 Setting Up Cross Compiler

1. Download `sdk-vai-5.1.sh` from the **Vitis AI Lounge**.  
2. Make it executable:
   ```bash
   chmod a+x sdk-vai-5.1.sh
   ```
3. Install the SDK (run **outside Docker**):
   ```bash
   ./sdk-vai-5.1.sh -d <path-to-sdk-installation>
   ```
4. After installation, source the environment:
   ```bash
   source <path-to-sdk-installation>/environment-setup-cortexa72-cortexa53-xilinx-linux
   ```
### 3.5 Configure NPU IP

```bash
# List available NPU IP configurations
source npu_ip/settings.sh LIST

# Example: Configure a specific NPU IP
source npu_ip/settings.sh VE2802_NPU_IP_O00_A304_M3
```

### 3.6 Launch Docker

```bash
# Start the Vitis AI Docker container
./docker/run.bash --acceptLicense -- /bin/bash
```

**Explanation:**
- `--acceptLicense`: Confirms acceptance of license terms.  
- The first launch may take several minutes as the Docker image is initialized.



---

## 4. Model Preparation

### 4.1 Selecting / Training Models

Use YOLOX as the baseline model for this tutorial.

```bash
cd /home/demo/YOLOX
```

### 4.2 Preparing for Quantization

```bash
# Define the number of calibration images
export VAISW_QUANTIZATION_NBIMAGES=1
```

---

## 5. Mixed Precision Quantization Workflow

### 5.1 Overview
Mixed precision combines BF16 and INT8 operations within a single model to achieve optimal speed-accuracy trade-offs.

## 5.1.1 Precision Configuration

During snapshot generation, set the desired precision using environment variables:

| **Variable** | **Description** |
|--------------|-----------------|
| `VAISW_FE_PRECISION=INT8` | Default: All layers use **INT8** precision. |
| `VAISW_FE_PRECISION=BF16` | All layers use **BF16** precision. |
| `VAISW_FE_PRECISION=MIXED` | Mix of **INT8** and **BF16** for optimized performance. |



## 5.1.2 Mixed Precision Control

These variables control how mixed precision is applied within the model:

| **Variable** | **Description** |
|--------------|-----------------|
| `VAISW_FE_MIXEDSELECTION=BEFORE_LAST_CONV` | Default: **INT8** used until the last convolution layer, then **BF16**. |
| `VAISW_FE_MIXEDSELECTION=TAIL` | **BF16** used in tail layers for accuracy preservation. |
| `VAISW_FE_MIXEDRATIO=0.05` | Specifies that **5% of GOPS** (operations) use BF16 precision. |
| `VAISW_FE_VIEWDTYPEINPUT=AUTO` | Automatically controls **input data type**. |
| `VAISW_FE_VIEWDTYPEOUTPUT=AUTO` | Automatically controls **output data type**. |

### 5.2 Configure Environment Variables

```bash
# Enable mixed precision feature extraction
export VAISW_FE_PRECISION=MIXED

```

---

## 6. Snapshot Generation

### 6.1 To set Snapshot Directory

```bash
export VAISW_SNAPSHOT_DIRECTORY=$VITIS_AI_REPO/snapshots/<name of the snapshot>
```

### 6.2 Generate Mixed Precision Snapshot

```bash
# Generate snapshot using test image
source $VITIS_AI_REPO/npu_ip/settings.sh && cd /home/demo/YOLOX && VAISW_FE_PRECISION=MIXED VAISW_SNAPSHOT_DIRECTORY=$VITIS_AI_REPO/snapshots/<snapshot_name> VAISW_QUANTIZATION_NBIMAGES=1 ./run assets/dog.jpg  m --batch 1 --save_result

```

### 6.3 Generate INT8 Snapshot

```bash
source $VITIS_AI_REPO/npu_ip/settings.sh && cd /home/demo/YOLOX && VAISW_SNAPSHOT_DIRECTORY=$VITIS_AI_REPO/snapshots/<snapshot_name> VAISW_QUANTIZATION_NBIMAGES=1 ./run assets/dog.jpg  m --batch 1 --save_result

# Note: Default computation uses quantized INT8.
```

### 6.4 Verify the generated snapshots

```bash
cd $VITIS_AI_REPO/snapshots
```
```bash
ls

# Note: You should see both the generated snapshots in this directory.
```

---

### 7 Setup the Board Hardware & SD Card

#### 7.1 **Connect the following components:**
   - **Power cable:** To power up the board.  
   - **Ethernet cable:** For host-target communication (required for `scp` and remote operations).  
   - **USB cable:** For serial console connection.

<img src="https://gitenterprise.xilinx.com/divyamy/vai-tutorial-skeleton/blob/main/images/header/VEK280%20Evaluation%20Board.png" width="75%">


#### 7.2 Flash the SD Card

Use the **Raspberry Pi Imager** (or any reliable flashing tool) to write the board image file to the SD card.

```bash
V5.1_VE2802_NPU_IP_O00_A304_M3_sd_card.img.gz
```

This file should be available at:

```bash
<Path>/sd_card/V5.1_VE2802_NPU_IP_O00_A304_M3_sd_card.img.gz
```

> ⚠️ **Note:**  
> The exact path may vary depending on your software package structure.  


#### 7.3 Insert SD Card

Once flashing is complete, safely eject the SD card and insert it into the **SD card slot** on the target board.



### 7.4 Setup Serial Terminal

To access the Linux shell on the board, connect via a serial terminal.

You can use tools such as:
- **Tabby Terminal**
- **PuTTY**
- **Minicom**
- **Tera Term**

#### Serial Configuration:

| **Parameter** | **Value** |
|----------------|------------|
| **Port** | Detected COM port or `/dev/ttyUSBx` |
| **Baud Rate** | 115200 |
| **Data Bits** | 8 |
| **Stop Bits** | 1 |
| **Parity** | None |

> 💡 *Tip:* On Linux, you can find your serial device using:
> ```bash
> dmesg | grep ttyUSB
> ```


### 7.5 Power On the Board

After connections are made and the SD card is inserted:
1. Switch on the board’s power supply.  
2. Wait for the boot process to complete.  
3. Observe the console messages on your serial terminal.



### 7.6 Login to the Board

Once the board boots, log in using the default credentials:

```bash
login: root
password: root
```
---

### 7.7 Configure Network (Static IP)

Set a static IP on the board to communicate with your host machine.

```bash
ifconfig end0 192.168.1.2
```

> This assigns the IP `192.168.1.2` to your target board’s Ethernet interface `end0`.

Ensure your **host PC** is configured with an IP in the same subnet (e.g., `192.168.1.1`).

#### 7.7.1 Verify Connection

Test network connectivity between the board and the host:

```bash
ping 192.168.1.1
```

If successful, you should see continuous ping replies — confirming network setup.

---

## 8. Deployment

After preparing your model snapshots on the host, follow these steps to deploy and run inference on the target.



### 8.1 Initialize Vitis AI Runtime Environment

On the **target board**, source the Vitis AI runtime environment setup script:

```bash
source /etc/vai.sh
```

> `/etc/vai.sh` is an initialization script for the **Vitis AI Runtime** on AMD boards.  
It:

- Sets environment variables and Python paths for Vitis AI tools.  
- Detects the **NPU IP** and locates the corresponding **xclbin** bitstream.  
- Creates missing device links (`/dev/npu_*`) for the NPU.  
- Provides backward compatibility symlinks for older Vitis AI commands.  
- Warns if only one CPU core is active (recommending a reboot).
This command configures all necessary environment variables and paths for Vitis AI tools such as `vart_ml_runner.py`.



### 8.2 Copy Model Snapshot from Host to Target

Use `scp` (Secure Copy Protocol) to transfer your compiled model snapshot directory to the target board:

```bash
scp -r <path-to-Vitis-AI>/snapshots/<snapshot_name> root@192.168.1.2:/root/
```


### 8.3 Run Inference on the Target

Once the model is copied, run inference using the **VART ML Runner**:

```bash
vart_ml_runner.py --snapshot <snapshot_name>/ --in_zero_copy --out_zero_copy
```

#### Explanation:
- `--snapshot` : Path to the snapshot directory 
- `--in_zero_copy` / `--out_zero_copy` : Enables zero-copy data movement for optimized performance.



### 8.4 Evaluate Performance

Enable detailed runtime summary logging by exporting the following environment variable:

```bash
export VAISW_RUNSESSION_SUMMARY=all
```

Re-run the inference command to see the performance of snapshot generated with Mixed Precision:

```bash
vart_ml_runner.py --snapshot <Mixed_Precision_snapshot_name>/ --in_zero_copy --out_zero_copy
```
<img src="https://gitenterprise.xilinx.com/divyamy/vai-tutorial-mixed-precision/blob/main/images/header/Mixed_Performance.png" width="75%">

Re-run the inference command to see the performance of snapshot generated with INT8:

```bash
vart_ml_runner.py --snapshot <INT8_snapshot_name> --in_zero_copy
```
<img src="https://gitenterprise.xilinx.com/divyamy/vai-tutorial-mixed-precision/blob/main/images/header/INT8_Performance.png" width="75%">

## Performance Comparison: INT8 vs Mixed Precision

This section compares inference performance between **INT8** and **Mixed Precision (INT8 + BF16)** modes on the **VEK280_REVB3** board.

### 8.5 Comparison Table

| Metric                          | INT8                              | Mixed Precision (INT8 + BF16)     |
|--------------------------------|-----------------------------------|-----------------------------------|
| **Batch Size**                 | 1                                 | 1                                 |
| **Input Tensor**               | `1x3x640x640 (INT8)`              | `1x3x640x640 (INT8)`              |
| **Output Tensor**              | `1x8400x85 (FLOAT32)`             | `1x8400x85 (BF16)`                |
| **Total Subgraphs**            | 2 (1 VART + 1 CPU)                | 1 (VART only)                     |
| **Whole Graph Median Time**    | 25.64 ms                          | 3.76 ms                           |
| **AI Acceleration Median Time**| 3.09 ms                           | 3.52 ms                           |
| **CPU Processing Time**        | 14.27 ms                          | 0.16 ms                           |
| **Output Reorder + Unquantize**| ~12.95 ms                         | N/A                               |
| **Median Throughput**          | 39.00 samples/sec                 | 265.96 samples/sec                |


### 8.6 Key Insights

- **Mixed Precision** delivers ~7× faster inference compared to INT8 mode.
- **INT8 mode** incurs significant CPU overhead due to output post-processing.
- **Mixed Precision** minimizes CPU usage by leveraging AIE acceleration for output handling.
- **Higher throughput** with Mixed Precision makes it ideal for real-time applications.

---
## 9. References

- [Vitis AI 5.1 User Guide](https://vitisai.docs.amd.com/en/latest/)
- [YOLOX Official Repository](https://docs.ultralytics.com/models/yolox/)


---

<p align="center">
  © 2025 Advanced Micro Devices, Inc. All rights reserved.
  
  ### Please Read: Important Legal Notices

The information presented in this document is for informational purposes only and may contain technical
inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and
may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes,
component and motherboard version changes, new model and/or product releases, product differences
between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any
computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD
assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the
right to revise this information and to make changes from time to time to the content hereof without
obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED "AS
IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND
ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN
THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY
PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING
FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

This document contains preliminary information and is subject to change without notice. Information
provided herein relates to products and/or services not yet available for sale, and provided solely for
information purposes and are not intended, or to be construed, as an offer for sale or an attempted
commercialization of the products and/or services referred to herein.

### AUTOMOTIVE APPLICATIONS DISCLAIMER

AUTOMOTIVE PRODUCTS (IDENTIFIED AS "XA" IN THE PART NUMBER) ARE NOT WARRANTED FOR USE IN
THE DEPLOYMENT OF AIRBAGS OR FOR USE IN APPLICATIONS THAT AFFECT CONTROL OF A VEHICLE
("SAFETY APPLICATION") UNLESS THERE IS A SAFETY CONCEPT OR REDUNDANCY FEATURE CONSISTENT
WITH THE ISO 26262 AUTOMOTIVE SAFETY STANDARD ("SAFETY DESIGN"). CUSTOMER SHALL, PRIOR TO
USING OR DISTRIBUTING ANY SYSTEMS THAT INCORPORATE PRODUCTS, THOROUGHLY TEST SUCH
SYSTEMS FOR SAFETY PURPOSES. USE OF PRODUCTS IN A SAFETY APPLICATION WITHOUT A SAFETY DESIGN
IS FULLY AT THE RISK OF CUSTOMER, SUBJECT ONLY TO APPLICABLE LAWS AND REGULATIONS GOVERNING
LIMITATIONS ON PRODUCT LIABILITY.

### Copyright

© Copyright 2025 Advanced Micro Devices, Inc. AMD, the AMD Arrow logo, Alveo, UltraScale+, Versal, Vitis,
Zynq, and combinations thereof are trademarks of Advanced Micro Devices, Inc. PCI, PCIe, and PCI Express are
trademarks of PCI-SIG and used under license. AMBA, AMBA Designer, Arm, ARM1176JZ-S, CoreSight, Cortex,
PrimeCell, Mali, and MPCore are trademarks of Arm Limited in the US and/or elsewhere. Other product names
used in this publication are for identification purposes only and may be trademarks of their respective
companies.
</p>
