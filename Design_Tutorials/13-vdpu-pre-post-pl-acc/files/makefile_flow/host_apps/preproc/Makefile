#
# Copyright 2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# this file was created by a computer. trust it.
#

# compiler tools


HOST_CXX ?= aarch64-linux-gnu-g++
RM = rm -f
RMDIR = rm -rf

#Absolute path of this makefile
ABS_PATH                := $(shell pwd -P)

#$(eval FATHER_PATH      := $(realpath $(ABS_PATH)/../..))
#DB: Note that this FATHER_PATH should be adapted to the place where you have the petaliux stuff
#FATHER_PATH ?=/media/danieleb/DATA/ZF/ZF_ProAI-main/NEW_ZF_PACKAGE_FINAL/
FATHER_PATH=$(DB_FATHER_PATH)


#Platform to build for
PLATFORM                 = $(FATHER_PATH)/platform/vck190_dpu_pcie/vck190_dpu_pcie.xpfm

VITIS_PLATFORM = vck190_dpu_pcie
VITIS_PLATFORM_DIR = $(FATHER_PATH)/platform/vck190_dpu_pcie/platform/zf_dpupcie_pf
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/vck190_dpu_pcie.xpfm

# host compiler global settings
CXXFLAGS += -std=c++1y -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I$(XILINX_VIVADO)/include/ -I$(FATHER_PATH)/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux/usr/include/xrt/ -O0 -g -Wall -c -fmessage-length=0 --sysroot=$(FATHER_PATH)/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux
LDFLAGS += -lxrt_coreutil -lrt -ldl -lcrypt -lstdc++ -lpthread -L$(FATHER_PATH)/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux/usr/lib/ --sysroot=$(FATHER_PATH)/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux

#
# host files
#

HOST_OBJECTS += src/ap_bmp.o
HOST_OBJECTS += src/dpupreproc_ref.o
HOST_OBJECTS += src/dpupreproc_tb.o
HOST_OBJECTS += src/host_preproc_xrt.o

HOST_EXE = host_preproc_xrt

BUILD_SUBDIRS += src/

#
# primary build targets
#

.PHONY: all clean
all:  $(HOST_EXE)

clean:
	-$(RM) $(HOST_EXE) $(HOST_OBJECTS)

.PHONY: incremental
incremental: all


nothing:

#
# host rules
#

src/ap_bmp.o: ./src/ap_bmp.cpp ./src/ap_bmp.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/dpupreproc_ref.o: ./src/dpupreproc_ref.cpp ./src/dpupreproc_defines.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/dpupreproc_tb.o: ./src/dpupreproc_tb.cpp ./src/dpupreproc_tb.h ./src/ap_bmp.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/host_preproc_xrt.o: ./src/host_preproc_xrt.cpp ./src/dpupreproc_tb.h $(FATHER_PATH)/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux/usr/include/xrt/experimental/xrt_kernel.h $(FATHER_PATH)/petalinux/xilinx-vck190-base-trd/images/linux/sdk/sysroots/aarch64-xilinx-linux/usr/include/xrt/experimental/xrt_profile.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)
