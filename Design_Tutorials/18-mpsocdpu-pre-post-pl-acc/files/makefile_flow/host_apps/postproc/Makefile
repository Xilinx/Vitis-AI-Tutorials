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
# this file was created by a computer and modified by a human being
#

# compiler tools

HOST_CXX ?= aarch64-linux-gnu-g++
RM = rm -f
RMDIR = rm -rf

#Absolute path of this makefile
ABS_PATH                := $(shell pwd -P)

#Platform to build for
#PLATFORM                 = $(ABS_PATH)/../../../dpu_trd/xilinx_zcu102_base_202110_1/xilinx_zcu102_base_202110_1.xpfm
PLATFORM=$(VITIS_PLATFORM_PATH)

# commented as already set up outside this makefile
#VITIS_PLATFORM = xilinx_zcu102_base_202110_1
#VITIS_PLATFORM_DIR = $(ABS_PATH)/../../downloads/${VITIS_PLATFORM}
#VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/${VITIS_PLATFORM}.xpfm
#VITIS_SYSROOTS = $(ABS_PATH)/../../../dpu_trd/xilinx-zynqmp-common-v2021.1/sdk/sysroots/cortexa72-cortexa53-xilinx-linux


# host compiler global settings
CXXFLAGS += -std=c++1y -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I$(XILINX_VIVADO)/include/ -I$(VITIS_SYSROOTS)/usr/include/xrt/ -O0 -g -Wall -c -fmessage-length=0 --sysroot=$(VITIS_SYSROOTS)
LDFLAGS += -lxrt_coreutil -lrt -ldl -lcrypt -lstdc++ -lpthread -L$(VITIS_SYSROOTS)/usr/lib/ --sysroot=$(VITIS_SYSROOTS)


#
# host files
#
HOST_OBJECTS += src/dpupostproc_ref.o
HOST_OBJECTS += src/dpupostproc_tb.o
HOST_OBJECTS += src/host_postproc_xrt.o

HOST_EXE = host_postproc_xrt

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

src/dpupostproc_ref.o: ./src/dpupostproc_ref.cpp ./src/dpupostproc_defines.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/dpupostproc_tb.o: ./src/dpupostproc_tb.cpp ./src/dpupostproc_tb.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/host_postproc_xrt.o: ./src/host_postproc_xrt.cpp ./src/dpupostproc_tb.h $(VITIS_SYSROOTS)/usr/include/xrt/experimental/xrt_kernel.h $(VITIS_SYSROOTS)/usr/include/xrt/experimental/xrt_profile.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)
