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
# this file was created by a computer (from Vitis GUI) but paths were adapted by a human being.
#

# compiler tools


HOST_CXX ?= aarch64-linux-gnu-g++
RM = rm -f
RMDIR = rm -rf

#Absolute path of this makefile
ABS_PATH                := $(shell pwd -P)


#Platform to build for
PLATFORM=$(VITIS_PLATFORM_PATH)
#PLATFORM  = $(ABS_PATH)/../../../dpu_trd/xilinx_zcu102_base_202110_1/xilinx_zcu102_base_202110_1.xpfm
#variables already set out of this Makefile
#VITIS_PLATFORM = xilinx_zcu102_base_202110_1
#VITIS_PLATFORM_DIR = $(ABS_PATH)/../../../dpu_trd/xilinx_zcu102_base_202110_1
#VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/xilinx_zcu102_base_202110_1.xpfm
#VITIS_SYSROOTS = $(ABS_PATH)/../../../dpu_trd/xilinx-zynqmp-common-v2021.1/sdk/sysroots/cortexa72-cortexa53-xilinx-linux



# host compiler global settings
CXXFLAGS += -std=c++17 -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I$(XILINX_VIVADO)/include/ \
-I$(VITIS_SYSROOTS)/usr/include/xrt/ \
-I$(VITIS_SYSROOTS)/usr/include/opencv4/ \
-I${TRD_HOME}/app/samples/include \
-O0 -g -Wall -c -fmessage-length=0 --sysroot=$(VITIS_SYSROOTS)

LDFLAGS += -lvart-runner \
-lopencv_videoio \
-lopencv_imgcodecs \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_core \
-lglog \
-lxir \
-lunilog \
-lxrt_core \
-lxrt_coreutil \
-lpthread \
-lrt \
-ldl \
-lcrypt \
-lstdc++ \
-L$(VITIS_SYSROOTS)/usr/lib/ \
-L${TRD_HOME}/app/samples/lib/ \
--sysroot=$(VITIS_SYSROOTS)


#
# host files
#

HOST_OBJECTS += src/common.o
HOST_OBJECTS += src/hw_pre2dpu2post.o
HOST_OBJECTS += src/main_mt.o

HOST_EXE = host_pre2dpu2post_xrt

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

src/common.o: ./src/common.cpp ./src/common.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/hw_pre2dpu2post.o: ./src/hw_pre2dpu2post.cpp ./src/hw_pre2dpu2post.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/main_mt.o: ./src/main_mt.cpp ./src/common.h ./src/hw_pre2dpu2post.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)
