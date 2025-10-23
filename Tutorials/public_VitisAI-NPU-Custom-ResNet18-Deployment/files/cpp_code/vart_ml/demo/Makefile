# ===========================================================
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License
# ===========================================================

# modified on 01 Oct. 2025


ifeq ($(shell uname -m),aarch64)
	BASEDIR=$(VAISW_INSTALL_DIR)/vart_ml
	OPENCV=opencv4
	LDFLAGS=-L$(BASEDIR)/npu_c_api -L$(BASEDIR)/vart_ml_io -L$(BASEDIR)/vart_ml_utils -L$(BASEDIR)/vart_ml_runner -L$(BASEDIR)/onnx_runner
else
	OPENCV=opencv
	LDFLAGS=-L$(VAISW_INSTALL_DIR)/lib/zebra
endif

CPPFLAGS=-I.
CPPFLAGS+=-I..
CXXFILES=common.cpp
CXXFLAGS=$(shell pkg-config --cflags $(OPENCV)) $(CXXFILES) -Wall -Wextra -Werror -std=c++17 -O3 -mcpu=native -pthread
LDLIBS=$(shell pkg-config --libs $(OPENCV))
LDLIBS+=-lnpu_c_api -lvart_ml_io -lvart_ml_utils -lvart_ml_runner -lonnx_runner

TARGETS:= connect simple_demo pipelining_demo threads_demo multi_models_demo vart_ml_demo test_vcor_resnet18

all: $(TARGETS)

clean:
	@$(RM) $(TARGETS) *.o $(CLEAN)

.PHONY: clean
