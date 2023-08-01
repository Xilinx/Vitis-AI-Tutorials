# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

### Makefile for creating dpupreproc kernel
#PLATFORM_CUSTOM ?= vck190_dpu_pcie
#BASE_PFM		    ?=./platform/${PLATFORM_CUSTOM}/${PLATFORM_CUSTOM}.xpfm
#XPFM 			      :=../../${BASE_PFM}
#KERNEL			    := vadd
#GHE_REPO		    := ../../../../ghe_daniele/VDPU-PRE-POST-PL-ACC/files

.PHONY: clean all

all: compile

compile:
	vitis_hls -f hls_script.tcl

clean:
	rm -f *.log *.xo
	rm -rf ./vhls_dpupreproc_prj/solution_hls_dpupreproc_m_axi/impl/ip/.Xil/*
	rm -rf HLS vhls_dpupreproc_prj
