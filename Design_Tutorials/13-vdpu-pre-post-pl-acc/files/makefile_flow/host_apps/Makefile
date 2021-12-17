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

.PHONY: all clean prepare_preproc prepare_postproc prepare_files


## here are the VHLS files
#VDPU_PRE_POST_PL_ACC=/media/danieleb/DATA/ZF/new_VDPU_PRE_POST_PL_ACC/files

all:
	make prepare_files
	make all -C preproc
	make all -C postproc
	make all -C pre2post
#	make all -C zc_pre2post


# make soft links to avoid too much replications of files (which ultimately is very error prone)
prepare_preproc:
	rm -f ./preproc/data_pre/* ./preproc/src/*  #clean directories
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/hls/data_pre/testing_0_1920x832.bmp  ./preproc/data_pre/testing_0_1920x832.bmp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/common_src/ap_bmp.cpp                ./preproc/src/ap_bmp.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/common_src/ap_bmp.h                  ./preproc/src/ap_bmp.h
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/common_src/dpupreproc_main.cpp       ./preproc/src/dpupreproc_main.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/common_src/dpupreproc_ref.cpp        ./preproc/src/dpupreproc_ref.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/common_src/dpupreproc_tb.cpp         ./preproc/src/dpupreproc_tb.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/common_src/dpupreproc_tb.h           ./preproc/src/dpupreproc_tb.h
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/vitis/host/host_preproc_xrt.cpp      ./preproc/src/host_preproc_xrt.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/preproc/vitis/host/dpupreproc_defines.h      ./preproc/src/dpupreproc_defines.h


prepare_postproc:
	rm -f ./postproc/data_post/* ./postproc/src/*  #clean directories
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/hls/data_post/notscaled_input.bin   ./postproc/data_post/notscaled_input.bin
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/common_src/dpupostproc_main.cpp     ./postproc/src/dpupostproc_main.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/common_src/dpupostproc_ref.cpp      ./postproc/src/dpupostproc_ref.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/common_src/dpupostproc_tb.cpp       ./postproc/src/dpupostproc_tb.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/common_src/dpupostproc_tb.h         ./postproc/src/dpupostproc_tb.h
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/common_src/luts.h                   ./postproc/src/luts.h
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/common_src/lut_exp.h                ./postproc/src/lut_exp.h
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/vitis/host/host_postproc_xrt.cpp    ./postproc/src/host_postproc_xrt.cpp
	ln -nsf $(VDPU_PRE_POST_PL_ACC)/postproc/vitis/host/dpupostproc_defines.h    ./postproc/src/dpupostproc_defines.h


prepare_files:
	make prepare_preproc
	make prepare_postproc

clean:
	make clean -C preproc
	make clean -C postproc
	make clean -C pre2post
#	make clean -C zc_pre2post
