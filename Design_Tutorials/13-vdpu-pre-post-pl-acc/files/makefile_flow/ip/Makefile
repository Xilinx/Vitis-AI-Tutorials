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

ECHO=@echo

# here are the VHLS files
VDPU-PRE-POST-PL-ACC=/media/danieleb/DATA/ZF/new_VDPU-PRE-POST-PL-ACC/files

.PHONY: clean prepare_preproc prepare_postproc vhls_dpupreproc vhls_dpupostproc


all: prepare_files vhls_dpupreproc  vhls_dpupostproc

# make soft links to avoid too much replications of files (which ultimately is very error prone)
prepare_preproc:
	rm -f ./vhls_dpupreproc/data_pre/* ./vhls_dpupreproc/src/*  #clean directories
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/hls/hls_script.tcl                   ./vhls_dpupreproc/hls_script.tcl
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/hls/data_pre/testing_0_1920x832.bmp  ./vhls_dpupreproc/data_pre/testing_0_1920x832.bmp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/common_src/ap_bmp.cpp                ./vhls_dpupreproc/src/ap_bmp.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/common_src/ap_bmp.h                  ./vhls_dpupreproc/src/ap_bmp.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/hls/src/dpupreproc_defines.h         ./vhls_dpupreproc/src/dpupreproc_defines.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/common_src/dpupreproc_main.cpp       ./vhls_dpupreproc/src/dpupreproc_main.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/common_src/dpupreproc_ref.cpp        ./vhls_dpupreproc/src/dpupreproc_ref.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/common_src/dpupreproc_tb.cpp         ./vhls_dpupreproc/src/dpupreproc_tb.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/common_src/dpupreproc_tb.h           ./vhls_dpupreproc/src/dpupreproc_tb.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/preproc/hls/src/dpupreproc_vhls.cpp          ./vhls_dpupreproc/src/dpupreproc_vhls.cpp

prepare_postproc:
	rm -f ./vhls_dpupostproc/data_post/* ./vhls_dpupostproc/src/*  #clean directories
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/hls/hls_script.tcl                  ./vhls_dpupostproc/hls_script.tcl
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/hls/data_post/notscaled_input.bin   ./vhls_dpupostproc/data_post/notscaled_input.bin
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/hls/src/dpupostproc_defines.h       ./vhls_dpupostproc/src/dpupostproc_defines.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/common_src/dpupostproc_main.cpp     ./vhls_dpupostproc/src/dpupostproc_main.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/common_src/dpupostproc_ref.cpp      ./vhls_dpupostproc/src/dpupostproc_ref.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/common_src/dpupostproc_tb.cpp       ./vhls_dpupostproc/src/dpupostproc_tb.cpp
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/common_src/dpupostproc_tb.h         ./vhls_dpupostproc/src/dpupostproc_tb.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/common_src/luts.h                   ./vhls_dpupostproc/src/luts.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/common_src/lut_exp.h                ./vhls_dpupostproc/src/lut_exp.h
	ln -nsf $(VDPU-PRE-POST-PL-ACC)/postproc/hls/src/dpupostproc_vhls.cpp        ./vhls_dpupostproc/src/dpupostproc_vhls.cpp





prepare_files:
	make prepare_preproc
	make prepare_postproc

vhls_dpupreproc:
	make all -C vhls_dpupreproc

vhls_dpupostproc:
	make all -C vhls_dpupostproc

clean:
	$(RM) -rf build
	$(RM) -rf *.xo *.xclbin *.xclbin.info *.xclbin.link_summary *.xo.compile_summary _x
	$(RM) -rf *.dat *.pb xsim.dir *.xml *.ltx *.csv *.json *.protoinst *.wdb *.wcfg
	find -name "*.log" -delete
	find -name "*.jou" -delete
	make clean -C vhls_dpupreproc
	make clean -C vhls_dpupostproc
#find -name ".Xil" -type d -delete
