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
# Top-level Makefile -- Calls sub-makefiles



ECHO                     = @echo

REQUIRED_VERSION         = 2021.1
VIVADO_VERSION          := $(shell vivado -version 2>/dev/null | grep "Vivado ")
VITIS_VERSION           := $(shell vitis -version 2>/dev/null | grep "Vitis " | sed 's/\*//g')
#PETALINUX_VERSION       := $(shell which petalinux-build)

export TARGET           := hw

.PHONY: clean help

version_check:	.version_check.done
prep:		.prep.done
ip:		.ip.done
host_apps:	.host_apps.done
vitis:		.vitis.done
package:	.package.done



help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) "      Command to generate everything for this design"
	$(ECHO) ""
	$(ECHO) "  make version_check"
	$(ECHO) "      checks out if the correct tools/versions are enabled"
	$(ECHO) "      checks out if the required environment variables are set"
	$(ECHO) "      (does not check correctness!)"
	$(ECHO) ""
	$(ECHO) "  make prep"
	$(ECHO) "      copies the prj/Vitis from Vitis-AI "
	$(ECHO) "      copies the HLS kernels from specified repository path "
	$(ECHO) "      establishes links to downloaded zcu102 platform and common software"
	$(ECHO) "      establishes link sysroot from common software"
	$(ECHO) "      Note: requires environment variables to be set accordingly"
	$(ECHO) "      Note: adapt/use script set_env_before_make.sh to setup environment"
	$(ECHO) ""
	$(ECHO) "  make ip"
	$(ECHO) "      Compile RTL and HLS kernels"
	$(ECHO) ""
	$(ECHO) "  make host_apps"
	$(ECHO) "      Compile PS applications"
	$(ECHO) "      * Requires all steps above"
	$(ECHO) ""
	$(ECHO) "  make vitis"
	$(ECHO) "      Extends and links the HW Platform with RTL and HLS kernels using Vitis v++"
	$(ECHO) "      Synthesize and Implements the complete design"
	$(ECHO) "      Packages the implemented design with the compiled PS apps"
	$(ECHO) "      Prepares and export SD card images"
	$(ECHO) "      * Requires all steps above"
	$(ECHO) ""
	$(ECHO) "  make package"
	$(ECHO) "      Packages the implemented design with the compiled PS apps"
	$(ECHO) "      Prepares and export SD card images"
	$(ECHO) "      * Requires all steps above"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

all: version_check prep ip vitis host_apps package

################################################################################
.version_check.done:
ifdef VITIS_AI_PATH
	$(ECHO) "Using VITIS_AI_PATH = $(VITIS_AI_PATH)"
else
	#export VITIS_AI_PATH=/sae26/projects/Vitis-AI
	$(ECHO) "VITIS_AI_PATH not explicitly set. Trying defaut VITIS_AI_PATH = $(VITIS_AI_PATH)"
	exit 1
endif
ifdef MPSOCDPU_PRE_POST_PL_ACC
	$(ECHO) "Using MPSOCDPU_PRE_POST_PL_ACC = $(MPSOCDPU_PRE_POST_PL_ACC)"
else
	#export MPSOCDPU_PRE_POST_PL_ACC=/sae26/projects/zynqMP/dpu-trd-ip-db
	$(ECHO) "MPSOCDPU_PRE_POST_PL_ACC not explicitly set. Trying defaut MPSOCDPU_PRE_POST_PL_ACC = $(MPSOCDPU_PRE_POST_PL_ACC)"
	exit 1
endif
ifdef VITIS_SYSROOTS
	$(ECHO) "Using VITIS_SYSROOTS = $(VITIS_SYSROOTS)"
else
	#export VITIS_SYSROOTS=/sae26/projects/zynqMP/dpu-trd-repo/downloads/xilinx-zynqmp-common-v2021.1/sdk/sysroots/cortexa72-cortexa53-xilinx-linux
	$(ECHO) "VITIS_SYSROOTS not explicitly set. Trying defaut VITIS_SYSROOTS = $(VITIS_SYSROOTS)"
	exit 1
endif

ifndef VIVADO_VERSION
	$(ECHO) "Vivado: not found"; exit 1
endif
ifeq (,$(findstring $(REQUIRED_VERSION),$(VIVADO_VERSION)))
	$(ECHO) "Vivado: $(REQUIRED_VERSION) not found"; exit 1
endif
ifndef PETALINUX_VERSION
	$(ECHO) "Petalinux: not found";
	#exit 1
endif
ifeq (,$(findstring $(REQUIRED_VERSION),$(PETALINUX_VERSION)))
	$(ECHO) "Petalinux: $(REQUIRED_VERSION) not found";
	#exit 1
endif
ifndef VITIS_VERSION
	$(ECHO) "Vitis: not found"; exit 1
endif
ifeq (,$(findstring $(REQUIRED_VERSION),$(VITIS_VERSION)))
	$(ECHO) "Vitis: $(REQUIRED_VERSION) not found"; exit 1
endif
	$(ECHO) "VERSION CHECK SUCCESSFUL"
	touch .version_check.done

################################################################################

.prep.done: .version_check.done
	$(ECHO) "Copying Vitis project from Vitis-AI $(VITIS_AI_PATH)"
	mkdir -p prj
	mkdir -p ip
	cp -r ${VITIS_AI_PATH}/dsa/DPU-TRD/prj/Vitis prj/.
	rm -rf ./ip/dpu_ip
	cp -rf ${VITIS_AI_PATH}/dsa/DPU-TRD/dpu_ip ./ip/dpu_ip
	cp -rf ${VITIS_AI_PATH}/dsa/DPU-TRD/app .
	$(ECHO) "Patching prj_config"
	mv prj/Vitis/config_file/prj_config prj/Vitis/config_file/prj_config.bak
	cp sources/config_file/prj_config prj/Vitis/config_file/
	$(ECHO) "Patching Makefile"
	mv prj/Vitis/Makefile prj/Vitis/Makefile.bak
	cp sources/Makefile prj/Vitis/.
	$(ECHO) "Patching package_dpu_kernel.tcl (to find hdl sources in subdir ip/dpu_ip)"
	mv prj/Vitis/scripts/package_dpu_kernel.tcl prj/Vitis/scripts/package_dpu_kernel.tcl.bak
	cp sources/package_dpu_kernel.tcl prj/Vitis/scripts/.
	$(ECHO) "Patching package_sfm_kernel.tcl (to find hdl sources in subdir ip/dpu_ip)"
	mv prj/Vitis/scripts/package_sfm_kernel.tcl prj/Vitis/scripts/package_sfm_kernel.tcl.bak
	cp sources/package_sfm_kernel.tcl prj/Vitis/scripts/.
	#linking ip to pre post kernel repo
	$(ECHO) "linking ip to pre post kernel repo"
	rm -rf ip/vhls_dpupreproc
	rm -rf ip/vhls_dpupostproc
	ln -nsf $(MPSOCDPU_PRE_POST_PL_ACC)/makefile_flow/ip/vhls_dpupreproc   ./ip/vhls_dpupreproc
	ln -nsf $(MPSOCDPU_PRE_POST_PL_ACC)/makefile_flow/ip/vhls_dpupostproc  ./ip/vhls_dpupostproc
	#linking host_apps
	ln -nsf $(MPSOCDPU_PRE_POST_PL_ACC)/makefile_flow/host_apps ./host_apps
	#$(MAKE) prepare_files -C host_apps
	touch .prep.done

################################################################################

.ip.done:  .prep.done .version_check.done
	$(MAKE) all -C ip/vhls_dpupreproc
	$(MAKE) all -C ip/vhls_dpupostproc
	touch .ip.done

################################################################################

.host_apps.done:  .prep.done .version_check.done
	$(MAKE) all -C host_apps
	touch .host_apps.done

################################################################################
.vitis.done: .prep.done .version_check.done .ip.done
	$(MAKE) binary_container_1/dpu.xclbin -C prj/Vitis
	touch .vitis.done

################################################################################
.package.done: .vitis.done .host_apps.done
	$(MAKE) package -C prj/Vitis
	touch .package.done

################################################################################
clean:
	rm -f .prep.done .vitis.done .host_apps.done .ip.done .version_check.done
	rm -rf ip/dpu_ip ip/vhls_dpupreproc ip/vhls_dpupostproc
	rm -f *.log *.str *.jou
	rm -rf .Xil app prj
	rm -rf prj ip/dpu_ip
	rm host_apps
	#$(MAKE) clean -C prj/Vitis
	#$(MAKE) clean -C host_apps
	#$(MAKE) clean -C ip


