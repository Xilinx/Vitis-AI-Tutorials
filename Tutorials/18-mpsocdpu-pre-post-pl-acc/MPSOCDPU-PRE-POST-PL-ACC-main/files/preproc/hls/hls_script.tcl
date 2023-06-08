# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


#### just for info ####
set tool "unknown"
set version "unknown"

foreach toolcheck {vivado_hls vitis_hls} {
    if {[ string first $toolcheck $argv0 ]>=0} {
        set tool $toolcheck
        set version "[exec which $toolcheck | sed -e "s:.*\\(20..\\..\\).*:\\1:g" ]"
    }
}

puts "*** tool: $tool is version: $version ***"
####

set do_csim      1
set do_csynth    1
set do_cosim     0
set do_export_xo 1

open_project vhls_dpupreproc_prj

# using variable mytop to set which version we're using:
# [] hls_dpupreproc is the reference model from Dan and uses AXIS interfaces
# [] hls_dpupreproc_m_axi is derived from the above but using the M_AXI interfaces
# commenting/uncommenting will change the solution name; note this is only driven from the script not possible from the GUI

#set mytop hls_dpupreproc
set mytop hls_dpupreproc_m_axi

set_top $mytop

set mycflags "-I ./src -std=c++14"

add_files     src/dpupreproc_vhls.cpp -cflags $mycflags
add_files -tb src/dpupreproc_ref.cpp  -cflags $mycflags
add_files -tb src/dpupreproc_tb.cpp   -cflags $mycflags
add_files -tb src/dpupreproc_main.cpp -cflags $mycflags
add_files -tb src/ap_bmp.cpp          -cflags $mycflags
add_files -tb data_pre

#open_solution "solution_$mytop" -flow_target vivado
open_solution "solution_$mytop" -flow_target vitis

##VCK190 ES1
## set_part {xcvc1902-vsva2197-2MP-e-S-es1}
#VCK190 Production
#set_part {xcvc1902-vsva2197-2MP-e}
# ZCU102
set_part {xczu9eg-ffvb1156-2-e}

create_clock -period 3 -name default

config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512 -m_axi_offset slave
config_rtl -register_reset_num 3
config_export -format xo -output ./$mytop.xo -rtl verilog

if {$do_csim > 0} {
    csim_design  -clean
}

if {$do_csynth > 0} {
    csynth_design
}

if {$do_cosim > 0} {
    cosim_design
}

#export_design -flow syn -rtl verilog -format ip_catalog
if {$do_export_xo > 0} {
    #export_design -flow impl -rtl verilog -format xo -output ./$mytop.xo
    export_design -rtl verilog -format xo -output ./$mytop.xo
}

exit
