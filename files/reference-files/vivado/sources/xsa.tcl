# *************************************************************************
# Copyright 2019 Xilinx Inc.
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
# *************************************************************************

set name [get_property NAME [current_project]]
set output_file [format ../hw_platform/%s.xsa $name]
set bd [format "%s.bd" [current_bd_design]]
set_property PFM_NAME [format "xilinx.com:board:%s:1.0" $name] [get_files $bd]

set_property platform.default_output_type "sd_card" [current_project]
set_property platform.design_intent.embedded "true" [current_project]
set_property platform.design_intent.server_managed "false" [current_project]
set_property platform.design_intent.external_host "false" [current_project]
set_property platform.design_intent.datacenter "false" [current_project]
set_property platform.post_sys_link_tcl_hook ./sources/dynamic_postlink.tcl [current_project]

# Get the xlconcat instance and pin number to work on now
set __xlconcat_inst_num 0
set __xlconcat_pin_num 0

set __xlconcat_inst [get_bd_cells -hierarchical -quiet -filter NAME=~xlconcat_interrupt_${__xlconcat_inst_num}]
set __xlconcat_pin [get_bd_pins -of_objects $__xlconcat_inst -quiet -filter NAME=~In${__xlconcat_pin_num}]

if {[llength $__xlconcat_pin] == 1} {
  if {[llength [get_bd_nets /xlconstant_gnd_dout]] == 1} {
    puts "Passed verify test"
    write_hw_platform -unified -force ../hw_platform/u96v2_mipi.xsa
  } else {
    puts "Missing required name for const net: net should be xlconstant_gnd_dout"
    puts "Halting XSA output"
  }
} else {
  puts "Halting XSA output"
}
