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

set_property PACKAGE_PIN N5 [get_ports {mipi_phy_if_data_p[0]}]
set_property PACKAGE_PIN N4 [get_ports {mipi_phy_if_data_n[0]}]
set_property PACKAGE_PIN M2 [get_ports {mipi_phy_if_data_p[1]}]
set_property PACKAGE_PIN M1 [get_ports {mipi_phy_if_data_n[1]}]
set_property PACKAGE_PIN N2 [get_ports mipi_phy_if_clk_p]
set_property PACKAGE_PIN P1 [get_ports mipi_phy_if_clk_n]

set_property PACKAGE_PIN E8 [get_ports clk_out_ov]
set_property PACKAGE_PIN A7 [get_ports {sensor_gpio_rst[0]}]
set_property IOSTANDARD LVCMOS18 [get_ports clk_out_ov]
set_property IOSTANDARD LVCMOS18 [get_ports {sensor_gpio_rst[0]}]
