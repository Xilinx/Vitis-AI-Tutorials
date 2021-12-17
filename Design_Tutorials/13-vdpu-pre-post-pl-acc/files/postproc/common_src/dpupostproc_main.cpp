/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpu_post_main.cpp
Purpose:		main for the HLS TB
Revision History:	23 July 2021
author:			daniele.bagni@xilinx.com

**************************************************************************************

Copyright 2021 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*******************************************************************************/

/*
This main() is just a placeholder for running a self checking TB.
In the real application, before starting its job this core has to take from the external DPU
the "output_fixpos" parameter which has values from 1 to 7
*/

#include "dpupostproc_tb.h"

//extern void Prepare7ExpLUTS(void);

int main(int argc, char **argv)
{

  // The ARM CPU could compute all the Look Up Table and send it to the post-processor
  // as an alternative architectural choice to save BRAMs
  //Prepare7ExpLUTS();

  // The ARM CPU should get the "output_fixpos"  from the DPU at run time

  int output_fixpos = POSTPR_FIXPOS;

  int ret =  POST_tb_main(argc, argv, output_fixpos);

  return ret;

}
