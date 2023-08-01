/*
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/


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
