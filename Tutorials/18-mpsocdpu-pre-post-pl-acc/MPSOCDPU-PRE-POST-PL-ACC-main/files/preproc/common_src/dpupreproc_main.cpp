/*

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/


#include "dpupreproc_tb.h"


int main(int argc, char **argv)
{


	float pre_means[3]  = {PRE_MEANS[0],   PRE_MEANS[1],  PRE_MEANS[2]};
	float pre_scales[3] = {PRE_SCALES[0], PRE_SCALES[1], PRE_SCALES[2]};

	// this parameter in the real life must be taken by the ARM CPU from the DPU xmodel subgraph
	// I put it here just to say that it must come from the external world
	int dpu_fixpos = PRE_FIXPOS;



	return PRE_tb_main(argc, argv, pre_means, pre_scales, dpu_fixpos);

}
