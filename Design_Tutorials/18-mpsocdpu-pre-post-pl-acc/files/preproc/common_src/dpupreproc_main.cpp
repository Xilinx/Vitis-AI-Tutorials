/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpupreproc_ref.cpp
Purpose:		Vitis HLS DPU pre-processor
Revision History:	23 July 2021
authors:		daniele.bagni@xilinx.com

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
