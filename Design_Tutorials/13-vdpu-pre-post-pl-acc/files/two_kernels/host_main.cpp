/*************************************************************************************
 Vendor:		Xilinx
 Associated Filename:	host_main.cpp
 Purpose:		Testbench file
 author:		daniele.bagni@xilinx.com

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


#include "dpupostproc_tb.h"
#include "dpupreproc_tb.h"

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_profile.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>


extern  void POST_Start(char* PRE_POST_xclbinFilename);
extern  int  POST_main( char* PRE_POST_xclbinFilename, int dpu_output_fixpos);
extern  void PRE_Start( char* PRE_POST_xclbinFilename);
extern  int  PRE_main(  char* PRE_POST_xclbinFilename, float pre_means[3], float pre_scales[3], int dpu_input_fixpos);

#define TWO_KERNELS //to prevent the compilation of main() from host_*proc_xrt.cpp


int main(int argc, char* argv[])
{

	// Get xclbin file name from the first argument
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
	for (int i = 0; i < argc; i++) fprintf(stderr,"%s ", argv[i]);
	fprintf(stderr,"V1 \n\n");


	char* PRE_POST_xclbinFilename = argv[1];
	std::cout << "PRE_POST_xclbinFilename " << PRE_POST_xclbinFilename << std::endl;

    // ________________________________________________________________________________________________________________________________________________
	// WARNING:
	// In the real application including the DPU, besides the two PL kernels, the ARM CPU should get at first the "*fixpos*" of the scaling factors
	// of the input and output CNN tensors from the DPU subgraph xmodel at run time. Here I am just emulating this behavior with a-priori values.

	// these PREPROC parameters must be set by the user, depending on the CNN xmodel to be loaded
	float pre_means[3]  = {PRE_MEANS[0],   PRE_MEANS[1],  PRE_MEANS[2]};
	float pre_scales[3] = {PRE_SCALES[0], PRE_SCALES[1], PRE_SCALES[2]};

	// these two parameters in the real life must be taken by the ARM CPU from the DPU xmodel subgraph with code similar to the following 4 lines:
    //int dpu_out_fixpos = (outputTensors[0])->template get_attr<int>("fix_point");
	////auto out_fix_scale = std::exp2f(1.0f * (float)out_fixpos);
	//int dpu_inp_fixpos = (inputTensors[0])->template get_attr<int>("fix_point");
	////auto inp_fix_scale = std::exp2f(1.0f * (float)in_fixpos);

	int dpu_input_fixpos  = PRE_FIXPOS;    // *fixpos* of input CNN tensor taken from DPU graph xmodel
	int dpu_output_fixpos = POSTPR_FIXPOS; // *fixpos* of output CNN tensor taken from DPU graph xmodel
    // ________________________________________________________________________________________________________________________________________________

	int PRE_ret  =  PRE_main(PRE_POST_xclbinFilename,  pre_means, pre_scales, dpu_input_fixpos);


	int POST_ret = POST_main(PRE_POST_xclbinFilename, dpu_output_fixpos);


	int ret = POST_ret + PRE_ret;

	return ret;

}
