/*************************************************************************************
 Vendor:		Xilinx
 Associated Filename:	host_main.cpp
 Purpose:		Testbench file
 author:		daniele.bagni@xilinx.com

 **************************************************************************************

 Copyright 2020 Xilinx Inc.

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
extern  int  POST_main( char* PRE_POST_xclbinFilename);
extern  void PRE_Start( char* PRE_POST_xclbinFilename);
extern  int  PRE_main(  char* PRE_POST_xclbinFilename);

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

	int POST_ret = POST_main(PRE_POST_xclbinFilename);

	int PRE_ret  =  PRE_main(PRE_POST_xclbinFilename);

	int ret = POST_ret + PRE_ret;

	return ret;

}
