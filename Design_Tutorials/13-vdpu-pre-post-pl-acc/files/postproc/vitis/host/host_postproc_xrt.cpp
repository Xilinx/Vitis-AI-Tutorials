/*************************************************************************************
 Vendor:		Xilinx
 Associated Filename:	host_postproc_xrt.cpp
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

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_profile.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>


#ifdef ARM_HOST //this code has to run ONLY on the ARM CPU of VERSAL ACAP


/* **************************************************************************************** */
/* **************************************************************************************** */

xrt::profile::user_range POST_range("Setup", "Start of execution to context creation");
xrt::profile::user_event POST_events;

xrtKernelHandle kernel_post;
xrtRunHandle    kernel_post_rhdl;
xrtBufferHandle inp_data, max_out, ind_out;
unsigned char   *POST_in_bomapped, *POST_out1_bomapped, *POST_out2_bomapped;
xrtDeviceHandle POST_device;
xrtXclbinHandle POST_xclbin;

/* **************************************************************************************** */
/* **************************************************************************************** */


void POST_Start(char* POST_xclbinFilename)
{
	// Assume the Alveo card is the first device
	POST_device = xrtDeviceOpen(0);

	// We are not validating the user input. A general program should verify it exists.
	POST_xclbin = xrtXclbinAllocFilename(POST_xclbinFilename);
	xrtDeviceLoadXclbinHandle(POST_device, POST_xclbin);
	xuid_t POST_xclbin_uuid;
	xrtXclbinGetUUID(POST_xclbin, POST_xclbin_uuid);
	printf("XCLBIN UUID 0x%px\n", POST_xclbin_uuid);

	kernel_post = xrtPLKernelOpen(POST_device, POST_xclbin_uuid, "hls_dpupostproc_m_axi");

	size_t inp_size_in_bytes = POST_MAX_HEIGHT * POST_MAX_WIDTH * MAX_NUM_OF_CLASSES * sizeof(char);
	printf(" INPUT size_in_byte=%ld\n", inp_size_in_bytes);
	size_t out_size_in_bytes = POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(char);
	printf("OUTPUT size_in_byte=%ld\n", out_size_in_bytes);


	POST_range.start("Setup", "Create Buffers");
	//create a buffer for the input data
	inp_data = xrtBOAlloc(POST_device, inp_size_in_bytes, 0, 0);
	POST_in_bomapped = reinterpret_cast<uint8_t*>(xrtBOMap(inp_data));
	memset(POST_in_bomapped, 0x0, inp_size_in_bytes);
	printf("Input   memory virtual  addr 0x%px\n", POST_in_bomapped);
	printf("Input   memory physical addr 0x%px\n", (void*) xrtBOAddress(inp_data));
	//create buffer1 for the output data
	max_out = xrtBOAlloc(POST_device, out_size_in_bytes, 0, 0);
	POST_out1_bomapped = reinterpret_cast<uint8_t*>(xrtBOMap(max_out));
	memset(POST_out1_bomapped, 0x00000000, out_size_in_bytes);
	printf("Output1 memory virtual  addr 0x%px\n", POST_out1_bomapped);
	printf("Output1 memory physical addr 0x%px\n", (void*) xrtBOAddress(max_out));
	//create buffer2 for the output data
	ind_out = xrtBOAlloc(POST_device, out_size_in_bytes, 0, 0);
	POST_out2_bomapped = reinterpret_cast<uint8_t*>(xrtBOMap(ind_out));
	memset(POST_out2_bomapped, 0x00000000, out_size_in_bytes);
	printf("Output2 memory virtual  addr 0x%px\n", POST_out2_bomapped);
	printf("Output2 memory physical addr 0x%px\n", (void*) xrtBOAddress(ind_out));
	POST_range.end();
	POST_events.mark("Setup done, Buffers created");
}

/* **************************************************************************************** */
/* **************************************************************************************** */

int POST_main(char* POST_xclbinFilename, int dpu_output_fixpos)
{
	int width, height;
	float POST_total_error;

	/* **************************************************************************************** */
	// memory allocation, all data to/from postproc kernel need to be created with xrtBOAlloc

	char *tempbuf1           = (         char*) malloc(MAX_PATH * sizeof(char));
	unsigned char *ref_max   = (unsigned char*) malloc( POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(char) );
	unsigned char *ref_index = (unsigned char*) malloc( POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(char) );

	/* **************************************************************************************** */
	// setup arguments

	POST_range.start("Arguments", "Setting Arguments");
	//set parameters
	width  = POST_MAX_WIDTH;
	height = POST_MAX_HEIGHT;
	POST_range.end();
	POST_events.mark("arguments set");

	/* **************************************************************************************** */
	// setup and start the kernel

	POST_Start(POST_xclbinFilename);
	signed char   *din       = (  signed char*) POST_in_bomapped;
	unsigned char *hls_max   = (unsigned char*) POST_out1_bomapped;
	unsigned char *hls_index = (unsigned char*) POST_out2_bomapped;

	/* **************************************************************************************** */
	// read input file: do not move it prior to POST_Start() or you will have SEGV ERROR

	POST_range.start("Reading input files", "Reading input files");
	sprintf(tempbuf1, "%s", "./data_post/notscaled_input.bin");
	Read_Bin_File_of_Schar(tempbuf1, POST_MAX_HEIGHT*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES, din);
	fprintf(stderr,"Loaded input data file  %s\n", tempbuf1);
	// just for debug
	sprintf(tempbuf1, "%s", "./data_post/dbg_notscaled_input.bin");
	Write_Bin_File_as_Schar(tempbuf1, POST_MAX_HEIGHT*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES, din);
	POST_range.end();
	POST_events.mark("input data loaded");

	/* **************************************************************************************** */
	// Execute the kernel

	POST_range.start("Kernel execution", "run kernel postproc");
	xrtURStart(0, "kernel execution", "kernel execution");
	// Execute
	kernel_post_rhdl = xrtKernelRun(kernel_post, inp_data, max_out, ind_out, dpu_output_fixpos, height, width );
	auto POST_state = xrtRunWait(kernel_post_rhdl); //wait for the kernel to finish
	xrtUREnd(0);
	POST_range.end();
	POST_events.mark("kernel finished");

	/* **************************************************************************************** */
	// call reference function

	POST_range.start("Ref function", "calling reference function");
	fprintf(stderr, "REF design with DPU output fixpos %d\n", dpu_output_fixpos);
	ref_dpupostproc(din, ref_max, ref_index, dpu_output_fixpos, height, width);
	//save reference results
	fprintf(stderr,"writing REF files\n");
	sprintf(tempbuf1, "%s", "./data_post/arm_ref_max.bin");
	Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, ref_max);
	sprintf(tempbuf1, "%s", "./data_post/arm_ref_index.bin");
	Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, ref_index);
	POST_range.end();
	POST_events.mark("Ref function end");

	/* **************************************************************************************** */
	//verify results here

	POST_range.start("verify", "verify results");
	fprintf(stderr, "HLS Checking results: REF vs. HLS\n");
	float total_max_error   = check_output_max(  ref_max,   hls_max,   height, width);
	float total_index_error = check_output_index(ref_index, hls_index, height, width);
	POST_total_error = total_max_error + total_index_error;
	fprintf(stderr, "error in max   value: %f\n", total_max_error);
	fprintf(stderr, "error in index value: %f\n", total_index_error);
	fprintf(stderr,"writing HLS Kernel output generated files\n");
	sprintf(tempbuf1, "%s", "./data_post/pl_hls_max.bin");
	Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, hls_max);
	sprintf(tempbuf1, "%s", "./data_post/pl_hls_index.bin");
	Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, hls_index);
	fprintf(stderr,"HLS Test done\n");
	if (POST_total_error > POST_REF_THRESHOLD)
	  {
	    fprintf(stderr, "TEST FAILED!: error = %10.8f\n", POST_total_error);
	  } else {
	  fprintf(stderr, "TEST SUCCESSFULL!\n");
	}
	POST_range.end();
	POST_events.mark("verified result");

	/* **************************************************************************************** */
	//Clean up

	//note: BO allocated buffers need to be freed first, otherwise error!
	xrtBOFree(inp_data);
	xrtBOFree(max_out);
	xrtBOFree(ind_out);
	xrtKernelClose(kernel_post);
	xrtDeviceClose(POST_device);
	xrtXclbinFreeHandle(POST_xclbin);
	xrtRunClose(kernel_post_rhdl);
	// free memory
	free(tempbuf1);
	free(ref_max);
	free(ref_index);

	/* **************************************************************************************** */
	//end

	std::cout << "PL PostProc done " << std::endl;
	return (EXIT_SUCCESS);
}

/* **************************************************************************************** */
/* **************************************************************************************** */
/* **************************************************************************************** */

#ifndef TWO_KERNELS
int main(int argc, char* argv[])
{

	// Get xclbin file name from the first argument
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}
	for (int i = 0; i < argc; i++) fprintf(stderr,"%s ", argv[i]);
	fprintf(stderr,"V3 \n\n");


	char* POST_xclbinFilename = argv[1];
	std::cout << "POST_xclbinFilename " << POST_xclbinFilename << std::endl;

	int dpu_output_fixpos = POSTPR_FIXPOS;
	int POST_ret = POST_main(POST_xclbinFilename, dpu_output_fixpos);
	return POST_ret;

}
#endif


#endif //#ifdef ARM_HOST
