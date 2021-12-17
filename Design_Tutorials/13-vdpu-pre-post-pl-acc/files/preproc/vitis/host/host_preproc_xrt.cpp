/*************************************************************************************
 Vendor:		Xilinx
 Associated Filename:	host_postproc_xrt.cpp
 Purpose:		Testbench file
 authors:		daniele.bagni@xilinx.com peters@xilinx.com

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

#include "dpupreproc_tb.h"

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_profile.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

#include "ap_bmp.h"


#ifdef ARM_HOST //this code has to run ONLY on the ARM CPU of VERSAL ACAP


/* **************************************************************************************************** */
/* just to check the reference output image and put it into 3 separate planes to write the BMP file later */

void format_ref_img(unsigned char *R, unsigned char *G, unsigned char *B,
		    char *out_rgb_img, unsigned short int height, unsigned short int width)
{
  unsigned short int rows = height;
  unsigned short int cols = width;

  int k = 0;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      unsigned char uB = (unsigned char) out_rgb_img[k + 0];
      unsigned char uG = (unsigned char) out_rgb_img[k + 1];
      unsigned char uR = (unsigned char) out_rgb_img[k + 2];
      B[r * PRE_MAX_WIDTH + c] = uB;
      G[r * PRE_MAX_WIDTH + c] = uG;
      R[r * PRE_MAX_WIDTH + c] = uR;
      k = k + 3;
    }
  }
}


/* **************************************************************************************************** */
/* **************************************************************************************************** */

static const std::string error_message = "Error: Result mismatch:\n"
  "i = %d CPU result = %d Device result = %d\n";


xrt::profile::user_range PRE_range("Setup", "Start of execution to context creation");
xrt::profile::user_event PRE_events;

xrtXclbinHandle PRE_xclbin;
xrtDeviceHandle PRE_device;
uint32_t        *PRE_in_bomapped, *PRE_out_bomapped;
xrtBufferHandle img_inp, img_out;
xrtKernelHandle kernel_pre;
xrtRunHandle    kernel_pre_rhdl;


/* **************************************************************************************************** */
/* **************************************************************************************************** */


void PRE_Start(  char* PRE_xclbinFilename)
{

  // Assume the Alveo card is the first device
  PRE_device = xrtDeviceOpen(0);

  // We are not validating the user input. A general program should verify it exists.
  PRE_xclbin = xrtXclbinAllocFilename(PRE_xclbinFilename);
  xrtDeviceLoadXclbinHandle(PRE_device, PRE_xclbin);
  xuid_t PRE_xclbin_uuid;
  xrtXclbinGetUUID(PRE_xclbin,      PRE_xclbin_uuid);
  printf("PRE XCLBIN UUID 0x%px\n", PRE_xclbin_uuid);

  kernel_pre = xrtPLKernelOpen(PRE_device, PRE_xclbin_uuid,"hls_dpupreproc_m_axi");

  size_t size_in_bytes = PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char) * 3;
  printf("size_in_byte=%ld\n", size_in_bytes);

  PRE_range.start("Setup", "Create Buffers");
  //create a buffer for the input image
  img_inp = xrtBOAlloc(PRE_device, size_in_bytes, 0, 0);
  PRE_in_bomapped = reinterpret_cast<uint32_t*>(xrtBOMap(img_inp));
  memset(PRE_in_bomapped, 0x0, size_in_bytes);
  printf("Input  memory virtual  addr 0x%px\n", PRE_in_bomapped);
  printf("Input  memory physical addr 0x%px\n", (void*) xrtBOAddress(img_inp));
  //create a buffer for the output image
  img_out = xrtBOAlloc(PRE_device, size_in_bytes, 0, 0);
  PRE_out_bomapped = reinterpret_cast<uint32_t*>(xrtBOMap(img_out));
  memset(PRE_out_bomapped, 0x00000000, size_in_bytes);
  printf("Output memory virtual  addr 0x%px\n", PRE_out_bomapped);
  printf("Output memory physical addr 0x%px\n", (void*) xrtBOAddress(img_out));
  PRE_range.end();
  PRE_events.mark("Setup done, Buffers created");
}

/* **************************************************************************************************** */
/* **************************************************************************************************** */

int PRE_main(char* PRE_xclbinFilename, float pre_means[3], float pre_scales[3], int dpu_input_fixpos)
{

  int index, row, col;

  
  /* **************************************************************************************** */
  // memory allocation, all data to/from postproc kernel need to be created with xrtBOAlloc

  unsigned char *R, *G, *B;
  R = (unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  G = (unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  B = (unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  char *tempbuf1 =  (char*) malloc(MAX_PATH * sizeof(char));
  signed char *ref_rgb_img = (signed char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(signed char) * 3);
  
  memset(R, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  memset(G, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  memset(B, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  
  /* **************************************************************************************** */
  // setup arguments
  
  PRE_range.start("Arguments", "Setting up arguments");
  //set parameters
  unsigned short int width = PRE_MAX_WIDTH;
  unsigned short int height = PRE_MAX_HEIGHT;
  PRE_range.end();
  PRE_events.mark("arguments set");

  /* **************************************************************************************** */
  // setup and start the kernel
 
  PRE_Start(PRE_xclbinFilename);
  unsigned char *din     = (unsigned char*) PRE_in_bomapped;
  signed char *hls_out   = (  signed char*) PRE_out_bomapped;

  /* **************************************************************************************** */
  // read input file: do not move it prior to PRE_Start() or you will have SEGV ERROR

  PRE_range.start("Reading input image", "Reading input image to buffers");
  sprintf(tempbuf1, "%s.bmp", PRE_INPUT_IMAGE);
  // Fill a frame with data
  int read_tmp = BMP_Read(tempbuf1, PRE_MAX_HEIGHT, PRE_MAX_WIDTH, R, G, B, din);
  if (read_tmp != 0) {
    printf("%s Loading image failed\n", tempbuf1);
    exit(1);
  }
  printf("Input image %s just read\n", tempbuf1);
  PRE_range.end();
  PRE_events.mark("input image loaded");

  /* **************************************************************************************** */
  // Execute the kernel

  PRE_range.start("Kernel execution", "run kernel preproc");
  //xrtURStart(0, "kernel execution", "kernel execution"); DB
  // Execute
  kernel_pre_rhdl = xrtKernelRun(kernel_pre, img_inp, img_out,
			pre_means[0], pre_means[1], pre_means[2],
			pre_scales[0],pre_scales[1],pre_scales[2],
			dpu_input_fixpos,
			height, width);
  auto state = xrtRunWait(kernel_pre_rhdl); //wait for the kernel to finish
  xrtUREnd(0);
  PRE_range.end();
  PRE_events.mark("kernel finished");

  /* **************************************************************************************** */
  // call reference function

  PRE_range.start("Ref function", "calling reference function");
  // call reference function
  arm_ref_dpupreproc(R, G, B, ref_rgb_img,
		  pre_means, pre_scales, dpu_input_fixpos,
		  height, width);
  //write out the reference bmp file re-using the same R G B buffers
  format_ref_img(R, G, B, ref_rgb_img, height, width);
  sprintf(tempbuf1, "%s_ref.bmp", PRE_INPUT_IMAGE); //construct the file name
  printf("Writing reference image %s \n", tempbuf1);
  BMP_Write(tempbuf1, height, width, R, G, B);
  PRE_range.end();
  PRE_events.mark("Ref function end");


  /* **************************************************************************************** */
  //verify results here
  
  PRE_range.start("verify", "verify results");
  //compare to the golden reference
  memset(R, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  memset(G, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  memset(B, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  int tot_err = 0;
  int tot_err_r = 0;
  int tot_err_g = 0;
  int tot_err_b = 0;
  for (row = 0; row < PRE_MAX_HEIGHT; row++) {
    for (col = 0; col < 3 * PRE_MAX_WIDTH; col = col + 3) {
      //ref data
      char rB = ref_rgb_img[row * PRE_MAX_WIDTH * 3 + col + 0];
      char rG = ref_rgb_img[row * PRE_MAX_WIDTH * 3 + col + 1];
      char rR = ref_rgb_img[row * PRE_MAX_WIDTH * 3 + col + 2];
      //hls data
      char hB = hls_out[row * PRE_MAX_WIDTH * 3 + col + 0];
      char hG = hls_out[row * PRE_MAX_WIDTH * 3 + col + 1];
      char hR = hls_out[row * PRE_MAX_WIDTH * 3 + col + 2];
      // check difference
      int diff_R = rR - hR;
      diff_R = ABS(diff_R);
      int diff_G = rG - hG;
      diff_G = ABS(diff_G);
      int diff_B = rB - hB;
      diff_B = ABS(diff_B);
      if (diff_R > 0)
	printf("diff R @ [%3d][%4d]: got %3d expected %3d\n", row, col / 3, hR, rR);
      if (diff_G > 0)
	printf("diff G @ [%3d][%4d]: got %3d expected %3d\n", row, col / 3, hG, rG);
      if (diff_B > 0)
	printf("diff B @ [%3d][%4d]: got %3d expected %3d\n", row, col / 3, hB, rB);
      tot_err += diff_R + diff_G + diff_B;
      tot_err_r += diff_R;
      tot_err_g += diff_G;
      tot_err_b += diff_B;
      //ps: write output image to R G B buffers for bmp write file
      index = (row * PRE_MAX_WIDTH) + (col / 3);
      R[index] = hR;
      G[index] = hG;
      B[index] = hB;
    }
  }
  printf("final row  : %d\n", row);
  printf("final col  : %d\n", col);
  printf("final index: %d\n", index);
  printf("total error: %d\n", tot_err);
  printf("R error: %d\n", tot_err_r);
  printf("G error: %d\n", tot_err_g);
  printf("B error: %d\n", tot_err_b);
  //write out the processed bmp file re-using the R G B buffers
  sprintf(tempbuf1, "%s_out.bmp", PRE_INPUT_IMAGE); //construct the file name
  printf("Writing output image %s \n", tempbuf1);
  BMP_Write(tempbuf1, PRE_MAX_HEIGHT, PRE_MAX_WIDTH, R, G, B);
  PRE_range.end();
  PRE_events.mark("verified result");
  
  /* **************************************************************************************** */
  //Clean up
  
  //note: img_inp and img_out BO need to be freed first, otherwise error!
  xrtBOFree(img_inp);
  xrtBOFree(img_out);
  xrtRunClose(kernel_pre_rhdl);
  xrtKernelClose(kernel_pre);
  xrtDeviceClose(PRE_device);
  xrtXclbinFreeHandle(PRE_xclbin);
  //free memory
  free(ref_rgb_img);
  free(R); free(G); free(B);

  /* **************************************************************************************** */
  //end 

  std::cout << "Preproc done " << std::endl;
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


	char* PRE_xclbinFilename = argv[1];
	std::cout << "PRE_xclbinFilename " << PRE_xclbinFilename << std::endl;

	//float pre_means[3]  = {PRE_MEANS[0],   PRE_MEANS[1],  PRE_MEANS[2]};
	float pre_means[3]  = { 0.0f,   0.0f, 0.0f };
	//float pre_scales[3] = {PRE_SCALES[0], PRE_SCALES[1], PRE_SCALES[2]};
	float pre_scales[3] = {1.0f, 1.0f, 1.0f};	
	// this parameter in the real life must be taken by the ARM CPU from the DPU xmodel subgraph
	// I put it here just to say that it must come from the external world
	//int dpu_fixpos = PRE_FIXPOS;
	int dpu_fixpos = 0; 


	int PRE_ret = PRE_main(PRE_xclbinFilename,
			       pre_means, pre_scales, dpu_fixpos);

	return PRE_ret;

}

#endif

#endif //#ifdef ARM_HOST 
