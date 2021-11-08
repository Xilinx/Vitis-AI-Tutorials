/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpupreproc_vhls.cpp
Purpose:		Vitis HLS DPU pre-processor
Revision History:	17 June 2021
authors:		daniele.bagni@xilinx.com, herve.ratigner@xilinx.com

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

#include "dpupreproc_defines.h"

// takes an RGB 24bit per pixel image (R, G, B each one of 8-bit) from a source (what?) and store it
// into DDR as 3*8-bit unsigned long word

extern "C"
void hls_dpupreproc(uPix3_t *inp_img, Dat3_t *img_out,
		    float norm_fact, float shift_fact, float scale_fact,
		    unsigned short int height, unsigned short int width)
{
#pragma HLS INTERFACE axis register_mode=both register depth=PRE_hls_IMGSZ port=inp_img
#pragma HLS INTERFACE axis register_mode=both register depth=PRE_hls_IMGSZ port=img_out
  /*
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1  depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH_OUT
  */

#pragma HLS INTERFACE s_axilite port=scale_fact bundle=control
#pragma HLS INTERFACE s_axilite port=norm_fact  bundle=control
#pragma HLS INTERFACE s_axilite port=shift_fact bundle=control
#pragma HLS INTERFACE s_axilite port=height   bundle=control
#pragma HLS INTERFACE s_axilite port=width    bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

  float gain = scale_fact;
  float norm = 1.0f / norm_fact;

  unsigned short int rows = height;
  unsigned short int cols = width;

  printf("DPU PreProc model.... !!!\n");
 L1:for (int r = 0; r < rows; r++)
    {
#pragma HLS LOOP_TRIPCOUNT max=PRE_hls_MAX_H min=1
    L2:for (int c = 0; c < cols; c++)
	{
#pragma HLS LOOP_TRIPCOUNT max=PRE_hls_MAX_W min=1
#pragma HLS PIPELINE II=1

	  uPix3_t triple_pix = inp_img[r*PRE_MAX_WIDTH+c];
	  unsigned char uR = triple_pix.range(3*BITS_PER_INP_PIXEL-1, 2*BITS_PER_INP_PIXEL);
	  unsigned char uG = triple_pix.range(2*BITS_PER_INP_PIXEL-1,   BITS_PER_INP_PIXEL);
	  unsigned char uB = triple_pix.range(  BITS_PER_INP_PIXEL-1,                    0);

	  // normalize
	  float R1 = uR;
	  float R2 = R1*norm;
	  R2 = R2 - shift_fact;
	  float G1 = uG;
	  float G2 = G1*norm;
	  G2 = G2 - shift_fact;
	  float B1 = uB;
	  float B2 = B1*norm;
	  B2 = B2 - shift_fact;
	  //scale
	  float R3 = gain * R2;
	  float G3 = gain * G2;
	  float B3 = gain * B2;
	  //clip
	  if (R3<-128) R3 = -128; if (R3> 127) R3 = 127;
	  if (G3<-128) G3 = -128; if (G3> 127) G3 = 127;
	  if (B3<-128) B3 = -128; if (B3> 127) B3 = 127;
	  //round 
	  int iR = (int) R3;
	  int iG = (int) G3;
	  int iB = (int) B3;

	  Dat3_t triple_data;
	  triple_data.range(3*BITS_PER_OUT_PIXEL-1, 2*BITS_PER_OUT_PIXEL) = iR;
	  triple_data.range(2*BITS_PER_OUT_PIXEL-1,   BITS_PER_OUT_PIXEL) = iG;
	  triple_data.range(  BITS_PER_OUT_PIXEL-1,                    0) = iB;
	  img_out[r*PRE_MAX_WIDTH+c] = triple_data;
	}
    }
}

ap_int<BITS_PER_OUT_PIXEL> colorconv(ap_uint<BITS_PER_INP_PIXEL> uR, 
				     float norm, float shift_fact, float gain)
{
  float R1 = uR;
  // normalize
  float R2 = R1*norm;
  float R2s = R2 - shift_fact;
  //scale
  float R3 = gain * R2s;
  //clip
  if (R3<-128) R3 = -128; if (R3> 127) R3 = 127;
  //round 
  return (int) R3;
}

extern "C"
void hls_dpupreproc_m_axi(m_axi_input_word *img_inp, m_axi_output_word *img_out, 
			  float norm_fact, float shift_fact, float scale_fact,
			  unsigned short int height, unsigned short int width)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem_in  depth=PRE_hls_IMGSZ
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem_out depth=PRE_hls_IMGSZ
  // VV
#pragma HLS INTERFACE s_axilite port=img_inp bundle=control
#pragma HLS INTERFACE s_axilite port=img_out  bundle=control
  // ^^
#pragma HLS INTERFACE s_axilite port=scale_fact bundle=control
#pragma HLS INTERFACE s_axilite port=norm_fact  bundle=control
#pragma HLS INTERFACE s_axilite port=shift_fact bundle=control
#pragma HLS INTERFACE s_axilite port=height   bundle=control
#pragma HLS INTERFACE s_axilite port=width    bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

  float my_gain_scale = scale_fact;
  float my_norm = 1.0f / norm_fact;
  float my_shift = shift_fact;

  unsigned short int rows = height;
  unsigned short int cols = width;

  // we read pixels from an AXI interface but they have a different bitwidth:
  // [] the AXI interface of bitwidth M_AXI_BITWIDTH, corresponds to
  //    VECTORIZATION_FACTOR colors
  // [] we need to read height * width pixels of 3 color components
  // -> so we need to read height * width * 3 color components divided by VECTORIZATION_FACTOR
  unsigned int words_to_read = rows * cols * 3 / VECTORIZATION_FACTOR; 

  printf("DPU PreProc VHLS HW-centric version.... !!!\n");
 L1:
  for (int index = 0; index < words_to_read; index++) {
#pragma HLS LOOP_TRIPCOUNT max=PRE_hls_trip_count_loop min=1
#pragma HLS PIPELINE II=1
    m_axi_input_word word_read=img_inp[index];
    m_axi_output_word word_to_write;

    // will be unrolled due to parent's pipeline directive
  will_be_unrolled:
    for(int u = 0 ; u < VECTORIZATION_FACTOR ; u++) {
      word_to_write[u] = colorconv( word_read[u], my_norm, my_shift, my_gain_scale);
      //            for tests only, injecting an error to make sure it fails and see if the TB detects it
      //            if(index==3 && u==1) { 
      //                word_to_write[u] *= (-1);
      //                printf(" **error inserted for index=%d & u=%d** ",index,u);
      //            }
      //            debug printf
      //            if(index<10) {
      //                printf(" hls [%d][%d] = %d -> %d\n", index, u, (int)word_read[u], (int)word_to_write[u]);
      //            }
    }
    img_out[index] = word_to_write;

  }
}

