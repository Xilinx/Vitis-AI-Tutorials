/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpupreproc_vhls.cpp
Purpose:		Vitis HLS DPU pre-processor
Revision History:	5Aug June 2021
authors:		daniele.bagni@xilinx.com, herve.ratigner@xilinx.com

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

#include "dpupreproc_defines.h"

// takes an RGB 24bit per pixel image (R, G, B each one of 8-bit) from a source (what?) and store it
// into DDR as 3*8-bit unsigned long word

extern "C"
void hls_dpupreproc(uPix3_t *inp_img, Dat3_t *img_out,
		    float means_0,float means_1,float means_2, float scales_0, float scales_1,float scales_2, int dpu_fixpos,
		    unsigned short int height, unsigned short int width)
{

#pragma HLS INTERFACE axis register_mode=both register depth=PRE_hls_IMGSZ port=inp_img
#pragma HLS INTERFACE axis register_mode=both register depth=PRE_hls_IMGSZ port=img_out

#pragma HLS INTERFACE s_axilite port=scales_0     bundle=control
#pragma HLS INTERFACE s_axilite port=means_0      bundle=control
#pragma HLS INTERFACE s_axilite port=scales_1     bundle=control
#pragma HLS INTERFACE s_axilite port=means_1      bundle=control
#pragma HLS INTERFACE s_axilite port=scales_2     bundle=control
#pragma HLS INTERFACE s_axilite port=means_2      bundle=control
#pragma HLS INTERFACE s_axilite port=dpu_fixpos bundle=control
#pragma HLS INTERFACE s_axilite port=height     bundle=control
#pragma HLS INTERFACE s_axilite port=width      bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

  // set the correct gain factor depending on the fixpos value	from the DPU input tensor
  float integ_dpu_scales[8] = {1.0, 2.0, 4.00, 8.000, 16.0000, 32.00000, 64.000000, 128.0000000};
  float fract_dpu_scales[8] = {1.0, 0.5, 0.25, 0.125,  0.0625,  0.03125,  0.015625,   0.0078125};
  float gain = (dpu_fixpos>0) ? integ_dpu_scales[dpu_fixpos] : fract_dpu_scales[dpu_fixpos];

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
	  float R2 = R1 - means_0;
	  R2 = R2 * scales_0;
	  float G1 = uG;
	  float G2 = G1 - means_1;
	  G2 = G2 * scales_1;
	  float B1 = uB;
	  float B2 = B1 - means_2;
	  B2 = B2 * scales_2;
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
				     float pre_means, float pre_scales, float dpu_gain)
{
  float R1 = uR;
  // normalize
  float R2  = R1 - pre_means;
  float R2s = R2 * pre_scales;
  //DPU scale
  float R3 = dpu_gain * R2s;
  //clip
  if (R3<-128) R3 = -128; if (R3> 127) R3 = 127;
  //round
  return (int) R3;
}

extern "C"
void hls_dpupreproc_m_axi(m_axi_input_word *img_inp, m_axi_output_word *img_out,
	    float means_0,float means_1,float means_2, float scales_0, float scales_1,float scales_2, int dpu_fixpos,
			  unsigned short int height, unsigned short int width)
{
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem_in  depth=PRE_hls_IMGSZ_V
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem_out depth=PRE_hls_IMGSZ_V
  // VV
#pragma HLS INTERFACE s_axilite port=img_inp    bundle=control
#pragma HLS INTERFACE s_axilite port=img_out    bundle=control
  // ^^
#pragma HLS INTERFACE s_axilite port=scales_0     bundle=control
#pragma HLS INTERFACE s_axilite port=means_0      bundle=control
#pragma HLS INTERFACE s_axilite port=scales_1     bundle=control
#pragma HLS INTERFACE s_axilite port=means_1      bundle=control
#pragma HLS INTERFACE s_axilite port=scales_2     bundle=control
#pragma HLS INTERFACE s_axilite port=means_2      bundle=control

#pragma HLS INTERFACE s_axilite port=dpu_fixpos bundle=control
#pragma HLS INTERFACE s_axilite port=height     bundle=control
#pragma HLS INTERFACE s_axilite port=width      bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    #if 0
  // set the correct gain factor depending on the fixpos value	from the DPU input tensor
  float integ_dpu_scales[8] = {1.0, 2.0, 4.00, 8.000, 16.0000, 32.00000, 64.000000, 128.0000000};
  float fract_dpu_scales[8] = {1.0, 0.5, 0.25, 0.125,  0.0625,  0.03125,  0.015625,   0.0078125};
  
  float gain = (dpu_fixpos>0) ? integ_dpu_scales[dpu_fixpos] : fract_dpu_scales[-dpu_fixpos]; 
  // ^ negative sign because that's the situation when the index is negative so needs to negate it..
#else
  // set the correct gain factor depending on the fixpos value	from the DPU input tensor
   
  // generates a float similar to 2 to the power of n or 1<<n, n positive or negative
  #define POW2(n) (n>=0?1.*(1<<n):1./(1<<-n))
  // using signed index as unsigned index means that the indexes are flipped around
  //
  // -1 == 4'b1111 is interpreted as 15 on 4 bits
  // -2 == 4'b1110 is interpreted as 14 on 4 bits
  // -3 == 4'b1101 is 13
  // ..
  // -8 == 4'b1000 is 8
  ap_uint<4> dpu_fixpos_index=dpu_fixpos; 
  float dpu_scales[16] = {
      // $ for i in `seq -8 7`; do printf "SG(%2d)," $i ; done
      POW2( 0),POW2( 1),POW2( 2),POW2( 3),POW2( 4),POW2( 5),POW2( 6),POW2( 7),
      POW2(-8),POW2(-7),POW2(-6),POW2(-5),POW2(-4),POW2(-3),POW2(-2),POW2(-1)
  };
  float gain = dpu_scales[dpu_fixpos_index];
  #endif

  unsigned short int rows = height;
  unsigned short int cols = width;

  // we read pixels from an AXI interface but they have a different bitwidth:
  // [] the AXI interface of bitwidth M_AXI_BITWIDTH, corresponds to
  //    VECTORIZATION_FACTOR colors
  // [] we need to read height * width pixels of 3 color components
  // -> so we need to read height * width * 3 color components divided by VECTORIZATION_FACTOR
  unsigned int words_to_read = rows * cols * 3 / VECTORIZATION_FACTOR;

  float means_lane0 = means_0;
  float means_lane1 = means_1;
  float means_lane2 = means_2;
  float scales_lane0 = scales_0;
  float scales_lane1 = scales_1;
  float scales_lane2 = scales_2;

  printf("DPU PreProc VHLS HW-centric version.... !!!\n");
 L1:
  for (int index = 0; index < words_to_read; index++) {
#pragma HLS LOOP_TRIPCOUNT max=PRE_hls_trip_count_loop min=1
#pragma HLS PIPELINE II=1
    m_axi_input_word word_read=img_inp[index];
    m_axi_output_word word_to_write;

    #if 0
    // previous code. 
    // will be unrolled due to parent's pipeline directive
  will_be_unrolled:
    for(int u = 0 ; u < VECTORIZATION_FACTOR ; u++) {

	//WARNING: THE CODE HAS TO BE CHANGED HERE. AT THE MOMENT IT DOES NOT ALLOW CAFFE-LIKE preprocessing
    // EACH COLORED PIXEL MUST BE PROCESSED BY ITS OWN MEAN AND SCALE, ONLY THE GAIN IS CONSTANT
	//TO BE UPDATE AFTER MY VACATIONS (DANIELE BAGNI)

      word_to_write[u] = colorconv( word_read[u], means_0, scales_0, gain);
    }
    #else

    // the new code is a bit more complex and assumes (for now) that 
    // VECTORIZATION_FACTOR is 4 and we put a compile-time bomb in the 
    // relevant header.
    //
    // the indexes correspond to the "channels"
    // first aligned access R,G,B, R or any subsequent aligned access
    //        second access G,B, R,G
    //         third access B, R,G,B
    // when looking by columns :
    // 1st column is getting the parameters from "lane0"
    // 2nd column is getting the parameters from "lane1"
    // 3rd column is getting the parameters from "lane2"
    // 4th column is getting the parameters from the same as lane0
    // when looking by lines, i.e. this is in terms of accesses:
    // for the next access, the lanes are shifted to the left, lane(i) takes lane(i+1)
      word_to_write[0] = colorconv( word_read[0], means_lane0, scales_lane0, gain);
      word_to_write[1] = colorconv( word_read[1], means_lane1, scales_lane1, gain);
      word_to_write[2] = colorconv( word_read[2], means_lane2, scales_lane2, gain);
      word_to_write[3] = colorconv( word_read[3], means_lane0, scales_lane0, gain);
      
      // rotate the coefficients for the next access
      float means_saved;
      means_saved = means_lane0;
      means_lane0 = means_lane1;
      means_lane1 = means_lane2;
      means_lane2 = means_saved;

      float scales_saved;
      scales_saved = scales_lane0;
      scales_lane0 = scales_lane1;
      scales_lane1 = scales_lane2;
      scales_lane2 = scales_saved;
    #endif
    img_out[index] = word_to_write;

  }
}
