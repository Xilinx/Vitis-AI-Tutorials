/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpupreproc_defines.h
Purpose:		header file with all the project defines
Revision History:	23 July 2021
author:			daniele.bagni@xilinx.com   herver@xilinx.com

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

#ifndef _H_DPUPREPROC_H_
#define _H_DPUPREPROC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>

//#define ARM_HOST

//#define DEBUG_TRANSPARENT


/* ******************************************************************************* */
// CONSTANT PARAMETERS for architectural choices

#define MAX_PATH      1000
#define PRE_MAX_HEIGHT     832  //((1080+8)/2)  // number of lines per image
#define PRE_MAX_WIDTH     1920  //(1920/2)      // number of pixels per line

const int BITS_PER_INP_PIXEL  =8;
const int BITS_PER_OUT_PIXEL  =8;

// the bitwidth of the master axi interface for reading and writing the data
const int PRE_M_AXI_BITWIDTH = 32;
// compile time error checking that input and output are the same bitwidth
const int input_and_output_bitwidth_dont_match[(BITS_PER_INP_PIXEL==BITS_PER_OUT_PIXEL)?1:-1]={0};
const int VECTORIZATION_FACTOR = PRE_M_AXI_BITWIDTH/BITS_PER_INP_PIXEL;


// preprocessing normalization factors
#ifndef DEBUG_TRANSPARENT //normal mode
#define NORM_FACT   (0.007843137) // which is 1.0f/127.5f
#define SHIFT_FACT  (127.5)
#define SCALE_FACT       64
#else
#define NORM_FACT   (1.0)
#define SHIFT_FACT  (0.0)
#define SCALE_FACT  (1.0)
#endif

#define PRE_FIXPOS 6


const float PRE_MEANS[ 3]={SHIFT_FACT,SHIFT_FACT,SHIFT_FACT};
const float PRE_SCALES[3]={NORM_FACT,  NORM_FACT, NORM_FACT};



/* ******************************************************************************* */
// parameters depending on the above ones (do not touch)

const int BITS_PER_INP_3PIXEL =3*BITS_PER_INP_PIXEL;
const int BITS_PER_OUT_3PIXEL =3*BITS_PER_OUT_PIXEL;

const int PRE_hls_IMGSZ   = (PRE_MAX_HEIGHT)*(PRE_MAX_WIDTH);
const int PRE_hls_IMGSZ_V = ((PRE_MAX_HEIGHT)*(PRE_MAX_WIDTH)*3)/VECTORIZATION_FACTOR;
const int PRE_hls_MIN_H = 1;                      //for HLS pragmas LOOP TRIPCOUNT
const int PRE_hls_MIN_W = 1;                      //for HLS pragmas LOOP TRIPCOUNT
const int PRE_hls_MAX_H = (PRE_MAX_HEIGHT);             //for HLS pragmas LOOP TRIPCOUNT
const int PRE_hls_MAX_W = (PRE_MAX_WIDTH);              //for HLS pragmas LOOP TRIPCOUNT
const int PRE_hls_trip_count_loop = PRE_hls_IMGSZ_V;



/* ******************************************************************************* */
// PARAMETERS for algorithm choices
const float  PRE_REF_THRESHOLD = (1e-6f);


/* ******************************************************************************* */
// I/O Image Settings
#define PRE_INPUT_IMAGE		"./data_pre/testing_0_1920x832"

/* ******************************************************************************* */
// SMALL FUNCTIONS IN MACRO
#define ABSDIFF(x,y)	( (x)>(y) ? (x - y) : (y - x) )
#define ABS(x)          ( (x)> 0  ? (x)     : -(x)    )
#define MIN(x,y)        ( (x)>(y) ? (y)     :  (x)    )
#define MAX(x,y)        ( (x)>(y) ? (x)     :  (y)    )


/* ******************************************************************************* */
// DATA TYPES

typedef struct u_rgb {
    unsigned char  B;
    unsigned char  G;
    unsigned char  R;
  } uRGB_t;

typedef struct i_rgb  {
    signed char B;
    signed char G;
    signed char R;
  } iRGB_t;

typedef union u2i_conv {
	unsigned long int i;
	  signed long int o;
} union_u2i_conv_t;


typedef union i2u_conv {
	  signed long int i;
	unsigned long int o;
} union_i2u_conv_t;




#ifndef ARM_HOST



/* ******************************************************************************* */
// HLS DATA TYPES

#include <ap_int.h> // HLS arbitrary width integer data types
#include <hls_vector.h> // support for hls::vector<T,N> class

typedef ap_uint<BITS_PER_INP_PIXEL >    uPix_t;      // input pixel
typedef hls::vector<ap_int< BITS_PER_INP_PIXEL>,VECTORIZATION_FACTOR> m_axi_output_word; // one word that we read from M_AXI is a vector of N uPix_t
typedef hls::vector<ap_uint<BITS_PER_INP_PIXEL>,VECTORIZATION_FACTOR> m_axi_input_word;  // one word that we read from M_AXI is a vector of N uPix_t
typedef ap_uint<BITS_PER_INP_3PIXEL>    uPix3_t;     // to pack 3 inp pixels into a single larger word
typedef  ap_int<BITS_PER_OUT_3PIXEL>     Dat3_t;     // to pack 3 out pixels into a single larger word


/* ******************************************************************************* */
// FUNCTION PROTOTYPES
void ref_dpupreproc(uRGB_t *inp_img, iRGB_t *out_img,
		    float means[3], float scales[3], int dpu_fixpos,
		    unsigned short int height, unsigned short int width);

float check_output_data( Dat3_t *hls_data, iRGB_t *ref_data, unsigned short int height, unsigned short int width);


extern "C"
void hls_dpupreproc(uPix3_t     *inp_img,  Dat3_t *img_out,
		//float norm_fact, float shift_fact, float scale_fact,
	    float means_0,float means_1,float means_2, float scales_0, float scales_1,float scales_2, int dpu_fixpos,
		unsigned short int height, unsigned short int width);

extern "C"
void hls_dpupreproc_m_axi(m_axi_input_word *inp_img, m_axi_output_word *img_out,
		//float norm_fact, float shift_fact, float scale_fact,
	    float means_0,float means_1,float means_2, float scales_0, float scales_1,float scales_2, int dpu_fixpos,
		unsigned short int height, unsigned short int width);

#else

void arm_ref_dpupreproc(unsigned char *R, unsigned char *G, unsigned char *B,
		signed char *out_rgb_img, float means[3], float scales[3], int dpu_fixpos,
		unsigned short int height, unsigned short int width);


void format_ref_img(unsigned char *R, unsigned char *G, unsigned char *B,
		signed char *out_rgb_img, unsigned short int height, unsigned short int width);




#endif //ARM_HOST


/* ******************************************************************************* */


#endif //_H_DPUPREPROC_H_
