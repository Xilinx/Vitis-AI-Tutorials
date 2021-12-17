/*************************************************************************************
Vendor:			Xilinx
Associated Filename:	dpupostproc_defines.h
Purpose:		header file with all the project defines
Revision History:	23 July 2021
author:			daniele.bagni@xilinx.com herver@xilinx.com

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

#ifndef _H_DPUPOSTPROC_H_
#define _H_DPUPOSTPROC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>


#define ARM_HOST


/* ******************************************************************************* */
// CONSTANT PARAMETERS for architectural choices

#define MAX_PATH      1000
#define POST_MAX_HEIGHT     832/2   //((1080+8)/2)  // number of lines per image
#define POST_MAX_WIDTH     1920/2   //(1920/2)      // number of pixels per line

const int BITS_PER_CLASS = 8;


//post processing scaling factor
#define POSTPR_FIXPOS  2
#define POSTPR_SCALE   (-POSTPR_FIXPOS) // 2^{-2} = 0.25f


// amount of classes or objects
#define     NUM_OF_CLASSES  12 //my tutorial case
#define MAX_NUM_OF_CLASSES  28 //ZF case


// the bitwidth of the master axi interface for reading and writing the data
const int POST_M_AXI_BITWIDTH = 256;
// compile time error checking that input and output are the same bitwidth
const int CLASSES_PER_INPUT_WORD = POST_M_AXI_BITWIDTH/BITS_PER_CLASS; // 256/8=32
// with current parameters we have 28*8 <= 256 -> OKAY
// we want to error out if the defaults parameters are changing because it will fail
const int bitwidths_dont_fit[MAX_NUM_OF_CLASSES*BITS_PER_CLASS<=POST_M_AXI_BITWIDTH?1:-1]={0};


// map the segmented classes in colored RGB images
const unsigned char colorB[19] = {128, 232, 70, 156, 153, 153,  30,   0,  35, 152, 180,  60,   0, 142, 70, 100, 100, 230,  32};
const unsigned char colorG[19] = { 64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130,  20,   0,   0,  0,  60,  80,   0,  11};
const unsigned char colorR[19] = {128, 244, 70, 102, 190, 153, 250, 220, 107, 152,  70, 220, 255,   0,  0,   0,   0,   0, 119};



/* ******************************************************************************* */
// parameters depending on the above ones (do not touch)

const int POST_hls_IMGSZ = (POST_MAX_HEIGHT)*(POST_MAX_WIDTH); //for HLS pragmas FIFO interfaces
const int POST_hls_MIN_H = 1;                        //for HLS pragmas LOOP TRIPCOUNT
const int POST_hls_MIN_W = 1;                        //for HLS pragmas LOOP TRIPCOUNT
const int POST_hls_MAX_H = (POST_MAX_HEIGHT);             //for HLS pragmas LOOP TRIPCOUNT
const int POST_hls_MAX_W = (POST_MAX_WIDTH);              //for HLS pragmas LOOP TRIPCOUNT
const int POST_hls_MAX_CLASS = (MAX_NUM_OF_CLASSES); //for HLS pragmas LOOP TRIPCOUNT
const int POST_hls_MIN_CLASS = (    NUM_OF_CLASSES); //for HLS pragmas LOOP TRIPCOUNT
const int POST_hls_MAXSZ = POST_hls_IMGSZ*MAX_NUM_OF_CLASSES; //for the HLS pragmas
const int POST_hls_MAXSZWORDS = POST_hls_MAXSZ/CLASSES_PER_INPUT_WORD; //for the HLS pragmas

//const int HLS_STREAM_DEPTH = 10;                //for HLS pragmas STREAM




/* ******************************************************************************* */
// PARAMETERS for algorithm choices
const float  POST_REF_THRESHOLD = (1e-6f);


/* ******************************************************************************* */
// I/O Image Settings
//#define INPUT_IMAGE		"./data/testing_0"
#define INPUT_IMAGE		"./data_post/notscaled_input"

/* ******************************************************************************* */
// SMALL FUNCTIONS IN MACRO
#define ABSDIFF(x,y)	( (x)>(y) ? (x - y) : (y - x) )
#define ABS(x)          ( (x)> 0  ? (x)     : -(x)    )
#define MIN(x,y)        ( (x)>(y) ? (y)     :  (x)    )
#define MAX(x,y)        ( (x)>(y) ? (x)     :  (y)    )


/* ******************************************************************************* */
// FUNCTION PROTOTYPES

float PrepareScalingFactor(int number_of_bits);
void  PrepareExpLUT(float scale_factor);

void ref_dpupostproc(signed char *inp_data,
		     //float *out_softmax,
		     unsigned char *out_max, unsigned char *out_index,
		     int output_fixpos, unsigned short int height, unsigned short int width);


#ifndef ARM_HOST



#include <ap_int.h> // HLS arbitrary width integer data types
#include <hls_vector.h> // support for hls::vector<T,N> class
typedef hls::vector<char,CLASSES_PER_INPUT_WORD> m_axi_input_word;
typedef hls::vector<ap_int<BITS_PER_CLASS>,MAX_NUM_OF_CLASSES> soft_max_input_t;



void hls_dpupostproc(signed char inp_data[POST_hls_MAXSZ],
		unsigned char out_max[POST_hls_IMGSZ], unsigned char out_index[POST_hls_IMGSZ],
		int output_fixpos, unsigned short int height, unsigned short int width);

extern "C"
void hls_dpupostproc_m_axi(m_axi_input_word *inp_data,
        unsigned char out_max[POST_hls_IMGSZ], unsigned char out_index[POST_hls_IMGSZ],
             int output_fixpos, unsigned short int height, unsigned short int width);


#else //Host application

// put your stuff inside here, if needed

#endif




/* ******************************************************************************* */
// check macros

#if (BITS_PER_PIXEL > 16)
#error <MAX NUMBER OF BITS IS 16!!!!>
#endif


#endif //_H_DPUPOSTPROC_H_
