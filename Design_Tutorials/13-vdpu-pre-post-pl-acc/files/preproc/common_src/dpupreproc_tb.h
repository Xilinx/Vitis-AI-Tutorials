/*************************************************************************************
 Vendor:		Xilinx
Associated Filename:	dpupreproc_tb.h
Purpose:		Vitis HLS DPU pre-processor
Revision History:	23 July 2021
authors:		daniele.bagni@xilinx.com

**************************************************************************************

 Copyright 20201Xilinx Inc.

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

#ifndef _H_DPUPREPROC_TB_H_

#define _H_DPUPREPROC_TB_H_

#include "dpupreproc_defines.h"


void Write_Bin_File_of_Schar(char *filename, int dim, signed char *buffer);

void Read_Bin_File_of_Schar(char *filename, int dim, char *buffer);


#ifndef ARM_HOST

void format_input_data(unsigned char *R, unsigned char *G, unsigned char *B,
		       uRGB_t *rgb_data, uPix3_t *hls_data, unsigned short int height,
		       unsigned short int width);

float check_input_data(uPix3_t *hls_data, uRGB_t *ref_data,
		       unsigned short int height, unsigned short int width);


float check_output_data(Dat3_t *hls_data, iRGB_t *ref_data,
			unsigned short int height, unsigned short int width);

float check_output_data_hls(m_axi_output_word *hls_data, iRGB_t *ref_data,
			    unsigned short int height, unsigned short int width);

void RGB_to_hlsvector(unsigned char RGB[PRE_MAX_HEIGHT*PRE_MAX_WIDTH], m_axi_input_word hlsvector[PRE_MAX_HEIGHT*PRE_MAX_WIDTH*3/VECTORIZATION_FACTOR]);


int PRE_tb_main(int argc, char **argv, float pre_means[3], float pre_scales[3], int dpu_input_fixpos);


#endif //#ifndef ARM_HOST

#endif //#define _H_DPUPREPROC_TB_H_
