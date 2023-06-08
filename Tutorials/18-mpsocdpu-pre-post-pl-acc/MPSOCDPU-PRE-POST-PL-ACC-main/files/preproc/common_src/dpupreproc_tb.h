/*

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/

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
