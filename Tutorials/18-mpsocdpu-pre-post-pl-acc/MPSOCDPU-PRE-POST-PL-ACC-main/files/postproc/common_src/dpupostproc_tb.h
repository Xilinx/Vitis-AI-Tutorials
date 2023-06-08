/*
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/


#ifndef _H_DPUPOSTPROC_TB_H_

#define _H_DPUPOSTPROC_TB_H_


#include "dpupostproc_defines.h"


void Write_Bin_File_as_Schar(char *filename, int dim, signed char *buffer);

void Write_Bin_File_as_Uchar(char *filename, int dim, unsigned char *buffer);

void Write_Bin_File_as_float(char *filename, int dim, float *buffer);

void Read_Bin_File_of_Schar(char *filename, int dim, signed char *buffer);

void Read_Bin_File_of_Uchar(char *filename, int dim, unsigned char *buffer);

void Read_Float_Txt_File(char *filename, int H, int W, float *data);

void Read_Byte_Txt_File(char *filename, int H, int W, char *data);

void Write_SoftMax_Txt_File(char *filename, int H, int W, float *data);

float check_output_max(unsigned char *ref_max, unsigned char *hls_max, unsigned short int height, unsigned short int width);

float check_output_index(unsigned char *ref_index, unsigned char *hls_index, unsigned short int height, unsigned short int width);


#ifndef ARM_HOST


float check_output_softmax(float *ref, float *hls, unsigned short int height, unsigned short int width);

float check_input_data(signed char *inp1, signed char *inp2, unsigned short int height, unsigned short int width);

void char_to_hlsvector(signed char input[POST_hls_MAXSZ], m_axi_input_word hlsvector[POST_hls_MAXSZWORDS]);

int POST_tb_main(int argc, char **argv,   int output_fixpos);


#endif // ARM_HOST
#endif
