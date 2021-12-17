/*************************************************************************************
 Vendor:				Xilinx
 Associated Filename:	dpupostproc_tb.h
 Purpose:				Testbench file
 Revision History:		23 July 2016 - initial release
 author:				daniele.bagni@xilinx.com

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
