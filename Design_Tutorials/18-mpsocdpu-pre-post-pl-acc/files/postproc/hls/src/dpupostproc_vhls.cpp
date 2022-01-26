/*************************************************************************************
Vendor:					Xilinx
Associated Filename:	dpupostproc_vhls.cpp
Purpose:				Vitis HLS DPU pre-processor
Revision History:		10 Nov 2021
authors:				danieleb@xilinx.com, herver@xilinx.com

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

#include <assert.h>
#include <hls_stream.h>

#include "dpupostproc_defines.h"

#include "luts.h"   // LUTs for all the 7 possible scaling factors of ANY CNN



/* ******************************************************************************* */

void hls_Arg_Soft_Max(signed char *inp_data, unsigned char *out_max, float *out_data, unsigned char *out_index, unsigned char size, int output_fixpos)
{
#pragma HLS INLINE
#pragma HLS PIPELINE
	float result[MAX_NUM_OF_CLASSES];
	float sum = 0.0f;
	unsigned char max=0;
	unsigned char index=0;
	L1:for (int i=0; i<size; i++)
	{
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_CLASS min=POST_hls_MIN_CLASS

		int byte = inp_data[i];
		byte = 128 + byte;
		assert( (byte<=255) & (byte>=0) );

		float val = LUT_EXP[output_fixpos-1][byte]; // already includes the multiplication per the scale_factor
		result[i] = val;
		sum += val;
	}

	float div = 1.0f / sum;
	float val;
	int i_val;
	L2:for (int i=0; i<size; i++)
	{
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_CLASS min=POST_hls_MIN_CLASS
		val = result[i] * div;
		out_data[i]= val;
		val = val * 255.0f;
		i_val = (int) val;
		assert( (i_val<=255) & (i_val>=0) );
		unsigned char u_val = i_val;
		if (u_val > max) {
			max = u_val;
			index = i;
		}
	}
	*out_max = max;
	*out_index = index;

}



//#ifdef __SYNTHESIS__
// computes the maximum of NUM_OF_CLASSES softmax probabilites and output such value and its integer index
void hls_dpupostproc(signed char inp_data[POST_hls_MAXSZ],
		unsigned char out_max[POST_hls_IMGSZ], unsigned char out_index[POST_hls_IMGSZ],
		int output_fixpos, unsigned short int height, unsigned short int width)
//#else
// computes the maximum of NUM_OF_CLASSES softmax probabilites and output such value and its integer index
//void hls_dpupostproc(char inp_data[POST_hls_MAXSZ], float out_softmax[POST_hls_IMGSZ],
//		unsigned char out_max[POST_hls_IMGSZ], unsigned char out_index[POST_hls_IMGSZ],
//		     int output_fixpos, unsigned short int height, unsigned short int width)
//#endif
{

#pragma HLS INTERFACE s_axilite port=height        bundle=control
#pragma HLS INTERFACE s_axilite port=width         bundle=control
#pragma HLS INTERFACE s_axilite port=output_fixpos bundle=control
#pragma HLS INTERFACE s_axilite port=return        bundle=control


	unsigned short int size = MAX_NUM_OF_CLASSES;

	unsigned short int rows = height;
	unsigned short int cols = width;

	float softmax[MAX_NUM_OF_CLASSES];
	//float vect[MAX_NUM_OF_CLASSES];
	signed char ch_vect[MAX_NUM_OF_CLASSES];
	unsigned char index, max;

	assert( (output_fixpos<=7) & (output_fixpos>=1) );

	L1:for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_H
		L2:for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_W
			L3:for(int cl=0; cl<size; cl++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_CLASS min=POST_hls_MIN_CLASS

				signed char  tmp_data  = inp_data[r*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES + c*MAX_NUM_OF_CLASSES + cl];
				//float tmp_float = scale_factor * tmp_data; //for this app scale_factor = 0.5f
				//vect[cl] =  tmp_float;
				ch_vect[cl] =  tmp_data;
			}
			hls_Arg_Soft_Max(ch_vect, &max, softmax, &index, size, output_fixpos);

			// store results
			out_max[r*POST_MAX_WIDTH + c] = max;
			out_index[r*POST_MAX_WIDTH + c] = index;
			//#ifndef __SYNTHESIS__
			//      L4:for(int cl=0; cl<size; cl++)
			//      {
			//    	  //#pragma HLS PIPELINE
			//    	  //#pragma HLS LOOP_TRIPCOUNT max=hls_MAX_CLASS min=POST_hls_MIN_CLASS
			//    	  out_softmax[r*POST_MAX_WIDTH*POST_MAX_NUM_OF_CLASSES + c*POST_MAX_NUM_OF_CLASSES + cl] = softmax[cl];
			//      }
			//#endif

		}
	}
} // end of fucntion

#define CAT3(A,B,C) A##B##C
#define L0(x,line) CAT3(x,,line)
#define LABEL(x) L0(CAT3(L,x,_line),__LINE__)

void hls_Arg_Soft_Max_two(soft_max_input_t inp_data, int dpu_fixpos, unsigned char *out_max, unsigned char *out_index, unsigned char size)
{
	//    #pragma HLS INLINE
	//    #pragma HLS PIPELINE II=1
	float result[MAX_NUM_OF_CLASSES];
	float sum = 0.0f;
	unsigned char max_value=0;
	unsigned char index_of_max=0;
	L1:
	for (int i=0; i<size; i++) {
		//        #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_CLASS min=POST_hls_MIN_CLASS

		int byte = inp_data[i];
		byte = 128 + byte;
		assert( (byte<=255) & (byte>=0) );

		float val = LUT_EXP[dpu_fixpos-1][byte]; // already includes the multiplication per the scale_factor
		result[i] = val;
		sum += val;
	}

	float div = 255.0f / sum;
	L2:
	for (int i=0; i<size; i++) {
		//       #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_CLASS min=POST_hls_MIN_CLASS
		float val = result[i] * div;
		int i_val = (int) val;
		assert( (i_val<=255) & (i_val>=0) );
		if (i_val > max_value) {
			max_value = i_val;
			index_of_max = i;
		}
	}
	*out_max = max_value;
	*out_index = index_of_max;
}


void mm2s(
		m_axi_input_word* mm,
		hls::stream<m_axi_input_word> &s,
		unsigned short rows,
		unsigned short cols)
{

	unsigned int words_to_read_per_line = cols * MAX_NUM_OF_CLASSES / CLASSES_PER_INPUT_WORD;
	const unsigned int POST_hls_MAX_W_WORDS = POST_hls_MAX_W * MAX_NUM_OF_CLASSES / CLASSES_PER_INPUT_WORD;
	unsigned int row_index=0;
	LABEL(inputreadframe):
		for (int r = 0; r < rows; r++) {
			//fprintf(stderr, "r = %3d\n", r);
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_H
			LABEL(inputreadoneline):
				for (int i = 0; i < words_to_read_per_line; i++) {
					//fprintf(stderr, "%3d ", i);
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_W_WORDS
#pragma HLS PIPELINE II=1
					s.write(mm[row_index+i]);
				}
			//fprintf(stderr, "\n");
			row_index+=POST_MAX_WIDTH*MAX_NUM_OF_CLASSES/CLASSES_PER_INPUT_WORD;
		}
}

// converts a stream of 32 bytes down to MAX_NUM_OF_CLASSES=28 bytes.
// this is for now hardcoded and the implementation cannot (without changes)
// accomodate anything else than 28.
// because 28=4*7 and 32=8*4 this means that 7 reads of 32 bytes
// will generate 8 values of 28 bytes.
// so we'll count the output pixels modulo 8 :
//   - for the 7 first cycles we'll read 32 bytes and
//   - generate 28 bytes over 8 cycles using the leftover from the previous cycle.

//DB4Nov21: need to change this into 12=4*3 and  16=4*4, so that 3 reads of 16 bytes will generate 4 values of 12 bytes.

// WARNING: this code works only in the case of POST_NUM_READS*CLASSES_PER_INPUT_WORD==POST_NUM_VALUES*MAX_NUM_OF_CLASSES
// and it was tested for the two cases: 7*32==8*28 and 3*16==4*12. FOR ANY FURTHER DIFFERENT CASE YOU NEED TO CHANGE THE CODE ACCORDINGLY


void shrinkstreambitwidth(
		hls::stream<m_axi_input_word> &larger,
		hls::stream<soft_max_input_t> &narrower,
		unsigned short rows,
		unsigned short cols,
		unsigned char num_values,
		unsigned char num_reads)
{
	unsigned int pixels_to_process = rows * cols;
	LABEL(resizedatapath):
		for (unsigned int j = 0; j < pixels_to_process; j++)
		{
			//fprintf(stderr, "%5d ", j);
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_IMGSZ
#pragma HLS PIPELINE II=1
			int cycle = j % num_values;
			m_axi_input_word data;
			soft_max_input_t out;
			//    static hls::vector<ap_int<BITS_PER_CLASS>,MAX_NUM_OF_CLASSES*2> remaining;
			static soft_max_input_t remaining;
			if (cycle<num_reads)
			{ // reads cycles 0-6
				data = larger.read();
			}

			switch (cycle)
			{
				// we have created this macro to consume N bytes from
				// previous remaining values & 28-N from new, new remaining is 32-28+N
				// so for example:
				// cycle 0 N= 0 : consume  0 from previous remaining & 28 new, remaining  4
				// cycle 1 N= 4 : consume  4 from previous remaining & 24 new, remaining  8
				// cycle 2 N= 8 : consume  8 from previous remaining & 20 new, remaining 12
				// cycle 3 N=12 : consume 12 from previous remaining & 16 new, remaining 16
				// etc
				// cycle 7: there is enough data in the previous remaining to not read new data
#define CONSUMEFROM(N) { \
			LABEL(A): for(int i=0;i<N;i++) out[i]=remaining[i]; \
			LABEL(B): for(int i=N;i<MAX_NUM_OF_CLASSES;i++) out[i]=data[i-N]; \
			LABEL(C): for(int i=0;i<CLASSES_PER_INPUT_WORD-MAX_NUM_OF_CLASSES+N;i++) remaining[i]=data[i+MAX_NUM_OF_CLASSES-N]; \
			break; \
			}
				case 0:
					CONSUMEFROM(0)
				case 1:
					CONSUMEFROM(4)
				case 2:
					CONSUMEFROM(8)
				case 3:
					if (num_reads==7)
						CONSUMEFROM(12)
					if (num_reads==3)
					{   // consume only from remaining 12 & nothing new, no new remaining
						LABEL(D): for(int i=0;i<MAX_NUM_OF_CLASSES;i++) out[i]=remaining[i];
						break;
					}
				case 4:
					if (num_reads==7)
						CONSUMEFROM(16)
				case 5:
					if (num_reads==7)
						CONSUMEFROM(20)
				case 6:
					if (num_reads==7)
						CONSUMEFROM(24)
				case 7:
					if (num_reads==7)
					{   // consume only from remaining 28 & nothing new, no new remaining
					LABEL(D): for(int i=0;i<MAX_NUM_OF_CLASSES;i++) out[i]=remaining[i];
					break;
					}
			}
			narrower.write(out);
		}
}

void softmax2mm(
		hls::stream<soft_max_input_t> &resizetosoftmax,
		unsigned short size,
		int dpu_fixpos,
		unsigned char out_max[POST_hls_IMGSZ],
		unsigned char out_index[POST_hls_IMGSZ],
		unsigned short rows,
		unsigned short cols)
{
#define MANUAL_LOOP_MERGE
#ifndef MANUAL_LOOP_MERGE
	L1:
	for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_H
		L2:
		for (int c = 0; c < cols; c++) {
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_MAX_W
#else
			unsigned int pixels_to_process = rows * cols;
			LABEL(flat):
				for (unsigned int j = 0, r=0, c=0; j < pixels_to_process; j++) {
#pragma HLS LOOP_TRIPCOUNT max=POST_hls_IMGSZ

#endif
#pragma HLS PIPELINE II=1
					soft_max_input_t read_ch_vect=resizetosoftmax.read();
					unsigned char index, max;
					hls_Arg_Soft_Max_two(read_ch_vect, dpu_fixpos, &max, &index, size);

					// store results
					out_max[r*POST_MAX_WIDTH + c] = max;
					out_index[r*POST_MAX_WIDTH + c] = index;
#ifndef MANUAL_LOOP_MERGE
				}
		}
#else
		if(c==cols-1) {
			r++;
			c=0;
		} else {
			c++;
		}
	}
#endif
}



extern "C"
void hls_dpupostproc_m_axi(m_axi_input_word *inp_data,
		unsigned char out_max[POST_hls_IMGSZ], unsigned char out_index[POST_hls_IMGSZ],
		int dpu_fixpos, unsigned short int height, unsigned short int width)
{

#pragma HLS INTERFACE m_axi     port=inp_data  offset=slave bundle=gmem_in depth=POST_hls_MAXSZWORDS //num of words accesses on the bus
#pragma HLS INTERFACE s_axilite port=inp_data bundle=control

#pragma HLS INTERFACE m_axi     port=out_max  offset=slave bundle=gmem_out_max depth=POST_hls_IMGSZ
#pragma HLS INTERFACE s_axilite port=out_max  bundle=control

#pragma HLS INTERFACE m_axi     port=out_index  offset=slave bundle=gmem_out_index depth=POST_hls_IMGSZ
#pragma HLS INTERFACE s_axilite port=out_index  bundle=control

#pragma HLS INTERFACE s_axilite port=height            bundle=control
#pragma HLS INTERFACE s_axilite port=width             bundle=control
#pragma HLS INTERFACE s_axilite port=dpu_fixpos        bundle=control
#pragma HLS INTERFACE s_axilite port=return            bundle=control

#pragma HLS DATAFLOW

	unsigned short int size = MAX_NUM_OF_CLASSES;
	unsigned short int rows = height;
	unsigned short int cols = width;
	unsigned char num_values = POST_NUM_VALUES;
	unsigned char num_reads  = POST_NUM_READS;

	assert( (dpu_fixpos<=7) & (dpu_fixpos>=1) );

	hls::stream<m_axi_input_word> inputtoresize("fifo m_axi reader to resizer");
#pragma HLS STREAM VARIABLE=inputtoresize DEPTH=HLS_STREAM_DEPTH

	hls::stream<soft_max_input_t> resizetosoftmax("fifo resizer to softmax");
#pragma HLS STREAM VARIABLE=resizetosoftmax DEPTH=HLS_STREAM_DEPTH

	mm2s(inp_data, inputtoresize,rows,cols);

	shrinkstreambitwidth(inputtoresize,resizetosoftmax,rows,cols, num_values, num_reads);

	softmax2mm( resizetosoftmax, size, dpu_fixpos, out_max, out_index, rows, cols);
}
