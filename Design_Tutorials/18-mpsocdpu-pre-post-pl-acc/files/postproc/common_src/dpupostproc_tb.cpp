/*************************************************************************************
 Vendor:		Xilinx
 Associated Filename:	dpupostproc_tb.cpp
 Purpose:		Testbench file
 Revision History:	15 Nov 2021
 author:		daniele.bagni@xilinx.com

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


#include "dpupostproc_tb.h"


// write signed char to a binary file
void Write_Bin_File_as_Schar(char *filename, int dim, signed char *buffer)
{
  FILE *fid = fopen(filename, "wb");

  // 2's complement
  for (int i = 0; i<dim; i++)
  {
	  signed char out_c;
	  signed char inp_c = buffer[i];
	  if (inp_c < 0 ) out_c = 256 + inp_c; //negative numbers go from 128 (i.e. -128) to 255 (i.e. -1);
	  else out_c = inp_c;
	  buffer[i]=out_c;
  }

  if(!fid)
	  printf("ERROR: could not open binary %s for writing\n",filename);
  else
	  fwrite(buffer, dim*sizeof(signed char), 1, fid);

  fclose(fid);
}

// write unsigned char to a binary file
void Write_Bin_File_as_Uchar(char *filename, int dim, unsigned char *buffer)
{
  FILE *fid = fopen(filename, "wb");

  if(!fid)
	  printf("ERROR: could not open binary %s for writing\n",filename);
  else
	  fwrite(buffer, dim*sizeof(unsigned char), 1, fid);

  fclose(fid);
}

// write float to a binary file
void Write_Bin_File_as_float(char *filename, int dim, float *buffer)
{
  FILE *fid = fopen(filename, "wb");

  if(!fid)
	  printf("ERROR: could not open binary %s for writing\n",filename);
  else
	  fwrite(buffer, dim*sizeof(float), 1, fid);

  fclose(fid);
}


// Read from a binary file of signed char
void Read_Bin_File_of_Schar(char *filename, int dim, signed char *buffer)
{
  FILE *fid = fopen(filename, "rb");
  if(!fid)
	  printf("ERROR: could not open binary %s for reading\n",filename);
  else
	  fread(buffer, dim*sizeof(signed char), 1, fid);

  // 2's complement
  for (int i = 0; i<dim; i++)
  {
	  signed char out_c;
	  signed char inp_c = buffer[i];
	  if (inp_c >=128 ) out_c = inp_c-256; //negative numbers go from -128 (i.e. 128) to -1 (i.e. 255);
	  else out_c = inp_c;
	  buffer[i]=out_c;
  }

  fclose(fid);
}



// Read from a binary file of unsigned char
void Read_Bin_File_of_Uchar(char *filename, int dim, unsigned char *buffer)
{
  FILE *fid = fopen(filename, "rb");
  if(!fid)
	  printf("ERROR: could not open binary %s for reading\n",filename);
  else
	  fread(buffer, dim*sizeof(unsigned char), 1, fid);

  fclose(fid);
}



// read from a text file of floating point data organized (ideally) into H lines of W columns and MAX_WIDTH stride
// each column contains NUM_OF_CLASSES data. In reality the files is stored sequentially, one line at a time with NUM_OF_CLASSES data.
// the memory buffer is larger than the data to be contained as MAX_NUM_OF_CLASSES >= NUM_OF_CLASSES and fuilled with zeros
void Read_Float_Txt_File(char *filename, int H, int W, float *data)
{
  FILE *fid = fopen(filename, "rt");
  if(!fid) printf("ERROR: could not open %s for reading\n",filename);

  unsigned short int y,x, c;
  float value;

  for (y = 0; y < H; y++)
  {
    for (x = 0; x < W; x++)
    {
      for (c = 0; c < NUM_OF_CLASSES; c++)
      {
    	  fscanf(fid, "%f ", &value);
    	  //printf( "%f ", (float) value);
    	  data[y*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES +x*MAX_NUM_OF_CLASSES +c] = value;
      }
      fscanf(fid, "\n");
      for (c = NUM_OF_CLASSES; c<MAX_NUM_OF_CLASSES; c++) //dummy data
      {
    	  data[y*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES +x*MAX_NUM_OF_CLASSES +c] = (float) 0;
      }
    }
  }
  fclose(fid);
}

// read from a text file of char data organized (ideally) into H lines of W columns and MAX_WIDTH stride
// each column contains NUM_OF_CLASSES data. In reality the files is stored sequentially, one line at a time with NUM_OF_CLASSES data.
// the memory buffer is larger than the data to be contained as MAX_NUM_OF_CLASSES >= NUM_OF_CLASSES and fuilled with zeros
void Read_Byte_Txt_File(char *filename, int H, int W, char *data)
{
  FILE *fid = fopen(filename, "rt");
  if(!fid) printf("ERROR: could not open %s for reading\n",filename);

  unsigned short int y,x, c;
  float value;

  for (y = 0; y < H; y++)
  {
    for (x = 0; x < W; x++)
    {
      for (c = 0; c < NUM_OF_CLASSES; c++)
      {
    	  fscanf(fid, "%f ", &value);
    	  //printf( "%f ", (float) value);
    	  data[y*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES +x*MAX_NUM_OF_CLASSES +c] = (char) value;
      }
      fscanf(fid, "\n");
      for (c = NUM_OF_CLASSES; c<MAX_NUM_OF_CLASSES; c++)
      {
    	  data[y*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES +x*MAX_NUM_OF_CLASSES +c] = 0;
      }
    }
  }
  fclose(fid);
}


// write to a text file of floating point data organized into H lines of W columns and MAX_WIDTH stride
template <typename T> void Write_Txt_File(char *filename, int H, int W, T *data)
{
  FILE *fid = fopen(filename, "wt");
  if(!fid) printf("ERROR: could not open %s for writing\n",filename);

  unsigned short int y,x;
  float value;

  for (y = 0; y < H; y++)
  {
    for (x = 0; x < W; x++)
    {
      value = (float) data[y*POST_MAX_WIDTH+x];
      fprintf(fid, "%20.10f ", value);
    }
    fprintf(fid, "\n");
  }
  fclose(fid);
}

// write to a text file of floating point data organized into H lines of W columns and MAX_WIDTH stride
void Write_SoftMax_Txt_File(char *filename, int H, int W, float *data)
{
  FILE *fid = fopen(filename, "wt");
  if(!fid) printf("ERROR: could not open %s for writing\n",filename);

  unsigned short int y, x, c;
  float value;

  for (y = 0; y < H; y++)
  {
    for (x = 0; x < W; x++)
    {
      for (c = 0; c < NUM_OF_CLASSES; c++)
      {
    	  value = data[y*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES +x*MAX_NUM_OF_CLASSES +c];
    	  fprintf(fid, "%20.10f ", value);
      }
      fprintf(fid, "\n");
    }
  }
  fclose(fid);
}

float check_output_max(unsigned char *ref_max, unsigned char *hls_max, unsigned short int height, unsigned short int width)
{

  int row = height;
  int col = width;
  float tot_diff = 0.0f;
  float diff, r_max, h_max;

  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {

      r_max = ref_max[row * POST_MAX_WIDTH + col];
      h_max = hls_max[row * POST_MAX_WIDTH + col];

      diff = r_max - h_max;
      diff = ABS(diff);
      if (diff > 1)
	{
    	  tot_diff +=diff; // if there was a mismatch this should not be higher than 1
    	  fprintf(stderr, "ERROR in MAX: got %4d expected %4d\n", (int) h_max, (int) r_max);
	}
    }
  }
  return tot_diff;
}

float check_output_index(unsigned char *ref_index, unsigned char *hls_index, unsigned short int height, unsigned short int width)
{

  int row = height;
  int col = width;
  int r_pix, h_pix;
  float diff;
  float tot_diff = 0.0f;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {

      r_pix = ref_index[row * POST_MAX_WIDTH + col];
      h_pix = hls_index[row * POST_MAX_WIDTH + col];

      diff = r_pix - h_pix;
      diff = ABS(diff);
      if (diff > POST_REF_THRESHOLD)
    	  tot_diff +=diff;

    }
  }
  return tot_diff;
}


#ifndef ARM_HOST

/*
float check_output_softmax(float *ref, float *hls, unsigned short int height, unsigned short int width)
{

  int row = height;
  int col = width;
  float r_pix, h_pix;
  float diff;
  float tot_diff = 0.0f;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
    	  for (int c = 0; c < MAX_NUM_OF_CLASSES; c++)
    	  {
      		  r_pix = ref[row*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES + col*MAX_NUM_OF_CLASSES +c];
    		  h_pix = hls[row*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES + col*MAX_NUM_OF_CLASSES +c];

    		  diff = r_pix - h_pix;
    		  diff = ABS(diff);
    		  if (diff > POST_REF_THRESHOLD)
    		  {
    			  tot_diff +=diff;
    	    	  fprintf(stderr, "ERROR in SOFTMAX: got %f expected %f\n", h_pix, r_pix);
    		  }
    	  }
    }
  }
  return tot_diff;
}
*/

float check_input_data(signed char *inp1, signed char *inp2, unsigned short int height, unsigned short int width)
{

  int row = height;
  int col = width;
  int r_pix, h_pix;
  float diff;
  float tot_diff = 0.0f;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {

        for (int c= 0; c < MAX_NUM_OF_CLASSES; c++)
        {
        	r_pix = inp1[row * POST_MAX_WIDTH*MAX_NUM_OF_CLASSES + col*MAX_NUM_OF_CLASSES + c];
        	h_pix = inp2[row * POST_MAX_WIDTH*MAX_NUM_OF_CLASSES + col*MAX_NUM_OF_CLASSES + c];
        	diff = r_pix - h_pix;
        	diff = ABS(diff);
        	if (diff > POST_REF_THRESHOLD) tot_diff += diff;
        }
    }
  }
  return tot_diff;
}

void char_to_hlsvector(signed char input[POST_hls_MAXSZ], m_axi_input_word hlsvector[POST_hls_MAXSZWORDS])
{
    for (int index = 0; index < POST_hls_MAXSZ; index++) {
//        if(index<10*3) printf(" input[%d] = %d \n", index, input[index]);
        hlsvector[index/CLASSES_PER_INPUT_WORD][index%CLASSES_PER_INPUT_WORD]=input[index];
    }
}

/* **************************************************************************************** */
/* **************************************************************************************** */
/* **************************************************************************************** */

extern float ref_LUT_EXP[256];

int POST_tb_main(int argc, char **argv, int output_fixpos)
{

  unsigned short int x, y;
  int width, height;
  char *tempbuf1, *tempbuf2;
  int check_results, ret_res = 0, status;
  float total_error;

  //Arrays to send and receive data from the accelerator
  signed char *inp_data, *inp_data2;
  m_axi_input_word *inp_data_m_axi;
  unsigned char *ref_index, *hls_index;
  //  float *hls_softmax, *ref_softmax;
  unsigned char *hls_max, *ref_max;


  /* **************************************************************************************** */

  // memory allocation
  tempbuf1 = (char*) malloc(MAX_PATH * sizeof(char));
  inp_data     = (signed char  *) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * MAX_NUM_OF_CLASSES * sizeof(signed char));
  inp_data2    = (signed char  *) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * MAX_NUM_OF_CLASSES * sizeof(signed char));
  ref_max      = (unsigned char*) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(unsigned char));
  hls_max      = (unsigned char*) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(unsigned char));
  ref_index    = (unsigned char*) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(unsigned char));
  hls_index    = (unsigned char*) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * sizeof(unsigned char));
  // ref_softmax  = (float *) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * MAX_NUM_OF_CLASSES * sizeof(float));
  // hls_softmax  = (float *) malloc(POST_MAX_HEIGHT * POST_MAX_WIDTH * MAX_NUM_OF_CLASSES * sizeof(float));
  inp_data_m_axi = new m_axi_input_word[POST_hls_MAXSZWORDS];

  /* **************************************************************************************** */
  // to reduce TB simulation time you can decrease the image size at runtime
  //  width  = POST_MAX_WIDTH/8;
  //  height = POST_MAX_HEIGHT/8;
  //width  = POST_MAX_WIDTH/4;
  //height = POST_MAX_HEIGHT/4;
  width  = POST_MAX_WIDTH;
  height = POST_MAX_HEIGHT;
  assert(width%8 == 0 && "width needs to be multiple of 8 for the implementation to be correct for the 28 classes output!");


  /* **************************************************************************************** */
  fprintf(stderr,"\nDPU Post-Processing on image size of W=%4d H=%4d and %2d classes\n", width, height, MAX_NUM_OF_CLASSES);
  fprintf(stderr,"\n\nRunning ");
  for (int i = 0; i < argc; i++)
    fprintf(stderr,"%s ", argv[i]);
  fprintf(stderr,"\n\n");

  /* **************************************************************************************** */

  //fprintf(stderr, "TIME to start:             "); int res1 = system("date");
  sprintf(tempbuf1, "%s", "./data_post/post_int8_din.bin");
  Read_Bin_File_of_Schar(tempbuf1, POST_MAX_HEIGHT*POST_MAX_WIDTH*MAX_NUM_OF_CLASSES, inp_data);
  char_to_hlsvector(inp_data,inp_data_m_axi);
  fprintf(stderr,"Loaded input data file  %s of size %4d %4d\n", tempbuf1, POST_MAX_HEIGHT, POST_MAX_WIDTH);
  //fprintf(stderr, "TIME to load bin inp file: "); int res2 = system("date");


  /* **************************************************************************************** */
  /* **************************************************************************************** */

  // no more needed
  ////prepare the scaling factor by creating a power of 2 number
  //float scaling_factor = PrepareScalingFactor(POSTPR_SCALE);
  //PrepareExpLUT(scaling_factor);
  // load scaling factors

  fprintf(stderr, "REF design with output_fixpos %d\n", output_fixpos);
  ref_dpupostproc(inp_data, //ref_softmax,
		  ref_max, ref_index, output_fixpos, height, width);
  //fprintf(stderr, "TIME to run REF postproc : ");  int res3 = system("date");


  //save reference results
  fprintf(stderr,"writing REF files\n");
  sprintf(tempbuf1, "%s", "./data_post/ref_max.bin");
  //Write_Txt_File<unsigned char>        (tempbuf1, height, width, ref_max);
  Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, ref_max);
  sprintf(tempbuf1, "%s", "./data_post/ref_index.bin");
  //Write_Txt_File<unsigned char>(tempbuf1, height, width, ref_index);
  Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, ref_index);
  /*
  sprintf(tempbuf1, "%s", "./data_post/ref_softmax.bin");
  //Write_SoftMax_Txt_File(tempbuf1, MAX_WIDTH*MAX_HEIGHT, ref_softmax);
  Write_Bin_File_as_float(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT*MAX_NUM_OF_CLASSES, ref_softmax);
  */
  //fprintf(stderr, "TIME to write binary REF : ");  int res4 = system("date");

  fprintf(stderr,"HLS DUT\n");
  // with debug  hls_dpupostproc(inp_data, hls_softmax, hls_max, hls_index, scaling_factor, height, width); // DUT: Design Under Test
  hls_dpupostproc(inp_data, hls_max, hls_index, output_fixpos, height, width); // DUT: Design Under Test
  //fprintf(stderr, "TIME to run HLS DUT      : ");   int res5 = system("date");

  fprintf(stderr,"HLS DUT M_AXI\n");
  //DB4nov21: the next line makes the CSIM stalling forever
  hls_dpupostproc_m_axi(inp_data_m_axi, hls_max, hls_index, output_fixpos, height, width); // DUT: Design Under Test
  //fprintf(stderr, "TIME to run HLS DUT M_AXI: ");   int res5_m_axi = system("date");

  fprintf(stderr,"writing HLS files\n");
  sprintf(tempbuf1, "%s", "./data_post/hls_max.bin");
  Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, hls_max);
  sprintf(tempbuf1, "%s", "./data_post/hls_index.bin");
  Write_Bin_File_as_Uchar(tempbuf1, POST_MAX_WIDTH*POST_MAX_HEIGHT, hls_index);
  //sprintf(tempbuf1, "%s", "./data_post/hls_softmax.bin");
  //Write_SoftMax_Txt_File(tempbuf1, height, width, hls_softmax);
  //Write_Bin_File_as_float(tempbuf1, MAX_WIDTH*MAX_HEIGHT*MAX_NUM_OF_CLASSES, hls_softmax);
  //fprintf(stderr, "TIME to write binary HLS : "); int res6 = system("date");


  /* **************************************************************************************** */
  /* **************************************************************************************** */

  // self checking test bench
  fprintf(stderr, "HLS Checking results: REF vs. HLS\n");
  float total_max_error   = 0;
  total_max_error   = check_output_max(ref_max, hls_max, height, width);
  float total_index_error = check_output_index(ref_index, hls_index, height, width);
  //float total_softm_error = check_output_softmax(ref_softmax, hls_softmax, height, width);
  float total_softm_error = 0;
  total_error = total_max_error + total_index_error + total_softm_error;
  fprintf(stderr, "error in max   value: %f\n", total_max_error);
  fprintf(stderr, "error in index value: %f\n", total_index_error);
  //fprintf(stderr, "error softmax  val  : %f\n", total_softm_error);
  //fprintf(stderr, "TIME to check results    : "); int res7 = system("date");

  fprintf(stderr,"HLS Test done\n");
  if (total_error > POST_REF_THRESHOLD)
  {
    fprintf(stderr, "TEST FAILED!: error = %10.8f\n", total_error);
    ret_res = 1;
  } else {
    fprintf(stderr, "TEST SUCCESSFULL!\n");
    ret_res = 0;
  }


  /* **************************************************************************************** */
  // free memory
  free(tempbuf1);
  free(inp_data);
  free(inp_data2);
  free(ref_max);
  free(hls_max);
  free(ref_index);
  free(hls_index);
  //free(ref_softmax); free(hls_softmax);
  delete [] inp_data_m_axi;

  fprintf(stderr, "DPU Post-Proc END\n");
  return ret_res;

}

#endif // ARM_HOST
