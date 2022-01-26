/*************************************************************************************
 Vendor:		Xilinx
Associated Filename:	dpupreproc_tb.cpp
Purpose:		Vitis HLS DPU pre-processor
Revision History:	23 July 2021
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

#include "dpupreproc_tb.h"
#include "ap_bmp.h"




// write signed char to a binary file
void Write_Bin_File_of_Schar(char *filename, int dim, signed char *buffer)
{
  FILE *fid = fopen(filename, "wb");

  if(!fid)
    printf("ERROR: could not open binary %s for writing\n",filename);
  else
    fwrite(buffer, dim*sizeof(signed char), 1, fid);

  fclose(fid);
}
// Read from a binary file of signed char
void Read_Bin_File_of_Schar(char *filename, int dim, char *buffer)
{
  FILE *fid = fopen(filename, "rb");
  if(!fid)
    printf("ERROR: could not open binary %s for reading\n",filename);
  else
    fread(buffer, dim*sizeof(char), 1, fid);

  fclose(fid);
}

#ifndef ARM_HOST

void format_input_data(unsigned char *R, unsigned char *G, unsigned char *B, unsigned char *RGB,
		       uRGB_t *rgb_data, uPix3_t *hls_data, unsigned short int height,
		       unsigned short int width) {

  unsigned char in_R, in_G, in_B;

  int row = height;
  int col = width;
  uRGB_t rgb_pix;
  uPix3_t triple_pix;
  unsigned int index;

  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
      in_B = B[row * PRE_MAX_WIDTH + col];
      in_G = G[row * PRE_MAX_WIDTH + col];
      in_R = R[row * PRE_MAX_WIDTH + col];

      // pack input data for reference function
      rgb_pix.R = in_R;
      rgb_pix.G = in_G;
      rgb_pix.B = in_B;
      rgb_data[row * PRE_MAX_WIDTH + col] = rgb_pix;

      // pack input data for HLS accelerator: R is the MSB, while B is the LSB
      triple_pix.range(  BITS_PER_INP_PIXEL - 1,                    0) = in_B;
      triple_pix.range(2*BITS_PER_INP_PIXEL - 1,   BITS_PER_INP_PIXEL) = in_G;
      triple_pix.range(3*BITS_PER_INP_PIXEL - 1, 2*BITS_PER_INP_PIXEL) = in_R;
      //std::cout << std::hex << triple_pix << std::endl;
      hls_data[row * PRE_MAX_WIDTH + col] = triple_pix;

      index = row*width+col;
      RGB[3*index+0] = in_B;
      RGB[3*index+1] = in_G;
      RGB[3*index+2] = in_R;

    }
  }

}

float check_input_data(uPix3_t *hls_data, uRGB_t *ref_data,
		       unsigned short int height, unsigned short int width) {

  int row = height;
  int col = width;

  uPix3_t triple_hls;
  uRGB_t triple_ref;
  unsigned char int_R, int_G, int_B;
  float hls_R, hls_G, hls_B, ref_R, ref_B, ref_G;
  float diff_r, diff_g, diff_b, tot_diff;

  tot_diff = 0.0f;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
      triple_hls = hls_data[row * PRE_MAX_WIDTH + col];
      // unpack input data from HLS accelerator: R is the MSB, while B is the LSB
      int_B = triple_hls.range(    BITS_PER_INP_PIXEL - 1,                    0);
      int_G = triple_hls.range(2 * BITS_PER_INP_PIXEL - 1,   BITS_PER_INP_PIXEL);
      int_R = triple_hls.range(3 * BITS_PER_INP_PIXEL - 1, 2*BITS_PER_INP_PIXEL);
      hls_R = int_R;
      hls_G = int_G;
      hls_B = int_B;

      triple_ref = ref_data[row * PRE_MAX_WIDTH + col];
      ref_B = triple_ref.B;
      ref_G = triple_ref.G;
      ref_R = triple_ref.R;

      diff_r = ref_R - hls_R;
      diff_r = ABS(diff_r);
      diff_g = ref_G - hls_G;
      diff_g = ABS(diff_g);
      diff_b = ref_B - hls_B;
      diff_b = ABS(diff_b);

      tot_diff += diff_r + diff_g + diff_b;
    }
  }
  return tot_diff;
}


float check_output_data(Dat3_t *hls_data, iRGB_t *ref_data,
			unsigned short int height, unsigned short int width) {

  int row = height;
  int col = width;

  Dat3_t triple_hls;
  iRGB_t triple_ref;
  signed char int_R, int_G, int_B;
  float hls_R, hls_G, hls_B, ref_R, ref_B, ref_G;
  float diff_r, diff_g, diff_b, tot_diff;

  FILE *fp1, *fp2, *fp3;
  if ((fp1 = fopen("./data_pre/out_ch1.txt", "w")) == NULL) printf("Error opening out_ch1.txt");
  if ((fp2 = fopen("./data_pre/out_ch2.txt", "w")) == NULL) printf("Error opening out_ch2.txt");
  if ((fp3 = fopen("./data_pre/out_ch3.txt", "w")) == NULL) printf("Error opening out_ch3.txt");

  tot_diff = 0.0f;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
      triple_hls = hls_data[row * PRE_MAX_WIDTH + col];
      // unpack input data from HLS accelerator: R is the MSB, while B is the LSB
      int_B = triple_hls.range(    BITS_PER_OUT_PIXEL - 1,                    0);
      int_G = triple_hls.range(2 * BITS_PER_OUT_PIXEL - 1,   BITS_PER_OUT_PIXEL);
      int_R = triple_hls.range(3 * BITS_PER_OUT_PIXEL - 1, 2*BITS_PER_OUT_PIXEL);
      hls_R = int_R;
      hls_G = int_G;
      hls_B = int_B;
      fprintf(fp1, "%18.14f  ", hls_R);
      fprintf(fp2, "%18.14f  ", hls_G);
      fprintf(fp3, "%18.14f  ", hls_B);

      triple_ref = ref_data[row * PRE_MAX_WIDTH + col];
      ref_B = triple_ref.B;
      ref_G = triple_ref.G;
      ref_R = triple_ref.R;

      diff_r = ref_R - hls_R;
      diff_r = ABS(diff_r);
      diff_g = ref_G - hls_G;
      diff_g = ABS(diff_g);
      diff_b = ref_B - hls_B;
      diff_b = ABS(diff_b);

      tot_diff += diff_r + diff_g + diff_b;
    }
    fprintf(fp1, "\n");
    fprintf(fp2, "\n");
    fprintf(fp3, "\n");
  }
  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
  return tot_diff;
}

float check_output_data_hls(m_axi_output_word *hls_data, iRGB_t *ref_data,
			    unsigned short int height, unsigned short int width) {

  int row = height;
  int col = width;

  Dat3_t triple_hls;
  iRGB_t triple_ref;
  signed char int_R, int_G, int_B;
  float hls_R, hls_G, hls_B, ref_R, ref_B, ref_G;
  float diff_r, diff_g, diff_b, tot_diff;

  FILE *fp1, *fp2, *fp3;
  if ((fp1 = fopen("./data_pre/out_ch1_hls.txt", "w")) == NULL) printf("Error opening out_ch1_hls.txt");
  if ((fp2 = fopen("./data_pre/out_ch2_hls.txt", "w")) == NULL) printf("Error opening out_ch2_hls.txt");
  if ((fp3 = fopen("./data_pre/out_ch3_hls.txt", "w")) == NULL) printf("Error opening out_ch3_hls.txt");

  tot_diff = 0.0f;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
      int color_index_b = (row * PRE_MAX_WIDTH + col)*3 + 0 ;
      int color_index_g = (row * PRE_MAX_WIDTH + col)*3 + 1 ;
      int color_index_r = (row * PRE_MAX_WIDTH + col)*3 + 2 ;
      // unpack input data from HLS accelerator: R is the MSB, while B is the LSB
      int_B = hls_data[color_index_b/VECTORIZATION_FACTOR][color_index_b%VECTORIZATION_FACTOR];
      int_G = hls_data[color_index_g/VECTORIZATION_FACTOR][color_index_g%VECTORIZATION_FACTOR];
      int_R = hls_data[color_index_r/VECTORIZATION_FACTOR][color_index_r%VECTORIZATION_FACTOR];
      hls_R = int_R;
      hls_G = int_G;
      hls_B = int_B;
      fprintf(fp1, "%18.14f  ", hls_R);
      fprintf(fp2, "%18.14f  ", hls_G);
      fprintf(fp3, "%18.14f  ", hls_B);

      triple_ref = ref_data[row * PRE_MAX_WIDTH + col];
      ref_B = triple_ref.B;
      ref_G = triple_ref.G;
      ref_R = triple_ref.R;

      diff_r = ref_R - hls_R;
      diff_r = ABS(diff_r);
      diff_g = ref_G - hls_G;
      diff_g = ABS(diff_g);
      diff_b = ref_B - hls_B;
      diff_b = ABS(diff_b);

      tot_diff += diff_r + diff_g + diff_b;
    }
    fprintf(fp1, "\n");
    fprintf(fp2, "\n");
    fprintf(fp3, "\n");
  }
  fclose(fp1);
  fclose(fp2);
  fclose(fp3);
  return tot_diff;
}


//void RGB_to_hlsvector(unsigned char RGB[PRE_MAX_HEIGHT*PRE_MAX_WIDTH*3],
//		m_axi_input_word hlsvector[2*(PRE_MAX_HEIGHT*PRE_MAX_WIDTH*3)/VECTORIZATION_FACTOR])
void RGB_to_hlsvector(unsigned char *RGB, m_axi_input_word *hlsvector)
{

	m_axi_input_word tmp_word;
  for (int index = 0; index < PRE_MAX_HEIGHT*PRE_MAX_WIDTH*3; index++) {
    //if (index<10*3) fprintf(stderr, " pixel[%d] RGB[%d] = %d \n", index/VECTORIZATION_FACTOR, index%VECTORIZATION_FACTOR, RGB[index]);

	  tmp_word[index%VECTORIZATION_FACTOR] = RGB[index];
    hlsvector[index/VECTORIZATION_FACTOR]  = tmp_word;
  }
}

/* **************************************************************************************** */
/* **************************************************************************************** */
/* **************************************************************************************** */

int PRE_tb_main(int argc, char **argv, float pre_means[3], float pre_scales[3], int dpu_fixpos)
{

  unsigned short int x, y, width, height;
  char *tempbuf1, *tempbuf2;
  int check_results, ret_res = 0;

  // Arrays to store image data
  unsigned char *R, *G, *B, *RGB;

  //Arrays to send and receive data from the accelerator
  uRGB_t  *ref_inp_img;
  uPix3_t *hls_inp_img;
  iRGB_t  *ref_out_img;
  Dat3_t *hls_out_img;

  /* **************************************************************************************** */

  // memory allocation
  tempbuf1 = (char*) malloc(MAX_PATH * sizeof(char));
  R = (unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  G = (unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  B = (unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char));
  RGB=(unsigned char*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char) * 3);
  ref_inp_img = (uRGB_t *) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(uRGB_t));
  hls_inp_img = (uPix3_t*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(uPix3_t));
  ref_out_img = (iRGB_t *) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(iRGB_t));
  hls_out_img = ( Dat3_t*) malloc(PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof( Dat3_t));

  memset(R          , 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);
  memset(G          , 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);
  memset(B          , 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);
  memset(RGB        , 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH*3);
  memset(ref_inp_img, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);
  memset(hls_inp_img, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);
  memset(ref_out_img, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);
  memset(hls_out_img, 0, PRE_MAX_HEIGHT * PRE_MAX_WIDTH);

/*
  fprintf(stderr, "sizeof(hls::vector<ap_int< BITS_PER_INP_PIXEL>,VECTORIZATION_FACTOR>) = %ld\n",
		  sizeof(hls::vector<ap_int< BITS_PER_INP_PIXEL>,VECTORIZATION_FACTOR> ) );
  fprintf(stderr, "sizeof(ap_int< BITS_PER_INP_PIXEL>) = %ld\n", sizeof(ap_int< BITS_PER_INP_PIXEL>) );
  fprintf(stderr, "sizeof(unsigned char) = %ld\n", sizeof(unsigned char) );
*/

  //WARNING: this is a workaround, see below explanation
   //m_axi_input_word  hls_RGB_in[ (PRE_MAX_HEIGHT*PRE_MAX_WIDTH*3)/VECTORIZATION_FACTOR];
   //m_axi_output_word hls_RGB_out[(PRE_MAX_HEIGHT*PRE_MAX_WIDTH*3)/VECTORIZATION_FACTOR];
   //WARNING: the two lines below generate an ERROR that seems a BUG (see my previous workaround)
   // the only way to get rid of this bug is the "*8", without it the bug remains
  m_axi_input_word  *hls_RGB_in =(m_axi_input_word *)malloc((8*PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char) * 3)/VECTORIZATION_FACTOR);
  m_axi_output_word *hls_RGB_out=(m_axi_output_word*)malloc((8*PRE_MAX_HEIGHT * PRE_MAX_WIDTH * sizeof(unsigned char) * 3)/VECTORIZATION_FACTOR);


  /* **************************************************************************************** */
  // if the input images are smaller than MAX_HEIGHT * MAX_WIDTH  change here the resolution
  //  for example 640 x 480 in case of VGA
  width  = PRE_MAX_WIDTH;
  height = PRE_MAX_HEIGHT; // /16; //just to reduce simulation time

  //Get image data
  sprintf(tempbuf1, "%s.bmp", PRE_INPUT_IMAGE);
  // Fill a frame with data
  int read_tmp = BMP_Read(tempbuf1, PRE_MAX_HEIGHT, PRE_MAX_WIDTH, R, G, B);
  if (read_tmp != 0) {
    printf("%s Loading image failed\n", tempbuf1);
    exit(1);
  }
  printf("Loaded image file %s of size height %4d width %4d\n", tempbuf1, PRE_MAX_HEIGHT, PRE_MAX_WIDTH);

  format_input_data(R, G, B, RGB, ref_inp_img, hls_inp_img, PRE_MAX_HEIGHT, PRE_MAX_WIDTH);
  RGB_to_hlsvector(RGB,hls_RGB_in);
  float total_error = check_input_data(hls_inp_img, ref_inp_img, height, width);
  printf("check on input data: error=%f\n", total_error);

  //zero padding, if needed
  for (y = height; y < PRE_MAX_HEIGHT; y++) {
    for (x = width; x < PRE_MAX_WIDTH; x++) {
      uRGB_t rgb_null;
      rgb_null.B = 0;
      rgb_null.B = 0;
      rgb_null.R = 0;
      uPix3_t pix3_null = 0;

      ref_inp_img[y * PRE_MAX_WIDTH + x] = rgb_null;
      hls_inp_img[y * PRE_MAX_WIDTH + x] = pix3_null;
    }
  }

  /* **************************************************************************************** */
  printf("\nDPU Pre-Processing on image size of W=%4d H=%4d\n", PRE_MAX_WIDTH, PRE_MAX_HEIGHT);
  printf("\n\nRunning ");
  for (int i = 0; i < argc; i++)
    printf("%s ", argv[i]);
#ifdef DEBUG_TRANSPARENT
  printf("in transparent mode");
#endif
  printf("\n\n");

  /* **************************************************************************************** */
  /* **************************************************************************************** */


  printf("REF design\n");
  ref_dpupreproc(ref_inp_img, ref_out_img,
		  pre_means, pre_scales, dpu_fixpos,
		  height, width);

  printf("HLS DUT\n");
  hls_dpupreproc(hls_inp_img, hls_out_img,
		  pre_means[0],pre_means[1],pre_means[2], pre_scales[0],pre_scales[1],pre_scales[2],dpu_fixpos,
		  height, width); // DUT: Design Under Test

  printf("HLS HW-centric\n");
  hls_dpupreproc_m_axi(hls_RGB_in, hls_RGB_out,
		  pre_means[0],pre_means[1],pre_means[2], pre_scales[0],pre_scales[1],pre_scales[2],dpu_fixpos,
		  height, width); // DUT: Design Under Test

  /* **************************************************************************************** */
  /* **************************************************************************************** */

  // self checking test bench
  printf("HLS Checking results: REF vs. HLS model\n");
  total_error += check_output_data(hls_out_img, ref_out_img, height, width);
  printf("check on output results data: error=%f\n", total_error);

  printf("HLS Checking results: REF vs. HLS HW\n");
  total_error += check_output_data_hls(hls_RGB_out, ref_out_img, height, width);
  printf("check on HLS HW output results data: error=%f\n", total_error);


  printf("HLS Test done\n");
  if (total_error > PRE_REF_THRESHOLD) {
    printf("TEST FAILED!: error = %10.8f\n", total_error);
    ret_res = 1;
  } else {
    printf("TEST SUCCESSFUL!\n");
    ret_res = 0;
  }

#ifdef DEBUG_TRANSPARENT //work in transparent mode for debug
  {

    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
	iRGB_t triple_ref = ref_out_img[row * PRE_MAX_WIDTH + col];
	signed char ref_B = triple_ref.B;
	signed char ref_G = triple_ref.G;
	signed char ref_R = triple_ref.R;
	R[row * PRE_MAX_WIDTH + col] = (unsigned char) ref_R;
	G[row * PRE_MAX_WIDTH + col] = (unsigned char) ref_G;
	B[row * PRE_MAX_WIDTH + col] = (unsigned char) ref_B;
      }
    }
    sprintf(tempbuf1, "%s", "./data/ref_transp_out.bmp");
    BMP_Write(tempbuf1, height, width, R, G, B);
  }
#endif


  /* **************************************************************************************** */
  // free memory
  free(R);
  free(G);
  free(B);
  free(RGB);
  free(tempbuf1);
  free(ref_inp_img);
  free(hls_inp_img);
  free(ref_out_img);
  free(hls_out_img);
  free(hls_RGB_in);
  free(hls_RGB_out);

  printf("DPU Pre-Proc END\n");

  return 0; //ret_res;

}


#endif //ARM_HOST
