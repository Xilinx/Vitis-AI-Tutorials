/*******************************************************************************
Vendor: Xilinx
Associated Filename: ap_bmp.h
Purpose: BMP image reader and writer header file for AutoESL
Revision History: February 13, 2012 - initial release

*******************************************************************************

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
#ifndef __XLNX__BITMAP__
#define __XLNX__BITMAP__

/*
// to suppress MS Visual C++ 2-10 Express compiler warnings
#pragma warning( disable : 4996 ) // fopen, fclose, etc
#pragma warning( disable : 4068 ) // unknown HLS pragma
#pragma warning( disable : 4102 ) // unreferenced labels
#pragma warning( disable : 4715 ) // not all control paths return a value
*/

// Basic color definitions
#define BLACK 0
#define WHITE 255

// Maximum image size
#define MAX_ROWS 1080
#define MAX_COLS 1920

//File Information Header
typedef struct{
  unsigned short FileType;
  unsigned int FileSize;
  unsigned short Reserved1;
  unsigned short Reserved2;
  unsigned short Offset;
}BMPHeader;

typedef struct{
  unsigned int Size;
  unsigned int Width;
  unsigned int Height;
  unsigned short Planes;
  unsigned short BitsPerPixel;
  unsigned int Compression;
  unsigned int SizeOfBitmap;
  unsigned int HorzResolution;
  unsigned int VertResolution;
  unsigned int ColorsUsed;
  unsigned int ColorsImportant;
}BMPImageHeader;

typedef struct{
  BMPHeader *file_header;
  BMPImageHeader *image_header;
  unsigned int *colors;
  unsigned char *data;
  unsigned char R[MAX_ROWS][MAX_COLS];
  unsigned char G[MAX_ROWS][MAX_COLS];
  unsigned char B[MAX_ROWS][MAX_COLS];
  unsigned char Y[MAX_ROWS][MAX_COLS];
  char U[MAX_ROWS][MAX_COLS];
  char V[MAX_ROWS][MAX_COLS];
}BMPImage;

//Read Function
int BMP_Read(char *file, int row, int col, unsigned char *R, unsigned char *G, unsigned char *B);
//int BMP_Read(char *file, int row, int col, unsigned char *R, unsigned char *G, unsigned char *B, unsigned char *RGB);

//Write Function
int BMP_Write(char *file, int row, int col, unsigned char *R, unsigned char *G, unsigned char *B);

#endif
