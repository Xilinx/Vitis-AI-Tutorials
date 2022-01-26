/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Author: Daniele Bagni & Peter Schillinger, Xilinx Inc.
// 15 Nov  2021

// WARNING: this code assumes that the image stored in the HD have the same size
// and do not need any resize

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "hw_pre2dpu2post.hpp"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

//this tutorial applies 12 classes only
int MAX_NUM_CLASSES=12;


//functions placed on the bottom of this file
void Write_Bin_File_as_Schar(char *filename, int dim, signed char *buffer);
void Write_Bin_File_as_Uchar(char *filename, int dim, unsigned char *buffer);



GraphInfo shapes;
int g_pre_type = 0;

// const string baseImagePath = "./src/img_test/";
string baseImagePath;  // they will get their values via argv[]

int num_threads = 0;
int is_running_0 = 1;
int num_of_images = 0;
int num_images_x_thread = 0;

int NUM_TEST_IMAGES = 3;

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64,  35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0,  0,   0,   60,  80,  0,   11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

static int8_t op_move(uint8_t i) {
  int8_t ret = int8_t(i - 128);
  return ret >> 1;
}

void transform_bgr(int w, int h, unsigned char *src, signed char *dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R) {
  float32x4_t shiftB = vdupq_n_f32(var_shift_B);
  float32x4_t shiftG = vdupq_n_f32(var_shift_G);
  float32x4_t shiftR = vdupq_n_f32(var_shift_R);

  float32x4_t scaleB = vdupq_n_f32(var_scale_B);
  float32x4_t scaleG = vdupq_n_f32(var_scale_G);
  float32x4_t scaleR = vdupq_n_f32(var_scale_R);

  for (int i = 0; i < h; i++) {
    int idx_base = i * w * 3;
    for (int j = 0; j < w; j += 8) {
      int idx = idx_base + j * 3;

      // init
      uint8x8x3_t sbgr_u8;
      uint16x8x3_t sbgr_u16;
      sbgr_u8 = vld3_u8(src + idx);
      sbgr_u16.val[0] = vmovl_u8(sbgr_u8.val[0]);
      sbgr_u16.val[1] = vmovl_u8(sbgr_u8.val[1]);
      sbgr_u16.val[2] = vmovl_u8(sbgr_u8.val[2]);

      // get low part u32
      uint32x4_t sb_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[0]));
      uint32x4_t sg_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[1]));
      uint32x4_t sr_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[2]));

      // get high part u32
      uint32x4_t sb_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[0]));
      uint32x4_t sg_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[1]));
      uint32x4_t sr_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[2]));

      // get low part float
      float32x4_t sb_low_f32 = vcvtq_f32_u32(sb_low_u32);
      float32x4_t sg_low_f32 = vcvtq_f32_u32(sg_low_u32);
      float32x4_t sr_low_f32 = vcvtq_f32_u32(sr_low_u32);

      // get high part float
      float32x4_t sb_high_f32 = vcvtq_f32_u32(sb_high_u32);
      float32x4_t sg_high_f32 = vcvtq_f32_u32(sg_high_u32);
      float32x4_t sr_high_f32 = vcvtq_f32_u32(sr_high_u32);

      // calculate low part float
      sb_low_f32 = vmulq_f32(vsubq_f32(sb_low_f32, shiftB), scaleB);
      sg_low_f32 = vmulq_f32(vsubq_f32(sg_low_f32, shiftG), scaleG);
      sr_low_f32 = vmulq_f32(vsubq_f32(sr_low_f32, shiftR), scaleR);

      // calculate low part float
      sb_high_f32 = vmulq_f32(vsubq_f32(sb_high_f32, shiftB), scaleB);
      sg_high_f32 = vmulq_f32(vsubq_f32(sg_high_f32, shiftG), scaleG);
      sr_high_f32 = vmulq_f32(vsubq_f32(sr_high_f32, shiftR), scaleR);

      // get the result low part int32
      int32x4_t db_low_s32 = vcvtq_s32_f32(sb_low_f32);
      int32x4_t dg_low_s32 = vcvtq_s32_f32(sg_low_f32);
      int32x4_t dr_low_s32 = vcvtq_s32_f32(sr_low_f32);

      // get the result high part int32
      int32x4_t db_high_s32 = vcvtq_s32_f32(sb_high_f32);
      int32x4_t dg_high_s32 = vcvtq_s32_f32(sg_high_f32);
      int32x4_t dr_high_s32 = vcvtq_s32_f32(sr_high_f32);

      // get the result low part int16
      int16x4_t db_low_s16 = vmovn_s32(db_low_s32);
      int16x4_t dg_low_s16 = vmovn_s32(dg_low_s32);
      int16x4_t dr_low_s16 = vmovn_s32(dr_low_s32);

      // get the result high part int16
      int16x4_t db_high_s16 = vmovn_s32(db_high_s32);
      int16x4_t dg_high_s16 = vmovn_s32(dg_high_s32);
      int16x4_t dr_high_s16 = vmovn_s32(dr_high_s32);

      // combine low and high into int16x8
      int16x8_t db_s16 = vcombine_s16(db_low_s16, db_high_s16);
      int16x8_t dg_s16 = vcombine_s16(dg_low_s16, dg_high_s16);
      int16x8_t dr_s16 = vcombine_s16(dr_low_s16, dr_high_s16);

      // combine low and high into int16x8
      int8x8x3_t dbgr;
      dbgr.val[0] = vmovn_s16(db_s16);
      dbgr.val[1] = vmovn_s16(dg_s16);
      dbgr.val[2] = vmovn_s16(dr_s16);

      // store...
      vst3_s8(dst + idx, dbgr);
    }
  }
}


/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images_list - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images_list) {
  images_list.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images_list.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief Run DPU Task for CNN
 *
 * @param taskFCN8 - pointer to FCN8 Task
 *
 * @return none
 */
void runCNN(vart::Runner *runner, int8_t *imageInputs, int8_t *FCResult) {
  // get in/out tensors and dims
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_dims();
  auto in_dims = inputTensors[0]->get_dims();
  // get shape info
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  int8_t *loc_imageInputs = imageInputs;
  int8_t *loc_FCResult = FCResult;

  // debug messages
  cout << "INSIDE THE DPU" << endl;
  cout << "outSize  " << outSize << endl;
  cout << " inSize  " << inSize << endl;
  cout << "outW     " << outWidth << endl;
  cout << "outH     " << outHeight << endl;
  cout << "inpW     " << inWidth << endl;
  cout << "inpH     " << inHeight << endl;
  cout << "# class  " << num_of_classes << endl;
  cout << "batchSize " << batchSize << endl;  // alway 1 for Edge

  for (unsigned int n = 0; n < num_images_x_thread; n += batchSize)  // this works correctly for either batchSize= 1 or 3
    {
      loc_imageInputs = imageInputs + n * inSize;
      loc_FCResult    = FCResult + n * outSize;

      // in/out tensor refactory for batch inout/output
      batchTensors.push_back(std::shared_ptr<xir::Tensor>(
							  xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
									      xir::DataType{xir::DataType::XINT, 8u})));
      inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
							     loc_imageInputs, batchTensors.back().get()));
      batchTensors.push_back(std::shared_ptr<xir::Tensor>(
							  xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
									      xir::DataType{xir::DataType::XINT, 8u})));
      outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
							      loc_FCResult, batchTensors.back().get()));

      // tensor buffer input/output
      inputsPtr.clear();
      outputsPtr.clear();
      inputsPtr.push_back(inputs[0].get());
      outputsPtr.push_back(outputs[0].get());

      // run
      auto job_id = runner->execute_async(inputsPtr, outputsPtr);
      runner->wait(job_id.first, -1);

      inputs.clear();
      outputs.clear();
    }
}

/**
 * @brief Entry for running FCN8 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy FCN8 on DPU platform.
 *
 */
int main(int argc, char *argv[])
{
  // using std::chrono::system_clock;
  system_clock::time_point t_start, t_end;
  system_clock::time_point t_start2, t_end2, r_end2;
  system_clock::time_point t_start3, t_end3;

  // Check args
  if (argc != 6) {
    cout << "Usage: run_cnn xmodel_path test_images_path thread_num (from 1 to "
      "6) use_hw_postproc(1:yes, 0:no) preproc_type(0:hw, 1:cpu, 2:neon)"
         << endl;
    return -1;
  }
  baseImagePath =
    std::string(argv[2]);  // path name of the folder with test images
  num_threads = atoi(argv[3]);
  assert((num_threads <= 6) & (num_threads >= 1));
  int use_post_processing = atoi(argv[4]);
  g_pre_type = atoi(argv[5]);
  cout << endl;
  for (int i=0; i<argc; i++) cout << argv[i] << " "; cout << endl; cout << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPARE DPU STUFF

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
    << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  // create runners
  auto runner  = vart::Runner::create_runner(subgraph[0], "run");
  auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner5 = vart::Runner::create_runner(subgraph[0], "run");

  // get in/out tensors and dims
  auto inputTensors  = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();
  auto out_dims      = outputTensors[0]->get_shape();
  auto in_dims       = inputTensors[0]->get_shape();

  // get in/out tensor shape
  int inputCnt  = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList  = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // get shape info
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;

  int out_fixpos = (outputTensors[0])->template get_attr<int>("fix_point");
  auto out_fix_scale = std::exp2f(1.0f * (float)out_fixpos);
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int in_fixpos = (inputTensors[0])->template get_attr<int>("fix_point");
  auto in_fix_scale = std::exp2f(1.0f * (float)in_fixpos);
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);

  // debug messages
  cout << endl;
  cout << "outSize       " << outSize << endl;
  cout << "inpSize       " << inSize << endl;
  cout << "outW          " << outWidth << endl;
  cout << "outH          " << outHeight << endl;
  cout << "outS          " << out_fix_scale << endl;
  cout << "inpW          " << inWidth << endl;
  cout << "inpH          " << inHeight << endl;
  cout << "inp fixpos    " << in_fixpos << endl;
  cout << "inp fix scale " << in_fix_scale << endl;
  cout << "out fixpos    " << out_fixpos << endl;
  cout << "out fix scale " << out_fix_scale << endl;
  cout << "num classes   " << num_of_classes << endl;
  cout << "batchSize     " << batchSize << endl;  // alway 1 for Edge
  cout << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // TIMERS CALIBRATION

  int num_of_trials = 200;
  std::chrono::duration<double, std::micro> avg_calibr_highres(0);
  for (int i =0; i<num_of_trials; i++)
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      auto t2 = std::chrono::high_resolution_clock::now();
      // floating-point duration: no duration_cast needed
      std::chrono::duration<double, std::micro> fp_us = t2 - t1;
      avg_calibr_highres  += fp_us;
      //if (i%20 ==0) cout << "[Timers calibration  ] " << fp_us.count() << "us" << endl;
    }
  cout << "[average calibration high resolution clock] " << avg_calibr_highres.count() / num_of_trials << "us"  << endl;
  //cout << "\n" << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MEMORY ALLOCATION

  // Load all image filenames
  vector<string> image_filename;
  ListImages(baseImagePath, image_filename);
  if (image_filename.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    exit(-1);
  } else {
    num_of_images = image_filename.size();
  }

  if (num_of_images > NUM_TEST_IMAGES) num_of_images = NUM_TEST_IMAGES;
  cout << "\n max num of images to read " << num_of_images << endl;

  // number of images per thread
  num_images_x_thread = num_of_images / num_threads;
  num_images_x_thread = (num_images_x_thread / batchSize) * batchSize;
  cout << "number of images per thread: " << num_images_x_thread << endl;
  // effective number of images as multiple of num_threads and batchSize
  num_of_images = num_images_x_thread * num_threads;

  // memory allocation
  vector<Mat> imagesList;
  Mat hw_segMat(outHeight, outWidth, CV_8UC3);
  Mat ref_segMat(outHeight, outWidth, CV_8UC3);
  Mat ref_showMat(outHeight, outWidth, CV_8UC3);
  Mat hw_showMat( outHeight, outWidth, CV_8UC3);
  int8_t *imageInputs     = new int8_t[(num_of_images)*inSize];
  int8_t *ref_imageInputs = new int8_t[(num_of_images)*inSize];
  int8_t *FCResult  = new int8_t [outSize*num_of_images];
  int8_t *POSTinput = new int8_t [num_of_images*outHeight*outWidth*MAX_NUM_CLASSES]; // the POST processor expects up to MAX_NUM_CLASSES];
  uint8_t *idx_out = new uint8_t[outHeight*outWidth*num_of_images];
  uint8_t *idx_ref = new uint8_t[outHeight*outWidth*num_of_images];
  uint8_t *max_out = new uint8_t[outHeight*outWidth*num_of_images];
  Mat image  = cv::Mat(inHeight, inWidth, CV_8UC3);
  Mat image2 = cv::Mat(inHeight, inWidth, CV_8UC3); //just for debug

  //void* memset( void* dest, int ch, size_t count );
  memset( FCResult, 0, sizeof(FCResult));


  /////////////////////////////////////////////////////////////////////////////////////////////
  // INITIALIZING THE PL ACCELERATORS
  cout << "\n" << endl;
  //initialization of xclbin and device
  auto hw     = new HW("/media/sd-mmcblk0p1/dpu.xclbin");
  auto hwPre  = new HW_PRE(hw->get_DeviceHandle(), hw->get_XclbinHandle(), "hls_dpupreproc_m_axi");
  auto hwPost = new HW_POST(hw->get_DeviceHandle(), hw->get_XclbinHandle(), "hls_dpupostproc_m_axi");
  hwPre->load(hw->get_DeviceHandle());
  hwPost->load(hw->get_DeviceHandle());
  cout << endl;
  cout << "PL Accelerators initialized" << endl;
  cout << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPROCESSING ALL IMAGES

#if 1
  vector<float> mean{128.0f, 128.0f, 128.0f};
  vector<float> scale{1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
  vector<float> real_scale{scale[0] * in_fix_scale, scale[1] * in_fix_scale, scale[2] * in_fix_scale};
#else   //transparent mode for debug
  vector<float> mean{ 0.0f, 0.0f, 0.0f};
  vector<float> scale{1.0f, 1.0f, 1.0f};
  in_fix_scale = 1;
  in_fixpos    = 0;
  vector<float> real_scale{scale[0] * in_fix_scale, scale[1] * in_fix_scale, scale[2] * in_fix_scale};
#endif
  cout << "scale factor: " << scale[0] << " dpu fix scale: " << in_fix_scale << " real scale: " << real_scale[0] << endl;
  cout << "num_of_images: " << num_of_images << endl;

  auto pre_load1 = std::chrono::high_resolution_clock::now();
  // load all images at once
  L1:for (unsigned int n = 0; n < num_of_images; n++)
  {
    image = imread(baseImagePath + image_filename[n]);
    cout << "reading " << image_filename[n] << " " <<image.size()<<endl;
    /*
      char s[20]; sprintf(s, "inp_%03d", n);
      cv::imshow(s,  image);
      cv::waitKey(1000);
      cv::destroyAllWindows();
      cv::imwrite(format("inp_%03d.png",n), image);
      cout << "\n writing " << format("inp_%03d.png",n) << endl;
    */
    imagesList.push_back(image);
  }
  auto pre_load2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> prepr_load_time = pre_load2 - pre_load1;
  cout << "\n" << endl;
  cout << "\n" << endl;
  cout << "[READ  Time ] " << prepr_load_time.count() << "us" << endl;
  cout << "[READ  FPS  ] " << num_of_images * 1000000.0 / prepr_load_time.count() << endl;
  cout << "\n" << endl;

  // RUN SW PREPROCESSING FOR REFERENCE
  auto pre_t1 = std::chrono::high_resolution_clock::now();
  L3:for (unsigned int n = 0; n < num_of_images; n++)
  {
    image = imagesList[n].clone();
    //cv::imwrite(format("inp_L3_%03d.png",n), image);
    if (g_pre_type == 1)
    {  // Original treatment
    	for (int y = 0; y < inHeight; y++)
    	{
	  for (int x = 0; x < inWidth; x++)
	  {
	    for (int c = 0; c < 3; c++)
	    {
	      // BGR format
	      float pix = image.at<Vec3b>(y, x)[c];
	      float dif = (pix - mean[c]) * real_scale[c];
	      //clipping
	      if (dif  >=  127) dif =  127;
	      if (dif  <= -128) dif = -128;

	      ref_imageInputs[n * inSize + 3 * (y * inWidth + x) + c] = (int8_t) dif;
	    }
	  }
    	}
    }
    // The following code can improve prepost performance, for reference
    if (g_pre_type == 2)
    { // Sequential storage
      if (0) transform(image.data, image.data + inSize, ref_imageInputs + n * inSize, op_move);
      if (1) {  // neon accelerate
    	  for (auto i = 0; i < inHeight; ++i) {
    		  transform_bgr(inWidth, 1, const_cast<uint8_t *>(image.data) + i * 3 * inWidth,
    				  ref_imageInputs + n * inSize + i * inWidth * 3, mean[0], real_scale[0],
					  mean[1], real_scale[1], mean[2], real_scale[2]);
    	  }
      }
    }
  } //end of L3
  auto pre_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> prepr_sw_time = pre_t2 - pre_t1; // - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[SW PREPROC  Time ] " << prepr_sw_time.count() << "us" << endl;
  cout << "[SW PREPROC  FPS  ] " << num_of_images*1000000.0/prepr_sw_time.count()  << endl;
  cout << "\n" << endl;

 // PL HW preprocessor runnning
  //xrtURStart(0, "HW PRE kernel start", "HW PRE kernel start");  //DB: if you uncomment it, it will stall forever
  pre_t1 = std::chrono::high_resolution_clock::now();
  L2:for (unsigned int n = 0; n < num_of_images; n++)
  {
    image = imagesList[n].clone();
    	  hwPre->run(image, imageInputs + n * inSize, mean, scale, in_fixpos); //DB: in_fixpos instead of real_scale
  }
  pre_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> prepr_hw_time = pre_t2 - pre_t1 - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[HW PREPROC  Time ] " << prepr_hw_time.count() << "us" << endl;
  cout << "[HW PREPROC  FPS  ] " << num_of_images*1000000.0/prepr_hw_time.count()  << endl;
  cout << "\n" << endl;

  //check PREPROCESSING results
  cout << "now checking preprocessing results: HW vs. SW\n" << endl;
  double preproc_total_error = 0;
  L4:for (unsigned int n = 0; n < num_of_images; n++)
  {
    for (int y = 0; y < inHeight; y++)
    {
      for (int x = 0; x < inWidth; x++)
      {
    	  for (int c = 0; c < 3; c++)
    	  {
	    int pre_pix =     imageInputs[n * inSize + 3 * (y * inWidth + x) + c];
	    int ref_pix = ref_imageInputs[n * inSize + 3 * (y * inWidth + x) + c];
	    image.at<Vec3b>(y, x)[c]  = pre_pix;
	    image2.at<Vec3b>(y, x)[c] = ref_pix;
	    int diff = pre_pix - ref_pix;
	    //take abs values
	    if (diff<0) diff = 0 - diff;
	    if (diff > 1)
	      {
		printf("PRE ERROR=%5d on frame %3d: got %4d expected %4d at row %4d col %4d channel %1d\n", diff, n, pre_pix, ref_pix, y, x, c);
		preproc_total_error += diff;
	      }
    	  }
      }
    }
    cv::imwrite(format("hw_preproc_out_%03d.png",n), image); // just for check
    cv::imwrite(format("sw_preproc_out_%03d.png",n), image2); // just for check
  } //end of L4
  printf("PRE TOTAL ERROR=%f \n", preproc_total_error);

  /*
  // just for debug
  imagesList.begin();
  for (unsigned int n = 0; n < num_of_images; n++)
  {
      image = imagesList[n];
      //cv::imshow(format("list_%03d", n),  imagesList[n]);
      //cv::waitKey(1000);
      //cv::imshow(format("clone_%03d", n),  image);
      //cv::waitKey(1000);
      //cv::destroyAllWindows();
      cv::imwrite(format("clone_%03d.png",n), image);
  }
  */

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MULTITHREADING DPU EXECUTION WITH BATCH

  // split images in chunks, each chunks for its own thead
  // avoid pointing to wrong memorycv::Mat> locations
  cout << "DPU Processing" << endl;
  int8_t *imagesInput0 =
    imageInputs + inSize * (num_threads == 1 ? 0 * num_images_x_thread : 0);
  int8_t *imagesInput1 =
    imageInputs + inSize * (num_threads == 2 ? 1 * num_images_x_thread : 0);
  int8_t *imagesInput2 =
    imageInputs + inSize * (num_threads == 3 ? 2 * num_images_x_thread : 0);
  int8_t *imagesInput3 =
    imageInputs + inSize * (num_threads == 4 ? 3 * num_images_x_thread : 0);
  int8_t *imagesInput4 =
    imageInputs + inSize * (num_threads == 5 ? 4 * num_images_x_thread : 0);
  int8_t *imagesInput5 =
    imageInputs + inSize * (num_threads == 6 ? 5 * num_images_x_thread : 0);
  int8_t *FCResult0 =
    FCResult + outSize * (num_threads == 1 ? 0 * num_images_x_thread : 0);
  int8_t *FCResult1 =
    FCResult + outSize * (num_threads == 2 ? 1 * num_images_x_thread : 0);
  int8_t *FCResult2 =
    FCResult + outSize * (num_threads == 3 ? 2 * num_images_x_thread : 0);
  int8_t *FCResult3 =
    FCResult + outSize * (num_threads == 4 ? 3 * num_images_x_thread : 0);
  int8_t *FCResult4 =
    FCResult + outSize * (num_threads == 5 ? 4 * num_images_x_thread : 0);
  int8_t *FCResult5 =
    FCResult + outSize * (num_threads == 6 ? 5 * num_images_x_thread : 0);
  // run multithreading
  thread workers[num_threads];
  auto dpu_t1 = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < num_threads; i++)
  {
    if (i == 0)
      workers[i] =
	thread(runCNN, runner.get(), ref(imagesInput0), ref(FCResult0));
    if (i == 1)
      workers[i] =
	thread(runCNN, runner1.get(), ref(imagesInput1), ref(FCResult1));
    if (i == 2)
      workers[i] =
	thread(runCNN, runner2.get(), ref(imagesInput2), ref(FCResult2));
    if (i == 3)
      workers[i] =
	thread(runCNN, runner3.get(), ref(imagesInput3), ref(FCResult3));
    if (i == 4)
      workers[i] =
	thread(runCNN, runner4.get(), ref(imagesInput4), ref(FCResult4));
    if (i == 5)
      workers[i] =
	thread(runCNN, runner5.get(), ref(imagesInput5), ref(FCResult5));
  }
  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }
  auto dpu_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> dpu_time = dpu_t2 - dpu_t1 - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[DPU tot Time ] " << dpu_time.count()                 << "us" << endl;
  //cout << "[DPU avg Time ] " << (dpu_time.count()/num_of_images) << "us" << endl;
  cout << "[DPU avg FPS  ] " << num_of_images*1000000.0/dpu_time.count()  << endl;
  cout << "\n" << endl;
  double total_time = 0.0;


  //the output from the DPU contains only 12 classes which is less than 28 max classes
  // we need to adapt the input buffer for the PL preprocessor
  int osize = outHeight * outWidth;
  for (unsigned int n = 0; n < num_of_images; n++)
  {
    int8_t *ptr_POST = &POSTinput[n*osize*MAX_NUM_CLASSES];
    int8_t *ptr_DPU  = &FCResult[ n*outSize];
    for (int y = 0; y < outHeight; y++)
    {
      for (int x = 0; x < outWidth; x++)
      {
	for (int c = 0; c < num_of_classes; c++)
	{
	  ptr_POST[ y*outWidth*MAX_NUM_CLASSES +x*MAX_NUM_CLASSES +c] =
	    ptr_DPU[y*outWidth*num_of_classes  +x*num_of_classes  +c];
	}

	//fill with wero data for the classes above 12
	for (int c = num_of_classes; c<MAX_NUM_CLASSES; c++)
	{
	  ptr_POST[y*outWidth*MAX_NUM_CLASSES +x*MAX_NUM_CLASSES +c] = 0;
	}
      }
    }
  }


  /////////////////////////////////////////////////////////////////////////////////////////////
  // POSTPROCESSING ALL THE IMAGES AT ONCE
  if (use_post_processing == 1)
  {
    cout << "\n DOING POST PROCESSING\n" << endl;

    // PL HW accelerator to compute argmax and index of all images
    auto postpr_hw_t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int n = 0; n < num_of_images; n++)
    {
      //int8_t *DPUres     = &FCResult[n*outSize]; //DB: this cannot be used as it contains only 12 classes
      int8_t   *DPUres     = &POSTinput[n*osize*MAX_NUM_CLASSES];
      uint8_t  *POSToutmax = &max_out [n*osize];
      uint8_t  *POSToutidx = &idx_out [n*osize];
      hwPost->run(DPUres, POSToutmax, POSToutidx, out_fixpos);
    }
    auto postpr_hw_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> postpr_hw_time = postpr_hw_t2 - postpr_hw_t1; // - avg_calibr_highres;
    cout << "\n" << endl;
    cout << "[HW POSTPROC Time ] " << postpr_hw_time.count() << "us" << endl;
    cout << "[HW_POSTPROC FPS  ] " << num_of_images*1000000.0/postpr_hw_time.count()  << endl;
    cout << "\n" << endl;
    total_time =  (double) postpr_hw_time.count();

    // SW computation of index (without SotfMax) as a reference
    LPP1:for (unsigned int n = 0; n < num_of_images; n++)
    {
      int8_t *inpdata      = &FCResult[n*outSize];
      uint8_t *loc_idx_ref = &idx_ref[ n*osize];;

      if (0) { // get float result
    	  vector<float> float_out_data;
    	  float_out_data.reserve(outSize);
    	  transform(inpdata, inpdata + osize*num_of_classes, float_out_data.begin(),
    			  std::bind2nd(multiplies<float>(), out_fix_scale));
      }
      for (int row = 0; row < outHeight; row++)
      {
    	  for (int col = 0; col < outWidth; col++)
    	  {
    		  int ii = row * outWidth * num_of_classes + col * num_of_classes;
    		  auto max_ind = max_element(inpdata + ii, inpdata + ii + num_of_classes);
    		  int posit = distance(inpdata + ii, max_ind);
    		  loc_idx_ref[row*outWidth+col] = posit;
    	  }
      }
    } // end of LPP1

  char tempbuf1[500];
  fprintf(stderr,"writing binary file with POST input data\n");
  sprintf(tempbuf1, "%s", "./data_pre2dpu2post/post_int8_din.bin");
  Write_Bin_File_as_Schar(tempbuf1, outSize*num_of_images, FCResult);

  char tempbuf2[500];
  fprintf(stderr,"writing binary file with POST index output data\n");
  sprintf(tempbuf2, "%s", "./data_pre2dpu2post/post_uint8_out_max.bin");
  Write_Bin_File_as_Uchar(tempbuf2, osize*num_of_images, max_out);
  sprintf(tempbuf2, "%s", "./data_pre2dpu2post/post_uint8_out_idx.bin");
  Write_Bin_File_as_Uchar(tempbuf2, osize*num_of_images, idx_out);
  sprintf(tempbuf2, "%s", "./data_pre2dpu2post/post_uint8_ref_idx.bin");
  Write_Bin_File_as_Uchar(tempbuf2, osize*num_of_images, idx_ref);

    //check POSTPROCESSING results
    cout << "now checking postprocessing results: HW vs. SW\n" << endl;
    double postproc_total_error = 0;
    LPP2:for (unsigned int n = 0; n < num_of_images; n++)
    {
      for (int y = 0; y < outHeight; y++)
      {
    	  for (int x = 0; x < outWidth; x++)
    	  {
	    int post_index = idx_out[n * osize + (y * outWidth + x)];
	    int ref_index  = idx_ref[n * osize + (y * outWidth + x)];
	    int diff = post_index - ref_index;
	    //take abs values
	    if (diff<0) diff = 0 - diff;
	    if (diff > 1)
	      {
		printf("POST ERR=%5d at frame %3d: got %4d expected %4d at row %4d col %4d\n", diff, n, post_index, ref_index, y, x);
		postproc_total_error += diff;
	      }
    	  }
      }
    } //end of LPP2
    printf("POST TOTAL ERROR=%f \n", postproc_total_error);

    // now use OpenCV functions to generate output segmented images
    Mat small_img;
    LPP3:for (unsigned int n = 0; n < num_of_images; n++)
    {
      cout << "\nImage : " << image_filename[n] << endl;
      image = imagesList[n].clone();
      cv::resize(image, small_img, hw_showMat.size(), 0, 0, INTER_AREA);
      //cv::imshow("Segmentation", small_img);
      // cv::waitKey(1000);
      // save the image
      cv::imwrite(image_filename[n], image);
      // cv::destroyAllWindows();
      uint8_t *loc_idx_out = &idx_out[osize*n]; //DB
      uint8_t *loc_idx_ref = &idx_ref[osize*n];;//DB

      for (int row = 0; row < outHeight; row++)
      {
        for (int col = 0; col < outWidth; col++)
        {
          int hw_posit = int(loc_idx_out[row * outWidth + col]);
          hw_segMat.at<Vec3b>(row, col) =  Vec3b(colorB[hw_posit],   colorG[hw_posit],  colorR[hw_posit]);
          int ref_posit = int(loc_idx_ref[row * outWidth + col]);
          ref_segMat.at<Vec3b>(row, col) =  Vec3b(colorB[ref_posit], colorG[ref_posit], colorR[ref_posit]);
        }
      }
      for (int ii = 0; ii < hw_showMat.rows * hw_showMat.cols * 3; ii++) {
        ref_showMat.data[ii] = small_img.data[ii] * 0.4 + ref_segMat.data[ii] * 0.6;
         hw_showMat.data[ii] = small_img.data[ii] * 0.4 +  hw_segMat.data[ii] * 0.6;
      }

      // just for debug
      if (n <= 3)
      {
    	  char s[20];
    	  sprintf(s, "out_%03d", n);
    	  // putText(image3, s, Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
    	  Mat hw_dst, ref_dst;
    	  cv::hconcat(small_img,  hw_segMat,  hw_dst);  // horizontal
    	  cv::hconcat(small_img, ref_segMat, ref_dst);  // horizontal
    	  //cv::imshow(s, dst);
    	  //cv::waitKey(1000);
    	  cv::imwrite(format("hw_out_%03d.png", n),  hw_dst);
    	  cv::imwrite(format("ref_out_%03d.png", n), ref_dst);
    	  //cv::destroyAllWindows();
      }
    } //end of LPP3

  } //end of   if (use_post_processing == 1)

  total_time += (double) prepr_hw_time.count();
  total_time += (double) dpu_time.count();
  cout << "[E2E Time ] " << total_time << "us" << endl;
  cout << "[E2E FPS  ] " << num_of_images * 1000000.0 / total_time  << endl;
  cout << "\n" << endl;

/////////////////////////////////////////////////////////////////////////////////////////////


  cout << "deleting imagesList  memory" << endl;
  imagesList.clear();
  cout << "deleting imageInputs memory" << endl;
  delete[] imageInputs;
  cout << "deleting ref_imgInputs mem" << endl;
  delete[] ref_imageInputs;
  cout << "deleting FCResult    memory" << endl;
  delete[] FCResult;
  delete[] POSTinput;
  cout << "deleting idx buffers memory" << endl;
  delete[]idx_out;
  delete[] idx_ref;
  cout << "deleting max buffer  memory" << endl;
  delete[] max_out;

  cout << "deleting hwPre " << endl;
  delete hwPre;
  cout << "deleting hwPost" << endl;
  delete hwPost;
  cout << "deleting hw    " << endl;
  delete hw;

  cout << "END OF PROGRAM" << endl;
  exit(0);
}






/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

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
