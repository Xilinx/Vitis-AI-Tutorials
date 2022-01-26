/*
 * Copyright 2021 Xilinx Inc.
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

#ifndef  _HW_PRE2POST_H_
#define  _HW_PRE2POST_H_


#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_profile.h"

//#define CL_HPP_CL_1_2_DEFAULT_BUILD
//#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
//#include <CL/cl2.hpp>
//#include "defines.hpp"

using namespace std;
using namespace cv;


// Customized buffer allocation for 4K boundary alignment
vector<int8_t> random_vector_char(size_t sz);
void argmax_c(const int8_t *input, unsigned int cls, unsigned int group,
              uint8_t *output);
void compare(int cls, int group, signed char *input, uint8_t *output1,
             uint8_t *output2);

template <typename T>
struct aligned_allocator {
  using value_type = T;
  T *allocate(std::size_t num) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T *>(ptr);
  }
  void deallocate(T *p, std::size_t num) { free(p); }
};

class HW {
 public:
  HW(const char *g_xclbin_Filename);
  xrtDeviceHandle get_DeviceHandle(void);
  xrtXclbinHandle get_XclbinHandle(void);
  ~HW(void);

 private:
  xrtDeviceHandle device_;
  xrtXclbinHandle xclbin_;
  xuid_t xclbin_uuid;
};


class HW_PRE  {
 public:
  HW_PRE(xrtDeviceHandle device_, xrtXclbinHandle xclbin_, const char *ip_name);
  void load(xrtDeviceHandle device_);
  void run(const Mat &input_img, int8_t *out_img, vector<float> means,
           vector<float> scales,  int dpu_fixpos); //DB added dpu_fixpos
  ~HW_PRE();

 private:
  xrtBufferHandle img_inp_, img_out_;
  uint32_t *in_bomapped_;
  int32_t *out_bomapped_;
  int MAX_HEIGHT = 224;
  int MAX_WIDTH =  224;
  xrtKernelHandle kernel_;
  xrtRunHandle kernel_rhdl_;
};

class HW_POST {
 public:
  HW_POST(xrtDeviceHandle device_, xrtXclbinHandle xclbin_, const char *ip_name);
  void load(xrtDeviceHandle device_);
  void run(int8_t *input, uint8_t *out_max, uint8_t *out_idx, int dpu_fixpos); //DB added dpu_fixpos
  ~HW_POST();

 private:
  xrtBufferHandle input_, max_out_, idx_out_;
  int32_t *in_bomapped_;
  uint32_t *idx_bomapped_;
  uint32_t *max_bomapped_;
  int MAX_HEIGHT = 224;
  int MAX_WIDTH  = 224;
  int MAX_NUM_CLASSES = 12;
  xrtKernelHandle kernel_;
  xrtRunHandle kernel_rhdl_;
};



#endif
