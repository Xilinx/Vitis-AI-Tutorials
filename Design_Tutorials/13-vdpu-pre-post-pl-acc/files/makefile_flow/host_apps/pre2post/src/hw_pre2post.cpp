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

#include "hw_pre2post.hpp"
#include <random>
#include <vitis/ai/profiling.hpp>


const char *g_xclbin_Filename = "/mnt/sd-mmcblk0p1/dpu.xclbin";
using namespace std;

vector<int8_t> random_vector_char(size_t sz) {
  static std::mt19937 rng(100);
  static std::uniform_int_distribution<char> dist;
  auto ret = vector<int8_t>(sz);
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = dist(rng);
  }
  return ret;
}

void argmax_c(const int8_t *input, unsigned int cls, unsigned int group,
              uint8_t *output) {
  for (unsigned int i = 0; i < group; ++i) {
    auto max_ind = max_element(input + i * cls, input + (i + 1) * cls);
    int8_t posit = (int8_t)distance(input + i * cls, max_ind);
    // if (i < 100) cout << " posit= " << posit - 0 << endl;  //
    output[i] = posit;
  }
}

void compare(int cls, int group, signed char *input, uint8_t *output1,
             uint8_t *output2) {
  for (auto g = 0; g < group; ++g) {
    auto idx = g;
    auto diff = output1[idx] - output2[idx];
    if ((diff != 0.0 && std::abs(diff) > 0.001)) {
      if (g < 10)
        cout << " g=" << g << " "  //
             << (int)input[idx * cls] << " " << (int)input[idx * cls + 1] << " "
             << ": " << output1[idx] - 0 << " " << output2[idx] - 0 << " "
             << diff << endl;
    }
  }
}

void HW::init(const char *ip_name) {
  device_ = xrtDeviceOpen(0);
  // We are not validating the user input. A general program should verify it
  // exists.
  xclbin_ = xrtXclbinAllocFilename(g_xclbin_Filename);
  xrtDeviceLoadXclbinHandle(device_, xclbin_);
  xuid_t xclbin_uuid;
  xrtXclbinGetUUID(xclbin_, xclbin_uuid);
  printf("XCLBIN UUID 0x%px\n", xclbin_uuid);

  kernel_ = xrtPLKernelOpen(device_, xclbin_uuid, ip_name);
}

void HW::close() {
  xrtXclbinFreeHandle(xclbin_);
}

HW_PRE::~HW_PRE() {
  // xrtBOUnmap(in_bomapped_);
  // xrtBOUnmap(out_bomapped_);
  xrtBOFree(img_inp_);
  xrtBOFree(img_out_);
  xrtRunClose(kernel_rhdl_);
  xrtKernelClose(kernel_);
  xrtDeviceClose(device_);
  //xrtXclbinFreeHandle(xclbin_);
}
HW_POST::~HW_POST() {
  // xrtBOUnmap(in_bomapped_);
  // xrtBOUnmap(idx_bomapped_);
  // xrtBOUnmap(max_bomapped_);
  xrtBOFree(input_);
  xrtBOFree(idx_out_);
  xrtBOFree(max_out_);
  xrtRunClose(kernel_rhdl_);
  xrtKernelClose(kernel_);
  xrtDeviceClose(device_);
  //xrtXclbinFreeHandle(xclbin_);
}



void HW_PRE::load() {
  size_t size_inp_bytes = MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char) * 3;
  // create a buffer for the input image
  img_inp_ = xrtBOAlloc(device_, size_inp_bytes, 0, 0);
  in_bomapped_ = reinterpret_cast<uint32_t *>(xrtBOMap(img_inp_));
  memset(in_bomapped_, 0x0, size_inp_bytes);
  printf("PreProc: input       memory virtual  addr %p\n", in_bomapped_);
  printf("preProc: input       memory physical addr %p\n", xrtBOAddress(img_inp_));
  // create a buffer for the output image
  img_out_ = xrtBOAlloc(device_, size_inp_bytes, 0, 0);
  out_bomapped_ = reinterpret_cast<int32_t *>(xrtBOMap(img_out_));
  memset(out_bomapped_, 0x00000000, size_inp_bytes);
  printf("PreProc: output      memory virtual  addr %p\n", out_bomapped_);
  printf("PreProc: output      memory physical addr %p\n", xrtBOAddress(img_out_));
  printf("PreProc:  size_inp_bytes=%7d\n", size_inp_bytes);

  //xrtURStart(0, "HW PRE kernel start", "HW PRE kernel start");  //DB: if you uncomment it, it will stall forever
}

void HW_PRE::run(const Mat &input_img, int8_t *out_img, vector<float> means,
		 vector<float> scales, int dpu_fixpos)
{
	__TIC__(run);
	size_t size_in_bytes = MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char) * 3;
	memcpy(in_bomapped_, input_img.data, size_in_bytes);
	// Execute
	__TIC__(runkernel);
	kernel_rhdl_ = xrtKernelRun(kernel_, img_inp_, img_out_,
		 means[0], means[1], means[2],
		 scales[0], scales[1], scales[2],
		 dpu_fixpos, MAX_HEIGHT, MAX_WIDTH); //DB: replaced 64.0f with dpu_fixpos
	auto state = xrtRunWait(kernel_rhdl_);  // wait for the kernel to finish
	__TOC__(runkernel);
	memcpy(out_img, out_bomapped_, size_in_bytes);
	//events_.mark("kernel finished");
	__TOC__(run);
}

void HW_POST::load() {
  size_t size_inp_bytes =  MAX_HEIGHT*MAX_WIDTH*MAX_NUM_CLASSES*sizeof(signed char);  // 960 x 416 * 28 (and not 12!)
  size_t size_out_bytes =  MAX_HEIGHT*MAX_WIDTH*sizeof(unsigned char);                //  960 x 416
  printf("PostProc: size_inp_bytes=%7d\n", size_inp_bytes);
  printf("PostProc: size_out_bytes=%7d\n", size_out_bytes);
  // create a buffer for the input image
  input_ = xrtBOAlloc(device_, size_inp_bytes, 0, 0);
  in_bomapped_ = reinterpret_cast<int32_t *>(xrtBOMap(input_));
  memset(in_bomapped_, 0x0, size_inp_bytes);
  printf("PostProc: input      memory virtual  addr %p\n", in_bomapped_);
  printf("PostProc: input      memory physical addr %p\n", xrtBOAddress(input_));
  // create buffers for the output images
  idx_out_ = xrtBOAlloc(device_, size_out_bytes, 0, 0);
  idx_bomapped_ = reinterpret_cast<uint32_t *>(xrtBOMap(idx_out_));
  memset(idx_bomapped_, 0x00000000, size_out_bytes);
  max_out_ = xrtBOAlloc(device_, size_out_bytes, 0, 0);
  max_bomapped_ = reinterpret_cast<uint32_t *>(xrtBOMap(max_out_));
  memset(max_bomapped_, 0x00000000, size_out_bytes);
  printf("PostProc: idx output memory virtual  addr %p\n", idx_bomapped_);
  printf("PostProc: idx output memory physical addr %p\n", xrtBOAddress(idx_out_));
  printf("PostProc: max output memory virtual  addr %p\n", max_bomapped_);
  printf("PostProc: max output memory physical addr %p\n", xrtBOAddress(max_out_));
  //xrtURStart(0, "POSTPROC kernel start", "POSTPROC kernel start");  //DB: if you uncomment it, there will stall forevever
}

void HW_POST::run(int8_t *input, uint8_t *out_max, uint8_t *out_ind, int dpu_fixpos)
{
	size_t size_in_bytes = MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char);
	__TIC__(allpost);
	memcpy(in_bomapped_, input, size_in_bytes*MAX_NUM_CLASSES); //DB: added memcpy
	// Execute
	__TIC__(startpost);
	kernel_rhdl_ = xrtKernelRun(kernel_, input_, max_out_, idx_out_, dpu_fixpos, //replaced 1.0f with dpu_fixpos
			MAX_HEIGHT, MAX_WIDTH);
	auto state = xrtRunWait(kernel_rhdl_);  // wait for the kernel to finish
	__TOC__(startpost);
	memcpy(out_max, max_bomapped_, size_in_bytes); 	 //DB: added memcpy
	memcpy(out_ind, idx_bomapped_, size_in_bytes);   //DB: added memcpy
	__TOC__(allpost);
	// events_.mark("kernel finished");
}
