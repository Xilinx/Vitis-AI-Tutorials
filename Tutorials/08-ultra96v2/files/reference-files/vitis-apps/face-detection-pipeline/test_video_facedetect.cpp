/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xilinx/ai/demo.hpp>
#include <xilinx/ai/facedetect.hpp>
#include <xilinx/ai/nnpp/facedetect.hpp>
#include "./process_result.hpp"
#include "mipiInput.h"

using namespace std;

int main(int argc, char *argv[]) {

  mipiInput test("/dev/video0");
  cv::VideoCapture inza("/dev/video1");

  //Note, we're using a glrenderer opbject to use Mali to display
  cv::VideoWriter output("appsrc ! glimagesink", cv::VideoWriter::fourcc('R', 'G', 'B', '3'), 30.0, cv::Size(640,480), true);

	std::cout << "starting test" << std::endl;

  auto ml_task = xilinx::ai::FaceDetect::create("densebox");

  auto t_last = std::chrono::steady_clock::now();
  auto t_now = std::chrono::steady_clock::now();

  cv::Mat othernewmat;

  while (true) {
    t_now = std::chrono::steady_clock::now();
    auto d_milli = std::chrono::duration_cast<std::chrono::milliseconds>( t_now - t_last ).count();
    t_last = t_now;

    test.rotateBuf();

    // cv::Mat newmat;
    test.getMat(othernewmat);

    // test.getMat();//->copyTo(othernewmat);
    // inza >> newmat;

    auto res = ml_task->run(othernewmat);

    process_result(&othernewmat, res, false, d_milli);

    //The getMat function lags the active/queued buffer by 4 frames
		output.write(othernewmat);
  }

}
