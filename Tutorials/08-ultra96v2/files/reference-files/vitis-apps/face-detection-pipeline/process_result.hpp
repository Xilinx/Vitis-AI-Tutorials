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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void process_result(cv::Mat *m1, const xilinx::ai::FaceDetectResult &result,
                       bool is_jpeg, long int time) {
  for (const auto &r : result.rects) {
    // LOG_IF(INFO, is_jpeg) << " " << r.score << " " //
    //                       << r.x << " "            //
    //                       << r.y << " "            //
    //                       << r.width << " "        //
    //                       << r.height;
    cv::rectangle((*m1),
                  cv::Rect{cv::Point(r.x * m1->cols, r.y * m1->rows),
                           cv::Size{(int)(r.width * m1->cols),
                                    (int)(r.height * m1->rows)}},
                  0xff);
  }
  // int fps = 1000/time;
  return;
}
