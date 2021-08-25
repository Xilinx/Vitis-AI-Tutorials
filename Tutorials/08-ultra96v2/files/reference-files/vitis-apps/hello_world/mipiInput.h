/*
Copyright 2020 Xilinx

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "linux/videodev2.h"
#include <iostream>
#include <string>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <unistd.h>
#include <sys/mman.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#define QSIZE 8

class mipiInput
{
private:
    struct v4l2_format fmt;
    struct v4l2_streamparm parm;
    struct v4l2_capability cap;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    int vfd;

    int index_in = 0;
    int index_out = 0;
    int queuedBufs = 0;

    std::string devname;
    uint8_t* internalbuffers[QSIZE];

    cv::Mat* internalMats[QSIZE];
    cv::Mat cvted;

    bool readied = 0;
    bool streaming = 0;

public:

    mipiInput(std::string name);
    void init();
    void rotateBuf();
    void getMat(cv::Mat&);
    void debugShowAll();
    void stopStream();
    void startStream();
    void enqueue();

    void err(std::string error, int num);
    void err(std::string error);

    ~mipiInput();

};
