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

#include "mipiInput.h"

void mipiInput::err(std::string error) {
	std::cout << error << ": " << std::to_string(errno) << std::endl;
	if(!ioctl(vfd, VIDIOC_STREAMOFF, &type))
	{
		std::cout << "terminated old stream" << std::endl;
	}
	close(vfd);
	exit(-1);
}

void mipiInput::err(std::string error, int num) {
	err(error);
}

mipiInput::mipiInput(std::string name)
{

	if(!ioctl(vfd, VIDIOC_STREAMOFF, &type))
	{
		std::cout << "terminated old stream" << std::endl;
	}

	devname = name;
	init();
}

void mipiInput::init(){

	if(readied){
		err("Unhandled reinit");
	}

	vfd = open(devname.c_str(), O_RDWR, 0);
	if (vfd < 0) {
		err("Can't open");
	}

	if (ioctl(vfd, VIDIOC_QUERYCAP, &cap))
	{
		err("Device not supported");
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) || !(cap.capabilities & V4L2_CAP_STREAMING))
	{
		err("Device not supported");
	}

	fmt = { 0 };
	fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

	if (ioctl(vfd, VIDIOC_G_FMT, &fmt))
	{
		err("Unable to get V4L2 device format");
	}

	fmt.fmt.pix.width = 640;
	fmt.fmt.pix.height = 480;
	fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_RGB24;
	fmt.fmt.pix_mp.num_planes = 1;
	fmt.fmt.pix_mp.plane_fmt[0].sizeimage = (640 * 480 * 3) + 1;
	fmt.fmt.pix_mp.plane_fmt[0].bytesperline = 1920;

	if (ioctl(vfd, VIDIOC_S_FMT, &fmt))
	{
		err("Unable to swap format");
	}

	struct v4l2_requestbuffers req = {0};
	req.count = 10;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	req.memory = V4L2_MEMORY_MMAP;

	if (ioctl(vfd, VIDIOC_REQBUFS, &req))
	{
		err("Request Buffer Failure");
	}

	for (int _bufcnt=0; _bufcnt<QSIZE; _bufcnt++) {

		struct v4l2_buffer buf;
		struct v4l2_plane _planes[1];
		buf = {0};

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.length = 1;
		buf.m.planes = _planes;

		buf.index=_bufcnt;

		if(ioctl(vfd, VIDIOC_QUERYBUF, buf))
		{
			err("Querying Buffer Buffer " + std::to_string(_bufcnt));
		}

		internalbuffers[_bufcnt] = (uint8_t*)(mmap (NULL, buf.m.planes[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, vfd, buf.m.planes[0].m.mem_offset));
		internalMats[_bufcnt] = new cv::Mat(480,640,CV_8UC3,(void*)(internalbuffers[_bufcnt]));
	}

	readied=true;

	return;
}

void mipiInput::enqueue() {
	struct v4l2_buffer buf;
	struct v4l2_plane _planes[1];
	buf = {0};

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.length = 1;
	buf.m.planes = _planes;
	buf.index = index_in++;
	if (index_in == QSIZE) {
		index_in = 0;
	}

	if(ioctl(vfd, VIDIOC_QBUF, &buf))
	{
		err("Queue Buffer failed");
	} else {
		// std::cout << "q success: " << buf.index << std::endl;
	}
}

void mipiInput::rotateBuf() {
	if(!readied){
		err("Buffers not allocated");
	}

	struct v4l2_buffer buf;
	struct v4l2_plane _planes[1];
	buf = {0};

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.length = 1;
	buf.m.planes = _planes;
	buf.index = 0;

  if (ioctl(vfd, VIDIOC_DQBUF, &buf))
  {
    std::cout << "dqueue failed" << std::endl;
  } else {
		// std::cout << "Successful dqueue: " << buf.index << std::endl;
	}

	index_out = buf.index;
}

void mipiInput::getMat(cv::Mat& out){
	cv::InputArray t1(*(internalMats[index_out]));
	cv::OutputArray t2(out);
	//
	cv::cvtColor(t1, t2, cv::COLOR_RGB2BGR);
	// return internalMats[index_out];
}

void mipiInput::debugShowAll(){
	for (int i=0; i<QSIZE; i++) {
		cv::imshow("test", *(internalMats[i]));
		cv::waitKey(0);
	}
}

void mipiInput::startStream(){
	if (streaming) {
		return;
	}
	if(ioctl(vfd, VIDIOC_STREAMON, &type))
	{
		err("Start Capture failed");
	} else {
		// std::cout << "Started stream" << std::endl;
		streaming=1;
	}
}

void mipiInput::stopStream(){
	if (!streaming) {
		return;
	}
	if(ioctl(vfd, VIDIOC_STREAMOFF, &type))
	{
		err("Stream stop failed");
	} else {
		streaming=0;
	}
}

mipiInput::~mipiInput()
{

	err("Object removed");

	if(ioctl(vfd, VIDIOC_STREAMOFF, &type))
	{
		err("Stream stop failed");
	}


}
