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


int main() {
	mipiInput test("/dev/video0");

	cv::Mat newmat;

	std::cout << "Starting Test" << std::endl;

	//Note, we're using a glrenderer opbject to use Mali to display
	cv::VideoWriter output("appsrc ! glimagesink", cv::VideoWriter::fourcc('R', 'G', 'B', '3'), 30.0, cv::Size(640,480), true);

	//Use 300 frames - camera input is set to 30fps (10 seconds video)
	for(int i=0; i<3000; i++) {
		//Enqueue buffer for next frame
		test.enqueue();

		//Start stream only if necessary
		test.startStream();

		//Dequeue buffer from frame
		test.rotateBuf();

		//Place frame into cv
		test.getMat(newmat);
	}
}
