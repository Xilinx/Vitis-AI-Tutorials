/*
 * Copyright 2022 Xilinx, Inc.
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

#ifndef _process_image_hpp_
#define _process_image_hpp_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>

class process_image
{
	private:		
		cv::Mat _img;
		float _ml_task_fps;
		std::unique_ptr<vitis::ai::YOLOv3> _ml_task;
		vitis::ai::YOLOv3Result _result;
		
		void convert_msg_to_mat(const sensor_msgs::ImageConstPtr& msg);
		void detect_objects();
		void process_result();
		cv::Scalar get_color(int label);

	public:
		process_image();
		void run(const sensor_msgs::ImageConstPtr& msg);
		void get_result();
		cv::Mat get_output_image();
		float get_fps();
		~process_image() { }
};

#endif
