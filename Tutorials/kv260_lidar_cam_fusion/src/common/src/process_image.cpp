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
 
#include "process_image.hpp"

process_image::process_image()
{
	//_ml_task = vitis::ai::YOLOv3::create("yolov3_bdd", true);
	_ml_task = vitis::ai::YOLOv3::create("yolov3_adas_pruned_0_9", true);
}

void process_image::run(const sensor_msgs::ImageConstPtr& msg)
{
	convert_msg_to_mat(msg);
	detect_objects();
	process_result();
}

void process_image::convert_msg_to_mat(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	_img = cv_ptr->image;
}

void process_image::detect_objects()
{
	ros::Time begin = ros::Time::now();
	_result = _ml_task->run(_img);
	ros::Time end = ros::Time::now();
	
	_ml_task_fps = 1/((end-begin).toSec());
}

void process_image::process_result()
{
	for (const auto bbox : _result.bboxes)
	{
		int label = bbox.label;
		float xmin = bbox.x * _img.cols + 1;
		float ymin = bbox.y * _img.rows + 1;
		float xmax = xmin + bbox.width * _img.cols;
		float ymax = ymin + bbox.height * _img.rows;
		float confidence = bbox.score;
		if (xmax > _img.cols)
		{
			xmax = _img.cols;
		}
		
		if (ymax > _img.rows)
		{
			ymax = _img.rows;
		}
		
		cv::rectangle(_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), get_color(label), 4, 1, 0);
	}
}

cv::Scalar process_image::get_color(int label)
{
	int c[3];
	for (int ii = 1, jj = 0; ii <= 9; ii *= 3, jj++)
	{
		c[jj] = ((label / ii) % 3) * 127;
	}
	
	return cv::Scalar(c[2], c[1], c[0]);
}

void process_image::get_result()
{
	ROS_WARN("process_pointcloud::get_result() not yet implemented");
}

cv::Mat process_image::get_output_image()
{
	return _img;
}

float process_image::get_fps()
{
	return _ml_task_fps;
}

