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

#include "cam_detection.hpp"

cam_detection::cam_detection(ros::NodeHandle n, ros::NodeHandle pn)
{
	init_cam_subscriber(pn);
	_pub_output_image = n.advertise<sensor_msgs::Image>("/cam_object_img", 10);
	_pub_ml_task_fps = n.advertise<std_msgs::Float32>("/cam_fps", 10);
}

void cam_detection::init_cam_subscriber(ros::NodeHandle pn)
{
	std::string cam_name;
	pn.param("cam", cam_name, std::string("cam"));
	ros::NodeHandle cam_nh(cam_name);
	
	_sub_image = cam_nh.subscribe("/cam/image_raw", 10, &cam_detection::recv_image_callback, this);
}

void cam_detection::recv_image_callback(const sensor_msgs::ImageConstPtr& msg)
{
	_image_processor.run(msg);
	publish_output_image(msg->header, _image_processor.get_output_image());
	publish_fps(_image_processor.get_fps());
}

void cam_detection::publish_output_image(const std_msgs::Header header, cv::Mat img)
{
	sensor_msgs::ImagePtr output_img_msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
	_pub_output_image.publish(output_img_msg);
}

void cam_detection::publish_fps(float processing_fps)
{
	std_msgs::Float32 fps_msg;
	fps_msg.data = processing_fps;
	_pub_ml_task_fps.publish(fps_msg);
}

