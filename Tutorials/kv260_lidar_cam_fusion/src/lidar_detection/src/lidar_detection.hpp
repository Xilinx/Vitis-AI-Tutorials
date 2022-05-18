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

#ifndef _lidar_detection_hpp_
#define _lidar_detection_hpp_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "../../common/include/process_pointcloud.hpp"

class lidar_detection
{
	private:
		ros::Subscriber _sub_pointcloud;
		ros::Publisher _pub_boxes;
		ros::Publisher _pub_pointcloud_2d_projection_image;
		ros::Publisher _pub_ml_task_fps;
		process_pointcloud _pointcloud_processor;

		void recv_pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
		void publish_boxes(const std_msgs::Header header, vitis::ai::PointPillarsResult* result);
		void publish_output_image(const std_msgs::Header header, cv::Mat bevmat);
		void publish_fps(float processing_fps);

	public:
		lidar_detection(ros::NodeHandle n, ros::NodeHandle pn);
		~lidar_detection() { }
};

#endif
