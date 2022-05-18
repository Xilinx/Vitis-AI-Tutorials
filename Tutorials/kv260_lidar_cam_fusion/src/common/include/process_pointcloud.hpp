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

#ifndef _process_pointcloud_hpp_
#define _process_pointcloud_hpp_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <velodyne_pcl/point_types.h>
#include <vitis/ai/pointpillars.hpp>
#include <opencv2/opencv.hpp>
#include "../include/display_transform_matrices.hpp"

typedef struct msg_offsets
{
	int x;
	int y;
	int z;
	int intensity;
} msg_offsets_t;

class process_pointcloud
{
	private:		
		vitis::ai::V1F _converted_pointcloud;
		float _ml_task_fps;
		std::unique_ptr<vitis::ai::PointPillars> _ml_task;
		vitis::ai::PointPillarsResult _result;
		cv::Mat _bevmat;
		display_transform_matrices _display_transform_matrices;

		void convert_pointcloud_msg_to_v1f(const sensor_msgs::PointCloud2ConstPtr& msg);
		void get_msg_offsets(const sensor_msgs::PointCloud2ConstPtr& msg, msg_offsets_t& msg_offsets);
		void detect_objects();
		void generate_display();

	public:
		process_pointcloud();
		void run(const sensor_msgs::PointCloud2ConstPtr& msg);
		vitis::ai::PointPillarsResult* get_result();
		cv::Mat get_bevmat();
		float get_fps();
		~process_pointcloud() { }
};

#endif
