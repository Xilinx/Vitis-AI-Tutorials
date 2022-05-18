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
 
#include "process_pointcloud.hpp"

namespace vitis
{
	namespace ai
	{
		extern cv::Mat bev_preprocess(const V1F& PointCloud);
	}
}

process_pointcloud::process_pointcloud()
{
	_ml_task = vitis::ai::PointPillars::create("pointpillars_kitti_12000_0_pt", "pointpillars_kitti_12000_1_pt");
}

void process_pointcloud::run(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	convert_pointcloud_msg_to_v1f(msg);
	detect_objects();
	generate_display();
}

void process_pointcloud::convert_pointcloud_msg_to_v1f(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	// Convert to V1F format V1F is defined in <vitis/ai/pointpillars.hpp> and is only a vector of floats
	msg_offsets_t msg_offsets;
	get_msg_offsets(msg, msg_offsets);
	
	_converted_pointcloud.clear();
	for (int ii = 0; ii < msg->width; ii++)
	{
		float x, y, z, intensity;
		x = *(float*)(&msg->data[0] + (msg->point_step*ii) + msg_offsets.x);
		y = *(float*)(&msg->data[0] + (msg->point_step*ii) + msg_offsets.y);
		z = *(float*)(&msg->data[0] + (msg->point_step*ii) + msg_offsets.z);
		intensity = *(float*)(&msg->data[0] + (msg->point_step*ii) + msg_offsets.intensity);
		
		_converted_pointcloud.push_back(x);
		_converted_pointcloud.push_back(y);
		_converted_pointcloud.push_back(z);
		_converted_pointcloud.push_back(intensity/255.0); // HDL64E is different from VLP32C. HDL64E is a float value. VLP32C is int8
	}
}

void process_pointcloud::get_msg_offsets(const sensor_msgs::PointCloud2ConstPtr& msg, msg_offsets_t& msg_offsets)
{
	for (int ii = 0; ii < msg->fields.size(); ii++)
	{		
		if (msg->fields[ii].name == "x")
		{
			msg_offsets.x = msg->fields[ii].offset;
		}
		if (msg->fields[ii].name == "y")
		{
			msg_offsets.y = msg->fields[ii].offset;
		}
		if (msg->fields[ii].name == "z")
		{
			msg_offsets.z = msg->fields[ii].offset;
		}
		if (msg->fields[ii].name == "intensity" || msg->fields[ii].name == "i") // HDL64E is different from VLP32C
		{
			msg_offsets.intensity = msg->fields[ii].offset;
		}
	}
}

void process_pointcloud::detect_objects()
{
	ros::Time begin = ros::Time::now();
	_result = _ml_task->run(_converted_pointcloud);
	ros::Time end = ros::Time::now();
	
	_ml_task_fps = 1/((end-begin).toSec());
}

void process_pointcloud::generate_display()
{
	vitis::ai::ANNORET annoret;
	cv::Mat rgbmat;
	
	_bevmat = vitis::ai::bev_preprocess(_converted_pointcloud);
	_ml_task->do_pointpillar_display(_result, vitis::ai::E_BEV, _display_transform_matrices.get_display_data(), rgbmat, _bevmat, 1242, 375, annoret);
}

vitis::ai::PointPillarsResult* process_pointcloud::get_result()
{
	return &_result;
}

cv::Mat process_pointcloud::get_bevmat()
{
	return _bevmat;
}

float process_pointcloud::get_fps()
{
	return _ml_task_fps;
}

