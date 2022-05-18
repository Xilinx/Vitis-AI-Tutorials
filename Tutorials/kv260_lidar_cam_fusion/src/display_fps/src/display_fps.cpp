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

#include "display_fps.hpp"

display_fps::display_fps(ros::NodeHandle n, ros::NodeHandle pn)
{
	n.param<float>("y_position", y_position, 4.0);

	_sub_fps = n.subscribe("/fps", 10, &display_fps::recv_fps_callback, this);
	_pub_fps_marker = n.advertise<visualization_msgs::Marker>("/fps_marker", 10);
}

void display_fps::recv_fps_callback(const std_msgs::Float32ConstPtr& msg)
{	
	visualization_msgs::Marker marker;
	marker.header.frame_id = "world";
	
	marker.text = std::to_string(msg->data)+"fps";
	marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker.action = visualization_msgs::Marker::ADD;
	
	 // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
	marker.pose.position.x = 0;
	marker.pose.position.y = y_position;
	marker.pose.position.z = 0;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;

	// Set the scale of the marker -- 1x1x1 here means 1m on a side
	marker.scale.x = 0.2;
	marker.scale.y = 0.2;
	marker.scale.z = 0.2;

	// Set the color -- be sure to set alpha to something non-zero!
	marker.color.r = 0.0f;
	marker.color.g = 1.0f;
	marker.color.b = 0.0f;
	marker.color.a = 1.0;
	
	_pub_fps_marker.publish(marker);
}

