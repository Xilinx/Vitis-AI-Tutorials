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

#ifndef _display_fps_hpp_
#define _display_fps_hpp_

#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/Marker.h>

class display_fps
{
	private:
		ros::Subscriber _sub_fps;
		ros::Publisher _pub_fps_marker;
		
		float y_position;

		void recv_fps_callback(const std_msgs::Float32ConstPtr& msg);

	public:
		display_fps(ros::NodeHandle n, ros::NodeHandle pn);
		~display_fps() { }
};

#endif
