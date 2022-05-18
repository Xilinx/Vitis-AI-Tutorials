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

#include "lidar_detection.hpp"

lidar_detection::lidar_detection(ros::NodeHandle n, ros::NodeHandle pn)
{
	_sub_pointcloud = n.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &lidar_detection::recv_pointcloud_callback, this);
	_pub_boxes = n.advertise<jsk_recognition_msgs::BoundingBoxArray>("/lidar_boxes", 10);
	_pub_pointcloud_2d_projection_image = n.advertise<sensor_msgs::Image>("/lidar_object_img", 10);
	_pub_ml_task_fps = n.advertise<std_msgs::Float32>("/lidar_fps", 10);
}

void lidar_detection::recv_pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	_pointcloud_processor.run(msg);
	publish_boxes(msg->header, _pointcloud_processor.get_result());
	publish_output_image(msg->header, _pointcloud_processor.get_bevmat());
	publish_fps(_pointcloud_processor.get_fps());
}

void lidar_detection::publish_boxes(const std_msgs::Header header, vitis::ai::PointPillarsResult* result)
{
	jsk_recognition_msgs::BoundingBoxArray boxes;
	boxes.header = header;
	for (int ii = 0; ii < result->ppresult.final_box_preds.size(); ii++)
	{
		jsk_recognition_msgs::BoundingBox box;
		box.header = header;		
		box.pose.position.x = result->ppresult.final_box_preds[ii][0];
		box.pose.position.y = result->ppresult.final_box_preds[ii][1];
		box.pose.position.z = result->ppresult.final_box_preds[ii][2];
		box.pose.orientation.w = 1.0;
		box.dimensions.x = result->ppresult.final_box_preds[ii][3];
		box.dimensions.y = result->ppresult.final_box_preds[ii][4];
		box.dimensions.z = result->ppresult.final_box_preds[ii][5];
		box.value = result->ppresult.final_scores[ii];
		box.label = ii;

		boxes.boxes.push_back(box);
	}
	
	_pub_boxes.publish(boxes);
}

void lidar_detection::publish_output_image(const std_msgs::Header header, cv::Mat bevmat)
{
	sensor_msgs::ImagePtr bevmat_msg = cv_bridge::CvImage(header, "bgr8", bevmat).toImageMsg();
	_pub_pointcloud_2d_projection_image.publish(bevmat_msg);
}

void lidar_detection::publish_fps(float processing_fps)
{
	std_msgs::Float32 fps_msg;
	fps_msg.data = processing_fps;
	_pub_ml_task_fps.publish(fps_msg);
}

