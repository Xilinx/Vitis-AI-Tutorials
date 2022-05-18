#
# Copyright 2022 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.0.2)
project(lidar_and_cam_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package (catkin REQUIRED COMPONENTS
	roscpp
	pcl_ros
	cv_bridge
	velodyne_pcl
)
find_package(OpenCV REQUIRED)
find_package(vitis_ai_library REQUIRED)

catkin_package(
	DEPENDS OpenCV	
)

include_directories(
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${vitis_ai_library_INCLUDE_DIRS}
	${velodyne_pcl_INCLUDE_DIRS}
	../common/include
	../lidar_detection/src
	../cam_detection/src
)

add_executable(lidar_and_cam_detection
	../common/src/process_image.cpp
	../common/src/display_transform_matrices.cpp
	../common/src/process_pointcloud.cpp
	../lidar_detection/src/lidar_detection.cpp
	../cam_detection/src/cam_detection.cpp
	src/lidar_and_cam_detection_node.cpp
)

add_dependencies(lidar_and_cam_detection
	${catkin_EXPORTED_TARGETS}
)
target_link_libraries(lidar_and_cam_detection
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${vitis_ai_library_LIBRARIES}
	vitis_ai_library-pointpillars
	vitis_ai_library-yolov3
	glog
)

