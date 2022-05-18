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
project(cam_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package (catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
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
	../common/include
)

add_executable(cam_detection
	../common/src/process_image.cpp
	src/cam_detection.cpp
	src/cam_detection_node.cpp
)

add_dependencies(cam_detection
	${catkin_EXPORTED_TARGETS}
)
target_link_libraries(cam_detection
	${catkin_LIBRARIES}
	${vitis_ai_library_LIBRARIES}
	vitis_ai_library-yolov3
	glog
)

