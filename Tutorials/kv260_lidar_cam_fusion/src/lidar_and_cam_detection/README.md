<table>
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>LIDAR + Camera Fusion on KV260</h1>
   </td>
 </tr>
 <tr>
 <td align="center"><h1>KV260 ROS Tutorial</h1>
 </td>
 </tr>
</table>

# Simultaneous Object Detection on LIDAR Point Cloud and Camera Frames Displayed in RViz

This demonstration is a combination of both the `cam_detection` and `lidar_detection` nodes into a single node that uses Vitis AI and the DPU hardware accelerator to do neural network inferencing on both the image data from the camera and the pointcloud from the LIDAR.

In this page, we will not again discuss ROS basic concepts or general organizational information which were discussed in the [cam_detection](../cam_detection) section. Similarly, pointcloud processing specifics are covered in [lidar_detection](../lidar_detection). This page will simply discuss the details specific for combining both nodes.

The source code for this demonstration is provided in the `kv260_lidar_cam_fusion/src/lidar_and_cam_detection` directory. It has the following directory structure:
```bash
.
├── CMakeLists.txt
├── launch
│   └── lidar_and_cam_detection.launch
├── package.xml
├── README.md
├── rviz
│   └── lidar_and_cam_detection.rviz
└── src
    └── lidar_and_cam_detection_node.cpp
```

We will start by discussing the source code itself then cover the details for building and launching the node.

## Writing the Code for the New Node
<details>
<summary>Click to see details</summary>

Because we encapsulated the `cam_detection` and `lidar_detection` nodes into separate classes, combining them together is nearly trivial. The only code that needs to be written is in `kv260_lidar_cam_fusion/src/lidar_and_cam_detection/src/lidar_and_cam_detection_node.cpp`
```C++
#include <ros/ros.h>
#include "../../lidar_detection/src/lidar_detection.hpp"
#include "../../cam_detection/src/cam_detection.hpp"

int main(int argc, char** argv)
{
	ros::init(argc, argv, "lidar_and_cam_detection");
	ros::NodeHandle n;
	ros::NodeHandle pn("~");

	cam_detection cam_node(n, pn);
	lidar_detection lidar_node(n, pn);

	ros::spin();
}
```

All we need to do is instantiate both `cam_detection` and `lidar_detection` nodes. Each node attaches its own callbacks to the specific sensor events and does its own processing asynchronously. Each one takes ownership of the Vitis AI object as needed.

</details>

## Building the New Node
<details>
<summary>Click to see steps</summary>

To build the new node, we have to include all the dependencies needed for both `cam_detection` and `lidar_detection`

```cmake
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
```

and again build with catkin_make
```bash
cd kv260_lidar_cam_fusion/
catkin_make
```

</details>

## Launching the New Node
<details>
<summary>Click to see steps</summary>

The launch file requires some special attention when combining both nodes. The full file is shown below

```xml
<?xml version="1.0"?>
<launch>
	<!-- Static TF transforms -->
	<node pkg="tf2_ros" type="static_transform_publisher" name="map_to_velodyne" args="0.0 0.0 0.0 0.0 0.0 0.0 map velodyne" />
	
	<!-- Launch Sensor nodes (optional based on command line - if not, expects you to play a .bag file) -->
	<group if="$(eval arg('omit_sensor') == 0)">
		<!-- Launch Camera node -->
		<node pkg="cv_camera" type="cv_camera_node" name="cam" output="screen"/>
		<param name="/cam/image_width" value="640" />
		<param name="/cam/image_height" value="480" />
		
		<!-- Launch Lidar node -->
		<include file="$(find velodyne_pointcloud)/launch/VLP-32C_points.launch"/>
	</group>

	<!-- Launch Lidar processing node -->
	<node pkg="lidar_and_cam_detection" type="lidar_and_cam_detection" name="lidar_and_cam_detection" output="screen" />
	
	<!-- Launch ml task FPS display nodes -->
	<group ns="fps_lidar">
		<remap from="/fps" to="/lidar_fps" />
		<remap from="/fps_marker" to="/lidar_fps_marker" />
		<include file="$(find display_fps)/launch/display_fps.launch" />
	</group>
	
	<group ns="fps_cam">
		<remap from="/fps" to="/cam_fps" />
		<remap from="/fps_marker" to="/cam_fps_marker" />
		<include file="$(find display_fps)/launch/display_fps.launch" />
		<param name="y_position" value="6.0" />
	</group>
</launch>
```

We now launch both sensors according to the `omit_sensor` argument again.

Also, two different instances of `display_fps` are launched, one for each processing node. However, special care needs to be taken because both of these nodes, by default, subscribe to the same "/fps" topic and publish to the same "/fps_marker" topic. To avoid collisions, we use the "remap" capability of ROS to change the name of the topics for that specific instance only.

We can now run the demonstration
```bash
roslaunch lidar_and_cam_detection lidar_and_cam_detection.launch &
```

Once the node is up and running, launch RViz
```bash
rviz -d src/lidar_and_cam_detection/rviz/lidar_and_cam_detection.rviz
```

</details>

# Support

GitHub issues will be used for tracking requests and bugs. For questions go to [forums.xilinx.com](http://forums.xilinx.com/).

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0]( http://www.apache.org/licenses/LICENSE-2.0 )



Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

<p align="center"> XD016 | &copy; Copyright 2021–2022 Xilinx, Inc.</p>
