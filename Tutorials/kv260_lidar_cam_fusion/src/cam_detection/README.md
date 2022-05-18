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

# Object Detection on Camera Displayed in RViz

First, make sure to download the yolov3 model to the correct location where Vitis AI will look for it in `/usr/share/vitis_ai_library/models`. If it's not there, download it with
```bash
mkdir -p /usr/share/vitis_ai_library/models
wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_bdd-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -O ~/Downloads/yolov3_bdd-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz
tar -xvf ~/Downloads/yolov3_bdd-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -C /usr/share/vitis_ai_library/models
```

Now that we can visualize the camera and LIDAR data together in RViz, we can now start processing the data. To do this, we will write C++ code for a new ROS node which subscribes to the topics that are publishing the camera and/or Lidar data. This allows us to run our own code in a callback whenever a new frame is received from the camera or a new point cloud is received from the LIDAR.

The source code for this demonstration is provided in the `kv260_lidar_cam_fusion/src/cam_detection` directory. It has the following directory structure:
```bash
.
├── CMakeLists.txt
├── launch
│   └── cam_detection.launch
├── package.xml
├── rviz
│   └── cam_detection.rviz
└── src
    ├── cam_detection.cpp
    ├── cam_detection.hpp
    ├── cam_detection_node.cpp
    └── process_result.hpp
```

We will start by discussing the source code itself then cover the details for building and launching the node.

## Writing the Code for the New Node
<details>
<summary>Click to see details</summary>
The new node is made up of the source files in `kv260_lidar_cam_fusion/src/cam_detection/src`. `cam_detection_node.cpp` spins up the actual node and creates an instance of `cam_detection` which is our custom object for handling the processing and is shown below

```C++
#include <ros/ros.h>
#include "cam_detection.hpp"

int main(int argc, char** argv)
{
	ros::init(argc, argv, "cam_detection");
	ros::NodeHandle n;
	ros::NodeHandle pn("~");

	cam_detection node(n, pn);

	ros::spin();
}
```

This is mostly boilerplate code for launching a new ROS node and instantiating an instance of our `cam_detection` node.

Next is the header file defining the `cam_detection` interface
```C++
#ifndef _cam_detection_hpp_
#define _cam_detection_hpp_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "../../common/include/process_image.hpp"

class cam_detection
{
	private:
		ros::Subscriber _sub_image;
		ros::Publisher _pub_output_image;
		ros::Publisher _pub_ml_task_fps;
		process_image _image_processor;
		
		void recv_image_callback(const sensor_msgs::ImageConstPtr& msg);
		void init_cam_subscriber(ros::NodeHandle pn);
		void publish_output_image(const std_msgs::Header header, cv::Mat img);
		void publish_fps(float processing_fps);

	public:
		cam_detection(ros::NodeHandle n, ros::NodeHandle pn);
		~cam_detection() { }
};

#endif
```

The first headers are for ROS packages that we are using
```C++
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
```
These include the main ROS api once again as well as libraries for interacting with the camera interface as well as OpenCV and OpenCV-to-ROS bridge.

Next are the files we need to run yolov3 neural network using Vitis AI. This image processing code is contained outside of this node in the `kv260_lidar_cam_fusion/src/common` directory. This is because this code is used in this node but also re-used in other nodes in this tutorial.
```C++
#include "../../common/include/process_image.hpp"
```

Now we can declare our class with member variables. ROS uses a publisher-subscriber model for communications between different nodes. This makes it very easy to scale the system to add sensors, processing nodes, displays, etc. Our new node subscribes to two existing camera nodes and publishes to one new node of its own
```C++
ros::Subscriber _sub_image;
ros::Publisher _pub_output_image;
ros::Publisher _pub_ml_task_fps;
```

The first subscription will attach to the `cv_camera` node to retreive the image data from the camera. We will publish the processed result (i.e. the raw image with bounding boxes drawn on it around detected objects) to the `_pub_output_image` variable. This node will also measure the performance of the yolov3 network by measuring the time spent in the function and converting to frames-per-second. This information is published via `_pub_ml_task_fps`.

We also create an instance of the image processor object mentioned previously.
```C++
process_image _image_processor;
```

Now we can declare our callbacks which will be called when a new message of the appropriate type is received.
```C++
void recv_image_callback(const sensor_msgs::ImageConstPtr& msg);
```

as well as several private member functions for intializing the camera and handling the final message publishing tasks
```C++
void init_cam_subscriber(ros::NodeHandle pn);
void publish_output_image(const std_msgs::Header header, cv::Mat img);
void publish_fps(float processing_fps);
```

Finally, our constructor which receives the node handles we declared in `cam_detection_node`
```C++
cam_detection(ros::NodeHandle n, ros::NodeHandle pn);
```

Now we can look at the implementation of the `cam_detection` class.
```C++
#include "cam_detection.hpp"

cam_detection::cam_detection(ros::NodeHandle n, ros::NodeHandle pn)
{
	init_cam_subscriber(pn);
	_pub_output_image = n.advertise<sensor_msgs::Image>("/cam_object_img", 1);
	_pub_ml_task_fps = n.advertise<std_msgs::Float32>("/cam_fps", 1);
}

void cam_detection::init_cam_subscriber(ros::NodeHandle pn)
{
	std::string cam_name;
	pn.param("cam", cam_name, std::string("cam"));
	ros::NodeHandle cam_nh(cam_name);
	
	_sub_image = cam_nh.subscribe("/cam/image_raw", 1, &cam_detection::recv_image_callback, this);
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
```

The main task of the constructor is to initializes the subscribers/publishers by attaching the callback functions (which are called each time something publishes a message of that type)
```C++
init_cam_subscriber(pn);
_pub_output_image = n.advertise<sensor_msgs::Image>("/cam_object_img", 1);
_pub_ml_task_fps = n.advertise<std_msgs::Float32>("/cam_fps", 1);
```

where the `init_cam_subscriber()` method is implemented as follows
```C++
std::string cam_name;
pn.param("cam", cam_name, std::string("cam"));
ros::NodeHandle cam_nh(cam_name);

_sub_image = cam_nh.subscribe("/cam/image_raw", 1, &cam_detection::recv_image_callback, this);
```

The first argument in the `.subscribe()` method is a string that corresponds to the node to which you want to subscribe. The third argument is the callback function that attaches to it.

Note also that the `.advertise()` method publishes to a new node named `/cam_object_image`. This name will show up in RViz and we can use it to display the processed output image.

Most of the work happens in the `recv_image_callback()` function where the image processor object implements the object detection and the results are published as shown below.
```C++
_image_processor.run(msg);
publish_output_image(msg->header, _image_processor.get_output_image());
publish_fps(_image_processor.get_fps());
```

The implementation of the image processor object will be discussed in the next subsection.

To publish the output image with bounding boxes, the `publish_output_image` function is called to convert the output `cv::Mat` image from Vitis AI back into ROS format and publishes it as a new message
```C++
sensor_msgs::ImagePtr output_img_msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
	_pub_output_image.publish(output_img_msg);
```

We also publish the output image as a single floating point value
```C++
std_msgs::Float32 fps_msg;
fps_msg.data = processing_fps;
_pub_ml_task_fps.publish(fps_msg);
```

### Image Processor Class

The `cam_detection` node uses an instance of the `process_image` class to handle the image processing task (in this case, object detection and classification). This provides a simple abstraction around the Vitis AI library and other housekeeping tasks. This interface is defined in `kv260_lidar_cam_fusion/src/common/include/process_image.hpp` shown below
```C++
#ifndef _process_image_hpp_
#define _process_image_hpp_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>

class process_image
{
	private:		
		cv::Mat _img;
		float _ml_task_fps;
		std::unique_ptr<vitis::ai::YOLOv3> _ml_task;
		vitis::ai::YOLOv3Result _result;
		
		void convert_msg_to_mat(const sensor_msgs::ImageConstPtr& msg);
		void detect_objects();
		void process_result();
		cv::Scalar get_color(int label);

	public:
		process_image();
		void run(const sensor_msgs::ImageConstPtr& msg);
		void get_result();
		cv::Mat get_output_image();
		float get_fps();
		~process_image() { }
};

#endif
```

We again include ROS and OpenCV functions that will be used
```C++
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
```

as well as the Vitis AI library functions specifically for yolov3
```C++
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>
```

There are several private member variables which hold the received image and processing results
```C++
cv::Mat _img;
float _ml_task_fps;
std::unique_ptr<vitis::ai::YOLOv3> _ml_task;
vitis::ai::YOLOv3Result _result;
```

Note that `_ml_task` is the actual Vitis AI object instance which will perform the hardware-accelerated neural network processing. The `_result` structure is the output from the neural network containing a vector of bounding box locations for detected objects.

Next, several private member functions are declared. The main public API is shown below
```C+++
process_image();
void run(const sensor_msgs::ImageConstPtr& msg);
void get_result();
cv::Mat get_output_image();
float get_fps();
~process_image() { }
```

Aside from the constructor/destructor, the primary methods are:
* `run()` - Run the image processing on the newly received message from the camera
* `get_result()` - Accessor method to retrieve the result struct from the processing
* `get_output_image()` - Accessor method to retrieve the result OpenCV Mat image containing the input image with colored bounding boxes drawn around detected objects
* `get_fps()` - Accessor method to retrieve the calculated frames per second of the network

The `process_image` class is implemented in `kv260_lidar_cam_fusion/src/common/src/process_image.cpp`.

The constructor simply initializes the `_ml_task` member by calling the `vitis::ai::YOLOv3::create` method
```C++
process_image::process_image()
{
	_ml_task = vitis::ai::YOLOv3::create("yolov3_bdd", true);
}
```

where the "yolov3_bdd" argument is the model name that will be used. This model must exist in the `/usr/share/vitis_ai_library/models` directory which is where Vitis AI library will look.

The `run()` method does the actual processing of the received message and is shown below
```C++
void process_image::run(const sensor_msgs::ImageConstPtr& msg)
{
	convert_msg_to_mat(msg);
	detect_objects();
	process_result();
}
```

We first convert the ROS message format (`sensor_msgs::ImageConstPtr`) to cv::Mat
```C++
void process_image::convert_msg_to_mat(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	_img = cv_ptr->image;
}
```

Then that image is processed by Vitis AI Yolov3 neural network to detect/classify objects in that image
```C++
void process_image::detect_objects()
{
	ros::Time begin = ros::Time::now();
	_result = _ml_task->run(_img);
	ros::Time end = ros::Time::now();
	
	_ml_task_fps = 1/((end-begin).toSec());
}
```

Note the use of the builtin `ros::Time` timestamping methods to calculate the performance of the network according to the amount of time that elapses during the `_ml_task->run()` execution.

Finally, bounding boxes are drawn on the output image by looping over the `_result.bboxes` bounding box vector and drawing different color boxes around those objects using `cv::rectangle()`.
```C++
void process_image::process_result()
{
	for (const auto bbox : _result.bboxes)
	{
		int label = bbox.label;
		float xmin = bbox.x * _img.cols + 1;
		float ymin = bbox.y * _img.rows + 1;
		float xmax = xmin + bbox.width * _img.cols;
		float ymax = ymin + bbox.height * _img.rows;
		float confidence = bbox.score;
		if (xmax > _img.cols)
		{
			xmax = _img.cols;
		}
		
		if (ymax > _img.rows)
		{
			ymax = _img.rows;
		}
		
		cv::rectangle(_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), get_color(label), 4, 1, 0);
	}
}
```

</details>

## Building the New Node
<details>
<summary>Click to see steps</summary>

ROS uses the catkin build tool which is like an extension of CMake making it easier to bring in ROS packages into the build. As such, any given package is normally compiled with CMake and we will follow this convention. As such, the `kv260_lidar_cam_fusion/src/cam_detection/CMakeLists.txt` is created as normal but also includes a few ROS/catkin specifics
```cmake
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
```

A few things are worth noting from this code. First, the project is called `cam_detection` which will be used when finding the node to launch with `roslaunch` later
```cmake
project(cam_detection)
```

Vitis AI requires C++17 so we tell this to CMake with
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

Note that `roscpp` and `cv_bridge` are added as dependencies. `roscpp` brings in all the C++ libraries and headers and `cv_bridge`  brings in the interface between the ROS message from `cv_camera` and OpenCV.
```cmake
find_package (catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
)
```

OpenCV and Vitis AI are also added which is what we'll use to grab frames from the camera and do some processing
```cmake
find_package(OpenCV REQUIRED)
find_package(vitis_ai_library REQUIRED)

catkin_package(
	DEPENDS OpenCV
)
```

Everything after that is standard CMake to add necessary headers/libaries/executables for building the entire project.

So that explains the `CMakeLists.txt` for actually compiling the node. Now we need to integrate this into the catkin system to build it. To do this, we create a 'package manifest' file `kv260_lidar_cam_fusion/src/cam_detection/package.xml`.

```xml
<?xml version="1.0"?>
<package format="2">
	<name>cam_detection</name>
	<version>0.0.0</version>
	<description>Object detection using camera on /dev/video0 using Rviz</description>
	<license>none</license>
	<maintainer email="bwiec@xilinx.com">Brian Wiec</maintainer>
	<buildtool_depend>catkin</buildtool_depend>
	<depend>roscpp</depend>
	<depend>cv_bridge</depend>
</package>
```

This XML file is mostly self-explanatory. It simply needs to be placed in the root directory of the package so catkin can tie the code to the launch file later. Note that the name matches the project name from the CMakeLists.txt
```xml
<name>cam_detection</name>
```

Now we are ready to build
```bash
cd kv260_lidar_cam_fusion/
catkin_make
```

*Note:* This is actually building all the nodes under `kv260_lidar_cam_fusion`.

</details>

## Launching the New Node
<details>
<summary>Click to see steps</summary>

We are now ready to launch the nodes to run this demonstration. This step will launch not just our new node, but also the other nodes we're using (i.e. cv_camera, RViz, etc). This is handled by another XML file called a 'launch' file in `kv260_lidar_cam_fusion/src/cam_detection/launch/cam_detection.launch`
```xml
<?xml version="1.0"?>
<launch>
	<!-- Static TF transforms -->
	<node pkg="tf2_ros" type="static_transform_publisher" name="map_to_world" args="0.0 0.0 0.0 0.0 0.0 0.0 map world" />

	<!-- Launch Camera node (optional based on command line - if not, expects you to play a .bag file) -->
	<group if="$(eval arg('omit_sensor') == 0)">
		<node pkg="cv_camera" type="cv_camera_node" name="cam" output="screen" />
		<param name="/cam/image_width" value="640" />
		<param name="/cam/image_height" value="480" />
	</group>

	<!-- Launch Camera processing node -->
	<node pkg="cam_detection" type="cam_detection" name="cam_detection" output="screen" />
	
	<!-- Launch ml task FPS display node -->
	<group ns="fps_cam">
		<remap from="/fps" to="/cam_fps" />
		<remap from="/fps_marker" to="/cam_fps_marker" />
		<include file="$(find display_fps)/launch/display_fps.launch" />
	</group>
</launch>
```

This file allows you to launch multiple nodes simultaneously and pass parameters to each one. The first node simply launches the `cv_camera` node. Note the *name* given is *cam* which matches the node handle from the C++ code.
```xml
<!-- Launch Camera node (optional based on command line - if not, expects you to play a .bag file) -->
<group if="$(eval arg('omit_sensor') == 0)">
	<node pkg="cv_camera" type="cv_camera_node" name="cam" output="screen" />
	<param name="/cam/image_width" value="640" />
	<param name="/cam/image_height" value="480" />
</group>
```
By default, this will open `/dev/video0`, but this can be changed via parameter. the `image_width` and `image_height` are set via parameters to retrieve a 640x480 resolution image from the sensor driver. Note that this node can be bypassed by passing `omit_sensor:=1` when using `roslaunch` to launch the node. This is useful for testing with rosbags instead of the actual sensor.

Now we can launch our custom node
```xml
<!-- Launch Camera processing node -->
<node pkg="cam_detection" type="cam_detection" name="cam_detection" output="screen" />
```
Note that the `pkg` is set to `cam_detection` which matches our package manifest file. The `type` set to `cam_detection` matches the CMakeLists build project.

Finally, we launch a custom `display_fps` node which simply converts a floating point message to `visualization_msgs::Marker` for display in RViz
```xml
<!-- Launch ml task FPS display node -->
<group ns="fps_cam">
	<remap from="/fps" to="/cam_fps" />
	<remap from="/fps_marker" to="/cam_fps_marker" />
	<include file="$(find display_fps)/launch/display_fps.launch" />
</group>
```

The `remap` tags in the node allow you to re-map topic names. This is useful for allowing you to change the topic name that a node interacts with without having to change any code or re-compile.

We can now run the demonstration
```bash
roslaunch cam_detection cam_detection.launch &
```
*Note:* If we have set everything up correctly, you will be able to use tab-completion on the previous line

Once the node is up and running, launch RViz
```bash
rviz -d src/cam_detection/rviz/cam_detection.rviz
```

</details>

# Support

GitHub issues will be used for tracking requests and bugs. For questions go to [forums.xilinx.com](http://forums.xilinx.com/).

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0]( http://www.apache.org/licenses/LICENSE-2.0 )



Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

<p align="center"> XD016 | &copy; Copyright 2021–2022 Xilinx, Inc.</p>
