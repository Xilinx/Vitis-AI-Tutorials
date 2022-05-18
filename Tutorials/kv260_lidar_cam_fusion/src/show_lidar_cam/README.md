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

# Display LIDAR Point Cloud and Camera Frames Simultaneously in RViz

RViz is an open-source visualization tool that is commonly used with ROS. It is capable of reading ROS topics and visualizing all different types of data. We will first simply open the camera and LIDAR data in RViz to become familiar with it and make sure all the hardware and software setup is correct. First, copy `rviz/lidar_and_cam.rviz` to the board. Next, we need to launch the ROS core:
```bash
roscore & # launch the ROS core
```
Once ROS core is up and running, we can launch nodes with
```bash
roslaunch show_lidar_cam show_lidar_cam.launch
```

RViz will now open and display both the camera image and point cloud
![show_lidar_cam](../../images/show_lidar_cam.gif)

## Displaying Results on Remote PC

RViz can take up a lot of cycles on the ARM A53 processor complex (PS) just to render the display. As an alternative, ROS topics can be sent over the network to other machines running ROS. That machine can then do the rendering for visualization and leave the more cycles on the A53's for more useful tasks. Setting this up is simply a matter of environment configuration.

On the host PC:
```bash
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=192.168.1.148
roscore &
rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 map velodyne 100 &
rviz &
```

On the KV260
```bash
export ROS_MASTER_URI=http://<host_pc_ip_address>:11311
export ROS_IP=<kv260_ip_address>
./run.sh lidar_cam_detection
```

# Support

GitHub issues will be used for tracking requests and bugs. For questions go to [forums.xilinx.com](http://forums.xilinx.com/).

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0]( http://www.apache.org/licenses/LICENSE-2.0 )



Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

<p align="center"> XD016 | &copy; Copyright 2021–2022 Xilinx, Inc.</p>
