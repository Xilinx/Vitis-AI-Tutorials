#!/bin/bash
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

cd /home/ubuntu/kv260_lidar_cam_fusion

# Installing some useful utilities
apt-get install -y net-tools mlocate libgoogle-glog-dev libpcap0.8-dev python

# Install ROS
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
apt update
apt install -y ros-noetic-desktop-full
apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
rosdep init
rosdep update
echo "source /opt/ros/noetic/setup.bash" >> /home/ubuntu/.bashrc
source /opt/ros/noetic/setup.bash

# Download, Build, and Install the Latest Velodyne ROS Package
cd /home/ubuntu
mkdir velodyne_ws && cd velodyne_ws
git clone https://github.com/ros-drivers/velodyne.git src
catkin_make
echo "source ${PWD}/devel/setup.bash" >> /home/ubuntu/.bashrc
source ${PWD}/devel/setup.bash

cat << EOF > /etc/systemd/network/20-wired.network
[Match]
Name=eth0

[Network]
DHCP=yes

[Route]
Destination=192.168.1.201
EOF
systemctl restart systemd-networkd.service

sed -i  's/  <arg name="device_ip".*/  <arg name="device_ip" default="192.168.1.201" \/>/' src/velodyne_pointcloud/launch/VLP-32C_points.launch

# Install cv_camera ROS Package
apt-get install -y ros-noetic-cv-camera

# Install ROS package for drawing bounding boxes
apt-get install -y ros-noetic-jsk-visualization

# Install the Xilinx KV260 Apps
snap install xlnx-config --edge --classic
(echo -e "Y\n" && cat) | xlnx-config.sysinit
snap install xlnx-nlp-smartvision
xlnx-config -q
xlnx-config --install nlp-smartvision
echo "xlnx-config --xmutil loadapp nlp-smartvision" > /usr/local/bin/load_firmware.sh
cat <<EOF > /etc/systemd/system/load_firmware.service
[Unit]
After=network.service

[Service]
ExecStart=/usr/local/bin/load_firmware.sh

[Install]
WantedBy=default.target
EOF
systemctl start load_firmware
systemctl enable load_firmware

xlnx-nlp-smartvision.set-mic
snap install xlnx-vai-lib-samples

# Download Vitis AI models that will be used
mkdir -p /usr/share/vitis_ai_library/models
mkdir -p /home/ubuntu/Downloads
wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_bdd-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -O /home/ubuntu/Downloads/yolov3_bdd-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz
tar -xvf /home/ubuntu/Downloads/yolov3_bdd-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -C /usr/share/vitis_ai_library/models

wget  https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -O /home/ubuntu/Downloads/yolov3_adas_pruned_0_9-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz
tar -xvf /home/ubuntu/Downloads/yolov3_adas_pruned_0_9-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -C /usr/share/vitis_ai_library/models

wget  https://www.xilinx.com/bin/public/openDownload?filename=ssd_adas_pruned_0_95-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -O /home/ubuntu/Downloads/ssd_adas_pruned_0_95-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz
tar -xvf /home/ubuntu/Downloads/ssd_adas_pruned_0_95-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -C /usr/share/vitis_ai_library/models

wget https://www.xilinx.com/bin/public/openDownload?filename=pointpillars_kitti_12000_0_pt-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -O /home/ubuntu/Downloads/pointpillars_kitti_12000_0_pt-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz
tar -xvf /home/ubuntu/Downloads/pointpillars_kitti_12000_0_pt-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -C /usr/share/vitis_ai_library/models

wget https://www.xilinx.com/bin/public/openDownload?filename=pointpillars_kitti_12000_1_pt-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -O /home/ubuntu/Downloads/pointpillars_kitti_12000_1_pt-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz
tar -xvf /home/ubuntu/Downloads/pointpillars_kitti_12000_1_pt-DPUCZDX8G_ISA0_B3136_MAX_BG2-1.3.1-r241.tar.gz -C /usr/share/vitis_ai_library/models

cd /home/ubuntu/kv260_lidar_cam_fusion
catkin_make
echo "source ${PWD}/devel/setup.bash" >> /home/ubuntu/.bashrc
source ${PWD}/devel/setup.bash
