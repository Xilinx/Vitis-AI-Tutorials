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

DEMO=$1
OMIT_SENSOR=$2

# If roscore is not yet running, start it
if ! rostopic list 2> /dev/null | grep -q "rosout"; then
	roscore &
fi

# Launch RViz for display
if [ -z $ROS_IP ]; then
	rviz -d src/$DEMO/rviz/$DEMO.rviz &
fi

# Launch the node
if [ -z $OMIT_SENSOR ] || [ $OMIT_SENSOR -eq 0 ]; then
	roslaunch --wait $DEMO $DEMO.launch omit_sensor:=0
else
	roslaunch --wait $DEMO $DEMO.launch omit_sensor:=1
fi

