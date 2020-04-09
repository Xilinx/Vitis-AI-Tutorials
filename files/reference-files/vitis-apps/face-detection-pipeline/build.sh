#!/bin/bash
CXX=${CXX:-g++}
$CXX -std=c++11 -I. -o test_video_facedetect test_video_facedetect.cpp mipiInput.cpp -lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ldpfacedetect  -pthread -lglog
