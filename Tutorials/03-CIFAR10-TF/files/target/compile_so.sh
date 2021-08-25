#!/bin/bash

# make shared library file (.so)
aarch64-xilinx-linux-gcc -fPIC  \
  -shared ./dpu_customcnn.elf -o ./dpuv2_rundir/libdpumodelcustomcnn.so
