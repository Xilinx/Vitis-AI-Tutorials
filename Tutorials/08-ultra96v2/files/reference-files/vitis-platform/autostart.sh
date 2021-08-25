#!/bin/bash
echo setting up media devices...
media-ctl -V '"ov5640 4-003c":0 [fmt:UYVY8_1X16/640x480@1/30 field:none colorspace:srgb]'
media-ctl -V '"b0000000.mipi_csi2_rx_subsystem":0 [fmt:UYVY8_1X16/640x480 field:none colorspace:srgb]'
media-ctl -V '"b0020000.v_proc_ss":1 [fmt:RBG888_1X24/640x480 field:none colorspace:srgb]'
media-ctl -V '"amba:axis_switch@0":1 [fmt:RBG888_1X24/640x480 field:none colorspace:srgb]'

cp /run/media/mmcblk0p1/dpu.xclbin /usr/lib
cp /run/media/mmcblk0p1/libs/*.so /usr/lib
cp /run/media/mmcblk0p1/libs/*.so.1 /usr/lib

echo setting up ethernet gadget...
modprobe g_ether
modprobe -r g_ether
sleep 3
modprobe g_ether
ifconfig usb0 149.199.50.187 up

modprobe wilc-sdio

export DISPLAY=:0.0
xrandr --output DP-1 --mode 800x600
xset s off -dpms
