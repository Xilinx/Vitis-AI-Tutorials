# Designing with Vitis AI for a Custom Edge Platform

At this point, you have now brought up a custom platform with the Vitis tool. You have tested the MIPI input source, used the Vitis tool to compose the DPUv2 Machine Learning inference accelerator into your design, and recompiled optimized models from the Model Zoo to target that configuration of the accelerator. Now, deploy a real inferencing application to your platform using the Vitis AI Library. The Vitis AI library is a high-level abstraction allowing fast prototyping and development with high-performance inferencing capability.

## Prepare the Cross Compilation Environment

To simplify the setup process and ease of deploying to a platform, the Vitis AI cross-compilation environment is set up in a Docker container publicly available on the Xilinx Docker Hub repository. Load this image to your machine, and then launch it to begin working on custom applications for your custom platform. You will also copy over the test application sources to the directory shared to the Docker container to easily access and compile the source.

1. Create a new directory for building the app under `build/ Vitis-AI` and call it face-detection-pipeline
1. Copy the `build.sh` and `test_video_facedetect.cpp` files from `reference-files/ vitis-apps/ face-detection-pipeline` into the new subdirectory.
1. Copy the `mipiInput.cpp` and `mipiInput.h` files from `reference-files/ vitis-apps/ hello_world` into the new subdirectory.
1. Open the source files from the `face-detection-pipeline` folder in your text editor for review and editing on your host system.
1. Navigate to `build/ Vitis-AI` and use the `docker_run.sh` script to launch the container. If you don't have the image installed, it will be downloaded at this time.\
  `./docker_run.sh xilinx/vitis-ai:runtime-1.0.0-cpu`
1. Inside the Docker container, navigate to the `/workspace/ face-detection-pipeline` This is the shared version of the folder on your host system.
1. Run the `build.sh` shell script for the Face Detection pipeline. This is a simple example using the `libdpfacedetect` shared library, calling the Create and Run functions to initialize the model in the DPU and then query the result for the given Mat.

## Deploy the Application to the Target

Now that you cross-compiled in the previous application for the DPU, and you will now transfer the application, model files, and necessary shared libraries onto the SD card. The DPU requires a number of shared libraries that are inside of the runtime Docker container, as well as a metadata configuration file for the runtime.

1. Inside the Docker container, create a "libs" directory in the [/workspace/] subdirectory.
1. Copy the following files from `docker, grab file location` to the `/workspace/ libs` subdirectory.\
`libUniLog.so`\
`libdpbase.so.1`\
`libdpcommon.so.1`\
`libdpfacedetect.so`\
`libdpfacedetect.so.1`\
`libdpmath.so.1`\
`libdpproto.so.1`\
`libdpuaol.so`\
`libhineon.so`\
`libn2cube.so`\
`libvart_dpu.so`\
`libvitis_dpu.so.1`\
`libxir.so`\
`libxlnx_dpu_controller.so`\
`libxlnx_dpu_controller.so.1`\
`libxlnx_elf_util.so`\
`libxlnx_elf_util.so.1`\
`libxlnx_xir_buffer_object.so`\
`libxlnx_xir_buffer_object.so.1`\
`libxlnx_xrt_device_handle.so`\
`libxlnx_xrt_device_handle.so.1`\
`libxnnpp.so`\
`libxnnpp.so.1`
1. Inside of the `/workspace/ face-detection-pipeline` directory, create a new directory called `densebox`.
1. Copy the `meta.json` and `densebox.prototxt` files from `reference-files/ vitis-apps/ face-detection-pipeline` into the new subdirectory.
1. Outside of the Docker container, copy the `dpu_densebox.elf` file stored in `build/` in the last step into the `densebox` folder.
1. Copy the `libs` and `densebox` directories onto your previous SD card image.
   >**IMPORTANT:** Do not delete any of the previous files.

## Run the Application and Design Compound Pipelines

Now that you have an SD card image, you can boot the Ultra96 and run the application to observe the Face Detection results with the  `./test_video_facedetect` command. There are a number of simple modifications that you can make to build different video pipelines.

Here are a few examples:

* Swap between cameras for a dynamic inputs
* Process frames from multiple cameras and stitch the resulting images together (or stitch in hardware with Vitis Vision Library)
* Create a compound pipeline, passing detected FaceDetection results to FaceLandmark
