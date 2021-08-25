# Working with the Model Zoo

Xilinx provides a number of pre-trained, quantized, and compiled (and in some cases, pruned) ML models in both TensorFlow and Caffe on Github: https://github.com/Xilinx/AI-Model-Zoo. As part of the Vitis AI release, a number of high-level libraries, demos, and samples are provided, that wrap around these models to abstract the usage of the high-performance ML acceleration processor. When porting to a custom platform, these models will need to be recompiled for the architecture of the processor used in the design. In this case, you will recompile the models to support the B2304.

## Prepare the Environment and Tools Docker

The Vitis AI compilation tools are provided through a dockerfile at https://github.com/Xilinx/Vitis-AI under the `docker` directory. To build the GPU-enabled version of the tools, navigate back to your clone of the Vitis AI repository from Part 1 of this tutorial. You can also follow this flow with the CPU-only docker container; if you do not have access to a GPU workstation or instance as the compilation-only flow is not compute intensive, use `-cpu` instead of `-gpu` in the following referenced Docker titles.

1. Navigate to `build/ workspace/ Hardware/ dpu.build/ link/ vivado/ vpl/ output` and copy `system.hdf` into the `build/ Vitis-AI` subdirectory.
2. Unzip `system.hdf`, and move the `ultra96v2_mipi.hwh` file into the `build/ Vitis-AI` subdirectory.
3. Navigate to `build/ Vitis-AI/ docker`, and use the `docker_build.sh` script to compile the GPU-enabled docker container.
4. Clone the Model Zoo repository from https://github.com/Xilinx/AI-Model-Zoo into the `build/ Vitis-AI` subdirectory.
5. Copy the Rebuild-Model-Zoo python tool (copy the whole folder) from `reference-files/ vitis-ai-utils/` into the `build/ Vitis-AI` subdirectory.

## Launch the Tools Docker and Compile the Model

Now that the Docker is built, you can launch the docker, and use the provided Conda environment  to generate a DPU Configuration File (dcf) with your hardware description file from the previous step. You will then use a generic Model Zoo python utility to generate our compiled model .elf files that will be used by the DPUv2 at runtime.

1. Launch the tools docker from the `build/ Vitis-AI` directory.\
  `./docker_run.sh xilinx/vitis-ai:tools-1.0.0-gpu`\
  This will share the current directory to /workspace inside of the Docker container.
1. Launch the Vitis AI Caffe Conda environment.\
  `conda activate vitis-ai-caffe`
1. Use the dlet tool to generate your .dcf file.\
  `dlet -f /workspace/system.hwh`
1. Copy the resulting .dcf file into `workspace/ Rebuild-Model-Zoo/ custom_arch_files`, and rename it `ultra96v2.dcf`.
1. Inside of the `workspace/ Rebuild-Model-Zoo` directory run the python util.\
  `python3 rebuild-model-zoo.py -z ../AI-Model-Zoo/ -n ultra96v2 -m 	cf_densebox_wider_360_640_1.11G`\
  This command will use the Model Zoo at the `-z` flag, target a dcf file named `ultra96v2.dcf`, and because a `-m` flag is provided, the command will only compile the densebox 640x360 model, instead of the whole Model Zoo.
1. Copy the resulting .elf file from the `build/ Vitis-AI/ Rebuild-Model-Zoo/ compiled_output/ cf_densebox_wider_360_640_1.11G` directory into your `build/`  directory to reserve the compiled model for deployment in the next part of this tutorial.
