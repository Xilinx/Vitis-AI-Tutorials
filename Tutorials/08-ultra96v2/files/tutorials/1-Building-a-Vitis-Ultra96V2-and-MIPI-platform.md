# Building a Vitis Ultra96V2 and MIPI Platform

Before you can deploy a high-performance Edge system, you must create a Vitis™ software platform for your custom board. To meet requirements for low-latency and a high framerate, you will work with MIPI input from the camera module on your mezzanine card. Beginning with a working MIPI MPSoC design on the Ultra96v2, this tutorial guides you through the process to enable it as a platform for dynamic design creation with the Vitis tool, and leveraging that capability to compose within Machine Learning Inference accelerators.

## Vivado Tool and the Hardware Platform

### Generate the Base MIPI Project

First, you will create the original **non-accelerated** MIPI project in the Vivado and PetaLinux tools. Next, you will have bootable hardware and software images to launch a pipeline to view the input MIPI video from the Ultra96.

1. Copy the `sources` folder from the [`reference-files/vivado/sources`](../reference-files/vivado/sources) directory to the [`build/vivado`](../build/vivado) directory.
1. Open Vivado 2019.2.
1. Create a new project in the [build/ vivado] directory.
   1. Name it `ultra96v2_mipi`.
   2. Choose to create a project subdirectory.
1. Select the **RTL project** option and **do not specify sources**.
1. Navigate to the **Boards** tab and select the Ultra96V2 board file as the part/board for the project.
1. Click **Finish**.
1. At the bottom of the screen, within the **Tcl Console** tab, change the directory to `build/vivado`.
1. Use the Tcl Console to call `source ./sources/u96v2_mipi.tcl`.
1. In the **Sources** tab, right-click `u96v2_mipi.bd`, and then click **Create HDL Wrapper** (let Vivado auto-update).
1. Click **File** > **Add Files** (add or create constraints).
1. From `build/vivado/sources`, select `cnst.xdc`, and then select to copy the constraints file into the project.

### Preparing for Dynamic Design Composition with the Vitis IDE

Now, you will make the necessary additions and modifications to the hardware design to prepare it for software-defined acceleration. To get started, open the base Vivado project.

### Configure Platform Interfaces

In order for the Vitis tool to insert your hardware acceleration blocks into the design, you need to leave open and designate the interfaces that it can use to connect the blocks. In this design, you need a few memory mapped interfaces, so that your kernels can connect into the PS DDR. You will open three HP Slave ports in this platform. This portion of the process also allows you to "name" the port, giving it a shorter nickname to designate connections later on.

1. Open the base Vivado project.
1. In the **Window** menu, click **Platform Interfaces**.
1. To add the three PS HPx slaves that are not already being used, right-click the interface and select **Enable** on the following:
   * S_AXI_HP0_FPD
   * S_AXI_HP1_FPD
   * S_AXI_HP2_FPD
1. Enable the HPM0_FPD master interface.

   >**NOTE**: Make sure this interface is disabled on the Zynq PS block (within the Processor Configuration Wizard), because this master will be used by the tools to connect to the accelerator.
1. For each slave interface enabled, add an **sptag** value in the **Options** tab that will be used to reference the port later in the flow: `HP0`, `HP1`, and `HP2`, respectively.

### Designate Platform Clocks

Similar to how you designated the interfaces for the platform, you now must indicate to the tools which clocks it should use for the accelerators that it places in the platform. In this case, your platform should use two clocks (a 1x and a 2x clock with the 2x provided for the optimized double-pumping of DSP48s), so you will indicate to the platform both a 150 and 300 MHz clock. The DPU hardware accelerator can be clocked faster or slower than this rate, and this rate was chosen to balance power and framerate performance in your application.

 1. Double-click the **clk_wiz_0 IP**, and make the following changes in the Output Clocks tab:
   `[clk_out3=150MHz], [clk_out4=300MHz], [Matched routing selected on clk_out3/4], [Reset Type = Active Low]`
 1. Right-click the block design, select **Add IP**, and add a processor system reset IP for each of the new clocks.
 1. Name the new clocks, `proc_sys_reset_dynamic_1` and `proc_sys_reset_dynamic_2`.
 1. Connect the `clk_out3` and `clk_out4` outputs of `clk_wiz_0` block to `proc_sys_reset_dynamic_1` and `proc_sys_reset_dynamic_2` `slowest_sync_clk` inputs, respectively.
 1. Connect the `ext_reset_in` to `pl_resetn0` on the MPSoC block.
 1. Connect the "locked" output of the Clock Wizard to the `dcm_locked` port of the processor reset blocks.
 1. Connect the `ext_reset_in` port of `each proc_sys_reset` block to the `pl_resetn0` port.
 1. In the Platform Interfaces tab, enable `clk_out3` and `clk_out4` of the `clk_wiz_0` instance.
 1. Set the slower clock (in this case, `clk_out3`) as the default. `clk_out3` should have its id set to 0, and `clk_out4` should have its id set to 1.
 1. Make sure the `proc_sys_reset` block listed in each window is set to the instance that is connected to that clock. Check the properties/options window when each clock is selected in platform interfaces, and verify the proc_sys_reset parameter matches.

### Enable Interrupt-Based Kernels

The default scheduling mode for the acceleration kernels is polled. To enable interrupt-based processing within your platform, you need to add an interrupt controller. Within the current design, you will connect  a constant "gnd" to the interrupt controller without connecting any valid interrupt sources at this time. Paired with the AXI-Interrupt Controller is a `dynamic_postlink` Tcl script in the Vivado tool sources, which will select the interrupt constant net, disconnect it from the concatenation block, and then automatically connect up your acceleration kernel after it is added by the Vitis tool.

1. Right-click the block design, select **Add IP**, and add an AXI Interrupt controller.
1. In the block properties for the interrupt controller, set the name to `axi_intc_0`.
1. Double-click the controller block to customize it.
1. Change the Auto/Manual switch to manual for "Edge or Level" and to "Single" for Interrupt Output Connection.
1. Select OK to exit.
1. Add a "Concat" IP to concatenate inputs into the interrupt controller.
1. In the block properties for the concat block, set the name to xlconcat_interrupt_0.
1. Double click the Concat block and modify the number of ports to 8.
1. Add a "Constant" IP to provide a constant "0" to the interrupt controller. This constant will get disconnected and replaced by a connection to acceleration interrupts by the tool at compile time.
1. Double-click `Constant IP`, and modify the constant value to **0**.
1. In the block properties, set the name of the constant to `xlconstant_gnd`.
1. Click the **Run Connection Automation** link in the Designer Assistance bar to auto-connect the AXI Interrupt controller's Slave AXI interface.
    1. Choose the **HPM0_LPD** because the HPM1_FPD is being used for the video subsystem.
    1. **NOTE:** Make sure to select a "new" interconnect
    1. Select the **clk_out1 (200 MHz)** clock port for all clock sources.
1. Connect the input of the interrupt controller to the concat block output.
1. Connect the constant output to the first input of the concat block and then each subsequent concat input to this net.
1. Connect the output of the interrupt controller to `pl_ps_irq0` on the PS block.

### Generate the Design and XSA

Now that you customized this design, it can be exported to the Vitis tool through a Xilinx Support Archive (XSA).

>**NOTE:** You are not going to build this project to a bitstream. The Vitis tool will utilize this archive to import the design, compose in your hardware accelerators, and at that point, it will build a bitstream. You will automate a portion of this process using the `xsa.tcl` script—this automates naming and platform details before exporting the XSA file to the `hw_platform` directory. This script also links your `dynamic_postlink.tcl` script, so that the script specific to this platform is included inside of the archive.

1. Generate the block design.
1. Export the hardware platform by running `source ./sources/xsa.tcl` in the Tcl Command window.

## Creating the Software Platform

The software platform requires several changes to the default Petalinux template. Begin by configuring the project to include a meta-layer, which builds in all necessary support for the MIPI mezzanine card and pipeline. Then, finish by adding the necessary Xilinx Runtime (XRT) components into the design.

### Customize the Template PetaLinux Project

The first step in creating your acceleration platform is to open a stock Petalinux project in the build directory. You will then add a meta-layer to enable the necessary support for the MIPI mezzanine and other elements on the Ultra96V2. Finally, you will add the acceleration library components previously mentioned XRT. These components come in the form of recipes, which you will add to the user layer within the Petalinux build. First, you will copy over the files and build recipes, and then you will enable them through the Petalinux configuration menus.

1. Change directory to the `build/` folder.
1. Create a new PetaLinux project with the ZynqMP template. \
  `petalinux-create -n petalinux --template zynqMP -t project`
1. Copy the `meta-ultra96v2mipi` folder from `reference-files/ petalinux] into [build/ petalinux/ components`.
1. Update the Petalinux project with the new exported XSA from the Vivado tool, and open the initial configuration menu.\
  `petalinux-config --get-hw-description=../hw_platform`
1. From the main menu, select **Subsystem AUTO settings** and under the Serial settings, change to psu_uart_1 as the primary.
1. From the main menu, select **DTG Settings**, and set the machine name to `avnet-ultra96-rev1`.
1. From the main menu, navigate to **Yocto Settings** > **User Layers** and add ${PROOT}/components/meta-ultra96v2mipi as user layer 0, and then exit the initial configuration menu.
1. To add the XRT drviers to the platform, add the recipes by copying the `recipes-xrt` directory from `reference-files/ petalinux` to `build/ petalinux/ project-spec/ meta-user`.
1. Add a recipe for an `autostart` script to run automatically after boot by copying the autostart directory from [reference-files/ petalinux/ autostart] to [build/ petalinux/ project-spec/ meta-user/ recipes-apps].
1. Add the recipes above to the Petalinux image configuration by editing [build/ petalinux/ project-spec/ meta-user/ conf/ user-rootfsconfig] to add all lines from [reference-files/ petalinux/ plnxrfscfg.txt]
1. Open the Petalinux root filesystem configuration GUI to enable the recipes above.\
  `petalinux-config -c rootfs`\
  and then enable all the recipes added to the `user-rootfsconfig` file within the "User Packages" and "Apps" sub menus
1. Within the rootfs configuration, under the PetaLinux Package Groups, enable the following options:
    * gstreamer
    * matchbox
    * opencv
    * v4lutils
    * x11
1. Exit the rootfs config menu.

### Modify the Linux Device Tree

The Linux Device Tree needs to be modified so that the several devices not automatically generated by the Device Tree Generator are correctly recognized by Linux—for instance, the OmniVision camera sensor. Modify the `system-user.dtsi` file to add the all these nodes to the tree.

1. Open `build/ petalinux/ project-spec/ meta-user/ recipes-bsp/ device-tree/ files/ system-user.dtsi`.
1. Replace the contents of the file with all the text provided in `reference-files/ petalinux/ dtfrag.txt`.

### Build PetaLinux and Package Software Components

Now that you have made all the necessary configuration changes for the Petalinux build, start the build. Depending on the processing power on your machine, build process times can vary. After the Linux build is complete, you need to move all the built software components into a common directory. By placing all your boot components in one directory, it makes it easier to package up the Hardware and Software sides into the resulting platform. You will also use Petalinux to build the sysroot to provide the complete cross-compilation environment for this software platform. This sysroot will also be included in the software portion of the platform because it provides the correct version of headers/includes when compiling the platform.

1. Build PetaLinux.\
  `petalinux-build`\
1. Copy all .elf files from the `build/ petalinux/ images/ linux` directory to `build/ sw_platform/ boot`.  
   This should copy over the following files:  
   * **Arm Trusted Firmware**: `b131.elf`  
   * **PMU Firmware**: `pmufw.elf`
   * **U-Boot**: `u-boot.elf`
   * **Zynq FSBL**: `zynqmp_fsbl.elf`
1. Copy the `linux.bif` file from the `reference_files/ vitis-platform` directory to `build/ sw_platform/ boot`.
1. Copy the `image.ub` file from the `build/ petalinux/ images/ linux` directory to `build/ sw_platform/ image`.
1. Copy the `autostart.sh` file from the `reference_files/ vitis-platform` directory to `build/ sw_platform/ image`.
1. Build the Yocto SDK (this provides your sysroot) from the project.
   `petalinux-build --sdk`
1. Move the `build/ petalinux/ images/ linux/ sdk.sh` file to `build/ sw_platform` and then extract SDK.
   `cd build/sw_platform`
   `./sdk.sh -d ./ -y`

## Generate the Vitis Platform

The Vitis Platform is a set of components that comprise everything needed to boot and develop for a particular board/design configuration and contains both a hardware and software component. Now that you have built the hardware (XSA) and software (Linux image and boot elf files) components for the platform, you can use these components to generate and export your custom user-defined platform. You will complete these steps in the Xilinx Vitis IDE.

1. Open the Vitis IDE.
1. For the workspace, select **build/ workspace**.
1. Select **File** > **New** > **Platform Project**.
1. Name the platform `ultra96v2_mipi`.
1. Select to create from hardware specification and select the XSA in [build/ hw_platform].
1. Select the Linux operating system and the psu_cortexa53 subsystem.
1. Deselect **Generate Boot Components** and click **Finish**.
1. Exit the Welcome tab.
1. In the File Navigator, double-click the `platform.spr` file to open the Platform Configuration page.
1. On the ultra96v2_mipi Platform page, select **Browse** for both FSBL and PMU Firmware, and navigate to the zynqmp_fsbl and pmufw elf files that you previously copied into the `sw_platform/boot` folder.
1. Customize the linux on psu_cortexa53 domain to point to the boot components and bif that were copied earlier to `sw_platform/boot`.
1. Customize the "linux on psu_cortexa53" domain to point to the image directory in `sw_platform/ image`.
1. Right click on the top-level project in the explorer and select **Build Project** to package up the custom Vitis Platform.

Now that the platform is generated, note that there is an `export` directory. This `export` directory is the complete, generated platform and can be zipped up and shared—providing the components to enable new developers on the custom platform.

## Creating a Vitis Application with DPUv2 and the MIPI Pipeline

For the final application, target your custom platform to bring up a simple demo showing the input from your MIPI input pipeline. You will also add in the DPUv2 IP from Vitis AI as a hardware accelerator and configure it to enable deployment of Vitis AI applications and demos in the next steps of this tutorial.

### Create a New Application Project

Start by creating the new application project. In the Vitis IDE, the Application Project exists inside of a System Project container in order to provide a method for cohesive system development across the enabled domains in the platform (for instance, Cortex-A53 and Cortex-R5). Because you are working in the same workspace, you can simply target the platform that you generated earlier, but you can also add additional platform repositories by clicking the **+** button and pointing to the directory containing your xpfm while inside the Platform Selection portion of the new app dialog.

1. Open the Vitis IDE.
1. For your workspace, select **build/ workspace**.
1. Select **File**> **New**> **Application Project**
1. Name the project, `hello_world`, and use the auto-generated system project name.
1. Select the **ultra96v2_mipi** platform that you just created.
1. Verify that the Linux domain is selected.
1. Select **Empty Application** as the template and **Finish**.

### Prepare the DPUv2 Kernel

The DPUv2 IP is provided in the sources in the Vitis AI 1.0 Github. You need to clone this repository and reconfigure the IP to meet you needs before packaging it in a .xo file to compose into your custom platform. You will modify a single header file to designate that you want the B2304 configuration of the DPUv2 IP, with the minimal DSP48 configuration.

1. Change directory to the `build` folder.
1. Clone the Vitis AI repository.\
  `git clone https://github.com/Xilinx/Vitis-AI.git`
1. Change directory to `build/ Vitis-AI/ DPU-TRD/ prj/ Vitis`.
1. Open the `dpu_conf.vh` file, and replace the contents with those in `reference-files/ vitis-apps/ hello-world/ dpu_conf.vh`.
1. Run "`make`" on the `dpu.xo` object to create the package `.xo file`.\
  `make binary_container_1/dpu.xo DEVICE=ultra96v2`

### Edit the Build Settings

1. Under your project in the file navigator, right-click the `src` folder, and select **Import Sources**.
1. Choose from directory `build/ Vitis-AI/ DPU-TRD/ prj/ Vitis/ binary_container_1` as the target location and import the `dpu.xo` file that you just created.
1. Import sources again, and add the cpp, header, and prj_config files from `reference-files/ vitis-apps/ hello_world`.
1. Open the `hello_world.prj` file and in the upper right-hand corner, change Emulation-SW to Hardware
1. Under Hardware Functions, click the lightning bolt icon to add a new accelerator.
1. Select the "dpu_xrt_top" included as part of the `dpu.xo` file that you included earlier.
1. Click **binary_container_1** to change the name to dpu.
1. Right-click **dpu** and select **Edit V++ options**.
1. Add `--config ../src/prj_config` to designate which port of the DPU will connect to your Platform Interfaces you created earlier.
1. In the Explorer tab, right-click the project folder, and in **GCC Host Linker** > **Libraries**, click the green **+** to add the following libraries:\
        * opencv_core\
        * opencv_imgcodecs\
        * opencv_highgui\
        * opencv_imgproc\
        * opencv_videoio\
1. In the GCC Host Compiler sub-menu, select **Includes**, and click the red **X** icon to remove the XILINX_VIVADO_HLS entry.
1. Click **Apply and Close**.
1. In the Assistant tab, right-click **Hardware**, and click **Build to kick off a build of the Hardware and Software Components**.

### Run the Demo

After the hardware build is complete, you have an `sd_card` folder in your `hello_world` project. You can copy everything in this folder over to an SD card, and load it on the Ultra96 to boot the platform and application. By default, the USB ethernet gadget will display with the static IP address, which you can use to SSH into the board from your host PC and forward the X11 graphics (to preview the images from this initial application). However, you are also able to bring up the board with the UART and configure USB to ethernet adapters or display your demo frames to a display port monitor.

In the next part of this tutorial, you will use this platform and the Vitis AI Runtime Docker Container to build custom applications to target the DPU in the platform.
