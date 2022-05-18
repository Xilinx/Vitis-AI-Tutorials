3.3 Setting cross-compilation environment
-----------------------
You can use Windows or Linux OS to download the board image and burn it to an SD card. This module could only be done on a Linux X86 server.
1. Download the [sdk-2021.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2021.1.0.0.sh) script on the host. The SDK file is generated through the ```petalinux-build --sdk``` command and applied to set up the sysroot headers, libs, and include files for cross compilation of applications running on the embedded platforms.
2. Run the script to install the cross-compilation system dependency and sysroot.
   ```
   $./sdk-2021.1.0.0.sh
   ```
  Follow the prompts to install.

    **Note:** You need the read and write permissions for the installation path to install. The installation takes around 10 minutes. By default, it is installed in the ``~/petalinux_sdk`` directory. You can change it to any other preferred location.
3. When the installation completes, follow the prompts and execute the following command to install the embedded ARM cross-compilation environment on X86 server.
   ```
   $source [SDK_INSTALLATION_PATH]/environment-setup-aarch64-xilinx-linux
   ```
  **Note:** The command needs to be re-executed each time when a new terminal interface is opened.
4. Add some AI-related libs and includes to the existing sysroot by downloading the Vitis AI denpendencies [vitis_ai_2021.1-r1.4.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2021.1-r1.4.0.tar.gz) and extract it to the sysroot.
   ```
   $tar -xzvf vitis_ai_2021.1-r1.4.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
   ```

5. The host setup is completed. The tutorial uses  refinedet as an example to do the cross-compile in the AI Library.
    ```
    $cd ~/Vitis-AI/demo/Vitis-AI-Library/samples/refinedet
    $bash -x build.sh
    ```
  You will find four executable files generated after compilation. Follow the instruction in [Module 4](../Module_4) for the next steps.

  <img src="images/cross-compile.png">


Copyright&copy; 2020-2002 Xilinx
