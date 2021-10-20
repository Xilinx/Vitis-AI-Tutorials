############################################################################
Machine Learning Tutorial for the ZCU104
############################################################################


===========================
Design Tutorials
===========================

.. toctree::
   :maxdepth: 3
   :caption: Introduction to Vitis AI
   :hidden:

   Module 1 <./Basic/Module_1/README>
   Module 2 <./Basic/Module_2/README>
   Module 3 <./Basic/Module_3/README>
   Module 4 <./Basic/Module_4/README>
   Module 5 <./Basic/Module_5/README>
   Module 6 <./Basic/Module_6/README>
   Module 7 <./Basic/Module_7/README>


The methodology for developing optimized accelerated applications is comprised of two major phases: architecting the application, and developing the kernels. In the first phase, you make key decisions about the application architecture by determining which software functions should be accelerated onto ACAP kernels, how much parallelism can be achieved, and how to deliver it in code. In the second phase, you implement the kernels by structuring the source code, and applying the necessary build options s to create the kernel architecture needed to achieve the optimized performance target. The following examples illustrate the use of this methodology in real-world applications.


.. list-table:: 
   :widths: 20 80
   :header-rows: 1
   
   * - Tutorial
     - Description
	 
   * - :doc:`Module 1 <./Basic/Module_1/README>`
     - Train, prune, and quantize a modified version of the AlexNet convolutional neural network (CNN) with the Kaggle Dogs vs. Cats dataset in order to deploy it on the Xilinx ZCU102 board.

   * - :doc:`Module 2 <./Basic/Module_2/README>`
     - Learn the Vitis AI TensorFlow design process for creating a compiled ELF file that is ready for deployment on the Xilinx DPU accelerator from a simple network model built using Python. This tutorial uses the MNIST test dataset.

   * - :doc:`Module 3 <./Basic/Module_3/README>`
     - Learn about the Vitis AI TensorFlow design process and how to go from a Python description of the network model to running a compiled model on the Xilinx DPU accelerator.

   * - :doc:`Module 4 <./Basic/Module_4/README>`
     - Quantize in fixed point some custom CNNs and deploy them on the Xilinx ZCU102 board, using Keras and the Xilinx7Vitis AI tool chain based on TensorFlow (TF).

   * - :doc:`Module 5 <./Basic/Module_5/README>`
     - Train the FCN8 and UNET Convolutional Neural Networks (CNNs) for Semantic Segmentation in Keras adopting a small custom dataset, quantize the floating point weights files to an 8-bit fixed point representation, and then deploy them on the Xilinx ZCU102 board using Vitis AI.

   * - :doc:`Module 6 <./Basic/Module_6/README>`
     - Implement a convolutional neural network (CNN) and run it on the DPUv3E accelerator IP.

   * - :doc:`Module 7 <./Basic/Module_7/README>`
     - Learn how to train, evaluate, convert, quantize, compile, and deploy YOLOv4 on Xilinx devices using Vitis AI.

   * - :doc:`Module 7 <./Basic/Module_7/README>`
     - 
       - Prepare SD card with the pre-built DPU platform
       - Boot the ZCU104 and verify basic functionality
