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

   Section 1 <./Workflows/README>
   Section 2 <./System_Setup/README>
   Section 3 <./Basic/ML_Introduction_Basic>

.. list-table:: 
   :widths: 20 80
   :header-rows: 1
   
   * - Tutorial
     - Description
	 
   * - :doc:`Section 1 <./Workflows/README>`
     - An overview of Vitis and Vitis-AI Workflows—see how Vitis unifies software, acceleration, and ML development under a single development platform.

   * - :doc:`Section 2 <./System_Setup/README>`
     - Vitis software platform and Vitis-AI setup.

   * - :doc:`Section 3 <./Basic/ML_Introduction_Basic>`
     - Deploy a DenseNet inference application on the ZCU104 board.


* [**Section 1**](./01-Workflows) :
  + An overview of Vitis and Vitis-AI Workflows
    * See how Vitis unifies software, acceleration, and ML development under a single development platform.
* [**Section 2**](./02-System_Setup) :
  + Vitis software platform setup
  + Vitis-AI setup
* [**Section 3**](./03-Basic) :
  + Deploy a DenseNet inference application on the ZCU104 board
    - Video file input
    - USB camera input
  + Increase overall system performance by using the [Vitis Vision Library](https://www.xilinx.com/products/design-tools/vitis/vitis-libraries/vitis-vision.html) to accelerate the image pre-processing

  + [**Module 1**](./03-Basic/Module_1)
    * Prepare SD card with the pre-built DPU platform
    * Boot the ZCU104 and verify basic functionality
  + [**Module 2**](./03-Basic/Module_2)
    * Setup cross-compilation environment
    * Update <code>glog</code> package
    * Cross-compile the Vitis-AI examples
  + [**Module 3**](./03-Basic/Module_3)
    * Update the board image
    * Run RefineNet demo
  + [**Module 4**](./03-Basic/Module_4)
    * Classification using Vitis-AI and Tensorflow
    * Running model through the Vitis-AI tool flow
    * Deploying the model to the ZCU104 and evaluating results
  + [**Module 5**](./03-Basic/Module_5)
    * Working with network and Vitis-AI
    * Modifying RefineDet model to work with Vitis-AI
    * Train model with modified dataset
    * Use Vitis-AI to generate deployment files
    * Running RefineDet on the ZCU104
  + [**Module 6**](./03-Basic/Module_6)
    * Review the Vitis-AI APIs for application development
    * Review the RefineDet application architecture
    * Cross-compiling RefineDet application using the cross-compilation environment
  + [**Module 7**](./03-Basic/Module_7)
    * Determining performance bottlenecks in RefineDet application
    * Accelerating the image pre-processing using the Vitis Vision libraries
    * Measuring end-to-end system performance

<p class="sphinxhide" align="center"><sup>Copyright&copy; 2020-2021 Xilinx</sup></p>
