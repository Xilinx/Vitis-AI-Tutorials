##############################################################################
Vitis-AI™ 1.4 — Machine Learning Tutorials
##############################################################################


===========================
Introduction
===========================


.. toctree::
   :maxdepth: 3
   :caption: Introduction to Vitis AI
   :hidden:

   Introduction to Vitis AI <./Introduction/Machine_Learning_Introduction>



.. list-table:: 
   :widths: 20 80
   :header-rows: 1
   
   * - Tutorial
     - Description
	 
   * - :doc:`Introduction to Vitis AI <./Introduction/Machine_Learning_Introduction>`
     - This tutorial puts in practice the concepts of FPGA acceleration of Machine Learning and illustrates how to quickly get started deploying both pre-optimized and customized ML models on Xilinx devices.




===========================
Design Tutorials
===========================

.. toctree::
   :maxdepth: 3
   :caption: Introduction to Vitis AI
   :hidden:

   Quantization and Pruning of AlexNet CNN trained in Caffe with Cats-vs-Dogs dataset <./Design_Tutorials/01-caffe_cats_vs_dogs/README>
   MNIST Classification using Vitis-AI and TensorFlow <./Design_Tutorials/02-MNIST_classification_tf/README>
   Using DenseNetX on the Xilinx DPU Accelerator <./Design_Tutorials/03-using_densenetx/README>
   Deep Learning with Custom GoogleNet and ResNet in Keras and Xilinx Vitis AI <./Design_Tutorials/04-Keras_GoogleNet_ResNet/README>
   FCN8 and UNET Semantic Segmentation with Keras and Xilinx Vitis AI <./Design_Tutorials/05-Keras_FCN8_UNET_segmentation/README>
   Vitis AI YOLOv4 <./Design_Tutorials/07-yolov4-tutorial/README>
   TensorFlow2 and Vitis AI design flow <./Design_Tutorials/08-tf2_flow/README>
   VPyTorch flow for Vitis AI <./Design_Tutorials/09-mnist_pyt-master/README>
   RF Modulation Recognition with TensorFlow 2 <./Design_Tutorials/tf2_var_autoenc/README>
   Denoising Variational Autoencoder with TensorFlow2 and Vitis-AI <./Design_Tutorials/tf2_var_autoenc/README>


The methodology for developing optimized accelerated applications is comprised of two major phases: architecting the application, and developing the kernels. In the first phase, you make key decisions about the application architecture by determining which software functions should be accelerated onto ACAP kernels, how much parallelism can be achieved, and how to deliver it in code. In the second phase, you implement the kernels by structuring the source code, and applying the necessary build options s to create the kernel architecture needed to achieve the optimized performance target. The following examples illustrate the use of this methodology in real-world applications.


.. list-table:: 
   :widths: 20 80
   :header-rows: 1
   
   * - Tutorial
     - Description
	 
   * - :doc:`Quantization and Pruning of AlexNet CNN trained in Caffe with Cats-vs-Dogs dataset <./Design_Tutorials/01-caffe_cats_vs_dogs/README>`
     - Train, prune, and quantize a modified version of the AlexNet convolutional neural network (CNN) with the Kaggle Dogs vs. Cats dataset in order to deploy it on the Xilinx ZCU102 board.

   * - :doc:`MNIST Classification using Vitis-AI and TensorFlow <./Design_Tutorials/02-MNIST_classification_tf/README>`
     - Learn the Vitis AI TensorFlow design process for creating a compiled ELF file that is ready for deployment on the Xilinx DPU accelerator from a simple network model built using Python. This tutorial uses the MNIST test dataset.

   * - :doc:`Using DenseNetX on the Xilinx DPU Accelerator <./Design_Tutorials/03-using_densenetx/README>`
     - Learn about the Vitis AI TensorFlow design process and how to go from a Python description of the network model to running a compiled model on the Xilinx DPU accelerator.

   * - :doc:`Deep Learning with Custom GoogleNet and ResNet in Keras and Xilinx Vitis AI <./Design_Tutorials/04-Keras_GoogleNet_ResNet/README>`
     - Quantize in fixed point some custom CNNs and deploy them on the Xilinx ZCU102 board, using Keras and the Xilinx7Vitis AI tool chain based on TensorFlow (TF).

   * - :doc:`FCN8 and UNET Semantic Segmentation with Keras and Xilinx Vitis AI <./Design_Tutorials/05-Keras_FCN8_UNET_segmentation/README>`
     - Train the FCN8 and UNET Convolutional Neural Networks (CNNs) for Semantic Segmentation in Keras adopting a small custom dataset, quantize the floating point weights files to an 8-bit fixed point representation, and then deploy them on the Xilinx ZCU102 board using Vitis AI.

   * - :doc:`Vitis AI YOLOv4 <./Design_Tutorials/07-yolov4-tutorial/README>`
     - Learn how to train, evaluate, convert, quantize, compile, and deploy YOLOv4 on Xilinx devices using Vitis AI.

   * - :doc:`TensorFlow2 and Vitis AI design flow <./Design_Tutorials/08-tf2_flow/README>`
     - Learn about the TF2 flow for Vitis AI. In this tutorial, you'll be trained on TF2, including conversion of a dataset into TFRecords, optimization with a plug-in, and compiling and execution on a Xilinx ZCU102 board or Xilinx Alveo U50 Data Center Accelerator card.

   * - :doc:`VPyTorch flow for Vitis AI <./Design_Tutorials/09-mnist_pyt-master/README>`
     - Learn how to use Vitis AI by using PyTorch. You'll use a simple ``get-you-started`` example to get started, and then be trained on quantization with a plug-in, and then compiling and execution on a Xilinx ZCU102 board or Xilinx Alveo U50 Data Center Accelerator card.

   * - :doc:`RF Modulation Recognition with TensorFlow 2 <./Design_Tutorials/10-RF_modulation_recognition/README>`
     - Machine learning applications are certainly not limited to image processing! Learn how to apply machine learning with Vitis AI to the recognition of RF modulation from signal data.

   * - :doc:`Denoising Variational Autoencoder with TensorFlow2 and Vitis-AI <./Design_Tutorials/tf2_var_autoenc/README>`
     - The Xilinx DPU can accelerate the execution of many different types of operations and layers that are commonly found in convolutional neural networks but occasionally we need to execute models that have fully custom layers. One such layer is the sampling function of a convolutional variational autoencoder. The DPU can accelerate the convolutional encoder and decoder but not the statistical sampling layer - this must be executed in software on a CPU. This tutorial will use the variational autoencoder as an example of how to approach this situation.



==================================
Feature Tutorials
==================================

.. toctree::
   :maxdepth: 3
   :caption: Introduction to Vitis AI
   :hidden:

   Freezing a Keras Model for use with Vitis AI <./Feature_Tutorials/01-freezing_a_keras_model/README>
   Profiling a CNN Using DNNDK or VART with Vitis AI <./Feature_Tutorials/02-profiling-example/README>
   TensorFlow AI Optimizer Example Using Low-level Coding Style <./Feature_Tutorials/04-tensorflow-ai-optimizer/README>
   Partitioning Vitis AI SubGraphs on CPU/DPU <./Feature_Tutorials/pytorch-subgraphs/README>
   Fine-Tuning TensorFlow2 quantized models <./Feature_Tutorials/tf2_quant_fine_tune/README>
   Vitis AI based Deployment Flow on VCK190 <./Feature_Tutorials/Vitis-AI-based-Deployment-Flow-on-VCK190/README>



Feature tutorials illustrate specific workflows or stages within Vitis AI.

.. list-table:: 
   :widths: 20 80
   :header-rows: 1
   
   * - Tutorial
     - Description
	 
   * - :doc:`Freezing a Keras Model for use with Vitis AI <./Feature_Tutorials/01-freezing_a_keras_model/README>`
     - Freeze a Keras model by generating a binary protobuf (``.pb``) file.

   * - :doc:`Profiling a CNN Using DNNDK or VART with Vitis AI <./Feature_Tutorials/02-profiling-example/README>`
     - Profile a CNN application running on the ZCU102 target board with Vitis AI.

   * - :doc:`TensorFlow AI Optimizer Example Using Low-level Coding Style <./Feature_Tutorials/04-tensorflow-ai-optimizer/README>`
     - Use AI Optimizer for TensorFlow to prune an AlexNet CNN by 80% while maintaining the original accuracy.

   * - :doc:`Partitioning Vitis AI SubGraphs on CPU/DPU <./Feature_Tutorials/pytorch-subgraphs/README>`
     - Learn how to deploy a CNN on the Xilinx `VCK190 <https://www.xilinx.com/products/boards-and-kits/vck190.html>`_ board using Vitis AI.

   * - :doc:`Fine-Tuning TensorFlow2 quantized models <./Feature_Tutorials/tf2_quant_fine_tune/README>`
     - Learn how to implement the Vitis-AI quantization fine-tuning for TensorFlow2.3.

   * - :doc:`Vitis AI based Deployment Flow on VCK190 <./Feature_Tutorials/Vitis-AI-based-Deployment-Flow-on-VCK190/README>`
     - DPU integration with VCK190 production platform.


.. sidebar:: More Information

    See Vitis Development Environment on `xilinx.com <https://www.xilinx.com/products/design-tools/vitis.html>`_.

    See Vitis-AI Development Environment on `xilinx.com <https://www.xilinx.com/products/design-tools/vitis/vitis-ai.htm>`_.




.. meta::
   :keywords: Vitis AI, Xilinx
   :description: Vitis AI Tutorials
   :xlnxdocumentclass: Document
   :xlnxdocumenttypes: Tutorials