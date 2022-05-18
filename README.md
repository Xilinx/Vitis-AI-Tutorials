<table class="sphinxhide">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis-AI™ 1.4 - Machine Learning Tutorials</h1>
    <a href="https://www.xilinx.com/products/design-tools/vitis.html">See Vitis™ Development Environment on xilinx.com</br></a>
    <a href="https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html">See Vitis-AI™ Development Environment on xilinx.com</a>
    </td>
 </tr>
</table>

### Introduction to Machine Learning with Vitis AI

Start here! This tutorial series will help to get you the lay of the land working with the
Vitis AI toolchain and machine learning on Xilinx devices.

 <table style="width:100%">
 <tr>
 <td width="35%" align="center"><b>Tutorial</b>
 <td width="65%" align="center"><b>Description</b>
 </tr>
 <tr>
 <td align="center"><a href="./Introduction/README.md">Introduction to Vitis AI</a></td>
 <td>This tutorial puts in practice the concepts of FPGA acceleration of Machine Learning and illustrates how to
 quickly get started deploying both pre-optimized and customized ML models on Xilinx devices.</td>
 </tr>
 </table>

### Design Tutorials

These tutorials illustrate end-to-end design concepts or workflows using Vitis AI.

 <table style="width:100%">
 <tr>
 <td width="35%" align="center"><b>Tutorial</b>
 <td width="65%" align="center"><b>Description</b>
 </tr>
 <tr>
 <td align="center">
   <a href="./Design_Tutorials/01-caffe_cats_vs_dogs/README.md">Quantization and Pruning of AlexNet CNN trained in Caffe with Cats-vs-Dogs dataset</a>
 </td>
 <td>Train, prune, and quantize a modified version of the AlexNet convolutional neural network (CNN) with the Kaggle Dogs vs. Cats dataset in order to deploy it on the Xilinx® ZCU102 board.</td>
 </tr>
 <tr>
 <td align="center">
   <a href="./Design_Tutorials/02-MNIST_classification_tf/README.md">MNIST Classification using Vitis AI and TensorFlow</a>
 </td>
 <td>Learn the Vitis AI TensorFlow design process for creating a compiled ELF file that is ready for deployment on the Xilinx DPU accelerator from a simple network model built using Python. This tutorial uses the MNIST test dataset.</td>
 </tr>
 <tr>
 <td align="center">
   <a href="./Design_Tutorials/03-using_densenetx/README.md">Using DenseNetX on the Xilinx DPU Accelerator</a>
 </td>
 <td>Learn about the Vitis AI TensorFlow design process and how to go from a Python description of the network model to running a compiled model on the Xilinx DPU accelerator.</td>
 </tr>
 <tr>
 <td align="center">
   <a href="./Design_Tutorials/04-Keras_GoogleNet_ResNet/README.md">Deep Learning with Custom GoogleNet and ResNet in Keras and Xilinx Vitis AI</a>
 </td>
 <td>Quantize in fixed point some custom CNNs and deploy them on the Xilinx ZCU102 board, using Keras and the Xilinx7Vitis AI tool chain based on TensorFlow (TF).</td>
 </tr>
 <tr>
 <td align="center">
   <a href="./Design_Tutorials/05-Keras_FCN8_UNET_segmentation/README.md">FCN8 and UNET Semantic Segmentation with Keras and Xilinx Vitis AI</a>
 </td>
 <td>Train the FCN8 and UNET Convolutional Neural Networks (CNNs) for Semantic Segmentation in Keras adopting a small custom dataset, quantize the floating point weights files to an 8-bit fixed point representation, and then deploy them on the Xilinx ZCU102 board using Vitis AI.</td>
 </tr>
  <tr>
 <td align="center">
   <a href="./Design_Tutorials/07-yolov4-tutorial/readme.md">Vitis AI YOLOv4</a>
 </td>
 <td>Learn how to train, evaluate, convert, quantize, compile, and deploy YOLOv4 on Xilinx devices using Vitis AI.</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/08-tf2_flow/README.md">TensorFlow2 and Vitis AI design flow</a>
 </td>
 <td>Learn about the TF2 flow for Vitis AI. In this tutorial, you'll be trained on TF2, including conversion of a dataset into TFRecords, optimization with a plug-in, and compiling and execution on a Xilinx ZCU102 board or Xilinx Alveo U50 Data Center Accelerator card.</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/09-mnist_pyt/README.md">PyTorch flow for Vitis AI</a>
 </td>
 <td>Introduces the Vitis AI TensorFlow design process and illustrates how to go from a python description of the network model to running a compiled model on a Xilinx evaluation board.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/10-RF_modulation_recognition/README.md">RF Modulation Recognition with TensorFlow 2</a>
 </td>
 <td>Machine learning applications are certainly not limited to image processing! Learn how to apply machine learning with Vitis AI to the recognition of RF modulation from signal data.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/11-tf2_var_autoenc/README.md">Denoising Variational Autoencoder with TensorFlow2 and Vitis-AI</a>
 </td>
 <td>The Xilinx DPU can accelerate the execution of many different types of operations and layers that are commonly found in convolutional neural networks but occasionally we need to execute models that have fully custom layers. One such layer is the sampling function of a convolutional variational autoencoder. The DPU can accelerate the convolutional encoder and decoder but not the statistical sampling layer - this must be executed in software on a CPU. This tutorial will use the variational autoencoder as an example of how to approach this situation.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/12-Alveo-U250-TF2-Classification/README.md">Alveo U250 TF2 Classification</a>
 </td>
 <td>Demonstrates image classification using the Alveo U250 card with Vitis AI 1.4 and the Tensorflow 2.x framework.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/13-vdpu-pre-post-pl-acc/README.md">Pre- and Post-processing PL Accelerators for ML with Versal DPU</a>
 </td>
 <td>A complete example of how using the <a href="https://github.com/Xilinx/Vitis-AI/tree/master/demo/Whole-App-Acceleration">WAA</a> flow with Vitis 2020.2 targeting the VCK190 PP board.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/14-caffe-ssd-pascal/README.md">Caffe SSD</a>
 </td>
 <td>Describes the Single Shot Detector (SSD), which is trained with Caffe on the <a href="https://pjreddie.com/projects/pascal-voc-dataset-mirror/">PASCAL VOC</a> dataset. This dataset contains 21 classes, one of which is background.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/15-caffe-segmentation-cityscapes/README.md">ML Caffe Segmentation</a>
 </td>
 <td>Describes how to train, quantize, compile, and deploy various segmentation networks using Vitis AI, including ENet, ESPNet, FPN, UNet, and a reduced compute version of UNet that we'll call Unet-lite. The training dataset used for this tutorial is the Cityscapes dataset, and the Caffe framework is used for training the models.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/16-profiler_introduction/README.md">Introduction Tutorial to the Vitis AI Profiler</a>
 </td>
 <td>Introduces the the Vitis AI Profiler tool flow and will illustrates how to profile an example from the Vitis AI runtime (VART).
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/17-PyTorch-CityScapes-Pruning/README.md">PyTorch CityScapes Pruning</a>
 </td>
 <td>Provides a starting point and demonstration of the PyTorch pruning capabilities for segmentation models.
</td>
 </tr>
 <tr>
  <td align="center">
   <a href="./Design_Tutorials/18-mpsocdpu-pre-post-pl-acc/README.md">Pre- and Post-processing Accelerators for Semantic Segmentation with Unet CNN on MPSoC DPU</a>
 </td>
 <td>A complete example of how using the <a href="https://github.com/Xilinx/Vitis-AI/tree/master/demo/Whole-App-Acceleration">WAA</a> flow targeting the MPSoC ZCU102 board.
</td>
 </tr>
</table>


### Feature Tutorials

 Feature tutorials illustrate specific workflows or stages within Vitis AI.

 <table style="width:100%">
 <tr>
 <td width="35%" align="center"><b>Tutorial</b>
 <td width="65%" align="center"><b>Description</b>
 </tr>
  <tr>
 <td align="center"><a href="./Feature_Tutorials/pytorch-subgraphs/README.md">Partitioning Vitis AI SubGraphs on CPU/DPU</a></td>
 <td>Learn how to deploy a CNN on the Xilinx <a href="https://www.xilinx.com/products/boards-and-kits/vck190.html">VCK190</a> board using Vitis AI.</td>
 </tr>
  <tr>
 <td align="center"><a href="./Feature_Tutorials/tf2_quant_fine_tune/README.md">Fine-Tuning TensorFlow2 quantized model</a></td>
 <td>Learn how to implement the Vitis-AI quantization fine-tuning for TensorFlow2.3.</td>
 </tr>
  <tr>
 <td align="center"><a href="./Feature_Tutorials/Vitis-AI-based-Deployment-Flow-on-VCK190/README.md">Vitis AI based Deployment Flow on VCK190</a></td>
 <td>DPU integration with VCK190 production platform.</td>
 </tr>
  <tr>
 <td align="center"><a href="./Feature_Tutorials/04-tensorflow-ai-optimizer/README.md">TensorFlow AI Optimizer Example Using Low-level Coding Style</a></td>
 <td>Use AI Optimizer for TensorFlow to prune an AlexNet CNN by 80% while maintaining the original accuracy.</td>
 </tr>
  </table>


</hr>
<p class="sphinxhide" align="center"><sup>Copyright&copy; 2020-2021 Xilinx</sup></p>
