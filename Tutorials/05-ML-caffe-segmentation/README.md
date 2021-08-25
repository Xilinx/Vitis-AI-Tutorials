# ML-Caffe-Segmentation-Tutorial
The following is a **tutorial** on how to **train, quantize, compile, and deploy** various segmentation networks including: **ENet, ESPNet, FPN, UNet, and a reduced compute version of UNet that we'll call Unet-lite** using Vitis-AI.  The training dataset used for this tutorial is the **Cityscapes dataset**, and the **Caffe** framework is used for training the models. After training, the **Vitis-AI tools** are used to quantize and compile the models, and ARM C++ application examples are included for deploying the models on a Xilinx **ZCU102** target board.  For background information on ESPNet, ENet, and general segmentation approaches, the [Segmentation Introduction Presentation](files/PDF/Segmentation_Introduction.pdf) has been provided.

Note that the goal of this tutorial is not to provide optimized high accuracy models, but rather to provide a framework and guidance under which segmentation models can be trained and deployed on Xilinx MPSoCs.

The tutorial is organized as follows:

#### 1) Environment Setup and Installation

#### 2) Prepare the Cityscapes database for Training Segmentation Models

#### 3) Training the Models

#### 4) Quantizing and Compiling the Segmentation networks for DPU implementation

#### 5) Evaluating the Floating Point Models on the Host PC

#### 6) Running the Models on the ZCU102

#### 7) Post processing the Hardware Inference Output

# Pre-Install Considerations for Caffe for Segmentation Models

The Vitis-AI tools come with a version of caffe that is supported for training the various segmentation models.  Section 1 will address the various setup steps required for configuring your environment with the Vitis-AI tools.

# 1.0 Environment Setup and Installation

An Ubuntu 16.04 host machine was used for the development and testing of this tutorial.  I use a GTX 1080ti though and the GPU docker has been tested with Docker 19.03.1, NVIDIA driver 410.xx (to work with CUDA 10.0) and nvidia-docker 2.2.2.  The Vitis AI user guide also mentions support for CentOS/RHEL 7.4 and 7.5 though this tutorial has not been tested in those environments.

Note that these networks have also been tested with the older DNNDK v3.0/v3.1 and the DPU v1.4.0 and have been found to work properly with this configuration.  

This tutorial was tested with a ZCU102 revision 1.0 (newer should also be ok),
Displayport monitor, keyboard, mouse, and USB hub with USB micro converter.

The Vitis-AI release used for testing this tutorial is [Vitis-AI-1.0](https://github.com/Xilinx/Vitis-AI/tree/v1.0).

## 1.1 PART 1: Vitis AI Environment Setup and Board Package Installation:

1) Install the Vitis-AI GPU tools via the following steps:

```
git clone https://github.com/Xilinx/Vitis-AI
cd Vitis-AI/docker
```

With version 1.0.0 there is an issue in the docker with the protobuf library version.  It can be resolved by patching the conda_requirements.txt.  To do so, open the conda_requirements.txt in a text editor and add the following two lines:

```
libprotobuf==3.10.1
protobuf==3.10.1
```

Next build the docker image using the following:

```
./docker_build.sh
cd ..
./docker_run.sh xilinx/vitis-ai:tools-1.0.0-gpu
conda activate vitis-ai-caffe
```

At this point the environment should be prepared such that the caffe command using the GPU should be available.  

2) Now copy the [Segment](files/Segment) folder to the docker workspace so that it is available inside the docker once launched.

3) Download the demo image for the desired board.  The images available on the AI Developer Hub support the ZCU104 and ZCU102.  This tutorial should work with either of the boards provided that the compiler is set to target the correct DPU within that board image.  

https://github.com/Xilinx/Vitis-AI/blob/master/mpsoc/README.md

![picture](files/ref_files/pictures/board_images.png)

4) Next follow the steps in the Vitis AI User Guide starting on p20-26 to flash the image to the SD card and copy over the Vitis AI package:

https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf


# 2.0 Prepare the Cityscapes database for Training Segmentation Models

For this tutorial, we’ll be training the models on the Cityscapes dataset.  Cityscapes is an automotive dataset created by Daimler which includes various driving scenes, mostly contained in Germany.  

The files from Cityscapes provide around 5000 images with fine annotations (class labels) for various city driving scenarios. There are two primary folders from the dataset that we'll be working with:

- leftImg8bit (includes all of the input images for training)
- gtFine (includes the class annotations as polygonal format (.json files))

There are also scripts that are separately downloaded for the dataset which are used to transform the class label .json files into class label images (.png files) which are used for training.  

Examples which show the different classes after being color coded and alpha-blended with the original image can be seen on the Cityscapes website: https://www.cityscapes-dataset.com/examples/.

The focus of this database for our purpose is on Semantic Annotations which consist of the following types (we only use the Fine Annotations for this tutorial, though it should also be possible to use coarse annotations and perhaps achieve even better results):

Course Annotations (20000 images)

Fine Annotations (5000 images)


There are 8 groups contained within the Cityscapes dataset with 19 classes.  There are 30 classes listed, but all classes with a ‘+’ next to them are treated as a single void class and preparation steps will change their values to ‘255’ which will subsequently be ignored in the training process.  Classes marked with an * have single instance based annotations available, though this tutorial does not cover instance based segmentation.

Group	Classes
flat	road · sidewalk · parking+ · rail track+
human	person* · rider*
vehicle	car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan*+ · trailer*+
construction	building · wall · fence · guard rail+ · bridge+ · tunnel+
object	pole · pole group+ · traffic sign · traffic light
nature	vegetation · terrain
sky	sky
void	ground+ · dynamic+ · static+

More information about the database can be found at the following URL:

https://www.cityscapes-dataset.com/dataset-overview/


Since preparing such a database for training requires a number of steps, the following detailed instructions are provided:

1) Download the Cityscapes dataset from: https://www.cityscapes-dataset.com/downloads/

  The specific packages needed are the [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) and [lefImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3). These files include the 5000 images with fine (pixel-wise) semantic annotations which are divided into train, test, and validation groups.  It would also be possible to train using the coarse annotations provided by Cityscapes and perhaps achieve better results, but only training with the fine annotations is covered in this tutorial.  

2) Extract these files into a folder named "Cityscapes" under the Segment folder on the Linux workstation inside the docker container.  After this you should have sub-folders labeled "gtFine" and "leftImg8bit" under .  From the introduction, it was noted that these folders contain the class labels and input images.  

3) There are various preparation, inspection, and evaluation scripts provided for Cityscapes which can be cloned from github.  The next step will be to download or clone these using the following and copy them into the Cityscapes folder inside the docker:
```
git clone https://github.com/mcordts/cityscapesScripts.git
```
4)  The scripts can be installed by changing directory into the cityscapesScripts folder then using pip.  If difficulties are encountered due to permissions within the docker with this step, you may either try executing the dataset preparation outside of the docker as only the images and ground truths are needed inside the docker or you can change the permissions by running sudo su and changing permissions using chown for the files needed to install with pip.

```
sudo pip install .
```

5)  Next we need to export the CITYSCAPES_DATASET variable to point to the directory where you extracted the leftimg8bit and gtFine folders.  This environment variable will be used by the preparatory scripts which pre-process the annotations into class labels.  In order to do this, first change directory to the location where the dataset was extracted, then run the following command.  Consider copy/pasting the command as it uses a backtick character surrounding "pwd" and not a single apostrophe.

```
export CITYSCAPES_DATASET=`pwd`
```

6)  The next step is to create the images which have class labels associated with each pixel and set the unused classes to value '255'.  This can be done by running the createTrainIdLabelImags.py script.  To run this script change directory to the cityscapesScripts/cityscapesscripts directory and run the following:
```
python preparation/createTrainIdLabelImags.py
```
  This will convert annotations in polygonal format (.json files) to .png images with label IDs, where pixels encode the “train IDs”.  Since we use the default 19 classes, you do not need to change anything in labels.py script at this time.  We will later go back and change the labels.py for use with evaluating the trained models.

  If you are new to datasets, it may be worthwhile to inspect the .json file to see how the polygon data is stored.  You'll notice it is basically an array of points that connect the lines for each polygonal area for each class.

  After running this script, you will see color coded images which denote the classes as well as trainable images which have the classes encoded in the order determined by the cityscapesscripts/helpers/labels.py.  

  Once the pixels are encoded with trainable values, the different classes are identified as values 0-18 and all of the ignored classes are set to value 255.  Notice that it's very difficult to distinguish the various classes in this image as they are such low values (with the exception of the ignored classes).  These images end in `_gtFine_labelTrainIds.png`.

  Note that it is possible to modify the `cityscapesscripts/helpers/labels.py` to change the class annotations during the preparatory step.

7) During the training process, a text file is used (the path to this file is specified in the input data layer of the model prototxt files) to identify the location of the training data and annotations.  This text file is located under [files/Segment/workspace/data/cityscapes/img_seg.txt](files/Segment/workspace/data/cityscapes/img_seg.txt).  If you placed your cityscapes dataset in a different directory, you will need to modify this text file as well to point to the absolute paths for the input images and associated label images which were just created, which should exist in the subdirectories where the Cityscapes data was extracted.  


  The left column in the img_seg.txt should point to the input image (these are stored under the `Cityscapes/leftImg8bit` directory), the right column should point to the labelTrainIds.png (which are the annotations or ground truths and are stored under the `Cityscapes/gtFine` directory).

  There are many classes that get ignored and their pixel values are set to '255'.  You can note that in the provided model prototxt files, the final softmax and accuracy layers in the network have set a label ignore parameter for value 255 to ignore these classes.  All of the other classes need to start at class 0 and increment.  The prototxt files referred to from here exist in [files/Segment/workspace/model](files/Segment/workspace/model) which includes folders for each of the models that are covered in the tutorial.  

At this point, the training dataset has been prepared and is ready for use to train the models and you can proceed to the next step which is 3.0 Training Models.

# 3.0 Training Models
Prototxt files are included which can be used to train the various models.  Note that these models may differ somewhat from the original models as they have been modified for end use in the DPU IP.  Some of the types of modifications that were made to these models include:

 - Replacing the un-pooling layer with deconvolution layer in the decoder module
 - Replacing all PReLU with ReLU
 - Removing spatial dropout layers
 - Replace Batchnorm layers with a merged Batchnorm + Scale layer
 - Position Batchnorm layers in parallel with ReLU
 - In UNet-full/Unet-lite models Batchnorm/scale layer combinations were inserted before relu layers (after d0c, d1c, d2c, and d3c) as the DPU doesn't support the data flow from Convolution to both the Concat and relu simultaneously

If further analysis is desired, the model prototxt files have been included so they can simply be diff'd from the original caffe prototxt file.

In terms of augmentation, the mean values from the dataset and a scale factor of 0.022 are applied to the input layers for each model.  

## 3.0.1 Training the Models from Scratch

When training from scratch, it is necessary to train ESPNet and ENet models in two stages:
For ESPNet, we will train a model similar to the **(c) ESPNet-C** architecture which can be seen in figure 4 of the [ESPNet publication](https://arxiv.org/pdf/1803.06815.pdf).

This essentially removes the decoder stage that is present in the **(d) ESPNet** model, and in place of that decoder stage, a single deconvolution layer is added to resize up 8x back to the original input size which matches the annotation size.

For ENet, a similar approach is taken and we train only the encoder stage by removing the decoder portion of the model and adding a single deconvolution layer to resize by a factor of 8x up to the original label size which matches the annotation size.

The FPN, Unet, and Unet-lite models can all be trained end to end, so the encoder/decoder two stage training process is not necessary for those models (though a similar process could be employed if desired and it may end up producing better results).

The pre-trained ESPNet/ENet encoder models were trained for 20K Iterations with an effective batch size of 50, and lr_base 0.0005.  Note that larger batch sizes can also be used and may ultimately produce more accurate results, though training time would be increased for larger batch sizes.

If you happen to encounter a situation where you receive a "CUDA out of memory" error, try reducing the batch size in the train_val.prototxt or train_val_encoder.prototxt and increase the corresponding iter_size in the solver.prototxt to maintain the same effective batch size.  Note again that all these models were trained with GTX 1080ti GPUs which have 12GB of memory.  If your GPU has less memory, you may need to adjust the batch sizes to fit within your GPU memory limits.

After this first step has been completed, we can train the full ESPNet prototxt using the weights from the first step to fine tune the model.  

Note that the initial training step for these models takes about 36 hours on my Xeon workstation using a GTX 1080ti graphics card.

### 3.0.1.1 Training the ESPNet (ESPNet-C) and ENet Encoder Models
The encoder models have been included with the caffe distribution. The files which will be needed as a starting point for this are the **solver_encoder.prototxt** and the **train_val_encoder.prototxt**. These files are located under the [files/Segment/workspace/model/espnet](files/Segment/workspace/model/espnet) and [files/Segment/workspace/model/enet](files/Segment/workspace/model/enet) paths respectively, and can be used for training the encoder only portions of these networks.

The full model prototxt files are also available under these paths and it is recommended to compare the two files using a text editor to understand what has been removed from the full model for the encoder only portion.

If you would like to skip training the encoder portion of the model, I have included a pre-trained encoder model for both networks which are stored under the [files/Segment/workspace/model/espnet/encoder_models](files/Segment/workspace/model/espnet/encoder_models) or [files/Segment/workspace/model/enet/encoder_models](files/Segment/workspace/model/enet/encoder_models) directories.  

1) The first step to train these models is to open the **solver_encoder.prototxt** file for the associated model.  It is important to understand the training parameters and paths for this file.  Notice the lines containing the "net: " definition and "snapshot_prefix: ".  

The first line specifies a path to where the train_val_encoder.prototxt exists and the second should point to an existing directory where you would like the model snapshots to be stored.  

  Note also how the other hyper-parameters are set in the solver prototxt.  **The base_lr**, **max_iter**, **iter_size**, and **device_id** are all important training parameters.  

  The **base_lr** is probably the most important parameter and if it is set to big or too small, the training process will never converge.  For this tutorial, it has been found that a size of 0.0005 is an appropriate value for training the models.

  The **iter_size** is used to determine the effective batch size.  If the batch size is set in the train_val_encoder.prototxt file to '5' in the input layer, then the iter_size essentially applies a multiplier to that batch size by not updating the weights until iter_size number of batches have been completed.  For example, if the iter_size is set to 10, then 10 x 5 = 50 is the effective batch size.  Batch size has a significant effect on the convergence of the training process as well as the accuracy of the model, so it is important to maintain a larger batch size when training the full models.  In the case of this tutorial, this parameter is used to enable the training process to maintain a larger effective batch size where there is a limited amount of GPU memory.

  The **device_id** parameter specifies the device id of the GPU card which will be used to accelerate the training process.  If you have only one GPU, specify '0', however, multiple GPUs can also be used by using a comma separated list and you can also train multiple models on different GPUs.

  As noted before the **max_iter** parameter determines how many times the model will see the training data during the training process.  If a dataset has N images and batch size is B, and P is the number of epochs, then the relationship between epochs and iterations is defined as:

  Iterations = (N * P) / B

  Since the training dataset is around 3000 images, we can re-arrange this equation to calculate the number of epochs by:

  (20K*50) /3K = 333 epochs.


2) Once the solver_encoder.prototxt has been verified, the model can be trained by changing directory to the model directory (e.g. /workspace/files/Segment/workspace/model/espnet) and the following command:

```
caffe train \
–solver solver_encoder.prototxt \
2>&1 | tee encoder_models/train_encoder_log.txt
```

I have included an [example log file](files/Segment/workspace/model/espnet/encoder_models/example_log.txt) from my console output during the training process for ESPNet which is stored under files/Segment/workspace/model/ESPNet/encoder_models/example_log.txt.  You should see something similar during the training process.  

Once the training process has completed, the full model can be trained which uses these pre-trained weights as a starting point for the encoder portion of the model.

### 3.0.1.2 Training the Full Models
The full models for ENet, ESPNet, FPN, Unet-Full, and Unet-Lite have been included with the tutorial files. The files which will be needed as a starting point for this are the **solver.prototxt** and the **train_val.prototxt**. These files are located under the [files/Segment/workspace/model/espnet](files/Segment/workspace/model/espnet), [files/Segment/workspace/model/enet](files/Segment/workspace/model/enet), [files/Segment/workspace/model/FPN](files/Segment/workspace/model/FPN), [files/Segment/workspace/model/unet-full](files/Segment/workspace/model/unet-full), and  [files/Segment/workspace/model/unet-lite](files/Segment/workspace/model/unet-lite) paths respectively, and can be used for training the full networks.

Since FPN, Unet-full, and Unet-lite can be trained end-to-end from scratch, there is no need to train the encoder portion separately.  Generally for training the full models, a larger batch size is desirable as it helps the model to approximate the full dataset better than a smaller batch size.  For this tutorial, I have used batch sizes >= 100 for training the full models.

1) Just like with the encoder training, the first step to train the full models is to open the associated **solver.prototxt** file and view the properties of the various hyper-parameters.  Note again that relative paths are used for the "net" and "snapshot_prefix" parameters, so if the files were placed as directed by the tutorial then these files should reside in the appropriate paths.

2) Once the solver.prototxt has been verified, the model can be trained by running one of the following commands (assuming the pretrained models are used, otherwise specify the name of your caffemodel):

**For ESPNet or ENet**:
```
caffe train \
–solver solver.prototxt \
–weights encoder_models/pretrained_encoder.caffemodel \
2>&1 | tee final_models/train_log.txt
```

**For FPN, Unet-Full, or Unet-Lite**:
```
caffe train \
–solver solver.prototxt  \
2>&1 | tee final_models/train_log.txt
```


I have included log files for each of the networks which show the output of the training process:

 - [ENet example log file](files/Segment/workspace/model/enet/final_models/example_log.txt)
 - [ESPNet example log file](files/Segment/workspace/model/espnet/final_models/example_log.txt)
 - [FPN example log file](files/Segment/workspace/model/FPN/final_models/example_log.txt)
 - [Unet-Full example log file](files/Segment/workspace/model/unet-full/final_models/example_log.txt)
 - [Unet-Lite example log file](files/Segment/workspace/model/unet-lite/final_models/example_log.txt)

  Note that these are stored respectively at:

   `files/Segment/workspace/model/enet/final_models/example_log.txt`

   `files/Segment/workspace/model/espnet/final_models/example_log.txt`

   `files/Segment/workspace/model/FPN/final_models/example_log.txt`

   `files/Segment/workspace/model/unet-full/final_models/example_log.txt`

   `files/Segment/workspace/model/unet-lite/final_models/example_log.txt`

You should see something similar during the training process for your full models.  

In general, training the full models is quite time consuming, in many cases >72 hours per model using my ML workstation.  

## 3.0.2 Training the Models using Transfer Learning
If you would like to accelerate the process of training the models, you can also train from transfer learning using the existing models that I have provided.

The pre-trained full models exist at the following paths:

**For ESPNet:**

[files/Segment/workspace/model/espnet/final_models/pretrained.caffemodel](files/Segment/workspace/model/espnet/final_models/)

**For ENet:**

[files/Segment/workspace/model/enet/final_models/pretrained.caffemodel](files/Segment/workspace/model/enet/final_models/)

**For FPN:**

[files/Segment/workspace/model/FPN/final_models/pretrained.caffemodel](files/Segment/workspace/model/FPN/final_models/)

**For UNet-Full:**
This model has been split into a multi-part zip file.  To extract it, 7-zip can be used and the user can CTRL+click and select each individual partial zip, then right click and select extract to extract the pre-trained model.

[files/Segment/workspace/model/unet-full/final_models/](files/Segment/workspace/model/unet-full/final_models/)

**For UNet-Lite:**

[files/Segment/workspace/model/unet-lite/final_models/pretrained.caffemodel](files/Segment/workspace/model/unet-lite/final_models/)

The following steps can be used to either use transfer learning to retrain only the encoder portion or the full model.  The caffemodel that is passed as an argument to the training step should be the appropriate model depending on what the desired approach is:

  - If you intend to use transfer learning with the encoder portion only, then use the pre-trained model under the encoder_models directory for the associated network.  After this step, you can train the full model using the output of this step as the input weights to train the full model.

  - If you intend to use transfer learning with the full model, then use the pre-trained model under the final_models directory for the associated network.  

1) Just like with the training from scratch steps, the first step to train the model is to open the associated **solver.prototxt** file and verify the hyper-parameters.  

2) Once the solver.prototxt has been verified, the models can be trained by changing directory to the desired model directory and running one of the following commands:

**For ESPNet**:
```
caffe train \
–solver solver.prototxt \
–weights final_models/pretrained.caffemodel \
2>&1 | tee caffe-fine-tune-full.log
```

At this point, the model training has been completed and you can proceed to the next step which to is evaluate the floating point model, however, if you are interested about the performance/training of the pre-trained models, please take a gander at the next section: "3.1.0 About the Pre-Trained Models".

## 3.1.0 About the Pre-Trained Models

The pre-trained models included with this tutorial have been trained for various # of iterations and with various batch sizes. Note that training all of the models end to end took about 3-4 weeks on my Xeon server with 2x GTX 1080ti graphics cards.

The full settings used for training these models are captured in the log files and solver prototxt files.  The initial training approach is outlined in the following table, and from this it can be seen that an attempt was made to train the models for ~1000 epochs each.  This extended amount of training allows for exploratory options when picking a suitable model for deployment.

 ![table](files/ref_files/pictures/training_params.png)

Each of the pre-trained models achieves different levels of accuracy in terms of mIOU and some of this variation is due to the training parameters used.  An initial effort was made to keep the total training epochs around 1000 for each model and the effective batch size around 170-175, with the exception of Unet-full as it was an exceptionally large model, so a reduced batch size (and therefore number of epochs) was used to speed up the training process.

Note again that the intention of this tutorial is not to benchmark different models against each other, or even to show a model that works exceptionally well, but rather to show how different segmentation models can be trained, quantized, then deployed in Xilinx SoCs while maintaining the floating point model accuracy.

As the training progressed, regular mIOU measurements were taken using Vitis AI (don't worry if you don't understand this yet, it's covered in section 5 - part 3) to score the models against the Cityscapes validation dataset (500 images). When viewing the plot, recall again that ENet and ESPNet had separate encoder training, so the reduced number of iterations shown in this plot do not visualize that fact.

 ![picture](files/ref_files/pictures/training_progress.png)

It can be seen from the plot that the model with the highest number of iterations does not necessarily correspond to the highest mIOU.  You can also see from the fluctuations in mIOU that perhaps it might be possible to achieve better results by adjusting the learning rate and lr_policy, or by training some of the models for more iterations.  In general, the models with the highest mIOUs were included as the pre-trained model for each respective network in this tutorial.  

- For ENet -> 6K iteration model
- For ESPNet -> 18K iteration model
- For FPN -> 10K iteration model
- For Unet-Lite -> 13K iteration models
- For Unet-Full -> 16K iteration model

Note that ESPNet continued to increase in mIOU at 12K iterations, so an additional 8K iterations of training were performed to find a higher mIOU model.  Additional exploratory training was done for some of the other models as well, but the final models included as pre-trained are captured in the table below which shows the training snapshot used as well as the mIOU as was measured for the floating point model, the quantized model on the host machine, and the model deployed on the ZCU102 hardware.  Again, don't worry if it isn't yet clear how these results were achieved.  The latter sections in this tutorial explain how to measure the mIOU for each of these scenarios.  Note that the one exception is ESPNet which was tested with an older DNNDK board image due to the channel augmentation issue with compiling ESPNet for the Vitis AI board image.

![picture](files/ref_files/pictures/pretrained_mIOU_summary_vai.png)


# 4.0 Quantizing and Compiling the Segmentation networks for DPU implementation

**IMPORTANT NOTE: The ESPNet model will not compile properly for the ZCU102 and ZCU104 boards due to the enablement of the channel augmentation setting in the DPU for those boards.  If you wish to run the ESPNet model, you will need to create your own board image without channel augmentation in the DPU settings or use an older board image from DNNDK v3.0 or v3.1 along with the older DNNDK tools to quantize and compile the model.**

NOTE: The target for this tutorial is the ZCU102, but it should be possible to target other boards as well by changing the target shown above when installing the tools and also modifying the compiler command to target the correct DPU.

Please refer to the [Vitis AI User Guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf) for more details on the Vitis AI tools.

If you would like to quantize and deploy the model, continue onto 4.0 part 1, otherwise if you would like to first test the quantized and floating point models and compare the mIOU between the two, then jump down to 4.0 part 2.

## 4.0 PART 1: Configuring the Files for Quantization, Compilation, and mIOU Testing:

1) I have included an example workspace in [files/Segment/VAI](files/Segment/VAI) to show how the VAI tools may be invoked as well as the necessary modifications to the prototxt files for both quantization/compilation and testing the float and quantized model mIOUs. Change directory to the VAI directory before proceeding to the next step.

2) Within the VAI directory, there is a subdirectory for each model.  Inside each model directory several files:

 - "float.prototxt" is used for quantizing/compiling the models for deployment on the target hardware

 - "float_test.prototxt" is used for testing the float and quantized models to report the mIOU against the cityscapes validation dataset

 - "float.caffemodel" is the pre-trained caffemodel.

 - "quantize_and_compile.sh" is a script that is used to perform both quantization and compilation (vai_q_caffe and vai_c_caffe) for deployment on the target hardware

 - "test_float_and_quantized.sh" is a script that will test both the floating point and quantized models and report out the mIOU for each


3) Open the "float.prototxt" that is included as an example in the VAI subfolders (i.e. ENet, ESPNet, etc.).

  The "float.prototxt" files should be mostly identical to your "train_val.prototxt" except for the following:

  - The input layer has changed from "ImageSegData" type to "ImageData"

  - Absolute paths have been specified to the calibration data so that they point to the correct locations if the steps in the tutorial were followed for populating the data (see step 4 below).

  - Note by default that the prototxt files are set to generate a 512x256 input size model which is intended for use with the xxx_video applications (e.g. fpn_video).  If you wish to run the evaluation in hardware on cityscapes validation images rather than on the recorded video (e.g. fpn_eval), the applications use 1024x512, so you will need to modify these input layers accordingly (the float_test.prototxt files have the input set for 1024x512 if you wish to use this as an example).

```
line 11:  source: "/workspace/files/Segment/VAI/data/cityscapes/calibration.txt"
line 12:  root_folder: "/workspace/files/Segment/VAI/data/cityscapes/calibration_images/"
```

  - The "SoftmaxWithLoss" layer has been changed to "SoftMax" and the "Accuracy" layer has been removed.  These layers were previously used to compute loss and accuracy for the training phase, so they have now been updated for deployment.

**Important note for ENet float.prototxt: the "UpsamplingBilinear2d_x" layers have been changed to "DeephiResize" type because vai_q_caffe doesn't support bilinear upsampling with the deconvolution layer**

You can use these prototxt files directly if the differences mentioned above are the only deltas between your train_val.prototxt file and float.prototxt.  Otherwise, if you are deploying the encoder model only or a modified version, you will need to update your train_val.prototxt to accommodate for the differences mentioned above, rename that file to "float.prototxt", and place it in the correct directory.

4) The calibration data needs to be populated into the [files/Segment/VAI/data/cityscapes/calibration_images](files/Segment/VAI/data/cityscapes/calibration_images) directory.  This data consists of a list of images which are specified in the [calibration.txt](files/Segment/VAI/data/cityscapes/calibration.txt) file, and 1000 test images from Cityscapes.  These will be used by the `vai_q_caffe quantize` process as stimulus for calibration of the model dynamic range during the quantization process.

  The data listed in the calibration.txt file calls out the following 1000 images:
   - the first 100 images from CITYSCAPES_DATASET/leftImg8bit/test/berlin
   - all images from $CITYSCAPES_DATASET/leftImg8bit/test/bielefeld
   - all images from $CITYSCAPES_DATASET/leftImg8bit/test/bonn
   - all images from $CITYSCAPES_DATASET/leftImg8bit/test/mainz
   - the first 373 images from $CITYSCAPES_DATASET/leftImg8bit/test/munich

  You will need to copy these images or potentially create soft links from the dataset directories listed about to the [files/Segment/VAI/data/cityscapes/calibration_images](files/Segment/VAI/data/cityscapes/calibration_images) directory.  You can use other calibration images if desired, however, the provided [calibration.txt](files/Segment/VAI/data/cityscapes/calibration.txt) file uses the images listed above.

5)  Next copy your latest trained model from Caffe into the **files/Segment/VAI/model_subdirectory_name** directory (or reuse the already populated float.caffemodel) and rename it "float.caffemodel".  This model should be located wherever the snapshot was saved from the the training step.

6) Next run the quantization tools using the following command (note that you made need to make the file executable first by running "chmod +x quantize_and_compile.sh"):

```
./quantize_and_compile.sh
```

If you open the script, you will see the following contents which indicate several things - first of all, you should make sure the GPUID environment variable is set correctly for your machine.  If you have only one GPU, this should be '0', otherwise, please change this to the index for the desired GPU to use for quantization.

For reference, I have included an [enet vai_q_caffe log file](files/Segment/VAI/enet/quantize/quantize.txt) that shows the output of my console after running the vai_q_caffe command.  You should see something similar after running the command on your machine.

Finally, the `vai_c_caffe` command is called which compiles the floating point model and produces a file called "dpu_segmentation_0.elf" under the vai_c_caffe_output directory.

For reference, I have included an [enet vai_c_caffe log file](files/Segment/VAI/enet/compile/compile.txt) that shows the output of my console after the vai_c_caffe command is run.  You should see something similar after running the command on your machine.

```
#!/usr/bin/env bash
export GPUID=0
net=segmentation

#working directory
work_dir=$(pwd)
#path of float model
model_dir=quantize
#output directory
output_dir=compile

echo "quantizing network: $(pwd)/float.prototxt"
vai_q_caffe quantize          \
          -model $(pwd)/float.prototxt     \
          -weights $(pwd)/float.caffemodel \
          -gpu $GPUID \
          -calib_iter 1000 \
          -output_dir ${model_dir} 2>&1 | tee ${model_dir}/quantize.txt

echo "Compiling network: ${net}"

vai_c_caffe   --prototxt=${model_dir}/deploy.prototxt \
        --caffemodel=${model_dir}/deploy.caffemodel \
        --output_dir=${output_dir} \
        --net_name=${net} \
        --arch=/opt/vitis_ai/compiler/arch/dpuv2/ZCU102/ZCU102.json 2>&1 | tee ${output_dir}/compile.txt
```      


At this point, an elf file should have been created in the **compile** directory which can be used in the final step which is to run the models on the ZCU102.  If desired, you can also proceed to the Part 3 of 4.0 which is testing the floating point and quantized models.

## 4.0 PART 2: Testing the Floating Point and Quantized Models
As mentioned in the previous section, files have been provided under the `files/Segment/VAI/model_subdirectory_name` filepath which can enable you to rapidly test the mIOU of both the floating point model as well as the quantized model on the cityscapes validation dataset.  In order to perform this testing, perform the following steps:

1)  Open the `files/Segment/VAI/data/val_img_seg_nomap.txt` file with a text editor.

2)  Notice that this file contains paths to the cityscapes validation dataset as would be expected if the dataset was placed in the recommended location. The left column has a path to the input image, and the right column has a path to the labels.  You may need to modify the root directory portion of both paths to point to the location of the cityscapes dataset on your machine if you placed your dataset in a location other than what was recommended in this tutorial.

3) Open the [float_test.prototxt](files/Segment/VAI/FPN/float_test.prototxt) file that corresponds to the model of interest.  Notice that there are several differences between this file and the [float.prototxt](files/Segment/VAI/FPN/float_test.prototxt) that was used for deployment.  The reason for this is that the DeephiResize layer causes some problems in the current version of vai_q_caffe which will prevent the compiler from compiling the model (it causes the input layer to be renamed to "resize_down" which causes the compiler to fail- for this reason two separate files are used, one for testing and one for deployment).

The new additions to this model are to support the auto_test and test commands:

- The input size of the model has been changed from 512x256 to 1024x512.  This is because the larger input size produces better mIOU results.  It would be possible to use other sizes such as the native input size for the cityscapes dataset which is 2048x1024, but testing the models would take longer and the Unet-full model will not work in this case because of some limitations on the Caffe distribution used within the vai_q_caffe tool.  Additionally, the models were trained with an input crop size of 512, so it is not necessarily expected that using the larger size will produce better results.

- An additional input layer "ImageSegData" has been added which has a path to the val_img_seg_nomap.txt file.  This is how the labels and input images are supplied for the testing procedure.

- A layer after this called "resize_down" has been added to scale the input image to the desired input size for the model (in this case 1024x512).

- A new layer at the end of the model has been added called "SegmentPixelIOU" which is a custom caffe layer packed up within the vai_q_caffe tool.  If you noticed, the val_img_seg_nomap.txt file actually points to the *gtFIne_labelIds* rather than *gtFine_labelTrainIds*. This is because the SegmentPixelIOU layer has been coded to automatically relabel the classes from the cityscapes labels such that the classes match gtFine_labelTrainIds and values 255 are ignored.

4) Open the one of the `test_float_and_quantized.sh` scripts.  The contents of this script are shown below.  You will only need to edit the GPUID to specify the correct GPU index for your tests.  Note that the log files will be captured under the test_results subdirectory for both the floating point and quantized results.

```
export GPUID=0
export WKDIR=`pwd`

cd $WKDIR
vai_q_caffe test -model float_test.prototxt -weights float.caffemodel -test_iter 500 -gpu $GPUID 2>&1 | tee test_results/float_model_test.txt

#working directory
work_dir=$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}

vai_q_caffe quantize            \
          -model ${model_dir}/float_test.prototxt     \
          -weights ${model_dir}/float.caffemodel \
          -gpu $GPUID \
          -calib_iter 1000 \
          -test_iter 500 \
          -auto_test \
          -output_dir ${output_dir} 2>&1 | tee test_results/quantized_model_test.txt

```

5) Execute the Script by running the following command.  This will take some time which varies depending on the GPU hardware available as well as which model is being run.  I have included example test results from a previous run under the associated model directories such as [files/Segment/VAI/FPN/test_results](files/Segment/VAI/espnet/test_results).  Note that the previous run results I have included does not necessarily represent the best performing snapshot - it is just an example of the output of running the test script.

```
./test_float_and_quantized.sh
```

At this point, the quantized and floating point models have been fully verified on the host and you are ready to proceed to deploying the models to the target hardware, however, if you skipped the section on pre-trained models you may be wondering how they scored.  Jump back up 3.1.0 About the Pre-Trained Models to see the results.

# 5.0 Evaluating the Floating Point Models on the Host PC

## 5.1 Evaluating the Models with Still Images and Displaying the Results
After training the models, it is useful to evaluate the model on the host PC before attempting to deploy it in the DPU.  In order to evaluate the models, scripts have been provided under the [files/Segment/workspace/scripts/test_scripts](files/Segment/workspace/scripts/test_scripts) directory.  

You can test both the encoder only models as well as the full models for all of the networks.  To do so, execute the following steps.

1) Change directory to the `files/Segment/workspace/scripts/test_scripts` directory

2) The following scripts can be used to evaluate the various models on a single image:

- test_enet_encoder.sh
- test_enet.sh
- test_espnet_encoder.sh
- test_epsnet.sh
- test_fpn.sh
- test_unet-full.sh
- test_unet-lite.sh

The evaluation scripts contain paths to the caffemodel which will be evaluated.  If you want to evaluate the pre-trained models that have been provided, you can simply execute them as is, otherwise, you will need to modify the path to the caffemodel and prototxt file.  An example is shown below for enet which uses relative paths to point up to the model directory where the prototxt and trained snapshot exist.

```
export PYTHONPATH=../../../caffe-master/python
python test_enet.py \
--model ../../model/enet/deploy.prototxt \
--weights ../../model/enet/final_models/pretrained.caffemodel \
--input ../munich_000000_000019_leftImg8bit.png \
--out_dir ./
```
As can be seen from the screenshot above, these scripts will call the underlying python script which will run one forward inference on the trained model using the "munich_000000_000019_leftImg8bit.png" image as the input.  

The output will then color code the classes and display the output image.  You could modify the python program to work on a recorded video or a webcam if desired, though the scope of this tutorial only provides an evaluation of a single images.

The input image and example outputs for the pre-trained networks can be seen below.  Note the differences between the two networks as well the smoothing effect that including the full decoder stage has on the output.

**Note: You will need to first copy the image "munich_000000_000019_leftImg8bit.png" from the cityscapes dataset leftImg8bit folder to the /workspace/files/Segment/workspace/scripts/ folder before running the test inferences.**

ENet Encoder Only Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bit_enet_encoder.png)

ENet Full Model Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bit_enet.png)

ESPNet Encoder Only Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bitESPNet_encoder.png)

ESPNet Full Model Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bitESPNet.png)

FPN Model Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bit_fpn.png)

Unet-Full Model Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bitUnet-full.png)

Unet-Lite Model Output:
![picture](files/Segment/workspace/scripts/test_scripts/munich_000000_000019_leftImg8bitUnet-lite.png)

**If the output of your model looks reasonable, you can now proceed to step 5, however, there are other tests that may be performed at this point which are outlined in the remainder of this section.**

It is also possible to test the inference time of the models on the host machine by running the following command from $CAFFE_ROOT:

**For ENet Encoder Only:**
```
./build/tools/caffe time \
-model ../workspace/model/enet/deploy_encoder.prototxt -gpu 0 -iterations 100
```
**For ENet Full Model:**
```
./build/tools/caffe time \
-model ../workspace/model/enet/deploy.prototxt -gpu 0 -iterations 100
```
**For ESPNet Encoder Only:**
```
./build/tools/caffe time \
-model ../workspace/model/espnet/deploy_encoder.prototxt -gpu 0 -iterations 100
```
**For ESPNet Full Model:**
```
./build/tools/caffe time \
-model ../workspace/model/espnet/deploy.prototxt -gpu 0 -iterations 100
```

My results for these various tests with a Xeon CPU and GTX 1080ti graphics card and a 512x256 input size for the models are as follows (note that the GPU power was estimated using the nvidia-smi command while running forward inference):

**For ENet Encoder Only: Average Forward Pass 17 ms, 94W**

**For ENet Full Model: Average Forward Pass 27 ms, 93W**

**For ESPNet Encoder Only: Average Forward Pass 9.1 ms, 108W**

**For ESPNet Full Model: Average Forward Pass 10.3 ms, 108W**

**For FPN Model: Average Forward Pass 15.5 ms, 114W**

**For UNet-Full Model: Average Forward Pass 33 ms, 220W**

**For UNet-Lite Model: Average Forward Pass 14 ms, 176W**


## 5.2 Measuring the Floating Point Model mIOU on the Cityscapes Validation dataset
In order to test the floating point and quantized model mIOU, the primary method is to use the vai_q_caffe test command.  A local copy of vai_q_caffe is provided rather than the publicly distributed vai_q_caffe, as this local copy provides the capability to test both the floating point and quantized models.

In each of the model subdirectories under the [files/Segment/VAI](files/Segment/VAI) filepath, a script has been provided that will perform the following actions:

- Test the floating point model against the cityscapes validation dataset (run inference and calculate mIOU)

- Quantize the model using the calibration images from the [files/Segment/VAI/data/cityscapes/calibration_images](files/Segment/VAI/data/cityscapes/calibration_images) directory (Need to first check step 4.0 on Quantization as directions are provided there for populating this directory).

**To use this primary method, please make sure you have covered section 4.0 on quantization as some of the steps covered in that section will be needed to complete this step.**

In addition to using the vai_q_caffe capabilities to test the floating point model, one secondary method is included in this section. This method employs the use of Caffe along with some python scripts to run forward inference on the floating point model and check the mIOU. 

#### Floating Point Model Test Using Python on Host Machine  
- Change directory to [files/Segment/workspace/scripts/eval](files/Segment/workspace/scripts/eval)

- Make sure your CITYSCAPES_DATASET variable is exported properly to the location of dataset.  If you have not done this, a default location will be used which will cause the script to fail.

- Next you need to prepare the validation images that will be used as a ground truth for comparison against the model output.  This can be done by running the cls34_to_cls19.py script by entering `python -cls34_to_cls19.py`.  This step only needs to be performed once for the ground truth images.  Note that these will be stored in a folder called `test_gtFine_cls19` where your CITYSCAPES_DATASET is located.

- Next you need to run forward inference using the validation images.  This can be done using the `forward_inference_model_name.sh` scripts provided (e.g. `./forward_inference_enet.sh`).  These scripts will create a soft link to the pretrained caffe models in the working directory where the snapshots of the training process are stored.  Then the script will run segmentation_miou_test.py to perform forward inference using that model and store the results under the "results" directory.

- Finally, with the results captured and ground truth images prepared, run `./eval_segmentation.sh` script to compare the ground truth images and report the mIOU.  The mIOU for all classes will be the first number reported and the other numbers reported will be the per class mIOU numbers.

# 6.0 Running the Models on the ZCU102
The final step in this tutorial is to run the models on the target hardware - in my case, a ZCU102, but other development boards may also be used (just make sure you changed the vai_c_caffe command when compiling the model to target the correct DPU json associated with your board).  In order to expedite the process for running on the target hardware, I have included software applications for each model that allow you to perform two different types of tests:

- Test the model using a video file and display the output.  By default, this uses a 512x256 input size, and a video file titled "traffic.mp4" has been included under [files/ZCU102/samples/video](files/ZCU102/samples/video) directory.  The software application in the "model_name_video" (e.g. enet_video) subfolder can be used to test the model with this video file.

- Run forward inference on the model using the cityscapes validation dataset and capture the results for post processing back on the host machine.   This uses a 1024x512 input size by default to match the mIOU tests that were done on the host previously, but other input sizes may also be used.  The results of running this software application will be captured under the `software/samples/model_name/results` folder.  The software application in the "model_name_eval" (e.g. enet_eval) subfolder can be used to perform this test.  

In order to run the tests, it is assumed that you have a ZCU102 revision 1.0 or newer connected to a DisplayPort monitor, keyboard and mouse, and an Ethernet connection to a host machine as described in step 1.1.  Then perform the following steps:

1) Copy the cityscapes validation images into the `files/ZCU102/samples/cityscapes` folder on your host machine.  After this you should have a directory structure that looks like the following:

![picture](files/ref_files/pictures/directory_structure_eval_application.png)

2) At this point you can either copy over your compiled model .elf file (output of vai_c_caffe) into the model subdirectory such as [files/ZCU102/samples/enet_eval/model/dpu_segmentation_0.elf](files/ZCU102/samples/enet_eval/model/), or use my pre-populated models that have already been included in that directory.  Note that the makefile assumes the name of the .elf file is "dpu_segmentation_0.elf" and it also assumes that you used the name "net=segmentation" when compiling the model with vai_c_caffe.  You can see this in the [files/ZCU102/samples/enet_eval/src/main.cc](files/ZCU102/samples/enet_eval/src/main.cc) files for each of the associated applications.

3) Boot the ZCU102 board

4) Launch the Linux terminal using the keyboard/mouse connected to the ZCU102

5) Configure the IP address for a compatible range with your host machine (I use 192.168.1.102 and then set my laptop to 192.168.1.101).  The command needed to perform this step is:

`ifconfig eth0 192.168.1.102`

6) Launch a file transfer program - I like to use MobaXterm, though others such as pscp or WinSCP can also be used.  

7) Transfer the entire samples directory over to the board

8) Change directory into one of the video based examples such as 'enet_video'

9) Run `make clean`, then `make -j4` to make the sample application.  This will compile the ARM application and link in the DPU model (located under the models subdirectory) with the ARM64 executable to run on the A53 cores.  

10) To execute the application, use the following command:

**for the *_video applications**
```
./segmentation ../video/traffic.mp4
```
**for the *_eval applications**
```
./segmentation
```
  - When running the video application, you should see the video play back with overlays that color code the various classes from the scene onto the monitor.  After the playback is completed, an FPS number will be reported in the terminal which summarizes the forward inference performance from the hardware run.

  - When running the evaluation application, you should see a list of the validation images on the console being read into DDR memory, then the text ***Processing***, after which there will be delay while forward inference is run on the images.  The final output will be 500 images with the class labels for pixel values stored into the results directory.  These images can then be post processed back on the host machine to extract the mIOU score for the model.

I have observed the following performance using the pre-trained models on the ZCU102 (the numbers may vary a bit here when reproducing the results on your board):

![picture](files/ref_files/pictures/zcu102_forward_inference_fps.png)

Using data gathered throughout this tutorial, we can compare the performance of the ZCU102 vs. the GTX1080ti graphics card that was used to time the models from section 4.1.  Albeit, this isn't a fair comparison for two reasons:
 1. We are comparing an embedded ~20W device with a 225W GPU
 2. The ZCU102 execution time includes reading/preparing the input and displaying the output whereas the GPU measurement only includes the forward inference time of the models

 That said, this still provides some data points which are useful to garner further understanding.  The following chart shows a comparison between the FPS as measured on the ZCU102 vs. the GTX1080ti.

![picture](files/ref_files/pictures/fps_zcu102.png)

What is perhaps a bit more useful than comparing raw FPS, however, is to compare FPS/W (performance/Watt) as this is a more generic comparison of what performance is achievable for a certain power cost.  Bear in mind, this is still not a fair comparison due to reason 2, but the value of a Xilinx SoC starts to shine a little more in this light.  In reality the advantage is even more pronounced if only the DPU throughput is considered.

In order to perform this comparison, ~20W was measured on the ZCU102 board during forward inference, and the nvidia-smi tool was used to read the power during forward inference of each of the models as part of section 4.1.  The comparison between the two can be seen in the following figure.

![picture](files/ref_files/pictures/fps_per_w.png)

The same data can be seen in table format below:

![picture](files/ref_files/pictures/FPS_per_watt_table.png)

This comparison is still based on an embedded application that has to deal with processing video files, so in order to make a more apples-to-apples comparison, the new Vitis AI Libraries were used to test the performance with different numbers of threads, and the performance increases significantly with this approach for several of the models.  The results can be seen in the following table.  Note that ESPNet is not included in this comparison as the aforementioned issue with the enablement of channel augmentation prevented this model from being run with the ZCU102 board image.  

![picture](files/ref_files/pictures/ai_sdk_performance.png)

At this point, you have verified the model functionality on the ZCU102 board and the only step left is to post process the images if you ran the evaluation software.  The mIOU score from this can then be compared the mIOU that vai_q_caffe measured previously on the host machine. In order to do this, proceed to the final step, "7.0 Post processing the Hardware Inference Output".

# 7.0 Post processing the Hardware Inference Output

In order to post process the output of the hardware run, scripts have been provided under the [files/Segment/workspace/scripts/postprocess](files/Segment/workspace/scripts/postprocess) folder.  

The following steps can be performed to complete this process:
- Copy the results folder from the target hardware back to the host machine into the [files/Segment/workspace/scripts/postprocess](files/Segment/workspace/scripts/postprocess) folder.

- Make sure your $CITYSCAPES_DATASET variable is exported properly to the location of dataset.  If you have not done this, a default location will be used which will cause the script to fail unless your locations matches what is used as the default in the script.

- Next you need to prepare the validation images that will be used as a ground truth for comparison against the model output.  This can be done by running the cls34_to_cls19.py script by entering `python -cls34_to_cls19.py`.  This step only needs to be performed once for the ground truth images.  If you already completed this as part of section 4.2, you can skip this step.  Note that these will be stored in a folder called `test_gtFine_cls19` where your $CITYSCAPES_DATASET is located.

- Now run the eval_segmentation.sh script by entering `./eval_segmentation.sh`.

  The output of this step should have a final line in parenthesis which are a list of IOUs.  The first value in this list is the mIOU for all the classes (this is the number to compare to the vai_q_caffe quantized model mIOU).  The other numbers are per class IOU numbers for the validation dataset.  I already completed this step for the pre-trained models and you can refer back to section "3.1.0 About the Pre-Trained Models" to see the results.



# Summary
Looking back, we've covered a lot of ground, including walking through the process of preparing, training, testing, and deploying 5 different segmentation models.  The goal of this tutorial was not to show a perfectly optimized solution, but rather to blaze a trail so you experts and explorers can streamline your own segmentation model development and rapidly deploy those models on a Xilinx SoC/MPSoC.  

The beauty of this solution is that there is a full portfolio of [Zynq-7000](https://www.xilinx.com/support/documentation/selection-guides/zynq-7000-product-selection-guide.pdf) and [Zynq Ultrascale+](https://www.xilinx.com/support/documentation/selection-guides/zynq-ultrascale-plus-product-selection-guide.pdf) devices (qualified for commercial, industrial, automotive, and aerospace and defense end markets) and various DPU configurations that allow you to scale for low power applications that that require < 3W as well as dial up the performance for higher end products (where you may need hundreds to thousands of FPS), such as deployments on PCIe accelerator class products such as [Alveo](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html).  

All of this is possible while using the same design approaches and tool flow, without re-inventing algorithms for low end vs. high end products.  You can even make trade-offs for DPU size vs. traditional pre/post-processing hardware acceleration (e.g. optical flow, stereo block matching, scaling, de-interlacing, FFTs, or even custom Image Sensor Pipelines).  The number of potential implementations are virtually endless, so you can truly build an optimal solution for your application that maximizes your differentiation.
