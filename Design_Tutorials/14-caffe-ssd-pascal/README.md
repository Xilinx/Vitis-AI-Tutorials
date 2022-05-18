<table class="sphinxhide">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis-AI™ 1.4 - Machine Learning Tutorials</h1>
    <a href="https://www.xilinx.com/products/design-tools/vitis.html">See Vitis™ Development Environment on xilinx.com</br></a>
    <a href="https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html">See Vitis-AI™ Development Environment on xilinx.com</a>
    </td>
 </tr>
</table>

# VAI Caffe SSD Tutorial

This tutorial on the Single Shot Detector (SSD) is trained with Caffe on the [PASCAL VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) dataset. This dataset contains 21 classes, one of which is background.

The topics covered in this tutorial include training, quantizing, and compiling SSD using PASCAL VOC 2007/2012 datasets, the Caffe framework, and Vitis AI tools. The model is then deployed on a Xilinx® ZCU102 target board and could also be deployed on other Xilinx development board targets (For example, Kria Starter Kit, ZCU104, and VCK190).
If you are new to the SSD network, the [SSD_0_Background.pdf](PDF/SSD_0_Background.pdf) contains a compilation of background information on the SSD network. It provides a helpful introduction.

One of the valuable things about SSD (besides providing solid detection results) is the ability to plug and play with different backbone feature extractors. This allows flexibility to make tradeoffs of performance between different backbones on your dataset and tradeoff performance versus latency.

This tutorial covers two different uses of SSD:

  * SSD with VGG16 backbone using 300x300 input size
  * Mobilenetv2-SSD using 480x360 input size

The goals of this tutorial are to:
- Demonstrate the workflow to go from Caffe to Xilinx embedded targets
- Highlight the viability of INT8 performance/accuracy
- Highlight the value of the Xilinx model zoo by leveraging a pre-trained Mobilenetv2-SSD model and retraining it on a different dataset with a different number of classes
- Demonstrate the Caffe quantized model finetuning flow
- Blaze a trail for those explorers among you who may be new to SSD or detection models

You can understand how to a model from scratch using a publicly available backbone and how we finetuned a model from the model zoo on a different dataset.

The tutorial is organized as follows:

1. Install the Caffe tool for SSD

2. Prepare the dataset and database

3. Train the SSD networks

4. Evaluate the Floating Point Network

5. Quantize and Compile the Network

6. Run the application on the ZCU102 board

You may also be wondering, "why a Caffe tutorial, isn't Caffe old?". The primary answer is that this is a reincarnation of an older DNNDK tutorial.  Caffe is also a great way to learn the fundamentals of ML which apply to all the different frameworks without actually having to know Python and framework APIs.  

## 1.0 Pre-Install Considerations for Caffe for SSD

The public version of SSD Caffe is posted by Wei Liu and the associated official version of SSD Caffe is available at https://github.com/weiliu89/caffe/tree/ssd.  You can find the original SSD publication here as well if you're interested in the theory and operation behind it:  https://arxiv.org/pdf/1512.02325.

This tutorial, however, uses Caffe provided within [Vitis AI 1.4](https://github.com/Xilinx/Vitis-AI/tree/v1.4).

**Note:** This tutorial is not compatible with Vitis AI 1.0 because the SSD network could not be trained in that environment.

## 1.1 Environment Setup and Installation

This tutorial is tested with a ZCU102 revision 1.0 (newer should also be ok), DisplayPort monitor, keyboard, mouse, and USB hub with USB micro converter.


### 1.1.0: Vitis AI Environment Setup and Board Package Installation:

1. Install the Vitis AI GPU tools using the following steps:

  ```
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI  
cd Vitis-AI
git checkout v1.4
cd docker
```

2. Build the docker image using the following:

  ```
./docker_build_gpu.sh
cd ..
./docker_run.sh xilinx/vitis-ai-gpu:latest
conda activate vitis-ai-caffe
```
At this point, the environment should be prepared and the Caffe command using the GPU should be available.  

  (**Tip**: Paths to datasets outside of the docker environment can be made visible within the docker by editing the `docker_run.sh` script.  Under the docker run command in the script, add a path with `-v path_outside_docker:path_inside_docker`.  For instance, if you have a path where data is stored locally at /data2/datasets, it can be made available inside the docker by adding `-v /data2/datasets:/data2/datasets`).

3. Copy the [SSD](SSD) folder to the docker workspace so that it is available inside the docker once launched.

4. Download the demo image for the desired board.  The images available in the Vitis AI repository include the [ZCU104, ZCU102, Kria KV260](https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc/VART), and [VCK190](https://github.com/Xilinx/Vitis-AI/tree/master/setup/vck190).  This tutorial should work with any of the boards provided that the compiler is set to target the correct DPU within that board image.  


5. Next follow the steps in the [Vitis AI User Guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_4/ug1414-vitis-ai.pdf) starting on page 25 and continue through page 33 to flash the image to the SD card.


## 2.0 Prepare the VOC database for SSD

To prepare the VOC0712 database, follow the recipe below:

1. Download the PASCAL VOC dataset from the following three links:
 - http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
 - http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
 - http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

2. Copy the three TAR files into the folder in the docker named /workspace/SSD/data. Extract all files from the three archives and merge them under a single VOCdevkit folder. To do so, CTRL+click all three files, and then right-click and select **Extract**.

  At the end of this process, you should have a new folder named **/workspace/SSD/data/VOCdevkit** containing the VOC2007 and VOC2012 subfolders.  **Note:** This dataset includes the training, validation, and testing images and the annotations, which are the bounding box locations included in .xml format.

  For the next step, you will be processing the annotations (labels) and the training images into LMDB files which Caffe can directly use for the training process.

  You need two files [create_list.sh](https://github.com/intel/caffe/blob/master/data/VOC0712/create_list.sh) and [labelmap_voc.prototxt](https://github.com/intel/caffe/blob/master/data/VOC0712/labelmap_voc.prototxt) which are available from this repository: https://github.com/intel/caffe/tree/master/data/VOC0712.

3. cd into the /workspace/SSD/data folder within the docker and copy the `create_list.sh` and labelmap_voc.txt files into this directory:

4. Open the `create_list.sh` script and add the following line at the start of the file:

  ```
DATAPATH=/workspace/SSD
```

5. Next change line 41 from:

  ```
$bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
```
  to:
  ```
get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
```

6. Execute the following commands:

  ```
source create_list.sh
bash ./build_lmdb.sh
```

  The first command creates the trainval.txt, test.txt, and test_name_size.txt files in the current directory. The second command calls a binary file "convert_annoset" which is a precompiled Caffe tool for preparing the lmdb (included in the Xilinx docker image).

You should now see something similar to what is shown in this example:

![Figure 1](ref_files/pictures/build_lmdb.png)

**Note:** This process created the LMDB databases stored in **/workspace/SSD/data/VOC/VOCdevkit/VOC0712/lmdb**.  The training lmdb is ~1.5GB the test lmdbd is ~445MB.

At this point, the training dataset has been prepared and is ready for use to train the SSD model and you can proceed to the next step which is training the model.

## 3.0 Training the SSD Models

### 3.0.1 Preparing the VGG-16 SSD Model for Training

1. Download the pre-trained [VGG_ILSVRC_16_layers_fc_reduced.caffemodel](https://github.com/conner99/VGGNet) (weights) files of the VGG network.  This will be used as the backbone (feature extraction layers) for our SSD network and since they are pre-trained, it helps to reduce the training time.  

2. Copy this file into the **/workspace/SSD/workspace** folder.

3. Example trainable [train_test_example.prototxt](SSD/workspace/VGG16-SSD/train_test_example.prototxt) and [solver.prototxt](SSD/workspace/VGG16-SSD/solver.prototxt) files have been provided under [SSD/workspace/VGG16-SSD](SSD/workspace/VGG16-SSD).  

  Output snapshots will be incrementally created under the following directory: **/workspace/SSD/workspace/VGG16-SSD/snapshots**.

### 3.0.2 Understanding the SSD Prototxt Files
There are a few notable differences between the official SSD model that gets created as part of Wei Liu's tutorial/caffe distribution.

We changed **mbox_conf_reshape**, **mbox_conf_softmax**, and **mbox_conf_flatten** layers to include the phase:test parameter so that these layers are only used for testing.  The entry that was added is:

```
include{
  phase: TEST
}
```

This was inserted in each of the layers just after the **top:** declaration and before the layers other parameters.

We replaced the Normalize layer with a BatchNorm/Scale combined layer because the DPU does not support Normalize.  In the original SSD model, this is called "conv4_3_norm".  Use the following steps to replace the layer:

1. Delete the Normalize layer named *conv4_3_norm*, which should start at around line 972 and end around line 985.

2. Insert the Batchnorm/scale layer *conv4_3* and *relu4_3* layers.

3. Replace the other bottom layers in the prototxt named *conv4_3_norm* with *conv4_3* (there should be three of these) and they appear as `bottom: "conv4_3_norm"`.

We modified the *fc6* layer by changing the num_output to 682 instead of 1024.  It is because the DPU only supports 12 bits to describe the number of parameters.  With a dilation of 6, the output parameters are 6143, which is too many for the DPU to capture.

Because of this change to the layer, we renamed both fc6 and fc7 layers: named *fc6* as *fc6_682*, and named *fc7* as *fc7_682*.

Taking a look at the [solver.prototxt](SSD/workspace/VGG16-SSD/solver.prototxt), this file has the training hyper-parameters and also points to the net that is being trained.  In this file, we use a combined train/test prototxt file instead of two separate files for train and test:

```
net:  "train_test_example.prototxt"
```

Notice the snapshot_prefix location, which is where the training will produce output models. This can be modified to place the snapshot output model in the desired directory. We are leaving the snapshots at this location with the recommended prefix for this tutorial. It allows for easier evaluation of the floating-point model without modifying other scripts.

```
snapshot_prefix: "./snapshots/snapshot"
```

The number or frequency of snapshots are determined based on the parameters specified in the solver and note that by default the first snapshot will occur after 10 thousand iterations.

The training process requires about 10GB of GPU memory in the current configuration.  If your GPU does not have sufficient memory, you can modify the train_test.prototxt file and change the `batch_size` parameter in the input layers to a smaller value such as 8, 4, 2, or 1.  

Since the training typically uses a batch_size of 32, if you modify this value, you should also change the solver.prototxt iter_size parameter such that the product of iter_size x batch_size = 32.  For instance, if you change the batch_size to be 8, you should change the iter_size to be 4 so that the effective batch size used for the training process is 4x8 = 32.  

### 3.0.3 Train It!

The Vitis AI docker comes with Caffe tools so to train the model, all you need to do is execute the following commands:

```
cd /workspace/SSD/workspace/VGG16-SSD
caffe train \
  -solver solver.prototxt \
  -weights VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
  -gpu 0 2>&1 | tee SSD_train.log
```

Notice that we are passing in the weights for the backbone from VGG. This is called finetuning or retraining and allows us to leverage someone else's hard work and energy bill to reduce our training time and provide us with better model accuracy.

The 120 thousand iterations training process takes about 60 hours on a Xeon machine with a GTX 1080 Ti graphics card, so go ahead and re-binge. Watch the last three seasons of your favorite series while you wait for this or do some other work.  

To extract it, 7-zip can be used in Windows, and you can CTRL+click and select each partial zip, then right-click and select **Extract**. If using Linux, right-click the part .001 file and select **Extract Here**.

Here is an example log file of the training process: [logfile_ssd_pascal.txt](SSD/workspace/VGG16-SSD/train.log). If everything goes smoothly, you will get something very similar.

The model has been trained and is now ready for evaluation. If you want to skip evaluating the floating point model and jump straight to quantization, go ahead to step 5.

## 3.1 Prepping and Training the Mobilenetv2-SSD model
I suspect many won't be so interested in the VGG16-SSD model these days, so maybe a more relevant (optimized) flavor is the Mobilenetv2-SSD model.  It's called Mobilenev2-SSD because it uses Mobilenetv2 for the backbone instead of VGG16.  With some imagination, you could extend this concept also to have an inception-SSD or ResNet-SSD, etc.

This model is available in the [Vitis AI model zoo](https://github.com/Xilinx/AI-Model-Zoo/tree/master).  For this tutorial, we use the  cf_ssdmobilenetv2_bdd_360_480_6.57G_1.1 model as the starting point. It can be downloaded directly through the following link:

https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdmobilenetv2_bdd_360_480_1.1.zip

The model initially has been trained on the 11 class BDD dataset.  Since we are using PASCAL VOC for this tutorial, we need to change the number of classes for the model.  Again, the hope here is that a concept is conveyed that models can be modified for different numbers of classes and datasets.  

If you are not interested in the mechanics, you can skip this section and use my pre-modified [SSD/workspace/Mobilenetv2-SSD/train_test_example.prototxt](SSD/workspace/Mobilenetv2-SSD/train_test_example.prototxt)

For SSD there are a few layers that we need to modify:

1. Download the model from the link above

2. Extract the package and copy the float/trainval.prototxt file and float/trainval.caffemodel into the SSD/workspace/Mobilenetv2-SSD directory

3. Open the trainval.prototxt file in a text editor

  a. First, change the paths to the dataset.  Since our dataset is located in a different spot and has a different name, update the following paths:

  - change line 48 to point to: "/workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb"
  - change line 135 to point to: "/workspace/SSD/data/labelmap_voc.prototxt"
  - change line 159 to point to: "/workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb"
  - change line 167 to point to: "/workspace/SSD/data/labelmap_voc.prototxt"
  - change line 4642 to point to: "/workspace/SSD/data/test_name_size.txt"

  b. Now, change the number of classes that the model targets.  This is mainly related to the number of bounding boxes that are produced for the various all these various classes, so it requires a little bit of knowledge about the SSD model architecture. The more obvious changes are as follows:

  - change line 4551 from '11' to '21'
  - change line 4577 from '11' to '21'
  - change line 4616 from '11' to '21'
  - change line 4638 from '11' to '21'

Ok, we're done, right?  Well not quite yet - you probably already guessed there's a little more to it.

If you go back and check out the [SSD_0_Background.pdf](PDF/SSD_0_Background.pdf), slide 19 indicates that there is a dependency on the number of output channels for some of these layers as to how many classes we are targeting.

I will take some of the mystery out of this step and tell you that the particular layers of interest are the confidence layers, and the names and corresponding channels (num_output parameter) for these are:

- conv4_3_norm_mbox_conf -> 88 channels
- fc7_mbox_conf -> 66 channels
- conv6_2_mbox_conf -> 66 channels
- conv7_2_mbox_conf -> 66 channels
- conv8_2_mbox_conf -> 44 channels

Essentially, each of these correspond to a certain number of boxes*classes per grid cell.  The grid is different for this model than for the VGG16-SSD which uses a 300x300 versus the 480x360 input size used for this model.

By analyzing the prototxt file in netscope to determine the grid size for each of these layers.  We also know the number of classes is 11 for BDD, so we have the following:

- 60x45 grid -> 11 classes, 88 output channels
- 30x23 grid -> 11 classes, 66 output channels
- 15x12 grid ->  11 classes, 66 output channels
- 8x6 grid -> 11 classes, 66 output channels
- 4x3 grid -> 11 classes, 44 output channels

Back to slide 19, we can pretty quickly determine the number of boxes (again, based off slide 19 of the pdf):

number_of_boxes_per_cell = output_channels / classes

- 60x45 grid -> 88/11 = 8 boxes per cell
- 30x23 grid -> 66/11 = 6 boxes per cell
- 15x12 grid -> 66/11 = 6 boxes per cell
- 8x6 grid -> 66/11 = 6 boxes per cell
- 4x3 grid -> 44/11 = 4 boxes per cell

Those among you who are very observant might say "Hey, this model doesn't have the larger 1x1 grid that the SSD 300x300 model did"?  Well, ok, it has more small boxes, so we can infer it is probably better at detecting more smaller objects and few larger objects.  Neat, huh?

Well you might also wonder, why can't I do that with the other VGG16 SSD model?  Answer: you can - do whatever you want!

Ok, back to the task at hand - we were supposed to be modifying this model to work for pascal VOC which has 21 classes.  To do this, we can  recalculate based on the new number of classes how many output channels each of those layers is supposed to have.

- 60x45 grid -> 8 boxes per cell x 21 classes = 168 output channels
- 30x23 grid -> 6 boxes per cell x 21 classes = 126 output channels
- 15x12 grid ->  6 boxes per cell x 21 classes = 126 output channels
- 8x6 grid -> 6 boxes per cell x 21 classes = 126 output channels
- 4x3 grid -> 4 boxes per cell x 21 classes = 84 output channels

So now we can change the following layers:
- conv4_3_norm_mbox_conf -> change num_outputs to 168
- fc7_mbox_conf -> change num_outputs to 126
- conv6_2_mbox_conf -> change num_outputs to 126
- conv7_2_mbox_conf -> change num_outputs to 126
- conv8_2_mbox_conf -> change num_outputs to 84


Now, Caffe has a nuance that, when finetuning from an existing model, if you change a layer definition, you need to rename that layer, so we will change the names of these layers by doing a find/replace for each layer and just append `_new` to the end of the layer name (remember that top and bottom connections need to be changed as well).

When doing the find/replace, make sure you use the caps/exact match feature of your text editor so you do not inadvertently replace other phrases that contain this name.  

I also changed the batch size in the train_test_example.prototxt for Mobilenetv2 to 10 and I updated the solver to a iter_size of 3 so it has an effective training batch size of 30 (whereas the SSD 300x300 uses a batch size of 32).

Do not forget to open your solver.prototxt and make sure it points to your new model and has all the various flavors that you want.  

Finally, we are ready to train - issue the following command to your CUDA troops and supersize your electric bill:

```
cd /workspace/SSD/workspace/Mobilenetv2-SSD
caffe train \
  -solver solver.prototxt \
  -weights trainval.caffemodel \
  -gpu 0 2>&1 | tee SSD_train.log
```

If run into any unexpected errors, modify your solver to use the pre-modified prototoxt file that I have provided [SSD/workspace/Mobilenetv2-SSD/train_test_example.prototxt](SSD/workspace/Mobilenetv2-SSD/train_test_example.prototxt) or diff your prototxt against mine.

I have also included a [log file](SSD/workspace/Mobilenetv2-SSD/train.log) from my console output during the training process.


## 4. Evaluating the Floating-Point SSD Model on the Host

Once the models have been trained, you will see .caffemodels and .solverstates appear in the snapshots directory as defined in the ["solver.prototxt"](SSD/workspace/Mobilenetv2-SSD/solver.prototxt) for each of the models respectively

```
snapshot_prefix: "SSD/workspace/VGG16-SSD/snapshots"
or
snapshot_prefix: "SSD/workspace/Mobilenetv2-SSD/snapshots"
```
I trained the models for 120k iterations, each with batch sizes of 32 for VGG16-SSD and 30 for Mobilenetv2-SSD.  I took snapshots and ran test iterations every 10K iterations.  Below is a plot of the test mAP on 619 images from the test dataset during the training process:  

![picture](ref_files/pictures/SSD_Training_mAP.png)

After training, it is possible to score the network against the dataset and evaluate it against a single image from the test dataset.

I have included some example scripts that can be used to evaluate the floating-point network models.  These scripts can be found in the [SSD/evaluation folder](SSD/evaluation/) under the model subdirectory of interest.

**IMPORTANT:** Make sure to open the desired script for either model ([score.sh](SSD/evaluation/VGG16-SSD/score.sh) or [detect.sh](SSD/evaluation/VGG16-SSD/detect.sh)) and change the model name/path to the model that you want to evaluate.

The following instructions show how to run two different scripts.  

### 4.1: Scoring the Network
To score the network, change the directory to SSD/evaluation and to the desired model subdirectory. Enter the following command:
  ```
bash ./score_VGG16_SSD.sh
or
bash ./score_Mobilenetv2-SSD.sh
```

  I have included a [VGG16-SSD scoring log file](SSD/evaluation/VGG16-SSD/score.log) which shows an mAP of 0.766879 and a [Mobilenetv2-SSD scoring log file](SSD/evaluation/Mobilenetv2-SSD/score.log) which shows an mAP of 0.713212.

### 4.2: Single Image Evaluation
To run a single image evaluation, I have included the [detect.sh](SSD/evaluation/VGG16-SSD/detect.sh) script and deploy.prototxt files for both models that can be used to evaluate single images.  To change the image used for detection, modify the detect.list file to point to the desired images.

```
bash ./detect_VGG16_SSD.sh
or
bash ./detect_Mobilenetv2-SSD.sh
```

This process calls the ssd_detect executable, which is part of the Caffe install inside the Conda virtual environment.  The output image with detection overlays should appear inside the evaluation directory.

**NOTE:** If you changed the location or name of the snapshots during training or want to evaluate a different model, you need to modify the script above and specify the correct path to the model.

At this point, you have tested the trained SSD model on the host machine and can proceed to step 5.

## 5.0 Quantizing the SSD Network for DPU Implementation

### 5.1: Configuring the Files for Quantization and Compilation
1. I have included an example workspace for both models in [SSD/VAI](SSD/VAI) to show how the VAI tools may be invoked and the necessary modifications to the prototxt files for both quantization/compilation and testing the float and quantized model mAPs. Change directory to the VAI directory before proceeding to the next step.

2. In the VAI directory, there is a subdirectory for each model.  Inside each model directory there are several files:

  - "float.prototxt": used for quantizing/compiling the models for deployment on the target hardware

  - "quantize_and_compile.sh": a script that is used to perform both quantization and compilation (vai_q_caffe and vai_c_caffe) for deployment on the target hardware

  - "test_float_and_quantized.sh": a script that  tests both the floating point and quantized models and report out the mIOU for each


3. Open the "float.prototxt" that is included as an example in the VAI subfolders.

  The "float.prototxt" files should be mostly identical to your "train_test_example.prototxt" except for the following:

  - The input layer has changed to "ImageData"

  - Paths have been specified to the calibration data and text file so that they point to the correct locations if the steps in the tutorial were followed for populating the training data.

  - The "Multiboxloss" layer has been removed since it was only needed to compute loss for the training phase, and is also not supported for deployment in the DPU.


4. Copy your latest trained model from Caffe into the **SSD/VAI/VGG16-SSD or SSD/VAI/Mobilenev2-SSD** directory and rename it *float.caffemodel*.  This model should be located wherever the snapshot was saved from the training step.

5.Run the quantization tools using the following command (**Note:** You need to make the file executable first by running `chmod +x quantize_and_compile.sh`):

```
bash ./quantize_and_compile.sh
```

If you open the script, you can see the following contents, which indicate several things. First, you should make sure the GPUID environment variable is set correctly for your machine. If you have only one GPU, this should be '0'. Otherwise, change this to the index for the desired GPU for quantization.

Finally, the `vai_c_caffe` command is called, which compiles the floating-point model and produces a file called "vgg16_ssd.xmodel" under the compile directory.


```
#!/usr/bin/env bash
export GPUID=0
net=vgg16_ssd

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
          -keep_fix_neuron \
          -calib_iter 1000 \
          -output_dir ${model_dir} 2>&1 | tee ${model_dir}/quantize.txt

echo "Compiling network: ${net}"

vai_c_caffe   --prototxt=${model_dir}/deploy.prototxt \
        --caffemodel=${model_dir}/deploy.caffemodel \
        --output_dir=${output_dir} \
        --net_name=${net} \
        --options="{'mode':'normal'}" \
        --arch=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json 2>&1 | tee ${output_dir}/compile.txt
```      


At this point, an ELF file should have been created in the `Compile` directory which can be used in the final step which is to run the models on the ZCU102.  

### 5.2: Testing the Floating Point and Quantized Models
As mentioned in the previous section, files have been provided under the `SSD/VAI/VGG16-SSD` or `SSD/VAI/Mobilenetv2-SSD` directories, which can enable you to rapidly test the mAP of both the floating point model and the quantized model on the cityscapes validation dataset.  To complete this testing, perform the following steps:

Open one of the `test_float_and_quantized.sh` scripts.  The contents of this script are shown below.  You will only need to edit the GPUID to specify the correct GPU index for your tests.  **Note**: The log files are captured under the test_results subdirectory for both the floating-point and quantized results.

```
export GPUID=0
export WKDIR=`pwd`

cd $WKDIR
vai_q_caffe test -model float.prototxt -weights float.caffemodel -test_iter 4952 -gpu $GPUID 2>&1 | tee test_results/float_model_test.txt

#working directory
work_dir=$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}

vai_q_caffe quantize            \
          -model ${model_dir}/float.prototxt     \
          -weights ${model_dir}/float.caffemodel \
          -gpu $GPUID \
          -calib_iter 1000 \
          -keep_fix_neuron \
          -test_iter 4952 \
          -auto_test \
          -output_dir ${output_dir} 2>&1 | tee test_results/quantized_model_test.txt

```

Execute the script by running the following command.  This could take some time depending on the available GPU hardware and the model being run.  I have included example test results from a previous run under the associated model directories such as [SSD/VAI/VGG16-SSD/test_results](SSD/VAI/VGG16-SSD/test_results).  

```
bash ./test_float_and_quantized.sh
```


My VGG16-SSD model floating-point model achieved 0.766879 mAP, and the quantized model achieved 0.760131 mAP in the test dataset, so I only lost about 0.67% mAP during quantization.  For Mobilenetv2-SSD, my floating-point model achieved 0.713212 mAP, and the quantized model achieved 0.648701 mAP, which is a loss of about 6.45% mAP.

The loss here for Mobilenetv2-SSD is somewhat significant during quantization, and knowing that Mobilenev2 is a problematic model for quantization, a logical next step is to finetune the model.

### 5.3: Finetuning the Mobilenev2-SSD Model

As part of the quantization process, a quantize_train_test.prototxt and quantize_train_test.caffemodel are produced.  I have included a finetuning workspace in [SSD/VAI/Mobilenetv2-SSD/finetune](SSD/VAI/Mobilenetv2-SSD/finetune).  The steps to prepare this are as follows:

1. Copy the quantize_train_test.prototxt and quantize_train_test.caffemodel from the quantize directory to the finetune subdirectory
2. Open the quantize_train_test.prototxt in a text editor and change the input layers to match the input layers in my trainable train_test_example.prototxt files.  Also added the detection_eval layer back in at the end of the model and the Multiboxloss layer.  You can diff the modified file that I included versus the generated quantize_train_test.prototxt to see exactly what changed
3. Copy the solver.prototxt from the float model training workspace into the finetuning workspace and edited this file to reduce the base_lr, max_iterations, testing, and snapshot intervals.

To run the finetuning process, copy your quantize_train_test.caffemodel into the finetune directory, then execute the following command from within this directory:

```
bash ./run_finetune.sh
```

Using the solver that I provided, you should get snapshots every 1000 iterations.  

After 2000 iterations, the mAP is 0.701439, which is only a degradation of ~1%.  I used this snapshot to generate deploy the model.

After the finetuning step, the model can be transformed and then compiled using the following commands:

```
bash ./deploy.sh
bash ./compile.sh
```

The compiled dpu_mobilenetv2_ssd.xmodel file is populated under the finetune/compile directory and this is the model I have included in the [ZCU102](SSD/ZCU102/mobilenetv2_ssd) package.  

You can find more information on quantize finetuning (also called Quantize Aware Training or QAT) in [UG1414 pg86](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_4/ug1414-vitis-ai.pdf).

At this point, the quantized and floating point models have been fully verified on the host and you are ready to proceed to deploying the models to the target hardware.

## 6.0 Running the SSD Model on the ZCU102
The final step of this tutorial is to run the trained and compiled model on the ZCU102 board.

1. The first step that needs to be done is to copy your SSD, compiled .xmodel file into the related model subdirectory.

  **Note:** The directory structure contains the following:

  - [vgg16_ssd](SSD/ZCU102/vgg16_ssd) directory: This is the directory where you need to copy your compiled vgg16_ssd model executable and make sure it is named **dpu_vgg16_ssd.xmodel**.  The vgg16_ssd.prototxt file is inside this directory, which is needed by the Vitis AI Libraries to deploy the model.  

  - The [vgg16_ssd.prototxt](SSD/ZCU102/vgg16_ssd/vgg16_ssd.prototxt) file is a config prototxt file which contains pre-processing and post-processing parameters.  Many of the parameters should be obvious, but these can be extracted in one form or another from the model prototxt file.  I have found that a conf_threshold of 0.3 seems to be reasonable for both models on a per-class basis.  

  - The prior_box_param fields are likely the most confusing, but if you open the deploy.prototxt model file for each of the models, you should be able to search for the Priorbox layers and extract many of these parameters with the exception of layer_height and layer_width parameters.  

  - One way to extract these two parameters is to load the deploy.prototxt into [netscope](https://dgschwend.github.io/netscope/#/editor) and view the graph created by the tool (just paste the prototxt contents into the left field and hit shift+enter to draw the graph).  Then, browse in the graph until you find the priorbox layers and the input dimensions should also be included.

  - [mobilenetv2_ssd](SSD/ZCU102/mobilenetv2_ssd) directory: This is the directory where you need to copy your compiled mobilenetv2_ssd model executable and make sure it is named **mobilenetv2_ssd.xmodel**.  See the previous description for the dpu_vgg16_ssd folder as the contents are the same, just adjusted for the mobilenetv2_ssd model.  

   The [Vitis AI Libraries](https://github.com/Xilinx/Vitis-AI/tree/v1.4/tools/Vitis-AI-Library) are providing all of the software functionality including image preprocessing, DPU model execution, and model post-processing (NMS etc.).  

2. The next step is to copy the model directories to your ZCU102 board using an Ethernet connection.  Assuming you have set up your board image correctly, the model directories should be copied to the following directory on the target: "/home/root/Vitis-AI/demo/Vitis-AI-Library/samples/ssd/"

3. With a webcam connected to the board (I used a Logitech C920), the models can be tested using video with the following commands:

  ```
./test_video_ssd vgg16_ssd/vgg16_ssd.xmodel -t 4 0
```
or
  ```
./test_video_ssd mobilenetv2_ssd/mobilenetv2_ssd.xmodel -t 4 0
```
The Vitis AI library application expects that there will be a model config prototxt file with the same name as the XMODEL file in the directory. The -t argument is the number of threads to use to execute the model. On a ZCU102, 4-6 threads seem to work reasonably well. The last argument, '0', specifies using the webcam for the input video. You can also specify the filename of a video recording for this argument.

5. To run the performance test, you need several sample images. I copied 12 images to a folder from the VOC dataset. Then you need to create an image.list file, which lists the paths to the input images, similar to what is used by the host evaluation process. Then you can run the performance evaluation using the following command structure:

  ```
./test_performance_ssd vgg16_ssd/vgg16_ssd.xmodel test_performance_ssd.list -t 6 -s 30
```
  or
  ```
./test_performance_ssd mobilenetv2_ssd/mobilenetv2_ssd.xmodel test_performance_ssd.list -t 8 -s 30
```

The performance that I achieved for the models (taken with VAI 1.3) was as follows:
- VGG16-SSD -> 46.72 FPS using 6 threads for 30 seconds
- Mobilenev2-SSD -> 69.51 FPS using 6 threads for 30 seconds

I also timed these models on my GTX1080ti graphics card by changing directory to the model workspace, using the following command and also captured the power dissipation using nvidia-smi.

```
caffe time -model deploy.prototxt -gpu 0 -iterations 100
```

The results were as follows:
- VGG16-SSD using a batch size of 1 ->  60.56FPS @220W
- Mobilenetv2-SSD using a batch size of 1 ->  26.9FPS @108W
- VGG16-SSD using a batch size of 3 -> 95.96FPS @220W
- Mobilenetv2-SSD using a batch size of 3 -> 34.78FPS @108W

You can see the comparison in terms of performance, and more importantly for edge applications, performance or watt in the chart below.

![picture](ref_files/pictures/performance_results_batch.png)

Based on this, you can see we are achieving about 10x the capability in terms of performance/W using the ZCU102 with an INT8 model.

With all the marketing hype about TOPS, actual efficiency versus theoretical TOPs is also a relevant discussion.  Granted that the GTX1080ti maybe is not the best device to compare, but for the sake of illustrating a point, let's proceed:

The GTX1080ti has a theoretical performance of 11.3 TFLOPS while the 3x B4096 DPUs (from VAI 1.3) in the ZCU102 @281MHz have a theoretical performance of 3.45 TOPs.  If you look at the model compute for these models, it is:

- VGG16-SSD -> 61.356 GOPS/image
- Mobilenetv2-SSD -> 8.736 GOPs/image

Now, we can calculate the efficiency by model_FPS / (peak_accelerator_ops/model_ops).  I have performed the calculation and included it in the table below.
![picture](ref_files/pictures/efficiency.png)

You can see here that the GTX1080ti is pretty efficient at inference for the VGG16 model since this model uses nice 3x3 convolutions throughout, but the efficiency of the mobilenetv2 model is pretty terrible primarily because depth-wise convolution requires a lot of data transfer.  We are achieving about 18% efficiency on the DPU, which is actually pretty good for this model.  We can actually do significantly better when using a 2x DPU configuration (if you run the numbers from the [Vitis AI Model Zoo performance numbers](https://github.com/Xilinx/AI-Model-Zoo/tree/master) on the ZCU104 we achieve almost 30% efficiency).  Part of the reason for this is that the 3x DPUs are being throttled by the available PS DDDR memory bandwidth, and the other part is that the model zoo is only processing 11 classes instead of 21.  

##Summary
This tutorial has demonstrated how to train two different variants of SSD and deploy those models on the ZCU102 from start to finish including installing the Vitis AI tools, preparing the PASCAL VOC dataset, training the SSD network, evaluating the floating point model, quantizing and compiling that model, and finally deploying the application on the ZCU102 board.  

Maybe some homework for the ambitious reader would be to try to enhance the performance of the SSD model by pruning the model using the [Vitis-AI Optimizer](https://www.xilinx.com/support/documentation/user_guides/ug1333-ai-optimizer.pdf).  Contact your local FAE for an eval license (this is the only component of the Vitis AI package that requires one) and hopefully this has helped you in some fashion with your ML voyage!
