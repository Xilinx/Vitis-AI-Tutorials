<table class="sphinxhide">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI 1.4 - Machine Learning Tutorials</h1>
    <a href="https://www.xilinx.com/products/design-tools/vitis.html">See Vitis Development Environment on xilinx.com</br></a>
    <a href="https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html">See Vitis AI Development Environment on xilinx.com</a>
    </td>
 </tr>
</table>

# Alveo U250 TF2 Classification Tutorial

This tutorial demonstrates image classification using the Alveo&trade; U250 with Vitis AI&trade; 1.4 and the Tensorflow 2.x framework.

The tutorial is organized into the following sections:

1. Prepping the dataset

2. Training the Model

3. Alveo Installation and Setup

4. Model Quantization and Compilation

5. Deployment on the U250

**Note:** This tutorial provides a pre-trained snapshot, so you can optionally skip step 2.

The pre-trained model has been provided in the [scripts/train_dir](./scripts/train_dir) folder.  It's a multi-part ZIP file and can be extracted on Linux or using 7-zip on windows.  You will need to unzip it to this directory if you wish to use it going forward.  On Linux, simply right click part *.001 and select extract here.

## Important: Environment Setup
It is important to first set up the Vitis AI docker on your host machine.  You can find instructions to do this on the Vitis AI landing page:

https://github.com/Xilinx/Vitis-AI/tree/v1.4

If training the model, a GPU should be used to accelerate the training process.  This tutorial uses the GTX 1080ti and can train the model overnight.

After setting up the environment, these tutorial files can be copied into the Vitis AI folder and the instructions below will walk through the remaining steps.

### 1) Prepping the Dataset
This tutorial is based on the Kaggle fruit-360 dataset. To train the model, we first need to download and prep the data into TF Records.

1.1) Download the dataset: https://www.kaggle.com/moltean/fruits

1.2) Extract the zipped archive.  Within the archive are several folders, the two important directories are named **Test** and **Training**.  

**Note:** Go ahead and move these two directories into a new folder called `images`.

Notice that the test and training directories each contain 131 sub-directories with the name of the class of image for each folder name.  In the case of classification networks, this is often how data is organized.

Since TensorFlow cannot use these folders directly as part of the training process, the data needs to be processed into a recognizable format which is the TF record.  You can find some detail on the TF Record format here for reference: https://www.tensorflow.org/tutorials/load_data/tfrecord.

The TF record is a format used by the TensorFlow training environment to feed data or labels to the model during training and validation.  Small datasets may be able to take advantage of other mechanisms to feed data, but for larger datasets that will not fit into the available GPU memory, TF records are the default mechanism for training.

1.3) This tutorial provides a script called ["tf_record.py"](./scripts/tf_record.py) to process the images into TF records.  This script is taken from the [Tensorflow Imagenet to GCS script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py).  

The original script from TensorFlow is modified for the fruit dataset in a number of ways, including using the folders for the validation labels instead of the JSON file for ImageNet, removing the portion that does uploading to GCS, as well as removing a few of the other ImageNet specific attributes.

1.3a) Open the `tf_record.py` script and set the following two variables for your system

- tf_record_dir='/path/to/tf_records_output'

- raw_data_dir='/path/to/fruits-360/Images'

1.3b) Note the training shard and validation shard numbers. These are how many TF shards will be created for the respective data.  The recommendation from TF is to have these shards be about 100MB each per the following [performance guide](https://docs.w3cub.com/tensorflow~guide/performance/performance_guide).

The size of the shard should correlate with the size of the input data divided by the number of shards created.

If required, walk through the script to understand how the data gets packed into shards.  One key point is that labels are assigned an int64 value ranging from 1-total number of subfolders.  The order in which they are assigned values depends
on sorted order of following python code:

labels = {v: k + 1 for k, v in enumerate(
    sorted(set(validation_synsets + training_synsets)))}

If required, it is possible to also do preprocessing, normalization, and/or resizing as part of the TF record generation step.

1.3c) Run the script either using a python debugger or `python3 tf_record.py`

After this step you should see messages about the shards being written to the previously defined location.  This process takes around 10 minutes on a workstation with 2x GTX 1080ti GPUs.

1.4) **Optional Step:** One other potential step in dataset preparation is to prepare the validation data as raw images for use with Vitis AI (for quantization and evaluation).  The quantization, validation, and testing procedures *can use the TF validation records for these processes*.  In fact, the following tutorial by default uses the TF Validation records for all of these processes, however, by default, the scripts provided with the Vitis AI model zoo ResNet50 model expects raw images from a directory along with a labels text file that lists the image name and corresponding class id on each line.

For more information, download the [Resnet50 pretrained model](https://www.xilinx.com/bin/public/openDownload?filename=tf2_resnet50_imagenet_224_224_7.76G_1.4.zip) from the Vitis AI Model Zoo and take a look at the following files:

- tf2_resnet50_imagenet_224_224_7.76G_1.3/readme.md

- tf2_resnet50_imagenet_224_224_7.76G_1.3/code/com/train_eval_h5.py

- tf2_resnet50_imagenet_224_224_7.76G_1.3/code/com/dataset.py

In the readme.md file, you can see a description of the ground truth starting on line 41.  Essentially it takes the form of a list of images followed by a space and then an integer string which is the class ID:

ILSVRC2012_val_00000001.JPEG 65

The code that executes this is in `train_eval_h5.py` on lines 129-131.  The `get_images_infor_from_file` function is used to grab the images from a directory and the labels from a file and put them into python lists.  

The python lists for the images and labels are then fed into the `ImagenetSequence` function which does the ImageNet preprocessing and puts the preprocessed values and labels into np.arrays.

Both of these functions are defined in the [scripts/dataset.py](./scripts/dataset.py) file.  

For example on how to process the validation images into the format compatible with the quantization, the [gen_validation_set.py](./scripts/gen_validation_set.py) is provided.  This script loops through all of the validation folders and copies the images into a single output folder while renaming them with the label name as part of the filename.  It also parses the various classes associated with the dataset and writes the file name with an integer value that represents the class ID into an output text file.

How to use this script:
- Open this script and set the input path variable to the location of your raw validation images

- Set the output path to where you want the validation set to be written as well as the output labels file

- Run the script in a debugger or with `python3 gen_validation_set.py`

### 2) Training the Model
To train the model, I have included a modified version of the following three key files from the Vitis AI Model Zoo:

- [train_eval_h5.py](./scripts/train_eval_h5.py)
- [dataset.py](./scripts/dataset.py)
- [imagenet_preprocessing.py](./scripts/imagenet_preprocessing.py)

There have been modifications made to each of the scripts to change from the 224x224 input size and ImageNet preprocessing to the fruits-360 dataset.  We recommend comparing these files to the originals to understand what changed for the new dataset.

I have also included a couple scripts for training the model.  The first is `train_from_scratch.sh`.  This script will set a boolean flag called `createnewmodel` within the `train_eval_h5.py` which will make a few modifications to the default ResNet50 ImageNet model:

- The new model uses input data that is 100x100 instead of the default 224x224 used by Resnet50
- The new model should classify 131 different fruits instead of the 1000 classes for ImageNet

The following code update to `train_eval_h5.py` will pull down the base Keras ResNet50 model with the default ImageNet weights and will change the input shape to 100x100 instead of 224x224.  

```
elif FLAGS.createnewmodel:
      #for training the model from scratch use the following:
      basemodel = ResNet50(weights="imagenet", include_top=True,input_tensor=Input(shape=(100, 100, 3)))
      base_output = basemodel.layers[175].output
      new_output = tf.keras.layers.Dense(activation="softmax", units=131)(base_output)
      model = tf.keras.models.Model(inputs=basemodel.inputs, outputs=new_output)

      print(model.summary())
```

As can be seen here, we are creating a tensor called `base_output` which is the output of layer 175 from the ResNet50 model (the layer indices can be seen when debugging the code with a python debugger). This is the last layer before the fully connected output of the model.  

We then define a `new_output` which is a set of dense layers using a softmax activation with 131 different classes and we connect its input to the `base_output` where we bi-furcated the original model.  

We then use the final line of code to create a new model that uses the basemodel input and the new output.

2.1) In order to train the model for the first time, use the following command:

- `./train_from_scratch.sh`

During training, you should see a series of messages with corresponding loss/accuracy produced in the console such as the following:

227/226 [==============================] - 10s 44ms/step - loss: 0.0082 - sparse_categorical_accuracy: 0.9975 - sparse_top_k_categorical_accuracy: 1.0000

Snapshots will be produced in the train_dir which correspond to the iterative epoch.

2.2) If training is stopped at some point and you would like to resume training, you can use the following command:

- `./resume_training.sh`

**Note:** You will first need to edit the `resume_training.sh` script to point to the desired checkpoint to start from.

### 3) Alveo Installation and Setup
The Alveo U250 card first needs to be installed into an x86 server chassis.  Details on the physical installation and server requirements can be found in the [Alveo U250 User Guide](https://www.xilinx.com/support/documentation/boards_and_kits/accelerator-cards/1_9/ug1301-getting-started-guide-alveo-accelerator-cards.pdf).

3.1) Host Environment Setup

Once the Alveo is installed in the server, a script provided in the Vitis AI repository can be used to install the required components on the host machine.  The instructions and script are found here and the key step to execute is the installation on the host using the `install.sh` script: https://github.com/Xilinx/Vitis-AI/tree/v1.4/setup/alveo.

If you are using this script to install XRT and the shells, it is recommended to first remove any existing XRT environment.  This should also remove any shells installed in the system and provide a clean starting point to install the Alveo software components.

A cold reboot of the machine is typically required at this point before proceeding to the next step because a new shell will have been flashed to the Alveo U250.
After installing on the shell and rebooting the machine, the output of 'sudo /opt/xilinx/xrt/bin/xbmgmt flash --scan' should yield the following:

---------------------------------------------------------------------
Deprecation Warning:
    The given legacy sub-command and/or option has been deprecated
    to be obsoleted in the next release.

    Further information regarding the legacy deprecated sub-commands
    and options along with their mappings to the next generation
    sub-commands and options can be found on the Xilinx Runtime (XRT)
    documentation page:

    https://xilinx.github.io/XRT/master/html/xbtools_map.html

    Please update your scripts and tools to use the next generation
    sub-commands and options.
---------------------------------------------------------------------
Card [0000:03:00.0]

    Card type: u250
    Flash type: SPI
    Flashable partition running on FPGA:
        xilinx_u250_gen3x16_base_3,[ID=0x48810c9d17860ef5],[SC=4.6.11]
    Flashable partitions installed in system:
        xilinx_u250_gen3x16_base_3,[ID=0x48810c9d17860ef5],[SC=4.6.11]


3.2) Docker Environment Setup

  - Before launching the docker environment it is necessary to manually program the shell once outside of the docker per the following Answer Record: https://www.xilinx.com/support/answers/75975.html.

    I have provided a script that should do this but the user is responsible to verify the card bdf matches what is in the provided script (Card [0000:03:00.0]).  

    To run this operation, open a command prompt and cd to the [scripts](./scripts) directory and run `./program_u250_shell.sh`.

  - The next step is to launch the Vitis AI docker environment and activate the vitis-ai-tensorflow2 Conda environment.

  - With this environment activate, source the [scripts/setup_dpu.sh](./scripts/setup_dpu.sh) script using `source setup_dpu.sh`.
    This will configure the various environment variables required for tools and deployment.

4. Model Quantization and Compilation
Scripts have been provided for quantization and compilation of the model.

  - Quantization can be run with [quantize.sh](./scripts/quantize.sh) using `./quantize.sh`

    Before running this script, open it with a text editor and set the paths to your dataset which will be used for calibration.

    Quantization can take a few minutes with a GPU enabled machine and upon completion, will populate the quantized.h5 model in the [scripts/tf2_resnet50/vai_q_output](./scripts/tf2_resnet50/vai_q_output) directory.

    A script has also been provided to evaluate the quantized model on the host machine if desired [scripts/eval_quantize_graph.sh](./scripts/eval_quantize_graph.sh)

  - Once the model has been quantized, it can be compiled with [compile.sh](./scripts/compile.sh) using `./compile.sh`

    **Note:** This script sets the input for a batch size of '4' which is what the DPUCADF8H implements.  This script takes a few minutes, but upon completion, will populate the compiled XMODEL file into the [scripts/tf2_resnet50/u250/](.scripts/tf2_resnet50/u250/) directory.


5. Deployment on the U250
Once the compiled model is available, the software for deployment can be compiled.  The software is provided under the [deployment](./deployment) directory.  

  - To compile the software, cd to [deployment](./deployment) and run `./build.sh`

  - Next copy the desired sample images from the dataset into a folder (e.g. imageDir100)

  - To execute the model, run `./resnet_example ../scripts/tf2_resnet50/u250/resnet50_tf2.xmodel ./imageDir100/`

  The output classes can be compared to the image name for correctness:
```
  Image : Kohlrabi220_100.jpg
    top[0] prob = 0.999035  name = Kohlrabi
    top[1] prob = 0.000335  name = Apple Golden 3
    top[2] prob = 0.000203  name = Cauliflower
    top[3] prob = 0.000123  name = Apple Red Yellow 2   
    top[4] prob = 0.000075  name = Tomato not Ripened
```
