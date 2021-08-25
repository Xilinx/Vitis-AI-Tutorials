<table style="width:100%">
  <tr>
    <th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>How to freeze a Keras model for use with Vitis AI</h2>
</th>
  </tr>

</table>

### Current status

Tested with Vitis AI 1.0

## 1. Introduction

When using TensorFlow and Vitis-AI, we are required to supply a 'frozen graph' in binary protobuf format (usually has a '.pb' file extension). When we are working directly in TensorFlow, generating a .pb file is relatively easy - just save an inference graph and a TensorFlow checkpoint, then run the freeze_graph script that comes with TensorFlow.

Things are not so easy when you use the Keras standalone library as its native 'save' format is HDF5 or a mix of JSON and HDF5, so we need a way of translating those into binary protobuf format.

``
There are three ways to save a Keras model:

+ Method #1: Using HDF5 checkpoints during training.
+ Method #2: By saving the complete model in HDF5 format.
+ Method #3: By saving the network architecture in JSON format and the weights, biases and other parameters in HDF5 format.


:warning: TensorFlow checkpoints and Keras checkpoints are **_not_** the same!


Methods #1 and #2 will save a complete training model, including the network architecture, its current weights/biases, training operations such as loss and optimizer functions and their current state. This will allow training to be resumed from its current point if required.

Method #3 creates a JSON file which describes the network architecture. The weights are saved into a separate HDF5 file. The trainign information is not saved, so this format cannot be used for resuming training, but it normally produces much smaller files and so is favored when going to deployment.

## 2. Method #1: Keras Checkpoint to Vitis-AI

If we are starting from a Keras checkpoint, then we first access the underlying TensorFlow back-end using the `backend` API. We set the learning phase to '0' to indicate to layers such as dropout or batch normalization that we are no longer training. We then load the checkpoint into a new model using `load_model`.

```python
# set learning phase for no training
backend.set_learning_phase(0)

# load weights & architecture into new model
loaded_model = load_model('keras_chkpt.h5')
```

Now we can write out the TensorFlow compatible checkpoint and inference graph which will be used later with the freeze_graph.py script to create the frozen model:

```python
# make list of output node names
output_names=[out.op.name for out in loaded_model.outputs]

# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
tf_session = backend.get_session()

# get the tensorflow session graph
input_graph_def = tf_session.graph.as_graph_def()

# write out tensorflow checkpoint & inference graph for use with freeze_graph script
save_path = saver.save(tf_session, 'tf_chkpt.ckpt')
tf.train.write_graph(input_graph_def, 'checkpoint_dir', 'tf_infer_graph.pb', as_text=False)
```

## 3. Method #2: Keras model to Vitis-AI

If the Keras model has been saved as a complete model using `save_model` like this:

```python
# save weights, model architecture & optimizer to an HDF5 format file
model.save('k_complete_model.h5')
```

..then it will be an HDF5 file that is identical to the one produced when we create a Keras checkpoint so just follow the same procedure as Method #1 above.


## 4. Method #3: JSON & HDF5 to Vitis-AI

If the Keras model has been saved as a JSON file for the architecture and an HDF5 file for the weights and biases using `save_model` like this:

```python
# save just the weights (no architecture) to an HDF5 format file
model.save_weights('k_model_weights.h5')

# save just the architecture (no weights) to a JSON file
with open('k_model_architecture.json', 'w') as f:
    f.write(model.to_json())
```

..then the model needs to recreated by first reading the JSON file and then loading the model with the weights stored in the HDF5 file:

```python
# set learning phase for no training
backend.set_learning_phase(0)

# load json and create model
json_file = open('k_model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('k_model_weights.h5')
```

Now we can write out the TensorFlow compatible checkpoint and inference graph which will be used later with the freeze_graph.py script to create the frozen model:

```python
# make list of output node names
output_names=[out.op.name for out in loaded_model.outputs]

# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
tf_session = backend.get_session()

# get the tensorflow session graph
input_graph_def = tf_session.graph.as_graph_def()

# write out tensorflow checkpoint & inference graph for use with freeze_graph script
save_path = saver.save(tf_session, 'tf_chkpt.ckpt')
tf.train.write_graph(input_graph_def, 'checkpoint_dir', 'tf_infer_graph.pb', as_text=False)
```


## 5. Prerequistes to run this tutorial

  + An Ubuntu host machine configured as per the requirements defined in Table 1 of the <a href="https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf">Vitis AI User Guide (UG1414)</a>.

  + The host machine should also have:
    + The vitis-ai-docker-runtime which is the runtime docker image for DPU-v2 development.
    + Either the vitis-ai-docker-tools GPU image or the vitis-ai-docker-tools CPU image.

See the <a href="https://github.com/Xilinx/Vitis-AI">Vitis-AI GitHub README.md</a> for details.

  + The host machine will need access to the internet for downloading files.
  + This tutorial assumes the user is familiar with Vitis-AI, Python3, TensorFlow and has some knowledge of machine learning principles.


## 6. Files in this repository

This repository contains examples for all three methods.

+ `train_save.py` will train a simple network to classify the CIFAR-10 dataset. It saves the trained model using the three methods outlined in this app note. Note that it only runs for a few epochs and the accuracy will be quite low.
+ `keras_2_tf.py` will create TensorFlow checkpoints and inference graphs from the saved Keras models.
+ `image_input_fn.py` contains the pre-processing function for the images - used during the calibration phase of quantization.
+ `generate_images.py` creates image files from the CIFAR10 dataset.
+ `run_all.sh` will run the complete flow.
+ `tools_docker.sh` contains an example of how to launch the Vitis-AI tools docker container.


## 7. Instructions

1. Clone or download this repository, open a terminal and 'cd' into the repo folder just created.

2. Start the Vitis-AI tools docker, the command will look something like this:

     `/home/mharvey/Vitis-AI/docker_run.sh xilinx/vitis-ai:tools-1.0.0-gpu`

3. Extra packages need to be installed into the docker container, run the following commands:

```shell
sudo su
conda activate vitis-ai-tensorflow
pip install keras==2.2.5
conda install -y pillow
exit
conda activate vitis-ai-tensorflow
```

:warning: Do not install Keras with the `conda install` command as this will overwrite the Vitis-AI version of TensorFlow.



4. To run the complete flow: `source ./run_all.sh`


When the flow is finished, you should find three folders; method1, method2 and method3 which contain all the Keras models, TensorFlow checkpoints and graphs and the .elf file.
