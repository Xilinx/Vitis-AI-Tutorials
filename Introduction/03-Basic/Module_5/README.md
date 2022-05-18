# Work with Networks in Vitis AI Docker

This module introduces how to use networks and scripts from Vitis AI model zoo to do the following tasks in Vitis AI docker:

- Prepare dataset
- Evaluate pretrained float model and quantized model, including test images and test accuracy
- Retrain the model
- Use Vitis AI toolchain to generate files for deployment

This tutorial uses Caffe RefineDet network as example. More examples with other frameworks will be added in the future.

## Prerequisite

- Active internet connection
- Vitis AI docker working
- Conda environment 'vitis-ai-caffe' is enabled by `conda activate vitis-ai-caffe`
- ZCU104 board is up and running with Vitis AI release image

## Download Network Package from Vitis AI Model Zoo

[Vitis AI model zoo](https://github.com/Xilinx/Vitis-AI/tree/master/models/AI-Model-Zoo) provides a variety of networks pre-trained in Caffe/TensorFlow or DarkNet, covering different applications, such as image classification, ADAS/AD, and video surveillance. Several mainstream networks are further optimized using [Xilinx Optimizer](https://www.xilinx.com/member/ai_optimizer.html) to achieve much higher performance.

In each network package, float models and quantized models are provided together with training and evaluation scripts. We use Caffe [RefineDet](https://arxiv.org/abs/1711.06897) (Model No. 10-13 in the Model Zoo) to build a people detection application.

RefineDet is a one-stage objection detection network that could achieve better accuracy than two-stage networks before
publication but still maintains comparable speed performance of one-stage networks.

RefineDet can be considered an improved version of SSD with the addition of an anchor refinement module (ARM), object detection module (OBM), and transfer connection block (TCB) to improve accuracy.

The detailed introduction about the network itself is beyond the scope of this module. For that, please refer to the original paper.

### Get RefineDet Model Package
After cloning the repository, go to the module 5 folder(Vitis-In-Depth-Tutorial/tree/v1.3/Machine_Learning/Introduction/03-Basic/Module_5) and run `sh get_network.sh` to download RefineDet network package. The package is trained with a people subset selected from COCO2014 and optimized to reduce 96% of the computation complexity (GOP).

If no error occurs, the model folder looks as follows:
```
cf_refinedet_coco_360_480_0.96_5.08G_1.3/
|-- code
|   |-- gen_data
|   |   |-- convert_coco2voc_like.py
|   |   |-- create_data.py
|   |   |-- gen_quantize_data_list.py
|   |   `-- get_dataset.sh
|   |-- test
|   |   |-- demo.py
|   |   |-- evaluation.py
|   |   |-- evaluation_py2.py
|   |   |-- quantized_test.sh
|   |   |-- quantize.sh
|   |   |-- test.py
|   |   `-- test.sh
|   `-- train
|       |-- solver.prototxt
|       `-- train.sh
|-- data
|   |-- demo_list.txt
|   |-- labelmap.prototxt
|   `-- test_name_size.txt
|-- float
|   |-- quantize.prototxt
|   |-- test.prototxt
|   |-- trainval.caffemodel
|   `-- trainval.prototxt
|-- quantized
|   |-- deploy.caffemodel
|   |-- deploy.prototxt
|   |-- DNNC
|   |   `-- deploy.prototxt
|   |-- quantized_test.prototxt
|   |-- quantized_train_test.caffemodel
|   `-- quantized_train_test.prototxt
`-- readme.md
```

Go to the `cf_refinedet_coco_360_480_0.96_5.08G_1.3/` folder and follow the steps described below:


## Prepare Customized COCO Dataset

COCO is large-scale object detection, segmentation, and captioning dataset including 330 thousand images (>200 thousand labeled) in 91 object types used widely in machine learning and artificial intelligence research. For detailed information about the dataset, go to [COCO official website](https://cocodataset.org/#home).

To retrain the network, evaluate the map before directly deploying the network onboard. The COCO dataset used by this network needs to be prepared with the necessary process.

The `code/gen_data` folder contains the related scripts:

```
gen_data/
|-- convert_coco2voc_like.py
|-- create_data.py
|-- gen_quantize_data_list.py
`-- get_dataset.sh
```

Use `sh get_dataset.sh` to download and generate the dataset which only contains **one class** *person* seleted from COCO2014. If you already have COCO2014 in the hard drive, please comment the download code in `get_dataset.sh` and put COCO2014 files into the `cf_refinedet_coco_480_360_0.96_5.08G/data` folder for simplicity. The expected file structure is shown as below:

```
coco2014/
      |->train2014/
      |->val2014/
      |->instances_train2014.json
      |->instances_val2014.json
```

If the script finishes without error, the generated LMBD files and images will have following structure:
```
data/
├── coco2014
│   ├── Annotations
│   ├── Images
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── train2014.txt
│   ├── train_image2anno.txt
│   ├── val2014.txt
│   └── val_image2anno.txt
├── coco2014_lmdb
│   ├── train2014_lmdb
│   └── val2014_lmdb
├── demo_list.txt
├── labelmap.prototxt
├── link_480_360
│   ├── train2014_lmdb -> ../../data/coco2014_lmdb/train2014_lmdb
│   └── val2014_lmdb -> ../../data/coco2014_lmdb/val2014_lmdb
└── test_name_size.txt
```
If the script finishes with out error, the md5sum values of train and val MDB files are as below:

|                 File Name                  |           Md5sum Value           |
| :----------------------------------------: | :------------------------------: |
| data/coco2014_lmdb/train2014_lmdb/data.mdb | 39891c36467f22498a98cbd431e43657 |
|  data/coco2014_lmdb/val2014_lmdb/data.mdb  | e99d3ad2ce3d5ffc864fa8d41b01d4b0 |




## From Official RefineDet to Model Zoo Version

The Model Zoo package provides Xilinx version RefineDet with modifications on the model structure to make it compatible with Vitis AI and network pruning optimization by Vitis AI optimizer.

The **[Journey from Original RefineDet](/Machine_Learning/Introduction/03-Basic/Module_5/network_training.md)** explains the detailed steps to start from original model and arrive at the baseline version (not optimized, 125GOPs) in the Model Zoo package. Optimized, 5GOPs model used in this example, but 25GOPs and 10GOPs models are also provided in the model zoo). The pretrained baseline model is provided here and could be used in the following parts.

In the Advanced section, there are separate modules to document the workflow of the AI optimizer and the steps to prune the model from baseline to optimized version.

## Work with Vitis AI Toolchain

Vitis AI toolchain quantifies float 32bit network model into INT8 model with optimization and then compiles it into instructions that run efficiently on the DPU in FPGA. The toolchain could support Caffe, DarkNet, and TensorFlow framework.

[Vitis AI User Documentation
](https://www.xilinx.com/html_docs/vitis_ai/1_3/zmw1606771874842.html) provides a comprehensive introduction of the Vitis AI toolchain. This tutorial only uses RefineDet as an example to demonstrate Caffe workflow. The following steps are based on the network package from Model Zoo (cf_refinedet_coco_480_360_0.96_5.08G), but it could be easily used with the baseline model with some changes on file paths in the scripts, prototxt, and commands.

### <span id = "tool">Evaluate Model</span>

In the network package, both float and quantized models are provided together with an evaluation script to run single images or test mAP over the whole validation dataset. These scripts work with the baseline model as well.

### Run single image

Go to the `cf_refinedet_coco_360_480_0.96_5.08G_1.3/code/test` folder.

Modify the image path on line 19 of `demo.py` per the need and run the inference on a single image with the `python demo.py` command. The output image is stored in `output/res_output.jpg`.

![Inference result](./res_output.jpg)

### Test mAP

The network package provides scripts to verify the mAP of the pretrained network (both float model and quantized model). To use the scripts, ensure the evaluation dataset are correctly prepared and file paths in 'parse_args' function (line 54 to 85 in `test.py`) are modified according to your environment.

By default, the script evaluates float model (trainval.caffemodel and test.prototxt) stored in the `cf_refinedet_coco_360_480_0.96_5.08G_1.3/float`folder.

Use the `python test.py` command to run the evaluation. The evaluation takes around 30 to 60 minutes depending on your CPU or GPU capability. If no error occurs, the mAP result displays as follows:
```
evaluate 21634 images
person AP: 0.6120358325758232
mAP: 0.6120358325758232
```
It is also easy to evaluate quantized model in the same way. Change the prototxt file and caffemodel file in the script into `quantize_test.prototxt` and `quantize_train_test.caffemodel`. After evaluation is complete, the result is displayed.
```
evaluate 21634 images
person AP: 0.6112254774886302
mAP: 0.611225477489
```

The float baseline model could also be evaluated using the same script. Change the prototxt file and caffemodel file into `refinedet_baseline.prototxt` and `refinedet_baseline.caffemodel` in the `cf_refinedet_coco_baseline_125G` folder.


## Deploy Model

### Quantize Float Model

The ready-to-use quantized models are already provided in the package. If you want to try the quantization steps by yourself, refer to [Vitis AI User Documentation
](https://www.xilinx.com/html_docs/vitis_ai/1_3/uim1570695919827.html).

For Caffe example, the `vai_q_caffe` is used to execute quantization with following commands:

```
vai_q_caffe quantize -model trainval.prototxt \
                     -weights trainval.caffemodel \
                     -calib_iter 20 \
                     -auto_test  \
                     -test_iter 100 \
                     -output_dir quantization \
                     -gpu 0 \
                     -ignore_layers arm_priorbox \
                     -keep_fixed_neuron
```

**Note:** The 'arm_priorbox' (which concats all previous priorbox layers together) needs to be ignored because it does not run on DPU and affects quantization bounding box results. Generated quantized model is stored in the folder specified by `-output_dir`, which in this case is `quantization`.

**Note:**: From Vitis AI 1.3, the Xmodel is also used on DPUCZDX8G, so 'keep_fixed_neuron' flag is needed when quantizing caffe network.  

```
I0112 02:46:12.526528  3810 net_test.cpp:308] Test Results:
I0112 02:46:12.526546  3810 net_test.cpp:309]     Test net output #0: detection_eval = 0.611557
I0112 02:46:12.527979  3810 net_test.cpp:450] Test Done!
I0112 02:46:13.067780  3810 vai_q.cpp:360] Start Deploy
I0112 02:46:13.100937  3810 vai_q.cpp:368] Deploy Done!
--------------------------------------------------
Output Quantized Train&Test Model:   "quantization/quantize_train_test.prototxt"
Output Quantized Train&Test Weights: "quantization/quantize_train_test.caffemodel"
Output Deploy Weights: "quantization/deploy.caffemodel"
Output Deploy Model:   "quantization/deploy.prototxt"
```

Using the same step and command, we could  generate a quantized model for baseline model and use it in the next step compilation. Ensure the right models are used per your needs and output files are saved properly.

### Compile Quantized Model

To deploy the quantized model on board, we use `vai_c_caffe` to generate the Xmodel file by compiling deploy.prototxt and deploy.caffemodel. Here we could use files in the `cf_refinedet_coco_480_360_0.96_5.08G/quantized` (provided in the network package) folder or `quantization` (generated by you).

Before compilation, in any deploy.prototxt, ensure all the layers after the layer block `odm_conf` are deleted as they are not needed for deployment and compilation.

Run the below command in the quantized model folder to compile the model into an ELF file  targeting DPU on ZCU104 released board image.

```
vai_c_caffe --prototxt    deploy.prototxt \
            --caffemodel  deploy.caffemodel \
            --arch        /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
            --output_dir  board_file_zcu104_5G \
            --net_name    refinedet_5G \
```

**Note:** You could also specify a preferred path for prototxt, caffemodel, and output_dir.

During compilation, the compiler gives the following output that displays information: the kernel name, input tensor names, input tensor shapes, output tensor names, and output tensor shapes. Save the information `refinedet_kernel.info` in `board_file_zcu104_5G` using the `save_kernel` option.

```
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] Namespace(inputs_shape=None, layout='NCHW', model_files=['deploy.caffemodel'], model_type='caffe', out_filename='board_file_zcu104_5G/refinedet_5G_org.xmodel', proto='deploy.prototxt')
[INFO] caffe model: deploy.caffemodel
[INFO] caffe model: deploy.prototxt
[INFO] parse raw model     :100%|███████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [00:00<00:00, 2777.05it/s]
[INFO] infer shape (NCHW)  :100%|███████████████████████████████████████████████████████████████████████████████████████████████| 168/168 [00:00<00:00, 23427.19it/s]
[INFO] infer shape (NHWC)  :100%|███████████████████████████████████████████████████████████████████████████████████████████████| 168/168 [00:00<00:00, 20118.86it/s]
[INFO] generate xmodel     :100%|███████████████████████████████████████████████████████████████████████████████████████████████| 168/168 [00:00<00:00, 12205.84it/s]
[INFO] generate xmodel: /workspace/Vitis-In-Depth-Tutorial/Machine_Learning/Introduction/03-Basic/Module_5/cf_refinedet_coco_360_480_0.96_5.08G_1.3/float/quantization/board_file_zcu104_5G/refinedet_5G_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/vitis-ai-user/log/xcompiler-20210112-034036-4505"
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Graph name: deploy, with op num: 360
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 7, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/Vitis-In-Depth-Tutorial/Machine_Learning/Introduction/03-Basic/Module_5/cf_refinedet_coco_360_480_0.96_5.08G_1.3/float/quantization/board_file_zcu104_5G/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/Vitis-In-Depth-Tutorial/Machine_Learning/Introduction/03-Basic/Module_5/cf_refinedet_coco_360_480_0.96_5.08G_1.3/float/quantization/board_file_zcu104_5G/refinedet_5G.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is cfa5b17ccea8932186ae1153edb0ef7d, and been saved to "/workspace/Vitis-In-Depth-Tutorial/Machine_Learning/Introduction/03-Basic/Module_5/cf_refinedet_coco_360_480_0.96_5.08G_1.3/float/quantization/board_file_zcu104_5G/md5sum.txt"

```
After compilation completes, The ELF file, kernel info file, and kernel graph files are generated and saved.

```
board_file_zcu104_5G/
├── md5sum.txt
├── meta.json
├── refinedet_5G_org.xmodel
└── refinedet_5G.xmodel
```

The 'refinedet_5G_org.xmodel' is the float Xmodel generated by the compiler and 'refinedet_5G.xmodel' is the compiled Xmodel for DPU execution.

Using the same step and command in the folder containing the deploy model for the baseline model, we could also generate the ELF file for the baseline model

```
board_file_zcu104_125G/
├── md5sum.txt
├── meta.json
├── refinedet_125G_org.xmodel
└── refinedet_125G.xmodel
```


## Test on Board

Before proceeding with this step, please ensure:

- The zcu104 board is running with Vitis AI 1.3 release board image and Vitis AI Library/VART are installed correctly based on the previous modules

- Host and zcu104 board are in same network. In our example, board IP (TARGET_IP) is 10.176.18.133

### Transfer Compiler Output to Board

Tranfer the compiler output `board_file_zcu104_5G` and `board_file_zcu104_125G` to board and place it under with `scp` command.

`scp -r /PATH_TO/board_file_zcu104_5G root@10.176.18.133:~/`

`scp -r /PATH_TO/board_file_zcu104_125G root@10.176.18.133:~/`


### Connect to Board

`ssh root@10.176.18.133`

Enter password root to connect to the board.

Go to refinedet example application folder

`cd /home/root/Vitis_AI/examples/Vitis-AI-Library/samples/refinedet`

### Copy the Xmodel files to Vitis AI Model Folder

The Vitis AI library provides pre-built models, but we will copy our compiled Xmodel to the model folders with the names required by the Vitis AI library.

`cp board_file_zcu104_5G/refinedet_5G.xmodel /usr/share/vitis_ai_library/models/refinedet_pruned_0_96/refinedet_pruned_0_96.xmodel`

`cp board_file_zcu104_125G/refinedet_125G.xmodel /usr/share/vitis_ai_library/models/refinedet_baseline/refinedet_baseline.xmodel`


### Run the application

- Performance test

  For example, we could test performance in 10s with 4 threads with the following command:

  `./test_performance_refinedet refinedet_pruned_0_96 test_performance_refinedet.list -t 4 -s 10`

  ```
  root@xilinx-zcu104-2020_2:~/Vitis_AI/examples/Vitis-AI-Library/samples/refinedet# ./test_performance_refinedet refinedet_pruned_0_96 test_performance_refinedet.list -t 4 -s 10
  WARNING: Logging before InitGoogleLogging() is written to STDERR
  I0110 18:53:33.324429  8773 benchmark.hpp:176] writing report to <STDOUT>
  I0110 18:53:33.451941  8773 benchmark.hpp:203] waiting for 0/10 seconds, 4 threads running
  I0110 18:53:43.452201  8773 benchmark.hpp:211] waiting for threads terminated
  FPS=216.327
  E2E_MEAN=18394.5
  DPU_MEAN=15711.6
  ```

  `./test_performance_refinedet refinedet_baseline test_performance_refinedet.list -t 4 -s 10`

  ```
  root@xilinx-zcu104-2020_2:~/Vitis_AI/examples/Vitis-AI-Library/samples/refinedet# ./test_performance_refinedet refinedet_baseline test_performance_refinedet.list -t 4 -s 10
  WARNING: Logging before InitGoogleLogging() is written to STDERR
  I0110 18:43:53.120203  7733 benchmark.hpp:176] writing report to <STDOUT>
  I0110 18:43:53.551745  7733 benchmark.hpp:203] waiting for 0/10 seconds, 4 threads running
  I0110 18:44:03.552002  7733 benchmark.hpp:211] waiting for threads terminated
  FPS=18.0589
  E2E_MEAN=217547
  DPU_MEAN=214647
  ```

You can observe huge performance improvement between the baseline model and the pruned model brought by the AI optimizer. The non-linear FPS improvement ratio vs. compression ratio is because :

- Network efficiency on DPU decreases to some extent after network size decreases.
- DPU time percentage significantly decreases in the whole application time. Software processing and scheduling overhead increases will affect end-to-end FPS performance.

In the other module in the advanced section, we will introduce how to use AI optimizer to achieve network optimization to reduce network from 125G to 5G.

## Summary

In this module, we use a Caffe detection network (RefineDet) to introduce following items:

- How to use a pre-trained model from Vitis AI model zoo, generate a dataset, and test it with provided scripts
- How to start from the original network, make necessary modifications, conduct training, and get to the version provided in the Model Zoo
- How to quantize and compile the model with ready-to-use Vitis AI toolchain inside docker
- How to deploy the network with Vitis AI library and run on the board

Hope you enjoy it, and let's move to [next module](../Module_6/) .

Copyright&copy; 2020-2022 Xilinx
