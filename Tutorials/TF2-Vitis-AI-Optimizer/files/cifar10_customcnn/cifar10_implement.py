#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Authors: Daniele Bagni (AMD)
# date:  01 Aug 2023


import os
#https://bobbyhadz.com/blog/disable-suppress-tensorflow-warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
#https://github.com/tensorflow/tensorflow/issues/8340
logging.getLogger("tensorflow").setLevel(logging.WARNING)


from config import cifar10_config as cfg #DB
print(cfg.SCRIPT_DIR)


import argparse
import numpy as np
import tensorflow as tf
import cv2, sys
from random import seed
from random import random
from random import shuffle
import glob
from datetime import datetime
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input,Conv2D,Dropout,Add,DepthwiseConv2D,Dense, Activation
from tensorflow.keras.layers import MaxPooling2D,BatchNormalization,ReLU, Flatten, Softmax
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from tensorflow.keras.preprocessing.image  import img_to_array
from tensorflow.keras.utils                import plot_model, to_categorical

#Vitis-AI Optimizer tool
try:
  from tf_nndct import IterativePruningRunner #VAI3.5
except:
  print("tf_nndct               not found") #VAI3.5
#Vitis-AI Quantizer tool
try:
  from tensorflow_model_optimization.quantization.keras import vitis_quantize
except:
  print("tensorflow_model_optimization.quantization.keras not found")


# ==========================================================================================
#   configuration
# ==========================================================================================
train_target_acc = cfg.train_target_acc
train_init_lr = cfg.train_init_lr
train_epochs = cfg.train_epochs
batchsize=cfg.BATCH_SIZE
input_shape=cfg.input_shape
model_name=cfg.model_name
init_prune_ratio=cfg.init_prune_ratio
incr_prune_ratio=cfg.incr_prune_ratio
prune_steps=cfg.prune_steps
finetune_init_lr=cfg.finetune_init_lr
DIVIDER=cfg.DIVIDER
CNN=cfg.cnn


# ==========================================================================================
# Get ResNet18 pre-trained model
# ==========================================================================================
from classification_models.keras import Classifiers
#print("\n[DB INFO] Get ResNet18 pretrained model...\n")

# original imagenet-based ResNet18 model
ResNet18, preprocess_input = Classifiers.get("resnet18")
#orig_model = ResNet18((224, 224, 3), weights="imagenet")
#print(orig_model.summary())

# build new model for CIFAR10
base_model = ResNet18(input_shape=(32,32,3), weights="imagenet", include_top=False)
#base_model = ResNet18(input_shape=(32,32,3), include_top=False)
x_layer = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax")(x_layer)
cnn_model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# ==========================================================================================
# miniVggNet CNN
# this is the same CNN publiced on previous Vitis-AI (TensorFlow1) tutorial here
# https://github.com/Xilinx/Vitis-AI-Tutorials/blob/3.0/Tutorials/Keras_GoogleNet_ResNet/files/code/custom_cnn.py
# just re-writtten for TensorFlow2
# ==========================================================================================
#from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation

def cbr(inputs, filters, kernel_size):
  # Convolution - BatchNorm - ReLU6
  net = Conv2D(filters, kernel_size, padding="same", kernel_initializer='he_uniform',
               use_bias=False)(inputs)
  net = BatchNormalization()(net)
  #net = ReLU(6.)(net)
  net = ReLU()(net)
  return net

# this CNN makes the Vitis_AI XIR Compiler failing
def ORIGINAL_miniVggNet(input_shape=(None,None,None),num_classes=None,bnEps=2e-5, bnMom=0.9,dropout=False):
  input_layer = Input(shape=input_shape)
  net = cbr(input_layer, 32, (3,3))
  net = cbr(net, 32, (3,3))
  net = MaxPooling2D(pool_size=(2, 2))(net)
  net = Dropout(rate=0.25)(net)
  net = cbr(net, 64, (3,3))
  net = cbr(net, 64, (3,3))
  net = MaxPooling2D(pool_size=(2, 2))(net)
  net = Dropout(rate=0.25)(net)
  net = Flatten()(net)
  net = Dense(512)(net)
  net = BatchNormalization()(net)
  net = ReLU()(net)
  net = Dropout(rate=0.50)(net)
  net = Dense(num_classes, activation="softmax")(net)
  #output_layer = Softmax(net)
  output_layer = net
  return Model(inputs=[input_layer], outputs=[output_layer])

# properly designed for VEK280
def miniVggNet(input_shape=(None,None,None),num_classes=None,bnEps=2e-5, bnMom=0.9,dropout=False):
  input_layer = Input(shape=input_shape)
  net = cbr(input_layer, 32, (3,3))
  net = cbr(net, 32, (3,3))
  net = MaxPooling2D(pool_size=(2, 2))(net)
  net = Dropout(rate=0.25)(net)
  net = cbr(net, 64, (3,3))
  net = cbr(net, 64, (3,3))
  net = MaxPooling2D(pool_size=(2, 2))(net)
  net = Dropout(rate=0.25)(net)
  net = cbr(net, 128, (3,3))
  net = cbr(net, 128, (3,3))
  net = MaxPooling2D(pool_size=(2, 2))(net)  
  net = Dropout(rate=0.25)(net)
  net = Flatten()(net)
  net = Dense(512)(net)
  net = BatchNormalization()(net)
  net = ReLU()(net)
  net = Dropout(rate=0.50)(net)
  net = Dense(num_classes, activation="softmax")(net)
  #output_layer = Softmax(net)
  output_layer = net
  return Model(inputs=[input_layer], outputs=[output_layer])


def build_CNN(weights=None,input_shape=(None,None,None),n_classes=None, cnn_type="ResNet18"):
  #build model and load weights if required
  if cnn_type=="miniVggNet" :
    model = miniVggNet(input_shape=input_shape,num_classes=n_classes)
  elif cnn_type=="ResNet18" :
    model = cnn_model #ResNet18
  else :
    print("ERROR: wrong CNN!")

  if (weights):
    print("Loading weights from",weights)
    #model.load_weights(weights).expect_partial() #DB: #https://github.com/tensorflow/tensorflow/issues/43554
    model.load_weights(weights)
  else:
    print("No weights to load")
  return model

# ==========================================================================================
# TRAINING
# ==========================================================================================

class EarlyStoponAcc(tf.keras.callbacks.Callback):
  '''
  Early stop on reaching target accuracy
  '''
  def __init__(self, target_acc):
    super(EarlyStoponAcc, self).__init__()
    self.target_acc=target_acc
  def on_epoch_end(self, epoch, logs=None):
    accuracy=logs["val_accuracy"]
    if accuracy >= self.target_acc:
      self.model.stop_training=True
      print("Reached target accuracy of ",self.target_acc,"...exiting.")


def train(model,output_ckpt,learnrate,train_dataset,test_dataset,epochs,batchsize,target_acc):

  def step_decay(epoch):
    '''
    Learning rate scheduler used by callback
    Reduces learning rate depending on number of epochs
    '''
    lr = learnrate
    if epoch > int(epochs*0.9):
      lr /= 100
    elif epoch > int(epochs*0.05):
      lr /= 10
    return lr

  # Call backs
  chkpt_call = ModelCheckpoint(filepath=output_ckpt,
                               monitor="val_accuracy",
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True)

  early_stop_call = EarlyStoponAcc(target_acc)
  #  lr_scheduler_call = LearningRateScheduler(schedule=step_decay,verbose=1)
  #  callbacks_list = [chkpt_call, early_stop_call, lr_scheduler_call]
  callbacks_list = [chkpt_call, early_stop_call]

  #  Compile model
  model.compile(optimizer=Adam(learning_rate=learnrate),
                #loss=SparseCategoricalCrossentropy(from_logits=True),
                loss="categorical_crossentropy",
                metrics=["accuracy"])

  # Training
  print("\n"+DIVIDER)
  print("Training model with training set...")
  print(DIVIDER)

  startTime1 = datetime.now() #DB
  # run training
  train_history=model.fit(train_dataset, #aug_generator,
                          epochs=epochs,
                          batch_size=batchsize,
                          steps_per_epoch=len(x_train)//batchsize,
                          validation_data=validation_generator,
                          #validation_steps=None,
                          validation_steps=len(x_valid)//batchsize,
                          callbacks=callbacks_list,
                          shuffle=True,
                          verbose=2)
  endTime1 = datetime.now()
  diff1 = endTime1 - startTime1
  print("\n")
  print("Elapsed time for Keras training (s): ", diff1.total_seconds())
  print("\n")

  return


def evaluate(model, x_test, y_test):
  '''
  Evaluate a keras model with the np arrays for test dataset
  '''
  model.compile(loss="categorical_crossentropy",
                #loss=SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",
                metrics=["accuracy"])
  scores = model.evaluate(x_test, y_test,
                          batch_size=batchsize,
                          steps=None,
                          verbose=0)
  return scores

# ==========================================================================================
#   PRUNING routines
# ==========================================================================================

def ana_eval(x_test, y_test, model):
    model.compile(loss="categorical_crossentropy",
                  #loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam",
                  metrics=["accuracy"])
    scores = model.evaluate(x_test, y_test,
                            batch_size=batchsize,
                            steps=None,
                            verbose=0)
    print("ANA EVAL accuracy: {0:.4f}".format(scores[1]*100),"%")
    return scores[1]

def prune(model, ratio, x_test, y_test):
  '''
  Prune the model
  '''
  input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
  runner = IterativePruningRunner(model, input_spec)
  import functools # see https://www.learnpython.org/en/Partial_functions
  eval_fn = functools.partial(ana_eval, x_test, y_test)
  runner.ana(eval_fn)
  return runner.prune(ratio)


# ==========================================================================================
#  Implement train -> prune -> transform -> quantize -> compile
# ==========================================================================================

def implement(build_dir, mode, target, network):
  # Implements training, pruning and transform modes

  global aug_generator, validation_generator
  global x_valid, y_valid, x_train, y_train, x_test, y_test
  
  # output checkpoints and folders for each mode
  train_output_ckpt = build_dir + cfg.train_output_ckpt
  prune_output_ckpt = build_dir + cfg.prune_output_ckpt
  transform_output_ckpt = build_dir + cfg.transform_output_ckpt
  quant_output_ckpt = build_dir + cfg.quant_output_ckpt
  compile_output_dir = build_dir+cfg.compile_dir+target

  if (mode=="train"):
    print("\n[DB INFO] IMPLEMENT TRAINING ...\n")

    # build CNN without weights
    model = build_CNN(input_shape=input_shape, n_classes=cfg.NUM_CLASSES, cnn_type = network)

    # make folder for saving trained model checkpoint
    os.makedirs(os.path.dirname(train_output_ckpt), exist_ok=True)

    # run initial training
    train(model,train_output_ckpt,train_init_lr,
          aug_generator, #train_dataset,
          validation_generator, #test_dataset,
          train_epochs,batchsize,train_target_acc)

    # eval trained checkpoint
    #model = build_CNN(weights=train_output_ckpt, input_shape=input_shape, n_classes=cfg.NUM_CLASSES)
    model.summary()
    # Prediction
    print("\n[DB INFO] evaluating network on Test and Validation datasets...\n")
    # Evaluate model accuracy with test set
    scores = evaluate(model, x_valid, y_valid)
    print("Validation Loss:     %.3f" % scores[0])
    print("Validation Accuracy: %.3f" % scores[1])
    scores = evaluate(model, x_test, y_test)
    print("Test Loss:           %.3f" % scores[0])
    print("Test Accuracy:       %.3f" % scores[1])
    # save final accuracy to a text file for use in pruning
    f = open(build_dir+"/trained_accuracy.txt", "w")
    f.write(str(scores[1]))
    f.close()
    # make predictions on the test set
    preds = model.predict(x_test)
    # show a nicely formatted classification report
    print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=cfg.labelNames_list))

  elif (mode=="prune"):
    print("\n[DB INFO] IMPLEMENT PRUNING ...\n")
    # build CNN with weights from initial training
    model = build_CNN(weights=train_output_ckpt, input_shape=input_shape, n_classes=cfg.NUM_CLASSES, cnn_type = network)

    prune_ratio=init_prune_ratio
    # fetch the required final accuracy
    f = open(build_dir+"/trained_accuracy.txt", "r")
    final_ft_acc = float(f.readline())
    f.close()
    print("\n[DB INFO] evaluating network on Test and Validation datasets...\n")
    # Evaluate model accuracy with test set
    scores = evaluate(model, x_valid, y_valid)
    print("Validation Loss:     %.3f" % scores[0])
    print("Validation Accuracy: %.3f" % scores[1])
    scores = evaluate(model, x_test, y_test)
    print("Test Loss:           %.3f" % scores[0])
    print("Test Accuracy:       %.3f" % scores[1])

    for i in range(1,prune_steps+1):
      print(DIVIDER)
      print("Pruning iteration ",i," of",prune_steps," Pruning ratio: ",prune_ratio)
      if (i==prune_steps):
        finetune_target_acc=final_ft_acc
      else:
        finetune_target_acc=final_ft_acc*0.97
      print("Target accuracy for this iteration: ",finetune_target_acc)
      # prune model
      pruned_model = prune(model,prune_ratio,x_test, y_test)
      # fine-tune pruned model
      train(pruned_model,prune_output_ckpt,finetune_init_lr,
      aug_generator, #train_dataset,
      validation_generator, #test_dataset,
      train_epochs,batchsize,finetune_target_acc)
      # increment the pruning ratio for the next iteration
      prune_ratio+=incr_prune_ratio

    # eval best fine-tuned checkpoint
    model = build_CNN(weights=prune_output_ckpt, input_shape=input_shape, n_classes=cfg.NUM_CLASSES, cnn_type = network)
    ##https://stackoverflow.com/questions/71929036/value-in-checkpoint-could-not-be-found-in-the-restored-object-root-optimizer
    model.load_weights(prune_output_ckpt).expect_partial()
    #model.summary()
    scores = evaluate(model,x_test, y_test)
    print("\n[DB INFO] Pruned model accuracy: {0:.4f}".format(scores[1]*100),"%")

  elif (mode=="transform"):
    print("\n[DB INFO] IMPLEMENT TRANSFORM ...\n")
    # build mobilenet with weights from last pruning iteration
    model = build_CNN(weights=prune_output_ckpt,input_shape=input_shape, n_classes=cfg.NUM_CLASSES, cnn_type = network)
    ##https://stackoverflow.com/questions/71929036/value-in-checkpoint-could-not-be-found-in-the-restored-object-root-optimizer
    
    model.load_weights(prune_output_ckpt).expect_partial()
    # make and save slim model
    input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
    runner = IterativePruningRunner(model, input_spec)
    slim_model = runner.get_slim_model()
    os.makedirs(os.path.dirname(transform_output_ckpt), exist_ok=True)
    # to silence warnings: see https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    slim_model.save(transform_output_ckpt)
    # eval slim model
    scores = evaluate(slim_model,x_test, y_test)
    print("\n[DB INFO] Slim model accuracy: {0:.4f}".format(scores[1]*100),"%")

  elif (mode=="quantize"):
    print("\n[DB INFO] IMPLEMENT QUANTIZATION ...\n")
    # make folder for saving quantized model
    os.makedirs(os.path.dirname(quant_output_ckpt), exist_ok=True)
    # load the transformed model if it exists
    # otherwise, load the trained model
    if (os.path.exists(transform_output_ckpt)):
      print("Loading transformed model...")
      float_model = load_model(transform_output_ckpt,compile=False)
    else:
      input_ckpt = train_output_ckpt
      print("Did not find transformed model, loading trained model...")
      # build CNN with weights
      float_model = build_CNN(weights=input_ckpt,input_shape=input_shape, n_classes=cfg.NUM_CLASSES, cnn_type = network)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=x_test[0:1999])
    # saved quantized model
    quantized_model.save(quant_output_ckpt)
    print("Saved quantized model to",quant_output_ckpt)
    # Evaluate quantized model
    print("\n"+DIVIDER)
    print ("Evaluating quantized model...")
    print(DIVIDER+"\n")
    quantized_model.compile(#loss=SparseCategoricalCrossentropy(from_logits=True),
                            loss="categorical_crossentropy",
                            optimizer="adam", metrics=["accuracy"])
    scores = quantized_model.evaluate(x_test, y_test,verbose=0)
    print("\n[DB INFO] Quantized model accuracy: {0:.4f}".format(scores[1]*100),"%")
    print("\n"+DIVIDER)

  elif (mode=="compile"):
    print("\n[DB INFO] IMPLEMENT GENERATE (COMPILE) XMODEL ...\n")
    # set the arch value for the compiler script
    arch_dict = {
      "zcu102": "DPUCZDX8G/ZCU102",
      "zcu104": "DPUCZDX8G/ZCU104",
      "kv260": "DPUCZDX8G/KV260",
      "vck190": "DPUCVDX8G/VCK190",
      "vek280": "DPUCV2DX8G/VEK280"
    }
    arch="/opt/vitis_ai/compiler/arch/"+arch_dict[target.lower()]+"/arch.json"
    # path to TF2 compiler script
    compiler_path = "vai_c_tensorflow2"
    # arguments for compiler script
    cmd_args = " --model " + quant_output_ckpt + " --output_dir " + compile_output_dir + " --arch " + arch + " --net_name " + model_name
    # run compiler python script
    os.system(compiler_path + cmd_args)

  else:
    print("INVALID MODE - valid modes are train, prune, transform, quantize, compile")
  return


# ==========================================================================================
#   MAIN routines
# ==========================================================================================

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-bd", "--build_dir", type=str, default="build", help="Path of build folder. Default is build")
  ap.add_argument("-m",  "--mode",      type=str, default="train", choices=["train","prune","transform","quantize","compile"],
                  help="Mode: train,prune,transform,quantize,compile. Default is train")
  ap.add_argument("-t" , "--target",    type=str, default="zcu102",help="Target platform. Default is zcu102")
  ap.add_argument("-n" , "--network",   type=str, default="ResNet18",help="ResNet18 / miniVggNet")
  args = ap.parse_args()
  print("\n"+DIVIDER)
  print("Keras version      : ",tf.keras.__version__)
  print("TensorFlow version : ",tf.__version__)
  print(sys.version)
  print(DIVIDER)
  #DB: test GPU
  if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
  else:
    print("Please install GPU version of TF2")


  global aug_generator, validation_generator
  global x_valid, y_valid, x_train, y_train, x_test, y_test
  
  # =========================
  # prepare your data
  # =========================
  print("\n[DB INFO] Creating numpy lists of images ...\n")
  startTime1 = datetime.now() #DB

  if ((args.mode=="train") or (args.mode=="prune")):
    # make a list of all files currently in the TRAIN folder
    imagesList = [img for img in glob.glob(cfg.TRAIN_DIR + "/*/*.png")]
    seed(42)
    shuffle(imagesList)
    x_train, y_train = list(), list()
    for img in imagesList:
      filename = os.path.basename(img)
      classname = filename.split("_")[0]
      #print(img)
      # read image with OpenCV
      img_orig = cv2.imread(img)
      #rs_img = cv2.resize(img_orig, (256,256))
      #cv2.imshow(classname, rs_img)
      #cv2.waitKey(0)
      img_array = img_to_array(img_orig, data_format=None)
      x_train.append(img_array)
      y_train.append(cfg.labelNames_dict[classname])
    print("[DB INFO] x_train: done")
    # one-hot encode the training labels
    y_train = to_categorical(y_train, 10)
    # preprocess
    x_train = np.asarray(x_train)
    x_train = cfg.Normalize(x_train)
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    # Data Generators
    print("\n[DB INFO] Data Generators ...\n")
    train_datagen = ImageDataGenerator()
    aug_datagen   = ImageDataGenerator(
      #rescale=1/255,
      rotation_range=5,
      horizontal_flip=True,
      height_shift_range=0.05,
      width_shift_range=0.05,
      shear_range=0.2,
      zoom_range=0.2)
    aug_generator = aug_datagen.flow(
      x_train, y_train,
      batch_size=batchsize)
    train_generator = train_datagen.flow(
      x_train, y_train,
      batch_size=batchsize)    
  
    # make a list of all files currently in the VALID folder
    imagesList = [img for img in glob.glob(cfg.VALID_DIR + "/*/*.png")]
    shuffle(imagesList)
    x_valid, y_valid = list(), list()
    for img in imagesList:
      filename = os.path.basename(img)
      classname = filename.split("_")[0]
      #print(img)
      # read image with OpenCV
      img_orig = cv2.imread(img)
      img_array = img_to_array(img_orig, data_format=None)
      x_valid.append(img_array)
      y_valid.append(cfg.labelNames_dict[classname])
    print("[DB INFO] x_valid: done")
    # one-hot encode the testing labels
    y_valid = to_categorical(y_valid, 10)
    # preprocess
    x_valid = np.asarray(x_valid)
    x_valid = cfg.Normalize(x_valid)
    print("x_valid shape: ", x_valid.shape)
    print("y_valid shape: ", y_valid.shape)
    valid_datagen = ImageDataGenerator()
    validation_generator = valid_datagen.flow(
      x_valid, y_valid,
      batch_size=batchsize)
    
    #assert True, ( len(x_train) > cfg.NUM_TRAIN_IMAGES)
    #assert True, ( len(x_test) >= (cfg.NUM_TRAIN_IMAGES+cfg.NUM_VAL_IMAGES))
    #assert True, ( cfg.NUM_TRAIN_IMAGES==cfg.NUM_VAL_IMAGES )

    
  if ((args.mode=="quantize") or (args.mode=="transform") or (args.mode=="train") or (args.mode=="prune")):
    # make a list of all files currently in the TEST folder
    imagesList = [img for img in glob.glob(cfg.TEST_DIR + "/*/*.png")]
    shuffle(imagesList)
    x_test, y_test = list(), list()
    for img in imagesList:
      filename = os.path.basename(img)
      classname = filename.split("_")[0]
      #print(img)
      # read image with OpenCV
      img_orig = cv2.imread(img)
      img_array = img_to_array(img_orig, data_format=None)
      x_test.append(img_array)
      y_test.append(cfg.labelNames_dict[classname])
    print("[DB INFO] x_test: done")
    ## one-hot encode the training and testing labels
    y_test  = to_categorical(y_test,  10)
    # preprocess
    x_test = np.asarray(x_test)
    x_test = cfg.Normalize(x_test)
    print("x_test  shape: ", x_test.shape)
    print("y_test  shape: ", y_test.shape)    
    test_datagen  = ImageDataGenerator()
    pred_generator = test_datagen.flow(
      x_test, y_test,
      batch_size=1)
  
  endTime1 = datetime.now()
  diff1 = endTime1 - startTime1
  print("\n")
  print("Elapsed time for Data Generation (s): ", diff1.total_seconds())
  print("\n")

  # ========================================
  # IMPLEMENT
  # ========================================

  cnn = args.network

  #check CNN name
  check_cnn1_name = ((cnn=="miniVggNet") or (cnn=="minivggnet") or (cnn=="MINIVGGNET"))
  check_cnn2_name = ((cnn=="ResNet18"  ) or (cnn=="resnet18"  ) or (cnn=="RESNET18"  ))
  if ((check_cnn1_name or check_cnn2_name) == False):
    print("CNN ERROR: either the name is mispelled or the cnn is not supported")
    return
  else:
    print("the CNN selected is ", cnn)
  
  #cnn = "miniVggNet"
  implement(args.build_dir, args.mode, args.target, cnn)


if __name__ == "__main__":
    run_main()
