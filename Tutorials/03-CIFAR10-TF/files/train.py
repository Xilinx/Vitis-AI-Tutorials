'''
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import os
import sys
import argparse
import tensorflow as tf
import numpy as np

from netmodel.customCNN import customCNN
from datadownload import datadownload

# reduce TensorFlow messages in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def train(input_height,input_width,input_chan,epochs,learnrate, \
          batchsize,output_ckpt_path,infer_graph_path,tboard_path):
    
    # Set up directories and files
    INFER_GRAPH_DIR = os.path.dirname(infer_graph_path)
    INFER_GRAPH_FILENAME =os.path.basename(infer_graph_path)
    
    print('\nDATASET PREPARATION:')
    # CIFAR10 dataset has 60k images. Training set is 50k, test set is 10k.
    # Each image is 32x32x8bits
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = datadownload()    


    # calculate total number of batches per epoch
    total_batches = int(len(x_train)/batchsize)


    #####################################################
    # Create the Computational graph
    #####################################################

    # define placeholders for the input images, labels, training mode and droput rate
    images_in = tf.placeholder(tf.float32, shape=[None,input_height,input_width,input_chan], name='images_in')
    labels = tf.placeholder(tf.int32, shape=[None,10], name='labels')
    train = tf.placeholder_with_default(False, shape=None, name='train')
    drop = tf.placeholder_with_default(0.0, shape=None, name='drop')


    # build the network, input comes from the 'images_in' placeholder
    # training mode and dropout rate are also driven by placeholders
    logits = customCNN(cnn_in=images_in, is_training=train, drop_rate=drop)


    # softmax cross entropy loss function
    # needs one-hot encoded labels
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))


    # Adaptive Momentum optimizer - minimize the loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=learnrate)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)


    # accuracy calculation during training:
    # Check to see if the predictions matches the labels and then
    # calculate accuracy as mean of the correct predictions
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    ground_truth_label = tf.argmax(labels, 1, output_type=tf.int32)
    correct_prediction = tf.equal(predicted_logit, ground_truth_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # TensorBoard data collection
    tf.summary.scalar('cross_entropy_loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('input_images', images_in)    

    # set up saver object
    saver = tf.train.Saver()



    #####################################################
    # Run the graph in a Session
    #####################################################
    # Launch the graph
    with tf.Session() as sess:

        sess.run(tf.initializers.global_variables())

        # TensorBoard writer
        writer = tf.summary.FileWriter(tboard_path, sess.graph)
        tb_summary = tf.summary.merge_all()

        # Training phase with training data
        print("\nTRAINING:")
        for epoch in range(epochs):

            # process all batches
            for i in range(total_batches):

                # fetch a batch from training dataset
                x_batch, y_batch = x_train[i*batchsize:i*batchsize+batchsize], \
                                   y_train[i*batchsize:i*batchsize+batchsize]

                # Run graph for optimization  - i.e. do the training
                train_feed_dict={images_in: x_batch, labels: y_batch, train: True, drop: 0.2}
                _, s = sess.run([train_op, tb_summary], feed_dict=train_feed_dict)
                writer.add_summary(s, (epoch*total_batches + i))

            # check training accuracy at end of epoch
            test_acc = sess.run(accuracy, feed_dict={images_in: x_test[:1000], labels: y_test[:1000]})
            print (" Epoch", epoch+1, "/", epochs, '- accuracy: {:1.4f}'.format(test_acc))

        writer.flush()
        writer.close()


        # Validation phase with validation dataset
        # calculate accuracy with 'unseen' data
        print("\nVALIDATION:")
        valid_feed_dict={images_in: x_valid[:1000], labels: y_valid[:1000]}
        valid_acc = sess.run(accuracy, feed_dict=valid_feed_dict)
        print (' Trained model accuracy with validation set: {:1.4f}'.format(valid_acc))

        # save post-training checkpoint, this saves all the parameters of the trained network
        print("\nSAVING:")
        saver.save(sess, output_ckpt_path)
        print(' Saved checkpoint to %s' % output_ckpt_path)


    #####  SESSION ENDS HERE #############


    
    #####################################################
    # Write out a graph for inference
    #####################################################
    # we cannot use the training graph for deployment
    # we need to create a new graph with is_training set to False to disable dropout & batch norm
    # this new graph does not have any nodes associated with training (loss, optimizer, etc)

    with tf.Graph().as_default():

      # define placeholders for the input data
      x_1 = tf.placeholder(tf.float32, shape=[None,input_height,input_width,input_chan], name='images_in')

      # call the miniVGGNet function with is_training=False
      logits_1 = customCNN(cnn_in=x_1, is_training=False, drop_rate=0.0)

      tf.train.write_graph(tf.get_default_graph().as_graph_def(), INFER_GRAPH_DIR, INFER_GRAPH_FILENAME, as_text=False)
      print(' Saved binary inference graph to %s' % infer_graph_path)
    

    print('\nFINISHED')
    print(' Run `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % tboard_path)

    return


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('-ih', '--input_height',
                    type=int,
                    default=32,
                    help='Input data height. Default is 32')                  
    ap.add_argument('-iw', '--input_width',
                    type=int,
                    default=32,
                    help='Input data width. Default is 32')                  
    ap.add_argument('-ic', '--input_chan',
                    type=int,
                    default=3,
                    help='Input data channels. Default is 3')                  
    ap.add_argument('-e', '--epochs',
                    type=int,
                    default=100,
                    help='Number of training epochs. Default is 100')                  
    ap.add_argument('-l', '--learnrate',
                    type=float,
                    default=0.0001,
                    help='Learning rate. Default is 0.0001')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=50,
                    help='Training batchsize. Default is 50')  
    ap.add_argument('-o', '--output_ckpt_path',
                    type=str,
                    default='./chkpt/float_model.ckpt',
                    help='Path and filename of trained checkpoint. Default is ./chkpt/float_model.ckpt')          
    ap.add_argument('-g', '--infer_graph_path',
                    type=str,
                    default='./chkpt/inference_graph.pb',
                    help='Path and filename of inference graph. Default is ./chkpt/inference_graph.pb')
    ap.add_argument('-t', '--tboard_path',
                    type=str,
                    default='./tb_log',
                    help='Path of TensorBoard logs. Default is ./tb_log')
                  
    args = ap.parse_args() 


    print('\n------------------------------------')
    print('Keras version      :',tf.keras.__version__)
    print('TensorFlow version :',tf.__version__)
    print('Python version     :',(sys.version))
    print('------------------------------------')
    print ('Command line options:')

    print (' --input_height    : ', args.input_height)
    print (' --input_width     : ', args.input_width)
    print (' --input_chan      : ', args.input_chan)
    print (' --epochs          : ', args.epochs)
    print (' --batchsize       : ', args.batchsize)
    print (' --learnrate       : ', args.learnrate)
    print (' --output_ckpt_path: ', args.output_ckpt_path)
    print (' --infer_graph_path: ', args.infer_graph_path)
    print (' --tboard_path     : ', args.tboard_path)
    print('------------------------------------\n')

    train(args.input_height,args.input_width,args.input_chan,args.epochs,args.learnrate, \
          args.batchsize,args.output_ckpt_path,args.infer_graph_path,args.tboard_path)


if __name__ == '__main__':
  main()

