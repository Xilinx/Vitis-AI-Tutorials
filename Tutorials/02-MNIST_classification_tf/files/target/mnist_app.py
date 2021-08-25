'''
Copyright 2020 Xilinx Inc.

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

from ctypes import *
import cv2
import numpy as np
import runner
import os
import math
import threading
import time
import argparse



'''
run CNN with batch
dpu: dpu runner
img: imagelist to be run
'''
def runDPU(dpu,img,batchSize,results,threadId,threadImages):

    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    tensorformat = dpu.get_tensor_format()
    if tensorformat == dpu.TensorFormat.NCHW:
        outputHeight = outputTensors[0].dims[2]
        outputWidth = outputTensors[0].dims[3]
        outputChannel = outputTensors[0].dims[1]
    elif tensorformat == dpu.TensorFormat.NHWC:
        outputHeight = outputTensors[0].dims[1]
        outputWidth = outputTensors[0].dims[2]
        outputChannel = outputTensors[0].dims[3]
    else:
        exit("Format error")
    outputSize = outputHeight*outputWidth*outputChannel
    #softmax = np.empty(outputSize)

    remaining = len(img)
    batchCount = 0

    while remaining > 0:

        if (remaining> batchSize):
            runSize = batchSize
        else:
            runSize = remaining
        remaining = remaining - batchSize
 
        shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndims)][1:])
        
        """ prepare batch input/output """
        outputData = []
        inputData = []
        outputData.append(np.empty((runSize,outputSize), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))
        
        """ init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = img[(batchCount*batchSize)+j].reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        """ run with batch """
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        """ calculate argmax over batch """
        for j in range(runSize):
            argmax = np.argmax(outputData[0][j])
            results[(threadId*threadImages)+(batchCount*batchSize)+j] = argmax

        batchCount += 1

    return



def runApp(batchSize, threadnum, json, image_dir):

    """ create runner """
    dpu = runner.Runner(json)

    listImage=os.listdir(image_dir)
    runTotal = len(listImage)


    """ pre-process all images """
    img = []
    for i in range(runTotal):
        image = cv2.imread(os.path.join(image_dir,listImage[i]), cv2.IMREAD_GRAYSCALE)
        image = image.reshape(28,28,1)
        image = image/255.0
        img.append(image)

    """ make a list to hold results - each thread will write into it """
    results = [None] * len(img)


    """run with batch """
    threadAll = []
   
    threadImages=int(len(img)/threadnum)+1

    # set up the threads
    for i in range(threadnum):
        startIdx = i*threadImages
        if ( (len(listImage)-(i*threadImages)) > threadImages):
            endIdx=(i+1)*threadImages
        else:
            endIdx=len(listImage)
        t1 = threading.Thread(target=runDPU, args=(dpu,img[startIdx:endIdx],batchSize,results,i,threadImages))
        threadAll.append(t1)

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("Throughput: %.2f FPS" %fps)

    # post-processing - compare results to ground truth labels
    # ground truth labels are first part of image file name
    classes = ['zero','one','two','three','four','five','six','seven','eight','nine']
    correct=0
    wrong=0

    for i in range(len(listImage)):
        gt = listImage[i].split('_')
        if (gt[0]==classes[results[i]]):
            correct+=1
        else:
            wrong+=1
            print(' ',i,':',listImage[i], classes[results[i]])

    acc = (correct/len(listImage))*100
    print('Correct:',correct,'Wrong:',wrong,'Accuracy: %.2f' %acc,'%')

    del dpu

    return


def main():

    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', '--json',
                    type=str,
                    required=True,
  	                help='Path of folder containing meta.json file. No default, must be supplied by user.')
    ap.add_argument('-i', '--image_dir',
                    type=str,
                    default='images',
  	                help='Path of images folder. Default is ./images')
    ap.add_argument('-t', '--threads',
                    type=int,
                    default=1,
  	                help='Number of threads. Default is 1')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=1,
  	                help='Input batchsize. Default is 1')
    args = ap.parse_args()


    runApp(args.batchsize, args.threads, args.json, args.image_dir)

    
if __name__ == '__main__':
    main()

