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

from ctypes import *
import cv2
import numpy as np
import runner
import os
import math
import threading
import time
import argparse


l = threading.Lock()

'''
Calculate softmax
data: data to be calculated
size: data size
return: softmax result
'''
def CPUCalcSoftmax(data,size):
    sum=0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i])
        sum +=result[i]
    for i in range(size):
        result[i] /=sum
    return result

'''
 ArgMax can be used instead of Softmax
'''
def CPUCalcArgMax(data):
    return (np.argmax(data)) 



'''
run CNN with batch
dpu: dpu runner
img: imagelist to be run
cnt: threadnum
'''
def runDPU(dpu, img, idx, batchSize, threadnum, runTotal):

    cnt = idx*batchSize

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
    softmax = np.empty(outputSize)

    count = cnt

    while count < runTotal:
        l.acquire()
        if (runTotal < (count+batchSize)):
            runSize = runTotal - count
        else:
            runSize = batchSize
        l.release()
        shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndims)][1:])
        
        """prepare batch input/output """
        outputData = []
        inputData = []
        outputData.append(np.empty((runSize,outputSize), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))
        
        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = img[count+j].reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        """run with batch """
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        """softmax calculate with batch """
        for j in range(runSize):
            #argmax = CPUCalcArgMax(outputData[0][j])
            softmax = CPUCalcSoftmax(outputData[0][j], outputSize)

        l.acquire()
        count = count + threadnum*runSize
        l.release()

    return



def runApp(batchSize, threadnum, json, image_dir):

    """create runner """
    dpu = runner.Runner(json)

    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    """ pre-process all images """
    img = []
    for i in range(runTotal):
        image = cv2.imread(os.path.join(image_dir,listimage[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.0
        img.append(image)

    """run with batch """
    threadAll = []
    time1 = time.time()
    for i in range(threadnum):
        t1 = threading.Thread(target=runDPU, args=(dpu, img, i, batchSize, threadnum, runTotal))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    time2 = time.time()

    timetotal = time2 - time1
    fps = float(runTotal / timetotal)
    print("%.2f FPS" %fps)

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

