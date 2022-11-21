#Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys


'''
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
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

def get_script_directory():
    path = os.getcwd()
    return path

global threadnum
threadnum = 0

def runRfClassify(dpu,rfIn,rfClass,rfSnr,cnt):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
	
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])
    output_ndim = tuple(outputTensors[0].dims)
	
    softmax = np.empty(pre_output_size)
    batchSize = inputTensors[0].dims[0]
    n_of_samples = len(rfIn)
    count = 0;
    top1_count = 0
    mods = [ 'OOK','4ASK','8ASK',
             'BPSK','QPSK','8PSK','16PSK','32PSK',
             '16APSK','32APSK','64APSK','128APSK',
             '16QAM','32QAM','64QAM','128QAM','256QAM',
             'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM',
             'GMSK','OQPSK','BFSK','4FSK','8FSK' ]
    limit = cnt - batchSize
    while count < (limit):
        runSize = batchSize
        shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])

        """prepare batch input/output """
        outputData = []
        inputData = []
        inputData.append(np.empty((shapeIn), dtype=np.int8, order = 'C'))
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]

        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = rfIn[(count+j)% n_of_samples].reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        """run with batch """
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        for j in range(len(outputData)):
            outputData[j] = outputData[j].reshape(runSize, pre_output_size)

        """softmax calculate with batch """
        for j in range(runSize):
           softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size)
           top1 = mods[np.argmax(softmax)]
           maxValue = np.amax(softmax)
           groundTruth = mods[np.argmax(rfClass[count+j])]
           SNR = np.amax(rfSnr[count+j]);
           print(count+j, ": SNR = ",SNR, "   \t Top1 = ", top1, "            \t ", f"{maxValue:.2f}  ", "\t Actual =  ", groundTruth)
           if(groundTruth == top1):
              top1_count = top1_count + 1
			  
        count = count + runSize
    
    print("Number of RF Samples Tested is ", limit)
    print("Batch Size is ", batchSize)
    print("Top1 accuracy = ",  f"{(top1_count/count):.2f}")

"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

	
def main(argv):
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)

    assert len(subgraphs) == 1 # only one DPU kernel
    all_dpu_runners = [];
    threadAll = []
    threadnum = 1
    for i in range(int(threadnum)):
       all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
	
		
    """input files"""
    rfIn=np.load('./rf_input.npy')
    rfClasses=np.load('./rf_classes.npy')
    rfSNRs=np.load('./rf_snrs.npy')
    cnt = len(rfSNRs)

    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runRfClassify, args=(all_dpu_runners[i], rfIn, rfClasses, rfSNRs, cnt))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please input dpu file.")
    else :
        main(sys.argv)