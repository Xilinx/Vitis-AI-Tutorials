# Copyright 2022 Xilinx Inc.
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

import numpy as np

class Lambda:
    def __init__(self,op):
        pass

    def calculate(self, output, input):
        np_output=np.array(output,copy=False)
        np_input=np.array(input[0],copy=False)
        L=len(np_output)
        if L==0:
            return
        batch, height, width, depth = np_input.shape
        increase_width = width * 2
        for b in range(batch):
            h1=np_input[b,:,:,0:depth//2]
            h2=np_input[b,:,:,depth//2:]
            h1=h1.reshape(1,height, increase_width, depth//4)
            h2=h2.reshape(1,height, increase_width, depth//4)
            outtest=[]
            for h in range(height):
                outtest.append((np.concatenate([h1[:,h,:,:],h2[:,h,:,:]])))
            
            np_output[b,:,:,:]=np.expand_dims((np.concatenate((outtest),axis=0)),axis=0)


if __name__ == '__main__':
    import os,sys
    tensors = []
    golden = []
    out_tensors = []
    base_directory = 'dump_results_0/'
    # define the tensor input and output dictionaries - this should be the filename of the 
    # tensor from the dumped files and the expected shape for the value pair
    tensor_inputs = {'quant_add_5.bin':(1,32,32,1024),
                     'quant_add_7.bin':(1,64,64,256),
                     'quant_add_9.bin':(1,128,128,64),
                     'quant_add_11.bin':(1,256,256,16)} 

    golden_outputs = {'quant_lambda.bin':(1,64,64,256),
                      'quant_lambda_1.bin':(1,128,128,64),
                      'quant_lambda_2.bin':(1,256,256,16),
                      'quant_lambda_3.bin':(1,512,512,4)}

    for file, shape in tensor_inputs.items():
        #read in the raw tensor values for input stimulus
        tensors.append(np.reshape(np.fromfile(base_directory+file,dtype='uint8'),shape))
    
    for file, shape in golden_outputs.items():
        #read in the golden values for cross checking
        golden.append(np.reshape(np.fromfile(base_directory+file,dtype='uint8'),shape))
        
        #generate output tensor placeholders for the lambda layer
        out_tensors.append(np.empty(shape,dtype='uint8'))

    myLambda=Lambda

    for i,o,g in zip(tensors,out_tensors,golden):
        #we need to expand the input dimensions so that the test tensor input looks like the tensor input from VART
        myLambda.calculate(myLambda,o,np.expand_dims(i,axis=0))
        if o.all() != g.all():
            print("Error, Output Doesn't Match Golden")
            sys.exit(-1)
    
    print("All Tensors found to Match!")


    #Basic Test Case:
    #height = 4
    #width = 4
    #chan = 16
    #batch = 3
    #in_test = np.zeros((batch,height,width,chan))
    #out_test=np.zeros((batch,height*2, width*2,chan//4))
    #cnt = 0
    #for b in range(batch):
    #    for h in range(height):
    #        for w in range(width):
    #            for c in range(chan):
    #                in_test[b][h][w][c]=cnt
    #                cnt = cnt+1
    #myLambda=Lambda
    #print(in_test)
    #print(in_test.shape)
    #myLambda.calculate(myLambda,out_test,np.expand_dims(in_test, axis=0))