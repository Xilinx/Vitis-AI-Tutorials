/*

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

*/


#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <thread>

#include "common.h"

using namespace std;
using namespace std::chrono;

GraphInfo shapes;

int num_threads   = 0;
int num_of_images = 0;
int num_images_x_thread = 0;

void runDPU(vart::Runner* runner, int8_t *imageInputs, int8_t *FCResult)
{
  // get in/out tensors and dims
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_dims();
  auto in_dims = inputTensors[0]->get_dims();
  int batchSize = in_dims[0];
  int outSize   = shapes.outTensorList[0].size;
  int inSize    = shapes.inTensorList[0].size;
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
  int8_t *loc_imageInputs = imageInputs;
  int8_t *loc_FCResult    = FCResult;
  for (unsigned int n = 0; n < num_images_x_thread; n += batchSize) //this works correctly for either batchSize= 1 or 3
  {

    loc_imageInputs = imageInputs+n * inSize;
    loc_FCResult    = FCResult   +n * outSize;
    // in/out tensor refactory for batch inout/output
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(inputTensors[0]->get_name(),
          in_dims, xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(loc_imageInputs, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(),
          out_dims,xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>( loc_FCResult, batchTensors.back().get()));
    //tensor buffer input/output
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());
    //run
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    inputs.clear();
    outputs.clear();
  }
}



int main(int argc, char* argv[])
{

  // Check args
  if (argc != 4) {
    cout << "Usage: get_dpu_fps xmodel_pathname num_of_threads(from 1 to 6) num_of_images" << endl;
    return -1;
  }
  num_threads   = atoi(argv[2]);
  assert( (num_threads<=6)&(num_threads>=1) );
  num_of_images = atoi(argv[3]);

  for (int i = 0; i< argc; i++)
    cout << argv[i] << " ";
  cout << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPARE DPU STUFF

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  //create up to 6 runners
  auto runner  = vart::Runner::create_runner(subgraph[0], "run");
  auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner5 = vart::Runner::create_runner(subgraph[0], "run");

  // get in/out tensors and dims
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  // get in/out tensor shape
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // get shape info
  int outSize   = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth  = shapes.outTensorList[0].width;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);

  //scaling factors
  //int out_fixpos = (outputTensors[0])->template get_attr<int>("fix_point");
  //auto out_fix_scale = std::exp2f(1.0f * (float)out_fixpos);
  //cout << "outS     " << out_fix_scale << endl;
  //int in_fixpos = (inputTensors[0])->template get_attr<int>("fix_point");
  //auto in_fix_scale = std::exp2f(1.0f * (float)in_fixpos);
  //cout << "inP      " << in_fixpos << endl;
  //cout << "inS      " << in_fix_scale << endl;
  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);

  // debug messages
  cout << "outSize   " << outSize << endl;
  cout << "inSize    " << inSize << endl;
  cout << "outW      " << outWidth << endl;
  cout << "outH      " << outHeight << endl;
  cout << "inpW      " << inWidth << endl;
  cout << "inpH      " << inHeight << endl;
  cout << "inp scale " << input_scale << endl;
  cout << "out scale " << output_scale << endl;
  cout << "# classes " << num_of_classes << endl;
  cout << "batchSize " << batchSize << endl;


  /////////////////////////////////////////////////////////////////////////////////////////////
  // TIMERS CALIBRATION

  int num_of_trials = 200;
  std::chrono::duration<double, std::micro> avg_calibr_highres(0);
  for (int i =0; i<num_of_trials; i++)
  {
      auto t1 = std::chrono::high_resolution_clock::now();
      auto t2 = std::chrono::high_resolution_clock::now();
      // floating-point duration: no duration_cast needed
      std::chrono::duration<double, std::micro> fp_us = t2 - t1;
      avg_calibr_highres  += fp_us;
      //if (i%10 ==0) cout << "[Timers calibration  ] " << fp_us.count() << "us" << endl;
    }
  cout << "[average calibration high resolution clock] " << avg_calibr_highres.count() / num_of_trials << "us"  << endl;
  cout << "\n" << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MEMORY ALLOCATION

  //number of images per thread
  num_images_x_thread = num_of_images/num_threads;
  num_images_x_thread = (num_images_x_thread/batchSize)*batchSize;
  cout << "\n number of dummy images per thread: " << num_images_x_thread << endl;
  //effective number of images as multiple of num_threads and batchSize
  num_of_images = num_images_x_thread * num_threads;

  // memory allocation
  int8_t * imageInputs = new int8_t [(num_of_images)*inSize];
  int8_t * FCResult    = new int8_t [(num_of_images)*outSize];
  cout << "\n allocated " << num_of_images* inSize << " bytes for  input buffer " << endl;
  cout << "\n allocated " << num_of_images*outSize << " bytes for output buffer " << endl;


  /////////////////////////////////////////////////////////////////////////////////////////////
  // split dummy images in chunks, each chunks for its own thead
  int8_t  *imagesInput0 = imageInputs+ inSize*(num_threads==1 ? 0*num_images_x_thread : 0);
  int8_t  *imagesInput1 = imageInputs+ inSize*(num_threads==2 ? 1*num_images_x_thread : 0);
  int8_t  *imagesInput2 = imageInputs+ inSize*(num_threads==3 ? 2*num_images_x_thread : 0);
  int8_t  *imagesInput3 = imageInputs+ inSize*(num_threads==4 ? 3*num_images_x_thread : 0);
  int8_t  *imagesInput4 = imageInputs+ inSize*(num_threads==5 ? 4*num_images_x_thread : 0);
  int8_t  *imagesInput5 = imageInputs+ inSize*(num_threads==6 ? 5*num_images_x_thread : 0);

  int8_t  *FCResult0    = FCResult+   outSize*(num_threads==1 ? 0*num_images_x_thread : 0);
  int8_t  *FCResult1    = FCResult+   outSize*(num_threads==2 ? 1*num_images_x_thread : 0);
  int8_t  *FCResult2    = FCResult+   outSize*(num_threads==3 ? 2*num_images_x_thread : 0);
  int8_t  *FCResult3    = FCResult+   outSize*(num_threads==4 ? 3*num_images_x_thread : 0);
  int8_t  *FCResult4    = FCResult+   outSize*(num_threads==5 ? 4*num_images_x_thread : 0);
  int8_t  *FCResult5    = FCResult+   outSize*(num_threads==6 ? 5*num_images_x_thread : 0);

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MULTITHREADING DPU EXECUTION WITH BATCH
  thread workers[num_threads];

  //t_start = system_clock::now();
  auto dpu_t1 = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < num_threads; i++)
    {
        if (i == 0) workers[i] = thread(runDPU, runner.get(),   ref(imagesInput0), ref(FCResult0) );
        if (i == 1) workers[i] = thread(runDPU, runner1.get(),  ref(imagesInput1), ref(FCResult1) );
        if (i == 2) workers[i] = thread(runDPU, runner2.get(),  ref(imagesInput2), ref(FCResult2) );
        if (i == 3) workers[i] = thread(runDPU, runner3.get(),  ref(imagesInput3), ref(FCResult3) );
        if (i == 4) workers[i] = thread(runDPU, runner4.get(),  ref(imagesInput4), ref(FCResult4) );
        if (i == 5) workers[i] = thread(runDPU, runner5.get(),  ref(imagesInput5), ref(FCResult5) );
    }
    // Release thread resources.
    for (auto &w : workers) {
      if (w.joinable()) w.join();
    }

    auto dpu_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> dpu_time = dpu_t2 - dpu_t1 - avg_calibr_highres;
    cout << "\n" << endl;
    double dpu_tot_time = (double) dpu_time.count();
    double dpu_avg_time = (dpu_tot_time*1000000.0)/num_of_images; //us
    double dpu_avg_fps  = (num_of_images*1000000.0)/dpu_tot_time;
    cout << "[DPU tot Time ] " << dpu_tot_time  << "us" << endl;
    cout << "[DPU avg Time ] " << dpu_avg_time  << "us" << endl;
    cout << "[DPU avg FPS  ] " << dpu_avg_fps   << endl;
    cout << "\n" << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////

    cout << "deleting memory buffer" << endl;
    delete[] imageInputs;
    delete[] FCResult;

    return 0;
  }
