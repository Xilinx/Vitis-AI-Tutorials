/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Author: Daniele Bagni, Xilinx Inc.
// date: 24 Sept 2021


#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;


GraphInfo shapes;

// const string baseImagePath = "./src/img_test/";
string baseImagePath;  // they will get their values via argv[]

int num_threads = 0;
int is_running_0 = 1;
int num_of_images = 0;
int num_images_x_thread = 0;

int NUM_TEST_IMAGES = 200;


uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64,  35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0,  0,   0,   60,  80,  0,   11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

static int8_t op_move(uint8_t i) {
  int8_t ret = int8_t(i - 128);
  return ret >> 1;
}


void ListImages(string const &path, vector<string> &images_list) {
  images_list.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images_list.push_back(name);
      }
    }
  }

  closedir(dir);
}

void runCNN(vart::Runner *runner, int8_t *imageInputs, int8_t *FCResult)
{
  // get in/out tensors and dims
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_dims();
  auto in_dims = inputTensors[0]->get_dims();
  // get shape info
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  int8_t *loc_imageInputs = imageInputs;
  int8_t *loc_FCResult = FCResult;

  cout << "inside RUN CNN " << endl;

  for (unsigned int n = 0; n < num_images_x_thread; n += batchSize)  // this works correctly for either batchSize= 1 or 3
  {
    loc_imageInputs = imageInputs + n * inSize;
    loc_FCResult = FCResult + n * outSize;

    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(loc_imageInputs, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>( loc_FCResult, batchTensors.back().get()));

    // tensor buffer input/output
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    // run
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);

    inputs.clear();
    outputs.clear();
  }
}

/**
 * @brief Entry for running FCN8 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy FCN8 on DPU platform.
 *
 */
int main(int argc, char *argv[]) {

  // Check args
  if (argc != 5) {
    cout << "Usage: run_cnn xmodel_path test_images_path thread_num (from 1 to "
            "6) use_post_proc(1:yes, 0:no)"
         << endl;
    return -1;
  }
  baseImagePath =
      std::string(argv[2]);  // path name of the folder with test images
  num_threads = atoi(argv[3]);
  assert((num_threads <= 6) & (num_threads >= 1));
  int use_post_processing = atoi(argv[4]);

  for (int i=0; i<argc; i++) cout << argv[i] << " "; cout << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPARE DPU STUFF

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  // create runners
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
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
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;

  int out_fixpos = (outputTensors[0])->template get_attr<int>("fix_point");
  auto out_fix_scale = std::exp2f(1.0f * (float)out_fixpos);
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int in_fixpos = (inputTensors[0])->template get_attr<int>("fix_point");
  auto in_fix_scale = std::exp2f(1.0f * (float)in_fixpos);
  int batchSize = in_dims[0];
  int num_of_classes = outSize / (outHeight * outWidth);

  float input_scale = get_input_scale(inputTensors[0]);
  float output_scale = get_output_scale(outputTensors[0]);

  // debug messages
  cout << "outSize  " << outSize << endl;
  cout << "inSize   " << inSize << endl;
  cout << "outW     " << outWidth << endl;
  cout << "outH     " << outHeight << endl;
  cout << "inpW     " << inWidth << endl;
  cout << "inpH     " << inHeight << endl;
  cout << "# class  " << num_of_classes << endl;
  cout << "batchSize " << batchSize << endl;  // alway 1 for Edge

  cout << "in_fixpos     " << in_fixpos << endl;
  cout << "in_fix_scale  " << in_fix_scale << endl;
  cout << "inputt_scale  " << input_scale << endl;
  cout << "out fix scale " << out_fix_scale << endl;
  cout << "output_scale "  << output_scale << endl;


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
      //if (i%20 ==0) cout << "[Timers calibration  ] " << fp_us.count() << "us" << endl;
    }
  cout << "[average calibration high resolution clock] " << avg_calibr_highres.count() / num_of_trials << "us"  << endl;
  cout << "\n" << endl;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // MEMORY ALLOCATION

  // Load all image filenames
  vector<string> image_filename;
  ListImages(baseImagePath, image_filename);
  if (image_filename.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    exit(-1);
  } else {
    num_of_images = image_filename.size();
  }

  if (num_of_images > NUM_TEST_IMAGES) num_of_images = NUM_TEST_IMAGES;
  cout << "\n max num of images to read " << num_of_images << endl;

  // number of images per thread
  num_images_x_thread = num_of_images / num_threads;
  num_images_x_thread = (num_images_x_thread / batchSize) * batchSize;
  cout << "\n number of images per thread: " << num_images_x_thread << endl;
  // effective number of images as multiple of num_threads and batchSize
  num_of_images = num_images_x_thread * num_threads;

  // memory allocation
  vector<Mat> imagesList;
  Mat segMat(outHeight, outWidth, CV_8UC3);
  Mat showMat(outHeight, outWidth, CV_8UC3);
  Mat image = cv::Mat(inHeight, inWidth, CV_8UC3);

  int8_t *imageInputs = new int8_t[(num_of_images)*inSize];
  int8_t *FCResult    = new int8_t[(num_of_images)*outSize];


  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPROCESSING ALL IMAGES AT ONCE

  auto pre_t1 = std::chrono::high_resolution_clock::now();

  for (unsigned int n = 0; n < num_of_images; n++)
  {
      image = imread(baseImagePath + image_filename[n]);
      //cout << "\n reading " << image_filename[n] << endl;
      imagesList.push_back(image);
      /*
        char s[20]; sprintf(s, "inp_%03d", n);
        cv::imshow(s,  image);
        cv::waitKey(1000);
        cv::destroyAllWindows();
        cv::imwrite(format("inp_%03d.png",n), image);
        cout << "\n writing " << format("inp_%03d.png",n) << endl;
      */
  }
  for (unsigned int n = 0; n < num_of_images; n++)
  {
      image = imagesList[n];

      for (int y = 0; y < inHeight; y++) {
	       for (int x = 0; x < inWidth; x++) {
            for (int c = 0; c < 3; c++) {
	             float tmp_pix = ((float) image.at<Vec3b>(y,x)[c])/127.5 -1.0;
               tmp_pix = tmp_pix * input_scale;
               imageInputs[n*inSize + 3*(y*inWidth+x) + c  ] = (int8_t) tmp_pix; //BGR format
	           //imageInputs[n*inSize + 3*(y*inWidth+x) + 2-c] = (int8_t) tmp_pix; //RGB format
            }
	       }
      }
      //cv::imshow(format("list_%03d.png",n),  imagesList[n]);
      //cv::waitKey(1000);
      //cv::destroyAllWindows();
  }

  auto pre_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> prepr_time = pre_t2 - pre_t1 - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[PREPROC  Time ] " << prepr_time.count() << "us" << endl;
  //cout << "[PREPROC  FPS  ] " << num_of_images*1000000.0/prepr_time.count()  << endl;
  cout << "\n" << endl;

  /*
  // just for debug
  imagesList.begin();
  for (unsigned int n = 0; n < num_of_images; n++)
  {
      image = imagesList[n];
      cv::imshow(format("list_%03d", n),  imagesList[n]);
      cv::waitKey(1000);
      cv::imshow(format("clone_%03d", n),  image);
      cv::waitKey(1000);
      cv::destroyAllWindows();
      cv::imwrite(format("clone_%03d.png",n), image);
  }
  */

  // split images in chunks, each chunks for its own thead
  // avoid pointing to wrong memorycv::Mat> locations
  int8_t *imagesInput0 =
      imageInputs + inSize * (num_threads == 1 ? 0 * num_images_x_thread : 0);
  int8_t *imagesInput1 =
      imageInputs + inSize * (num_threads == 2 ? 1 * num_images_x_thread : 0);
  int8_t *imagesInput2 =
      imageInputs + inSize * (num_threads == 3 ? 2 * num_images_x_thread : 0);
  int8_t *imagesInput3 =
      imageInputs + inSize * (num_threads == 4 ? 3 * num_images_x_thread : 0);
  int8_t *imagesInput4 =
      imageInputs + inSize * (num_threads == 5 ? 4 * num_images_x_thread : 0);
  int8_t *imagesInput5 =
      imageInputs + inSize * (num_threads == 6 ? 5 * num_images_x_thread : 0);

  int8_t *FCResult0 =
      FCResult + outSize * (num_threads == 1 ? 0 * num_images_x_thread : 0);
  int8_t *FCResult1 =
      FCResult + outSize * (num_threads == 2 ? 1 * num_images_x_thread : 0);
  int8_t *FCResult2 =
      FCResult + outSize * (num_threads == 3 ? 2 * num_images_x_thread : 0);
  int8_t *FCResult3 =
      FCResult + outSize * (num_threads == 4 ? 3 * num_images_x_thread : 0);
  int8_t *FCResult4 =
      FCResult + outSize * (num_threads == 5 ? 4 * num_images_x_thread : 0);
  int8_t *FCResult5 =
      FCResult + outSize * (num_threads == 6 ? 5 * num_images_x_thread : 0);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // MULTITHREADING DPU EXECUTION WITH BATCH
  thread workers[num_threads];

  auto dpu_t1 = std::chrono::high_resolution_clock::now();


  for (auto i = 0; i < num_threads; i++) {
    if (i == 0)
      workers[i] =
          thread(runCNN, runner.get(), ref(imagesInput0), ref(FCResult0));
    if (i == 1)
      workers[i] =
          thread(runCNN, runner1.get(), ref(imagesInput1), ref(FCResult1));
    if (i == 2)
      workers[i] =
          thread(runCNN, runner2.get(), ref(imagesInput2), ref(FCResult2));
    if (i == 3)
      workers[i] =
          thread(runCNN, runner3.get(), ref(imagesInput3), ref(FCResult3));
    if (i == 4)
      workers[i] =
          thread(runCNN, runner4.get(), ref(imagesInput4), ref(FCResult4));
    if (i == 5)
      workers[i] =
          thread(runCNN, runner5.get(), ref(imagesInput5), ref(FCResult5));
  }
  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }

  auto dpu_t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> dpu_time = dpu_t2 - dpu_t1 - avg_calibr_highres;
  cout << "\n" << endl;
  cout << "[DPU tot Time ] " << dpu_time.count()                 << "us" << endl;
  //cout << "[DPU avg Time ] " << (dpu_time.count()/num_of_images) << "us" << endl;
  cout << "[DPU avg FPS  ] " << num_of_images*1000000.0/dpu_time.count()  << endl;
  cout << "\n" << endl;

  double total_time = 0.0;

/////////////////////////////////////////////////////////////////////////////////////////////
  // POSTPROCESSING ALL THE IMAGES AT ONCE
  if (use_post_processing == 1) {
    cout << "\n DOING POST PROCESSING\n" << endl;

    auto postpr_t1 = std::chrono::high_resolution_clock::now();

    for (unsigned int n = 0; n < num_of_images; n++) {
      // cout << "\nImage : " << image_filename[n] << endl;
      image = imagesList[n].clone();
      Mat small_img;
      cv::resize(image, small_img, showMat.size(), 0, 0, INTER_AREA);

      // cv::imshow("Segmentation", small_img);
      // cv::waitKey(1000);
      // save the image
      // cv::imwrite(image_filename[n], image);
      // cv::destroyAllWindows();

      int8_t *OutData = &FCResult[n * outSize];
      for (int row = 0; row < outHeight; row++) {
        for (int col = 0; col < outWidth; col++) {
          int ii = row * outWidth * num_of_classes +
                   col * num_of_classes;  // to map the segmented image into
                                          // colors uncomment this line
          auto max_ind =
              max_element(OutData + ii, OutData + ii + num_of_classes);
          int posit = distance(OutData + ii, max_ind);
          segMat.at<Vec3b>(row, col) =
              Vec3b(colorB[posit], colorG[posit], colorR[posit]);
        }
      }
      for (int ii = 0; ii < showMat.rows * showMat.cols * 3; ii++) {
        showMat.data[ii] = small_img.data[ii] * 0.4 + segMat.data[ii] * 0.6;
      }

      // just for debug
      if (n <= 3) {
        char s[20];
        sprintf(s, "out_%03d", n);
        // putText(image3, s, Point(10, 10), FONT_HERSHEY_PLAIN, 1.0,
        //        CV_RGB(0, 255, 0), 2.0);

        Mat dst;
        cv::hconcat(small_img, segMat, dst);  // horizontal
        // cv::imshow(s, dst);
        // cv::waitKey(1000);
        cv::imwrite(format("out_%03d.png", n), dst);
        cv::destroyAllWindows();
      }
    }
    auto postpr_t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> postpr_time = postpr_t2 - postpr_t1 - avg_calibr_highres;
    cout << "\n" << endl;
    cout << "[POSTPROC Time ] " << postpr_time.count() << "us" << endl;
    //cout << "[POSTPROC FPS  ] " << num_of_images*1000000.0/postpr_time.count()  << endl;
    cout << "\n" << endl;
    total_time =  (double) postpr_time.count();
  }

  total_time += (double) prepr_time.count();
  total_time += (double) dpu_time.count();
  //cout << "[TOTAL Computation Time (DPU+CPU)        ] " << total_time  << "us" << endl;
  cout << "[average FPS with pre- & post-processing ] " << num_of_images*1000000.0/total_time  << "us" << endl;


  /////////////////////////////////////////////////////////////////////////////////////////////

  // cout << "deleting softmax     memory" << endl;
  // delete[] softmax;
  cout << "deleting imageInputs memory" << endl;
  delete[] imageInputs;
  cout << "deleting FCResult    memory" << endl;
  delete[] FCResult;
  cout << "deleting imagesList  memory" << endl;
  imagesList.clear();

  return 0;
}
