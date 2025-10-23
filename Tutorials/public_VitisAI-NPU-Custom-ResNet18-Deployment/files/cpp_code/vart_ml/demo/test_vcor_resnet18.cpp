/* 
# ===========================================================
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License
# ===========================================================

# modified on 02 Oct. 2025
*/


#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "vart_ml_utils/parsing_combinators.h"
#include <vart_ml_runner/runner.h>
#include <vart_ml_utils/vart_ml_log.h>
#include <vart_ml_utils/vcd_stats.h>

#include "common.h"
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iomanip> // For std::setprecision

#define NOT_NATIVE 0
#define NATIVE 1
#define PHYSICAL_NATIVE 2

namespace fs = std::filesystem;

//DB: added by me
// Function to check if a string ends with a given suffix
bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
}


//DB: added by me
// Function to collect image paths and class names
std::pair<std::vector<std::string>, std::vector<std::pair<std::string, std::string>>> 
getImagesFromFolders(const std::string& input_folder) {
    std::vector<std::pair<std::string, std::string>> image_data; // Vector to hold image name and class name
    std::vector<std::string> absolute_paths; // Vector to hold absolute image paths

    // Iterate through each subdirectory in the input folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (fs::is_directory(entry)) { // Check if it's a directory
            std::string class_name = entry.path().filename().string(); // Get the class name from the folder
			//std::cout << "class name: " << class_name << std::endl;

            // Iterate through each file in the subdirectory
            for (const auto& file : fs::directory_iterator(entry)) {
                if (fs::is_regular_file(file)) { // Check if it's a file
                    std::string image_name = file.path().filename().string(); // Get the image name (without path)
                    std::string image_path = file.path().string(); // Get the absolute path

                    // Check if the file has a valid image extension
                    if (endsWith(image_name, ".png") || endsWith(image_name, ".jpg") ||
                        endsWith(image_name, ".jpeg") || endsWith(image_name, ".gif") ||
                        endsWith(image_name, ".bmp") ||
						endsWith(image_name, ".PNG") || endsWith(image_name, ".JPG") ||
 						endsWith(image_name, ".JPEG") || endsWith(image_name, ".GIF") ||
 						endsWith(image_name, ".BMP")  ) 
						{
                        absolute_paths.push_back(image_path); // Add to the absolute path vector
                        image_data.emplace_back(image_name, class_name); // Add to the vector
                    }
                }
            }
        }
    }

    return {absolute_paths, image_data}; // Return both absolute paths and image data
}

//DB: added by me
// Function to save the class names and image names to a file
void saveToFile(const std::vector<std::pair<std::string, std::string>>& image_data) {
    // Open a file in write mode
    std::ofstream ofs("vcor_val_GroundTruth.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    // Write the class name and image name to the file
    for (const auto& entry : image_data) {
        ofs << entry.first << " " << entry.second << std::endl; // Write image name and class name separated by a tab
    }

    ofs.close(); // Close the file
}

// dir must have only subdirectories which are the class name, images are below them
bool isEmptyDirWithSubdirsOnly(const fs::path& dir) 
{
    // Check if the given path is a directory
    if (!fs::is_directory(dir)) 
	{
        std::cerr << "The specified path is not a directory." << std::endl;
        return false;
    }
    bool hasFiles = false;
    for (const auto& entry : fs::directory_iterator(dir)) 
	{
        if (fs::is_regular_file(entry)) {
            // If we find a regular file, set hasFiles to true
            hasFiles = true;
            break;
        }
    }
    // If we found files, return false
    if (hasFiles) return false;
    // Now check if there are subdirectories
    for (const auto& entry : fs::directory_iterator(dir)) 
	{
        if (fs::is_directory(entry)) {
            // If we find any subdirectory, return true
            return true;
        }
    }
    // If no subdirectories are found, return false
    return false;
}


//DB: added by me
std::vector<std::string> main_getImagesFromFolders(const std::string& input_folder) 
{
    //std::string input_folder = "path/to/val"; // Change this to your actual folder path
	std::cout << "folder with input images: " << input_folder << std::endl;	
	if (isEmptyDirWithSubdirsOnly(input_folder) == true)
	{ 
		auto result = getImagesFromFolders(input_folder); // Get absolute paths and class-image pairs
		auto absolute_paths = result.first;
		auto image_data = result.second;
		// Output the image data (class name, image name) and the absolute paths to the console
    	std::cout << "Image Data (Class Name, Image Name, Absolute Path):" << std::endl;
    	for (size_t i = 0; i < image_data.size(); ++i) {
        	const std::string& class_name = image_data[i].second;
        	const std::string& image_name = image_data[i].first;
        	const std::string& path = absolute_paths[i];
        	std::cout << class_name << "\t" << image_name << "\t" << path << std::endl; // Merge the output
    	}
	    // Save the data to a text file: you need it only once!
    	//saveToFile(image_data); 
		return absolute_paths;
	}
	else
	{
		std::cout << "ERROR: The folder " << input_folder << " contains files and not subdirectories." << std::endl;	
		return std::vector<std::string>(); // this is empty 
	}

}

//DB: added by me
void printMap2(const std::map<std::string, std::string>& gold) 
{
	int cnt = 1;
    // Iterate through each key-value pair in the map
    for (const auto& pair : gold) {
        std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
		if (cnt == 10) break;
		cnt++;
    }
}


int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		usage(argv[0], "Missing arguments");
		return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE, "Missing arguments.\n");
	}
	std::cout << "now running: " << std::endl;
	for (size_t i=0; i<size_t(argc); i++)
	{
		std::cout << argv[i] << " ";
	}
	std::cout << std::endl;
	//static int dbg_cnt=0; //DB		
	struct options           options;
	std::vector<std::string> images_paths;

	int err = read_options(argc, argv, options, images_paths);
	if (err)
		return err;
	/* //DB 
	std::cout << "1) options.imgPath " << options.imgPath <<  std::endl;

	std::cout << "read_options             : images_paths" << std::endl;
	if (images_paths.empty()) {
    	std::cout << "The vector is empty." << std::endl;
	} else {
    	std::cout << "The vector is not empty." << std::endl;
	}
	for(size_t i = 0; i < images_paths.size(); ++i) 
	{
    	std::cout << images_paths[i] << std::endl;
		if (i > 5) break;
	}
	*/
	/* Check if we need to enable VART ML Runner's npu_only. */
	bool npuOnly = options.raw["useCpuSubgraphs"].empty();

	/* Check if images are provided. Otherwise, random data will be used */
	bool images_not_provided = options.raw["imgPath"].empty() && images_paths.empty();
	std::cout<< "images_not_provided=" << images_not_provided << std::endl;

	/* Create VART ML Runners. */
	std::vector<std::unique_ptr<vart::Runner>> runners;
	for (const auto& path : options.snapshots)
		runners.push_back(vart::Runner::create_runner(
		    path.c_str(), images_not_provided ? "default" : "NHWC", "default", std::nullopt, npuOnly));

	err = read_images_options(options, images_paths, runners[0]->get_batch_size());
	if (err) return err;
	
	/* //DB
	std::cout << "2) options.imgPath " << options.imgPath <<  std::endl;
	std::cout<< "read_images_options: images_paths " << std::endl;
	for(size_t i = 0; i < images_paths.size(); ++i) 
	{
		std::cout << images_paths[i] << std::endl;
		if (i > 5) break;
	}		
	*/

	size_t nbImages  = options.nbImages;
	size_t batchSize = options.batchSize;

	/* Number of images that got compared to a gold file. */
	size_t nbComparedImages = nbImages;

	float accuracy[runners.size()][2];
	memset(accuracy, 0, sizeof(accuracy));

	/* Options are applied identically to all networks. */
	bool in_isNative  = false;
	bool out_isNative = false;
	bool phyNative    = false;
	if (!options.raw["dataFormat"].empty())
	{
		if (options.raw["dataFormat"] == "phyNative")
		{
			phyNative    = true;
			in_isNative  = true;
			out_isNative = true;
		}
		else if (options.raw["dataFormat"] == "native")
		{
			in_isNative  = true;
			out_isNative = true;
		}
		else if (options.raw["dataFormat"] == "inNative")
			in_isNative = true;
		else if (options.raw["dataFormat"] == "outNative")
			out_isNative = true;
	}

	/* Use external quantization */
	bool useExternalQuant = (!options.raw["useExternalQuant"].empty());

	/* Configure multi-threading */
	bool   isMultiThread = (!options.raw["nbThreads"].empty());
	size_t nbThreads     = isMultiThread ? strtoul(options.raw["nbThreads"].c_str(), NULL, 0) : 1;

	std::string fpga_arch = "aieml";

	if (!options.raw["fpgaArch"].empty())
		fpga_arch = options.raw["fpgaArch"];

	/* Determine repeat count for input image set. */
	size_t repeatCnt = 1;
	if (options.raw.find("repeat") != options.raw.end())
	{
		std::stringstream ss(options.raw["repeat"].c_str());
		ss >> repeatCnt;
	}

	/* Create the input buffers */
	std::vector<std::vector<const vart::vart_ml_tensor_t*>> inputTensors(runners.size());
	std::vector<size_t>                                     inputCnt(runners.size());
	std::vector<std::vector<void*>>                         inbuf(runners.size());
	std::vector<std::vector<uint64_t>>                      phy_inbuf(runners.size());

	/* Create the output buffers */
	std::vector<std::vector<const vart::vart_ml_tensor_t*>> outputTensors(runners.size());
	std::vector<size_t>                                     outputCnt(runners.size());
	std::vector<std::vector<void*>>                         outbuf(runners.size());
	std::vector<std::vector<uint64_t>>                      phy_outbuf(runners.size());

	/* Those variables are used to check if all images have been stored in the ddr before the run*/
	std::vector<size_t> nb_img_in_buff(runners.size(), nbImages);
	std::vector<int>    inbuf_alloc_fail(runners.size(), false);

	/* Note: FPS will be displayed if all input images were preprocessed prior to inference execution. */
	bool display_fps = true;

	/* DDR list for inputs and outputs buffer, only used if native or physical native are enable*/
	auto ddr_in_list  = options.ddr_in_list;
	auto ddr_out_list = options.ddr_out_list;

	// Get the VCOR Images //DB 
	images_paths.clear();
	images_paths = main_getImagesFromFolders(options.imgPath);
	/*  
	std::cout<< "main_getImagesFromFolders: images_paths " << std::endl;
	for(size_t i = 0; i < images_paths.size(); ++i) 
	{
		std::cout << images_paths[i] << std::endl;
		if (i > 5) break;
	}	
	*/


	/* Configure runners */
	for (size_t m = 0; m < runners.size(); m++)
	{
		/* Set runner name if not provided */
		if (options.networks[m] == "N/A")
			options.networks[m] = runners[m]->get_model_name();

		/* Configure ddr list for physical input and output buffers */
		uint8_t nb_ddrs = runners[m]->get_nb_ddrs();
		if (ddr_in_list[m].empty())
			for (size_t i = 0; i < nb_ddrs; i++)
				ddr_in_list[m].push_back(i);
		else if (*std::max_element(ddr_in_list[m].begin(), ddr_in_list[m].end()) >= nb_ddrs)
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "DDR ID (%d) exceeds actual DDR ID range (%d)\n",
			                           *std::max_element(ddr_in_list[m].begin(), ddr_in_list[m].end()),
			                           nb_ddrs - 1);

		if (ddr_out_list[m].empty())
			for (size_t i = 0; i < nb_ddrs; i++)
				ddr_out_list[m].push_back(i);
		else if (*std::max_element(ddr_out_list[m].begin(), ddr_out_list[m].end()) >= nb_ddrs)
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "DDR ID (%d) exceeds actual DDR ID range (%d)\n",
			                           *std::max_element(ddr_out_list[m].begin(), ddr_out_list[m].end()),
			                           nb_ddrs - 1);

		if (in_isNative && (err = runners[m]->set_input_native_formats(phyNative ? PHYSICAL_NATIVE : NATIVE)))
			return err;

		if (out_isNative
		    && (err = runners[m]->set_output_native_formats(phyNative ? PHYSICAL_NATIVE : NATIVE)))
			return err;

		if ((err = runners[m]->set_input_cacheable_attribute(options.raw["setNonCacheableInput"].empty())))
			return err;

		if ((err = runners[m]->set_output_cacheable_attribute(options.raw["setNonCacheableOutput"].empty())))
			return err;

		/* Get in/out tensor */
		inputTensors[m]  = runners[m]->get_input_tensors();
		outputTensors[m] = runners[m]->get_output_tensors();

		/* Get in/out tensor size */
		inputCnt[m]  = inputTensors[m].size();
		outputCnt[m] = outputTensors[m].size();

		//DB: added by me
		{
			std::cout << " input  tensors = " <<  inputCnt[m] << std::endl;
			std::cout << "output  tensors = " << outputCnt[m] << std::endl;
			for (size_t i = 0; i < inputCnt[m]; i++)
			{
				std::cout << "  input tensor["<<m<<"] height = " <<  inputTensors[m][i]->shape[0] << std::endl;
				std::cout << "  input tensor["<<m<<"] width  = " <<  inputTensors[m][i]->shape[1] << std::endl;
				std::cout << "  input tensor["<<m<<"] chan   = " <<  inputTensors[m][i]->shape[2] << std::endl;
			}
			for (size_t i = 0; i < outputCnt[m]; i++)
			{
				std::cout << "output[" <<m<< "]tensor height = " << outputTensors[m][i]->shape[0] << std::endl;
				std::cout << "output[" <<m<< "]tensor width  = " << outputTensors[m][i]->shape[1] << std::endl;
				std::cout << "output[" <<m<< "]tensor chan   = " << outputTensors[m][i]->shape[2] << std::endl;
			}
		}				

		/* Initialize the input buffers */
		inbuf[m].resize(inputCnt[m] * nbImages, NULL);
		phy_inbuf[m].resize(inputCnt[m] * nbImages);
		for (size_t i = 0; i < inputCnt[m]; i++)
		{
			if ((in_isNative || useExternalQuant)
			    && (err = runners[m]->set_data_type(inputTensors[m][i],
			                                        runners[m]->get_native_data_type(inputTensors[m][i]))))
				return err;

			for (size_t n = 0; n < nbImages; n++)
			{
				size_t idx = n * inputCnt[m] + i;

				if (in_isNative)
				{
					/* The performance are better when the load is shared between the DDRs */
					inbuf[m][idx] = runners[m]->malloc_buffer(inputTensors[m][i]->native_size,
					                                          ddr_in_list[m][idx % ddr_in_list[m].size()]);

					if (inbuf[m][idx] == NULL)
					{
						if (isMultiThread)
							return vart_ml_log_err_msg(
							    vart_ml_error::DEVICE_DDR_MALLOC_FAILURE,
							    "Failed to allocate input buffers for all %zu images.\n"
							    "Unable to switch to pre-process-on-the-fly mode in "
							    "multi-thread run. Abort.\n",
							    nbImages);

						vart_ml_log_err_msg(vart_ml_error::DEVICE_DDR_MALLOC_FAILURE,
						                    "Failed to allocate input buffers for all %zu images. "
						                    "Switching to pre-process-on-the-fly mode.\n",
						                    nbImages);
						inbuf_alloc_fail[m] = true;
						break;
					}

					phy_inbuf[m][idx] = runners[m]->get_physical_addr(inbuf[m][idx]);
				}
				else
					inbuf[m][idx] = std::malloc(inputTensors[m][i]->nbytes);
			}

			if (inbuf_alloc_fail[m])
			{
				/* Clean-up previously allocated buffers before break */
				for (size_t n = 0; n < inputCnt[m] * nbImages; n++)
					if (inbuf[m][n] != NULL)
						runners[m]->free_buffer(inbuf[m][n]);
				break;
			}
		}

		/* If failed to allocate buffers for all input images, try only for a single batch */
		if (inbuf_alloc_fail[m])
		{
			inbuf[m].resize(inputCnt[m] * batchSize, NULL);
			phy_inbuf[m].resize(inputCnt[m] * batchSize);

			/* Inputs has to be native to reach this code*/
			for (size_t i = 0; i < inputCnt[m]; i++)
			{
				if ((err = runners[m]->set_data_type(inputTensors[m][i],
				                                     runners[m]->get_native_data_type(inputTensors[m][i]))))
					return err;

				for (size_t n = 0; n < batchSize; n++)
				{
					size_t idx = n * inputCnt[m] + i;

					/* The performance are better when the load is shared between the DDRs */
					inbuf[m][idx] = runners[m]->malloc_buffer(inputTensors[m][i]->native_size,
					                                          ddr_in_list[m][idx % ddr_in_list[m].size()]);

					if (inbuf[m][idx] == NULL)
					{
						return vart_ml_log_err_msg(
						    vart_ml_error::DEVICE_DDR_MALLOC_FAILURE,
						    "Failed to allocate input buffers for a single batch. Abort.\n");
					}
					phy_inbuf[m][idx] = runners[m]->get_physical_addr(inbuf[m][idx]);
				}
			}
			nb_img_in_buff[m] = batchSize;
			display_fps       = false;
		}

		/* Initialize the output buffers */
		outbuf[m].resize(outputCnt[m] * nbImages);
		phy_outbuf[m].resize(outputCnt[m] * nbImages, 0);
		for (size_t i = 0; i < outputCnt[m]; i++)
		{
			if ((out_isNative || useExternalQuant)
			    && (err = runners[m]->set_data_type(outputTensors[m][i],
			                                        runners[m]->get_native_data_type(outputTensors[m][i]))))
				return err;

			for (size_t n = 0; n < nbImages; n++)
			{
				size_t idx = n * outputCnt[m] + i;

				if (out_isNative)
				{
					/* The performance are better when the load is shared between the DDRs */
					outbuf[m][idx] = runners[m]->malloc_buffer(outputTensors[m][i]->native_size,
					                                           ddr_out_list[m][idx % ddr_out_list[m].size()]);

					if (outbuf[m][idx] == NULL)
						return vart_ml_log_err_msg(vart_ml_error::DEVICE_DDR_MALLOC_FAILURE,
						                           "Fail to allocate output buffer.\n");

					phy_outbuf[m][idx] = runners[m]->get_physical_addr(outbuf[m][idx]);
				}
				else
					outbuf[m][idx] = std::malloc(outputTensors[m][i]->nbytes);
			}
		}

		/* Preprocess all images here to allow repeat and/or multi-threading */
		if (nb_img_in_buff[m] == nbImages)
		{
			if (options.raw["useSnapshotGold"].empty())
			{
				err = preprocess_batch(inputTensors[m],
				                       0,
				                       nbImages,
				                       fpga_arch,
				                       in_isNative,
				                       useExternalQuant,
				                       images_paths,
				                       inbuf[m]);
				if (err)
					return err;
			}
			else
			{
				for (size_t n = 0; n < nbImages; n += batchSize)
					for (size_t i = 0; i < inputCnt[m]; i++)
					{
						void*                gold_in;
						std::vector<float>   gold_float;
						std::vector<uint8_t> gold_uint8;
						size_t               gold_size;

						if (options.data_types[m][inputTensors[m][i]->name] == "FLOAT32")
						{
							err = readBinFile(options.goldFiles[m][inputTensors[m][i]->name][n / batchSize],
							                  gold_float);
							gold_in   = gold_float.data();
							gold_size = inputTensors[m][i]->size * sizeof(float);
						}
						else if (options.data_types[m][inputTensors[m][i]->name] == "UINT8")
						{
							err = readBinFile(options.goldFiles[m][inputTensors[m][i]->name][n / batchSize],
							                  gold_uint8);
							gold_in   = gold_uint8.data();
							gold_size = inputTensors[m][i]->size * sizeof(uint8_t);
						}
						else
							return vart_ml_log_err_msg(
							    vart_ml_error::CONFIG_INVALID_QUANTIZATION_TYPE,
							    "Gold input %s data_type %s is unsupported. Please use a valid data type.\n",
							    inputTensors[m][i]->name.c_str(),
							    options.data_types[m][inputTensors[m][i]->name].c_str());
						if (err)
							return err;

						for (size_t b = 0; b < batchSize; b++)
							if (in_isNative)
								force_native_input(fpga_arch,
								                   runners[m]->get_shape_format(inputTensors[m][i]),
								                   runners[m]->get_native_shape_format(inputTensors[m][i]),
								                   inputTensors[m][i],
								                   &((uint8_t*)gold_in)[b * gold_size],
								                   inbuf[m][(n + b) * inputCnt[m] + i]);
							else
								memcpy(inbuf[m][(n + b) * inputCnt[m] + i],
								       &((uint8_t*)gold_in)[b * gold_size],
								       gold_size);
					}
			}
		}
	}

	auto runner_batch_size = runners[0]->get_batch_size();

	/* Declaration of arrays of pointers dedicated to single-model multi-thread run. */
	const void* tmp_inbuf_ptr[nbThreads][inputCnt[0] * runner_batch_size];
	void*       tmp_outbuf_ptr[nbThreads][outputCnt[0] * runner_batch_size];

	std::pair<uint32_t, int> job_id[nbThreads];
	size_t                   thread_idx = 0;

	/* Declaration of arrays of pointers dedicated to multi-model run. */
	const void** mm_tmp_inbuf_ptr[runners.size()];
	void**       mm_tmp_outbuf_ptr[runners.size()];

	for (size_t m = 0; m < runners.size(); m++)
	{
		mm_tmp_inbuf_ptr[m]  = new const void*[inputCnt[m] * runner_batch_size];
		mm_tmp_outbuf_ptr[m] = new void*[outputCnt[m] * runner_batch_size];
	}

	std::pair<uint32_t, int> mm_job_id[runners.size()];

	/* Execution loop */
	const auto start     = std::chrono::high_resolution_clock::now();
	auto       elapsed_s = std::chrono::duration<double>(0);

	int8_t execute_loop_id = vcd_register_global_event("execute_loop");
	for (size_t r = 0; r < repeatCnt; r++)
	{
		for (size_t n = 0; n < nbImages; n += batchSize)
		{
			vcd_event(execute_loop_id, 1);

			if (n + batchSize > nbImages)
				batchSize = nbImages - n;

			if (batchSize > runner_batch_size)
			{
				return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
				                           "given batchSize %ld is higher than the snapshot batchSize %ld.\n",
				                           batchSize,
				                           runner_batch_size);
			}

			/* Multi-model or single-model/single-thread run. */
			if (runners.size() > 1 || !isMultiThread)
			{
				for (size_t m = 0; m < runners.size(); m++)
				{
					size_t img_idx_offset = 0;
					if (nb_img_in_buff[m] < nbImages)
						/* Preprocess batch of input images here */
						err = preprocess_batch(inputTensors[m],
						                       n,
						                       batchSize,
						                       fpga_arch,
						                       in_isNative,
						                       useExternalQuant,
						                       images_paths,
						                       inbuf[m]);
					else
						/* bump index */
						img_idx_offset = n;

					if (batchSize == runner_batch_size)
					{
						if (phyNative)
						{
							if (runners.size() > 1)
								job_id[m] = runners[m]->execute_async(
								    &(phy_inbuf[m].data()[img_idx_offset * inputCnt[m]]),
								    &(phy_outbuf[m].data()[n * outputCnt[m]]));
							else
								err =
								    runners[m]->execute(&(phy_inbuf[m].data()[img_idx_offset * inputCnt[m]]),
								                        &(phy_outbuf[m].data()[n * outputCnt[m]]));
						}
						else
						{
							if (runners.size() > 1)
								job_id[m] = runners[m]->execute_async(
								    (const void**)&inbuf[m][img_idx_offset * inputCnt[m]],
								    &outbuf[m][n * outputCnt[m]]);
							else
								err =
								    runners[m]->execute((const void**)&inbuf[m][img_idx_offset * inputCnt[m]],
								                        &outbuf[m][n * outputCnt[m]]);
						}
						if (err)
						{
							std::cout << red << "[TEST_ERROR] " << options.networks[m]
							          << " failed during execute." << reset << std::endl;
							goto clean;
						}
					}
					else
					{
						/* Fill the array of pointers. */
						for (size_t b = 0; b < runner_batch_size; b++)
							if (b < batchSize)
							{
								for (size_t i = 0; i < inputCnt[m]; i++)
									if (phyNative)
										mm_tmp_inbuf_ptr[m][b * inputCnt[m] + i] =
										    (void*)phy_inbuf[m][(b + img_idx_offset) * inputCnt[m] + i];
									else
										mm_tmp_inbuf_ptr[m][b * inputCnt[m] + i] =
										    inbuf[m][(b + img_idx_offset) * inputCnt[m] + i];

								for (size_t i = 0; i < outputCnt[m]; i++)
									if (phyNative)
										mm_tmp_outbuf_ptr[m][b * outputCnt[m] + i] =
										    (void*)phy_outbuf[m][(b + n) * outputCnt[m] + i];
									else
										mm_tmp_outbuf_ptr[m][b * outputCnt[m] + i] =
										    outbuf[m][(b + n) * outputCnt[m] + i];
							}
							else
							{
								for (size_t i = 0; i < inputCnt[m]; i++)
									mm_tmp_inbuf_ptr[m][b * inputCnt[m] + i] = NULL;
								for (size_t i = 0; i < outputCnt[m]; i++)
									mm_tmp_outbuf_ptr[m][b * outputCnt[m] + i] = NULL;
							}

						if (phyNative)
						{
							if (runners.size() > 1)
								job_id[m] = runners[m]->execute_async((const uint64_t*)mm_tmp_inbuf_ptr[m],
								                                      (uint64_t*)mm_tmp_outbuf_ptr[m]);
							else
								err = runners[m]->execute((const uint64_t*)mm_tmp_inbuf_ptr[m],
								                          (uint64_t*)mm_tmp_outbuf_ptr[m]);
						}
						else
						{
							if (runners.size() > 1)
								job_id[m] =
								    runners[m]->execute_async(mm_tmp_inbuf_ptr[m], mm_tmp_outbuf_ptr[m]);
							else
								err = runners[m]->execute(mm_tmp_inbuf_ptr[m], mm_tmp_outbuf_ptr[m]);
						}
						if (err)
						{
							std::cout << red << "[TEST_ERROR] " << options.networks[m]
							          << " failed during execute." << reset << std::endl;
							goto clean;
						}
					}
				}

				if (runners.size() > 1)
					for (size_t m = 0; m < runners.size(); m++)
					{
						err = runners[m]->wait(mm_job_id[m].first, -1);
						if (err)
						{
							std::cout << red << "[TEST_ERROR] " << options.networks[m]
							          << " failed during execute." << reset << std::endl;
							goto clean;
						}
					}

				if (display_fps)
				{
					const auto curr = std::chrono::high_resolution_clock::now();
					elapsed_s       = std::chrono::duration<double>(curr - start);

					std::cout << "\t" << std::setw(8) << std::setprecision(2) << std::setfill(' ')
					          << std::fixed
					          << (n + batchSize + r * nbImages) * runners.size() / elapsed_s.count()
					          << " imgs/s. (" << std::setw(std::to_string(nbImages).size())
					          << (n + batchSize + r * nbImages) << " images)\r" << std::flush;
				}
			}
			else /* Single-model/multi-thread run. */
			{
				if (batchSize == runner_batch_size)
				{
					if (phyNative)
						job_id[thread_idx] =
						    runners[0]->execute_async(&(phy_inbuf[0].data()[n * inputCnt[0]]),
						                              &(phy_outbuf[0].data()[n * outputCnt[0]]));
					else
						job_id[thread_idx] = runners[0]->execute_async(
						    (const void**)&inbuf[0][n * inputCnt[0]], &outbuf[0][n * outputCnt[0]]);
				}
				else
				{
					/* Fill the array of pointers. */
					for (size_t b = 0; b < runner_batch_size; b++)
						if (b < batchSize)
						{
							for (size_t i = 0; i < inputCnt[0]; i++)
								if (phyNative)
									tmp_inbuf_ptr[thread_idx][b * inputCnt[0] + i] =
									    (void*)phy_inbuf[0][(b + n) * inputCnt[0] + i];
								else
									tmp_inbuf_ptr[thread_idx][b * inputCnt[0] + i] =
									    inbuf[0][(b + n) * inputCnt[0] + i];

							for (size_t i = 0; i < outputCnt[0]; i++)
								if (phyNative)
									tmp_outbuf_ptr[thread_idx][b * outputCnt[0] + i] =
									    (void*)phy_outbuf[0][(b + n) * outputCnt[0] + i];
								else
									tmp_outbuf_ptr[thread_idx][b * outputCnt[0] + i] =
									    outbuf[0][(b + n) * outputCnt[0] + i];
						}
						else
						{
							for (size_t i = 0; i < inputCnt[0]; i++)
								tmp_inbuf_ptr[thread_idx][b * inputCnt[0] + i] = NULL;
							for (size_t i = 0; i < outputCnt[0]; i++)
								tmp_outbuf_ptr[thread_idx][b * outputCnt[0] + i] = NULL;
						}

					if (phyNative)
						job_id[thread_idx] =
						    runners[0]->execute_async((const uint64_t*)tmp_inbuf_ptr[thread_idx],
						                              (uint64_t*)tmp_outbuf_ptr[thread_idx]);
					else
						job_id[thread_idx] =
						    runners[0]->execute_async(tmp_inbuf_ptr[thread_idx], tmp_outbuf_ptr[thread_idx]);
				}

				thread_idx++;

				// Once all threads are spawned, wait for completion.
				if (thread_idx == nbThreads || (n + batchSize) == nbImages)
				{
					err = runners[0]->wait(-1, -1);
					if (err)
					{
						std::cout << red << "[TEST_ERROR] " << options.networks[0]
						          << " failed during execute." << reset << std::endl;
						goto clean;
					}

					const auto curr = std::chrono::high_resolution_clock::now();
					elapsed_s       = std::chrono::duration<double>(curr - start);

					std::cout << "\t" << std::setw(8) << std::setprecision(2) << std::setfill(' ')
					          << std::fixed << (n + batchSize + r * nbImages) / elapsed_s.count()
					          << " imgs/s. (" << std::setw(std::to_string(nbImages).size())
					          << (n + batchSize + r * nbImages) << " images)\r" << std::flush;

					vcd_event(execute_loop_id, 0);

					thread_idx = 0;
				}
			}
		}
	}

	if (!options.raw["useSnapshotGold"].empty())
	{
		for (size_t n = 0; n < nbImages; n += batchSize)
			for (size_t m = 0; m < runners.size(); m++)
				for (size_t i = 0; i < outputCnt[m]; i++)
				{
					std::string          gold_data_type = options.data_types[m][outputTensors[m][i]->name];
					std::vector<float>   gold_float;
					std::vector<uint8_t> gold_uint8;

					void*                results;
					std::vector<float>   results_float;
					std::vector<uint8_t> results_uint8;
					size_t               results_size;

					if (gold_data_type == "FLOAT32")
					{
						err = readBinFile(options.goldFiles[m][outputTensors[m][i]->name][n / batchSize],
						                  gold_float);
						results_float.resize(gold_float.size());
						results      = results_float.data();
						results_size = outputTensors[m][i]->size * sizeof(float);
					}
					else if (gold_data_type == "UINT8")
					{
						err = readBinFile(options.goldFiles[m][outputTensors[m][i]->name][n / batchSize],
						                  gold_uint8);
						results_uint8.resize(gold_uint8.size());
						results      = results_uint8.data();
						results_size = outputTensors[m][i]->size * sizeof(uint8_t);
					}
					else
						return vart_ml_log_err_msg(
						    vart_ml_error::CONFIG_INVALID_QUANTIZATION_TYPE,
						    "Gold output %s data_type %s is unsupported. Please use a valid data type.\n",
						    outputTensors[m][i]->name.c_str(),
						    gold_data_type.c_str());

					for (size_t b = 0; b < batchSize; b++)
						if (out_isNative)
							force_native_output(fpga_arch,
							                    runners[m]->get_shape_format(outputTensors[m][i]),
							                    runners[m]->get_native_shape_format(outputTensors[m][i]),
							                    outputTensors[m][i],
							                    outbuf[m][(b + n) * outputCnt[m] + i],
							                    &((uint8_t*)results)[b * results_size]);
						else
							memcpy(&((uint8_t*)results)[b * results_size],
							       outbuf[m][(b + n) * outputCnt[m] + i],
							       outputTensors[m][i]->size * sizeof(float));

					if ((gold_data_type == "FLOAT32" && results_float == gold_float)
					    || (gold_data_type == "UINT8" && results_uint8 == gold_uint8))
					{
						accuracy[m][0] += batchSize;
						accuracy[m][1] += batchSize;
					}
					else
						vart_ml_log(LOG_INFO,
						            "%sMismatch at image %ld for output %s of model %s%s\n",
						            red.c_str(),
						            n / batchSize,
						            outputTensors[m][i]->name.c_str(),
						            runners[m]->get_model_name().c_str(),
						            reset.c_str());
				}

		std::cout << std::endl;
		std::cout << "============================================================" << std::endl;
		std::cout << "Accuracy Summary:" << std::endl;
		for (size_t m = 0; m < runners.size(); m++)
		{
			print_accuracy_summary_headless(
			    runners[m]->get_model_name() + ' ', accuracy[m], nbImages * outputCnt[m]);

			if (accuracy[m][0] != nbImages * outputCnt[m])
			{
				vart_ml_log(LOG_INFO,
				            "[AMD] %s[TEST_ERROR]: %s: found mismatch with gold.%s\n",
				            red.c_str(),
				            runners[m]->get_model_name().c_str(),
				            reset.c_str());
				err++;
				break;
			}
		}
	}
	// Without the categories or the gold, the accuracy can not be calculated.
	else if (!images_paths.empty() && !options.categories.empty() && !options.gold.empty())
	{
		for (size_t m = 0; m < runners.size(); m++)
		{
			batchSize = options.batchSize;
			for (size_t n = 0; n < nbImages; n += batchSize)
			{
				if (n + batchSize > nbImages)
					batchSize = nbImages - n;

				std::vector<std::vector<float>> unquant_buf_batch(outputCnt[m]);
				std::vector<std::vector<float>> results;
				for (size_t i = 0; i < outputCnt[m]; i++)
				{
					for (size_t b = 0; b < batchSize; b++)
					{
						std::vector<float> unquant_buf;

						if (runners[m]->get_data_type(outputTensors[m][i]) == vart::DataType::FLOAT32)
							unquant_buf = { (float*)outbuf[m][(b + n) * outputCnt[m] + i],
								            (float*)outbuf[m][(b + n) * outputCnt[m] + i]
								                + outputTensors[m][i]->size };
						else if (runners[m]->get_data_type(outputTensors[m][i]) == vart::DataType::INT8)
							for (size_t j = 0; j < outputTensors[m][i]->size; j++)
								unquant_buf.push_back(((int8_t*)outbuf[m][(b + n) * outputCnt[m] + i])[j]
								                      / outputTensors[m][i]->coeff);
						else if (runners[m]->get_data_type(outputTensors[m][i]) == vart::DataType::UINT8)
							for (size_t j = 0; j < outputTensors[m][i]->size; j++)
								unquant_buf.push_back(((uint8_t*)outbuf[m][(b + n) * outputCnt[m] + i])[j]
								                      / outputTensors[m][i]->coeff);
						else
							return vart_ml_log_err_msg(
							    vart_ml_error::CONFIG_INVALID_QUANTIZATION_TYPE,
							    "Output %ld data_type is unsupported. Please use a valid data type.\n",
							    i);

						unquant_buf_batch[i].insert(unquant_buf_batch[i].end(),
						                            unquant_buf.begin(),
						                            unquant_buf.begin() + outputTensors[m][i]->size);
					}

					softmax(unquant_buf_batch[i]);

					for (size_t b = 0; b < batchSize; b++)
						results.push_back(std::vector<float>(
						    unquant_buf_batch[i].begin() + b * outputTensors[m][i]->size,
						    unquant_buf_batch[i].begin() + (b + 1) * outputTensors[m][i]->size));
				}

				compare_gold(options,
				             batchSize,
				             n,
				             images_paths,
				             results,
				             nbComparedImages,
				             accuracy[m],
				             options.networks[m]);
			}
		}

		if (nbComparedImages > 0)
		{
			std::cout << std::endl;
			std::cout << "============================================================" << std::endl;
			std::cout << "Accuracy Summary:" << std::endl;

			for (size_t m = 0; m < runners.size(); m++)
				print_accuracy_summary_headless(options.networks[m] + ' ', accuracy[m], nbComparedImages);
		}
	}

	if (in_isNative || out_isNative)
	{
		std::cout << "[AMD] VART ML runner data format was set to ";
		if (in_isNative && out_isNative)
			std::cout << (phyNative ? "PHYSICAL NATIVE." : "NATIVE.");
		else
		{
			std::cout << (in_isNative ? "IN-NATIVE." : "");
			std::cout << (out_isNative ? "OUT-NATIVE." : "");
		}
		std::cout << std::endl;
	}

	if (display_fps)
		std::cout << "[AMD] " << (repeatCnt * nbImages) * runners.size() / elapsed_s.count() << " imgs/s ("
		          << repeatCnt * nbImages << " images)" << std::endl;
	else
		std::cout << "[AMD] Unable to report imgs/s performance using pre-process-on-the-fly mode."
		          << std::endl;

	if (isMultiThread)
		std::cout << "WARNING: Multi-threaded execution (" << nbThreads << ") "
		          << "may have harmed statistics in the performance summary." << std::endl;

// Clean-up
clean:
	for (size_t m = 0; m < runners.size(); m++)
	{
		for (size_t i = 0; i < inputCnt[m]; i++)
			for (size_t n = 0; n < nb_img_in_buff[m]; n++)
				if (in_isNative)
					runners[m]->free_buffer(inbuf[m][n * inputCnt[m] + i]);
				else
					std::free(inbuf[m][n * inputCnt[m] + i]);

		for (size_t i = 0; i < outputCnt[m]; i++)
			for (size_t n = 0; n < nbImages; n++)
				if (out_isNative)
					runners[m]->free_buffer(outbuf[m][n * outputCnt[m] + i]);
				else
					std::free(outbuf[m][n * outputCnt[m] + i]);

		delete[] mm_tmp_inbuf_ptr[m];
		delete[] mm_tmp_outbuf_ptr[m];
	}

	return err;
}
