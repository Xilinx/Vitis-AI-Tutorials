/* 
===========================================================
Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
MIT License
===========================================================
 
last change: 30 Sep. 2025
*/

#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip> // Required for std::fixed, std::setw, std::setprecision //DB
#include <math.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "vart_ml_runner/runner.h"
#include "vart_ml_utils/vart_ml_log.h"

static int COLORFMT = ColorFormat::BGR; //defaul value (0)

#define DEFAULT_NB_BATCH 10

static std::pair<std::string, std::vector<int>> resizeType;

static std::vector<size_t> channelOrder;

static std::vector<float> MEAN;
static std::vector<float> STD;

bool isDirectory(const char* dir)
{
	DIR* d = nullptr;

	d = opendir(dir);
	if (!d)
		return false;

	closedir(d);
	return true;
}

bool isFile(const char* path, const char* mode)
{
	FILE* f = nullptr;

	f = fopen(path, mode);
	if (!f)
		return false;

	fclose(f);
	return true;
}

static int channelIdx(char channel)
{
	static const char channels[] = { 'B', 'G', 'R' };

	auto ptr = std::find(channels, channels + 3, channel);

	// Getting index from pointer
	return ptr - channels;
}

static void panScan(const cv::Mat& inputImage, cv::Mat& outputImage, int height, int width, int resize)
{
	// Get image size
	int img_h, img_w;
	img_h = static_cast<int>(inputImage.rows);
	img_w = static_cast<int>(inputImage.cols);

	// Compute new size
	int h, w, newHeight, newWidth;
	h         = (resize == 0) ? height : resize;
	w         = (resize == 0) ? width : resize;
	newHeight = std::max(img_h * w / img_w, img_h * h / img_h);
	newWidth  = std::max(img_w * w / img_w, img_w * h / img_h);

	cv::Mat resizedImage;
	/* Resize shortest, this line triggers error messages about Xilinx XRT, disregard them */
	cv::resize(inputImage, resizedImage, cv::Size(newWidth, newHeight));

	// Crop longest
	int      x = (resizedImage.cols - width) / 2;
	int      y = (resizedImage.rows - height) / 2;
	cv::Rect cropRegion(x, y, width, height);

	outputImage = resizedImage(cropRegion).clone();
}

static void listDdr(std::string& list_str, std::vector<std::vector<uint8_t>>& list)
{
	if (!list_str.empty())
	{
		while (list_str.find('+') != std::string::npos)
		{
			std::string          sub_list_str = list_str.substr(0, list_str.find('+'));
			std::vector<uint8_t> sub_list;
			while (sub_list_str.find(':') != std::string::npos)
			{
				sub_list.push_back(
				    static_cast<uint8_t>(std::stoi(sub_list_str.substr(0, sub_list_str.find(':')))));
				sub_list_str.erase(0, sub_list_str.find(':') + 1);
			}

			if (!sub_list_str.empty())
				sub_list.push_back(static_cast<uint8_t>(std::stoi(sub_list_str)));

			list.push_back(sub_list);
			list_str.erase(0, list_str.find('+') + 1);
		}

		if (!list_str.empty())
		{
			std::vector<uint8_t> sub_list;
			while (list_str.find(':') != std::string::npos)
			{
				sub_list.push_back(static_cast<uint8_t>(std::stoi(list_str.substr(0, list_str.find(':')))));
				list_str.erase(0, list_str.find(':') + 1);
			}

			if (!list_str.empty())
				sub_list.push_back(static_cast<uint8_t>(std::stoi(list_str)));

			list.push_back(sub_list);
		}
	}
}

void usage(std::string cmd, std::string reason)
{
	// The demo is the name of the binary executed.
	std::string demo = cmd.substr(cmd.find_last_of('/') + 1);

	if (!reason.empty())
		std::cout << cmd << ": " << reason << std::endl;

	if (demo == "multi_models_demo")
		std::cout << "Usage: " << demo << " --imagePath PATH... --snapshot PATH[+PATH]... [OPTION]..."
		          << std::endl;
	else
		std::cout << "Usage: " << demo << " --imagePath PATH... --snapshot PATH [OPTION]..." << std::endl;
	std::cout << std::endl;

	std::cout << "Mandatory arguments:" << std::endl;
	std::cout << "  --colorFmt COLORFMT   Color Format, either 1 (RGB) or 0 (BGR, the default)" << std::endl; //DB	
	if (demo == "multi_models_demo")
		std::cout << "  --snapshot PATH[+PATH]... Paths to the snapshot directories" << std::endl;
	else
		std::cout << "  --snapshot PATH         Path to the snapshot directory" << std::endl;
	std::cout << std::endl;

	std::cout << "Options:" << std::endl;
	std::cout << "  --batchSize BATCHSIZE   Size of a batch of images to process, defaults to the snapshot\n"
	             "                          batch size"
	          << std::endl;
	std::cout << "  --channelOrder          Expected order of the channels, defaults to BGR" << std::endl;
	std::cout << "  --imgPath PATH...       Either a directory or a list of images,\n"
	          << "                            if it is a directory, run on the nbImages first images\n"
	          << "                            if it is a list of images, run on them (overrides nbImages)\n"
	          << "                            if no image is given, random data will be used" << std::endl;
	std::cout << "  --goldFile PATH         Path to the file containing the gold results. If none given,\n"
	          << "                          does not perform comparison" << std::endl;
	std::cout
	    << "  --labels PATH           Path to the file containing labels of results, defaults to 'labels'"
	    << std::endl;
	std::cout << "  --mean MEAN             Mean of a pixel (depends on the framework), defaults to 0"
	          << std::endl;
	std::cout << "  --nbImages NBIMAGES     Number of images to process, defaults to " << DEFAULT_NB_BATCH
	          << " times the snapshot\n"
	             "                          batch size"
	          << std::endl;
	if (demo == "multi_models_demo")
		std::cout << "  --network NETWORK[+NETWORK]... networks to display" << std::endl;
	else
		std::cout << "  --network NETWORK       Network to display" << std::endl;
	std::cout << "  --resizeType          Type of resize to apply to the input images, defaults to PanScan"
	          << std::endl;
	std::cout << "  --std STD               Standard deviation (depends on the framework)" << std::endl;

	if (demo == "threads_demo" || demo == "vart_ml_demo")
		std::cout << "  --nbThreads nb          Number of threads to use" << std::endl;

	if (demo == "vart_ml_demo")
	{
		std::cout << "  --repeat nb             Run nb (default is 1) times the input images (for profiling\n"
		          << "                          with large amount of images)" << std::endl;
		std::cout
		    << "  --useExternalQuant      Forces app-level quantization before feed to VART ML and Forces\n"
		    << "                          app-level unquantization after retrieval from VART ML."
		    << std::endl;
		std::cout
		    << "  --dataFormat            Force input and/or output data to be uploaded to/downloaded from"
		    << std::endl
		    << "                          VART ML in native-format. Possible arguments are: " << std::endl
		    << "                            - 'native' (in and out)." << std::endl
		    << "                            - 'inNative' (input only)." << std::endl
		    << "                            - 'outNative' (output only)." << std::endl;
		std::cout
		    << "  --setNonCacheableInput  By default we skip the copy of input data. Doing so can improve\n"
		    << "                          performance assuming that input data is already located in a\n"
		    << "                          cacheable memory region. On the other hand, if this option is set\n"
		    << "                          whereas data is NOT stored in a cacheable memory region, it can\n"
		    << "                          result in performance degradation." << std::endl;
		std::cout
		    << "  --setNonCacheableOutput By default we skip the copy of output data. Doing so can improve\n"
		    << "                          performance assuming that output data is already located in a\n"
		    << "                          cacheable memory region. On the other hand, if this option is set\n"
		    << "                          whereas data is NOT stored in a cacheable memory region, it can\n"
		    << "                          result in performance degradation." << std::endl;
		std::cout
		    << "  --fpgaArch              Specify the FPGA architecture for native format transformation,\n"
		    << "                          defaults to 'aieml'" << std::endl;
		std::cout << "  --useCpuSubgraphs       Execute CPU nodes of the given model, defaults to 'False'."
		          << std::endl;
		std::cout << "  --useSnapshotGold       Use the gold files from the snapshot instead of the images\n"
		             "                          and gold file given, defaults to 'False'."
		          << std::endl;
		std::cout
		    << "  --forceInOutDdr         Specify which DDR memories will be used to allocate input and\n"
		    << "                          output buffers by passing an ordered column-seperated list\n"
		    << "                          of IDs." << std::endl;
		std::cout << "  --forceInDdr            Specify which DDR memories will be used to allocate input\n"
		          << "                          buffers by passing an ordered column-seperated list of IDs."
		          << std::endl;
		std::cout << "  --forceOutDdr           Specify which DDR memories will be used to allocate output\n"
		          << "                          buffers by passing an ordered column-seperated list of IDs."
		          << std::endl;
	}
}

int read_options(int argc, char* argv[], struct options& options, std::vector<std::string>& images_paths)
{
	std::string cmd(argv[0]);
	std::string demo = cmd.substr(cmd.find_last_of('/') + 1);

	std::set<std::string> valid_options;
	valid_options.insert("batchSize");
	valid_options.insert("channelOrder");
	valid_options.insert("goldFile");
	valid_options.insert("imgPath");
	valid_options.insert("labels");
	valid_options.insert("mean");
	valid_options.insert("nbImages");
	valid_options.insert("network");
	valid_options.insert("resizeType");
	valid_options.insert("snapshot");
	valid_options.insert("std");
	valid_options.insert("colorFmt");    //DB
	if (demo == "threads_demo" || demo == "vart_ml_demo")
		valid_options.insert("nbThreads");

	if (demo == "vart_ml_demo")
	{
		valid_options.insert("dataFormat");
		valid_options.insert("repeat");
		valid_options.insert("fpgaArch");
		valid_options.insert("setNonCacheableInput");
		valid_options.insert("setNonCacheableOutput");
		valid_options.insert("useExternalQuant");
		valid_options.insert("useCpuSubgraphs");
		valid_options.insert("useSnapshotGold");
		valid_options.insert("forceInOutDdr");
		valid_options.insert("forceInDdr");
		valid_options.insert("forceOutDdr");
	}
	if (demo == "test_vcor_resnet18") //DB
		valid_options.insert("dataFormat");

	std::map<std::string, std::string> options_map;
	std::string                        last_option;
	for (size_t i = 1; i < (size_t)argc; i++)
	{
		if (argv[i][0] == '-' and argv[i][1] == '-')
		{
			last_option = std::string(argv[i] + 2, argv[i] + strlen(argv[i]));
			if (valid_options.find(last_option) == valid_options.end())
			{
				usage(argv[0], "");
				return vart_ml_log_err_msg(
				    vart_ml_error::CONFIG_BAD_ARGUMENT, "Option %s doesn't exist\n", last_option.c_str());
			}
			if (last_option != "imgPath")
				options_map[last_option] = "true";
		}
		else if (last_option == "imgPath")
		{
			if (isDirectory(argv[i]))
				options_map[last_option] = argv[i];
			else if (isFile(argv[i], "r"))
				images_paths.push_back(argv[i]);
			else
				return vart_ml_log_err_msg(
				    vart_ml_error::CONFIG_BAD_ARGUMENT, "Unsupported file mode %s.\n", argv[i]);
		}
		else
			options_map[last_option] = argv[i];
	}

	if (!options_map["resizeType"].empty())
	{
		std::string      resize = options_map["resizeType"];
		std::vector<int> arg;

		resizeType.first = resize.substr(0, resize.find(' '));
		resize.erase(0, resize.find(' ') + 1);

		while (resize.find(' ') != std::string::npos)
		{
			resizeType.second.push_back(strtoul(resize.substr(0, resize.find(' ')).c_str(), NULL, 0));
			resize.erase(0, resize.find(' ') + 1);
		}
		if (!resize.empty())
			resizeType.second.push_back(strtoul(resize.c_str(), NULL, 0));
	}
	else
		resizeType = std::make_pair("PanScan", std::vector<int>(2, 224));

	if (!options_map["channelOrder"].empty())
		for (auto& c : options_map["channelOrder"])
			channelOrder.push_back(channelIdx(c));
	else
		channelOrder = { 0, 1, 2 };

	if (!options_map["mean"].empty())
	{
		std::string mean = options_map["mean"];
		mean.erase(std::remove(mean.begin(), mean.end(), '\''), mean.end());

		while (mean.find(' ') != std::string::npos)
		{
			MEAN.push_back(strtoul(mean.substr(0, mean.find(' ')).c_str(), NULL, 0));
			mean.erase(0, mean.find(' ') + 1);
		}
		if (!mean.empty())
			MEAN.push_back(strtoul(mean.c_str(), NULL, 0));
	}
	else
		MEAN.resize(3, 0.0);

	if (!options_map["std"].empty())
	{
		std::string std = options_map["std"];
		std.erase(std::remove(std.begin(), std.end(), '\''), std.end());

		while (std.find(' ') != std::string::npos)
		{
			STD.push_back(strtoul(std.substr(0, std.find(' ')).c_str(), NULL, 0));
			std.erase(0, std.find(' ') + 1);
		}
		if (!std.empty())
			STD.push_back(strtoul(std.c_str(), NULL, 0));

		for (auto& s : STD)
			if (s == 0)
				s = 1;
	}
	else
		STD.resize(3, 255.0);
	
	if (!options_map["colorFmt"].empty())
		set_colorFmt(strtoul(options_map["colorFmt"].c_str(), NULL, 0));
	std::cout << "preprocessing will apply " << ((COLORFMT == ColorFormat::RGB) ? "RGB" : "BGR") << " color format" << std::endl;		

	int err;
	err = read_categories(options_map["labels"], options.categories);
	if (err)
		return err;

	err = read_gold(options_map["goldFile"], options.gold);
	if (err)
		return err;

	// Check for mandatory options
	if (options_map["snapshot"].empty())
	{
		usage(argv[0], "Missing argument snapshot");
		return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE, "Missing argument snapshot");
	}

	if (options_map["snapshot"].find('+') != std::string::npos)
	{
		if (demo != "multi_models_demo" && demo != "vart_ml_demo")
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "Multi-model execution is not supported by %s. Please use "
			                           "multi_models_demo or vart_ml_demo.\n",
			                           demo.c_str());

		if (!options_map["nbThreads"].empty())
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "%s does not support 'nbThreads' option in multi-model execution.\n",
			                           demo.c_str());

		std::string snapshot_str = options_map["snapshot"];
		while (snapshot_str.find('+') != std::string::npos)
		{
			options.snapshots.push_back(snapshot_str.substr(0, snapshot_str.find('+')));

			snapshot_str.erase(0, snapshot_str.find('+') + 1);
		}
		options.snapshots.push_back(snapshot_str.substr(0, snapshot_str.find('+')));
	}
	else
		options.snapshots.push_back(options_map["snapshot"]);

	if (!options_map["network"].empty())
	{
		std::string network_str = options_map["network"];

		while (network_str.find('+') != std::string::npos)
		{
			options.networks.push_back(network_str.substr(0, network_str.find('+')));
			network_str.erase(0, network_str.find('+') + 1);
		}
		if (!network_str.empty())
			options.networks.push_back(network_str);
	}

	while (options.networks.size() < options.snapshots.size())
		options.networks.push_back("N/A");

	if (!options_map["dataFormat"].empty())
	{
    	if ( (demo != "vart_ml_demo") && (demo != "test_vcor_resnet18") ) //DB
        	return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
                                   "native dataformat is not supported by %s. Please use vart_ml_demo or test_vcor_resnet18.\n",
                                   argv[0]); //DB

    if ((options_map["dataFormat"] != "native") && (options_map["dataFormat"] != "inNative")
        && (options_map["dataFormat"] != "outNative") && (options_map["dataFormat"] != "phyNative"))
        return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
                                   "unknown dataformat argument '%s'. Valid arguments are: 'native' (in "
                                   "and out), 'inNative', 'outNative', 'phyNative'.\n",
                                   options_map["dataFormat"].c_str());
	}


	if (!options_map["setCacheableInput"].empty())
	{
		if (demo != "vart_ml_demo")
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "forcing assumption regarding cachability of the memory where input "
			                           "data is located is not supported by %s. Please use vart_ml_demo.\n",
			                           argv[0]);

		if (options_map["setCacheableInput"] != "True" && options_map["setCacheableInput"] != "False")
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "unknown argument for '%s'. Expected either 'True' or 'False'.\n",
			                           options_map["setCacheableInput"].c_str());
	}

	if (!options_map["setCacheableOutput"].empty())
	{
		if (demo != "vart_ml_demo")
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "forcing assumption regarding cachability of the memory where output "
			                           "data is located is not supported by %s. Please use vart_ml_demo.\n",
			                           argv[0]);

		if (options_map["setCacheableOutput"] != "True" && options_map["setCacheableOutput"] != "False")
			return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
			                           "unknown argument for '%s'. Expected either 'True' or 'False'.\n",
			                           options_map["setCacheableOutput"].c_str());
	}

	if (!options_map["fpgaArch"].empty() && (demo != "vart_ml_demo"))
		return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
		                           "fpgaArch is not supported by %s. Please use vart_ml_demo.\n",
		                           demo.c_str());

	if (!options_map["useSnapshotGold"].empty())
		options_map["useCpuSubgraphs"] = true;

	if (!options_map["forceInOutDdr"].empty() || !options_map["forceInDdr"].empty()
	    || !options_map["forceOutDdr"].empty())
	{
		if (demo != "vart_ml_demo")
			return vart_ml_log_err_msg(
			    vart_ml_error::TOOLS_BAD_USAGE,
			    "forcing DDR ID restriction for buffer creation is not supported by %s. "
			    "Please use vart_ml_demo.\n",
			    argv[0]);

		std::string ddr_in_list_str, ddr_out_list_str;

		if (!options_map["forceInOutDdr"].empty()
		    && (!options_map["forceInDdr"].empty() || !options_map["forceOutDdr"].empty()))
			return vart_ml_log_err_msg(
			    vart_ml_error::TOOLS_BAD_USAGE,
			    "forceInOutDdr option cannot be used simultaneously with forceInDdr or forceOutDdr.\n");
		else if (!options_map["forceInOutDdr"].empty())
			ddr_in_list_str = ddr_out_list_str = options_map["forceInOutDdr"];
		else
		{
			if (!options_map["forceInDdr"].empty())
				ddr_in_list_str = options_map["forceInDdr"];
			if (!options_map["forceOutDdr"].empty())
				ddr_out_list_str = options_map["forceOutDdr"];
		}

		listDdr(ddr_in_list_str, options.ddr_in_list);
		listDdr(ddr_out_list_str, options.ddr_out_list);
	}

	options.ddr_in_list.resize(options.snapshots.size());
	options.ddr_out_list.resize(options.snapshots.size());

	options.raw = options_map;

	return vart_ml_error::SUCCESS;
}

int read_images_options(struct options&           options,
                        std::vector<std::string>& images_paths,
                        size_t                    default_batchSize)
{
	// Parse size of batchs, defaults to snapshot batch size if none given.
	options.batchSize = (options.raw["batchSize"].empty())
	                        ? default_batchSize
	                        : strtoul(options.raw["batchSize"].c_str(), NULL, 0);

	if (options.batchSize > default_batchSize)
		return vart_ml_log_err_msg(vart_ml_error::TOOLS_BAD_USAGE,
		                           "Batch size of %ld not supported. The batch size needs to be smaller or "
		                           "equal to the snapshot batch size %ld.\n",
		                           options.batchSize,
		                           default_batchSize);

	// Parse number of images, defaults to DEFAULT_NB_BATCH * batchSize size if none given.
	options.nbImages = images_paths.empty() ? ((options.raw["nbImages"].empty())
	                                               ? options.batchSize * DEFAULT_NB_BATCH
	                                               : strtoul(options.raw["nbImages"].c_str(), NULL, 0))
	                                        : images_paths.size();

	// Not enough images to fill a full batch
	if (options.batchSize > options.nbImages)
		options.batchSize = options.nbImages;

	if (options.raw["imgPath"].empty() && images_paths.empty() && options.raw["useSnapshotGold"].empty())
		vart_ml_log(
		    LOG_WARN,
		    "No images provided. This run will use random buffers and will not have an accuracy score\n");
	options.imgPath = options.raw["imgPath"];

	// Get paths of all required images if a directory was given.
	if (!options.imgPath.empty() && images_paths.empty() && options.raw["useSnapshotGold"].empty())
		for (size_t n = 0; n < options.nbImages; n += options.batchSize)
			for (size_t b = 0; b < options.batchSize; b++)
			{
				std::stringstream num;
				num << std::setfill('0') << std::setw(8) << std::to_string(n + b + 1);
				images_paths.push_back(options.imgPath + "/ILSVRC2012_val_" + num.str() + ".JPEG");
			}
	else if (!options.raw["useSnapshotGold"].empty())
	{
		if (!options.imgPath.empty() || !options.categories.empty() || !options.gold.empty())
			vart_ml_log(LOG_WARN,
			            "useSnapshotGold is enable, imgPath, labels and goldFile will be ignored\n");

		options.goldFiles.resize(options.snapshots.size());
		options.data_types.resize(options.snapshots.size());
		for (size_t i = 0; i < options.snapshots.size(); i++)
		{
			std::string snapshot_path =
			    (options.snapshots[i].back() == '/') ? options.snapshots[i] : options.snapshots[i] + "/";

			std::vector<std::string> inputs_path;
			for (const auto& entry : std::filesystem::directory_iterator(snapshot_path + "golds/inputs"))
				inputs_path.push_back(entry.path());

			std::sort(inputs_path.begin(), inputs_path.end());

			for (auto& input_path : inputs_path)
			{
				size_t pos = (snapshot_path + "golds/inputs/").size();

				std::string input_name = input_path.substr(pos, input_path.find("_batch") - pos);
				std::string input_type = input_name.substr(input_name.find_last_of('_') + 1);

				input_name = input_name.substr(0, input_name.find_last_of('_'));

				options.data_types[i][input_name] = input_type;
				options.goldFiles[i][input_name].push_back(input_path);
			}

			std::vector<std::string> outputs_path;
			for (const auto& entry : std::filesystem::directory_iterator(snapshot_path + "golds/outputs"))
				outputs_path.push_back(entry.path());

			std::sort(outputs_path.begin(), outputs_path.end());

			for (auto& output_path : outputs_path)
			{
				size_t pos = (snapshot_path + "golds/outputs/").size();

				std::string output_name = output_path.substr(pos, output_path.find("_batch") - pos);
				std::string output_type = output_name.substr(output_name.find_last_of('_') + 1);

				output_name = output_name.substr(0, output_name.find_last_of('_'));

				options.data_types[i][output_name] = output_type;
				options.goldFiles[i][output_name].push_back(output_path);
			}

			if (options.goldFiles[i].begin()->second.size() * options.batchSize < options.nbImages)
				options.nbImages = options.goldFiles[i].begin()->second.size() * options.batchSize;
		}
	}

	return vart_ml_error::SUCCESS;
}

std::vector<std::pair<size_t, float>> topk(std::vector<float>& buf, size_t k)
{
	std::vector<std::pair<size_t, float>> tmp;

	size_t i = 0;
	for (auto x : buf)
		tmp.push_back(std::make_pair(i++, x));

	std::sort(tmp.begin(), tmp.end(), [](auto& l, auto r) { return l.second > r.second; });

	tmp.resize(k);
	return tmp;
}

int read_categories(std::string labels_file, std::vector<std::string>& categories)
{
	std::ifstream catfile(labels_file.empty() ? "labels" : labels_file);
	std::string   str;

	if (!catfile)
	{
		vart_ml_log(LOG_WARN, "No labels file provided. This run will not have an accuracy score.\n");
		return vart_ml_error::SUCCESS;
	}

	while (std::getline(catfile, str))
		categories.push_back(str);

	catfile.close();
	return vart_ml_error::SUCCESS;
}

void printMap(const std::map<std::string, std::string>& gold) //DB 
{
  // Iterate through each key-value pair in the map
  for (const auto& pair : gold) {
      std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
  }
}

int read_gold(std::string gold_file, std::map<std::string, std::string>& gold)
{
	// If there is no gold file, return an empty map.
	if (gold_file.empty())
		return vart_ml_error::SUCCESS;

	// Open the gold file.
	std::ifstream catfile(gold_file);
	if (!catfile)
		return vart_ml_log_err_msg(vart_ml_error::FILE_ACCESS_OPEN_FAILURE,
		                           "Failed to open the gold file.\n");

	// Read gold file and insert elements in gold map.
	std::string str;
	while (std::getline(catfile, str))
		gold[str.substr(0, str.find(" "))] = str.substr(str.find(" ") + 1);

	catfile.close();
	return vart_ml_error::SUCCESS;
}

void softmax(std::vector<float>& buf)
{
	double sum = 0.0f;

	for (auto& x : buf)
	{
		x = exp(x);
		sum += x;
	}

	for (auto& x : buf)
		x /= sum;
}

void set_colorFmt (int new_colFmt) //DB
{ 
	if (new_colFmt > 1)
	{
		std::cout << "ERROR: colorFmt can be only either 1 (RGB) or 0 (BGR)! Forcing it to BGR" << std::endl;
		new_colFmt = ColorFormat::BGR;
	}
	COLORFMT = (new_colFmt == 1) ? ColorFormat::RGB : ColorFormat::BGR;
}
int preprocess(float* const buf, std::string image_path, size_t height, size_t width, size_t channel)
{
	if (channel > MEAN.size() || channel > STD.size())
		return vart_ml_log_err_msg(
		    vart_ml_error::TOOLS_BAD_ARG,
		    "The preprocess provided does not correspond to the number of channel of the input.\n");

	cv::Mat preprocessed = cv::Mat(height, width, CV_8UC3);
	cv::Mat image        = cv::imread(image_path);

	if (resizeType.first == "PanScan")
	{
		if (resizeType.second.size() >= 3)
			panScan(image, preprocessed, height, width, resizeType.second[2]);
		else
			panScan(image, preprocessed, height, width, 0);
	}
	else
		return vart_ml_log_err_msg(
		    vart_ml_error::TOOLS_BAD_ARG, "Resize type %s is not supported.\n", resizeType.first.c_str());

	for (size_t h = 0; h < height; h++)
	{
		for (size_t w = 0; w < width; w++)
		{
			for (size_t c = 0; c < channel; c++)
				if (COLORFMT == ColorFormat::BGR) //DB
					buf[width * channel * h + channel * w +   c] = (preprocessed.at<cv::Vec3b>(h, w)[channelOrder[c]] - MEAN[c]) / STD[c];
				else //(COLORFMT == ColorFormat::RGB) //DB
					buf[width * channel * h + channel * w + 2-c] = (preprocessed.at<cv::Vec3b>(h, w)[channelOrder[c]] - MEAN[c]) / STD[c];
		}
	}

	return vart_ml_error::SUCCESS;
}

int preprocess(uint8_t* const buf, std::string image_path, size_t height, size_t width, size_t channel)
{
	cv::Mat preprocessed = cv::Mat(height, width, CV_8UC3);
	cv::Mat image        = cv::imread(image_path);

	if (resizeType.first == "PanScan")
	{
		if (resizeType.second.size() >= 3)
			panScan(image, preprocessed, height, width, resizeType.second[2]);
		else
			panScan(image, preprocessed, height, width, 0);
	}
	else
		return vart_ml_log_err_msg(
		    vart_ml_error::TOOLS_BAD_ARG, "Resize type %s is not supported.\n", resizeType.first.c_str());

	for (size_t h = 0; h < height; h++)
	{
		for (size_t w = 0; w < width; w++)
		{
			for (size_t c = 0; c < channel; c++)
			if (COLORFMT == ColorFormat::BGR) //DB
				buf[width * channel * h + channel * w +   c] = preprocessed.at<cv::Vec3b>(h, w)[channelOrder[c]];
			else //(COLORFMT == ColorFormat::RGB) //DB
				buf[width * channel * h + channel * w + 2-c] = preprocessed.at<cv::Vec3b>(h, w)[channelOrder[c]];
		}
	}

	return vart_ml_error::SUCCESS;
}

static int clip(int x, int min, int max)
{
	if (x >= max)
		return max;
	if (x <= min)
		return min;
	return x;
}

void force_native_input(std::string                   arch,
                        std::string                   shape_format,
                        std::string                   ddr_format,
                        const vart::vart_ml_tensor_t* in,
                        const void*                   src,
                        void*                         dst)
{
	uint8_t*            input;
	std::vector<int8_t> quant(in->size);

	if (in->coeff > 0)
	{
		for (size_t i = 0; i < in->size; i++)
			quant[i] = clip(nearbyintf(((float*)src)[i] * in->coeff), -128, 127);
		input = (uint8_t*)quant.data();
	}
	else
		input = (uint8_t*)src;

	size_t H, W, C, S;
	if (arch == "aieml")
	{
		if (ddr_format == "NHWC")
		{
			if (shape_format == "NHWC" && in->shape.size() == 4)
			{
				H = in->shape[1];
				W = in->shape[2];
				C = in->strides[2]; // C * S after padding is added by user
				S = in->strides[3]; // Data size

				// Need to be aligned with a multiple of bigPixel
				size_t nbBigPixel = (C / S + in->big_pixel - 1) / in->big_pixel;

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* src_base = input + h * in->strides[1];
					uint8_t* dst_base = (uint8_t*)dst + h * W * S * in->big_pixel * nbBigPixel;
					for (size_t w = 0; w < W; w++)
						memcpy(dst_base + w * S * in->big_pixel * nbBigPixel, src_base + w * C, C);
				}
			}
			else if (shape_format == "NCHW" && in->shape.size() == 4)
			{
				C = in->shape[1];
				H = in->shape[2];
				W = in->strides[2]; // W * S after padding is added by user
				S = in->strides[3]; // Data size
				// Need to be aligned with a multiple of bigPixel
				size_t nbBigPixel = (C + in->big_pixel - 1) / in->big_pixel;

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* src_base = input + h * W;
					uint8_t* dst_base = (uint8_t*)dst + h * W * in->big_pixel * nbBigPixel;
					for (size_t w = 0; w < W; w += S)
					{
						uint8_t* src = src_base + w;
						uint8_t* dst = dst_base + w * in->big_pixel * nbBigPixel;
						for (size_t c = 0; c < C; c++)
							memcpy(dst + c * S, src + c * in->strides[1], S);
					}
				}
			}
			else
				memcpy(dst, input, in->nbytes);
		}
		else if (ddr_format == "NCHWc")
		{
			if (shape_format == "NHWC" && in->shape.size() == 4)
			{
				H = in->shape[1];
				W = in->shape[2];
				C = in->strides[2]; // C * S after padding is added by user
				S = in->strides[3]; // Data size

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* src_base = input + h * W * C;
					uint8_t* dst_base = (uint8_t*)dst + h * W * S * in->big_pixel;
					for (size_t w = 0; w < W; w++)
					{
						uint8_t* src = src_base + w * C;
						uint8_t* dst = dst_base + w * S * in->big_pixel;
						for (size_t c = 0; c < C; c += in->big_pixel)
							memcpy(dst + c * H * W * S,
							       src + c * S,
							       std::min((int)(C / S - c), (int)in->big_pixel) * S);
					}
				}
			}
			else if (shape_format == "NCHW" && in->shape.size() == 4)
			{
				C = in->shape[1];
				H = in->shape[2];
				W = in->strides[2]; // W * S after padding is added by user
				S = in->strides[3]; // Data size

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* src_base = input + h * W;
					uint8_t* dst_base = (uint8_t*)dst + h * W * in->big_pixel;
					for (size_t w = 0; w < W; w += S)
					{
						uint8_t* src = src_base + w;
						uint8_t* dst = dst_base + w * in->big_pixel;
						for (size_t c = 0; c < C; c++)
							memcpy(dst + (c % in->big_pixel) * S
							           + (c / in->big_pixel) * H * W * in->big_pixel,
							       src + c * in->strides[1],
							       S);
					}
				}
			}
			else
				memcpy(dst, input, in->nbytes);
		}
		else
			memcpy(dst, input, in->nbytes);
	}
	else
	{
		if (shape_format == "NHWC" && in->shape.size() == 4)
		{
			H = in->shape[1];
			W = in->shape[2];
			C = in->strides[2]; // C * S after padding is added by user
			S = in->strides[3]; // Data size

			size_t nbcol = (C / S + in->big_pixel - 1) / in->big_pixel;
			// Real size after 4-byte blocks convertion
			size_t line_size = std::max((int)(in->big_pixel * W), 32) * S;
			// neurons are grouped by column of 64 pixels (16 big-pixels)
			size_t in_ddr_line_size = ((W + 15) / 16) * ((C + 3) / 4) * 64;
			// Real size of the line in DDR
			in_ddr_line_size *= in->strides[3];
			// Number of lines fitting in a single ddr line
			size_t nb = in_ddr_line_size / (line_size * nbcol);

			for (size_t h = 0; h < H; h++)
			{
				uint8_t* src_base = input + h * in->strides[1];
				uint8_t* dst_base = (uint8_t*)dst + (h / nb) * in_ddr_line_size + (h % nb) * line_size;
				for (size_t w = 0; w < W; w++)
				{
					uint8_t* src = src_base + w * C;
					uint8_t* dst = dst_base + 64 * nbcol * (w / 16) * S + (w % 16) * S * in->big_pixel;
					for (size_t c = 0; c < C; c += in->big_pixel)
						memcpy(dst + 16 * c * S,
						       src + c * S,
						       std::min((int)(C / S - c), (int)in->big_pixel) * S);
				}
			}
		}
		else if (shape_format == "NCHW" && in->shape.size() == 4)
		{
			C = in->shape[1];
			H = in->shape[2];
			W = in->strides[2]; // W * S after padding is added by user
			S = in->strides[3]; // Data size

			size_t nbcol = (C + in->big_pixel - 1) / in->big_pixel;
			// Real size after 4-byte blocks convertion
			size_t line_size = std::max((int)(in->big_pixel * W / S), 32) * S;
			// neurons are grouped by column of 64 pixels (16 big-pixels)
			size_t in_ddr_line_size = ((W + 15) / 16) * ((C + 3) / 4) * 64;
			// Real size of the line in DDR
			in_ddr_line_size *= in->strides[3];
			// Number of lines fitting in a single ddr line
			size_t nb = in_ddr_line_size / (line_size * nbcol);

			for (size_t h = 0; h < H; h++)
			{
				uint8_t* src_base = input + h * W;
				uint8_t* dst_base = (uint8_t*)dst + (h / nb) * in_ddr_line_size + (h % nb) * line_size;
				for (size_t w = 0; w < W; w += S)
				{
					uint8_t* src = src_base + w;
					uint8_t* dst = dst_base + 64 * nbcol * (w / 16) + (w % 16) * in->big_pixel;
					for (size_t c = 0; c < C; c++)
						memcpy(dst + (c % in->big_pixel) * S + (c / in->big_pixel) * S * 64,
						       src + in->strides[1],
						       S);
				}
			}
		}
		else
			memcpy(dst, input, in->nbytes);
	}
}

void force_native_output(std::string                   arch,
                         std::string                   shape_format,
                         std::string                   ddr_format,
                         const vart::vart_ml_tensor_t* out,
                         const void*                   src,
                         void*                         dst)
{
	std::vector<int8_t> quant(out->size);
	uint8_t*            output;

	if (out->coeff > 0)
		output = (uint8_t*)quant.data();
	else
		output = (uint8_t*)dst;

	size_t H, W, C, S;
	if (arch == "aieml")
	{
		if (ddr_format == "NHWC")
		{
			if (shape_format == "NHWC" && out->shape.size() == 4)
			{
				H = out->shape[1];
				W = out->shape[2];
				C = out->strides[2]; // C * S after padding is added by user
				S = out->strides[3]; // Data size
				// Need to be aligned with a multiple of bigPixel
				size_t nbBigPixel = (C / S + out->big_pixel - 1) / out->big_pixel;

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* dst_base = output + h * out->strides[1];
					uint8_t* src_base = (uint8_t*)src + h * W * S * out->big_pixel * nbBigPixel;
					for (size_t w = 0; w < W; w++)
						memcpy(dst_base + w * C, src_base + w * S * out->big_pixel * nbBigPixel, C);
				}
			}
			else if (shape_format == "NCHW" && out->shape.size() == 4)
			{
				C = out->shape[1];
				H = out->shape[2];
				W = out->strides[2]; // W * S after padding is added by user
				S = out->strides[3]; // Data size
				// Need to be aligned with a multiple of bigPixel
				size_t nbBigPixel = (C + out->big_pixel - 1) / out->big_pixel;

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* dst_base = output + h * W;
					uint8_t* src_base = (uint8_t*)src + h * W * out->big_pixel * nbBigPixel;
					for (size_t w = 0; w < W; w += S)
					{
						uint8_t* dst = dst_base + w;
						uint8_t* src = src_base + w * out->big_pixel * nbBigPixel;
						for (size_t c = 0; c < C; c++)
							memcpy(dst + c * out->strides[1], src + c * S, S);
					}
				}
			}
			else
				memcpy(output, src, out->nbytes);
		}
		else if (ddr_format == "NCHWc")
		{
			if (shape_format == "NHWC" && out->shape.size() == 4)
			{
				H = out->shape[1];
				W = out->shape[2];
				C = out->strides[2]; // C * S after padding is added by user
				S = out->strides[3]; // Data size

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* dst_base = output + h * out->strides[1];
					uint8_t* src_base = (uint8_t*)src + h * W * S * out->big_pixel;
					for (size_t w = 0; w < W; w++)
					{
						uint8_t* dst = dst_base + w * C;
						uint8_t* src = src_base + w * S * out->big_pixel;
						for (size_t c = 0; c < C; c += out->big_pixel)
							memcpy(dst + c * S,
							       src + c * H * W * S,
							       std::min((int)(C / S - c), (int)out->big_pixel) * S);
					}
				}
			}
			else if (shape_format == "NCHW" && out->shape.size() == 4)
			{
				C = out->shape[1];
				H = out->shape[2];
				W = out->strides[2]; // W * S after padding is added by user
				S = out->strides[3]; // Data size

				for (size_t h = 0; h < H; h++)
				{
					uint8_t* dst_base = output + h * W;
					uint8_t* src_base = (uint8_t*)src + h * W * out->big_pixel;
					for (size_t w = 0; w < W; w += S)
					{
						uint8_t* dst = dst_base + w;
						uint8_t* src = src_base + w * out->big_pixel;
						for (size_t c = 0; c < C; c++)
							memcpy(dst + c * out->strides[1],
							       src + (c % out->big_pixel) * S
							           + (c / out->big_pixel) * H * W * out->big_pixel,
							       S);
					}
				}
			}
			else
				memcpy(output, src, out->nbytes);
		}
		else
			memcpy(output, src, out->nbytes);
	}
	else
	{
		if (shape_format == "NHWC" && out->shape.size() == 4)
		{
			size_t H, W, C, S;
			H = out->shape[1];
			W = out->shape[2];
			C = out->strides[2]; // C * S after padding is added by user
			S = out->strides[3]; // Data size

			size_t nbcol = (C / S + out->big_pixel - 1) / out->big_pixel;
			// Real size after 4-byte blocks convertion
			size_t line_size = std::max((int)(out->big_pixel * W), 32) * S;
			// neurons are grouped by column of 64 pixels (16 big-pixels)
			size_t out_ddr_line_size = ((W + 15) / 16) * ((C + 3) / 4) * 64;
			// Real size of the line in DDR
			out_ddr_line_size *= out->strides[3];
			// Number of lines fitting in a single ddr line
			size_t nb = out_ddr_line_size / (line_size * nbcol);

			for (size_t h = 0; h < H; h++)
			{
				uint8_t* dst_base = output + h * out->strides[1];
				uint8_t* src_base = (uint8_t*)src + (h / nb) * out_ddr_line_size + (h % nb) * line_size;
				for (size_t w = 0; w < W; w++)
				{
					uint8_t* dst = dst_base + w * C;
					uint8_t* src = src_base + 64 * nbcol * (w / 16) * S + (w % 16) * S * out->big_pixel;
					for (size_t c = 0; c < C; c += out->big_pixel)
						memcpy(dst + c * S,
						       src + 16 * c * S,
						       std::min((int)(C / S - c), (int)out->big_pixel) * S);
				}
			}
		}
		else if (shape_format == "NCHW" && out->shape.size() == 4)
		{
			C = out->shape[1];
			H = out->shape[2];
			W = out->strides[2]; // W * S after padding is added by user
			S = out->strides[3]; // Data size

			size_t nbcol = (C + out->big_pixel - 1) / out->big_pixel;
			// Real size after 4-byte blocks convertion
			size_t line_size = std::max((int)(out->big_pixel * W / S), 32) * S;
			// neurons are grouped by column of 64 pixels (16 big-pixels)
			size_t out_ddr_line_size = ((W + 15) / 16) * ((C + 3) / 4) * 64;
			// Real size of the line in DDR
			out_ddr_line_size *= out->strides[3];
			// Number of lines fitting in a single ddr line
			size_t nb = out_ddr_line_size / (line_size * nbcol);

			for (size_t h = 0; h < H; h++)
			{
				uint8_t* dst_base = output + h * W;
				uint8_t* src_base = (uint8_t*)src + (h / nb) * out_ddr_line_size + (h % nb) * line_size;
				for (size_t w = 0; w < W; w += S)
				{
					uint8_t* dst = dst_base + w;
					uint8_t* src = src_base + 64 * nbcol * (w / 16) + (w % 16) * out->big_pixel;
					for (size_t c = 0; c < C; c++)
						memcpy(dst + c * out->strides[1],
						       src + (c % out->big_pixel) * S + (c / out->big_pixel) * S * 64,
						       S);
				}
			}
		}
		else
			memcpy(output, src, out->nbytes);
	}

	if (out->coeff > 0)
		for (size_t i = 0; i < out->size; i++)
			((float*)dst)[i] = quant[i] / out->coeff;
}

int preprocess_batch(std::vector<const vart::vart_ml_tensor_t*> in_tensors,
                     size_t                                     img_idx,
                     size_t                                     nb_images,
                     std::string                                fpga_arch,
                     bool                                       in_isNative,
                     bool                                       useExternalQuant,
                     std::vector<std::string>                   images_paths,
                     std::vector<void*>&                        inbuf)
{
	static bool display_loading = false;
	size_t      input_cnt       = in_tensors.size();
	int         err;

	const auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < input_cnt; i++)
		for (size_t n = 0; n < nb_images; n++)
		{
			if (!display_loading)
			{
				const auto curr      = std::chrono::high_resolution_clock::now();
				auto       elapsed_s = std::chrono::duration<double>(curr - start);

				if (elapsed_s.count() > 1)
					display_loading = true;
			}

			if (display_loading)
				std::cout << "Loading images: " << (n + i * nb_images) * 100 / (nb_images * input_cnt) + 1
				          << "%\r" << std::flush;

			size_t idx = n * input_cnt + i;

			if (images_paths.empty())
			{
				if (in_isNative)
					for (size_t j = 0; j < in_tensors[i]->native_size; j++)
						((int8_t*)inbuf[idx])[j] = rand();
				else if (in_tensors[i]->data_type == vart::DataType::FLOAT32)
					for (size_t j = 0; j < in_tensors[i]->size; j++)
						((float*)inbuf[idx])[j] = rand();
				else
					for (size_t j = 0; j < in_tensors[i]->size; j++)
						((int8_t*)inbuf[idx])[j] = rand();
			}
			else if (in_isNative)
			{
				if (in_tensors[i]->coeff > 0)
				{
					std::vector<float> input(in_tensors[i]->size);

					err = preprocess(input.data(),
					                 images_paths[img_idx + n],
					                 in_tensors[i]->shape[1],
					                 in_tensors[i]->shape[2],
					                 in_tensors[i]->shape[3]);
					if (err)
						return err;

					force_native_input(fpga_arch, "NHWC", "NHWC", in_tensors[i], input.data(), inbuf[idx]);
				}
				else
				{
					std::vector<uint8_t> input(in_tensors[i]->size);

					err = preprocess(input.data(),
					                 images_paths[img_idx + n],
					                 in_tensors[i]->shape[1],
					                 in_tensors[i]->shape[2],
					                 in_tensors[i]->shape[3]);
					if (err)
						return err;

					force_native_input(fpga_arch, "NHWC", "NHWC", in_tensors[i], input.data(), inbuf[idx]);
				}
			}
			else if (useExternalQuant)
			{
				if (in_tensors[i]->coeff <= 0)
					return vart_ml_log_err_msg(vart_ml_error::CONFIG_INVALID_QUANTIZATION_TYPE,
					                           "No external quantization available for input data type.\n");

				std::vector<float> input(in_tensors[i]->size);

				err = preprocess(input.data(),
				                 images_paths[img_idx + n],
				                 in_tensors[i]->shape[1],
				                 in_tensors[i]->shape[2],
				                 in_tensors[i]->shape[3]);
				if (err)
					return err;

				for (size_t j = 0; j < in_tensors[i]->size; j++)
					((int8_t*)inbuf[idx])[j] = clip(nearbyintf(input[j] * in_tensors[i]->coeff), -128, 127);
			}
			else
			{
				if (in_tensors[i]->data_type == vart::DataType::FLOAT32)
					err = preprocess((float*)inbuf[idx],
					                 images_paths[img_idx + n],
					                 in_tensors[i]->shape[1],
					                 in_tensors[i]->shape[2],
					                 in_tensors[i]->shape[3]);
				else
					err = preprocess((uint8_t*)inbuf[idx],
					                 images_paths[img_idx + n],
					                 in_tensors[i]->shape[1],
					                 in_tensors[i]->shape[2],
					                 in_tensors[i]->shape[3]);
				if (err)
					return err;
			}
		}
	std::cout << std::endl;

	return vart_ml_error::SUCCESS;
}

void compare_gold(struct options                  options,
                  size_t                          batchSize,
                  size_t                          first_image_index,
                  std::vector<std::string>        images_paths,
                  std::vector<std::vector<float>> results,
                  size_t&                         nbComparedImages,
                  float                           accuracy[2],
                  std::string                     network)
{
	// Without the categories or the gold, the accuracy can not be calculated.
	if (images_paths.empty() || options.categories.empty() || options.gold.empty())
	{
		nbComparedImages = 0;
		return;
	}

	for (size_t b = 0; b < batchSize; b++)
	{
		size_t result_index = first_image_index + b;

		// Get the name of the current image.
		std::string currentImg =
		    images_paths[result_index].substr(images_paths[result_index].find_last_of('/') + 1);

		std::vector<std::pair<size_t, float>> top5 = topk(results[b], 5);

		std::string color[5]   = { "" };
		auto        goldResult = options.gold.find(currentImg);
		// If gold file has no goldResult for this image, decrement compared images counter.
		if (goldResult == options.gold.end())
			--nbComparedImages;
		else if (goldResult->second == options.categories[top5[0].first])
		{
			accuracy[0]++;
			accuracy[1]++;
			color[0] = green;
		}
		else
		{
			for (size_t i = 1; i < 5; i++)
			{
				if (goldResult->second == options.categories[top5[i].first])
				{
					accuracy[1]++;
					if (top5[0].second == top5[i].second)
					{
						accuracy[0]++;
						color[i] = green;
					}
					else
						color[i] = yellow;
					break;
				}
			}
		}

		std::cout << network << " Image " << result_index << " (" << result_index << ":0) " << currentImg
		          << std::endl;

		std::cout << network << "    GOLD - ";
		if (goldResult == options.gold.end())
			std::cout << "no gold results for this image, omitted for accuracy summary";
		else
			std::cout << goldResult->second << " - 1.000000";
		std::cout << std::endl;

		for (size_t i = 0; i < 5; i++)
			std::cout << network << "    PRED - " << options.categories[top5[i].first] << " - " << color[i]
			          << top5[i].second << reset << std::endl;
		std::cout << network << std::endl;
	}
}

void print_accuracy_summary_headless(std::string network, float accuracy[2], size_t nbComparedImages)
{
	std::cout << "[AMD] [" << network << "TEST top1] " << 100 * accuracy[0] / nbComparedImages << "% passed."
	          << std::endl;
	std::cout << "[AMD] [" << network << "TEST top5] " << 100 * accuracy[1] / nbComparedImages << "% passed."
	          << std::endl;
	std::cout << "[AMD] [" << network << "ALL TESTS] " << 100 * accuracy[0] / nbComparedImages << "% passed."
	          << std::endl;
}

void print_accuracy_summary(std::string network, float accuracy[2], size_t nbComparedImages)
{
	// Set network name used for accuracy section. Use tabulation as default if none given.
	std::string network_accur = (network.empty()) ? "" : (network + ' ');

	std::cout << std::endl;
	std::cout << "============================================================" << std::endl;
	std::cout << "Accuracy Summary:" << std::endl;
	print_accuracy_summary_headless(network_accur, accuracy, nbComparedImages);
}
