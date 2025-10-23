/*
# ===========================================================
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License
# ===========================================================

# last change: 30 Sep. 2025
*/

#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <vart_ml_runner/runner.h>

const std::string red   ("\033[1;31m");
const std::string green ("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string reset ("\033[0m")  ;

// for DEBUG purpose     //DB
#define DB_DEBUG      1  //DB
#define DB_NO_DEBUG   0  //DB
#define DB_NUMIMAGES  5  //DB
#define SET_W        10  //DB
#define SET_P         4  //DB

namespace ColorFormat 
{
    constexpr int rgb = 1;
    constexpr int RGB = 1;
    constexpr int bgr = 0;
    constexpr int BGR = 0;
}
struct options
{
	size_t                                                       nbImages;
	size_t                                                       batchSize;
	std::vector<std::string>                                     networks;
	std::vector<std::string>                                     categories;
	std::map<std::string, std::string>                           gold;
	std::string                                                  imgPath;
	std::vector<std::string>                                     snapshots;
	std::map<std::string, std::string>                           raw;
	std::vector<std::map<std::string, std::string>>              data_types;
	std::vector<std::vector<uint8_t>>                            ddr_in_list;
	std::vector<std::vector<uint8_t>>                            ddr_out_list;
	std::vector<std::map<std::string, std::vector<std::string>>> goldFiles;
};

/**
 * @brief Check if the file is a directory
 *
 * @param dir Path to the file to check
 * @return bool True if dir points to a directory, false otherwise
 */
bool isDirectory(const char* dir);

/**
 * @brief Check if the file is a regular file
 *
 * @param path Path to the file to check
 * @param mode Mode to open the file for
 * @return bool True if path points to a regular file, false otherwise
 */
bool isFile(const char* path, const char* mode);

/**
 * @brief Display help for the program and exits
 *
 * @param cmd Command used to execute the demo
 * @param reason Reason why the function has been called
 */
void usage(std::string demo, std::string reason);

/**
 * @brief Read cli options
 *
 * @param argc Number of cli parameters
 * @param argv Cli parameters
 * @param options struct that will be populated with interpreted generic options
 * @param images_paths vector of paths to images to process
 * @return int The Vart ML error (vart_ml_error_id) code of the operation.
 */
int read_options(int argc, char* argv[], struct options& options, std::vector<std::string>& images_paths);

/**
 * @brief Read the images options
 *
 * @param options struct that will be populated with interpreted generic options
 * @param images_paths vector of paths of images to process
 * @param snapshots_paths vector of paths of snapshots to process
 * @param default_batchSize batch size of the snapshot
 * @return int The Vart ML error (vart_ml_error_id) code of the operation.
 */
int read_images_options(struct options&           options,
                        std::vector<std::string>& images_paths,
                        size_t                    default_batchSize);

/**
 * @brief Get the k results with the highest probability
 *
 * @param buf Vector of results for each label
 * @param k Number of top value requested
 * @return std::vector<std::pair<size_t, float>> k pair of label number and probability value
 */
std::vector<std::pair<size_t, float>> topk(std::vector<float>& buf, size_t k);

/**
 * @brief Get all labels
 *
 * @param labels_file Labels file path
 * @param categories Returned vector of labels
 * @return int The Vart ML error (vart_ml_error_id) code of the operation.
 */
int read_categories(std::string labels_file, std::vector<std::string>& categories);

/**
 * @brief Get all correct labels form the gold file
 *
 * @param gold_file Gold file path
 * @param gold Returned map with file names as key and gold labels as value
 * @return int The Vart ML error (vart_ml_error_id) code of the operation.
 */
int read_gold(std::string gold_file, std::map<std::string, std::string>& gold);

/**
 * @brief Apply the sofmax function to all elements
 *
 * @param buf Vector of element on which to apply the softmax function
 */
void softmax(std::vector<float>& buf);

void set_colorFmt (int new_colFmt); //DB

/**
 * @brief Apply the preprocessing to image
 *
 * @param buf Buffer to fill with processed image
 * @param image_path Path of the target image file
 * @param height Height of the image
 * @param width Width of the image
 * @param channel Channel count of the image
 * @return int The Vart ML error (vart_ml_error_id) code of the operation.
 */
int preprocess(float* const buf, std::string image_path, size_t height, size_t width, size_t channel);
int preprocess(int8_t* const buf, std::string image_path, size_t height, size_t width, size_t channel);

/**
 * @brief Convert data into DDR native format
 *
 * @param arch Backend architecture
 * @param shape_format Shape format of the input data
 * @param ddr_format Format expected by the DDR
 * @param in Input tensor to match preprocessing of images to
 * @param src Source address of the buffer
 * @param dst Destination address of the buffe
 */
void force_native_input(std::string                   arch,
                        std::string                   shape_format,
                        std::string                   ddr_format,
                        const vart::vart_ml_tensor_t* in,
                        const void*                   src,
                        void*                         dst);

/**
 * @brief Convert data from DDR native format
 *
 * @param arch Backend architecture
 * @param shape_format Shape format of the input data
 * @param ddr_format Format expected by the DDR
 * @param in Input tensor to match preprocessing of images to
 * @param src Source address of the buffer
 * @param dst Destination address of the buffe
 */
void force_native_output(std::string                   arch,
                         std::string                   shape_format,
                         std::string                   ddr_format,
                         const vart::vart_ml_tensor_t* out,
                         const void*                   src,
                         void*                         dst);

/**
 * @brief Apply the preprocessing to a set of images
 *
 * Preprocess all input images. If quantization option is enabled, proceed with quantization of images as
 * well.
 * Note: depending on whether or not input data is assumed in native format of not, re-organize data in
 * a way that does not require further byte-level manipulation (force_native_input).
 *
 * @param in_tensors Input tensor to match preprocessing of images to
 * @param img_idx Index of first image
 * @param nb_images Number of images to process
 * @param fpga_arch Target fpga architecture
 * @param in_isNative Output data format nativeness
 * @param useExternalQuant Datas need to be quantized
 * @param images_paths Path to an image file
 * @param inbuf Buffer input data
 * @return int The Vart ML error (vart_ml_error_id) code of the operation.
 */
int preprocess_batch(std::vector<const vart::vart_ml_tensor_t*> in_tensors,
                     size_t                                     img_idx,
                     size_t                                     nb_images,
                     std::string                                fpga_arch,
                     bool                                       in_isNative,
                     bool                                       useExternalQuant,
                     std::vector<std::string>                   images_paths,
                     std::vector<void*>&                        inbuf);

/**
 * @brief Compare the results to gold results and display results
 *
 * @param options Program options
 * @param batchSize Size of a batch of images
 * @param first_image_index Index of the first image in the batch
 * @param images_paths Input images
 * @param results Inference results
 * @param nbComparedImages Number of images that got compared to gold results
 * @param accuracy[2] Array of accuracies to fill
 * @param network Network name to display
 */
void compare_gold(struct options                  options,
                  size_t                          batchSize,
                  size_t                          first_image_index,
                  std::vector<std::string>        images_paths,
                  std::vector<std::vector<float>> results,
                  size_t&                         nbComparedImages,
                  float                           accuracy[2],
                  std::string                     network);

/**
 * @brief Prints the accuracy summary without header
 *
 * @param network Network name to be displayed
 * @param accuracy Accuracy computed by demo
 * @param nbComparedImages Number of images compared to a gold file
 */
void print_accuracy_summary_headless(std::string network, float accuracy[2], size_t nbComparedImages);

/**
 * @brief Prints the accuracy summary
 *
 * @param network Network name as given to program
 * @param accuracy Accuracy computed by demo
 * @param nbComparedImages Number of images compared to a gold file
 */
void print_accuracy_summary(std::string network, float accuracy[2], size_t nbComparedImages);
