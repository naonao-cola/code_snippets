/**
 * @FilePath     : /mqis4_algo/src/algo_tool.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-05-31 13:56:33
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#pragma once
#ifndef __ALGO_TOOL_H__
#define __ALGO_TOOL_H__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
namespace nao {
namespace utils {
/**
 * @brief			Read the file, load the TRT model file.
 * @param file
 * @return
 */
std::vector<char> load_file(const std::string& file);

/**
 * @brief			Obtain the angle with the x-axis, and convert it to radians.
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @return
 */
double get_angle_x(double x1, double y1, double x2, double y2);
} // namespace utils

namespace vision {
/**
 * @brief			crnn Model scaling picture size
 * @param img
 * @param resize_img
 * @param wh_ratio
 * @param rec_image_shape
 */
void crnn_resize_img(cv::InputArray img, cv::OutputArray dst, float wh_ratio, std::initializer_list<int> rec_image_shape);

/**
 * @brief			normalization
 * @param src
 * @param dst
 * @param mean
 * @param scale
 * @param is_scale
 */
void normalize(cv::InputArray src, cv::OutputArray dst, std::initializer_list<float> mean, std::initializer_list<float> scale, const bool is_scale);

/**
 * @brief			Transfer channel
 * @param src
 * @param data
 */
void permute_batch(cv::InputArray src, float* data);
} // namespace vision
} // namespace nao

#endif //__ALGO_TOOL_H__