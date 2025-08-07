/**
 * @FilePath     : /test02/src/test.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-07-08 12:00:36
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-28 11:32:22
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/


#include <string>


#include <fstream>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <vector>

std::string base64_enCode(const char* Data, unsigned int in_len);

std::string base64_deCode(std::string const& encoded_string);

void SaveImage(const std::string& save_path, const cv::Mat& img);

void img2dat(cv::Mat src, std::string datName);