/**
 * @FilePath     : /test02/src/main.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-07-08 11:15:50
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-28 16:49:24
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#include "test.h"
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <vector>

#define USE_JPG 0
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

std::string getFilenameWithoutExt(const std::string& path)
{
    size_t lastSlash = path.find_last_of("/");
    size_t lastDot = path.find_last_of('.');

    if (lastSlash == std::string::npos)
        lastSlash = 0;
    else
        lastSlash += 1;

    if (lastDot == std::string::npos || lastDot < lastSlash) {
        return path.substr(lastSlash);
    }

    return path.substr(lastSlash, lastDot - lastSlash);
}



int main(int argc, char** argv)
{
    std::vector<cv::String> path_vec;
    cv::glob("./img/", path_vec);

    struct timeval start_time, stop_time;
    int64_t time_count;
#if USE_JPG

    for (int i = 0; i < path_vec.size();i++) {
        cv::Mat temp = cv::imread(path_vec[i]);
        std::string filename = getFilenameWithoutExt(path_vec[i]);
        std::string save_path = std::string(R"(./jpg_test/)") + filename + ".jpg";

        gettimeofday(&start_time, NULL);
        std::vector<int> param = std::vector<int>(2);
        param[0] = cv::IMWRITE_JPEG_QUALITY;
        param[1] = 80; // default(95) 0-100
        std::vector<unsigned char> buff;
        cv::imencode(".jpg", temp, buff, param);
        //std::string filename = std::experimental::filesystem::path(path_vec[i]).stem().string();
        std::ofstream ofs(save_path.c_str(), std::ios::binary);
        ofs.write(reinterpret_cast<char*>(buff.data()), buff.size());
        ofs.close();
        gettimeofday(&stop_time, NULL);
        time_count += (__get_us(stop_time) - __get_us(start_time)) / 1000;
        printf("压缩 保存 : %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    }
#else
    for (int i = 0; i < path_vec.size(); i++) {

        cv::Mat temp = cv::imread(path_vec[i]);


        //std::string filename = std::experimental::filesystem::path(path_vec[i]).stem().string();
        std::string filename = getFilenameWithoutExt(path_vec[i]);
        std::string save_path = std::string(R"(./bmp_test/)") + filename + ".bmp";
        gettimeofday(&start_time, NULL);
        SaveImage(save_path, temp);
        gettimeofday(&stop_time, NULL);
        time_count += (__get_us(stop_time) - __get_us(start_time)) / 1000;
        printf(" bmp  保存 : %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    }
#endif
    printf("压缩 保存 平均 : %zu ms\n", time_count / path_vec.size());
    // std::string imgBase64 = base64_enCode(reinterpret_cast<const char*>(buff.data()), buff.size()); // 编码
    // std::cout << "img base64 encode data:" << imgBase64 << std::endl;
    // std::cout << "img base64 encode size:" << imgBase64.size() << std::endl;
    // std::string imgdecode64 = base64_deCode(imgBase64); // 解码
    // std::cout << "img decode:" << imgdecode64 << std::endl;
    // std::cout << "img decode size:" << imgdecode64.size() << std::endl;

    return 0;
}
