/**
 * @FilePath     : /mqis4_algo/src/algo_tool.cpp
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-05-31 14:14:05
 * @Copyright    : G AUTOMOBILE RESEARCH in_fileSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#include <fstream>
#include <iostream>
#include "algo_tool.h"

namespace nao {
namespace utils {
std::vector<char> load_file(const std::string& file)
{
    std::ifstream in_file(file, std::ios::in | std::ios::binary);
    if (!in_file.is_open())
        return {};

    in_file.seekg(0, std::ios::end);
    size_t length = in_file.tellg();
    std::vector<char> data;
    if (length > 0) {
        in_file.seekg(0, std::ios::beg);
        data.resize(length);
        in_file.read(data.data(), length);
    }
    in_file.close();
    return data;
}

double get_angle_x(double x1, double y1, double x2, double y2)
{
    if (std::abs(x1 - x2) < 0.001) {
        return CV_PI / 2;
    }
    double k = (y1 - y2) / (x1 - x2);
    double angle = atanl(k);
    return angle;
}
} // namespace utils

namespace vision {
void crnn_resize_img(cv::InputArray img, cv::OutputArray resize_img, float wh_ratio, std::initializer_list<int> rec_image_shape)
{
    std::initializer_list<int>::iterator it = rec_image_shape.begin();
    int imgC = *it;
    int imgH = *(++it);
    int imgW = *(++it);
    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols()) / float(img.rows());
    int resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f, cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, int(imgW - resize_img.cols()), cv::BORDER_CONSTANT, {127, 127, 127});
}

void normalize(cv::InputArray src, cv::OutputArray dst, std::initializer_list<float> mean, std::initializer_list<float> scale, const bool is_scale)
{
    std::vector<float> mean_v(mean);
    std::vector<float> scale_v(scale);
    double e = 1.0;
    if (is_scale) {
        e /= 255.0;
    }
    src.getMat().convertTo(dst, CV_32FC3, e);
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(dst, bgr_channels);
    for (auto i = 0; i < bgr_channels.size(); i++) {
        bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale_v[i], (0.0 - mean_v[i]) * scale_v[i]);
    }
    cv::merge(bgr_channels, dst);
}

void permute_batch(cv::InputArray src, float* data)
{
    int rh = src.rows();
    int rw = src.cols();
    int rc = src.channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(src, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
    }
}
} // namespace vision
} // namespace nao