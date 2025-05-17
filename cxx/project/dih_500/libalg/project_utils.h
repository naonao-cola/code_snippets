//
// Created by y on 24-12-9.
//

#ifndef DIH_ALG_PROJECT_UTILS_H
#define DIH_ALG_PROJECT_UTILS_H
#include <list>

#include "neural_network.h"


int CountBoxCategoryConf(const std::list<NNetResult_t>& src, const std::vector<float>& conf_v,
                     std::list<NNetResult_t>& dst);

int MergePltImg(const cv::Mat& img_bri, const cv::Mat& img_flu, cv::Mat& output);




#endif  // DIH_ALG_PROJECT_UTILS_H
