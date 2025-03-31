/**
 * @FilePath     : /connector_ai/src/project/utils.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-10-12 14:43:36
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-10-12 14:44:10
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/

#include "nlohmann/json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace xx
{
/**
 * @brief: 获取json 的键
 * @param json_obj
 * @param key
 * @param def_val
 * @return
 * @note :
 **/
template<typename T>
T GetProperty(const nlohmann::json& json_obj, const std::string& key, const T& def_val)
{
    return json_obj.contains(key) ? json_obj[key].get<T>() : def_val;
}

/**
 * @brief: 画出结果图
 * @param dispaly
 * @return
 * @note :
 **/
void DrawImg(cv::Mat dispaly, nlohmann::json json_info);


double computeIoU(cv::Rect box1, cv::Rect box2);

double FliterBox(std::vector<cv::Rect>& box_vec, cv::Rect box2, double thre);


nlohmann::json pt_json(std::vector<std::vector<cv::Point>> mask);

nlohmann::json pt_json(std::vector<cv::Point2f> mask);

template<typename T>
std::vector<T> order_pts(const std::vector<T> src_pt)
{
    std::vector<T> tmp, dst;
    if (src_pt.size() != 4) {
        std::cerr << "输入的原点个数错误" << std::endl;
        return dst;
    }
    tmp.assign(src_pt.begin(), src_pt.end());
    // 按照x值大小升序排列，x值小的两个点位左侧的两个点
    std::sort(tmp.begin(), tmp.end(), [=](const T pt1, const T pt2) { return pt1.x < pt2.x; });
    if (tmp[0].y > tmp[1].y) {
        if (tmp[2].y > tmp[3].y) {
            dst.emplace_back(tmp[1]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[0]);
        }
        else {
            dst.emplace_back(tmp[1]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[0]);
        }
    }
    else {
        if (tmp[2].y > tmp[3].y) {
            dst.emplace_back(tmp[0]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[1]);
        }
        else {
            dst.emplace_back(tmp[0]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[1]);
        }
    }
    return dst;
}

cv::Vec4i get_value(std::vector<cv::Point>);
cv::Point get_center(std::vector<cv::Point> pts);
}   // namespace xx