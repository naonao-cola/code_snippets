#pragma once
#include <string>
#include <algorithm>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>

#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "logger.h"

using json = nlohmann::json;

class DynamicThreshold {
 public:
    void config(const std::string& type, double img_scale = 1.0 );
    void forwad(const cv::Mat& input_img, cv::Mat& out_thr_map, int blank_thr = 255, int char_thr = 0 );

 private:
    template<class Vtype>
    bool read_json_value(const json& param, const std::string& key, Vtype&);
    void get_char_region(const cv::Mat& input, cv::Mat& char_region_bin_img);
    static json get_paper_std(const std::string& ptype_str);

 private:
    int m_char_region_thr{20};
    int m_black_region_thr{50};
    double m_img_scale{1.0};
    int m_bin_thr;
    int m_expand_size_w;
    int m_expand_size_h;
    int m_remove_size;
    int m_open_size;
    int m_close_size;
};

