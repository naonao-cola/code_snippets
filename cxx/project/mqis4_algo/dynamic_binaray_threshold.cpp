#include "dynamic_binaray_threshold.h"

void DynamicThreshold::config(const std::string& type, double img_scale) {
    img_scale = std::min(10.0, img_scale);
    img_scale = std::max(0.1, img_scale);
    m_img_scale = img_scale;
    json param = get_paper_std(type);
    param["dyn_thr_expand_size"] = static_cast<int>(param["dyn_thr_expand_size"] * m_img_scale);
    param["dyn_thr_remove_size"] = static_cast<int>(param["dyn_thr_remove_size"] * m_img_scale);
    LOG_INFO("param: {}", param.dump());

    bool read_success = true;
    read_success &= read_json_value<int>(param, "dyn_thr_bin_thr", m_bin_thr);
    read_success &= read_json_value<int>(param, "dyn_thr_expand_size", m_expand_size_w);
    m_expand_size_h = m_expand_size_w /2;
    read_success &= read_json_value<int>(param, "dyn_thr_remove_size", m_remove_size);
    read_success &= read_json_value<int>(param, "dyn_thr_open_size", m_open_size);
    read_success &= read_json_value<int>(param, "dyn_thr_close_size", m_close_size);
    assert(read_success);
}

json DynamicThreshold::get_paper_std(const std::string& ptype_str) {
    json dynamic_std;
    if (ptype_str == "HGZ_B") {
        dynamic_std = {
            {"dyn_thr_bin_thr", 200},
            {"dyn_thr_expand_size", 240},
            {"dyn_thr_remove_size", 100},
            {"dyn_thr_open_size", 3},
            {"dyn_thr_close_size", 10},
        };
    } else if (ptype_str == "HGZ_A") {
        dynamic_std = {
            {"dyn_thr_bin_thr", 190},
            {"dyn_thr_expand_size", 450},
            {"dyn_thr_remove_size", 100},
            {"dyn_thr_open_size", 3},
            {"dyn_thr_close_size", 10},
        };
    } else if (ptype_str == "COC") {
        dynamic_std = {
            {"dyn_thr_bin_thr", 200},
            {"dyn_thr_expand_size", 300},
            {"dyn_thr_remove_size", 100},
            {"dyn_thr_open_size", 3},
            {"dyn_thr_close_size", 10},
        };
    } else if (ptype_str == "HBZ_A") {
        dynamic_std = {
            {"dyn_thr_bin_thr", 190},
            {"dyn_thr_expand_size", 350},
            {"dyn_thr_remove_size", 100},
            {"dyn_thr_open_size", 3},
            {"dyn_thr_close_size", 10},
        };
    } else if (ptype_str == "HBZ_B") {
        dynamic_std = {
            {"dyn_thr_bin_thr", 200},
            {"dyn_thr_expand_size", 250},
            {"dyn_thr_remove_size", 100},
            {"dyn_thr_open_size", 3},
            {"dyn_thr_close_size", 10},
        };
    } else if (ptype_str == "RYZ") {
        dynamic_std = {
            {"dyn_thr_bin_thr", 100},
            {"dyn_thr_expand_size", 300},
            {"dyn_thr_remove_size", 100},
            {"dyn_thr_open_size", 3},
            {"dyn_thr_close_size", 10},
        };
    }
    return dynamic_std;
}

void DynamicThreshold::forwad(const cv::Mat& input_img_o, cv::Mat& out_thr_map, int blank_thr, int char_thr) {
    // 提取文字区域
    blank_thr = std::max(0, blank_thr);
    blank_thr = std::min(255, blank_thr);

    char_thr = std::max(0, char_thr);
    char_thr = std::min(255, char_thr);
    cv::Mat input_img;
    int dw, dh;
    // dw = static_cast<int>(input_img_o.cols * m_img_scale);
    // dh = static_cast<int>(input_img_o.rows * m_img_scale);
    // cv::resize(input_img_o, input_img, cv::Size2i(dw, dh), m_img_scale, m_img_scale);
    get_char_region(input_img_o, out_thr_map);
    cv::Mat lut = cv::Mat::zeros(1, 256, CV_8UC1);
    uchar* p = lut.data;
    p[0] = static_cast<uchar>(blank_thr);
    p[255] = static_cast<uchar>(char_thr);
    cv::LUT(out_thr_map, lut, out_thr_map);
    // dw = input_img_o.cols;
    // dh = input_img_o.rows;
    // cv::resize(out_thr_map, out_thr_map, cv::Size2i(dw, dh), 1.0/m_img_scale, 1.0/m_img_scale);
}

void DynamicThreshold::get_char_region(const cv::Mat& input, cv::Mat& char_region_bin_img) {
    cv::Mat gray_img;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = input;
    }

    cv::threshold(gray_img, char_region_bin_img, m_bin_thr, 255, cv::THRESH_BINARY_INV);

    int size_open = static_cast<int>(m_open_size * m_img_scale * 10.0);
    int size_close = static_cast<int>(m_close_size * m_img_scale * 10.0);
    cv::Mat elem_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size2i(size_open, size_open));
    cv::Mat elem_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size2i(size_close, size_close));
    cv::morphologyEx(char_region_bin_img, char_region_bin_img, cv::MORPH_CLOSE, elem_close);
    cv::morphologyEx(char_region_bin_img, char_region_bin_img, cv::MORPH_OPEN, elem_open);

    cv::Mat elem_expand = cv::getStructuringElement(cv::MORPH_RECT, cv::Size2i(m_expand_size_w, m_expand_size_h));
    cv::Mat elem_remove = cv::getStructuringElement(cv::MORPH_RECT, cv::Size2i(m_remove_size, m_remove_size));
    cv::morphologyEx(char_region_bin_img, char_region_bin_img, cv::MORPH_DILATE, elem_expand);
    cv::morphologyEx(char_region_bin_img, char_region_bin_img, cv::MORPH_ERODE, elem_remove);
    // 填充为凸集
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy(contours.size());
    cv::findContours(char_region_bin_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point());
    // cv::drawContours(char_region_bin_img, contours, -1, cv::Scalar(0), 2);
    int img_area = char_region_bin_img.rows * char_region_bin_img.cols;
    for (int i = 0; i< contours.size(); ++i) {
        cv::Rect2i ret = cv::boundingRect(contours[i]);
        if ( ret.area() > img_area/2 ) {
            continue;
        }
        cv::fillConvexPoly(char_region_bin_img, contours[i], cv::Scalar(255));
    }
}

template<class Vtype>
bool DynamicThreshold::read_json_value(const json& param, const std::string& key, Vtype& value) {
    if (param.contains(key)) {
        value = param[key].get<Vtype>();
        return true;
    } else {
        LOG_WARN("param not contains {}", key);
        return false;
    }
}
