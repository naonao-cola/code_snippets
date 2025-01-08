#pragma once

#include <vector>
#include <algorithm>
#include <set>
#include <list>
#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <nlohmann/json.hpp>
#include "logger.h"
#include "defines.h"

using json = nlohmann::json;

namespace DynamicOCR {
inline int get_rotatedRect_longside(const cv::RotatedRect rotret) {
    return rotret.size.width > rotret.size.height ? rotret.size.width : rotret.size.height;
}

inline int get_rotatedRect_shortside(const cv::RotatedRect rotret) {
    return rotret.size.width < rotret.size.height ? rotret.size.width : rotret.size.height;
}

inline bool cmp_by_x(cv::RotatedRect a, cv::RotatedRect b) {
    if (a.center.x < b.center.x)
        return true;
    return false;
}

inline bool cmp_by_y(cv::RotatedRect a, cv::RotatedRect b) {
    if (a.center.y < b.center.y)
        return true;
    return false;
}

inline bool cmp_by_xy(cv::RotatedRect a, cv::RotatedRect b) {
    int dist = std::abs(a.center.y - b.center.y);
    if (dist > 50) {
        return cmp_by_y(a, b);
    } else {
        return cmp_by_x(a, b);
    }
    return true;
}

inline bool cmp_rot_rect(cv::RotatedRect a, cv::RotatedRect b) {
    cv::Point p_a = a.center;
    cv::Point p_b = b.center;
    if (p_a.y < p_b.y) {
        return true;
    }
    return false;
}

/**
 * 动态文本区域检测辅助类, 负责拆分文本，缺陷判断等
*/
class DynamicCharDet {
 public:
    typedef std::shared_ptr<DynamicCharDet> m_ptr;
     /**
     * @brief 将多行文本切分为单行文本图片，并判断是否存在缺陷
     *
     * @param input_img
     * @param line_text_imgs ： 切分为单行文字的图片
     * @param OK_rrect ：OK 的旋转矩形区域
     * @param NG_rrect ：NG 的旋转矩形区域
     */
    void spilt_muti_line_text_img(const cv::Mat input_img,
                                  std::vector<cv::Mat> &line_text_imgs,
                                  std::vector<cv::RotatedRect> &OK_rrect,
                                  std::vector<cv::RotatedRect> &NG_rrect,
                                  std::string label);  // NOLINT
    void config(const std::string& paper_type, bool has_stamp);

 private:
    void spilt_line_by_blob(cv::Mat input, std::vector<cv::RotatedRect> &det_rst, std::string label);

    void get_text_mask(const cv::Mat &input_img, cv::Mat &txt_mask);

    void det_defect(std::vector<cv::RotatedRect> &det_rst,
                              std::vector<cv::RotatedRect> &NG_rrst,
                              std::vector<cv::RotatedRect> &OK_rrst,
                              std::string label = "");

    void crop_single_line_text_img(const cv::Mat input_img,
                                   const std::vector<cv::RotatedRect> &OK_rrect,
                                   std::vector<cv::Mat> &line_text_imgs );
    void remove_stamp(cv::Mat &input_img, cv::Mat &remove_stamp_img);

    void get_stamp_region(const cv::Mat &input_img, cv::Rect2i &stamp_bbox);

    void text_extract_algo(cv::Mat &input_img);

    json get_ocr_defect_std(std::string ptype_str);

 private:
    json m_config;
    json m_dynamic_defect_std;
    json m_param;
    int m_img_height;
    int m_img_width;
    cv::Mat m_bin_img;
    bool m_has_stamp{false};
    std::string m_paper_type;
    int m_bin_thr;
};

}  // namespace DynamicOCR
