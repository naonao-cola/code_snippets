#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "defines.h"

using json = nlohmann::json;

enum TFM_MODE
{
    TFM_NONE, TFM_REF, TFM_INFER
};

/**
 * 图像预处理工具
 * 1. 纸张区域、ROI区域提取等
 * 2. 根据在模板图上的标注信息，对推理图进行坐标转换
 * 3. 印章提取
 * 4. barcode提取
*/
class RefImgTool {
public:
    RefImgTool();
    cv::Mat get_ref_img(bool gray=false);
    cv::Mat get_masked_img(cv::Mat img, const json& in_param);
    void config(const json& config);
    cv::Mat set_test_img(cv::Mat &img, const json& in_param, bool& locate_ok);
    cv::Mat perspective(cv::Mat &img, std::vector<cv::Point> pts, int out_w, int out_h);
    // std::vector<cv::Point> tfm_pts(std::vector<cv::Point> pts);
    json get_roi_img(cv::Mat img, cv::Mat &roi_img, json roi, int width, int height, TFM_MODE tfmmode=TFM_INFER);
    cv::Rect get_pad_roi_img(cv::Mat img, cv::Mat &roi_img, json roi, double pad_px, TFM_MODE tfmmode=TFM_INFER);

    template<typename T>
    cv::Mat get_perspective_M(std::vector<T> pts, int width, int height, int& out_w, int& out_h);
    void get_paper_loc(const cv::Mat &img, std::vector<cv::Point2f>& roi);
    json transform_roi(json roi, bool is_ref);
    json transform_result(json polygon, bool need_round=true);
    json transform_stamp_result(json polygon, bool need_round=true);
    bool is_intersect_with_stamp(json region);

    LocateInfo get_ref_loc() { return m_ref_loc; }
    LocateInfo get_img_loc() { return m_img_loc; }
    json get_stamp_bbox() { return m_stamp_bbox; }
    json get_shape_pts_by_name(std::string name);

    bool has_stamp() { return m_stamp_bbox.size() == 2; }
    cv::Mat get_stamp_img(cv::Mat img, const json& in_param);
    cv::Mat get_stamp_img_keep_black(cv::Mat img, cv::Mat stamp_temp, const json& in_param, int& xoff, int& yoff);

    cv::Mat stamp_extract(cv::Mat img, float red_ratio, bool locate);
    cv::Mat barcode_extract(cv::Mat img, std::vector<cv::Point>& char_pts);

private:
    cv::Mat m_ref_img;
    cv::Mat m_ref_gray_img;
    cv::Mat m_ref_paper_M;      // 参考图裁剪纸张区域的变换矩阵
    cv::Mat m_paper_M;          // 推理图裁剪纸张区域的变换矩阵
    cv::Mat m_content_M;        // 印刷内容矫正矩阵
    cv::Mat m_ref2test_M;

    cv::Mat m_mark_a_temp;
    cv::Mat m_mark_b_temp;
    cv::Mat m_mark_a_pad;
    cv::Mat m_mark_b_pad;

    json m_mask_shapes;
    json m_mark_a;
    json m_mark_b;

    cv::Mat m_stamp;
    json m_stamp_bbox;

    LocateInfo m_ref_loc;
    LocateInfo m_img_loc;

    std::string m_ptype_str;
};
