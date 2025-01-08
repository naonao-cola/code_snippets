#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "utils.h"
#include "ref_img_tool.h"
#include "crnn_inference.h"
#include <nlohmann/json.hpp>
#include "logger.h"
#include "save_roi_label.h"
#include "dynamic_ocr_algo.h"


using json = nlohmann::json;

/**
 * 动态区域检测
 * 动态区域是指文字内容不固定的区域，因为内容不固定，AI无监督模型无法穷举所有OK样本，
 * 所以针对该区域目前检测方案主要通过CV来完成：
 * 1. 针对污渍缺陷，blob分析拆分文本会导致行高异常，通过判断行高是否在正常范围来判断是否有缺陷
 * 2. 字符缺失，通过判断文本框间隙是否过大
 * 3. 文字残缺，通过OCR对比
*/
class DynamicalOCR {
 public:
    explicit DynamicalOCR(CrnnInference* crnn);
    ~DynamicalOCR();
    void DynamicalOCR::config(json config, RefImgTool *ref);
    json DynamicalOCR::forward(cv::Mat img, const json &in_param);

 private:
    RefImgTool *m_ref;
    CrnnInference *m_crnn;
    DynamicOCR::DynamicCharDet::m_ptr m_ptr_dynamic_ocr_det{nullptr};
    json m_config;
    json m_dynamic_defect_std;
    json m_param;
    int m_img_height;
    int m_img_width;
    cv::Mat m_bin_img;

#ifdef COLLECT_OCR_DATA
    RoiLabelSave m_label_save;
#endif
};
