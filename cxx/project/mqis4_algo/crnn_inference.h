#pragma once

#include "trt_inference.h"
#include "ref_img_tool.h"
#include <fstream>

// OCR识别模型
class CrnnInference: public TrtInference {

public:
    CrnnInference(char *ptr, int size, int device_id, json info);
    ~CrnnInference();
    void config(json config, RefImgTool *ref);

    virtual void preprocess(cv::Mat img, const json& in_param, TrtBufferManager &buffers);

    virtual json post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers);

    virtual json forward(cv::Mat img, const json& in_param);
    std::string forward_pure(cv::Mat img, const json& in_param);
    json get_info();
private:
    void save_ocr_data(cv::Mat img, const json& task, const json& in_param, const std::string& result_txt, const std::string& gt_txt);

private:
    RefImgTool* m_ref;
    json m_config;
    json m_info;

    // Data collect
    std::fstream dynamic_ok_fs;
    std::fstream dynamic_ng_fs;
    std::fstream static_ok_fs;
    std::fstream static_ng_fs;
};
