#pragma once

#include "ref_img_tool.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// 重影检查，目前未使用
class DoublePrintCheck {
public:
    DoublePrintCheck(json info);
    void config(json config, RefImgTool *ref);
    virtual json forward(cv::Mat img);

private:
    RefImgTool* m_ref;
    json m_info;
    json m_roi_list;
    json m_config;
};
