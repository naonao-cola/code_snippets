#pragma once

#include "ref_img_tool.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

/**
 * 印章残缺检测
 * 提取印章，和标准印章对齐后相减
*/
class StampDet {
public:
    StampDet(json info);
    void config(json config, RefImgTool *ref);
    json forward(cv::Mat img, const json& in_param);

private:
    RefImgTool* m_ref;
    json m_info;
    json m_config;
};
