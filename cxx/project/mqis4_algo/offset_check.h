#pragma once

#include "ref_img_tool.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * 打印偏移检测，检查打印的文本内容是否有偏斜
 * 算法通过模板匹配证书上的两个锚点（mark_a,mark_b)，和标准图进行对比来确认
*/
class OffsetCheck {
public:
    OffsetCheck(json info);
    ~OffsetCheck();
    void config(json config, RefImgTool *ref);
    virtual json forward(cv::Mat img, const json& in_param);

private:
    json m_config;
    RefImgTool* m_ref;
    json m_info;
    json m_cur_task;
    json m_refloc;
};

