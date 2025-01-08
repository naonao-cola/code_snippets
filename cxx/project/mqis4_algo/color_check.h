#pragma once

#include "ref_img_tool.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class ColorCheck {
public:
    ColorCheck(json info);
    void config(RefImgTool *ref);
    virtual json forward(cv::Mat img);

private:
    RefImgTool* m_ref;
    json m_info;
};
