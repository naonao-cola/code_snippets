#pragma once
#include "ref_img_tool.h"
#include <nlohmann/json.hpp>
#include <string.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

class BarcodeDecoder {
public:
    BarcodeDecoder(json info);

    void config(json config, RefImgTool *ref);

    json decode(cv::Mat img, float scale=0.5);

    virtual json forward(cv::Mat img);

private:
    RefImgTool* m_ref;
    json m_config;
    json m_info;
};
