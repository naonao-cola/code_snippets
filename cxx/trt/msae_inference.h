#pragma once

#include "trt_inference.h"
#include "ref_img_tool.h"
#include "dynamic_binaray_threshold.h"

class Bbox{
public:
    Bbox(int ix1, int iy1, int ix2, int iy2, float iscore=0.0);
    Bbox(json pts, std::string type, float iscore=0.0);
    float iou(Bbox obox);

private:
    int x1;
    int y1;
    int x2;
    int y2;
    int w;
    int h;
    float score;
};

/**
 * 无监督学习正样本模型，检测OK样本中未出现过得异常（脏污、破损、字符缺失等）
*/
class MsaeInference: public TrtInference{
public:
    MsaeInference(char *ptr, int size, int device_id, json info, bool is_stamp=false);
    void config(json config, RefImgTool *ref);
    virtual void preprocess(cv::Mat img, const json& in_param, TrtBufferManager &buffers);
    virtual json post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers);
    virtual json forward(cv::Mat img, const json& in_param);

private:
    cv::Mat preprocess_img(cv::Mat img, const json& in_param);
    bool is_intersect_important_area(const json& bbox);

private:
    DynamicThreshold m_dt_tool;
    json m_info;
    json m_config;
    RefImgTool* m_ref;
    bool m_is_stamp;
    cv::Mat m_processed_img;
};