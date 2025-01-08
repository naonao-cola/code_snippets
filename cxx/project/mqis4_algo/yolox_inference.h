#pragma once

#include "trt_inference.h"

class YoloxInference: public TrtInference {

public:
    YoloxInference(char *ptr, int size, int device_id, json info);

    virtual void preprocess(cv::Mat img, TrtBufferManager &buffers);

    virtual json post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers);

private:
    json m_info;
};
