#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <nlohmann/json.hpp>
#include <string.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "trt_logging.h"
#include "trt_buffers.h"

using json = nlohmann::json;

class TrtInference {

public:
    TrtInference(char *ptr, int size, int device_id);
    ~TrtInference();
    virtual void preprocess(cv::Mat img, const json& in_param, TrtBufferManager &buffers) = 0;
    virtual json forward(cv::Mat img, const json& in_param);
    virtual json post_process(cv::Mat img, const json& in_param, TrtBufferManager &buffers) = 0;

protected:
    TrtLoggerImpl m_logger;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine{nullptr};
    nvinfer1::IExecutionContext* m_context;
    cudaStream_t m_stream;
};
