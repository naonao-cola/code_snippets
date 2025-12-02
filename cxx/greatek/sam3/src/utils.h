#ifndef __UTILS_H__
#define __UTILS_H__

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <vector>
#include "NvInferPlugin.h"

class MyLogger : public nvinfer1::ILogger
{
    void log(Severity s, const char* msg) noexcept override
    {
        if (s <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};


std::shared_ptr<float> sam_preprocess(const cv::Mat& img, int target_width, int target_height, float mean, float std);

std::vector<char> load_engine(const std::string& path);



void infer(std::string engine_path, std::string img_path, std::string prompt);

#endif   // __UTILS_H__