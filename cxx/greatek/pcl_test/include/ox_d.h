/**
 * @FilePath     : /pcl_test/include/ox_d.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-15 09:42:10
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-15 09:56:14
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

#include "ox.h"
#include <algorithm>
#include <numeric>

class YOLOv8Detector : public YoloOnnxBase
{
public:
    YOLOv8Detector(const std::string& model_path, bool use_gpu = false, float conf_threshold = 0.25f, float iou_threshold = 0.45f);

protected:
    cv::Mat Preprocess(const cv::Mat& image, std::vector<int64_t>& input_tensor_shape) override;

    DetectionResult Postprocess(const std::vector<Ort::Value>& output_tensors, const cv::Size& orig_size, const cv::Size& input_size) override;

private:
    // YOLOv8特定的后处理工具函数
    void NMS(std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids, std::vector<int>& indices);

    float ComputeIOU(const cv::Rect& box1, const cv::Rect& box2);
    LetterboxInfo letterbox_info_;   // 保存letterbox信息
};

#endif   // YOLOV8_DETECTOR_H