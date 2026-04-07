/**
 * @FilePath     : /pcl_test/include/ox_seg.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-16 09:19:42
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-16 09:19:43
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#ifndef YOLOV8_SEG_DETECTOR_H
#define YOLOV8_SEG_DETECTOR_H

#include "ox.h"
#include <algorithm>
#include <numeric>

class YOLOv8SegDetector : public YoloOnnxBase
{
public:
    YOLOv8SegDetector(const std::string& model_path, bool use_gpu = false, float conf_threshold = 0.25f, float iou_threshold = 0.45f, float mask_threshold = 0.5f);

protected:
    cv::Mat Preprocess(const cv::Mat& image, std::vector<int64_t>& input_tensor_shape) override;

    DetectionResult Postprocess(const std::vector<Ort::Value>& output_tensors, const cv::Size& orig_size, const cv::Size& input_size) override;

private:
    // 后处理工具函数
    void ProcessMaskProtos(const float* mask_proto_data, const std::vector<int64_t>& mask_proto_shape, const std::vector<std::vector<float>>& mask_coeffs, const std::vector<cv::Rect>& boxes, std::vector<cv::Mat>& output_masks, const cv::Size& target_size);

    void NMSWithMasks(std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids, std::vector<std::vector<float>>& mask_coeffs, std::vector<int>& indices);

    float ComputeIOU(const cv::Rect& box1, const cv::Rect& box2);

    // 掩码相关参数
    float         mask_threshold_;   // 掩码二值化阈值
    LetterboxInfo letterbox_info_;
};

#endif   // YOLOV8_SEG_DETECTOR_H