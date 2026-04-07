/**
 * @FilePath     : /pcl_test/include/ox.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-14 17:28:34
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-15 13:58:27
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#ifndef YOLO_ONNX_BASE_H
#define YOLO_ONNX_BASE_H

#include "onnxruntime_c_api.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>



// 通用检测结果结构
// 在 ox.h 中修改 DetectionResult 结构
struct DetectionResult
{
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      class_ids;
    std::vector<cv::Mat>  masks;   // 新增：实例分割掩码
};
// Letterbox信息，用于前后处理坐标转换
struct LetterboxInfo
{
    float    scale;       // 缩放比例
    int      dw;          // 宽度padding
    int      dh;          // 高度padding
    cv::Size new_shape;   // 缩放后的尺寸
};


// 抽象基类：定义通用推理接口
class YoloOnnxBase
{
public:
    YoloOnnxBase(const std::string& model_path, bool use_gpu = false, float conf_threshold = 0.25f, float iou_threshold = 0.45f);
    virtual ~YoloOnnxBase() = default;

    // 禁止拷贝，允许移动
    YoloOnnxBase(const YoloOnnxBase&)            = delete;
    YoloOnnxBase& operator=(const YoloOnnxBase&) = delete;

    // 主检测接口
    virtual DetectionResult Detect(const cv::Mat& image);

protected:
    // 子类必须实现的接口
    virtual cv::Mat Preprocess(const cv::Mat& image, std::vector<int64_t>& input_tensor_shape) = 0;

    virtual DetectionResult Postprocess(const std::vector<Ort::Value>& output_tensors, const cv::Size& orig_size, const cv::Size& input_size) = 0;

    // 通用工具方法
    std::vector<Ort::Value> RunInference(const cv::Mat& input_tensor);

    // 获取输入输出节点信息
    void InitializeModel(const std::string& model_path, bool use_gpu);

    // 成员变量
    Ort::Env                      env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions           session_options_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    size_t                   num_inputs_ = 1;
    size_t                   num_outputs_ = 1;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;



    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    float conf_threshold_;
    float iou_threshold_;
    bool  is_dynamic_input_shape_ = false;
};
#endif   // YOLO_ONNX_BASE_H