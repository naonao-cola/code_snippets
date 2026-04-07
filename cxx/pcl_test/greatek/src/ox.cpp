/**
 * @FilePath     : /pcl_test/src/ox.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-14 17:28:48
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-15 15:36:35
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "ox.h"

// 基类实现
YoloOnnxBase::YoloOnnxBase(const std::string& model_path, bool use_gpu, float conf_threshold, float iou_threshold)
    : conf_threshold_(conf_threshold)
    , iou_threshold_(iou_threshold)
{
    InitializeModel(model_path, use_gpu);
}

inline void YoloOnnxBase::InitializeModel(const std::string& model_path, bool use_gpu)
{
    env_             = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO");
    session_options_ = Ort::SessionOptions();

    // 配置GPU
    if (use_gpu) {
        auto providers = Ort::GetAvailableProviders();
        if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
            OrtCUDAProviderOptions cuda_options;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        }
    }

    // 加载模型
#ifdef _WIN32
    std::wstring w_model_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, w_model_path.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    // 获取输入节点信息
    Ort::AllocatorWithDefaultOptions allocator;
    size_t                           num_input_nodes = session_->GetInputCount();
    num_inputs_                                      = num_input_nodes;
    input_names_.reserve(num_input_nodes);
    input_shapes_.reserve(num_input_nodes);
    input_node_names_.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto        input_name = session_->GetInputNameAllocated(i, allocator);
        std::string str_tmp    = input_name.get();
        input_names_.push_back(str_tmp);

        Ort::TypeInfo type_info   = session_->GetInputTypeInfo(i);
        auto          tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes_.push_back(tensor_info.GetShape());

        // 检查动态输入形状
        if (input_shapes_[i][2] == -1 && input_shapes_[i][3] == -1) {
            is_dynamic_input_shape_ = true;
        }
    }

    for (unsigned int i = 0; i < num_input_nodes; ++i) {
        input_node_names_[i] = input_names_[i].c_str();
        std::cout << "输入节点 " << i << input_node_names_[i] << std::endl;
    }

    // 获取输出节点信息
    size_t num_output_nodes = session_->GetOutputCount();
    num_outputs_            = num_output_nodes;
    output_names_.reserve(num_output_nodes);
    output_shapes_.reserve(num_output_nodes);
    output_node_names_.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto        output_name = session_->GetOutputNameAllocated(i, allocator);
        std::string str_tmp     = output_name.get();
        output_names_.push_back(str_tmp);

        Ort::TypeInfo type_info   = session_->GetOutputTypeInfo(i);
        auto          tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_shapes_.push_back(tensor_info.GetShape());
    }

    for (unsigned int i = 0; i < num_output_nodes; i++) {
        output_node_names_[i] = output_names_[i].c_str();
        std::cout << "输出节点 " << i << output_node_names_[i] << std::endl;
    }
}

inline std::vector<Ort::Value> YoloOnnxBase::RunInference(const cv::Mat& input_tensor)
{
    // 创建输入张量
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo         memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    // std::vector<int64_t> input_shape = {1, input_tensor.channels(), input_tensor.rows, input_tensor.cols};
    std::vector<int64_t> input_shape = {1, 3, 640, 640};

    if (input_tensor.type() != CV_32F) {
        throw std::runtime_error("Input tensor must be of type CV_32F");
    }

    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_tensor.ptr<float>()), input_tensor.total() * input_tensor.channels(), input_shape.data(), input_shape.size()));

    // 执行推理
    try {
        return session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), input_tensors.data(), input_tensors.size(), output_node_names_.data(), output_names_.size());
    }
    catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        throw;   // 重新抛出异常
    }
}

inline DetectionResult YoloOnnxBase::Detect(const cv::Mat& image)
{
    // 预处理
    std::vector<int64_t> input_tensor_shape;
    cv::Mat              preprocessed = Preprocess(image, input_tensor_shape);

    // 推理
    auto output_tensors = RunInference(preprocessed);

    // 后处理
    return Postprocess(output_tensors, image.size(), preprocessed.size());
}
