#pragma once

#include <chrono>
#include "nlohmann/json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define USE_LICENSE 1

using json = nlohmann::json;
using TimePoint = std::chrono::system_clock::time_point;

#define ALGO_GROUP_SUFFIX "AlgoGroup"

#ifndef DECLARE_SINGLETON
#define DECLARE_SINGLETON(class_name) \
    static class_name* get_instance() { \
        static class_name instance; \
        return &instance; \
    }
#endif // !DECLARE_SINGLETON

#ifndef IMPLEMENT_SINGLETON
#define IMPLEMENT_SINGLETON(class_name) \
    class_name(); \
    ~class_name(); \
    class_name(const class_name&) = delete; \
    class_name& operator=(const class_name&) = delete;
#endif // !IMPLEMENT_SINGLETON

#define DCLEAR_ALGO_GROUP_REGISTER(name) static const bool name##_ag_registered;
#define DCLEAR_ALGO_REGISTER(name) static const bool name##_a_registered;
#define REGISTER_ALGO_GROUP(name) const bool name::name##_ag_registered = (AlgoManager::RegisterAlgoGroup(#name, [](){ return new name(); }), true);
#define REGISTER_ALGO(name) const bool name::name##_ag_registered = (AlgoManager::RegisterAlgo(#name, [](){ return new name(); }), true);


struct stTaktTime
{
	TimePoint startTime;
	TimePoint endTime;
	long long costTimeMs;
};

// 算法运行状态
enum class RunStatus: int
{
    OK = 0,                     // 正常运行
    ABNORMAL_IMAGE,             // 图片异常
    ABNORMAL_IMG_BRIGHTNESS,  // 图片亮度异常
    ABNORMAL_ANGLE,             // 角度异常
    NOT_FOUND_TARGET,           // 目标查找失败
    OUT_OF_MEMORY,              // 内存不足
    WRONG_PARAM,                // 参数异常
    NOT_READY,                  // 未准备好（未初始化）
    TIME_OUT,                   // 检测超时
    DUPLICATE_ALGO_NAME,        // 算法名重复
    INVALID_IMG_DATA,           // 图片buffer异常，转换cvMat失败
    QUEUE_OVERFLOW,             // 队列溢出
    WRONG_STATE,                // 状态错误
    LICENSE_ERROR,              // 授权检查失败
    AI_INIT_FAIL,               // AI 初始化Runtime失败
    AI_LOAD_MODEL_FAIL,         // AI 加载模型失败
    AI_INFER_FAIL,              // AI 推理失败
    INVALID_HANDLE,             // 非法句柄
    UNKNOWN_ERROR = 999         // 未知错误
};


// 推理任务
struct stInferTask
{
    unsigned char* img_data;    // 图像buffer数据
    unsigned char* img_data2;
    cv::Mat image;
    cv::Mat image2;
    json image_info;            // 图像信息
    json image_info2;
};

// 算法结果
struct stAlgoResult
{
    RunStatus status = RunStatus::OK;
    int judge_result = 1;
    json result_info;                   // 算法结果信息，格式参考飞书文档
    std::vector<cv::Mat> result_imgs;   // 算法输出图片列表
    stTaktTime tt;                      // 算法运行时间
};

// 图像最终结果
struct stFinalResult
{
    json image_info;    // 图像信息，和InferTask一致
    json results;       // 图像最终结果，格式参考飞书文档
    stTaktTime tt;      // 算法运行时间
};

using InferTaskPtr = std::shared_ptr<stInferTask>;
using AlgoResultPtr = std::shared_ptr<stAlgoResult>;
using FinalResultPtr = std::shared_ptr<stFinalResult>;

