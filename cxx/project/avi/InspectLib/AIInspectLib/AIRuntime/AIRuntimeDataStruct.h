#ifndef __AIRUNTIMEDATASTRUCT_H__
#define __AIRUNTIMEDATASTRUCT_H__
#include <memory>
#include <opencv2/opencv.hpp>
#include <future>
#include "AIRuntimeUtils.h"

struct stAIConfigInfo {
    bool    usePinMemory{ true };
    size_t  workSpaceSize{ 2048 };
    size_t  GPUCachSize{ 1024 };
    size_t  CPUCachSize{ 1024 };
    int     preProcessThreadCnt{ 8 };
    int     preProcessThreadPriority{ 1 };
    int     inferThreadCnt{ 8 };
    int     inferThreadPriority{ 1 };
    stAIConfigInfo() = default;
    stAIConfigInfo(json config) {
        usePinMemory = get_param<bool>(config, "usePinMemory", usePinMemory);
        workSpaceSize = get_param<size_t>(config, "workSpaceSize", workSpaceSize);
        GPUCachSize = get_param<size_t>(config, "GPUCachSize", GPUCachSize);
        CPUCachSize = get_param<size_t>(config, "CPUCachSize", CPUCachSize);
        preProcessThreadPriority = get_param<int>(config, "preProcessThreadPriority", preProcessThreadPriority);
        preProcessThreadCnt = get_param<int>(config, "preProcessThreadCnt", preProcessThreadCnt);
        inferThreadCnt = get_param<int>(config, "inferThreadCnt", inferThreadCnt);
        inferThreadPriority = get_param<int>(config, "inferThreadPriority", inferThreadPriority);
    }
};

enum eAIErrorCode
{
    E_OK = 0,
    E_OUT_OF_MEMORY,
    E_CREATE_MODEL_FAILED,
    E_FILE_NOT_EXIST,
    E_QUEUUE_FULL,
};

enum eAIAlgoType : int {
    Classfication,
    ObjectDetection,
    Segmentation
};

struct stAIInferParam {
    std::vector<int> gpuId{ 0 };
    int maxBatchSize{ 1 };
    float confidenceThreshold{ 0.0 };
    float nmsThreshold{ 1.0 };
    int maxObjectNums{ 1024 };
    stAIInferParam() = default;
    stAIInferParam(const json& info) {
        maxBatchSize = get_param<int>(info, "maxBatchSize", maxBatchSize);
        confidenceThreshold = get_param<float>(info, "confidenceThreshold", confidenceThreshold);
        nmsThreshold = get_param<float>(info, "nmsThreshold", nmsThreshold);
        maxObjectNums = get_param<int>(info, "maxObjectNums", maxObjectNums);
    }
};

struct stAIModelInfo {
    typedef std::shared_ptr<stAIModelInfo> mPtr;
    int modelVersion{ 1 };
    int modelId{ 0 };
    eAIAlgoType algoType{ eAIAlgoType::ObjectDetection };
    std::string modelName{ "" };
    std::string modelPath{ "" };  // model file path
    stAIInferParam inferParam;
    stAIModelInfo() = default;
    stAIModelInfo(const json& info) {
        modelVersion = get_param<int>(info, "modelVersion", modelVersion);
        modelId = get_param<int>(info, "modelId", modelId);
        modelName = get_param<std::string>(info, "modelName", modelName);
        modelPath = get_param<std::string>(info, "modelPath", modelPath);
        inferParam = stAIInferParam(info["InferParam"]);
    }

    std::string ModelInfo() {
        char buff[400];
        std::string rst = " \n ============================================================================";
        sprintf_s(buff, "%s\n model version:\t\t %d \n model id:\t\ %d \n algoType:\t\t %d \n modelPath:\t\t%s %s", 
                rst.c_str(), modelVersion, modelId, (int)algoType, modelPath.c_str(), rst.c_str());
        return std::string(buff);
    }
};

struct stImageBuf
{
    int width;
    int height;
    int channel;
    unsigned char* data;
};

struct stTaskInfo
{
    int modelId;
    int taskId;
    int orgImageId;
    TimeCost tt;
    long long preCostTime;
    long long inferCostTime;
    long long hostCostTime;
    long long totalCostTime;
    std::shared_ptr<void> inspParam;
    std::vector<cv::Mat> imageData;
    void* promiseResult{ nullptr };
    std::string Info() {
        // std::string rst = " \n ============================================================================";
        char buff[200];
        sprintf_s(buff, "\n model id:\t\t%d\n image size:\t\t%d", modelId, imageData.size());
        /*std::string rst = "";
        rst += fmt::format("\n model id:\t\t{}", modelId);
        rst += fmt::format("\n image size:\t\t{}", imageData.size());*/
        return std::string(buff);
    }
};

struct stPoint
{
    float x;
    float y;
    stPoint(float x_, float y_) : x(x_), y(y_) {}
};

struct stResultItem
{
    int code;
    int shape;
    float confidence;
    std::vector<stPoint> points;
    std::string Info() {
        // std::string rst = " \n ============================================================================";
        std::string rst = "";
      /*  rst += fmt::format("\n code:\t\t{}", code);
        rst += fmt::format("\n shape:\t\t{}", shape);
        rst += fmt::format("\n confidence:\t\t{}", confidence);*/
        char buff[200];
        sprintf_s(buff, "\n code:\t\t %d \n shape:\t\t %d\n confidence:\t\t %g", code, shape, confidence);
        if (points.size() == 2) {
            //rst += fmt::format("\n points:\t\t[{}  {}  {}  {}]", points[0].x, points[0].y, points[1].x, points[1].y);
            sprintf_s(buff, "%s\n points:\t\t[%g  %g  %g  %g]", buff, points[0].x, points[0].y, points[1].x, points[1].y);
        }
        else {
            rst += "[]";
        }
        rst = std::string(buff);
        // rst += fmt::format("\n ============================================================================");
        return rst;
    }
};

struct stModelResult
{
    std::shared_ptr<stTaskInfo> taskInfo;
    std::vector<std::vector<stResultItem>> itemList;
};

struct stGPUInfo {
    int    gpuId;
    size_t totalMemorySize;
    size_t usedMemorySize;
    size_t avaliableMemorySize;
    float  gpuUtilRate;
};

using ModelResultPtr = std::shared_ptr<stModelResult>;
using TaskInfoPtr = std::shared_ptr<stTaskInfo>;

#endif // __DATA_STRUCT_H__