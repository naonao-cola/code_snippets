#pragma once
#include <iostream>
#include <condition_variable>
#include <mutex>

#include "../utils/ThreadPool.h"
#include "../utils/SafeQueue.h"
#include "Defines.h"
#include "nlohmann/json.hpp"
#include "AlgoManager.h"

using json = nlohmann::json;

enum class LogLv;
enum class ErrorCode;
class BaseAlgo;
class BaseAlgoGroup;

// 结果回调函数原型
typedef void(*ResultCallbackFunc)(const char* img_info_json, const char* result_json);
// 日志回调函数原型
typedef void(*LogCallbackFunc)(int level, const char* log_msg);


// 推理引擎类，负责引擎的初始化、配置、算法运行、调度等
class InferenceEngine
{
public:
    DECLARE_SINGLETON(InferenceEngine)

    // 系统配置
    ErrorCode ConfigSystemParams(const json& common_cfg);
    // 算法配置
    ErrorCode ConfigAlgoParams(const json& algo_all_cfg);
    // 提交算法任务，异步执行
    ErrorCode CommitInferTask(InferTaskPtr task);
    // 注册结果回调函数
    void RegisterResultCallback(ResultCallbackFunc callback);
    // 注册日志回调函数
    void RegisterLogCallback(LogCallbackFunc callback);
    // 销毁资源
    void Destroy();

    // 是否上报日志（如果有注册日志回调就上报，否则本地输出日志）
    inline bool IsReportLog() { return m_log_callback != nullptr; }
    // 通过日志回调函数上报日志
    inline void ReportLog(LogLv level, std::string msg);
    inline json& GetCommonCfg() {return m_common_cfg;}
    inline json& GetAlgoCfg() {return m_algo_cfg;}
    inline ThreadPool* GetThreadPool() { return m_algo_thread_pool; }

    ErrorCode Init();
    ErrorCode LicenseVerify();

private:
    IMPLEMENT_SINGLETON(InferenceEngine)

    FinalResultPtr RunTask(InferTaskPtr task);
    void ResultCallbackWorker();


private:
    AlgoManager* m_algo_mgr;
    tvalgo::SafeQueue<std::future<FinalResultPtr>> m_future_results;

    std::mutex m_queue_lock;
    std::condition_variable m_cond;
    std::atomic_bool m_stopped;
    std::thread* m_result_thread;
    ThreadPool* m_task_thread_pool;
    ThreadPool* m_algo_thread_pool;
    ResultCallbackFunc m_result_callback;
    LogCallbackFunc m_log_callback;
    json m_common_cfg;
    json m_algo_cfg;
    LogLv m_log_level;
};