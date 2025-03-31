#pragma once
#include <iostream>
#include "fmt/core.h"
#include "easylogging++.h"
#include "../framework/Defines.h"
#include "../framework/InferenceEngine.h"


enum class LogLv:int
{
    DEBUG_LV = 0,
    VERBOSE_LV,
    INFO_LV,
    WARNING_LV,
    ERROR_LV,
    FATAL_LV
};


#define ALOG(level, fmt_str, ...) \
{\
    std::string s = fmt::format(fmt_str, __VA_ARGS__);\
    if (InferenceEngine::get_instance()->IsReportLog()) { \
        InferenceEngine::get_instance()->ReportLog(level, fmt::format("[ALGO]: {}", s)); \
    } else {\
        if (level == LogLv::DEBUG_LV) \
            LOG(DEBUG) <<"ALGO: "<< s; \
        else if (level == LogLv::VERBOSE_LV) \
            LOG(TRACE) <<"ALGO: "<< s; \
        else if (level == LogLv::INFO_LV) \
             LOG(INFO) <<"ALGO: "<< s; \
        else if (level == LogLv::WARNING_LV) \
             LOG(WARNING) <<"ALGO: "<< s; \
        else if (level == LogLv::ERROR_LV) \
             LOG(ERROR) <<"ALGO: "<< s; \
        else if (level == LogLv::FATAL_LV) \
             LOG(FATAL) <<"ALGO: "<< s; \
    } \
}

#define LOGD(fmt_str, ...) ALOG(LogLv::DEBUG_LV, fmt_str, __VA_ARGS__)
#define LOGV(fmt_str, ...) ALOG(LogLv::VERBOSE_LV, fmt_str, __VA_ARGS__)
#define LOGI(fmt_str, ...) ALOG(LogLv::INFO_LV, fmt_str, __VA_ARGS__)
#define LOGW(fmt_str, ...) ALOG(LogLv::WARNING_LV, fmt_str, __VA_ARGS__)
#define LOGE(fmt_str, ...) ALOG(LogLv::ERROR_LV, fmt_str, __VA_ARGS__)
#define LOGF(fmt_str, ...) ALOG(LogLv::FATAL_LV, fmt_str, __VA_ARGS__)

