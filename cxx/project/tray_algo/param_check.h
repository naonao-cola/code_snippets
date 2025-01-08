/**
 * @FilePath     : /tray_algo/src/custom/param_check.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2024-06-20 14:19:53
 **/
#pragma once
#include <sstream>
#include "../framework/Defines.h"
#include "../utils/Logger.h"

#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(__clang__)
static inline bool(likely)(bool x)
{
    return __builtin_expect((x), true);
}
static inline bool(unlikely)(bool x)
{
    return __builtin_expect((x), false);
}
#else
static inline bool(likely)(bool x)
{
    return x;
}
static inline bool(unlikely)(bool x)
{
    return x;
}
#endif

#define TVALGO_FUNCTION_BEGIN                                     \
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>(); \
    algo_result->status = ErrorCode::OK;                          \
    LOGD("algo start run file {}, line {}", __FILE__, __LINE__);

#define TVALGO_FUNCTION_END                                                                                       \
    LOGD("algo end run file {}, line {} info {}", __FILE__, __LINE__, Utils::DumpJson(algo_result->result_info)); \
    LOGD("algo end run file {}, line {}", __FILE__, __LINE__);                                                    \
    return algo_result;

#define TVALGO_FUNCTION_RETURN_ERROR_PARAM(info)                                                                  \
    LOGD("algo end run file {}, line {} info {}", __FILE__, __LINE__, info);                                      \
    algo_result->status = ErrorCode::WRONG_PARAM;                                                                 \
    LOGD("algo end run file {}, line {} info {}", __FILE__, __LINE__, Utils::DumpJson(algo_result->result_info)); \
    return algo_result;

#define TVALGO_FUNCTION_LOG(info) \
    LOGD("algo log run file {}, line {} info {}", __FILE__, __LINE__, info);

#define START_TIMER \
    double start_time = clock()

#define END_TIMER                                               \
    double end_time = clock();                                  \
    double duration = (end_time - start_time) / CLOCKS_PER_SEC; \
    LOGD("[{}] \t [{}]msc.", __FUNCTION__, duration);

/**
 * @brief json 获取默认参数
 * @param name
 * @param val
 * @param valSet
 * @param canAny
 * @return
 */
bool InIntSet(const std::string& name, int val, std::set<int> valSet, bool canAny);
bool InStringSet(const std::string& name, std::string val, std::set<std::string> valSet, bool canEmpty);
bool InDoubleRange(const std::string& name, double val, double minVal, double maxVal, bool canAny);
bool InIntRange(const std::string& name, int val, int minVal, int maxVal, bool canAny);
bool PositiveInt(const std::string& name, int val, bool canAny);
