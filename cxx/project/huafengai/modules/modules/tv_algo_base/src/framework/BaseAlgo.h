#pragma once

#include <iostream>
#include "nlohmann/json.hpp"
#include "ErrorDefine.h"
#include "Defines.h"
#include "AlgoManager.h"

using json = nlohmann::json;


// 算法基类
class BaseAlgo
{
public:
    BaseAlgo() {};
    virtual ~BaseAlgo() {};

    // 设置算法参数，每个type_id(对应一类图片)有一套单独的算法参数
    virtual void SetParam(const std::string type_id, const json& params);
    // 运行算法，task是推理的任务对象，包含图像和相关信息。pre_results是预处理算法输出的结果
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results = std::vector<AlgoResultPtr>());
    // 推理引擎退出时会调用该方法，如果有需要释放的资源可以在该方法中进行
    virtual void Destroy() {};
    int GetAlgoIndex(const std::string type_id);
    inline std::string GetName() { return m_name; }
    json GetTaskParams(InferTaskPtr task);

    bool IsDebug();
protected:
    static const bool registered;
    // 每个算法可以有多套参数，对应不同的图片
    std::map<std::string, json> m_param_map;
protected:
    std::mutex m_mtx;
    std::string m_name;
};