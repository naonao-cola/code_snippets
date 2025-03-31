#pragma once
#include <iostream>
#include <vector>
#include "BaseAlgo.h"
#include "nlohmann/json.hpp"
#include "ErrorDefine.h"

using json = nlohmann::json;



class ThreadPool;
class InferenceEngine;

//
class BaseAlgoGroup
{
public:
    BaseAlgoGroup();
    virtual ~BaseAlgoGroup();

    ErrorCode SetParams(const json& algo_group_cfg);
    virtual FinalResultPtr RunGroup(InferTaskPtr task);
    // virtual json MergeResults(InferTaskPtr task);
    std::string GetTypeID() { return m_type_id; };
    std::string GetTypeName() { return m_type_name; };


private:
    ErrorCode SetAlgoParam(const json& algo_param, bool is_preprocess);
private:
    DCLEAR_ALGO_GROUP_REGISTER(BaseAlgoGroup)
    std::vector<BaseAlgo*> m_algo_list;
    std::vector<BaseAlgo*> m_pre_list;
    std::string m_type_name;
    std::string m_type_id;
};