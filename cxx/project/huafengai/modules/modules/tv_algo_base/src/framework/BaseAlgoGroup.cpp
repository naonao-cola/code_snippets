#include "../utils/Utils.h"
#include "../utils/logger.h"
#include "BaseAlgoGroup.h"
#include "AlgoManager.h"

BaseAlgoGroup::BaseAlgoGroup()
{
}

BaseAlgoGroup::~BaseAlgoGroup()
{
}

// 设置算法组的参数，如果第一次调用则创建相关算法对象
ErrorCode BaseAlgoGroup::SetParams(const json &algo_group_cfg)
{
    m_type_id = algo_group_cfg["type_id"];
    m_type_name = algo_group_cfg["type_name"];
    json pre_algo_param_list = algo_group_cfg["pre_algo_params"];
    json algo_param_list = algo_group_cfg["algo_params"];

    ErrorCode e_code = ErrorCode::OK;
    for (auto pre_algo_param : pre_algo_param_list)
    {
        e_code = SetAlgoParam(pre_algo_param, true);
        CHECK_ERRORCODE_RETURN(e_code)
    }

    for (auto algo_param : algo_param_list)
    {
        e_code = SetAlgoParam(algo_param, false);
        CHECK_ERRORCODE_RETURN(e_code)
    }

    // 预处理算法按照算法编号“algo_index”进行排序，串行执行
    std::sort(m_pre_list.begin(), m_pre_list.end(), [=](BaseAlgo *a, BaseAlgo *b) { return a->GetAlgoIndex(m_type_id) < b->GetAlgoIndex(m_type_id); });

    return ErrorCode::OK;
}

/** 设置算法参数，如果是第一次设置则创建算法对象，否则更新算法参数
* @algo_param：算法参数json
* @is_preprocess: 是否是预处理算法
*/
ErrorCode BaseAlgoGroup::SetAlgoParam(const json &algo_param, bool is_preprocess)
{
    std::string algo_name = Utils::GetProperty(algo_param, "algo_name", std::string("unknown"));
    if (algo_name == "unknown")
    {
        LOGE("Algo parameter wrong. Must specify [algo_name]!")
        return ErrorCode::WRONG_PARAM;
    }

    // 如果算法对象不存在则创建后再设置参数
    BaseAlgo *pAlgo = AlgoManager::get_instance()->GetAlgo(algo_name);
    if (pAlgo == nullptr)
    {
        pAlgo = AlgoManager::CreateAlgo(algo_name);

        if (pAlgo == nullptr)
        {
            LOGE("Create [Algo] fail. Invalid algo name:{}", algo_name);
            return ErrorCode::WRONG_PARAM;
        }

        AlgoManager::get_instance()->AddAlgo(pAlgo);
    }

    if (is_preprocess)
        m_pre_list.emplace_back(pAlgo);
    else
        m_algo_list.emplace_back(pAlgo);

    // 设置对应type_id的算法参数
    pAlgo->SetParam(m_type_id, algo_param);

    return ErrorCode::OK;
}

FinalResultPtr BaseAlgoGroup::RunGroup(InferTaskPtr task)
{
    FinalResultPtr final_result = std::make_shared<stFinalResult>();
    final_result->image_info = task->image_info;
    final_result->results = {
        {"class_list", json::array()},
        {"status", "OK"},
        {"judge_result", 1},
        {"shapes", json::array()}
    };

    std::vector<AlgoResultPtr> pre_results;
    for (auto preAlgo : m_pre_list)
    {
        AlgoResultPtr pre_result = preAlgo->RunAlgo(task);
        if (pre_result->status != RunStatus::OK)
        {
            final_result->results["status"] = Utils::GetStatusCode(pre_result->status);
            return final_result;
        }
        pre_results.emplace_back(pre_result);
    }

    std::vector<std::future<AlgoResultPtr>> future_results;
    InferenceEngine *pEngine = InferenceEngine::get_instance();

    for (auto algo : m_algo_list)
    {
        future_results.emplace_back(pEngine->GetThreadPool()->enqueue([=, &pre_results]()
                                                                        { return algo->RunAlgo(task, pre_results); }));
    }

    for (auto && result : future_results)
    {
        LOGI("--------- Wait algo Result ----------------------");
        AlgoResultPtr ar = result.get();
        if (ar->status != RunStatus::OK) {
            final_result->results["status"] = Utils::GetStatusCode(ar->status);
            return final_result;
        }
        if (ar->result_info.empty()) {
            final_result->results["judge_result"] = 0;
        }
        if (ar->judge_result==0) {
            final_result->results["judge_result"] = 0;
        }
        for (json shape : ar->result_info) {
            final_result->results["shapes"].emplace_back(shape);
            if (ar->judge_result == 0) {
                final_result->results["judge_result"] = 0;
            }
        }
    }

    std::string img_name = task->image_info["img_name"];
    LOGI("AlgoGroup:[{}] RunGroup Complete!", m_type_name);
    return final_result;
}

