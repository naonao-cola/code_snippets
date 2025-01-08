#pragma once
#include "../framework/BaseAlgo.h"


class PreBarcodeMatching : public BaseAlgo
{
public:
    PreBarcodeMatching();
    ~PreBarcodeMatching();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map);

private:
    DCLEAR_ALGO_GROUP_REGISTER(PreBarcodeMatching)
};