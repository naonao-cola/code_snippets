#pragma once
#include "../framework/BaseAlgo.h"

class AlgoB : public BaseAlgo
{
public:
    AlgoB();
    ~AlgoB();
    AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
private:
    DCLEAR_ALGO_GROUP_REGISTER(AlgoB)
};