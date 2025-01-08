#pragma once
#include "../framework/BaseAlgo.h"

class AlgoPreA : public BaseAlgo
{
public:
    AlgoPreA();
    ~AlgoPreA();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
private:
    DCLEAR_ALGO_GROUP_REGISTER(AlgoPreA)
};