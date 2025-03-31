#pragma once
#include "../framework/BaseAlgo.h"

class AlgoA : public BaseAlgo
{
public:
    AlgoA();
    ~AlgoA();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
private:
    DCLEAR_ALGO_GROUP_REGISTER(AlgoA)
};