#pragma once
#include "../framework/BaseAlgo.h"

class AlgoPreB : public BaseAlgo
{
public:
    AlgoPreB();
    ~AlgoPreB();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
private:
    DCLEAR_ALGO_GROUP_REGISTER(AlgoPreB)
};