#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "AlgoPreA.h"

REGISTER_ALGO(AlgoPreA)

AlgoPreA::AlgoPreA()
{

}

AlgoPreA::~AlgoPreA()
{

}

AlgoResultPtr AlgoPreA::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("AlgoPreA run finished!");
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status = RunStatus::OK;

    return algo_result;
}