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
    // Sleep(100);
    LOGI("AlgoPreA run finished!");
    return std::make_shared<stAlgoResult>();
}