#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "AlgoPreB.h"

REGISTER_ALGO(AlgoPreB)

AlgoPreB::AlgoPreB()
{

}

AlgoPreB::~AlgoPreB()
{

}

AlgoResultPtr AlgoPreB::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    Sleep(100);
    LOGI("AlgoPreB run finished!");
    return std::make_shared<stAlgoResult>();
}