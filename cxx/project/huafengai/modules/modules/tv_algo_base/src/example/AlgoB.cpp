#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "AlgoB.h"

REGISTER_ALGO(AlgoB)

AlgoB::AlgoB()
{

}

AlgoB::~AlgoB()
{

}

AlgoResultPtr AlgoB::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("AlgoB start run!");
    Sleep(50);
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->result_info = {
        {"label", "AlgoB_defect"},
        {"shapeType", "rectangle"},
        {"points", {{30.0, 25.0}, {90.0, 150.0}}},
        {"result", {{"confidence", 0.98}, {"area", 10000}}},
    };
    LOGI("AlgoB run finished!");
    return algo_result;
}