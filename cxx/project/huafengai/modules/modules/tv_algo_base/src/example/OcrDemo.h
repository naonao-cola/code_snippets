#pragma once
#include <AIRuntimeDataStruct.h>
#include "../framework/DataStruct.h"
#include "../framework/OcrBaseAlgo.h"



class OcrDemo : public OcrBaseAlgo
{
public:
    OcrDemo();
    virtual ~OcrDemo();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

protected:
    virtual void FilterDetResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params = {});
    virtual void FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params = {});
private:
    DCLEAR_ALGO_GROUP_REGISTER(OcrDemo)
};
