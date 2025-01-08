#pragma once
#include <AIRuntimeDataStruct.h>
#include "../framework/DataStruct.h"
#include "../framework/OcrBaseAlgo.h"



class LabelOcr : public OcrBaseAlgo
{
public:
    LabelOcr();
    virtual ~LabelOcr();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

protected:
    virtual void FilterDetResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params = {});
    virtual void FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params = {});
    void align_results(AlgoResultPtr algo_result);

    int IncludeChinese(char *str);
    void is_zero(cv::Mat img,std::vector<OcrDataPtr>& ocrdata_list);
private:
    DCLEAR_ALGO_GROUP_REGISTER(LabelOcr)
};