#include <windows.h>
#include "../utils/logger.h"
#include "../utils/Utils.h"
#include "OcrDemo.h"
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>


REGISTER_ALGO(OcrDemo)

OcrDemo::OcrDemo()
{
}

OcrDemo::~OcrDemo()
{
}

void OcrDemo::FilterDetResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params)
{
    // auto it = ocrdata_list.begin();
    // while (it != ocrdata_list.end()) {
    //     OcrDataPtr ocrdata = *it;
    //     if (ocrdata->GetAspectRatio() < 6) {
    //         it = ocrdata_list.erase(it);
    //     } else {
    //         ++it;
    //     }
    // }
}

void OcrDemo::FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params)
{
    auto it = ocrdata_list.begin();
    while (it != ocrdata_list.end()) {
        OcrDataPtr ocrdata = *it;
        std::set<char> firstCharSet = {'A', 'B', 'C'};
        if (ocrdata->text.length() < 12 ||
            firstCharSet.find(ocrdata->text[0]) == firstCharSet.end()) {
            LOGD("OcrDemo skip result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);
            it = ocrdata_list.erase(it);
        } else {
            ocrdata->labelName = "Catalog";
            ++it;
        }
    }
}

AlgoResultPtr OcrDemo::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    // LOGI("OcrDemo start run!");
    json params = GetTaskParams(task);
    bool is_debug = Utils::GetProperty(params, "debug", 0);

    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();

    // cv::Mat gray;
    // cv::cvtColor(task->image, gray, 16544);

    std::vector<OcrDataPtr> ocrdata_list;
    TextDet(task->image, ocrdata_list);
    // FilterDetResult(ocrdata_list);
    TextRec(task->image, ocrdata_list);
    // FilterRecResult(ocrdata_list);

    algo_result->result_info = json::array();
    for (auto ocrdata : ocrdata_list) {
        algo_result->result_info.emplace_back(ocrdata->ToJsonResult());
    }

    // LOGI("OcrDemo run finished! {}", algo_result->result_info.dump(2));
    return algo_result;
}
