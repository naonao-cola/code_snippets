#include <windows.h>
#include "../utils/logger.h"
#include "../utils/Utils.h"
#include "trayCharRec.h"
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeInterface.h>
#include <AIRuntimeUtils.h>
#include "../utils/Utils.h"
#include "JsonHelper.h"
#include "param_check.h"
#include "algo_tool.h"

REGISTER_ALGO(trayCharRec)

trayCharRec::trayCharRec() {}

trayCharRec::~trayCharRec() {}

AlgoResultPtr trayCharRec::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status = ErrorCode::OK;
    algo_result->result_info.push_back(
        {
            {"label", "trayCharRec"},
            {"shapeType", "rectangle"},
            {"points", {{0, 0}, {0, 0}}},
            {"result", {{"confidence", 0}, {"area", 0}}},
        });
    json params = GetTaskParams(task);
    cv::Mat src;
    cv::resize(task->image, src, cv::Size(task->image.cols / 4, task->image.rows / 4));
    if (src.channels() < 3) {
        cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
    }
    cv::Scalar mean_value = cv::mean(src);
    if (mean_value[0] < 130) {
        cv::Mat src2 = cv::Mat::zeros(src.size(), src.type());
        cv::Mat dst;
        cv::addWeighted(src, 1.25, src2, 0, 70, dst);
        src = dst.clone();
    }
    std::vector<OcrDataPtr> ocrdata_list;
    TextDet(src, ocrdata_list);
    TextRec(src, ocrdata_list);
    FilterRecResult(ocrdata_list, params);
    algo_result->result_info = json::array();
    for (auto ocrdata : ocrdata_list) {
        if (ocrdata->labelName == "CNG") {
            algo_result->judge_result = 0;
        }
        algo_result->result_info.emplace_back(ocrdata->ToJsonResult());
    }
    LOGD("trayCharRec run finished!");
    return algo_result;
}

void trayCharRec::FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params)
{
    auto it = ocrdata_list.begin();
    while (it != ocrdata_list.end()) {
        OcrDataPtr ocrdata = *it;
        std::set<char> firstCharSet = {'1'};
        if (ocrdata->text.length() > 9 && (ocrdata->text[0] == '1' || ocrdata->text[0] == '[')) {
            LOGD("trayCharRec skip result: {}, len:{}, score:{}", ocrdata->text, ocrdata->text.length(), ocrdata->conf);
            int length = ocrdata->text.length();
            ocrdata->labelName = ocrdata->text.substr(1, length - 1);
            ++it;
        } else if (ocrdata->text.length() < 9) {
            ocrdata->labelName = "CNG";
            ++it;
        } else {
            ocrdata->labelName = ocrdata->text;
            ++it;
        }
    }
}
