/**
 * @FilePath     : /tray_algo/src/custom/trayCharRec.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2024-06-20 14:21:43
 **/
#pragma once
#include <AIRuntimeDataStruct.h>
#include "../framework/DataStruct.h"
#include "../framework/OcrBaseAlgo.h"
namespace fs = std::filesystem;

class trayCharRec : public OcrBaseAlgo {
public:
    trayCharRec();
    ~trayCharRec();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    /**
     * @brief             字符串过滤与修正
     * @param ocrdata_list
     * @param params
     */
    void FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params);
    cv::Point p1_, p2_;
    DCLEAR_ALGO_GROUP_REGISTER(trayCharRec)
};