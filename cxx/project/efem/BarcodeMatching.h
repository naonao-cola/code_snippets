#pragma once
#include "../framework/BaseAlgo.h"
#include <filesystem> // C++17
namespace fs = std::filesystem;
#define SAVE_

class BarcodeMatching : public BaseAlgo
{
public:
    BarcodeMatching();
    ~BarcodeMatching();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
    void result_to_json(json& result_info, std::string result);
    cv::Mat GetRotateCropImage(const cv::Mat& srcimage, std::vector<std::vector<int>> box);
    bool is_match(std::string Rcode, std::string Ccode);
private:
    void write_debug_img(std::string name, cv::Mat img);
    bool isAbnormal(InferTaskPtr task);

    DCLEAR_ALGO_GROUP_REGISTER(BarcodeMatching)
    bool saveImg = true;

};