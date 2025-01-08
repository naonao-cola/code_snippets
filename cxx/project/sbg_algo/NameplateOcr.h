#pragma once
#include <AIRuntimeDataStruct.h>
#include "../framework/DataStruct.h"
#include "../framework/OcrBaseAlgo.h"


class NameplateOcr : public OcrBaseAlgo
{
public:
    NameplateOcr();
    ~NameplateOcr();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

    enum NUM_CODE
    {
        CCC = 0,
        ASTA,
        CE,
        KEMA,
        UL
    };
protected:
    virtual void FilterDetResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params = {});
    virtual void FilterRecResult(std::vector<OcrDataPtr>& ocrdata_list, const json& params = {});

    static std::string RemoveSpacesAndPunctuation(const std::string& str);
    void CheckKeyWords(OcrDataPtr ocrdata, std::map<std::string, OcrDataPtr>& keyItems, bool is_english);

    void align_results(AlgoResultPtr algo_result);
    std::string print_rec_str(AlgoResultPtr algo_result);

    void shape_det(cv::Mat img,AlgoResultPtr algo_result );
    void is_zero(cv::Mat img,OcrDataPtr ocrdata_list);
    int sauvola(const cv::Mat& src, cv::Mat& dst, const double& k, const int& wnd_size);
    cv::Mat zero_img_;
private:
    DCLEAR_ALGO_GROUP_REGISTER(NameplateOcr)

    //标签识别的部分
};

