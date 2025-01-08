#pragma once
#include "../framework/BaseAlgo.h"

enum ENUM_ERROR_CODE
{
    KD = 0,
    HJ,
    TB,
    DXB,
    FSJ,
    AXFSJ,
    ML,
    HSFHJ,
    LS,
    BOX,
};

#define SAVE_DEBUG

class sbgObjDetec : public BaseAlgo
{
public:
    sbgObjDetec();
    ~sbgObjDetec();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
    //
    void KDfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //
    void HJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //
    void TBfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);

     void DXBfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result,
                  json& BOXdata, cv::Mat& img);
    //
    void FSJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    
     //
    void AXFSJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result,
                     json& BOXdata, cv::Mat& img);
    //marlyƬ
    void MLfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //
    void HSFHJfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //
    void LSfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
   
private:
    bool isPointInsideRectangle(int pointX, int pointY, int rectLeft, int rectTop, int rectRight, int rectBottom);
    bool isPointInsideRectangle(cv::Rect lhs, cv::Rect rhs);
    void write_debug_img(InferTaskPtr task, std::string name, cv::Mat img);
    DCLEAR_ALGO_GROUP_REGISTER(sbgObjDetec)
};