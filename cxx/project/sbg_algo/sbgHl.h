/**
 * @FilePath     : /sbg_algo/src/custom/sbgHl.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-09-06 10:41:09
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-09-06 10:41:09
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#pragma once
#include "../framework/BaseAlgo.h"

enum ENUM_ERROR_CODE
{
    tb = 0,
    ibj,
    marly,
    wa_gray_s,
    wa_black_s,
    screw_head,
    leak_hole,
    bs,
    gwb,
};

#define SAVE_DEBUG

class sbgHl : public BaseAlgo
{
public:
    sbgHl();
    ~sbgHl();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

    //条巴
    void tbfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //绝缘黑胶
    void ibgfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //marly
    void marlyfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);

    //灰色防水胶
    void wgsfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result,json& BOXdata, cv::Mat& img);
    //黑色防水胶
    void wbsfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //接头螺丝
    void screw_headfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //漏水孔
    void leak_holefilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //黑色海绵
    void bsfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);
    //地线巴
    void gwbfilter(InferTaskPtr task, json& indata, AlgoResultPtr algo_result, json& BOXdata, cv::Mat& img);

private:
    bool isPointInsideRectangle(int pointX, int pointY, int rectLeft, int rectTop, int rectRight, int rectBottom);
    bool isPointInsideRectangle(cv::Rect lhs,cv::Rect rhs);
    void write_debug_img(InferTaskPtr task, std::string name, cv::Mat img);

    DCLEAR_ALGO_GROUP_REGISTER(sbgHl)
};