/**
 * @FilePath     : /t3cg6/src/project/LL.h
 * @Description  : 左长边
 * @Author       : naonao
 * @Date         : 2024-12-03 15:50:40
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-04 13:41:49
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#pragma once
#ifndef NAONAO_LL_H
#    define NAONAO_LL_H

#    include "../../base/src/framework/BaseAlgo.h"
#    include "VBlob.h"
#    include <opencv2/opencv.hpp>
#    include <string.h>

class RL : public BaseAlgo
{
public:
    RL(){};
    ~RL(){};
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    DCLEAR_ALGO_GROUP_REGISTER(RL)

    /**
     * @brief: 获取参数
     * @return
     * @note :
     **/
    void getParam(InferTaskPtr task);

    std::vector<cv::Rect>         img_process(cv::Mat src, InferTaskPtr task, AlgoResultPtr algo_result);

    std::vector<STRU_DEFECT_ITEM> judgement_vec;    // 判断规则
    VBlob                         blobexec_;        // 特征提取器
    int                           start_pix_ = 0;   // 边缘开始像素
    int                           end_pix_   = 0;   // 边缘结束像素
    int                           blur_w_      = 5;
    int                           blur_h_      = 5;
    int                           th_high_     = 255;
    int                           th_low_      = 0;
    int                           blur_length_ = 200;
    int                           continuous_  = 100;
    int                           th_value_    = 100;
    int                           diff_th_value_ = 100;
};

#endif   // NAONAO_LL_H