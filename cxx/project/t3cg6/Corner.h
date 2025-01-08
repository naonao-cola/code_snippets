/**
 * @FilePath     : /t3cg6/src/project/Corner.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-12-03 16:35:29
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-03 16:36:41
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#pragma once
#ifndef NAONAO_CORNER_H
#    define NAONAO_CORNER_H

#    include "../../base/src/framework/BaseAlgo.h"
#    include "VBlob.h"
#    include <opencv2/opencv.hpp>
#    include <string.h>


class Corner : public BaseAlgo
{
public:
    Corner(){};
    ~Corner(){};
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    DCLEAR_ALGO_GROUP_REGISTER(Corner)

    /**
     * @brief: 获取参数
     * @return
     * @note :
     **/
    void getParam(InferTaskPtr task);


    /**
     * @brief  主函数
     * @param src
     * @param th_low
     * @param th_high
     * @param task
     * @param tmp_path
     * @return
     */
    bool img_process(cv::Mat src, AlgoResultPtr task, std::string tmp_path = "");

    int    low_th_     = 50;     // 低阈值
    int    high_th_    = 100;    // 高阈值
    double low_angle_  = 35.5;   // 低角度
    double high_angle_ = 47.5;   // 高角度
    int    open_width_ = 15;     // 开操作宽
    int    open_hight_ = 15;     // 开操作高
    int    blur_width_ = 3;      // 模糊宽
    int    blur_hight_ = 3;      // 模糊高
    int    grad_width_ = 5;      // 梯度宽
    int    grad_heigh_ = 5;      // 梯度高
    int    h_width_    = 50;     // 水平开操作的宽
    int    v_heigh_    = 50;     // 垂直开操作的高
};

#endif   // NAONAO_CORNER_H