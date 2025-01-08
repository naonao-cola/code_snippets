/**
 * @FilePath     : /connector/src/custom/Curved_Bow_Train.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-01-08 11:53:39
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#pragma once
#include "../framework/BaseAlgo.h"
#include "./sub_3rdparty/tival/include/ShapeBasedMatching.h"
class Curved_Bow_Train : public BaseAlgo {
public:
    Curved_Bow_Train();
    ~Curved_Bow_Train();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);


    std::tuple<std::string, json>get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map);



    void train_process(const cv::Mat& src, const cv::Mat& mask,AlgoResultPtr algo_result);



private:
    DCLEAR_ALGO_GROUP_REGISTER(Curved_Bow_Train)

    
    //训练模板参数    
    int num_levels_ = 3;
    int angle_min_ = -10;
    int angle_max_ = 10;
    double min_score_ = 0.5;
    int contrast_ = 30;
    std::string path_ = "";

    //追加参数
    int num_ = 100;
    double similarity_ = 0.55;
    double scale_min_ = 0.8;
    double scale_max_ = 1.2;
    double greediness_ = 0.8;

    //模板检测参数
    //double min_score_ = 0.5;
    //int contrast_ = 30;
    double max_overlap_ = 0.5;
    double strength_ = 0.5;
   
    bool sort_by_score_ = false;

    //训练区域
    int left_top_x_ = 0;
    int left_top_y_ = 0;
    int box_width_ = 0;
    int box_height_ = 0;


    //检测区域
    int detect_left_x_ = 0;
    int detect_left_y_ = 0;
    int detect_width_ = 9344;
    int detect_height_ = 7000;
    /*
    训练种类
    1，表示训练
    2，表示训练完成，之后进行检测
    3，训练完成之后，保存
    */
    int train_flag_ = 0;


    Tival::ShapeBasedMatching sbm_;
    cv::Mat roi_img_;
};