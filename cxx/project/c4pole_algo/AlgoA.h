#pragma once
#include "../framework/BaseAlgo.h"

class AlgoA : public BaseAlgo {
public:
    AlgoA();
    ~AlgoA();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    DCLEAR_ALGO_GROUP_REGISTER(AlgoA)

    int inside_dia_min_ = 110;
    int inside_dia_max_ = 130;
    int outside_dis_min_ =190;
    int outside_dis_max_ = 210;
    int circle_cnt_ = 1;
    int distance_max_ = 100;
    int diatance_min_ =20;
    float judge_dis_ = 6.5;



    bool get_circle_all_pix(cv::Mat img, cv::Point center, int radius);
    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map);
    bool img_process(cv::Mat src,cv::Rect& circle_rect);
};