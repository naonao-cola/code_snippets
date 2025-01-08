#pragma once
#include "../../modules/tv_algo_base/src/framework/BaseAlgo.h"
#include "ShapeBasedMatching.h"
#include <opencv2/opencv.hpp>
#include <string.h>



class Train : public BaseAlgo
{
public:
    Train(){};
    ~Train(){};
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    DCLEAR_ALGO_GROUP_REGISTER(Train)

    cv::Mat template_img_;   // 模板图片
    cv::Mat template_mask_img_;
    // 模板匹配参数
    // 训练模板参数
    int    num_levels_ = 3;
    int    angle_min_  = -10;
    int    angle_max_  = 10;
    double min_score_  = 0.5;
    int    contrast_   = 30;
    // 保存路径
    std::string path_        = "";
    std::string yml_         = "";
    double      max_overlap_ = 0.5;
    double      strength_    = 0.5;

    // 追加参数
    int    num_           = 100;
    double similarity_    = 0.55;
    double scale_min_     = 0.8;
    double scale_max_     = 1.2;
    double greediness_    = 0.8;
    bool   sort_by_score_ = false;


    // 训练区域
    int left_top_x_ = 0;
    int left_top_y_ = 0;
    int box_width_  = 0;
    int box_height_ = 0;
    //检测区域
    int detect_left_x_ = 0;
    int detect_left_y_ = 0;
    int detect_width_ = 9344;
    int detect_height_ = 7000;



    Tival::ShapeBasedMatching sbm_;

    /**
     * @brief: 获取参数
     * @return
     * @note :
     **/
    void getParam(InferTaskPtr task);

    /**
     * @brief: 训练模型
     * @return
     * @note :
     **/
    void train(const cv::Mat& src, AlgoResultPtr algo_result);
    bool test(const cv::Mat& src);
};