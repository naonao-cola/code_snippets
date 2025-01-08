#include "train.h"
#include "../../modules/tv_algo_base/src/framework/InferenceEngine.h"
/**
 * @FilePath     : /connector_ai/src/project/train.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-10-24 19:18:34
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-10-30 18:51:06
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
/**
 * @FilePath     : /connector_ai/src/project/train.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-10-24 19:18:34
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-10-29 17:23:01
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#include "../../modules/tv_algo_base/src/utils/logger.h"
#include "details.h"

#include <windows.h>

#if USE_AI_DETECT
#    include <AIRuntimeDataStruct.h>
#    include <AIRuntimeInterface.h>
#    include <AIRuntimeUtils.h>
#endif   // USE_AI_DETECT

REGISTER_ALGO(Train)

AlgoResultPtr Train::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("Train start run!");
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    getParam(task);

    train(task->image, algo_result);
    test(task->image);
    LOGI("Train run finished!");
    return algo_result;
}

void Train::getParam(InferTaskPtr task)
{
    json param_json = GetTaskParams(task);
    if (template_img_.empty()) {
        std::string p1     = xx::GetProperty<std::string>(param_json["param"], "template_path", "");
        std::string p2     = xx::GetProperty<std::string>(param_json["param"], "tempmask_path", "");
        template_img_      = cv::imread(p1, 0);
        template_mask_img_ = cv::imread(p2, 0);
    }
    num_    = xx::GetProperty<int>(param_json["param"], "num", 24);
    num_levels_    = xx::GetProperty<int>(param_json["param"], "num_levels", 4);
    angle_min_     = xx::GetProperty<int>(param_json["param"], "angle_min", -10);
    angle_max_     = xx::GetProperty<int>(param_json["param"], "angle_max", 10);
    min_score_     = xx::GetProperty<double>(param_json["param"], "min_score", 0.4);
    contrast_      = xx::GetProperty<int>(param_json["param"], "contrast", 35);
    path_          = xx::GetProperty<std::string>(param_json["param"], "path", "");
    yml_           = xx::GetProperty<std::string>(param_json["param"], "yml", "");
    scale_min_     = xx::GetProperty<double>(param_json["param"], "scale_min", 0.9);
    scale_max_     = xx::GetProperty<double>(param_json["param"], "scale_max", 1.1);
    max_overlap_   = xx::GetProperty<double>(param_json["param"], "max_overlap", 0.2);
    strength_      = xx::GetProperty<double>(param_json["param"], "strength", 0.8);
    sort_by_score_ = xx::GetProperty<bool>(param_json["param"], "sort_by_score", false);
    detect_left_x_ = xx::GetProperty<int>(param_json["param"], "detect_left_x", 23);
    detect_left_y_ = xx::GetProperty<int>(param_json["param"], "detect_left_y", 119);
    detect_width_  = xx::GetProperty<int>(param_json["param"], "detect_width", 9132);
    detect_height_ = xx::GetProperty<int>(param_json["param"], "detect_height", 6679);
}

void Train::train(const cv::Mat& src, AlgoResultPtr algo_result)
{
    cv::Rect          roi_rect;
    cv::Mat           roi_img;
    cv::Mat           mask;
    cv::Mat           dst1;
    cv::Mat           dst2;
    Tival::SbmResults sbm_ret;

    roi_img = template_img_;
    mask    = template_mask_img_;
    cv::resize(roi_img, dst1, cv::Size(roi_img.cols / 2, roi_img.rows / 2));
    cv::resize(mask, dst2, cv::Size(mask.cols / 2, mask.rows / 2));
    if (dst1.channels() > 1)
        cv::cvtColor(dst1, dst1, cv::COLOR_BGR2GRAY);
    cv::Mat ret = dst2 & dst1;
    // cv::bitwise_not(dst2,dst2);
    cv::threshold(dst2, dst2, 200, 255, cv::THRESH_BINARY);

    // 训练参数
    json params = {
        {"NumLevels", num_levels_},
        {"AngleMin", angle_min_},
        {"AngleMax", angle_max_},
        {"ScaleMin", scale_min_},
        {"ScaleMax", scale_max_},
        {"MinScore", min_score_},
        {"Contrast", contrast_},
    };

    sbm_ret                                        = sbm_.CreateByImageWithMask(dst1, dst2, params);
    cv::Mat                             dstlDraw   = dst1.clone();
    std::vector<std::vector<cv::Point>> modelShape = sbm_.GetModelFeaturePoints(1).ToIntContours();
    cv::drawContours(dstlDraw, modelShape, -1, cv::Scalar(0, 255, 0), 1, 8);

    if (sbm_ret.center.size() != 1) {
        return;
    }
    if (sbm_ret.center.size() == 1) {
        sbm_.Save(path_);
        cv::FileStorage fs(yml_, cv::FileStorage::WRITE);
        fs << "template_width" << template_img_.cols;
        fs << "template_height" << template_img_.rows;
    }
}

bool Train::test(const cv::Mat& src)
{
    cv::Mat gray_img;
    cv::Mat dst;
    cv::Mat dis;
    int     bw;
    int     bh;

    dis = src.clone();
    cv::FileStorage fs(yml_, cv::FileStorage::READ);
    fs["template_width"] >> bw;
    fs["template_height"] >> bh;

    cv::Rect detect_rect(detect_left_x_, detect_left_y_, detect_width_, detect_height_);
    cv::Mat  detect_img = src(detect_rect);
    if (detect_img.channels() > 1)
        cv::cvtColor(detect_img, gray_img, cv::COLOR_BGR2GRAY);
    else
        gray_img = detect_img;
    cv::resize(detect_img, dst, cv::Size(detect_img.cols / 2, detect_img.rows / 2));
    nlohmann::json            match_params = {{"AngleMin", angle_min_},
                                              {"AngleMax", angle_max_},
                                              {"MinScore", min_score_},
                                              {"ScaleMin", scale_min_},
                                              {"ScaleMax", scale_max_},
                                              {"Contrast", contrast_},
                                              {"SortByScore", sort_by_score_},
                                              {"MaxOverlap", max_overlap_},
                                              {"Strength", strength_},
                                              {"Num", num_}};
    Tival::ShapeBasedMatching sbm;
    sbm.Load(path_);
    Tival::SbmResults ret = sbm.Find(dst, match_params);

    //! 坐标转换
    int count = (std::min)(num_, (int)ret.center.size());
    for (int m = 0; m < count; m++) {
        int             sx     = ret.center[m].x * 2 + bw / 2.0;
        int             sy     = ret.center[m].y * 2 + bh / 2.0;
        double          sangle = ret.angle[m];
        double          sscore = ret.score[m];
        double          sscale = ret.scale[m];
        cv::RotatedRect rotate_rect(cv::Point2d(sx + detect_left_x_, sy + detect_left_y_), cv::Size(bw, bh), -sangle * 180 / CV_PI);
        cv::Point2f     vertex[4];
        rotate_rect.points(vertex);
        std::vector<cv::Point2d> pt_vec;
        pt_vec.push_back(vertex[0]);
        pt_vec.push_back(vertex[1]);
        pt_vec.push_back(vertex[2]);
        pt_vec.push_back(vertex[3]);
        pt_vec = xx::order_pts(pt_vec);

        for (size_t j = 0; j < 4; j++)
            cv::line(dis, pt_vec[j], pt_vec[(j + 1) % 4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    return true;
}
