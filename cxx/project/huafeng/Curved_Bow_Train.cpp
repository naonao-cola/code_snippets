/**
 * @FilePath     : /connector/src/custom/Curved_Bow_Train.cpp
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-01-08 11:54:34
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "Curved_Bow_Train.h"
#include <windows.h>
#include "algo_tool.h"
#include "xml_wr.h"
#include "./sub_3rdparty/tival/include/JsonHelper.h"
#include "param_check.h"
#include "../utils/Utils.h"

REGISTER_ALGO(Curved_Bow_Train)

Curved_Bow_Train::Curved_Bow_Train(){}

Curved_Bow_Train::~Curved_Bow_Train()
{
}

std::tuple<std::string, json> Curved_Bow_Train::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map) {

    std::string task_type_id = task->image_info["type_id"];
    json task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}


AlgoResultPtr Curved_Bow_Train::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    TVALGO_FUNCTION_BEGIN
    algo_result->result_info.push_back(
        {
            {"label","Curved_Bow_Train"},
            {"shapeType","default"},
            {"points",{{0,0},{0,0}}},
            {"result",{{"confidence",0},{"area",0}}},
        }
    );
    //获取参数
    std::tuple<std::string, json> details_info = get_task_info(task, pre_results, m_param_map);
    json task_param_json = std::get<1>(details_info);

    try
    {
        num_levels_    = Tival::JsonHelper::GetParam(task_param_json["param"], "num_levels", 4);
        angle_min_     = Tival::JsonHelper::GetParam(task_param_json["param"], "angle_min", -10);
        angle_max_     = Tival::JsonHelper::GetParam(task_param_json["param"], "angle_max", 10);
        min_score_     = Tival::JsonHelper::GetParam(task_param_json["param"], "min_score", 0.4);
        contrast_      = Tival::JsonHelper::GetParam(task_param_json["param"], "contrast", 35);
        path_          = Tival::JsonHelper::GetParam(task_param_json["param"], "path", std::string(".\\AlgoModels\\TPL-0c281ba6b3b441f2bcc0f005dee51693_fb47d9a7-72dc-4eed-be23-f4b9349b531f"));
        scale_min_     = Tival::JsonHelper::GetParam(task_param_json["param"], "scale_min", 0.9);
        scale_max_     = Tival::JsonHelper::GetParam(task_param_json["param"], "scale_max", 1.1);
        max_overlap_   = Tival::JsonHelper::GetParam(task_param_json["param"], "max_overlap", 0.2);
        strength_      = Tival::JsonHelper::GetParam(task_param_json["param"], "strength", 0.8);
        left_top_x_    = Tival::JsonHelper::GetParam(task_param_json["param"], "left_top_x", 4468);
        left_top_y_    = Tival::JsonHelper::GetParam(task_param_json["param"], "left_top_y", 2862);
        sort_by_score_ = Tival::JsonHelper::GetParam(task_param_json["param"], "sort_by_score", false);
        box_width_     = Tival::JsonHelper::GetParam(task_param_json["param"], "box_width", 603);
        box_height_    = Tival::JsonHelper::GetParam(task_param_json["param"], "box_height", 425);
        detect_left_x_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_x", 23);
        detect_left_y_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_y", 119);
        detect_width_  = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_width", 9132);
        detect_height_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_height", 6679);
        train_flag_    = Tival::JsonHelper::GetParam(task_param_json["param"], "train_flag", 1);
        bool status = true;

        status &= InIntRange("left_top_x", left_top_x_, 0, 9344, false);
        status &= InIntRange("left_top_y", left_top_y_, 0, 7000, false);
        status &= InIntRange("box_width",  box_width_, 0, 9344, false);
        status &= InIntRange("box_height", box_height_, 0, 9344, false);

        status &= InIntRange("detect_left_x", detect_left_x_, 0, 9344, false);
        status &= InIntRange("detect_left_y", detect_left_y_, 0, 7000, false);
        status &= InIntRange("detect_width", detect_width_, 0, 9344, false);
        status &= InIntRange("detect_height", detect_height_, 0, 9344, false);

        if (!status) {
            TVALGO_FUNCTION_RETURN_ERROR_PARAM("param error")
        }
    }
    catch (const std::exception& e )
    {
        TVALGO_FUNCTION_RETURN_ERROR_PARAM(e.what())
    }
    if (train_flag_ != 1) {
        TVALGO_FUNCTION_END
    }

    //获取原图并转为单通道
    cv::Mat task_img = task->image.clone();
    cv::Mat dst1,dst2;
    if (task_img.channels() > 1)
        cv::cvtColor(task_img, dst1, cv::COLOR_BGR2GRAY);
    
    //判断非空
    if (!task->image2.empty()) {
        dst2 = task->image2.clone();
        if (dst2.channels() == 3) {
            cv::cvtColor(dst2, dst2, cv::COLOR_BGR2GRAY);
        }
        else if(dst2.channels() == 4) {
            cv::cvtColor(dst2, dst2, cv::COLOR_BGRA2GRAY);
        }
    }
    else {
        dst2 = cv::Mat(cv::Size(dst1.cols,dst2.rows),CV_8UC1,cv::Scalar(255));
    }
    if (dst2.cols != dst1.cols || dst2.rows != dst1.rows) {
        algo_result->judge_result = 0;
        TVALGO_FUNCTION_LOG("Curved_Bow_Train  train_process flase!")
        TVALGO_FUNCTION_END
    }
    if (train_flag_ == 1) {
        train_process(dst1, dst2,algo_result);
    }
    TVALGO_FUNCTION_END
}

void Curved_Bow_Train::train_process(const cv::Mat& src, const cv::Mat& mask, AlgoResultPtr algo_result) {
    LOGI("Curved_Bow_Train run  train_process start!");
    //获取roi区域图像
    cv::Rect roi_rect(left_top_x_, left_top_y_, box_width_, box_height_);
    roi_img_ = src;
    //训练参数
    nlohmann::json params = {
        {"NumLevels",num_levels_},
        {"AngleMin",angle_min_},
        {"AngleMax",angle_max_},
        {"ScaleMin",scale_min_},
        {"ScaleMax",scale_max_},
        {"MinScore",min_score_},
        {"Contrast",contrast_},
    };
    //原图与mask缩放,
    cv::Mat dst1,dst2;
    cv::resize(roi_img_,dst1,cv::Size(roi_img_.cols/2, roi_img_.rows/2));
    cv::resize(mask, dst2, cv::Size(mask.cols / 2, mask.rows / 2));

    cv::bitwise_not(dst2,dst2);
    TVALGO_FUNCTION_LOG("train  start")
    cv::threshold(dst2,dst2,200,255,cv::THRESH_BINARY);

    Tival::SbmResults sbm_ret;
    try
    {
        sbm_ret = sbm_.CreateByImageWithMask(dst1, dst2, params);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    
    if (sbm_ret.center.size() != 1) {
        algo_result->judge_result = 0;
        algo_result->result_info.emplace_back();
        TVALGO_FUNCTION_LOG("Curved_Bow_Train  train_process flase!")
        return;
    }

    TVALGO_FUNCTION_LOG("train end")
    if (sbm_ret.center.size() == 1) {
        sbm_.Save(path_ + "//Curved_Bow_Train.mdl");
        //写入模板信息
        nao::xml::Xmlw xmlw(1, path_ + "//Curved_Bow_Train.xml");
        xmlw.writeValue("template_x", left_top_x_);
        xmlw.writeValue("template_y", left_top_y_);
        xmlw.writeValue("template_width", box_width_);
        xmlw.writeValue("template_height", box_height_);
        xmlw.writeValue("template_cx", sbm_ret.center[0].x * 2 + left_top_x_ );
        xmlw.writeValue("template_cy", sbm_ret.center[0].y * 2 + left_top_y_ );
        algo_result->judge_result = 1;
    }
   
    TVALGO_FUNCTION_LOG("Curved_Bow_Train run  train_process end!")
}
