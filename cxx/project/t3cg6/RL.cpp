/**
 * @FilePath     : /t3cg6/src/project/LL.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-12-03 15:51:03
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-03 19:13:03
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#include <windows.h>

#include "../../base/src/framework/InferenceEngine.h"
#include "../../base/src/utils/logger.h"
#include "RL.h"
#include "details.h"

// #define DRAW 0
REGISTER_ALGO(RL)

AlgoResultPtr RL::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("RL start run! exec update 2024/12/25");
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    cv::Mat task_img = task->image.clone();
    
    if (task_img.channels() > 1)
        cv::cvtColor(task_img, task_img, cv::COLOR_BGR2GRAY);

    getParam(task);
    img_process(task_img, task, algo_result);
    LOGI("RL run finished!");
    return algo_result;
}

void RL::getParam(InferTaskPtr task)
{
    json param_json = GetTaskParams(task);
    start_pix_      = xx::GetProperty<int>(param_json["param"], "start_pix", 100);
    end_pix_        = xx::GetProperty<int>(param_json["param"], "end_pix", 100);

    blur_w_      = xx::GetProperty<int>(param_json["param"], "blur_w", 5);
    blur_h_      = xx::GetProperty<int>(param_json["param"], "blur_h", 5);
    th_low_      = xx::GetProperty<int>(param_json["param"], "th_low", 0);
    th_high_     = xx::GetProperty<int>(param_json["param"], "th_high", 255);
    blur_length_ = xx::GetProperty<int>(param_json["param"], "blur_length", 200);
    continuous_  = xx::GetProperty<int>(param_json["param"], "continuous", 120);
    th_value_    = xx::GetProperty<int>(param_json["param"], "th_value", 100);
    diff_th_value_ = xx::GetProperty<int>(param_json["param"], "diff_th_value", 100);
    // 判断规则
    json judgements = xx::GetProperty<json>(param_json, "judgement", json());
    for (const auto& item : judgements) {
        STRU_DEFECT_ITEM singal_defect_param;
        singal_defect_param.strItemName = xx::GetProperty<std::string>(item, "name", "");
        // singal_defect_param.bDefectItemUse = xx::GetProperty<bool>(item, "use", true);
        for (const auto& param : item["judgeparams"]) {
            STRU_JUDGEMENT singal_param;
            singal_param.bUse          = xx::GetProperty<bool>(param, "enable", true);
            singal_param.dValue        = xx::GetProperty<double>(param, "value", 0.1);
            singal_param.feature_index = blobexec_.getIndex(xx::GetProperty<std::string>(param, "name", ""));
            singal_param.nSign         = blobexec_.getSignFromSymbol(xx::GetProperty<std::string>(param, "symbol", ""));
            singal_defect_param.Judgment.emplace_back(singal_param);
        }
        judgement_vec.emplace_back(singal_defect_param);
    }
}

 std::vector<cv::Rect> RL::img_process(cv::Mat src, InferTaskPtr task, AlgoResultPtr algo_result)
{

    int row_beg = 0;
    int row_end = src.rows;
    int col_beg = start_pix_;
    int col_end = end_pix_;
    // 查找边界
    xx::get_edge(src, th_low_, th_high_, blur_w_, blur_h_, row_beg, row_end, col_beg, col_end, continuous_, th_value_, 1);

    std::vector<cv::Rect> ret_vec;
    if (col_beg == 0 || col_end == 0 || std::abs(row_beg - row_end) < 20)
        return ret_vec;


    int     temp_r = (col_beg + 600) < src.cols ? 600 : src.cols - col_beg;
    cv::Mat ROI    = src(cv::Rect(col_beg, row_beg, temp_r, row_end - row_beg));

    cv::Mat dst_blur;
    cv::Mat diff;
    cv::Mat sobel_img;
    cv::Mat binary;
    cv::Mat diff_th;

    int blur_p_h = 50;
    if (row_beg >src.rows/3) {
        blur_p_h = 50;
    }
    if (row_end< src.rows-200 ){
        blur_p_h = 150;
        blur_length_ = 250;
    }

    if (row_end < src.rows / 2 + 400) {
        blur_p_h     = 100;
        blur_length_ = 150;
    }
    cv::blur(ROI, dst_blur, cv::Size(5,blur_length_), cv::Point(1, blur_p_h));
    //cv::blur(ROI, dst_blur, cv::Size(30, 30), cv::Point(-1,-1));
    cv::absdiff(ROI, dst_blur, diff);
    cv::threshold(diff, diff_th, diff_th_value_, 255, cv::THRESH_TOZERO);
    cv::Sobel(diff_th, sobel_img, CV_16S, 0, 1, -1);
    cv::convertScaleAbs(sobel_img, sobel_img);
    cv::threshold(sobel_img, sobel_img, 25, 255, cv::THRESH_TOZERO);



    blobexec_.DoBlobCalculate(sobel_img);
    std::vector<tBLOB_FEATURE> feature = blobexec_.DoDefectBlobSingleJudgment(judgement_vec);

    cv::Mat dis = src.clone();
    cv::cvtColor(dis, dis, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < feature.size(); i++) {
        cv::Rect tmp_rect = feature[i].rectBox;
        tmp_rect.x += col_beg;
        tmp_rect.y += row_beg;

        bool a = std::abs(tmp_rect.x - col_beg) <= 18;
        bool b = std::abs(tmp_rect.y - row_beg) < 20;
        bool c = (std::abs(tmp_rect.y + tmp_rect.height / 2 - row_end) < 50) || (std::abs(tmp_rect.y + tmp_rect.height - row_end) < 20);
        if (a && (b || c)) {
            continue;
        }

        cv::rectangle(dis, tmp_rect, cv::Scalar(0, 0, 255), 1);
        algo_result->result_info.push_back({
            {"label", "RL"},
            {"shapeType", "rectangle"},
            {"points", {{tmp_rect.tl().x, tmp_rect.tl().y}, {tmp_rect.br().x, tmp_rect.br().y}}},
            {"result", {{"confidence", 0}, {"angle", "ng"}}},
        });
    }
    return ret_vec;
}