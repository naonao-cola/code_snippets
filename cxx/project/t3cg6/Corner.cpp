/**
 * @FilePath     : /t3cg6/src/project/Corner.cpp
/**
 * @FilePath     : /code_snippets/cxx/project/t3cg6/Corner.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-01-06 14:59:16
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-01-06 14:59:17
 * @Copyright (c) 2025 by G, All Rights Reserved.
**/
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-12-03 16:35:42
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-03 16:44:50
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#include <windows.h>

#include "../../base/src/framework/InferenceEngine.h"
#include "../../base/src/utils/logger.h"
#include "Corner.h"
#include "details.h"

REGISTER_ALGO(Corner)

AlgoResultPtr Corner::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("Corner start run! exec update 2024/11/25");
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();

    cv::Mat task_img = task->image.clone();
    getParam(task);

    if (task_img.channels() > 1)
        cv::cvtColor(task_img, task_img, cv::COLOR_BGR2GRAY);
    bool angle = img_process(task_img, algo_result, "");
    LOGI("Corner process end !");
    if (!angle) {
        algo_result->result_info.push_back({
            {"label", "Angle_defect"},
            {"shapeType", "rectangle"},
            {"points", {{0, 0}, {0, 0}}},
            {"result", {{"confidence", 0}, {"angle", "ng"}}},
        });
    }
    LOGI("Corner run finished!");
    return algo_result;
}

void Corner::getParam(InferTaskPtr task)
{
    json param_json = GetTaskParams(task);
    low_th_         = xx::GetProperty<int>(param_json["param"], "low_th", 50);
    high_th_        = xx::GetProperty<int>(param_json["param"], "high_th", 50);
    low_angle_      = xx::GetProperty<int>(param_json["param"], "low_angle", 50);
    high_angle_     = xx::GetProperty<int>(param_json["param"], "high_angle", 50);
    open_width_     = xx::GetProperty<int>(param_json["param"], "open_width", 50);
    open_hight_     = xx::GetProperty<int>(param_json["param"], "open_hight", 50);
    blur_width_     = xx::GetProperty<int>(param_json["param"], "blur_width", 50);
    blur_hight_     = xx::GetProperty<int>(param_json["param"], "blur_hight", 50);
    grad_width_     = xx::GetProperty<int>(param_json["param"], "grad_width", 50);
    grad_heigh_     = xx::GetProperty<int>(param_json["param"], "grad_heigh", 50);
    h_width_        = xx::GetProperty<int>(param_json["param"], "h_width", 50);
    v_heigh_        = xx::GetProperty<int>(param_json["param"], "v_heigh", 50);
}

bool Corner::img_process(cv::Mat src, AlgoResultPtr task, std::string tmp_path)
{
    cv::Mat dst;
    cv::Mat h_dst;
    cv::Mat v_dst;
    cv::Mat h_diff;
    cv::Mat v_diff;
    double  h_angle = 0;
    int     h_count = 0;
    double  v_angle = 0;
    double  v_count = 0;

    if (src.channels() > 1) {
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    }

#ifdef DEBUG_ON
    cv::Mat dis_1 = src.clone();
    cv::cvtColor(dis_1, dis_1, cv::COLOR_GRAY2BGR);
#endif

    // 预处理
    cv::threshold(src, dst, low_th_, high_th_, cv::THRESH_BINARY);
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(open_width_, open_hight_)));
    cv::blur(dst, dst, cv::Size(blur_width_, blur_hight_));
    cv::morphologyEx(dst, dst, cv::MORPH_GRADIENT, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(grad_width_, grad_heigh_)));
    cv::threshold(dst, dst, low_th_, high_th_, cv::THRESH_BINARY);

    // 水平操作
    cv::Mat h = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(h_width_, 1), cv::Point(-1, -1));
    cv::morphologyEx(dst, h_dst, cv::MORPH_OPEN, h);
    std::vector<std::vector<cv::Point>> h_contours;
    std::vector<cv::Vec4i>              h_hierarchy;
    cv::findContours(h_dst, h_contours, h_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (size_t t = 0; t < h_contours.size(); t++) {
        cv::Rect rect = boundingRect(h_contours[t]);
        int      m    = (std::max)(rect.width, rect.height);
        if (rect.width < 50)
            continue;
        // 直线拟合
        cv::Vec4f oneline;
        cv::fitLine(h_contours[t], oneline, cv::DIST_L1, 0, 0.01, 0.01);
        float cosθ = oneline[0];
        float sinθ = oneline[1];
        float x0   = oneline[2];
        float y0   = oneline[3];
        float k    = sinθ / cosθ;
        float b    = y0 - k * x0;
        h_angle    = h_angle + k;
        h_count++;
        float x = 0;
        float y = k * x + b;
#ifdef DEBUG_ON
        cv::line(dis_1, cv::Point(x0, y0), cv::Point(x, y), cv::Scalar(0, 0, 255), 2, 8, 0);
#endif
    }
    if (std::abs(h_angle) > 0) {
        h_angle = h_angle / h_count * 1.0;
    }

    // 垂直操作
    cv::absdiff(dst, h_dst, h_diff);
    cv::Mat v = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(1, v_heigh_), cv::Point(-1, -1));
    cv::morphologyEx(dst, v_dst, cv::MORPH_OPEN, v);
    cv::absdiff(h_diff, v_dst, v_diff);
    cv::Mat diff_dst;
    cv::morphologyEx(v_diff, diff_dst, cv::MORPH_OPEN, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1)));

    std::vector<std::vector<cv::Point>> v_contours;
    std::vector<cv::Vec4i>              v_hierarchy;
    cv::findContours(diff_dst, v_contours, v_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    for (size_t t = 0; t < v_contours.size(); t++) {
        cv::Rect rect = boundingRect(v_contours[t]);
        int      m    = (std::max)(rect.width, rect.height);
        if (rect.width < 60)
            continue;
        cv::Vec4f oneline;
        cv::fitLine(v_contours[t], oneline, cv::DIST_L1, 0, 0.01, 0.01);
        float cosθ = oneline[0];
        float sinθ = oneline[1];
        float x0   = oneline[2];
        float y0   = oneline[3];

        float k = sinθ / cosθ;
        float b = y0 - k * x0;
        v_angle = v_angle + k;
        v_count++;
        float x = 0;
        float y = k * x + b;
#ifdef DEBUG_ON
        cv::line(dis_1, cv::Point(x0, y0), cv::Point(x, y), cv::Scalar(0, 0, 255), 2, 8, 0);
#endif
    }

    if (std::abs(v_angle) > 0) {
        v_angle = v_angle / v_count * 1.0;
    }

    double ret_angle = 0;
    if (std::abs(v_angle) > 0 && std::abs(h_angle) > 0) {
        ret_angle = std::atan(std::abs((v_angle - h_angle) / (1 + (v_angle * h_angle)))) / CV_PI * 180;
    }
    if (ret_angle > -370 || ret_angle < 370) {
        ret_angle = int(ret_angle);
    }

    if (ret_angle == 0) {
        return false;
    }
    if (ret_angle >= low_angle_ && ret_angle <= high_angle_) {
        return true;
    }
    return false;
}