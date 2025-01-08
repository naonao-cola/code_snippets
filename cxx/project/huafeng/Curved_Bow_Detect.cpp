/**
 * @FilePath     : /connector/src/custom/Curved_Bow_Detect.cpp
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-01-27 16:53:14
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#include <future>
#include <windows.h>

#include "../framework/InferenceEngine.h"
#include "../utils/Utils.h"
#include "../utils/logger.h"
#include "./sub_3rdparty/tival/include/JsonHelper.h"
#include "Curved_Bow_Detect.h"
#include "algo_tool.h"
#include "param_check.h"
#include "spinlock.h"
#include "xml_wr.h"
#include  <filesystem>
REGISTER_ALGO(Curved_Bow_Detect)
USpinLock g_pin_lock;
Curved_Bow_Detect::Curved_Bow_Detect() { }
Curved_Bow_Detect::~Curved_Bow_Detect() { }

AlgoResultPtr Curved_Bow_Detect::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    TVALGO_FUNCTION_BEGIN
    algo_result->result_info.push_back(
        {
            {"label","Curved_Bow_Detect"},
            {"shapeType","default"},
            {"points",{{0,0},{0,0}}},
            {"result",{{"confidence",0},{"area",0}}},
        }
    );
    try {
        bool status = get_param(task, pre_results);
        if (!status) {
            TVALGO_FUNCTION_RETURN_ERROR_PARAM("error param")
        }
    } catch (const std::exception& e) {
        TVALGO_FUNCTION_RETURN_ERROR_PARAM(e.what())
    }
    
    // 算法部分
    cv::Mat task_img = task->image.clone();
    cv::Mat dst, roi_img;
    if (task_img.channels() > 1)
        cv::cvtColor(task_img, dst, cv::COLOR_BGR2GRAY);
    else {
        dst = task_img.clone();
    }
    if (detect_flag_ == 3) {
        img_process(dst, algo_result);
    }
    TVALGO_FUNCTION_END
}

std::tuple<std::string, json> Curved_Bow_Detect::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map)
{
    std::string task_type_id = task->image_info["type_id"];
    json task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

std::vector<cv::Point2f> get_rotrect_coords(double x, double y, double w, double h, double r, bool lt_first)
{
    double centX = x;
    double centY = y;
    double hw = w / 2;
    double hh = h / 2;
    double phi = r / 180 * CV_PI; // 弧度
    double ox[] = { centX - hw, centX + hw, centX + hw, centX - hw };
    double oy[] = { centY - hh, centY - hh, centY + hh, centY + hh };

    std::vector<cv::Point2f> tmp_pts;
    for (int i = 0; i < 4; i++) {
        double x = (ox[i] - centX) * cos(-phi) - (oy[i] - centY) * sin(-phi) + centX;
        double y = (ox[i] - centX) * sin(-phi) + (oy[i] - centY) * cos(-phi) + centY;
        tmp_pts.push_back(cv::Point2f(x, y));
    }
    return tmp_pts;
}

bool Curved_Bow_Detect::get_param(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    // 获取参数
    std::tuple<std::string, json> details_info = get_task_info(task, pre_results, m_param_map);
    json task_param_json = std::get<1>(details_info);
    bool status = true;

    num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "num", 4);
    angle_min_ = Tival::JsonHelper::GetParam(task_param_json["param"], "angle_min", -10);
    angle_max_ = Tival::JsonHelper::GetParam(task_param_json["param"], "angle_max", 10);
    scale_min_ = Tival::JsonHelper::GetParam(task_param_json["param"], "scale_min", 0.9);
    scale_max_ = Tival::JsonHelper::GetParam(task_param_json["param"], "scale_max", 1.1);
    contrast_ = Tival::JsonHelper::GetParam(task_param_json["param"], "contrast", 35);
    min_score_ = Tival::JsonHelper::GetParam(task_param_json["param"], "min_score", 0.4);
    strength_ = Tival::JsonHelper::GetParam(task_param_json["param"], "strength", 0.8);
    max_overlap_ = Tival::JsonHelper::GetParam(task_param_json["param"], "max_overlap", 0.2);
    sort_by_score_ = Tival::JsonHelper::GetParam(task_param_json["param"], "sort_by_score", false);
    path_ = Tival::JsonHelper::GetParam(task_param_json["param"], "path", std::string(""));
    detect_flag_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_flag", 3);
    basis_x_num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "basis_x_num", 12);
    basis_y_num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "basis_y_num", 8);
    detect_left_x_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_x", 0);
    detect_left_y_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_y", 0);
    detect_width_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_width", 9344);
    detect_height_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_height", 7000);
    gamma_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gamma_value", 0.8);
    small_film_t_brightness_ = Tival::JsonHelper::GetParam(task_param_json["param"], "small_film_t_brightness", 255);
    small_film_b_brightness_ = Tival::JsonHelper::GetParam(task_param_json["param"], "small_film_b_brightness", 220);
    small_film_t_area_ = Tival::JsonHelper::GetParam(task_param_json["param"], "small_film_t_area", 3300);
    small_film_b_area_ = Tival::JsonHelper::GetParam(task_param_json["param"], "small_film_b_area", 1200);

    // 读xml
    std::filesystem::path xml_path(path_ + "//Curved_Bow_Train.xml");
    if (!std::filesystem::exists(xml_path)) {
        TVALGO_FUNCTION_LOG("xml file not exist")
        return false;
    }
    nao::xml::Xmlr xmlr(path_ + "//Curved_Bow_Train.xml");
    xmlr.readValue("template_x", template_x_);
    xmlr.readValue("template_y", template_y_);
    xmlr.readValue("template_width", template_width_);
    xmlr.readValue("template_height", template_height_);
    xmlr.readValue("template_cx", template_cx_);
    xmlr.readValue("template_cy", template_cy_);

    // 读取卡尺参数
    caliper_param_vec_.clear();
    json caliper_vec = task_param_json["param"]["calipers"];
    if (caliper_vec.is_array() && !caliper_vec.empty()) {
        // 卡尺数量
        auto size = caliper_vec.size();
        for (int i = 0; i < size; i++) {
            auto item = caliper_vec.at(i);
            std::shared_ptr<caliper_param> sub_caliper_param = std::make_shared<caliper_param>();
            // 卡尺参数
            sub_caliper_param->caliper_num = Tival::JsonHelper::GetParam(item, "caliper_num", 25);
            sub_caliper_param->caliper_length = Tival::JsonHelper::GetParam(item, "caliper_length", 30);
            sub_caliper_param->caliper_width = Tival::JsonHelper::GetParam(item, "caliper_width", 10);
            sub_caliper_param->sigma = Tival::JsonHelper::GetParam(item, "sigma", 1);
            sub_caliper_param->transition = Tival::JsonHelper::GetParam(item, "transition", std::string("all"));
            sub_caliper_param->num = Tival::JsonHelper::GetParam(item, "num", 1);
            sub_caliper_param->contrast = Tival::JsonHelper::GetParam(item, "contrast", 45);
            // 卡尺区域
            sub_caliper_param->center_x = Tival::JsonHelper::GetParam(item, "center_x", 4513);
            sub_caliper_param->center_y = Tival::JsonHelper::GetParam(item, "center_y", 3117);
            sub_caliper_param->box_width = Tival::JsonHelper::GetParam(item, "box_width", 49);
            sub_caliper_param->box_height = Tival::JsonHelper::GetParam(item, "box_height", 49);
            sub_caliper_param->angle = Tival::JsonHelper::GetParam(item, "angle", 49);
            sub_caliper_param->sort_by_score = Tival::JsonHelper::GetParam(item, "sort_by_score", true);

            status = InStringSet("transition", sub_caliper_param->transition, { "all", "positive", "negative" }, false);
            caliper_param_vec_.push_back(sub_caliper_param);
        }
    }
    // 读取缺失基座
    miss_.clear();
    json miss_vec = task_param_json["param"]["miss"];
    for (int i = 0; i < miss_vec.size(); i++) {
        auto item = miss_vec[i];
        miss_.push_back(cv::Point(item[0], item[1]));
    }

    // 读取理论值
    d_value_.clear();
    double d_tmp_value, e_tmp_value;

    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d1", 1.6);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d2", 1.6);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d3", 0.162);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d4", 0.162);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d5", 1.475);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d6", 1.475);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d7", 1.0);
    d_value_.push_back(d_tmp_value);
    d_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "d8", 1.0);
    d_value_.push_back(d_tmp_value);

    error_value_.clear();
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_1", 0.11);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_2", 0.11);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_3", 0.13);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_4", 0.13);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_5", 0.13);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_6", 0.13);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_7", 2.0);
    error_value_.push_back(e_tmp_value);
    e_tmp_value = Tival::JsonHelper::GetParam(task_param_json["param"], "error_8", 2.0);
    error_value_.push_back(e_tmp_value);

    ratio_vec_.clear();
    double ratio_value_tmp, constant_value_tmp;
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_1", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_2", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_3", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_4", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_5", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_6", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_7", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);
    ratio_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "ratio_8", 1.0);
    ratio_vec_.push_back(ratio_value_tmp);

    constant_vec_.clear();
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_1", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_2", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_3", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_4", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_5", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_6", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_7", 0.0);
    constant_vec_.push_back(constant_value_tmp);
    constant_value_tmp = Tival::JsonHelper::GetParam(task_param_json["param"], "constant_8", 0.0);
    constant_vec_.push_back(constant_value_tmp);

    pix_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "pix_value", 5.26);

    status &= InIntRange("detect_left_x", detect_left_x_, 0, 9344, false);
    status &= InIntRange("detect_left_y", detect_left_y_, 0, 7000, false);
    status &= InIntRange("detect_width", detect_width_, 0, 9344, false);
    status &= InIntRange("detect_height", detect_height_, 0, 9344, false);
    status &= InIntRange("small_film_t_brightness", small_film_t_brightness_, 200, 255, false);
    status &= InIntRange("small_film_b_brightness", small_film_b_brightness_, 150, 255, false);
    status &= InIntRange("small_film_t_area", small_film_t_area_, 3000, 5000, false);
    status &= InIntRange("small_film_b_area", small_film_b_area_, 800, 2000, false);
    return status;
}


cv::Point2f Curved_Bow_Detect::get_caliper_center(cv::Point2f param[], const cv::Mat& img, std::string direction, cv::Point2f& pt_center,Tival::FindLineResult line_ret, AlgoResultPtr algo_result)
{
    // 左右屏蔽片中心，固定大小
    cv::Point2f ret_pt;
    int x_collect[4] = { param[0].x, param[1].x, param[2].x, param[3].x };
    int y_collect[4] = { param[0].y, param[1].y, param[2].y, param[3].y };

    // 当前外接矩形
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    int a1 = std::min(std::max(left - 50, 0), img.cols);
    int a2 = std::min(std::max(top - 80, 0), img.rows);

    int b1 = std::min(std::max(right + 50, 0), img.cols);
    int b2 = std::min(std::max(bottom + 80, 0), img.rows);
    // 外延
    cv::Point2f p1(a1, a2);
    cv::Point2f p2(b1, b2);

    cv::Rect roi_rect(p1, p2);

    cv::Mat sub_img, gamma_img, blur_img, th_img, bit_not_img;
    sub_img = img(roi_rect);
    // 先gamma增强
    gamma_img = connector::gamma_trans(sub_img, gamma_value_);
    th_img = gamma_img.clone();
    cv::threshold(th_img, th_img, small_film_b_brightness_, small_film_t_brightness_, cv::THRESH_BINARY);

    // y方向膨胀，再腐蚀
    cv::dilate(th_img, th_img, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 25)));
    cv::erode(th_img, th_img, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 25)));

    std::vector<std::vector<cv::Point>> contours = connector::get_contours(th_img);
    cv::Rect draw_rect;
    ;
    for (int i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        double area = cv::contourArea(contours[i]);
        // 去除面积不符合的，去除长宽比不符合的
        if (area <= small_film_b_area_ || area >= small_film_t_area_ || rect.width * 3 >= rect.height)
            continue;
        // 去处中心不在中间的
        if (rect.x + rect.width / 2 >= sub_img.cols - 35 || rect.x + rect.width / 2 <= 35)
            continue;
        if (rect.y + rect.height / 2 >= sub_img.rows - 35 || rect.y + rect.height / 2 <= 35)
            continue;
        cv::Point2f center;
        cv::Mat cn_img = th_img(rect);
        connector::get_center(cn_img, center);
        draw_rect = rect;
        center.y = rect.tl().y + center.y;
        center.x = rect.tl().x + center.x;
        /* draw_rect.x = rect.tl().y + draw_rect.x;
         draw_rect.y = rect.tl().y + draw_rect.y;*/
        ret_pt = center;
    }
    ret_pt.x = ret_pt.x + roi_rect.x;
    ret_pt.y = ret_pt.y + roi_rect.y;

    draw_rect.x = roi_rect.x + draw_rect.x;
    draw_rect.y = roi_rect.y + draw_rect.y;

    pt_center = ret_pt;
    ret_pt = connector::calculate_foot_point(line_ret.start_point[0], line_ret.end_point[0], ret_pt);


    //中心点找错了
    int mid_y = (top + bottom) / 2;
    if (std::abs(ret_pt.y - mid_y)>50) {
        ret_pt.x = -1;
        ret_pt.y = -1;
        pt_center.x = -1;
        pt_center.y = -1;
    }

    g_pin_lock.lock();
    algo_result->result_info.push_back(
        { { "label", "Curved_Bow_Train_defect" },
            { "shapeType", "point" },
            { "points", { { ret_pt.x, ret_pt.y } } },
            { "result", { { "confidence", 1 }, { "area", 0 } } } });

    algo_result->result_info.push_back(
        { { "label", "Curved_Bow_Train_defect" },
            { "shapeType", "rectangle" },
            { "points", { { draw_rect.x, draw_rect.y }, { draw_rect.x + draw_rect.width, draw_rect.y }, { draw_rect.x + draw_rect.width, draw_rect.y + draw_rect.height }, { draw_rect.x, draw_rect.y + draw_rect.height } } },
            { "result", { { "confidence", 1 }, { "area", 0 }, { "index", -2 } } } });

    g_pin_lock.unlock();
    return ret_pt;
}

void Curved_Bow_Detect::img_process(const cv::Mat& src, AlgoResultPtr algo_result)
{
    // 首先进行匹配
    int basis_num = basis_x_num_ * basis_y_num_ - miss_.size();
    nlohmann::json match_params = {
        { "AngleMin", angle_min_ },
        { "AngleMax", angle_max_ },
        { "MinScore", min_score_ },
        { "ScaleMin", scale_min_ },
        { "ScaleMax", scale_max_ },
        { "Contrast", contrast_ },
        { "SortByScore", sort_by_score_ },
        { "MaxOverlap", max_overlap_ },
        { "Strength", strength_ },
        { "Num", basis_num }
    };

    std::filesystem::path xml_path(path_ + "//Curved_Bow_Train.mdl");
    if (!std::filesystem::exists(xml_path)) {
        TVALGO_FUNCTION_LOG("model file not exist")
        return;
    }

    Tival::ShapeBasedMatching sbm;
    // 模板是否加载，未加载进行训练
    if (!sbm.IsLoaded()) {
        sbm.Load(path_ + "//Curved_Bow_Train.mdl");
    }
    status_flag = true;
    // 画出检测区域
    cv::Rect detect_rect(detect_left_x_, detect_left_y_, detect_width_, detect_height_);
    cv::Mat detect_img = src(detect_rect).clone();
    // 缩放到一半
    cv::Mat dst;
    cv::resize(detect_img, dst, cv::Size(detect_img.cols / 2, detect_img.rows / 2));
    Tival::SbmResults ret = sbm.Find(dst, match_params);

    cv::Mat dis = src.clone();
  

    std::vector<match_ret> order_ret;
    std::vector<basis_info> basis_info_vec;
    bool cvt_ret = cvt_match(ret, order_ret, algo_result, src, basis_info_vec);
    //connector::draw_results(dis, algo_result->result_info);
    if (!cvt_ret) {
        LOGI("Curved_Bow_Detect match error");
        algo_result->judge_result = 0;
        algo_result->result_info.push_back(
            {
                { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "image_ret" },
                { "points", {}},
                { "result", { { "confidence", 0 }, { "area", 0 } } },
            }
        );
        return;
    }
    // connector::draw_results(dis, algo_result->result_info);
  
 #pragma omp parallel for 
    for (int i = 0; i < basis_info_vec.size(); i++) {
        // 循环检测每个小基座
        basis_find_line(src, algo_result, basis_info_vec[i], basis_info_vec[i].index);
    }

    // 判断个数
    if (ret.center.size() != basis_num || !status_flag) {
        algo_result->judge_result = 0;
        algo_result->result_info.push_back(
            {
                { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "image_ret" },
                { "points", {}},
                { "result", { { "confidence", 0 }, { "area", 0 } } },
            }
        );

    } else {
        algo_result->judge_result = 1;
        algo_result->result_info.push_back(
            {
                { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "image_ret" },
                { "points", {}},
                { "result", { { "confidence", 1 }, { "area", 0 } } },
            }
        );
    }
   
   /* connector::draw_results(dis, algo_result->result_info);
    cv::imwrite("E:\\demo\\cxx\\connector\\data\\test.jpg", dis);*/

}


int Curved_Bow_Detect::get_nearest_point_idx(cv::Mat points, cv::Point2d refPoint, double& minDist) {
    cv::Mat query = (cv::Mat_<float>(1, 2) << refPoint.x, refPoint.y);
    cv::flann::Index flannIndex(points, cv::flann::KDTreeIndexParams());
    cv::Mat indices, dists;
    flannIndex.knnSearch(query, indices, dists, 2, cv::flann::SearchParams());
    minDist = dists.at<float>(0, 0);
    return  indices.at<int>(0, 0);
}


bool Curved_Bow_Detect::cvt_match(const Tival::SbmResults& ret, std::vector<match_ret>& order_ret, AlgoResultPtr algo_result, const cv::Mat& src, std::vector<basis_info>& basis_info_vec)
{
    cv::Mat dis = src.clone();
    if (dis.channels() < 3) {
        cv::cvtColor(dis,dis,cv::COLOR_GRAY2BGR);
    }

    order_ret.clear();
    //匹配中心与 矩形中心的偏差
    cv::Point2f template_center(template_x_ + template_width_ / 2, template_y_ + template_height_ / 2);
    cv::Point2f template_match_center(template_cx_, template_cy_);
    double x_match_diff = template_match_center.x - template_center.x;
    double y_match_diff = template_match_center.y - template_center.y;

    double angle_avg = 0;
    double angle_count = 0;
    std::vector<double> angle_vec;
    // 还原到原图坐标，在排序
    for (int m = 0; m < ret.center.size(); m++) {
        match_ret tmp_ret;
        tmp_ret.x = ret.center[m].x * 2 + detect_left_x_;
        tmp_ret.y = ret.center[m].y * 2 + detect_left_y_;
        tmp_ret.angle = ret.angle[m];
        tmp_ret.score = ret.score[m];
        tmp_ret.scale = ret.scale[m];
        basis_info signal_basis_info;
        signal_basis_info.match_info = tmp_ret;
        basis_info_vec.push_back(signal_basis_info);

        //添加结果，添加偏移量，画框不会歪
        cv::RotatedRect rotate_rect(cv::Point2d(signal_basis_info.match_info.x- x_match_diff, signal_basis_info.match_info.y- y_match_diff), cv::Size(template_width_, template_height_), -signal_basis_info.match_info.angle * 180 / CV_PI);
        cv::Point2f vertex[4];
        rotate_rect.points(vertex);
        std::vector<cv::Point2d> pt_vec;
        pt_vec.push_back(vertex[0]);
        pt_vec.push_back(vertex[1]);
        pt_vec.push_back(vertex[2]);
        pt_vec.push_back(vertex[3]);
        pt_vec = connector::order_pts(pt_vec);
        algo_result->result_info.push_back(
            {
                { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "rectangle" },
                { "points", { { pt_vec[0].x, pt_vec[0].y }, { pt_vec[1].x, pt_vec[1].y }, { pt_vec[2].x, pt_vec[2].y }, { pt_vec[3].x, pt_vec[3].y } } },
                { "result", { { "confidence", 1 }, { "area", 0 }, { "index", m} } },
            });


        // 根据匹配的前6个模板计算图像的角度
        if (m < 12) {
            // 前6个模板计算上下屏蔽片的角度
            cv::Mat tmpM = connector::vector_angle_to_M(template_cx_, template_cy_, 0, tmp_ret.x, tmp_ret.y, tmp_ret.angle * 180 / CV_PI);
            // 上屏蔽片
            caliper_param top_parm = *caliper_param_vec_[2];
            std::vector<cv::Point2f> top_vertex = get_rotrect_coords(top_parm.center_x, top_parm.center_y, top_parm.box_width, top_parm.box_height, top_parm.angle, 1);
            // 上方屏蔽片卡尺的4个点
            cv::Point2f top_pt_vec[4];
            top_pt_vec[0] = connector::TransPoint(tmpM, top_vertex[0]);
            top_pt_vec[1] = connector::TransPoint(tmpM, top_vertex[1]);
            top_pt_vec[2] = connector::TransPoint(tmpM, top_vertex[2]);
            top_pt_vec[3] = connector::TransPoint(tmpM, top_vertex[3]);
            bool error_flag = true;
            error_flag &= 0 <= top_pt_vec[0].x && top_pt_vec[0].x <= src.cols && top_pt_vec[0].y >= 0 && top_pt_vec[0].y <= src.rows;
            error_flag &= 0 <= top_pt_vec[1].x && top_pt_vec[1].x <= src.cols && top_pt_vec[1].y >= 0 && top_pt_vec[1].y <= src.rows;
            error_flag &= 0 <= top_pt_vec[2].x && top_pt_vec[2].x <= src.cols && top_pt_vec[2].y >= 0 && top_pt_vec[2].y <= src.rows;
            error_flag &= 0 <= top_pt_vec[3].x && top_pt_vec[3].x <= src.cols && top_pt_vec[3].y >= 0 && top_pt_vec[3].y <= src.rows;
            if (!error_flag) {
                continue;
            }
            nlohmann::json top_line_params = {
                { "CaliperNum", top_parm.caliper_num },
                { "CaliperLength", top_parm.caliper_length },
                { "CaliperWidth", top_parm.caliper_width },
                { "Transition", top_parm.transition },
                { "Sigma", top_parm.sigma },
                { "Num", top_parm.num },
                { "Contrast", 120 },
                { "SortByScore", top_parm.sort_by_score }
            };

            cv::Mat cur_img;
            Tival::TPoint start, end;
            cv::Rect caliper_rect;
            get_caliper_rect_img(top_parm, src, caliper_rect, cur_img, start, end, m, top_pt_vec);
            Tival::FindLineResult top_line_ret = Tival::FindLine::Run(cur_img, start, end, top_line_params);
            if (top_line_ret.start_point.size() > 0) {
                double k = (top_line_ret.start_point[0].y - top_line_ret.end_point[0].y) / (top_line_ret.start_point[0].x - top_line_ret.end_point[0].x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);

                angle_avg = angle_avg + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
            // 下屏蔽片
            caliper_param bot_parm = *caliper_param_vec_[3];
            std::vector<cv::Point2f> bot_vertex = get_rotrect_coords(bot_parm.center_x, bot_parm.center_y, bot_parm.box_width, bot_parm.box_height, bot_parm.angle, 1);
            // 下方屏蔽片卡尺的4个点
            cv::Point2f bot_pt_vec[4];
            bot_pt_vec[0] = connector::TransPoint(tmpM, bot_vertex[0]);
            bot_pt_vec[1] = connector::TransPoint(tmpM, bot_vertex[1]);
            bot_pt_vec[2] = connector::TransPoint(tmpM, bot_vertex[2]);
            bot_pt_vec[3] = connector::TransPoint(tmpM, bot_vertex[3]);
            error_flag = true;
            error_flag &= 0 <= bot_pt_vec[0].x && bot_pt_vec[0].x <= src.cols && bot_pt_vec[0].y >= 0 && bot_pt_vec[0].y <= src.rows;
            error_flag &= 0 <= bot_pt_vec[1].x && bot_pt_vec[1].x <= src.cols && bot_pt_vec[1].y >= 0 && bot_pt_vec[1].y <= src.rows;
            error_flag &= 0 <= bot_pt_vec[2].x && bot_pt_vec[2].x <= src.cols && bot_pt_vec[2].y >= 0 && bot_pt_vec[2].y <= src.rows;
            error_flag &= 0 <= bot_pt_vec[3].x && bot_pt_vec[3].x <= src.cols && bot_pt_vec[3].y >= 0 && bot_pt_vec[3].y <= src.rows;
            if (!error_flag) {
                continue;
            }
            nlohmann::json bot_line_params = {
                { "CaliperNum", bot_parm.caliper_num },
                { "CaliperLength", bot_parm.caliper_length },
                { "CaliperWidth", bot_parm.caliper_width },
                { "Transition", bot_parm.transition },
                { "Sigma", bot_parm.sigma },
                { "Num", bot_parm.num },
                { "Contrast", 120 },
                { "SortByScore", bot_parm.sort_by_score }
            };

            get_caliper_rect_img(bot_parm, src, caliper_rect, cur_img, start, end, m, bot_pt_vec);
            Tival::FindLineResult bot_line_ret = Tival::FindLine::Run(cur_img, start, end, bot_line_params);
            if (bot_line_ret.start_point.size() > 0) {
                double k = (bot_line_ret.start_point[0].y - bot_line_ret.end_point[0].y) / (bot_line_ret.start_point[0].x - bot_line_ret.end_point[0].x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle_avg = angle_avg + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
    }
    //connector::draw_results(dis, algo_result->result_info);

    if (angle_count<=0) {
        return false;
    }
    // 求得旋转矩阵
    angle_avg = angle_avg / angle_count;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point2f(src.cols / 2, src.rows / 2), angle_avg, 1);
    for (int i = 0; i < basis_info_vec.size(); i++) {
        basis_info_vec[i].trans_point.x = m.at<double>(0, 0) * basis_info_vec[i].match_info.x + m.at<double>(0, 1) * basis_info_vec[i].match_info.y + m.at<double>(0, 2);
        basis_info_vec[i].trans_point.y = m.at<double>(1, 0) * basis_info_vec[i].match_info.x + m.at<double>(1, 1) * basis_info_vec[i].match_info.y + m.at<double>(1, 2);
       
    }
    // 比较排序
    std::sort(basis_info_vec.begin(), basis_info_vec.end(), [&](const basis_info& lhs, const basis_info& rhs) {
        // y轴相差500以内是同一行
        if (abs(lhs.trans_point.y - rhs.trans_point.y) <= 400) {
            if (lhs.trans_point.x < rhs.trans_point.x) {
                return true;
            } else {
                return false;
            }
        } else {
            // 不在同一行
            if (lhs.trans_point.y < rhs.trans_point.y) {
                return true;
            } else {
                return false;
            }
        }
    });

    if (basis_info_vec.size() > 1) {
        std::vector<std::vector<cv::Point2d>> rank;
        std::vector<cv::Point2d> swap_vec;
        for (int i = 0; i < basis_info_vec.size() - 1; i++) {
            cv::Point2d cur_pt = basis_info_vec[i].trans_point;
            cv::Point2d next_pt = basis_info_vec[i + 1].trans_point;
            if (std::abs(cur_pt.y - next_pt.y) > 400) {
                // 不是同一行则前一个是上一行，后一个是下一行
                swap_vec.push_back(cur_pt);
                std::vector<cv::Point2d> tmp_vec = swap_vec;
                // 将上一行放进去，另起新行
                rank.push_back(tmp_vec);
                swap_vec.clear();
            } else {
                // 是同一行
                swap_vec.push_back(cur_pt);
                if (i == basis_info_vec.size() - 2) {
                    // 最后一行，最后一个，收尾
                    swap_vec.push_back(next_pt);
                    std::vector<cv::Point2d> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }

        // 所有行第一个最小的值为 作为起始的x点，基座间距是横向坐标为 大约为611像素，第一行的y值作为起始的y点坐标
        double start_x = 0;
        double start_y = 0;
        for (int i = 0; i < rank[0].size(); i++) {
            start_y = start_y + rank[0][i].y;
        }
        start_y = start_y / rank[0].size();

        //x值取最小值,避免第一列有找缺
        start_x = 9999;
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i][0].x <= start_x){
                start_x = rank[i][0].x;
            }
        }
        //需要修改bug, 当第一排或者第一列 检测有缺失时，有误差，纵向 573， 横向 620
        double dis_x_dis = std::abs((rank[0][0].x - rank[0][rank[0].size() - 1].x)) / (rank[0].size() - 1);
        double dis_y_dis = std::abs((rank[0][0].y - rank[rank.size() - 1][0].y)) / (rank.size() - 1);
        //偏差过大时，表示有缺失，使用默认值
        dis_x_dis = std::abs(dis_x_dis - 620) < 50 ? dis_x_dis : 620;
        dis_y_dis = std::abs(dis_y_dis - 573) < 50 ? dis_y_dis : 573;

        //构造设计图纸坐标
        cv::Mat org_point = cv::Mat::zeros(basis_y_num_, basis_x_num_, CV_32FC(6));
        for (int i = 0; i < basis_y_num_; i++) {
            for (int j = 0; j < basis_x_num_;j++) {
                cv::Vec6f& pixel = org_point.at<cv::Vec6f>(i, j);
                pixel[0] = i * basis_x_num_ + j;    //序号
                pixel[1] = start_x + j * dis_x_dis; //图纸的水平，转换到模板的x
                pixel[2] = start_y + i * dis_y_dis; //图纸的水平，转换到模板的y
                pixel[5] = 0;                       //是否为缺失的
                //pixel[3] = ;  //原图的x
                //pixel[4] = ;  //原图的y
            }
        }

        //flann 搜索
        cv::Mat sub_org_point[6];
        cv::split(org_point, sub_org_point);
        std::vector<cv::Mat> channels = { sub_org_point[1], sub_org_point[2]};
        cv::Mat merged;
        cv::merge(channels, merged);
        merged = merged.reshape(1, basis_y_num_ * basis_x_num_);
        //填补图纸的空缺位置，，补齐基座序号
        for (int i = 0; i < basis_info_vec.size(); i++) {
            cv::Point2f cur_pt = basis_info_vec[i].trans_point;
            double minDist = 0;
            int idx = get_nearest_point_idx(merged, cur_pt,minDist);
            int row_idx = idx / basis_x_num_;
            int col_idx = idx % basis_x_num_;
            cv::Vec6f& pixel = org_point.at<cv::Vec6f>(row_idx, col_idx);
            pixel[3] = basis_info_vec[i].match_info.x;
            pixel[4] = basis_info_vec[i].match_info.y;
            basis_info_vec[i].index = (int)pixel[0];
        }


        //补缺根据 模板设计图是否有原图的坐标进行补缺
        cv::Mat l = (cv::Mat_<double>(3, 3) << m.at<double>(0, 0), m.at<double>(0, 1), m.at<double>(0, 2), m.at<double>(1, 0), m.at<double>(1, 1), m.at<double>(1, 2), 0, 0, 1);
        cv::Mat l_inv = l.inv();
        for (int i = 0; i < basis_y_num_; i++) {
            for (int j = 0; j < basis_x_num_; j++) {
                cv::Vec6f& pixel = org_point.at<cv::Vec6f>(i, j);
                //原图缺的位置。
                if (pixel[3]<=1) {
                    bool exist_flag = false;
                    for (int m = 0; m < miss_.size(); m++) {
                        int idx = miss_[m].x * basis_x_num_ +miss_[m].y;
                        if (idx == (int)pixel[0]) {
                            exist_flag = true;
                            break;
                        }
                    }
                    if (exist_flag) continue;
                    //图纸坐标反算到原图
                    cv::Mat r = (cv::Mat_<double>(3, 1) << pixel[1] , pixel[2], 1);
                    cv::Mat tmp = l_inv * r;
                    //补全缺失信息
                    pixel[3] = tmp.at<double>(0, 0);
                    pixel[4] = tmp.at<double>(1, 0);
                    pixel[5] = 1;

                    cv::Point2d pt;
                    pt.x = tmp.at<double>(0, 0);
                    pt.y = tmp.at<double>(1, 0);
                    
                    cv::Point2d p1(pt.x - 40, pt.y - 40);
                    cv::Point2d p2(pt.x + 40, pt.y - 40);
                    cv::Point2d p3(pt.x + 40, pt.y + 40);
                    cv::Point2d p4(pt.x - 40, pt.y + 40);
                    cv::Rect det_rect(detect_left_x_, detect_left_x_, detect_width_, detect_height_);
                    if (det_rect.contains(p1) && det_rect.contains(p2) && det_rect.contains(p3) && det_rect.contains(p4))
                    {
                        algo_result->result_info.push_back(
                            { { "label", "Curved_Bow_Train_defect" },
                                { "shapeType", "basis" },
                                { "points", { { pixel[3],pixel[4] }} },
                                { "result", { { "dist", { -1, -1, -1, -1, -1, -1, -1, -1 } },
                                              { "status", { 0, 0, 0, 0, 0, 0, 0, 0 } },
                                              { "error", { -1, -1, -1, -1, -1, -1, -1, -1} },
                                              { "index", (int)pixel[0]},
                                              { "points", {
                                                  { p1.x, p1.y }, { p2.x, p2.y }, { p3.x, p3.y }, { p4.x, p4.y },
                                                  { p2.x, p2.y }, { p3.x, p3.y }, { p1.x, p1.y }, { p4.x, p4.y },
                                                  { -1, -1 }, { -1,-1 }, { -1, -1 }, { -1, -1 },
                                                  { -1, -1 }, { -1, -1 }, { -1, -1 }, { -1, -1 } }
                                                  }}} });
                    }
                }
            }
        }
        return true;
    }
    return false;
}

void Curved_Bow_Detect::get_caliper_rect_img(const caliper_param& param, const cv::Mat& src, cv::Rect& caliper_rect, cv::Mat& caliper_img, Tival::TPoint& start, Tival::TPoint& end, int index, const cv::Point2f cur_pt_vec[])
{
    std::vector<cv::Point2f> pt_vec;
    pt_vec.push_back(cur_pt_vec[0]);
    pt_vec.push_back(cur_pt_vec[1]);
    pt_vec.push_back(cur_pt_vec[2]);
    pt_vec.push_back(cur_pt_vec[3]);
    pt_vec = connector::order_pts(pt_vec);
    // 当前卡尺的旋转矩形
    cv::RotatedRect rotate_rect(pt_vec[0], pt_vec[1], pt_vec[2]);
    // 当前卡尺的外接矩形，此时的坐标相对于大图
    caliper_rect = rotate_rect.boundingRect2f();
    caliper_img = src(caliper_rect).clone();
    start.X = (cur_pt_vec[0].x + cur_pt_vec[1].x) / 2;
    start.Y = (cur_pt_vec[0].y + cur_pt_vec[1].y) / 2;
    end.X = (cur_pt_vec[2].x + cur_pt_vec[3].x) / 2;
    end.Y = (cur_pt_vec[2].y + cur_pt_vec[3].y) / 2;
    // 返回相对于小图的坐标
    start.X = (start.X - caliper_rect.tl().x) >0 ? (start.X - caliper_rect.tl().x):0;
    start.Y = start.Y - caliper_rect.tl().y;
    end.X = end.X - caliper_rect.tl().x;
    end.Y = end.Y - caliper_rect.tl().y;
}

void Curved_Bow_Detect::basis_find_line(const cv::Mat& src, AlgoResultPtr algo_result, basis_info& cur_basis, int index)
{

    std::vector<cv::Rect> rect_vec;
    std::vector<Tival::FindLineResult> detect_point_vec;
    // 7个卡尺的点
    std::vector<std::vector<cv::Point2f>> cur_pt;
    cv::Mat tmpM = connector::vector_angle_to_M(template_cx_, template_cy_, 0, cur_basis.match_info.x, cur_basis.match_info.y, cur_basis.match_info.angle * 180 / CV_PI);

    for (int i = 0; i < caliper_param_vec_.size(); i++) {
        // 除了第一排的上屏蔽片，其他排不处理上屏蔽片
        Tival::FindLineResult find_line_ret;
        if (index >= basis_x_num_ && i == 2) {
            find_line_ret.start_point.push_back(cv::Point2f(-1,-1)); ;
            find_line_ret.end_point.push_back(cv::Point2f(-1,-1));
            find_line_ret.mid_point.push_back(cv::Point2f(-1,-1));
            detect_point_vec.push_back(find_line_ret);
            continue;
        }

        // 获取原始卡尺的四个点
        caliper_param tmp_caliper_param = *caliper_param_vec_[i];
        std::vector<cv::Point2f> vertex = get_rotrect_coords(tmp_caliper_param.center_x, tmp_caliper_param.center_y, tmp_caliper_param.box_width, tmp_caliper_param.box_height, tmp_caliper_param.angle, 1);

        std::vector<cv::Point2f> cur_caliper_pt;
        // 当前卡尺的4个点
        cv::Point2f cur_pt_vec[4];
        cur_pt_vec[0] = connector::TransPoint(tmpM, vertex[0]);
        cur_pt_vec[1] = connector::TransPoint(tmpM, vertex[1]);
        cur_pt_vec[2] = connector::TransPoint(tmpM, vertex[2]);
        cur_pt_vec[3] = connector::TransPoint(tmpM, vertex[3]);
        cur_caliper_pt.push_back(cur_pt_vec[0]);
        cur_caliper_pt.push_back(cur_pt_vec[1]);
        cur_caliper_pt.push_back(cur_pt_vec[2]);
        cur_caliper_pt.push_back(cur_pt_vec[3]);

        bool error_flag = true;
        error_flag &= 0 <= cur_pt_vec[0].x && cur_pt_vec[0].x <= src.cols && cur_pt_vec[0].y >= 0 && cur_pt_vec[0].y <= src.rows;
        error_flag &= 0 <= cur_pt_vec[1].x && cur_pt_vec[1].x <= src.cols && cur_pt_vec[1].y >= 0 && cur_pt_vec[1].y <= src.rows;
        error_flag &= 0 <= cur_pt_vec[2].x && cur_pt_vec[2].x <= src.cols && cur_pt_vec[2].y >= 0 && cur_pt_vec[2].y <= src.rows;
        error_flag &= 0 <= cur_pt_vec[3].x && cur_pt_vec[3].x <= src.cols && cur_pt_vec[3].y >= 0 && cur_pt_vec[3].y <= src.rows;

        // 变换之后超出图像边界，不进行计算
        if (!error_flag) {
            cur_basis.error_flag = 1;
            return;
        }
        cur_pt.push_back(cur_caliper_pt);

        // 当前图片
        cv::Mat cur_img;
        Tival::TPoint start, end;
        cv::Rect caliper_rect;
        get_caliper_rect_img(tmp_caliper_param, src, caliper_rect, cur_img, start, end, i, cur_pt_vec);

        rect_vec.push_back(caliper_rect);
        // 卡尺数据，卡尺有7个
        caliper_param sub_parm = *caliper_param_vec_[i];
        nlohmann::json find_line_params = {
            { "CaliperNum", sub_parm.caliper_num },
            { "CaliperLength", sub_parm.caliper_length },
            { "CaliperWidth", sub_parm.caliper_width },
            { "Transition", sub_parm.transition },
            { "Sigma", sub_parm.sigma },
            { "Num", sub_parm.num },
            { "Contrast", sub_parm.contrast },
            { "SortByScore", sub_parm.sort_by_score }
        };

        find_line_ret = Tival::FindLine::Run(cur_img, start, end, find_line_params);

        // 第一次未找到，改变极性，减小对比度进行检测
        if (find_line_ret.start_point.size() <= 0) {
            if (find_line_params["Transition"] == "positive") {
                find_line_params["Transition"] = "negative";
            }
            else {
                find_line_params["Transition"] = "positive";
            }
            find_line_params["Contrast"] = find_line_params["Contrast"] - 15;
            find_line_ret = Tival::FindLine::Run(cur_img, start, end, find_line_params);
            if (find_line_ret.start_point.size() <= 0) {
                find_line_params["Transition"] = "all";
                find_line_ret = Tival::FindLine::Run(cur_img, start, end, find_line_params);
            }
            //下屏蔽片重新找
            if (i == 3 && find_line_ret.start_point.size() <= 0) {
                //还原极性
                find_line_params["Transition"] = sub_parm.transition;
                cv::Mat thre_img;
                cv::threshold(cur_img, thre_img,240,255,cv::THRESH_BINARY);
                std::vector<std::vector<cv::Point>> contours = connector::get_contours(thre_img);
                for (const auto& item : contours) 
                {
                    cv::Rect item_rect = cv::boundingRect(item);
                    if (item_rect.width > thre_img.cols - 5) {
                        start.X = item_rect.x + item_rect.width;
                        start.Y = item_rect.y;
                        end.X = item_rect.x;
                        end.Y = item_rect.y;
                        break;
                    }
                }
                find_line_ret = Tival::FindLine::Run(cur_img, start, end, find_line_params);
            }
        }
        // 坐标还原
        if (find_line_ret.start_point.size() != 0) {
            find_line_ret.start_point[0].x = find_line_ret.start_point[0].x + caliper_rect.tl().x;
            find_line_ret.start_point[0].y = find_line_ret.start_point[0].y + caliper_rect.tl().y;
            find_line_ret.end_point[0].x = find_line_ret.end_point[0].x + caliper_rect.tl().x;
            find_line_ret.end_point[0].y = find_line_ret.end_point[0].y + caliper_rect.tl().y;
            find_line_ret.mid_point[0].x = find_line_ret.mid_point[0].x + caliper_rect.tl().x;
            find_line_ret.mid_point[0].y = find_line_ret.mid_point[0].y + caliper_rect.tl().y;
            detect_point_vec.push_back(find_line_ret);
        }
        else {
            // 未找到线
            find_line_ret.start_point.push_back(cv::Point2f(-1, -1)); ;
            find_line_ret.end_point.push_back(cv::Point2f(-1, -1));
            find_line_ret.mid_point.push_back(cv::Point2f(-1, -1));
            detect_point_vec.push_back(find_line_ret);
        }

        cur_basis.rect_vec = rect_vec;
        cur_basis.line_vec = detect_point_vec;
        cur_basis.pt = cur_pt;
        g_pin_lock.lock();
        // 辅助线
        std::vector<cv::Point2f> pt_vec;
        pt_vec.push_back(cur_pt_vec[0]);
        pt_vec.push_back(cur_pt_vec[1]);
        pt_vec.push_back(cur_pt_vec[2]);
        pt_vec.push_back(cur_pt_vec[3]);
        pt_vec = connector::order_pts(pt_vec);

        if (find_line_ret.start_point.size() >= 0 && find_line_ret.start_point[0].x<0) {
            algo_result->result_info.push_back(
                { { "label", "Curved_Bow_Train_defect" },
                    { "shapeType", "rectangle" },
                    { "points", { { pt_vec[0].x, pt_vec[0].y }, { pt_vec[1].x, pt_vec[1].y }, { pt_vec[2].x, pt_vec[2].y }, { pt_vec[3].x, pt_vec[3].y } } },
                    { "result", { { "confidence", 0 }, { "area", 0 }, { "index", -2 } } } });
        }


        g_pin_lock.unlock();
    }

    // 找线结果
    g_pin_lock.lock();
    for (int i = 0; i < detect_point_vec.size(); i++) {
        Tival::FindLineResult cur_line = detect_point_vec[i];
        algo_result->result_info.push_back(
            { { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "line" },
                { "points", { { cur_line.start_point[0].x, cur_line.start_point[0].y}, {cur_line.end_point[0].x, cur_line.end_point[0].y}}},
                { "result", { { "confidence", 1 }, { "area", 0 } } } });
    }
    g_pin_lock.unlock();


    // 计算线段信息
    if (detect_point_vec.size() >= 6) {
        if (detect_point_vec[0].start_point[0].x < 0 || detect_point_vec[1].start_point[0].x < 0 || detect_point_vec[3].start_point[0].x < 0 || detect_point_vec[4].start_point[0].x < 0 || detect_point_vec[5].start_point[0].x < 0 || detect_point_vec[6].start_point[0].x < 0) {
            g_pin_lock.lock();
           cv::RotatedRect rotate_rect(cv::Point2d(cur_basis.match_info.x, cur_basis.match_info.y), cv::Size(template_width_, template_height_), -cur_basis.match_info.angle * 180 / CV_PI);
            cv::Point2f vertex[4];
            rotate_rect.points(vertex);
            std::vector<cv::Point2d> pt_vec;
            pt_vec.push_back(vertex[0]);
            pt_vec.push_back(vertex[1]);
            pt_vec.push_back(vertex[2]);
            pt_vec.push_back(vertex[3]);
            pt_vec = connector::order_pts(pt_vec);

            algo_result->result_info.push_back(
                { { "label", "Curved_Bow_Train_defect" },
                    { "shapeType", "basis" },
                    { "points", { {cur_basis.match_info.x,cur_basis.match_info.y }} },
                    { "result", { { "dist", { -1, -1, -1, -1, -1, -1, -1, -1 } },
                                  { "status", { 0, 0, 0, 0, 0, 0, 0, 0 } },
                                  { "error", { -1, -1, -1, -1, -1, -1, -1, -1} },
                                  { "index", (int)cur_basis.index},
                                  { "points", {
                                      { -1, -1 }, { -1, -1 }, { -1, -1 }, { -1, -1 },
                                      { -1, -1 }, { -1, -1 }, { -1, -1 }, { -1, -1 },
                                      { -1, -1 }, { -1,-1 }, { -1, -1 }, { -1, -1 },
                                      { -1, -1 }, { -1, -1 }, { -1, -1 }, { -1, -1 } }
                                      }}} });
            status_flag = false;
            g_pin_lock.unlock();
            return;
        }
        cv::Point2f param_one[4], param_two[4];
        param_one[0] = cur_pt[0][0];
        param_one[1] = cur_pt[0][1];
        param_one[2] = cur_pt[0][2];
        param_one[3] = cur_pt[0][3];
        param_two[1] = cur_pt[1][0];
        param_two[0] = cur_pt[1][1];
        param_two[2] = cur_pt[1][2];
        param_two[3] = cur_pt[1][3];

        //中线
        Tival::FindLineResult med_line = connector::get_med_line_2(detect_point_vec[4], detect_point_vec[5], detect_point_vec[6]);
        // 求 y基准与x基准的交点,下基座即为y基准
        cv::Point2d cross_x_y = connector::get2lineIPoint(med_line.start_point[0], med_line.end_point[0], detect_point_vec[6].start_point[0], detect_point_vec[6].end_point[0]);

        //左右分开计算

        cv::Point2f pl_5, pr_6;

        cv::Point2f pl = get_caliper_center(param_one, src, "L", pl_5,detect_point_vec[0], algo_result);
        cv::Point2f pr = get_caliper_center(param_two, src, "R", pr_6,detect_point_vec[1], algo_result);

        double d1 = -1;
        double d2 = -1;
        double d3 = -1;
        double d4 = -1;
        double d5 = -1;
        double d6 = -1;
        double d7 = -1;
        double d8 = -1;

        if (pl.x > 0) {
            // 左屏蔽片到基座的x方向距离
            d1 = connector::dist_p2l(pl, med_line.start_point[0], med_line.end_point[0]);
            // 左屏蔽片到基座的Y方向距离
            d3 = connector::dist_p2l(pl, detect_point_vec[6].start_point[0], detect_point_vec[6].end_point[0]);
            // 左屏蔽片到长屏蔽片的的y方向距离
            d5 = connector::dist_p2l(pl_5, detect_point_vec[3].start_point[0], detect_point_vec[3].end_point[0]);
            // 第一排计算与上面的距离,且最上面的线找到
            if (index < basis_x_num_ && detect_point_vec[2].start_point[0].x > 0) {
                // 左屏蔽片到长屏蔽片的的y方向距离
                d7 = connector::dist_p2l(pl_5, detect_point_vec[2].start_point[0], detect_point_vec[2].end_point[0]);
            }
        }


        if (pr.x>0) {
            // 右屏蔽片到基座的x方向距离
           d2 = connector::dist_p2l(pr, med_line.start_point[0], med_line.end_point[0]);
            // 右屏蔽片到基座的Y方向距离
           d4 = connector::dist_p2l(pr, detect_point_vec[6].start_point[0], detect_point_vec[6].end_point[0]);
            //右屏蔽片到长屏蔽片的的y方向距离
           d6 = connector::dist_p2l(pr_6, detect_point_vec[3].start_point[0], detect_point_vec[3].end_point[0]);
           // 第一排计算与上面的距离,且最上面的线找到
           if (index < basis_x_num_ && detect_point_vec[2].start_point[0].x > 0) {
               //右屏蔽片到长屏蔽片的的y方向距离
               d8 = connector::dist_p2l(pr_6, detect_point_vec[2].start_point[0], detect_point_vec[2].end_point[0]);
           }

        }

        //实际距离
        if (pl.x > 0) {
            d1 = d1 * pix_value_ / 1000.0 * ratio_vec_[0] + constant_vec_[0];
            d3 = d3 * pix_value_ / 1000.0 * ratio_vec_[2] + constant_vec_[2];
            d5 = d5 * pix_value_ / 1000.0 * ratio_vec_[4] + constant_vec_[4];
            d7 = d7 * pix_value_ / 1000.0 * ratio_vec_[6] + constant_vec_[6];
        }
        if (pr.x>0) {
            d2 = d2 * pix_value_ / 1000.0 * ratio_vec_[1] + constant_vec_[1];
            d4 = d4 * pix_value_ / 1000.0 * ratio_vec_[3] + constant_vec_[3];
            d6 = d6 * pix_value_ / 1000.0 * ratio_vec_[5] + constant_vec_[5];
            d8 = d8 * pix_value_ / 1000.0 * ratio_vec_[7] + constant_vec_[7];
        }
        if (std::isnan(d1) || std::isnan(d2) || std::isnan(d3) || std::isnan(d4) || std::isnan(d5) || std::isnan(d6) || std::isnan(d7) || std::isnan(d8)) {
            LOGD("algo log run file {}, line {} info{}", __FILE__, __LINE__, "cal  fail");
        }
        //误差
        double error_1=0;
        double error_3=0;
        double error_5=0;
        double error_7=0;
        double error_2=0;
        double error_4=0;
        double error_6=0;
        double error_8=0;
        if (pl.x > 0) {
            error_1 = abs(d1 - d_value_[0]);
            error_3 = abs(d3 - d_value_[2]);
            error_5 = abs(d5 - d_value_[4]);
            error_7 = abs(d7 - d_value_[6]);
        }
        if (pr.x > 0) {
            error_2 = abs(d2 - d_value_[1]);
            error_4 = abs(d4 - d_value_[3]);
            error_6 = abs(d6 - d_value_[5]);
            error_8 = abs(d8 - d_value_[7]);
        }

        //状态
        int status1=0;
        int status2=0;
        int status3=0;
        int status4=0;
        int status5=0;
        int status6=0;
        int status7=0;
        int status8=0;
        
        if (pl.x>0) {
            status1 = error_1 > error_value_[0] ? 0 : 1;
            status3 = error_3 > error_value_[2] ? 0 : 1;
            status5 = error_5 > error_value_[4] ? 0 : 1;
            status7 = error_7 > error_value_[6] ? 0 : 1;
        }
        if (pr.x>0) {
            status2 = error_2 > error_value_[1] ? 0 : 1;
            status4 = error_4 > error_value_[3] ? 0 : 1;
            status6 = error_6 > error_value_[5] ? 0 : 1;
            status8 = error_8 > error_value_[7] ? 0 : 1;
        }

        //屏蔽不需要计算的
        if (index >= basis_x_num_) {
            d7 = -1;
            d8 = -1;
            error_7 = -1;
            error_8 = -1;
            status7 = 1;
            status8 = 1;
        }
        else if(index < basis_x_num_ && (index + 1) % basis_x_num_ != 0){
            //第一行且不是最后一个
            d8 = -1;
            error_8 = -1;
            status8 = 1;
        }
        // 取余，每行不是最后一个，只取左侧的屏蔽片
        if ((index + 1) % basis_x_num_ != 0) {
            //2024年6月17日11:04:04 修改，取值右侧屏蔽片数据
           /* d6 = -1;
            error_6 = -1;
            status6 = 1;*/
            int row_idx = (index + 1) / basis_x_num_;
            int col_idx = (index + 1) % basis_x_num_;
            int next_exit_flag = 1;
            for (int m = 0; m < miss_.size(); m++) {
                if ((row_idx == miss_[m].x && col_idx == miss_[m].y)) {
                    next_exit_flag = 0;
                    break;
                }
            }
            if (next_exit_flag == 0) {
            //下一个不存在不操作，
            }
            else {
                //下一个存在，只取左边的
                d6 = -1;
                error_6 = -1;
                status6 = 1; 
            }

            /*d4 = -1;
            error_4 = -1;
            status4 = -1;*/
        }
        //全图状态
        if (status1 && status2 && status3 && status4 && status5 && status6 && status7 && status8) {

        } else {
            status_flag = false;
        }

        // 求画图的交点，左右分开画
        //辅助数据
        cv::Point2f tmp_pt;
        double diff_x = 0;
        //左边
        cv::Point2f pt_l_l(-1,-1);
        cv::Point2f pt_l_r(-1,-1);
        cv::Point2f pt_l_t(-1, -1);
        cv::Point2f pt_l_b(-1, -1);
        cv::Point2f pt_l_d5_t(-1, -1);
        cv::Point2f pt_l_d5_b(-1, -1);
        cv::Point2f d7_l_t(-1,-1);
        cv::Point2f d7_l_b(-1,-1);
        if (pl.x>0) {
            // 左边的d1的两个点
            tmp_pt = connector::calculate_foot_point(med_line.start_point[0], med_line.end_point[0], pl);
            pt_l_l.y = tmp_pt.y - 85;
            pt_l_l.x = connector::get_line_x(med_line.start_point[0], med_line.end_point[0], pt_l_l.y);
            double diff_x = pt_l_l.x - tmp_pt.x;
            pt_l_r.y = pl.y - 85;
            pt_l_r.x = pl.x + diff_x;
           // 左边d3的两个点
            pt_l_t = connector::calculate_foot_point(detect_point_vec[6].start_point[0], detect_point_vec[6].end_point[0], pl);
            pt_l_b = pl;

            if ((index + 1) % basis_x_num_ != 0) {
                // 左边d5的两个点
                pt_l_d5_t = pl_5;
                pt_l_d5_b = connector::calculate_foot_point(detect_point_vec[3].start_point[0], detect_point_vec[3].end_point[0], pl_5);
            }
            else {
                // 左边d5的两个点
                pt_l_d5_t = pl_5;
                pt_l_d5_b = connector::calculate_foot_point(detect_point_vec[3].start_point[0], detect_point_vec[3].end_point[0], pl_5);
            }
            // 左边d7的两个点
            if (index >= basis_x_num_) {
                d7_l_t = cv::Point2f(-1, -1);
                d7_l_b = cv::Point2f(-1, -1);
            
            }
            else if (index < basis_x_num_ && (index + 1) % basis_x_num_ != 0) {
                d7_l_b = pl_5;
                d7_l_t = connector::calculate_foot_point(detect_point_vec[2].start_point[0], detect_point_vec[2].end_point[0], d7_l_b);
            }
            else {
                d7_l_b = pl_5;
                d7_l_t = connector::calculate_foot_point(detect_point_vec[2].start_point[0], detect_point_vec[2].end_point[0], d7_l_b);
            }
        }

        //右边
        cv::Point2f pt_r_r(-1,-1);
        cv::Point2f pt_r_l(-1,-1);
        cv::Point2f pt_r_t(-1, -1);
        cv::Point2f pt_r_b(-1, -1);
        cv::Point2f pt_r_d6_t(-1,-1);
        cv::Point2f pt_r_d6_b(-1,-1);
        cv::Point2f d8_r_t(-1,-1);
        cv::Point2f d8_r_b(-1,-1);
        if (pr.x>0) {
            // 右边的d2的两个点
            tmp_pt = connector::calculate_foot_point(med_line.start_point[0], med_line.end_point[0], pr);
            pt_r_r.y = tmp_pt.y - 85;
            pt_r_r.x = connector::get_line_x(med_line.start_point[0], med_line.end_point[0], pt_r_r.y);
            diff_x = pt_r_r.x - tmp_pt.x;
            pt_r_l.y = pr.y - 85;
            pt_r_l.x = pr.x + diff_x;
            // 右边d4的两个点
            pt_r_t = connector::calculate_foot_point(detect_point_vec[6].start_point[0], detect_point_vec[6].end_point[0], pr);
            pt_r_b = pr;
            if ((index + 1) % basis_x_num_ != 0) {
                // 右边d6的两个点
                pt_r_d6_t = cv::Point2f(-1, -1);
                pt_r_d6_b = cv::Point2f(-1, -1);
                int row_idx = (index+1) / basis_x_num_;
                int col_idx = (index+1) % basis_x_num_;

                for (int m = 0; m < miss_.size();m++) {
                    if (row_idx == miss_[m].x && col_idx==miss_[m].y) {
                        pt_r_d6_t = pr_6;
                        pt_r_d6_b = connector::calculate_foot_point(detect_point_vec[3].start_point[0], detect_point_vec[3].end_point[0], pr_6);
                    }
                }
               
            }
            else {
                // 右边d6的两个点
                pt_r_d6_t = pr_6;
                pt_r_d6_b = connector::calculate_foot_point(detect_point_vec[3].start_point[0], detect_point_vec[3].end_point[0], pr_6);
            }
            // 右边d8的两个点
            if (index >= basis_x_num_) {
                d8_r_t = cv::Point2f(-1, -1);
                d8_r_b = cv::Point2f(-1, -1);
            }
            else if (index < basis_x_num_ && (index + 1) % basis_x_num_ != 0) {
                d8_r_t = cv::Point2f(-1, -1);
                d8_r_b = cv::Point2f(-1, -1);
            }
            else {
                d8_r_b = pr_6;
                d8_r_t = connector::calculate_foot_point(detect_point_vec[2].start_point[0], detect_point_vec[2].end_point[0], d8_r_b);
            }
        }

        g_pin_lock.lock();
        algo_result->result_info.push_back(
            { { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "point" },
                { "points", { { pl.x, pl.y } } },
                { "result", { { "confidence", 1 }, { "area", 0 } } } });
        algo_result->result_info.push_back(
            { { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "point" },
                { "points", { { pr.x, pr.y } } },
                { "result", { { "confidence", 1 }, { "area", 0 } } } });

        algo_result->result_info.push_back(
            { { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "line" },
                { "points", { { med_line.start_point[0].x, med_line.start_point[0].y}, {med_line.end_point[0].x, med_line.end_point[0].y}}},
                { "result", { { "confidence", 1 }, { "area", 0 } } } });
        algo_result->result_info.push_back(
            { { "label", "" },
                { "shapeType", "basis" },
                { "points", { { cross_x_y.x, cross_x_y.y } } },
                { "result", { { "dist", { d1, d2, d3, d4, d5, d6, d7, d8 } }, { "status", { status1, status2, status3, status4, status5, status6, status7, status8 } }, { "error", { error_1, error_2, error_3, error_4, error_5, error_6, error_7, error_8 } }, { "index", cur_basis.index }, { "points", { { pt_l_l.x, pt_l_l.y }, { pt_l_r.x, pt_l_r.y }, { pt_r_l.x, pt_r_l.y }, { pt_r_r.x, pt_r_r.y }, { pt_l_t.x, pt_l_t.y }, { pt_l_b.x, pt_l_b.y }, { pt_r_t.x, pt_r_t.y }, { pt_r_b.x, pt_r_b.y }, { pt_l_d5_t.x, pt_l_d5_t.y }, { pt_l_d5_b.x, pt_l_d5_b.y }, { pt_r_d6_t.x, pt_r_d6_t.y }, { pt_r_d6_b.x, pt_r_d6_b.y }, { d7_l_t.x, d7_l_t.y }, { d7_l_b.x, d7_l_b.y }, { d8_r_t.x, d8_r_t.y }, { d8_r_b.x, d8_r_b.y } }

                                                                                                                                                                                                                                                                                               } } } });
        g_pin_lock.unlock();
    }
    return;
}
