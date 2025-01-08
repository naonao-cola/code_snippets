#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "trayTowerDet.h"
#include <AIRuntimeInterface.h>
#include <AIRuntimeDataStruct.h>
#include <AIRuntimeUtils.h>
#include "JsonHelper.h"
#include <filesystem>
#include "../utils/Utils.h"
#include "param_check.h"
#include "algo_tool.h"
#include "../../3rdparty/ed/EDLib.h"
REGISTER_ALGO(trayTowerDet)

static int SAVE_IMG_TOWER_INDEX = 0;
constexpr int MIN_TEMPLATE_HEIGHT = 1150;
constexpr int MAX_TEMPLATE_HEIGHT = 1750;
constexpr int MIN_TEMPLATE_WEIGHT = 3100;
constexpr int MAX_TEMPLATE_WEIGHT = 3800;

trayTowerDet::trayTowerDet() {}
trayTowerDet::~trayTowerDet() {}
cv::Point2f get_cross_pt_2(const cv::Mat& img, const cv::Rect& rect, int type);

AlgoResultPtr trayTowerDet::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    TVALGO_FUNCTION_BEGIN
    algo_result->result_info.push_back(
        {
            {"label", "trayTowerDet"},
            {"shapeType", "rectangle"},
            {"points", {{0, 0}, {0, 0}}},
            {"result", {{"confidence", 0}, {"area", 0}}},
        });
#pragma region param_section
    CV_DbgAssert(!task->image.empty());
    cv::Mat dst;
    try {
        get_param(task);
    } catch (const std::exception& e) {
        algo_result->judge_result = 0;
        TVALGO_FUNCTION_RETURN_ERROR_PARAM(e.what())
    }
    if (task->image.channels() > 1) {
        cv::cvtColor(task->image, dst, cv::COLOR_BGR2GRAY);
    } else {
        dst = task->image.clone();
    }
#pragma endregion

#pragma region algo_section
    try {
        if (tower_type_ == 0) {
            img_process1(dst, algo_result);
        } else {
            img_process2(dst, algo_result);
        }
    } catch (const std::exception& e) {
        algo_result->judge_result = 0;
        TVALGO_FUNCTION_RETURN_ERROR_PARAM(e.what())
    }
#pragma endregion
    TVALGO_FUNCTION_LOG(algo_result->judge_result)
    TVALGO_FUNCTION_END
}

std::tuple<std::string, json> trayTowerDet::get_task_info(InferTaskPtr task, std::map<std::string, json> param_map) const
{
    std::string task_type_id = task->image_info["type_id"];
    json task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

bool trayTowerDet::get_param(InferTaskPtr task)
{
    auto [task_type_id, task_param_json] = get_task_info(task, m_param_map);
    template_img_path_1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "template_img_path_1", std::string(""));
    template_img_path_2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "template_img_path_2", std::string(""));
    tower_type_ = Tival::JsonHelper::GetParam(task_param_json["param"], "tower_type", 0);
    area_th_ = Tival::JsonHelper::GetParam(task_param_json["param"], "area_th", 120);
    img_th_ = Tival::JsonHelper::GetParam(task_param_json["param"], "img_th", 150);
    img_th_2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "img_th_2", 150);
    if (!template_img_path_1_.empty() && template_img_1_.empty()) {
        template_img_1_ = cv::imread(template_img_path_1_, 0);
    }
    if (!template_img_path_2_.empty() && template_img_2_.empty()) {
        template_img_2_ = cv::imread(template_img_path_2_, 0);
    }
    if (IsDebug()) {
        image_file_name_ = task->image_info["img_path"];
        image_file_name_ = std::filesystem::path(image_file_name_).filename().string();
    }
    return true;
}

cv::Point get_his(const cv::Mat& img, int type)
{
    cv::Mat row_mat, col_mat;
    cv::reduce(img, col_mat, 0, cv::REDUCE_AVG);
    cv::reduce(img, row_mat, 1, cv::REDUCE_AVG);
    std::vector<int> row_vec = row_mat.reshape(1, 1);
    std::vector<int> col_vec = col_mat.reshape(1, 1);
    int result_i = 0;
    int result_j = 0;
    if (type == 1) {
        for (int i = 0; i < col_vec.size(); i++) {
            if (col_vec[i] >= 40) {
                result_i = i;
                break;
            }
        }
        for (int i = 0; i < row_vec.size(); i++) {
            if (row_vec[i] >= 40) {
                result_j = i;
                break;
            }
        }
    }
    if (type == 2) {
        for (int i = col_vec.size() - 1; i > 0; i--) {
            if (col_vec[i] >= 40) {
                result_i = i;
                break;
            }
        }
        for (int i = 0; i < row_vec.size(); i++) {
            if (row_vec[i] >= 40) {
                result_j = i;
                break;
            }
        }
    }
    if (type == 3) {
        for (int i = col_vec.size() - 1; i > 0; i--) {
            if (col_vec[i] >= 40) {
                result_i = i;
                break;
            }
        }
        for (int i = row_vec.size() - 1; i > 0; i--) {
            if (row_vec[i] >= 40) {
                result_j = i;
                break;
            }
        }
    }
    if (type == 4) {
        for (int i = 0; i < col_vec.size(); i++) {
            if (col_vec[i] >= 40) {
                result_i = i;
                break;
            }
        }
        for (int i = row_vec.size() - 1; i > 0; i--) {
            if (row_vec[i] >= 40) {
                result_j = i;
                break;
            }
        }
    }
    return cv::Point(result_i, result_j);
}

std::vector<cv::Point> trayTowerDet::re_get_rotated_pt(const cv::Mat& input_img, int input_th_value, const std::vector<cv::Point2f>& input_pts)
{
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 2));
    cv::Rect left_top_rect(input_pts[0].x - 50, input_pts[0].y - 50, 150, 150);
    cv::Mat left_top_img = input_img(left_top_rect);
    cv::Mat left_top_th_img;
    cv::threshold(left_top_img, left_top_th_img, input_th_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(left_top_th_img, left_top_th_img, elementX);
    cv::erode(left_top_th_img, left_top_th_img, elementX);
    cv::Point lt_pt = get_his(left_top_th_img, 1);
    lt_pt.x = lt_pt.x + left_top_rect.x;
    lt_pt.y = lt_pt.y + left_top_rect.y;

    cv::Rect right_top_rect(input_pts[1].x - 100, input_pts[1].y - 50, 150, 150);
    cv::Mat right_top_img = input_img(right_top_rect);
    cv::Mat right_top_th_img;
    cv::threshold(right_top_img, right_top_th_img, input_th_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(right_top_th_img, right_top_th_img, elementX);
    cv::erode(right_top_th_img, right_top_th_img, elementX);
    cv::Point rt_pt = get_his(right_top_th_img, 2);
    rt_pt.x = rt_pt.x + right_top_rect.x;
    rt_pt.y = rt_pt.y + right_top_rect.y;

    cv::Rect right_bottom_rect(input_pts[2].x - 100, input_pts[2].y - 80, 150, 150);
    cv::Mat right_bottom_img = input_img(right_bottom_rect);
    cv::Mat right_bottom_th_img;
    cv::threshold(right_bottom_img, right_bottom_th_img, input_th_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(right_bottom_th_img, right_bottom_th_img, elementX);
    cv::erode(right_bottom_th_img, right_bottom_th_img, elementX);
    cv::Point rb_pt = get_his(right_bottom_th_img, 3);
    rb_pt.x = rb_pt.x + right_bottom_rect.x;
    rb_pt.y = rb_pt.y + right_bottom_rect.y;

    cv::Rect left_bottom_rect(input_pts[3].x - 50, input_pts[3].y - 100, 150, 150);
    cv::Mat left_bottom_img = input_img(left_bottom_rect);
    cv::Mat left_bottom_th_img, left_bottom_canny_img;
    cv::threshold(left_bottom_img, left_bottom_th_img, input_th_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(left_bottom_th_img, left_bottom_th_img, elementX);
    cv::erode(left_bottom_th_img, left_bottom_th_img, elementX);
    cv::Point lb_pt = get_his(left_bottom_th_img, 4);
    lb_pt.x = lb_pt.x + left_bottom_rect.x;
    lb_pt.y = lb_pt.y + left_bottom_rect.y;

    std::vector<cv::Point> result_pt_vec;
    result_pt_vec.push_back(lt_pt);
    result_pt_vec.push_back(rt_pt);
    result_pt_vec.push_back(rb_pt);
    result_pt_vec.push_back(lb_pt);
    return result_pt_vec;
}

std::vector<cv::Point2f> trayTowerDet::get_pts(const cv::Mat& img, const std::vector<cv::Point2f>& pt_vec, int threshold_value)
{
#pragma region rotate_section
    std::vector<cv::Point2f> sorted_pts = tool::order_pts(pt_vec);
    double k1 = (sorted_pts[1].y - sorted_pts[0].y) / (sorted_pts[1].x - sorted_pts[0].x);
    double k2 = (sorted_pts[2].y - sorted_pts[3].y) / (sorted_pts[2].x - sorted_pts[3].x);
    double angle = atanl((k1 + k2) / 2) * 180.0 / CV_PI;
    cv::Mat m, inv_m, rotate_img;
    std::vector<cv::Point2f> rotated_pts;
    m = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    cv::warpAffine(img, rotate_img, m, img.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    for (int i = 0; i < sorted_pts.size(); i++) {
        cv::Point2d tm_pt = tool::TransPoint(m, sorted_pts[i]);
        rotated_pts.push_back(tm_pt);
    }
#pragma endregion

    std::vector<cv::Point> re_vec_pt = re_get_rotated_pt(rotate_img, threshold_value, rotated_pts);
    std::vector<cv::Point2f> ret_vec;
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 2));

#pragma region LT_PT
    cv::Rect left_top_rect(re_vec_pt[0].x - 30, re_vec_pt[0].y - 30, 100, 100);
    cv::Mat left_top_img = rotate_img(left_top_rect);
    cv::Mat left_top_th_img;
    cv::threshold(left_top_img, left_top_th_img, threshold_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(left_top_th_img, left_top_th_img, elementX);
    cv::erode(left_top_th_img, left_top_th_img, elementX);
    auto contours = tool::get_contours(left_top_th_img);
    std::vector<std::vector<cv::Point>> left_top_contours = contours.value();
    for (int i = 0; i < left_top_contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(left_top_contours[i]);
        int max_w = std::max(rect.width, rect.height);
        std::vector<std::vector<cv::Point>> draw_conts = {left_top_contours[i]};
        if (max_w < 30) {
            cv::drawContours(left_top_th_img, draw_conts, 0, 0, -1);
        }
    }
    cv::Point2f left_top_pt = get_cross_pt_2(left_top_th_img, left_top_rect, 0);
    if (left_top_pt.x < 5) {
        ret_vec.clear();
        return ret_vec;
    }
    ret_vec.emplace_back(left_top_pt);
#pragma endregion

#pragma region RT_PT
    cv::Rect right_top_rect(re_vec_pt[1].x - 60, re_vec_pt[1].y - 30, 100, 100);
    cv::Mat right_top_img = rotate_img(right_top_rect);
    cv::Mat right_top_th_img;
    cv::threshold(right_top_img, right_top_th_img, threshold_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(right_top_th_img, right_top_th_img, elementX);
    cv::erode(right_top_th_img, right_top_th_img, elementX);
    contours = tool::get_contours(right_top_th_img);
    std::vector<std::vector<cv::Point>> right_top_contours = contours.value();
    for (int i = 0; i < right_top_contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(right_top_contours[i]);
        int max_w = std::max(rect.width, rect.height);
        std::vector<std::vector<cv::Point>> draw_conts = {right_top_contours[i]};
        if (max_w < 30) {
            cv::drawContours(right_top_th_img, draw_conts, 0, 0, -1);
        }
    }
    cv::Point2f right_top_pt = get_cross_pt_2(right_top_th_img, right_top_rect, 1);
    if (right_top_pt.x < 5) {
        ret_vec.clear();
        return ret_vec;
    }
    ret_vec.emplace_back(right_top_pt);
#pragma endregion

#pragma region RB_LT
    cv::Rect right_bottom_rect(re_vec_pt[2].x - 60, re_vec_pt[2].y - 70, 100, 100);
    cv::Mat right_bottom_img = rotate_img(right_bottom_rect);
    cv::Mat right_bottom_th_img;
    cv::threshold(right_bottom_img, right_bottom_th_img, threshold_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(right_bottom_th_img, right_bottom_th_img, elementX);
    cv::erode(right_bottom_th_img, right_bottom_th_img, elementX);
    contours = tool::get_contours(right_bottom_th_img);
    std::vector<std::vector<cv::Point>> right_bottom_contours = contours.value();
    for (int i = 0; i < right_bottom_contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(right_bottom_contours[i]);
        std::vector<std::vector<cv::Point>> draw_conts = {right_bottom_contours[i]};
        int max_w = std::max(rect.width, rect.height);
        double area = cv::contourArea(right_bottom_contours[i]);
        if (max_w < 30 || area < 600) {
            cv::drawContours(right_bottom_th_img, draw_conts, 0, 0, -1);
        }
    }
    cv::Point2f right_bottom_pt = get_cross_pt_2(right_bottom_th_img, right_bottom_rect, 2);
    if (right_bottom_pt.x < 5) {
        ret_vec.clear();
        return ret_vec;
    }
    ret_vec.emplace_back(right_bottom_pt);
#pragma endregion

#pragma region LB_PT
    cv::Rect left_bottom_rect(re_vec_pt[3].x - 30, re_vec_pt[3].y - 70, 100, 100);
    cv::Mat left_bottom_img = rotate_img(left_bottom_rect);
    cv::Mat left_bottom_th_img;
    cv::threshold(left_bottom_img, left_bottom_th_img, threshold_value, 255, cv::THRESH_BINARY_INV);
    cv::dilate(left_bottom_th_img, left_bottom_th_img, elementX);
    cv::erode(left_bottom_th_img, left_bottom_th_img, elementX);
    contours = tool::get_contours(left_bottom_th_img);
    std::vector<std::vector<cv::Point>> left_bottom_contours = contours.value();
    for (int i = 0; i < left_bottom_contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(left_bottom_contours[i]);
        std::vector<std::vector<cv::Point>> draw_conts = {left_bottom_contours[i]};
        int max_w = std::max(rect.width, rect.height);
        double area = cv::contourArea(left_bottom_contours[i]);
        if (max_w < 30 || area < 600) {
            cv::drawContours(left_bottom_th_img, draw_conts, 0, 0, -1);
        }
    }
    cv::Point2f left_bottom_pt = get_cross_pt_2(left_bottom_th_img, left_bottom_rect, 3);
    if (left_bottom_pt.x < 5) {
        ret_vec.clear();
        return ret_vec;
    }
    ret_vec.emplace_back(left_bottom_pt);
#pragma endregion
    std::vector<cv::Point2f> finnal_pt;
    for (int i = 0; i < ret_vec.size(); i++) {
        cv::Point2f tm_pt = tool::TransPoint(inv_m, ret_vec[i]);
        finnal_pt.push_back(tm_pt);
    }
    return finnal_pt;
}

void trayTowerDet::img_process1(const cv::Mat& src, AlgoResultPtr algo_result)
{
    if (!template_img_path_1_.empty() && template_img_1_.empty()) {
        template_img_1_ = cv::imread(template_img_path_1_, 0);
    }
    cv::Mat template_img_th_img;
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

#pragma region TMEPLATE_PRE
    if (img_process_1_flag_ == 0) {
        cv::RotatedRect temp_rotate_rect;
        cv::medianBlur(template_img_1_, template_img_th_img, 7);
        tool::sauvola(template_img_th_img, 0.1, 15);
        cv::erode(template_img_th_img, template_img_th_img, elementX);
        cv::bitwise_not(template_img_th_img, template_img_th_img);
        auto contours_ret = tool::get_contours(template_img_th_img);
        std::vector<std::vector<cv::Point>> contours_temp_img = contours_ret.value();
        for (size_t i = 0; i < contours_temp_img.size(); ++i) {
            cv::RotatedRect rot_rect = cv::minAreaRect(contours_temp_img[i]);
            cv::Rect rect = cv::boundingRect(contours_temp_img[i]);
            int width = std::max(rot_rect.size.width, rot_rect.size.height);
            int heigh = std::min(rot_rect.size.width, rot_rect.size.height);
            if (width > MIN_TEMPLATE_WEIGHT && heigh > MIN_TEMPLATE_HEIGHT && width < MAX_TEMPLATE_WEIGHT) {
                temp_rotate_rect = rot_rect;
                temp_rect_1_ = rect;
            }
        }
        // original template image
        template_img_org_1_ = template_img_1_.clone();
        // obtain rotated points and images
        auto rotate_ret = tool::get_rotated_rect_pt_trans_and_rotate(temp_rotate_rect, template_img_org_1_);
        temp_pts_1_ = rotate_ret.value();
        temp_pts_1_ = get_pts(template_img_org_1_, temp_pts_1_, img_th_);
        /*
        旋转过后，不平整，做精细化调整
        */
        if (abs(temp_pts_1_[0].y - temp_pts_1_[1].y) > 5) {
            cv::Point2f src_pts[4], dst_pts[4];
            cv::Mat m, inv_m, rotate_img;
            double w = std::sqrtf(std::powf((temp_pts_1_[0].x - temp_pts_1_[1].x), 2) + std::powf((temp_pts_1_[0].y - temp_pts_1_[1].y), 2));
            double h = std::sqrtf(std::powf((temp_pts_1_[2].x - temp_pts_1_[1].x), 2) + std::powf((temp_pts_1_[2].y - temp_pts_1_[1].y), 2));
            for (int i = 0; i < temp_pts_1_.size(); i++) {
                src_pts[i] = temp_pts_1_[i];
            }
            dst_pts[0] = temp_pts_1_[0];
            dst_pts[1] = cv::Point2d(temp_pts_1_[0].x + w, temp_pts_1_[0].y);
            dst_pts[2] = cv::Point2d(temp_pts_1_[0].x + w, temp_pts_1_[0].y + h);
            dst_pts[3] = cv::Point2d(temp_pts_1_[0].x, temp_pts_1_[0].y + h);
            m = cv::getPerspectiveTransform(src_pts, dst_pts);
            cv::warpPerspective(template_img_org_1_, template_img_org_1_, m, template_img_org_1_.size());
            temp_pts_1_.clear();
            for (int i = 0; i < 4; i++) {
                temp_pts_1_.push_back(dst_pts[i]);
            }
        }
        img_process_1_flag_++;
    }
#pragma endregion
    // generate mask
    cv::Mat temp_mask = cv::Mat::zeros(template_img_org_1_.size(), template_img_org_1_.type());
    for (int i = 0; i < temp_mask_1_.size(); i++) {
        cv::Vec4i cur = temp_mask_1_[i];
        cv::Rect cur_rect(cur[0] + temp_pts_1_[0].x, cur[1] + temp_pts_1_[0].y, cur[2], cur[3]);
        temp_mask(cur_rect) = 255;
    }
    // process the original image
    cv::Mat src_img;
    cv::Rect src_rect;
    cv::RotatedRect src_rotate_rect;
    std::vector<std::vector<cv::Point>> contours_src_img;
    src_img = src.clone();
    tool::sauvola(src_img, 0.1, 15);
    cv::erode(src_img, src_img, elementX);
    cv::bitwise_not(src_img, src_img);
    auto contours_src_ret = tool::get_contours(src_img);
    contours_src_img = contours_src_ret.value();
    for (size_t i = 0; i < contours_src_img.size(); ++i) {
        cv::RotatedRect rot_rect = cv::minAreaRect(contours_src_img[i]);
        cv::Rect rect = cv::boundingRect(contours_src_img[i]);
        int width = std::max(rot_rect.size.width, rot_rect.size.height);
        int heigh = std::min(rot_rect.size.width, rot_rect.size.height);
        if (width > MIN_TEMPLATE_WEIGHT && heigh > MIN_TEMPLATE_HEIGHT && width <= MAX_TEMPLATE_WEIGHT) {
            src_rotate_rect = rot_rect;
            src_rect = rect;
        }
    }
    int width = std::max(src_rotate_rect.size.width, src_rotate_rect.size.height);
    int heigh = std::min(src_rotate_rect.size.width, src_rotate_rect.size.height);
    if (width <= MIN_TEMPLATE_WEIGHT || heigh > MAX_TEMPLATE_HEIGHT || width >= MAX_TEMPLATE_WEIGHT) {
        algo_result->judge_result = 0;
        return;
    }
    // get rotated images with points
    auto src_rotate_pts_op = tool::get_rotated_rect_pts(src_rotate_rect);
    std::vector<cv::Point2f> src_rotate_pts = src_rotate_pts_op.value();
    cv::Mat dis_img;
    if (IsDebug()) {
        cv::cvtColor(src, dis_img, cv::COLOR_GRAY2BGR);
        for (size_t j = 0; j < 4; j++) {
            cv::line(dis_img, src_rotate_pts[j], src_rotate_pts[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    src_rotate_pts = get_pts(src, src_rotate_pts, img_th_);
    // second find
    if (src_rotate_pts.size() < 3) {
        cv::threshold(src_img, src_img, 75, 255, cv::THRESH_TOZERO);
        contours_src_ret = tool::get_contours(src_img);
        contours_src_img = contours_src_ret.value();
        for (size_t i = 0; i < contours_src_img.size(); ++i) {
            cv::RotatedRect rot_rect = cv::minAreaRect(contours_src_img[i]);
            cv::Rect rect = cv::boundingRect(contours_src_img[i]);
            int width = std::max(rot_rect.size.width, rot_rect.size.height);
            int heigh = std::min(rot_rect.size.width, rot_rect.size.height);
            if (width > MIN_TEMPLATE_WEIGHT && heigh > MIN_TEMPLATE_HEIGHT && width <= MAX_TEMPLATE_WEIGHT) {
                src_rotate_rect = rot_rect;
                src_rect = rect;
            }
        }
        src_rotate_pts_op = tool::get_rotated_rect_pts(src_rotate_rect);
        src_rotate_pts = src_rotate_pts_op.value();
        src_rotate_pts = get_pts(src, src_rotate_pts, img_th_);
    }
    std::vector<std::vector<cv::Point2f>> dis_pt_vec;
    if (src_rotate_pts.size() < 4) {
        algo_result->judge_result = 0;
        TVALGO_FUNCTION_LOG("src rotate_pts size is 0")
        write_debug_img(src, dis_pt_vec, 1);
        return;
    }
    auto equal_ret = tool::get_equal_img_2(src, src_rotate_pts, template_img_org_1_, temp_pts_1_);
    cv::Mat equal_img = equal_ret.value();
    if (IsDebug()) {
        for (size_t m = 0; m < src_rotate_pts.size(); m++) {
            cv::circle(dis_img, src_rotate_pts[m], 3, cv::Scalar(0, 0, 255));
        }
        std::string save_name = "item_tower_pt_" + std::to_string(SAVE_IMG_TOWER_INDEX) + ".jpg";
    }
    // perspective change and difference
    cv::Mat wrap_mat, src_warp_img, cdf, cdf_roi, csf_th_img, csf_copy;

    tool::perspective(src_rotate_pts, temp_pts_1_, wrap_mat);
    cv::warpPerspective(equal_img, src_warp_img, wrap_mat, equal_img.size());
    cv::absdiff(src_warp_img, template_img_org_1_, cdf);
    cdf.copyTo(cdf_roi, temp_mask);
    cv::threshold(cdf_roi, csf_th_img, img_th_2_, 255, cv::THRESH_BINARY);
    algo_result->judge_result = 1;

    // result determination
    csf_copy = csf_th_img.clone();
    for (int i = 0; i < temp_mask_1_.size(); i++) {
        cv::Vec4i cur = temp_mask_1_[i];
        cv::Rect cur_rect(cur[0] + temp_pts_1_[0].x, cur[1] + temp_pts_1_[0].y, cur[2], cur[3]);
        cv::Mat cur_img = csf_th_img(cur_rect);
        int no_zero = cv::countNonZero(cur_img);
        if (no_zero <= area_th_) {
            csf_copy(cur_rect) = 0;
        } else {
            // 模板匹配，中点，是为了去除偏差
            cv::Point center_pt{cur_rect.x + cur_rect.width / 2, cur_rect.y + cur_rect.height / 2};
            cv::Mat crop_src_img = src_warp_img(cv::Rect(center_pt.x - 25, center_pt.y - 25, 50, 50));
            cv::Mat crop_temp_img = template_img_org_1_(cv::Rect(center_pt.x - 40, center_pt.y - 40, 80, 80));
            cv::Mat match_r;
            cv::matchTemplate(crop_src_img, crop_temp_img, match_r, cv::TM_CCOEFF_NORMED);
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(match_r, &min_val, &max_val, &min_loc, &max_loc);
            if (max_val > 0.85) {
                // 匹配到则裁剪图片
                cv::Mat compare_temp_img = crop_temp_img(cv::Rect(max_loc.x, max_loc.y, crop_src_img.cols, crop_src_img.rows));
                cv::Mat compare_ret, compare_th_img;
                cv::absdiff(crop_src_img, compare_temp_img, compare_ret);
                cv::threshold(compare_ret, compare_th_img, img_th_2_, 255, cv::THRESH_BINARY);
                int compare_no_zero = cv::countNonZero(compare_th_img);
                if (compare_no_zero <= area_th_) {
                    csf_copy(cur_rect) = 0;
                } else {
                    // 面积大于阈值
                    algo_result->judge_result = 0;
                    std::vector<cv::Point2f> cur_pts = wrap_point(cur_rect, wrap_mat, algo_result);
                    dis_pt_vec.push_back(cur_pts);
                }
            } else {
                // 没有匹配到
                algo_result->judge_result = 0;
                std::vector<cv::Point2f> cur_pts = wrap_point(cur_rect, wrap_mat, algo_result);
                dis_pt_vec.push_back(cur_pts);
            }
        }
    }
    write_debug_img(src, dis_pt_vec);
}

void trayTowerDet::img_process2(const cv::Mat& src, AlgoResultPtr algo_result)
{
    if (!template_img_path_2_.empty() && template_img_2_.empty()) {
        template_img_2_ = cv::imread(template_img_path_2_, 0);
    }
    cv::Mat template_img_th_img;
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

#pragma region TMEPLATE_PRE
    if (img_process_2_flag_) {
        cv::medianBlur(template_img_2_, template_img_th_img, 7);
        tool::sauvola(template_img_th_img, 0.1, 15);
        cv::erode(template_img_th_img, template_img_th_img, elementX);
        cv::bitwise_not(template_img_th_img, template_img_th_img);
        auto contours_ret = tool::get_contours(template_img_th_img);
        std::vector<std::vector<cv::Point>> contours_temp_img = contours_ret.value();
        cv::RotatedRect temp_rotate_rect;
        for (size_t i = 0; i < contours_temp_img.size(); ++i) {
            cv::RotatedRect rot_rect = cv::minAreaRect(contours_temp_img[i]);
            cv::Rect rect = cv::boundingRect(contours_temp_img[i]);
            int width = std::max(rot_rect.size.width, rot_rect.size.height);
            int heigh = std::min(rot_rect.size.width, rot_rect.size.height);
            if (width > MIN_TEMPLATE_WEIGHT && heigh > MIN_TEMPLATE_HEIGHT && width <= MAX_TEMPLATE_WEIGHT) {
                temp_rotate_rect = rot_rect;
                temp_rect_2_ = rect;
            }
        }
        template_img_org_2_ = template_img_2_.clone();
        auto rotate_ret = tool::get_rotated_rect_pt_trans_and_rotate(temp_rotate_rect, template_img_org_2_);
        temp_pts_2_ = rotate_ret.value();
        temp_pts_2_ = get_pts(template_img_org_2_, temp_pts_2_, img_th_);
        // The difference is greater than 5
        if (abs(temp_pts_2_[0].y - temp_pts_2_[1].y) > 5) {
            cv::Point2f src_pts[4], dst_pts[4];
            cv::Mat m, inv_m, rotate_img;
            for (int i = 0; i < temp_pts_2_.size(); i++) {
                src_pts[i] = temp_pts_2_[i];
            }
            double w = std::sqrtf(std::powf((temp_pts_2_[0].x - temp_pts_2_[1].x), 2) + std::powf((temp_pts_2_[0].y - temp_pts_2_[1].y), 2));
            double h = std::sqrtf(std::powf((temp_pts_2_[2].x - temp_pts_2_[1].x), 2) + std::powf((temp_pts_2_[2].y - temp_pts_2_[1].y), 2));
            dst_pts[0] = temp_pts_2_[0];
            dst_pts[1] = cv::Point2d(temp_pts_2_[0].x + w, temp_pts_2_[0].y);
            dst_pts[2] = cv::Point2d(temp_pts_2_[0].x + w, temp_pts_2_[0].y + h);
            dst_pts[3] = cv::Point2d(temp_pts_2_[0].x, temp_pts_2_[0].y + h);
            m = cv::getPerspectiveTransform(src_pts, dst_pts);
            cv::warpPerspective(template_img_org_2_, template_img_org_2_, m, template_img_org_2_.size());
            temp_pts_2_.clear();
            for (int i = 0; i < 4; i++) {
                temp_pts_2_.push_back(dst_pts[i]);
            }
        }
        img_process_2_flag_++;
    }
#pragma endregion
    // mask
    cv::Mat temp_mask = cv::Mat::zeros(template_img_org_2_.size(), template_img_org_2_.type());
    for (int i = 0; i < temp_mask_2_.size(); i++) {
        cv::Vec4i cur = temp_mask_2_[i];
        cv::Rect cur_rect(cur[0] + temp_pts_2_[0].x, cur[1] + temp_pts_2_[0].y, cur[2], cur[3]);
        temp_mask(cur_rect) = 255;
    }
    // process the original image
    cv::Mat src_img;
    cv::RotatedRect src_rotate_rect;
    cv::Rect src_rect;
    src_img = src.clone();
    tool::sauvola(src_img, 0.1, 15);
    cv::erode(src_img, src_img, elementX);
    cv::bitwise_not(src_img, src_img);
    auto contours_src_ret = tool::get_contours(src_img);
    std::vector<std::vector<cv::Point>> contours_src_img = contours_src_ret.value();
    for (size_t i = 0; i < contours_src_img.size(); ++i) {
        cv::RotatedRect rot_rect = cv::minAreaRect(contours_src_img[i]);
        cv::Rect rect = cv::boundingRect(contours_src_img[i]);
        int width = std::max(rot_rect.size.width, rot_rect.size.height);
        int heigh = std::min(rot_rect.size.width, rot_rect.size.height);
        if (width > MIN_TEMPLATE_WEIGHT && heigh > MIN_TEMPLATE_HEIGHT && width <= MAX_TEMPLATE_WEIGHT) {
            src_rotate_rect = rot_rect;
            src_rect = rect;
        }
    }
    int width = std::max(src_rotate_rect.size.width, src_rotate_rect.size.height);
    int heigh = std::min(src_rotate_rect.size.width, src_rotate_rect.size.height);
    if (width <= MIN_TEMPLATE_WEIGHT || heigh > MAX_TEMPLATE_HEIGHT || width >= MAX_TEMPLATE_WEIGHT) {
        algo_result->judge_result = 0;
        return;
    }
    auto src_rotate_pts_op = tool::get_rotated_rect_pts(src_rotate_rect);
    std::vector<cv::Point2f> src_rotate_pts = src_rotate_pts_op.value();
    cv::Mat dis_img;
    if (IsDebug()) {
        cv::cvtColor(src, dis_img, cv::COLOR_GRAY2BGR);
        for (size_t j = 0; j < 4; j++) {
            cv::line(dis_img, src_rotate_pts[j], src_rotate_pts[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
    src_rotate_pts = get_pts(src, src_rotate_pts, img_th_);
    // second find
    if (src_rotate_pts.size() < 3) {
        cv::threshold(src_img, src_img, 75, 255, cv::THRESH_TOZERO);
        contours_src_ret = tool::get_contours(src_img);
        contours_src_img = contours_src_ret.value();
        for (size_t i = 0; i < contours_src_img.size(); ++i) {
            cv::RotatedRect rot_rect = cv::minAreaRect(contours_src_img[i]);
            cv::Rect rect = cv::boundingRect(contours_src_img[i]);
            int width = std::max(rot_rect.size.width, rot_rect.size.height);
            int heigh = std::min(rot_rect.size.width, rot_rect.size.height);
            if (width > MIN_TEMPLATE_WEIGHT && heigh > MIN_TEMPLATE_HEIGHT && width <= MAX_TEMPLATE_WEIGHT) {
                src_rotate_rect = rot_rect;
                src_rect = rect;
            }
        }
        src_rotate_pts_op = tool::get_rotated_rect_pts(src_rotate_rect);
        src_rotate_pts = src_rotate_pts_op.value();
        src_rotate_pts = get_pts(src, src_rotate_pts, img_th_);
    }
    std::vector<std::vector<cv::Point2f>> dis_pt_vec;
    if (src_rotate_pts.size() <= 0) {
        algo_result->judge_result = 0;
        TVALGO_FUNCTION_LOG("src rotate_pts size is 0")
        write_debug_img(src, dis_pt_vec, 1);
        return;
    }
    // histogram matching
    auto equal_ret = tool::get_equal_img_2(src, src_rotate_pts, template_img_org_2_, temp_pts_2_);
    cv::Mat equal_img = equal_ret.value();
    // perspective change and difference
    cv::Mat wrap_mat, src_warp_img, cdf, cdf_roi, csf_th_img, csf_copy;
    tool::perspective(src_rotate_pts, temp_pts_2_, wrap_mat);
    cv::warpPerspective(equal_img, src_warp_img, wrap_mat, equal_img.size());
    cv::absdiff(src_warp_img, template_img_org_2_, cdf);
    cdf.copyTo(cdf_roi, temp_mask);
    cv::threshold(cdf_roi, csf_th_img, img_th_2_, 255, cv::THRESH_BINARY);
    algo_result->judge_result = 1;
    csf_copy = csf_th_img.clone();
    // result determination
    for (int i = 0; i < temp_mask_2_.size(); i++) {
        cv::Vec4i cur = temp_mask_2_[i];
        cv::Rect cur_rect(cur[0] + temp_pts_2_[0].x, cur[1] + temp_pts_2_[0].y, cur[2], cur[3]);
        cv::Mat cur_img = csf_th_img(cur_rect);
        int no_zero = cv::countNonZero(cur_img);
        if (no_zero <= area_th_) {
            csf_copy(cur_rect) = 0;
        } else {
            algo_result->judge_result = 0;
            std::vector<cv::Point2f> cur_pts = wrap_point(cur_rect, wrap_mat, algo_result);
            dis_pt_vec.push_back(cur_pts);
        }
    }
    write_debug_img(src, dis_pt_vec);
}

std::vector<cv::Point2f> trayTowerDet::wrap_point(const cv::Rect& input_rect, const cv::Mat& wrap_mat, AlgoResultPtr algo_result)
{
    std::vector<cv::Point2f> wrap_pts;
    cv::Point2f p_tl, p_tr, p_bl, p_br;
    tool::point_inv_transform(cv::Point2f(input_rect.tl()), wrap_mat, p_tl);
    tool::point_inv_transform(cv::Point2f(input_rect.tl().x + input_rect.width, input_rect.tl().y), wrap_mat, p_tr);
    tool::point_inv_transform(cv::Point2f(input_rect.tl().x, input_rect.tl().y + input_rect.height), wrap_mat, p_bl);
    tool::point_inv_transform(cv::Point2f(input_rect.br()), wrap_mat, p_br);
    algo_result->result_info.push_back(
        {
            {"label", "trayTowerDet"},
            {"shapeType", "polygon"},
            {"points", {{p_tl.x, p_tl.y}, {p_tr.x, p_tr.y}, {p_br.x, p_br.y}, {p_bl.x, p_bl.y}}},
            {"result", {{"confidence", 0}, {"area", 0}}},
        });
    wrap_pts.push_back(p_tl);
    wrap_pts.push_back(p_tr);
    wrap_pts.push_back(p_br);
    wrap_pts.push_back(p_bl);
    return wrap_pts;
}

void trayTowerDet::write_debug_img(const cv::Mat& input_img, const std::vector<std::vector<cv::Point2f>>& input_pts, int unknown_flag)
{
    if (IsDebug()) {
        if (unknown_flag == 1) {
            std::string save_name = "item_tower_unknown_" + image_file_name_ + ".jpg";
            SaveDebugImage(input_img, save_name);
            SAVE_IMG_TOWER_INDEX++;
            return;
        }
        if (input_pts.size() > 0) {
            cv::Mat dis_img;
            cv::cvtColor(input_img, dis_img, cv::COLOR_GRAY2BGR);
            for (size_t m = 0; m < input_pts.size(); m++) {
                for (size_t j = 0; j < 4; j++) {
                    cv::line(dis_img, input_pts[m][j], input_pts[m][(j + 1) % 4], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                }
            }
            std::string save_name = "item_tower_ng_" + image_file_name_ + ".jpg";
            SaveDebugImage(dis_img, save_name);
            SAVE_IMG_TOWER_INDEX++;
        } else {
            std::string save_name = "item_tower_ok_" + image_file_name_ + ".jpg";
            SaveDebugImage(input_img, save_name);
            SAVE_IMG_TOWER_INDEX++;
        }
    }
}

cv::Point2f get_cross_pt_2(const cv::Mat& img, const cv::Rect& rect, int type)
{
    ED testED = ED(img, LSD_OPERATOR, 36, 8, 1, 8, 1.0, true);
    EDLines testEDLines = EDLines(testED);
    std::vector<LS> lines = testEDLines.getLines();
    if (lines.size() <= 1) {
        return cv::Point2f(0, 0);
    }
    std::vector<LS> h_ls, v_ls;
    for (int i = 0; i < lines.size(); i++) {
        LS c_l = lines[i];
        double dy = fabs(c_l.start.y - c_l.end.y);
        double dis = cv::norm(c_l.start - c_l.end);
        double diff_y = c_l.start.y - c_l.end.y;
        double diff_x = c_l.start.x - c_l.end.x;
        double k = 0;
        if (diff_x != 0) {
            k = diff_y / diff_x;
            k = fabs(k);
        } else {
            k = 20;
        }
        if (dis < 19)
            continue;
        if (k > 10) {
            v_ls.emplace_back(c_l);
        }

        if (k < 0.2) {
            h_ls.emplace_back(c_l);
        }
    }
    std::sort(v_ls.begin(), v_ls.end(), [&](const LS& lhs, const LS& rhs) {
        cv::Point2f l_c((lhs.start.x + lhs.end.x) / 2, (lhs.start.y + lhs.end.y) / 2);
        cv::Point2f r_c((rhs.start.x + rhs.end.x) / 2, (rhs.start.y + rhs.end.y) / 2);
        if (l_c.x < r_c.x) {
            return true;
        } else {
            return false;
        }
    });
    std::sort(h_ls.begin(), h_ls.end(), [&](const LS& lhs, const LS& rhs) {
        cv::Point2f l_c((lhs.start.x + lhs.end.x) / 2, (lhs.start.y + lhs.end.y) / 2);
        cv::Point2f r_c((rhs.start.x + rhs.end.x) / 2, (rhs.start.y + rhs.end.y) / 2);
        if (l_c.y < r_c.y) {
            return true;
        } else {
            return false;
        }
    });
    // calculated point
    cv::Point2f ret_pt(0, 0);
    if (h_ls.size() == 0 || v_ls.size() == 0) {
        return ret_pt;
    }
    std::function get_cross_pt = [&](const LS& l1, const LS& l2) -> cv::Point2f {
        float x1 = l1.start.x, y1 = l1.start.y, x2 = l1.end.x, y2 = l1.end.y, x3 = l2.start.x, y3 = l2.start.y, x4 = l2.end.x, y4 = l2.end.y;
        float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
        float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
        float px = x1 + t * (x2 - x1);
        float py = y1 + t * (y2 - y1);
        return cv::Point2f(px, py);
    };
    if (type == 0) {
        LS l1 = h_ls[h_ls.size() - 1];
        LS l2 = v_ls[v_ls.size() - 1];
        ret_pt = get_cross_pt(l1, l2);
    } else if (type == 1) {
        LS l1 = h_ls[h_ls.size() - 1];
        LS l2 = v_ls[0];
        ret_pt = get_cross_pt(l1, l2);
    } else if (type == 2) {
        LS l1 = h_ls[0];
        LS l2 = v_ls[0];
        ret_pt = get_cross_pt(l1, l2);
    } else if (type == 3) {
        LS l1 = h_ls[0];
        LS l2 = v_ls[v_ls.size() - 1];
        ret_pt = get_cross_pt(l1, l2);
    }
    ret_pt.x = ret_pt.x + rect.x;
    ret_pt.y = ret_pt.y + rect.y;
    return ret_pt;
}
