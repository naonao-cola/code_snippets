#include "Z_Male_Detect.h"
#include "../../modules/tv_algo_base/src/framework/InferenceEngine.h"
#include "../../modules/tv_algo_base/src/utils/Utils.h"
#include "../../modules/tv_algo_base/src/utils/logger.h"
#include "JsonHelper.h"
#include "algo_tool.h"
#include "param_check.h"
#include "spinlock.h"
#include <execution>
#include <opencv2/core.hpp>
#include <opencv2/flann/flann.hpp>
#include <windows.h>

REGISTER_ALGO(Z_Male_Detect)

Z_Male_Detect::Z_Male_Detect() { }

Z_Male_Detect::~Z_Male_Detect() { }

AlgoResultPtr Z_Male_Detect::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    TVALGO_FUNCTION_BEGIN

    algo_result->result_info.push_back(
        {
            {"label","Z_Male_Detect"},
            {"shapeType","default"},
            {"points",{{0,0},{0,0}}},
            {"result",{{"confidence",0},{"area",0}}},
        }
    );
    try {
        get_param(task, pre_results);
    } catch (const std::exception& e) {
        TVALGO_FUNCTION_RETURN_ERROR_PARAM(e.what())
    }

    cv::Mat task_img = task->image.clone();
    cv::Mat dst, gray_mask;
    if (task_img.channels() > 1)
        cv::cvtColor(task_img, dst, cv::COLOR_BGR2GRAY);
    else {
        dst = task_img.clone();
    }

    // 算法部分
    cv::Rect img_rect(detect_left_x_, detect_left_y_, detect_width_, detect_height_);
    double angle1, angle2;

    cv::Mat mask_img = cv::Mat::zeros(dst.size(),dst.type());
    cv::Mat tmp_mask = mask_img(img_rect);

    dst(img_rect).copyTo(tmp_mask);

    std::vector<cv::Point2f> pt_vec_1 = img_process_1(task_img, mask_img,algo_result, angle1);
    std::vector<cv::Point2f> pt_vec_2 = img_process_2(task_img, mask_img, algo_result, angle2);
    if(pt_vec_1.size()<5 || pt_vec_2.size()<5 ){
        algo_result->judge_result = 0;
        TVALGO_FUNCTION_END
    }
    cal_data(task_img, dst, pt_vec_1, pt_vec_2, angle1, algo_result);

    // 绘图
    cv::Mat dis = task_img.clone();
    connector::draw_results(dis, algo_result->result_info);
    TVALGO_FUNCTION_END
}

std::tuple<std::string, json> Z_Male_Detect::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map) const
{
    std::string task_type_id = task->image_info["type_id"];
    json task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

bool Z_Male_Detect::get_param(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    std::tuple<std::string, json> details_info = get_task_info(task, pre_results, m_param_map);
    json task_param_json = std::get<1>(details_info);
    pix_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "pix_value", 5.26);
    z_male_x_num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "z_male_x_num", 6);
    z_male_y_num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "z_male_y_num", 8);

    gamma_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gamma_value", 0.8);
    area_range_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "area_range_b", 1.1);
    rect_height_t_ = Tival::JsonHelper::GetParam(task_param_json["param"], "rect_height_t", 35);
    rect_height_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "rect_height_b", 0.4);
    rect_width_t_ = Tival::JsonHelper::GetParam(task_param_json["param"], "rect_width_t", 35);
    rect_width_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "rect_width_b", 0.4);
    rect_rate_ = Tival::JsonHelper::GetParam(task_param_json["param"], "rect_rate", 0.4);
    threshold_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "threshold_value", 180);

    error_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_value", 0.12);

    s_threshold_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_threshold_value", 200);
    s_gamma_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_gamma_value_", 0.8);
    s_area_range_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_area_range_b", 100);
    s_rect_height_t_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_rect_height_t", 20);
    s_rect_height_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_rect_height_b", 7);
    s_rect_width_t_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_rect_width_t", 25);
    s_rect_width_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_rect_width_b", 10);
    s_rect_rate_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_rect_rate", 3.2);

    s_angle_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_angle", 85.0);
    s_angle_max_ = Tival::JsonHelper::GetParam(task_param_json["param"], "s_angle_max", 5.0);
    s_no_zero_ = Tival::JsonHelper::GetParam(task_param_json["param"], "no_zero",0.4);
    // mode = Tival::JsonHelper::GetParam(task_param_json["param"], "clockwise_rotation_90", 90);

    detect_left_x_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_x", 600);
    detect_left_y_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_y", 1739);
    detect_width_ = Tival::JsonHelper::GetParam(task_param_json["param"],  "detect_width", 3648);
    detect_height_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_height", 1880);


    tempImagePoint_1_.clear();
    json x_coords_vec_1 = task_param_json["param"]["x_coords1"];
    json y_coords_vec_1 = task_param_json["param"]["y_coords1"];
    for (int i = 0; i < x_coords_vec_1.size(); i++) {
        auto item_x = x_coords_vec_1[i];
        auto item_y = y_coords_vec_1[i];
        tempImagePoint_1_.push_back(cv::Point2f(item_x, item_y));
    }
    // 构建黄针检测范围
    if (!tempImagePoint_1_.empty()) {
        cv::Mat ret = get_col_row(tempImagePoint_1_);
        template_mat_1_.release();
        if (z_male_x_num_ == 19) {
            template_mat_1_ = cv::Mat::zeros(z_male_y_num_, z_male_x_num_, CV_32FC(6));
            template_mat_1_ = ret(cv::Range(0, z_male_y_num_), cv::Range(0, z_male_x_num_)).clone();
        } else {
            template_mat_1_ = cv::Mat::zeros(z_male_y_num_, z_male_x_num_ * 2, CV_32FC(6));
            template_mat_1_ = ret(cv::Range(0, z_male_y_num_), cv::Range(0, z_male_x_num_ * 2)).clone();
        }
    }
    tempImagePoint_2_.clear();
    json x_coords_vec_2 = task_param_json["param"]["x_coords2"];
    json y_coords_vec_2 = task_param_json["param"]["y_coords2"];
    for (int i = 0; i < x_coords_vec_2.size(); i++) {
        auto item_x = x_coords_vec_2[i];
        auto item_y = y_coords_vec_2[i];
        tempImagePoint_2_.push_back(cv::Point2f(item_x, item_y));
    }

    // 构建黄铜片检测范围
    template_mat_2_.release();
    if (!tempImagePoint_2_.empty()) {
        cv::Mat ret = get_col_row(tempImagePoint_2_);
        template_mat_2_ = cv::Mat::zeros(z_male_y_num_, z_male_x_num_, CV_32FC(6));
        template_mat_2_ = ret(cv::Range(0, z_male_y_num_), cv::Range(0, z_male_x_num_)).clone();
    }

    miss_1_.clear();
    json x_empty_vec_1 = task_param_json["param"]["x_empty1"];
    json y_empty_vec_1 = task_param_json["param"]["y_empty1"];
    for (int i = 0; i < x_empty_vec_1.size(); i++) {
        auto item_x = x_empty_vec_1[i];
        auto item_y = y_empty_vec_1[i];
        miss_1_.push_back(cv::Point2f(item_x, item_y));
    }

    miss_2_.clear();
    json x_empty_vec_2 = task_param_json["param"]["x_empty2"];
    json y_empty_vec_2 = task_param_json["param"]["y_empty2"];
    for (int i = 0; i < x_empty_vec_2.size(); i++) {
        auto item_x = x_empty_vec_2[i];
        auto item_y = y_empty_vec_2[i];
        miss_2_.push_back(cv::Point2f(item_x, item_y));
    }

    features_1_ = cv::Mat(tempImagePoint_1_).reshape(1, tempImagePoint_1_.size());
    features_1_.convertTo(features_1_, CV_32FC1);
    features_2_ = cv::Mat(tempImagePoint_2_).reshape(1, tempImagePoint_2_.size());
    features_2_.convertTo(features_2_, CV_32FC1);
    return true;
}

cv::Mat Z_Male_Detect::get_col_row(std::vector<cv::Point2f> basis_info_vec)
{
    std::vector<std::vector<cv::Point2f>> rank;
    std::vector<cv::Point2f> swap_vec;
    for (int i = 0; i < basis_info_vec.size() - 1; i++) {
        cv::Point2f cur_pt = basis_info_vec[i];
        cv::Point2f next_pt = basis_info_vec[i + 1];
        if (std::abs(cur_pt.y - next_pt.y) > 1.4) {
            // 不是同一行则前一个是上一行，后一个是下一行
            swap_vec.push_back(cur_pt);
            std::vector<cv::Point2f> tmp_vec = swap_vec;
            // 将上一行放进去，另起新行
            rank.push_back(tmp_vec);
            swap_vec.clear();
        } else {
            // 是同一行
            swap_vec.push_back(cur_pt);
            if (i == basis_info_vec.size() - 2) {
                // 最后一行，最后一个，收尾
                swap_vec.push_back(next_pt);
                std::vector<cv::Point2f> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
        }
    }
    cv::Mat ret = cv::Mat::zeros(rank.size(), rank[0].size(), CV_32FC(6));
    for (int i = 0; i < rank.size(); i++) {
        for (int j = 0; j < rank[0].size(); j++) {
            cv::Vec6f& pixel = ret.at<cv::Vec6f>(i, j);
            pixel[0] = rank[i][j].x;
            pixel[1] = rank[i][j].y;
        }
    }
    return ret;
}
// 获取小铜片
std::vector<cv::Point2f> Z_Male_Detect::img_process_1(const cv::Mat& src, const cv::Mat& gray_img,AlgoResultPtr algo_result, double& angle) noexcept
{
    cv::Mat th_img;
    cv::Mat gamma_img = connector::gamma_trans(gray_img, s_gamma_value_);
    // 阈值处理
    int thre_value = s_threshold_value_;

    cv::threshold(gamma_img, th_img, thre_value, 255, cv::THRESH_BINARY);
    // 形态学处理
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
    ; // X方向腐蚀
    cv::dilate(th_img, th_img, elementX);
    cv::erode(th_img, th_img, elementX);
    //
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img);
    // mask
    cv::Mat gray_mask = cv::Mat::zeros(src.size(), CV_8UC1);

    std::vector<double> area_rate_vec;
    // 取初值mask
    for (size_t i = 0; i < filter_contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        // double area_rate = (area / ());
        if (area < s_area_range_b_ || rect.height > rect.width)
            continue;

        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (r_rect.angle <= s_angle_ && r_rect.angle >= s_angle_max_) {
            continue;
        }
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        double rate = width / (height * 1.0);
        if (height <= s_rect_height_t_ && height >= s_rect_height_b_ && width >= s_rect_width_b_ && width < s_rect_width_t_ && rate >= s_rect_rate_) {
            cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        }
    }

    // 求重心
    std::vector<std::vector<cv::Point>> center_contours = connector::get_contours(gray_mask);
    std::vector<cv::Point2f> total_center;

    angle = 0;
    double angle_count = 0;
    std::vector<double> angle_vec;

    // #pragma omp parallel for num_threads(12)
    for (int i = 0; i < center_contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(center_contours[i]);
        cv::RotatedRect rot_rect = cv::minAreaRect(center_contours[i]);
        cv::Point2f box_pts[4];
        rot_rect.points(box_pts);

        std::vector<cv::Point2f> pt_vec;
        pt_vec.emplace_back(box_pts[0]);
        pt_vec.emplace_back(box_pts[1]);
        pt_vec.emplace_back(box_pts[2]);
        pt_vec.emplace_back(box_pts[3]);
        std::vector<cv::Point2f> order_pt_vec = connector::order_pts(pt_vec);

        double k = (order_pt_vec[1].y - order_pt_vec[0].y) / (order_pt_vec[1].x - order_pt_vec[0].x);
        angle_vec.push_back(atanl(k) * 180.0 / CV_PI);

        angle = angle + atanl(k) * 180.0 / CV_PI;
        angle_count++;

        cv::Rect af_rect;
        af_rect.x = rect.x - 5;
        af_rect.y = rect.y - 5;
        af_rect.width = rect.width + 10;
        af_rect.height = rect.height + 10;
        cv::Mat sub_img = gray_img(af_rect).clone();

        std::vector<cv::Point2f> dst_pt;
        // 均值在70到110 之间
        int mean_value = cv::mean(sub_img)[0];
        // 以均值为基础，向上迭代5次
        int end_value = min(mean_value + 50, 254);
        std::map<int, cv::Point2f> center_info;
        for (int cur_th = mean_value; cur_th <= end_value; cur_th = cur_th + 10) {
            cv::Mat thre_img;
            cv::threshold(sub_img, thre_img, cur_th, 255, cv::THRESH_BINARY);
            cv::Point2f center;
            connector::get_center(thre_img, center);
            // 此时的center是相对于裁剪之后的，需要还原
            center.x = af_rect.x + center.x;
            center.y = af_rect.y + center.y;
            if (std::isnan(center.y) || std::isnan(center.x))
                continue;
            center_info.insert(std::pair<int, cv::Point2f>(cur_th, center));
        }
        double sumX = 0, sumY = 0;
        double total_w = 0;
        for (auto item : center_info) {
            if (std::isnan(item.second.y) || std::isnan(item.second.x))
                continue;
            int w = 255 - item.first; // 权重
            total_w = total_w + w;
            sumX = sumX + w * item.second.x;
            sumY = sumY + w * item.second.y;
        }
        cv::Point2f ret_center;
        ret_center.x = sumX / total_w;
        ret_center.y = sumY / total_w;
        total_center.emplace_back(ret_center);
        if (!std::isnan(ret_center.x)) {
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "polygon" },
                    { "points", { { box_pts[0].x, box_pts[0].y }, { box_pts[1].x, box_pts[1].y }, { box_pts[2].x, box_pts[2].y }, { box_pts[3].x, box_pts[3].y } } },
                    { "result", 1 } });
        }
    }
    if (angle_count > 3) {
        angle = angle / angle_count;
    }
    return total_center;
}

std::vector<cv::Point2f> Z_Male_Detect::img_process_2(const cv::Mat& src, const cv::Mat& gray_img, AlgoResultPtr algo_result, double& angle) noexcept
{

    cv::Mat thre_img;
    // 阈值处理
    int thre_value = threshold_value_;
    cv::Mat gamma_img = connector::gamma_trans(gray_img, gamma_value_);

    cv::threshold(gamma_img, thre_img, thre_value, 255, cv::THRESH_BINARY);
    // X方向腐蚀
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::dilate(thre_img, thre_img, elementX);
    // cv::erode(thre_img, thre_img, elementX);

    std::vector<std::vector<cv::Point>> contours = connector::get_contours(thre_img);

    // 初步过滤
    cv::Mat gray_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        double area = cv::contourArea(contours[i]);
        if (area < area_range_b_)
            continue;
        cv::RotatedRect r_rect = cv::minAreaRect(contours[i]);
        std::vector<std::vector<cv::Point>> draw_conts = { contours[i] };
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        double rate = width / (height * 1.0);
        if (height <= rect_height_t_ && height >= rect_height_b_ && width >= rect_width_b_ && width <= rect_width_t_ && rate >= rect_rate_) {
            cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        }
    }

    std::vector<cv::Point2f> center_pt_info;
    std::vector<std::vector<cv::Point>> line_pt = connector::get_contours(gray_mask);
    // #pragma omp parallel for num_threads(12)
    angle = 0;
    double angle_count = 0;
    std::vector<double> angle_vec;
    for (int j = 0; j < line_pt.size(); j++) {
        cv::Rect rect = cv::boundingRect(line_pt[j]);
        cv::RotatedRect r_rect = cv::minAreaRect(line_pt[j]);
        cv::Point2f box_pts[4];
        r_rect.points(box_pts);
        algo_result->result_info.push_back(
            { { "label", "fuzhu" },
                { "shapeType", "polygon" },
                { "points", { { box_pts[0].x, box_pts[0].y }, { box_pts[1].x, box_pts[1].y }, { box_pts[2].x, box_pts[2].y }, { box_pts[3].x, box_pts[3].y } } },
                { "result", 1 } });

        cv::Moments M = cv::moments(line_pt[j], false);
        cv::Point center(M.m10 / M.m00, M.m01 / M.m00);

        std::vector<cv::Point2f> pt_vec;
        pt_vec.emplace_back(box_pts[0]);
        pt_vec.emplace_back(box_pts[1]);
        pt_vec.emplace_back(box_pts[2]);
        pt_vec.emplace_back(box_pts[3]);
        std::vector<cv::Point2f> order_pt_vec = connector::order_pts(pt_vec);

        double k = (order_pt_vec[1].y - order_pt_vec[0].y) / (order_pt_vec[1].x - order_pt_vec[0].x);
        angle_vec.push_back(atanl(k) * 180.0 / CV_PI);

        angle = angle + atanl(k) * 180.0 / CV_PI;
        angle_count++;

        // angle_vec.emplace_back(angle);
        // angle = angle + r_rect.angle -90;
        // angle_count++;

        double center_x = M.m10 / M.m00;
        // 左右个去除 1/14，保留上部分整体
        int width = rect.width;
        int step_width = width / 14;

        cv::Rect af_rect;
        af_rect.x = rect.x + step_width;
        af_rect.width = rect.width - step_width * 2;
        af_rect.y = rect.y - 15;
        af_rect.height = rect.height;

        cv::Mat sub_img = gray_img(af_rect).clone();

        int mean_value = cv::mean(sub_img)[0];
        int start_value = mean_value - 20 > 160 ? mean_value + 20 : 160;
        int end_value = mean_value + 20 < 240 ? 220 : mean_value + 20;
        std::map<int, double> center_info;
        int step_th = 20;
        for (int cur_th = start_value; cur_th <= end_value; cur_th = cur_th + step_th) {
            cv::Mat sub_thre_img;
            cv::threshold(sub_img, sub_thre_img, cur_th, 255, cv::THRESH_BINARY);
            // 排除掉干扰
            std::vector<std::vector<cv::Point>> tmp_pt = connector::get_contours(sub_thre_img);
            for (int m = 0; m < tmp_pt.size(); m++) {
                cv::Rect rect = cv::boundingRect(tmp_pt[m]);
                std::vector<std::vector<cv::Point>> draw_conts = { tmp_pt[m] };
                if (rect.width < sub_thre_img.cols - 20) {
                    cv::drawContours(sub_thre_img, draw_conts, 0, 0, -1);
                }
            }
            std::vector<cv::Point2f> dst_pt;
            connector::StegerLine(sub_thre_img, dst_pt);
            double sumY = 0;
            for (int m = 0; m < dst_pt.size(); m++) {
                sumY = sumY + dst_pt[m].y;
            }
            sumY = sumY / dst_pt.size();
            // 此时的center是相对于裁剪之后的，需要还原
            sumY = sumY + af_rect.y;
            center_info.insert(std::pair<int, double>(cur_th, sumY));
        }
        double sumY = 0;
        double total_w = 0;
        // 权重考虑是否加指数
        for (auto item : center_info) {
          if (std::isnan(item.second)) {
            continue;
          }
            int w = 255 - item.first; // 权重
            total_w = total_w + w;
            sumY = sumY + w * item.second;
        }
        cv::Point2f ret_center;
        ret_center.x = center_x;
        ret_center.y = sumY / total_w;
        center_pt_info.emplace_back(ret_center);
    }
    if (angle_count > 3) {
        angle = angle / angle_count;
    }
    return center_pt_info;
}

#include <limits>
float calculateSlope(const cv::Point2f& p1, const cv::Point2f& p2)
{
    if (p1.x == p2.x) {
        return std::numeric_limits<float>::infinity();
    }
    return (p2.y - p1.y) / (p2.x - p1.x);
}

double reget_angle(std::vector<cv::Point2f> pts, int min_points_in_line)
{
    std::map<float, std::vector<std::pair<int, int>>> line_groups;
    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = i + 1; j < pts.size(); ++j) {
            float slope = calculateSlope(pts[i], pts[j]);
            float slope_key = roundf(slope * 100) / 100;
            line_groups[slope_key].push_back(std::make_pair(i, j));
        }
    }
    int max_value = 0;
    double angle = 0;
    for (const auto& pair : line_groups) {
        if (pair.second.size() >= min_points_in_line) {
            if (pair.second.size() >= max_value) {
                max_value = pair.second.size();
                angle = pair.first;
            }
        }
    }
    angle = atanl(angle) * 180.0 / CV_PI;
    return angle;
}

cv::Mat Z_Male_Detect::fit_line(const cv::Mat& src, const cv::Mat& gray_img, std::vector<cv::Point2f> pts, double angle)
{

    if (z_male_x_num_ > 16) {

        angle = reget_angle(pts, 19);
    }

    // 左上角的点
    cv::Point2f lt = { -1, -1 };
    for (auto pt : pts) {
        if (lt.x == -1 || (pt.x + pt.y) < (lt.x + lt.y)) {
            lt.x = pt.x;
            lt.y = pt.y;
        }
    }
    cv::Mat tmpM = connector::vector_angle_to_M(lt.x, lt.y, 0, 0, 0, angle);
    std::vector<std::vector<cv::Point2f>> pt_vec(z_male_y_num_);
    double row_height_mm = (std::max)(double(tempImagePoint_1_[z_male_x_num_ * 2].y) - double(tempImagePoint_1_[0].y), 1.9);
    if (z_male_x_num_ >= 16) {
        row_height_mm = (std::max)(double(tempImagePoint_1_[z_male_x_num_].y) - double(tempImagePoint_1_[0].y), 1.9);
    }
    double row_height_px = row_height_mm * 1000 / pix_value_;
    ;

    std::vector<cv::Point2f> tmp_pt_vec;
    for (int i = 0; i < pts.size(); i++) {
        cv::Point2f tmp_pt = connector::TransPoint(tmpM, pts[i]);
        tmp_pt_vec.emplace_back(tmp_pt);
        int y_to_lt = std::abs(tmp_pt.y);
        int row_idx = int(y_to_lt / row_height_px + 0.5);
        double diff = std::abs(row_idx * row_height_px - y_to_lt);
        if (diff < 100 && row_idx < z_male_y_num_) {
            pt_vec[row_idx].push_back(pts[i]);
        }
    }
    int count = 0;
    double angel_sum = 0;
    for (int i = 0; i < pt_vec.size(); i++) {
        if (pt_vec[i].size() <= 0)
            continue;
        cv::Vec4f lineParams;
        cv::fitLine(pt_vec[i], lineParams, cv::DIST_HUBER, 0, 0.01, 0.01);
        double angleRad = std::atan(lineParams[1] / lineParams[0]);
        angel_sum += angleRad;
        count += 1;
    }
    angle = (angel_sum / count) * 180.0 / CV_PI;
    // 变换矩阵
    cv::Mat M = connector::vector_angle_to_M(lt.x, lt.y, 0.0, 0.0, 0.0, angle);
    return M;
}

int Z_Male_Detect::get_nearest_point_idx(cv::Mat points, cv::Point2f refPoint, double& minDist)
{
    cv::Mat query = (cv::Mat_<float>(1, 2) << refPoint.x, refPoint.y);
    cv::flann::Index flannIndex(points, cv::flann::KDTreeIndexParams(2), cvflann::FLANN_DIST_L2);
    cv::Mat indices, dists;
    flannIndex.knnSearch(query, indices, dists, 2, cv::flann::SearchParams(32));
    minDist = std::sqrt(dists.at<float>(0, 0));
    /*if (minDist > error_value_ *2) {
    return -1;
    }*/
    if (minDist < 0) {
        minDist = 0;
        return -1;
    }
    if (indices.at<int>(0, 0)<0 || indices.at<int>(0, 0)> points.rows * points.cols) {
        minDist = 0;
        return -1;
    }
    return indices.at<int>(0, 0);
}

bool Z_Male_Detect::compare(const point_info& lhs, point_info& rhs)
{
    if (std::abs(lhs.tran_pt.y - rhs.tran_pt.y) < 50) {
        if (lhs.tran_pt.x < rhs.tran_pt.x) {
            return true;
        } else {
            return false;
        }
    } else {
        if (lhs.tran_pt.y < rhs.tran_pt.y) {
            return true;
        } else {
            return false;
        }
    }
}
void Z_Male_Detect::cal_data(const cv::Mat& src, const cv::Mat& gray_img, std::vector<cv::Point2f> pt_vec_1, std::vector<cv::Point2f> pt_vec_2, double angle, AlgoResultPtr algo_result) noexcept
{
    if (pt_vec_1.size() < 20) {
        // 找错的情况下直接返回
        algo_result->judge_result = 0;
        return;
    }

    status_flag = true;
    // 拟合直线求矩阵
    cv::Mat trans_M = fit_line(src, gray_img, pt_vec_1, angle);
    // 结果信息
    std::vector<point_info> pt_info_vec_1;
    std::vector<point_info> pt_info_vec_2;
    // 铜针
    for (int i = 0; i < pt_vec_1.size(); i++) {
        if (std::isnan(pt_vec_1[i].x) || std::isnan(pt_vec_1[i].y))
            continue;
        cv::Point2f trans_pt = connector::TransPoint(trans_M, pt_vec_1[i]);
        point_info pi;
        pi.img_pt = pt_vec_1[i];
        pi.tran_pt = trans_pt;
        pt_info_vec_1.push_back(pi);
    }
    // 黄铜片
    for (int i = 0; i < pt_vec_2.size(); i++) {
        if (std::isnan(pt_vec_2[i].x) || std::isnan(pt_vec_2[i].y))
            continue;
        cv::Point2f trans_pt = connector::TransPoint(trans_M, pt_vec_2[i]);
        point_info pi;
        pi.img_pt = pt_vec_2[i];
        pi.tran_pt = trans_pt;
        pt_info_vec_2.push_back(pi);
    }
    // 排序，计算误差
    std::sort(std::execution::par, pt_info_vec_1.begin(), pt_info_vec_1.end(), Z_Male_Detect::compare);
    std::sort(std::execution::par, pt_info_vec_2.begin(), pt_info_vec_2.end(), Z_Male_Detect::compare);


    // 计算黄针的位置偏差
    double diff_x_1 = 0;
    double diff_y_1 = 0;
    double found_cnt_1 = 0;

    for (int i = 0; i < pt_info_vec_1.size(); i++) {
        point_info pi = pt_info_vec_1[i];
        double cur_x = pt_info_vec_1[i].tran_pt.x;
        double cur_y = pt_info_vec_1[i].tran_pt.y;
        cur_x = cur_x * pix_value_ / 1000.0;
        cur_y = cur_y * pix_value_ / 1000.0;
        pt_info_vec_1[i].cal_pt.x = cur_x;
        pt_info_vec_1[i].cal_pt.y = cur_y;
        double min_dist;

        int idx = get_nearest_point_idx(features_1_, pt_info_vec_1[i].cal_pt, min_dist);
        int x_num = 0;
        int y_num = 0;
        if (z_male_x_num_ == 19) {
            x_num = idx / z_male_x_num_;
            y_num = idx % z_male_x_num_;
        } else {
            x_num = idx / (z_male_x_num_ * 2);
            y_num = idx % (z_male_x_num_ * 2);
        }

        cv::Vec6f& pixel = template_mat_1_.at<cv::Vec6f>(x_num, y_num);

        if (min_dist > error_value_ * 3) {
            if (pixel[3] != 1)
                pixel[3] = 0;
            pt_info_vec_1[i].index = -1;
            continue;
        }
        pixel[3] = 1;
        pt_info_vec_1[i].offset_x = (cur_x - tempImagePoint_1_[idx].x);
        pt_info_vec_1[i].offset_y = (cur_y - tempImagePoint_1_[idx].y);
        pt_info_vec_1[i].org_pt = tempImagePoint_1_[idx];
        pt_info_vec_1[i].index = idx;
        pt_info_vec_1[i].is_ok = std::abs(pt_info_vec_1[i].offset_x) > error_value_ ? false : true;
        pt_info_vec_1[i].is_ok = pt_info_vec_1[i].is_ok && (pt_info_vec_1[i].offset_y > error_value_ ? false : true);
        /*if (!pt_info_vec_1[i].is_ok) {
                status_flag = false;
        }*/
        if (std::abs(pt_info_vec_1[i].offset_x) <= error_value_ && std::abs(pt_info_vec_1[i].offset_y) <= error_value_) {
            diff_x_1 = diff_x_1 + (cur_x - tempImagePoint_1_[idx].x);
            diff_y_1 = diff_y_1 + (cur_y - tempImagePoint_1_[idx].y);
            found_cnt_1++;
        }
    }
    diff_x_1 = diff_x_1 / found_cnt_1;
    diff_y_1 = diff_y_1 / found_cnt_1;
    //黄铜针未找全
    if (pt_info_vec_1.size()< template_mat_1_.rows * template_mat_1_.cols) {
        for (int i = 0; i < template_mat_1_.rows; i++) {
            for (int j = 0; j < template_mat_1_.cols; j++) {
                cv::Vec6f& pixel = template_mat_1_.at<cv::Vec6f>(i, j);
                if (pixel[3] ==1) {
                    continue;
                }
                //未找到的反算重找
                cv::Point2f org_pt = cv::Point2f(pixel[0], pixel[1]);
                org_pt.x = (org_pt.x + diff_x_1) * 1000 / pix_value_;
                org_pt.y = (org_pt.y + diff_y_1) * 1000 / pix_value_;
                double error_pix = error_value_ * 1000 / pix_value_;
                org_pt = connector::TransPoint_inv(trans_M, org_pt);

                int idx = i * template_mat_1_.cols + j;
                point_info t_z = refind_point_1(gray_img, org_pt, idx, trans_M, error_pix);
                if (t_z.index != -1) {
                    pt_info_vec_1.emplace_back(t_z);
                    pixel[3] = 1;
                }
                else {
                    pixel[3] = 0;
                }

            }
        }
    }

    //排除多余的,金针数量多，则排除误差最大的,多出的个数不定
    if (pt_info_vec_1.size() > template_mat_1_.rows * template_mat_1_.cols) {
        int d_count = pt_info_vec_1.size() - template_mat_1_.rows * template_mat_1_.cols;
        std::vector<std::pair<int, double>> d_vec;
        for (int i = 0; i < pt_info_vec_1.size(); i++) {
            double min_dist;
            int idx = get_nearest_point_idx(features_1_, pt_info_vec_1[i].cal_pt, min_dist);
            if (idx<0 || idx>template_mat_1_.rows * template_mat_1_.cols || std::isnan(min_dist) || min_dist<0) {
                continue;
            }
            d_vec.push_back(std::make_pair(i,min_dist));
        }
        std::sort(d_vec.begin(), d_vec.end(), [](const std::pair<int, double>lhs, const std::pair<int, double>rhs) { return (lhs.second > rhs.second); });
        for (int j = 0; j < d_count;j++) {
            int idx = d_vec[j].first;
            pt_info_vec_1[idx].index = -1;
        }
    }

    // 计算黄铜片的位置偏差
    double diff_x_2 = 0;
    double diff_y_2 = 0;
    double found_cnt_2 = 0;

    for (int i = 0; i < pt_info_vec_2.size(); i++) {

        point_info pi = pt_info_vec_2[i];
        double cur_x = pt_info_vec_2[i].tran_pt.x;
        double cur_y = pt_info_vec_2[i].tran_pt.y;
        if (std::isnan(cur_x) || std::isnan(cur_y))
            continue;
        cur_x = cur_x * pix_value_ / 1000.0;
        cur_y = cur_y * pix_value_ / 1000.0;
        pt_info_vec_2[i].cal_pt.x = cur_x;
        pt_info_vec_2[i].cal_pt.y = cur_y;
        double min_dist;
        int idx = get_nearest_point_idx(features_2_, pt_info_vec_2[i].cal_pt, min_dist);

        int x_num = 0;
        int y_num = 0;
        x_num = idx / z_male_x_num_;
        y_num = idx % z_male_x_num_;

        cv::Vec6f& pixel = template_mat_2_.at<cv::Vec6f>(x_num, y_num);

        if (min_dist > error_value_ * 3) {
            pt_info_vec_2[i].index = -1;
            if (pixel[3] != 1)
                pixel[3] = 0;
            continue;
        }
        pixel[3] = 1;
        pt_info_vec_2[i].offset_x = (cur_x - tempImagePoint_2_[idx].x);
        pt_info_vec_2[i].offset_y = (cur_y - tempImagePoint_2_[idx].y);
        pt_info_vec_2[i].org_pt = tempImagePoint_2_[idx];
        pt_info_vec_2[i].index = idx;
        pt_info_vec_2[i].is_ok = std::abs(pt_info_vec_2[i].offset_x) > error_value_ ? false : true;
        pt_info_vec_2[i].is_ok = pt_info_vec_2[i].is_ok && (pt_info_vec_2[i].offset_y > error_value_ ? false : true);
        /*if (!pt_info_vec_2[i].is_ok) {
                status_flag = false;
        }*/
        if (std::abs(pt_info_vec_2[i].offset_x) <= error_value_ && std::abs(pt_info_vec_2[i].offset_y) <= error_value_) {
            diff_x_2 = diff_x_2 + (cur_x - tempImagePoint_2_[idx].x);
            diff_y_2 = diff_y_2 + (cur_y - tempImagePoint_2_[idx].y);
            found_cnt_2++;
        }
    }
    diff_x_2 = diff_x_2 / found_cnt_2;
    diff_y_2 = diff_y_2 / found_cnt_2;

    // 图纸范围：
    for (int i = 0; i < template_mat_1_.rows; i++) {
        for (int j = 0; j < template_mat_1_.cols; j++) {

            cv::Vec6f pixel = template_mat_1_.at<cv::Vec6f>(i, j);
            cv::Point2f org_pt = cv::Point2f(pixel[0], pixel[1]);
            org_pt.x = (org_pt.x + diff_x_1) * 1000 / pix_value_;
            org_pt.y = (org_pt.y + diff_y_1) * 1000 / pix_value_;
            double error_pix = error_value_ * 1000 / pix_value_;
            org_pt = connector::TransPoint_inv(trans_M, org_pt);
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "polygon" },
                    { "points", { { org_pt.x - error_pix, org_pt.y - error_pix }, { org_pt.x + error_pix, org_pt.y - error_pix }, { org_pt.x + error_pix, org_pt.y + error_pix }, { org_pt.x - error_pix, org_pt.y + error_pix } } },
                    { "result", pixel[3] } });
            if (pixel[3] == 0) {
                algo_result->result_info.push_back(
                    { { "label", "J" },
                        { "shapeType", "point" },
                        { "points", { { org_pt.x, org_pt.y } } },
                        { "result", { { "is_ok", false }, { "x_off", 0 }, { "y_off", 0 }, { "std_x", org_pt.x }, { "std_y", org_pt.y }, { "measured_x", 0 }, { "measured_y", 0 }, { "TP", 0 }, { "index", i * template_mat_2_.rows + j } } } });
            }
        }
    }
    for (int i = 0; i < template_mat_2_.rows; i++) {
        for (int j = 0; j < template_mat_2_.cols; j++) {
            cv::Vec6f pixel = template_mat_2_.at<cv::Vec6f>(i, j);
            cv::Point2f org_pt = cv::Point2f(pixel[0], pixel[1]);
            org_pt.x = (org_pt.x + diff_x_2) * 1000 / pix_value_;
            org_pt.y = (org_pt.y + diff_y_2) * 1000 / pix_value_;
            double error_pix = error_value_ * 1000 / pix_value_;
            org_pt = connector::TransPoint_inv(trans_M, org_pt);
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "polygon" },
                    { "points", { { org_pt.x - error_pix, org_pt.y - error_pix }, { org_pt.x + error_pix, org_pt.y - error_pix }, { org_pt.x + error_pix, org_pt.y + error_pix }, { org_pt.x - error_pix, org_pt.y + error_pix } } },
                    { "result", pixel[3] } });

            if (pixel[3] == 0) {
                algo_result->result_info.push_back(
                    { { "label", "T" },
                        { "shapeType", "point" },
                        { "points", { { org_pt.x, org_pt.y } } },
                        { "result", { { "is_ok", false }, { "x_off", 0 }, { "y_off", 0 }, { "std_x", org_pt.x }, { "std_y", org_pt.y }, { "measured_x", 0 }, { "measured_y", 0 }, { "TP", 0 }, { "index", i * template_mat_2_.rows + j } } } });
            }
        }
    }


    std::unordered_map<int, int> pin_map;
    // 误差校正
    for (int i = 0; i < pt_info_vec_1.size(); i++) {
        if (pt_info_vec_1[i].index < 0)
            continue;
        pt_info_vec_1[i].offset_x = pt_info_vec_1[i].offset_x - diff_x_1;
        pt_info_vec_1[i].offset_y = pt_info_vec_1[i].offset_y - diff_y_1;
        pt_info_vec_1[i].cal_pt.x = pt_info_vec_1[i].cal_pt.x - diff_x_1;
        pt_info_vec_1[i].cal_pt.y = pt_info_vec_1[i].cal_pt.y - diff_y_1;
        pt_info_vec_1[i].is_ok = std::abs(pt_info_vec_1[i].offset_x) > error_value_ ? false : true;
        pt_info_vec_1[i].is_ok = pt_info_vec_1[i].is_ok && (std::abs(pt_info_vec_1[i].offset_y) > error_value_ ? false : true);
        //查询是否有重复的
        if (pin_map.count(pt_info_vec_1[i].index) > 0) {
            //有重复的判断状态,有一个ok，就将当前的置为ok
            if (pin_map[pt_info_vec_1[i].index] > 0) {
                pt_info_vec_1[i].is_ok = true;
            }
        }
        //插入比较集
        if (pt_info_vec_1[i].is_ok) {
            pin_map.insert(std::pair<int,int>(pt_info_vec_1[i].index, 1));
        }
        else {
            pin_map.insert(std::pair<int, int>(pt_info_vec_1[i].index, 0));
        }
        if (!pt_info_vec_1[i].is_ok) {
            status_flag = false;
        }

        // 金针标签
        algo_result->result_info.push_back(
            { { "label", "J" },
                { "shapeType", "point" },
                { "points", { { pt_info_vec_1[i].img_pt.x, pt_info_vec_1[i].img_pt.y } } },
                { "result", { { "is_ok", pt_info_vec_1[i].is_ok }, { "x_off", pt_info_vec_1[i].offset_x }, { "y_off", pt_info_vec_1[i].offset_y }, { "std_x", pt_info_vec_1[i].org_pt.x }, { "std_y", pt_info_vec_1[i].org_pt.y }, { "measured_x", pt_info_vec_1[i].cal_pt.x }, { "measured_y", pt_info_vec_1[i].cal_pt.y }, { "TP", 2 * sqrt(pt_info_vec_1[i].offset_x * pt_info_vec_1[i].offset_x + pt_info_vec_1[i].offset_y * pt_info_vec_1[i].offset_y) }, { "index", pt_info_vec_1[i].index } } } });
    }
    for (int i = 0; i < pt_info_vec_2.size(); i++) {
        if (pt_info_vec_2[i].index < 0)
            continue;
        pt_info_vec_2[i].offset_x = pt_info_vec_2[i].offset_x - diff_x_2;
        pt_info_vec_2[i].offset_y = pt_info_vec_2[i].offset_y - diff_y_2;
        pt_info_vec_2[i].cal_pt.x = pt_info_vec_2[i].cal_pt.x - diff_x_2;
        pt_info_vec_2[i].cal_pt.y = pt_info_vec_2[i].cal_pt.y - diff_y_2;
        pt_info_vec_2[i].is_ok = std::abs(pt_info_vec_2[i].offset_x) > error_value_ ? false : true;
        pt_info_vec_2[i].is_ok = pt_info_vec_2[i].is_ok && (std::abs(pt_info_vec_2[i].offset_y) > error_value_ ? false : true);
        if (!pt_info_vec_2[i].is_ok) {
            status_flag = false;
        }
        // 铜片标签
        algo_result->result_info.push_back(
            { { "label", "T" },
                { "shapeType", "point" },
                { "points", { { pt_info_vec_2[i].img_pt.x, pt_info_vec_2[i].img_pt.y } } },
                { "result", { { "is_ok", pt_info_vec_2[i].is_ok }, { "x_off", pt_info_vec_2[i].offset_x }, { "y_off", pt_info_vec_2[i].offset_y }, { "std_x", pt_info_vec_2[i].org_pt.x }, { "std_y", pt_info_vec_2[i].org_pt.y }, { "measured_x", pt_info_vec_2[i].cal_pt.x }, { "measured_y", pt_info_vec_2[i].cal_pt.y }, { "TP", 2 * sqrt(pt_info_vec_2[i].offset_x * pt_info_vec_2[i].offset_x + pt_info_vec_2[i].offset_y * pt_info_vec_2[i].offset_y) }, { "index", pt_info_vec_2[i].index } } } });
    }
    if (!status_flag) {
        algo_result->judge_result = 0;
    } else {
        algo_result->judge_result = 1;
    }
}

Z_Male_Detect::point_info Z_Male_Detect::refind_point_1(const cv::Mat& img, cv::Point2f pt,  int idx,cv::Mat trans_m, double error_pix ,int pad)
{
    point_info pi;
    //判断有无

    cv::Rect sec_rect = cv::Rect( pt.x- error_pix-5,pt.y - error_pix - 5, 2*error_pix +10, 2 * error_pix+10);
    cv::Mat sec_img = img(sec_rect).clone();

    //没有亮斑返回
    cv::Rect fir_rect = cv::Rect(pt.x - error_pix, pt.y - error_pix, 2 * error_pix, 2 * error_pix );
    cv::Mat fir_img = img(fir_rect).clone();
    cv::Mat fir_test_img;
    cv::threshold(fir_img, fir_test_img, 200, 255, cv::THRESH_BINARY);
    int fir_area = cv::countNonZero(fir_test_img);
    if (fir_area<8) {
        pi.index = -1;
        return pi;
    }

    //亮斑过多返回
    cv::Mat test_img;
    cv::threshold(sec_img, test_img, 200, 255, cv::THRESH_BINARY);
    double no_zero = cv::countNonZero(test_img) / (test_img.cols * test_img.rows * 1.0);
    if (no_zero > s_no_zero_) {
        pi.index = -1;
        return pi;
    }


    int start_value = 180;
    int end_value =140;
    std::map<int, cv::Point2f> center_info;

    for (int cur_th = start_value; cur_th >=end_value; cur_th = cur_th - 10) {
        cv::Mat thre_img;
        cv::threshold(sec_img, thre_img, cur_th, 255, cv::THRESH_BINARY);
        cv::Point2f center;
        connector::get_center(thre_img, center);
        // 此时的center是相对于裁剪之后的，需要还原
        center.x = sec_rect.x + center.x;
        center.y = sec_rect.y + center.y;
        if (std::isnan(center.y) || std::isnan(center.x))
            continue;
        center_info.insert(std::pair<int, cv::Point2f>(cur_th, center));
    }

    double sumX = 0, sumY = 0;
    double total_w = 0;
    for (auto item : center_info) {
        if (std::isnan(item.second.y) || std::isnan(item.second.x))
            continue;
        int w = 255 - item.first; // 权重
        total_w = total_w + w;
        sumX = sumX + w * item.second.x;
        sumY = sumY + w * item.second.y;
    }

    if (total_w <10) {
        pi.index = -1;
        return pi;
    }
    cv::Point2f ret_center;
    ret_center.x = sumX / total_w;
    ret_center.y = sumY / total_w;


    cv::Point2f trans_pt = connector::TransPoint(trans_m, ret_center);
    pi.img_pt = ret_center;
    pi.tran_pt = trans_pt;


    double cur_x = pi.tran_pt.x;
    double cur_y = pi.tran_pt.y;
    cur_x = cur_x * pix_value_ / 1000.0;
    cur_y = cur_y * pix_value_ / 1000.0;
    pi.cal_pt.x = cur_x;
    pi.cal_pt.y = cur_y;
    pi.index = -1;

    pi.offset_x = (cur_x - tempImagePoint_1_[idx].x);
    pi.offset_y = (cur_y - tempImagePoint_1_[idx].y);
    pi.org_pt = tempImagePoint_1_[idx];
    pi.index = idx;
    pi.is_ok = std::abs(pi.offset_x) > error_value_ ? false : true;
    pi.is_ok = pi.is_ok && (pi.offset_y > error_value_ ? false : true);
    return pi;
}