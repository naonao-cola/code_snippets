#include "W_Female_Detect.h"
#include "../framework/InferenceEngine.h"
#include "../utils/Utils.h"
#include "../utils/logger.h"
#include "./sub_3rdparty/tival/include/FindLine.h"
#include "./sub_3rdparty/tival/include/JsonHelper.h"
#include "./sub_3rdparty/tival/include/ShapeBasedMatching.h"
#include "algo_tool.h"
#include "param_check.h"
#include "spinlock.h"
#include "xml_wr.h"
#include <execution>
#include <opencv2/core.hpp>
#include <opencv2/flann/flann.hpp>
#include <windows.h>
#include "img_feature.h"

USpinLock w_lock;
cv::Mat g_dis, g_dis_2, g_dis_3;
static int g_conut = 0;

REGISTER_ALGO(W_Female_Detect)
W_Female_Detect::W_Female_Detect() {
    svm_obj._train_size = cv::Size(88, 34);
    svm_obj.init("./", "lc_3_8.mdl");
    //_base_path + "\\" + _model_name
    LOGI("svm_obj path {} ", svm_obj._base_path + "\\" + svm_obj._model_name);

    svm_obj_kai._train_size = cv::Size(100, 55);
    svm_obj_kai.init("./", "kai.mdl");

    svm_obj_kbai._train_size = cv::Size(100, 55);
    svm_obj_kbai.init("./", "kbai.mdl");

    svm_obj_origin_k._train_size = cv::Size(100, 45);
    svm_obj_origin_k.init("./", "k_origin.mdl");



    svm_obj_top_tan._train_size = cv::Size(60, 30);
    svm_obj_top_tan.init("./", "lc_s.mdl");
  
    svm_obj_lc_ce._train_size = cv::Size(50, 90);
    svm_obj_lc_ce.init("./", "lc_ce.mdl");

}
W_Female_Detect::~W_Female_Detect() { }

AlgoResultPtr W_Female_Detect::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    TVALGO_FUNCTION_BEGIN

    algo_result->result_info.push_back(
        {
            {"label","W_Female_Detect"},
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

    cv::Rect detect_rect(detect_left_x_, detect_left_y_, detect_width_, detect_height_);
    if ((detect_left_x_ + detect_width_) > task->image.cols || (detect_left_y_ + detect_height_) > task->image.rows || detect_left_x_ < 0 || detect_left_y_ < 0) {
        TVALGO_FUNCTION_END
    }

    cv::Mat task_img_2 = cv::Mat::zeros(task->image.size(), task->image.type());
    cv::Mat task_img_1 = cv::Mat::zeros(task->image.size(), task->image.type());

    cv::Mat mask1, mask2;
    mask1 = task_img_1(detect_rect);
    mask2 = task_img_2(detect_rect);

    task->image(detect_rect).copyTo(mask2);
    task->image2(detect_rect).copyTo(mask1);

    input_img_1 = task_img_1.clone();
    input_img_2 = task_img_2.clone();

    cv::Mat dst_1, dst_2, gamma_img_1, gamma_img_2;

    if (task_img_1.channels() > 1)
        cv::cvtColor(task_img_1, dst_1, cv::COLOR_BGR2GRAY);
    if (task_img_2.channels() > 1)
        cv::cvtColor(task_img_2, dst_2, cv::COLOR_BGR2GRAY);

    // 通道分离
    cv::Mat hsv1, hsv2;
    cv::cvtColor(task_img_1, hsv1, cv::COLOR_BGR2HSV);
    cv::cvtColor(task_img_2, hsv2, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_v1, hsv_v2;
    cv::split(hsv1, hsv_v1);
    cv::split(hsv2, hsv_v2);

    LOGI("W_Female_Detect detect start");
    //3*8LC
    if (detect_flag_ == 1) {
        img_process_1(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //4*6 LC
    if (detect_flag_ == 2) {
        img_process_2(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //开口
    if (detect_flag_ == 3) {
        img_process_3(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //单体开口
    if (detect_flag_ == 4) {
        img_process_4(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //3*8 LC 单列
    if (detect_flag_ == 5) {
        img_process_5(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //乳白单体开口
    if (detect_flag_ == 6) {
        img_process_6(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //LC 3*8  灰色
    if (detect_flag_ == 7) {
        img_process_7(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
    //开口 10*13 跟最原始的开口一个形态
    if (detect_flag_ == 8) {
        img_process_8(dst_1, dst_2, hsv_v1, hsv_v2, algo_result);
    }
   
    LOGI("W_Female_Detect detect end");
    TVALGO_FUNCTION_END
}

std::tuple<std::string, json> W_Female_Detect::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map) const
{
    std::string task_type_id = task->image_info["type_id"];
    json task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

std::string g_path;
bool W_Female_Detect::get_param(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    std::tuple<std::string, json> details_info = get_task_info(task, pre_results, m_param_map);
    json task_param_json = std::get<1>(details_info);
    bool status = true;
    
    LOGI("W_Female_Detect get param start");
    
    //std::string tmp_path = nao::fl::parent_parh(task->image_info["img_path"]);
    //size_t pos = tmp_path.find_last_of("\\");
    //g_path = tmp_path.substr(pos+1);

	detect_flag_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_flag", 3);
	basis_x_num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "basis_x_num", 12);
	basis_y_num_ = Tival::JsonHelper::GetParam(task_param_json["param"], "basis_y_num", 8);
	detect_left_x_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_x", 0);
	detect_left_y_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_left_y", 0);
	detect_width_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_width", 9344);
	detect_height_ = Tival::JsonHelper::GetParam(task_param_json["param"], "detect_height", 7000);
	gamma_value_1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gamma_value_1", 0.8);
	gamma_value_2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gamma_value_2", 0.8);
	area_th_ = Tival::JsonHelper::GetParam(task_param_json["param"], "area_th", 15);

    a_ = Tival::JsonHelper::GetParam(task_param_json["param"], "a", 0.8);
    b1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "b1", 0.8);
    b2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "b2", 0.8);
    b3_ = Tival::JsonHelper::GetParam(task_param_json["param"], "b3", 0.8);
    c1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "c1", 0.8);
    c2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "c2", 0.8);
    d1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "d1", 0.8);
    d2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "d2", 0.8);
    e_ = Tival::JsonHelper::GetParam(task_param_json["param"], "e", 0.8);
    p_ = Tival::JsonHelper::GetParam(task_param_json["param"], "p", 0.8);
    f_ = Tival::JsonHelper::GetParam(task_param_json["param"], "f", 0.8);
  

    error_a_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_a", 0.8);
    error_b1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_b1", 0.8);
    error_b2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_b2", 0.8);
    error_b3_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_b3", 0.8);
    error_c1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_c1", 0.8);
    error_c2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_c2", 0.8);
    error_d1_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_d1", 0.8);
    error_d2_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_d2", 0.8);
    error_e_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_e", 0.8);
    error_p_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_p", 0.8);
    error_f_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_f", 0.8);
  

    tl_ = Tival::JsonHelper::GetParam(task_param_json["param"], "tl", 0.8);
    tr_ = Tival::JsonHelper::GetParam(task_param_json["param"], "tr", 0.8);
    dl_ = Tival::JsonHelper::GetParam(task_param_json["param"], "dl", 0.8);
    dr_ = Tival::JsonHelper::GetParam(task_param_json["param"], "dr", 0.8);
    md_ = Tival::JsonHelper::GetParam(task_param_json["param"], "md", 0.8);
    gap_u_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gap_u", 0.8);
    gap_m_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gap_m", 0.8);
    gap_d_ = Tival::JsonHelper::GetParam(task_param_json["param"], "gap_d", 0.8);
    error_tl_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_tl", 0.8);
    error_tr_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_tr", 0.8);
    error_dl_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_dl", 0.8);
    error_dr_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_dr", 0.8);
    error_md_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_md", 0.8);
    error_gap_u_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_gap_u", 0.8);
    error_gap_m_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_gap_m", 0.8);
    error_gap_d_ = Tival::JsonHelper::GetParam(task_param_json["param"], "error_gap_d", 0.8);

    l_th_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "l_th_value", 90);
    d_th_value_t_ = Tival::JsonHelper::GetParam(task_param_json["param"], "d_th_value_t", 15);
    d_th_value_b_ = Tival::JsonHelper::GetParam(task_param_json["param"], "d_th_value_b", 60);
    yellow_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "yellow_value", 80);

    dis_p_ = Tival::JsonHelper::GetParam(task_param_json["param"], "dis_p", 18.0);
    dis_p_t_ = Tival::JsonHelper::GetParam(task_param_json["param"], "dis_p_t", 18.0);
    top_tan_ = Tival::JsonHelper::GetParam(task_param_json["param"], "top_tan", false);

    svm_threshold_ = Tival::JsonHelper::GetParam(task_param_json["param"], "svm_threshold", 0.8);
    // 读取缺失基座
    miss_.clear();
    json miss_vec = task_param_json["param"]["miss"];
    for (int i = 0; i < miss_vec.size(); i++) {
        auto item = miss_vec[i];
        miss_.push_back(cv::Point(item[0], item[1]));
    }
    //指定ok 的基座
    right_.clear();
    json right_vec = task_param_json["param"]["right"];
    for (int i = 0; i < right_vec.size(); i++) {
        auto item = right_vec[i];
        right_.push_back(cv::Point(item[0], item[1]));
    }
    pix_value_ = Tival::JsonHelper::GetParam(task_param_json["param"], "pix_value", 5.26);

    LOGI("W_Female_Detect get param end");
    status &= InIntRange("detect_left_x", detect_left_x_, 0, 5120, false);
    status &= InIntRange("detect_left_y", detect_left_y_, 0, 5120, false);
    status &= InIntRange("detect_width", detect_width_, 0, 5120, false);
    status &= InIntRange("detect_height", detect_height_, 0, 5120, false);
    return status;
}

cv::Mat W_Female_Detect::get_edge(const cv::Mat& src, int th_value)
{
    cv::Mat soble_img_x, sobel_img_y, edge, edge_th;
    cv::Sobel(src, sobel_img_y, CV_16S, 0, 1,-1);
    cv::Sobel(src, soble_img_x, CV_16S, 1, 0,-1);
    edge = soble_img_x + sobel_img_y;
    cv::convertScaleAbs(edge, edge);
    cv::threshold(edge, edge_th, th_value, 255, cv::THRESH_BINARY);
    return edge_th;
}

// 0 代表按列求均值，被处理成一行,  1代表按行求均值，被处理成为一列,，投影
std::vector<int> W_Female_Detect::get_his(const cv::Mat& img, int type, int th_value)
{

    std::vector<int> col_vec;
    if (type == 0) {
        for (int i = 0; i < img.cols; i++) {
            int count = 0;
            int sum = 0;
            for (int j = 0; j < img.rows; j++) {
                int value = img.at<uchar>(j, i);
                if (value < th_value)
                    continue;
                count++;
                sum = sum + value;
            }
            int avage;
            if (count > 0) {
                avage = sum / count;
            } else {
                avage = 0;
            }
            col_vec.push_back(avage);
        }
    }
    if (type == 1) {
        for (int i = 0; i < img.rows; i++) {
            int count = 0;
            int sum = 0;
            for (int j = 0; j < img.cols; j++) {
                int value = img.at<uchar>(i, j);
                if (value < th_value)
                    continue;
                count++;
                sum = sum + value;
            }
            int avage;
            if (count > 0) {
                avage = sum / count;
            } else {
                avage = 0;
            }
            col_vec.push_back(avage);
        }
    }
    return col_vec;
}
// 最大值滤波
std::vector<int> max_filter(std::vector<int> data, int step) noexcept
{
    std::vector<int> output;
    output.resize(data.size());
    int halfKernel = step / 2;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < 100) {
            output[i] = 0;
            continue;
        }
        // 计算当前窗口的开始和结束索引
        int start = std::max<int>(0, i - halfKernel);
        int end = std::min<int>(static_cast<int>(data.size()) - 1, i + halfKernel);
        // 在当前窗口内找到最大值
        int maxVal = *std::max_element(data.begin() + start, data.begin() + end + 1);
        int index = std::max_element(data.begin() + start, data.begin() + end + 1) - data.begin();

        // 将最大值存储在输出向量的相应位置
        output[index] = maxVal;
    }
    return output;
}

cv::Mat W_Female_Detect::find_edge(const cv::Mat& img, int& tv, int& bv, int& lv, int& rv, EDGE_TYPE edge_type, int st, int sb, int sl, int sr, int th_val, int his_th, int step) noexcept
{
    // 初始值
    tv = bv = lv = rv = 0;
    // 得到一行的投影
    std::vector<int> col_value = get_his(img, 0, his_th);
    // 得到一行的投影
    std::vector<int> row_value = get_his(img, 1, his_th);

    std::vector<int> col_diff;
    col_diff.resize(col_value.size());
    for (size_t i = 1; i < col_value.size(); ++i)
        col_diff[i] = abs(col_value[i] - col_value[i - 1]);

    std::vector<int> row_diff;
    row_diff.resize(row_value.size());
    for (size_t i = 1; i < row_value.size(); i++)
        row_diff[i] = abs(row_value[i] - row_value[i - 1]);

    cv::Mat dis = cv::Mat::zeros(img.size(), img.type());

    // 中间向两边
    if (edge_type == EDGE_TYPE::M2LR) {
        for (size_t i = sr; i < col_diff.size(); i++) {
            if (col_diff[i] >= th_val) {
                rv = i;
                break;
            }
        }
        for (int i = sl; i >= 0; i--) {
            if (col_diff[i] >= th_val && i >= 0) {
                lv = i;
                break;
            }
            if (i <= 0) {
                lv = -1;
                break;
            }
        }
    }
    // 中间向上下
    if (edge_type == EDGE_TYPE::M2TB) {
        for (size_t i = sb; i < row_diff.size(); i++) {
            if (row_diff[i] >= th_val) {
                bv = i;
                break;
            }
        }
        for (int i = st; i >= 0; i--) {
            if (row_diff[i] >= th_val && i >= 0) {
                tv = i;
                break;
            }
            if (i <= 0) {
                tv = -1;
                break;
            }
        }
    }
    // 中间向四周
    if (edge_type == EDGE_TYPE::M2F) {
        for (size_t i = sr; i < col_diff.size(); i++) {
            if (col_diff[i] >= th_val) {
                rv = i;
                break;
            }
        }
        for (int i = sl; i >= 0; i--) {
            if (col_diff[i] >= th_val && i >= 0) {
                lv = i;
                break;
            }
            if (i <= 0) {
                lv = -1;
                break;
            }
        }
        for (size_t i = sb; i < row_diff.size(); i++) {
            if (row_diff[i] >= th_val) {
                bv = i;
                break;
            }
        }
        for (int i = st; i >= 0; i--) {
            if (row_diff[i] >= th_val && i >= 0) {
                tv = i;
                break;
            }
            if (i <= 0) {
                tv = -1;
                break;
            }
        }
    }
    // 下到上
    if (edge_type == EDGE_TYPE::B2T) {
        for (int i = st; i > 0; i--) {
            if (row_value[i] > 100) {
                bv = i;
                break;
            }
            if (row_diff[i] >= th_val && i >= 0 && i < row_diff.size()) {
                bv = i;
                break;
            }
            if (i <= 0) {
                bv = -1;
                break;
            }
        }
    }
    // 上到下
    if (edge_type == EDGE_TYPE::T2B) {
        for (size_t i = sb; i < row_diff.size(); i++) {
            if (row_diff[i] >= th_val) {
                tv = i;
                break;
            }
        }
    }
    // 两边到中间
    if (edge_type == EDGE_TYPE::LR2M) {
        for (int i = sr; i >= 0; i--) {
            if (i <= 0) {
                rv = -1;
                break;
            }
            if (col_diff[i] >= th_val && i >= 0 && i < col_diff.size()) {
                rv = i;
                break;
            }
        }
        for (size_t i = sl; i < col_diff.size(); i++) {
            if (col_diff[i] >= th_val) {
                lv = i;
                break;
            }
        }
    }

    if (edge_type == EDGE_TYPE::L2R) {
    }
    if (edge_type == EDGE_TYPE::B2T2) {
        // 最大值滤波
        std::vector<int> filter_data = max_filter(row_value, 5);
        for (int i = st; i > 0; i--) {
            if (filter_data[i] >= th_val && i >= 0 && i < filter_data.size()) {
                bv = i;
                break;
            }
            if (i <= 0) {
                bv = -1;
                break;
            }
        }
    }
    if (tv != 0) {
        cv::line(dis, cv::Point(0, tv), cv::Point(img.cols, tv), cv::Scalar::all(255));
    }
    if (bv != 0) {
        cv::line(dis, cv::Point(0, bv), cv::Point(img.cols, bv), cv::Scalar::all(255));
    }
    if (lv != 0) {
        cv::line(dis, cv::Point(lv, 0), cv::Point(lv, img.rows), cv::Scalar::all(255));
    }
    if (rv != 0) {
        cv::line(dis, cv::Point(rv, 0), cv::Point(rv, img.rows), cv::Scalar::all(255));
    }

    return dis;
}

cv::Mat W_Female_Detect::find_edge_2(const cv::Mat& img, int& tv, int& bv, int& lv, int& rv, EDGE_TYPE edge_type, int st, int sb, int sl, int sr, int th_val)
{

    tv = bv = lv = rv = 0;
    cv::Mat th_img;
    int mean = cv::mean(img)[0];
    cv::threshold(img, th_img, th_val, 255, cv::THRESH_BINARY);

    // 中间向两边
    if (edge_type == EDGE_TYPE::M2LR) {
        if (th_val >= 120 && th_val <= 200) {

            /*cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 7));
            cv::Mat open_img;
            cv::morphologyEx(th_img, open_img, cv::MORPH_OPEN, element);*/

            std::vector<std::vector<cv::Point>> draw_2_contor = connector::get_contours(th_img);
            for (int i = 0; i < draw_2_contor.size(); i++) {
                cv::Rect rect = cv::boundingRect(draw_2_contor[i]);
                cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
                double area = cv::contourArea(draw_2_contor[i]);
                if (c_pt.y < 10 || c_pt.x < 10 || c_pt.x > th_img.cols - 10)
                    continue;
                if (area < 70)
                    continue;
                if (c_pt.x < 10 && c_pt.x > th_img.cols - 10)
                    continue;
                if (rect.x < th_img.cols / 2 - 10) {
                    lv = rect.x + rect.width;
                } else {
                    rv = rect.x;
                }
            }
        } else if (th_val > 200) {
            // 在最下行补黑
            /*cv::Mat bottom_img = th_img(cv::Rect(0, th_img.rows - 2, th_img.cols, 2));
            bottom_img = 0;*/
            th_img(cv::Rect(0, th_img.rows - 5, th_img.cols, 5)).setTo(cv::Scalar(0));

            std::vector<std::vector<cv::Point>> draw_2_contor = connector::get_contours(th_img);
            for (int i = 0; i < draw_2_contor.size(); i++) {
                cv::Rect rect = cv::boundingRect(draw_2_contor[i]);
                cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
                double area = cv::contourArea(draw_2_contor[i]);
                if (c_pt.y < 15 || c_pt.x < 10 || c_pt.x > th_img.cols - 10)
                    continue;
                if (c_pt.x < 10 || c_pt.x > th_img.cols - 10)
                    continue;
                if (area < 50)
                    continue;
                if (rect.x < th_img.cols / 2 - 10) {
                    lv = rect.x + rect.width + 0.7;
                } else {
                    rv = rect.x - 0.7;
                }
            }
        } else {
            int count = 0;
            for (size_t i = sl; i < sr; i++) {
                int cur_value = th_img.at<uchar>(th_img.rows / 2 + 10, i);
                int next_value = th_img.at<uchar>(th_img.rows / 2 + 10, i + 1);
                int diff = abs(cur_value - next_value);
                if (diff > 200) {
                    if (count == 0) {
                        lv = i;
                    }
                    if (count == 1) {
                        rv = i;
                    }
                    if (count > 1) {
                        break;
                    }
                    count++;
                }
            }
        }
    }
    // LR2M = 6,   //两边到中间
    if (edge_type == EDGE_TYPE::LR2M) {
        if (th_val >= 180) {
            for (size_t i = th_img.cols / 2 + 10; i < th_img.cols - 1; i++) {
                int cur_value = th_img.at<uchar>(th_img.rows / 2 + 15, i);
                if (cur_value < 50) {
                    rv = i;
                    break;
                }
            }
            for (size_t i = th_img.cols / 2 - 10; i >= 0; i--) {
                int cur_value = th_img.at<uchar>(th_img.rows / 2 + 15, i);
                if (cur_value < 50 && i >= 0) {
                    lv = i;
                    break;
                }
            }
            if (lv <= 0) {
                lv = 0;
            }
        } else if (th_val >= 120 && th_val < 180) {
            std::vector<std::vector<cv::Point>> draw_2_contor = connector::get_contours(th_img);
            for (int i = 0; i < draw_2_contor.size(); i++) {
                cv::Rect rect = cv::boundingRect(draw_2_contor[i]);
                cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
                double area = cv::contourArea(draw_2_contor[i]);
                if (c_pt.y < 15)
                    continue;
                if (area < 100)
                    continue;
                if (rect.x < th_img.cols / 2 - 10) {
                    lv = rect.x;
                } else {
                    rv = rect.x + rect.width;
                }
            }
        } else {

            for (size_t i = th_img.cols / 2 + 10; i < th_img.cols - 1; i++) {
                int cur_value = th_img.at<uchar>(th_img.rows / 2 + 15, i);
                if (cur_value < 50) {
                    rv = i;
                    break;
                }
            }
            for (size_t i = th_img.cols / 2 - 10; i >= 0; i--) {
                int cur_value = th_img.at<uchar>(th_img.rows / 2 + 15, i);
                if (cur_value < 50 && i >= 0) {
                    lv = i;
                    break;
                }
            }
            if (lv == 0) {
                lv = 0;
            }
        }
    }
    // B2T = 4,   // 下到上
    if (edge_type == EDGE_TYPE::B2T) {
    }

    cv::Mat dis = cv::Mat::zeros(img.size(), img.type());
    if (tv != 0) {
        cv::line(dis, cv::Point(0, tv), cv::Point(img.cols, tv), cv::Scalar::all(255));
    }
    if (bv != 0) {
        cv::line(dis, cv::Point(0, bv), cv::Point(img.cols, bv), cv::Scalar::all(255));
    }
    if (lv != 0) {
        cv::line(dis, cv::Point(lv, 0), cv::Point(lv, img.rows), cv::Scalar::all(255));
    }
    if (rv != 0) {
        cv::line(dis, cv::Point(rv, 0), cv::Point(rv, img.rows), cv::Scalar::all(255));
    }
    return dis;
}

void get_line_pt(cv::Mat img, int height, int& start, int& end)
{
    std::vector<std::pair<int, int>> se;
    for (int i = 0; i < img.cols - 1; i++) {
        int p_value = img.at<uchar>(height, i);
        int p_n_value = img.at<uchar>(height, i + 1);
        int diff = abs(p_value - p_n_value);
        if (diff > 0) {
            if (start == 0) {
                start = i + 1;
            } else {
                end = i + 1;
                se.push_back(std::pair(start, end));
                start = i + 1;
                // break;
            }
        }
    }
    if (se.size() == 1) {
        start = se[0].first;
        end = se[0].second;
    }
    if (se.size() > 1) {
        int index = 0;
        int min_value = 100;
        for (int i = 0; i < se.size(); i++) {
            int tmp_start = se[i].first;
            int tmp_end = se[i].second;
            double mid = (tmp_start + tmp_end) / 2;
            double diff = abs(mid - img.cols / 2);
            if (diff < min_value) {
                index = i;
                min_value = diff;
            }
        }
        start = se[index].first;
        end = se[index].second;
    }
    if (se.size() == 0) {
        start = 0;
        end = 0;
    }
}

void W_Female_Detect::get_top_line_pt(cv::Mat img, cv::Point2f& l_pt, cv::Point2f& r_pt, cv::Rect rect)
{

    cv::Mat th_img;
    cv::threshold(img, th_img, 80, 255, cv::THRESH_BINARY);

    // 左侧 四分之一
    cv::Rect l_rect(0, 0, th_img.cols / 4, th_img.rows);
    cv::Mat l_img = th_img(l_rect).clone();
    std::vector<int> l_his = get_his(l_img, 1, 0);
    // 右侧 四分之一
    cv::Rect r_rect(th_img.cols / 4 * 3, 0, th_img.cols / 4, th_img.rows);
    cv::Mat r_img = th_img(r_rect).clone();
    std::vector<int> r_his = get_his(r_img, 1, 0);

    std::vector<int> l_diff;
    l_diff.resize(l_his.size());
    for (size_t i = 0; i < l_his.size() - 1; ++i)
        l_diff[i] = abs(l_his[i] - l_his[i + 1]);

    std::vector<int> r_diff;
    r_diff.resize(r_his.size());
    for (size_t i = 0; i < r_his.size() - 1; i++)
        r_diff[i] = abs(r_his[i] - r_his[i + 1]);

    int l_index = std::max_element(l_diff.begin(), l_diff.end()) - l_diff.begin();
    int r_index = std::max_element(r_diff.begin(), r_diff.end()) - r_diff.begin();

    l_pt = cv::Point2f(0, l_index);
    r_pt = cv::Point2f(img.cols, r_index);

    l_pt.x = l_pt.x + rect.x;
    l_pt.y = l_pt.y + rect.y;

    r_pt.x = r_pt.x + rect.x;
    r_pt.y = r_pt.y + rect.y;
}


cv::Rect W_Female_Detect::reget_rect(const cv::Mat& img, const cv::Rect& rect)
{
    // 得到一行的投影
    std::vector<int> col_value = get_his(img, 0, 0);
    // 得到一行的投影
    std::vector<int> row_value = get_his(img, 1, 0);

    int tv, bv, lv, rv;
    tv = bv = lv = rv = 0;
    for (size_t i = 0; i < col_value.size(); i++) {
        if (col_value[i] >= 10) {
            lv = i;
            break;
        }
    }
    for (int i = col_value.size() - 1; i >= 0; i--) {
        if (col_value[i] >= 10 && i >= 0) {
            rv = i;
            break;
        }
        if (i < 0) {
            rv = -1;
            break;
        }
    }
    for (size_t i = 0; i < row_value.size(); i++) {
        if (row_value[i] >= 10) {
            tv = i;
            break;
        }
    }
    for (int i = row_value.size() - 1; i >= 0; i--) {
        if (row_value[i] >= 10 && i >= 0) {
            bv = i + 2;
            break;
        }
        if (i < 0) {
            bv = -1;
            break;
        }
    }
    cv::Rect ret_rect(lv, tv, abs(rv - lv), abs(bv - tv));
    ret_rect.x = ret_rect.x + rect.x;
    ret_rect.y = ret_rect.y + rect.y;
    return ret_rect;
}
//
void W_Female_Detect::img_process_3(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept
{
    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 25;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::dilate(th_img_1, th_img_1, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;

    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 140 || rect.height > 90)continue;
        if (width > 175)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }

    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                } else {
                    return false;
                }
            } else {
                if (lhs.y < rhs.y) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            } else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    } else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;
#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 140 || rect.height > 90 || rect.width <= 90)continue;

        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 175)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
         cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        /*cv::drawContours(gray_mask, draw_conts, 0, 255, -1);*/
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            } else {
                return false;
            }
        } else {
            if (lhs.y < rhs.y) {
                return true;
            } else {
                return false;
            }
        }
    });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }

    std::vector<W_Female_Detect::w_female> w_female_vec;
    cv::Vec4i af_vec(131, 0, 103, 47);
    cv::Vec4i pre_vec(-144, 0, 103, 47);

    // 同一行矩形的相对关系
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 105, 48),
        cv::Vec4i(146, 0, 94, 48),
        cv::Vec4i(302, 0, 94, 48),
        cv::Vec4i(434, 0, 105, 48),
        cv::Vec4i(579, 0, 94, 48),
        cv::Vec4i(735, 0, 94, 48),
        cv::Vec4i(868, 0, 105, 48),
        cv::Vec4i(1013, 0, 94, 48),
        cv::Vec4i(1168, 0, 94, 48),
        cv::Vec4i(1300, 0, 105, 48),
        cv::Vec4i(1444, 0, 94, 48),
        cv::Vec4i(1600, 0, 94, 48),
        cv::Vec4i(1733, 0, 105, 48),
        cv::Vec4i(1877, 0, 94, 48),
        cv::Vec4i(2033, 0, 94, 48),
        cv::Vec4i(2166, 0, 105, 48),
        cv::Vec4i(2309, 0, 94, 48),
        cv::Vec4i(2465, 0, 94, 48),
        cv::Vec4i(2599, 0, 105, 48),
    };

    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            ;
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            } else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }

        for (int i = 0; i < rank.size(); i++) {
            // 每一行进行处理,最后一行特殊处理
            bool estimate_flag = false;
            //if (rank[i].size() != 12 || i == 5) {
            //    estimate_flag = false;
            //}
            //if (rank[i].size() == 12) {
            //    // 求间距，两两求间距。求标准间距，间距不合规，则表示 有漏检的
            //    double dis = 0;
            //    for (int j = 0; j < rank[i].size() - 1; j = j + 2) {
            //        cv::Point p1(rank[i][j].x + rank[i][j].width / 2, rank[i][j].y + rank[i][j].height / 2);
            //        cv::Point p2(rank[i][j + 1].x + rank[i][j + 1].width / 2, rank[i][j + 1].y + rank[i][j + 1].height / 2);
            //        dis = dis + std::sqrt(std::pow((p1.x - p2.x), 2) + std::pow(p1.y - p2.y, 2));
            //    }
            //    dis = dis / 6;
            //    if (dis > 170) {
            //        estimate_flag = false;
            //    }
            //}
            if (estimate_flag) {
                // 当前行的个数是12个，要进行间隔插一个
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    if (j % 2 == 0) {
                        cv::Rect tmp_rect(cur_rect.x + pre_vec[0], cur_rect.y + pre_vec[1], pre_vec[2], pre_vec[3]);
                        process_rect_vec.push_back(tmp_rect);
                        process_rect_vec.push_back(cur_rect);
                        cv::rectangle(gray_mask, tmp_rect, cv::Scalar::all(255));
                        cv::rectangle(g_dis, tmp_rect, cv::Scalar(0, 0, 255));
                        cv::rectangle(g_dis_2, tmp_rect, cv::Scalar(0, 0, 255));

                    } else if (j == rank[i].size() - 1) {
                        cv::Rect tmp_rect(cur_rect.x + af_vec[0], cur_rect.y + af_vec[1], af_vec[2], af_vec[3]);
                        process_rect_vec.push_back(cur_rect);
                        process_rect_vec.push_back(tmp_rect);
                        cv::rectangle(gray_mask, tmp_rect, cv::Scalar::all(255));
                        cv::rectangle(g_dis, tmp_rect, cv::Scalar(0, 0, 255));
                        cv::rectangle(g_dis_2, tmp_rect, cv::Scalar(0, 0, 255));
                    } else {
                        process_rect_vec.push_back(cur_rect);
                    }
                }
            } else {
                // 当前行未找全的，特殊处理
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                if (i % 2 == 0) {
                    // 偶数行
                    s_col_idx = (s_rect.x - detect_left_x_) / 144;
                }
                if (i % 2 == 1) {
                    // 奇数行
                    s_col_idx = (s_rect.x - 126 - detect_left_x_) / 144;
                }
                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    c_col_idx = ((cur_rect.x - s_rect.x) / 144.0 + 0.5) + s_col_idx;
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }

    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            } else {
                return false;
            }
        } else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            } else {
                return false;
            }
        }
    });

    std::vector<cv::Mat> rbg_v1, rbg_v2;
    cv::split(input_img_1, rbg_v1);
    cv::split(input_img_2, rbg_v2);

//#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i++) {
        w_female singal_female = cal_3(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, rbg_v1, rbg_v2,algo_result, process_rect_vec[i], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        w_female_vec.push_back(singal_female);
        w_lock.unlock();
    }
    data_cvt_3(w_female_vec, algo_result);
}

std::vector<cv::Rect> W_Female_Detect::get_complete_rect(std::vector<cv::Vec4i> estimate_rect_1, cv::Rect cur_rect, int c_col_idx)
{
    std::vector<cv::Rect> ret_rect_vec;
    // 当前矩形
    // 当前矩形的序号
    for (int i = 0; i < estimate_rect_1.size(); i++) {
        int diff = i - c_col_idx;
        int c_x = 0;
        if (diff <= 0) {
            c_x = -(estimate_rect_1[c_col_idx][0] - estimate_rect_1[i][0]) + cur_rect.x;
        } else {
            c_x = (estimate_rect_1[i][0] - estimate_rect_1[c_col_idx][0]) + cur_rect.x;
        }
        cv::Rect tmp_rect(c_x, cur_rect.y, estimate_rect_1[i][2], estimate_rect_1[i][3]);
        ret_rect_vec.push_back(tmp_rect);
    }
    return ret_rect_vec;
}

void W_Female_Detect::data_cvt_3(std::vector<W_Female_Detect::w_female> w_female_vec, AlgoResultPtr algo_result)
{
    status_flag = true;

    for (int i = 0; i < w_female_vec.size(); i++) {
        w_female tmp_lc = w_female_vec[i];

        // 中线
        cv::Point2f p1 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_lc.mid_line.startPointX, tmp_lc.mid_line.startPointY));
        cv::Point2f p2 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_lc.mid_line.endPointX, tmp_lc.mid_line.endPointY));
        algo_result->result_info.push_back(
            { { "label", "fuzhu" },
                { "shapeType", "line" },
                { "points", { { p1.x, p1.y }, { p2.x, p2.y } } },
                { "result", 1 } });

        cv::Rect tp_rect = tmp_lc.template_rect;
        cv::Point2f lt = tp_rect.tl();
        cv::Point2f lb(tp_rect.tl().x, tp_rect.tl().y + tp_rect.height);
        cv::Point2f rt(tp_rect.br().x, tp_rect.br().y - tp_rect.height);
        cv::Point2f rb = tp_rect.br();

        cv::Point2f pc(tp_rect.x + tp_rect.width / 2, tp_rect.y + tp_rect.height / 2);

        cv::Point2f org_lt = connector::TransPoint(tmp_lc.inv_h, lt);
        cv::Point2f org_lb = connector::TransPoint(tmp_lc.inv_h, lb);
        cv::Point2f org_rt = connector::TransPoint(tmp_lc.inv_h, rt);
        cv::Point2f org_rb = connector::TransPoint(tmp_lc.inv_h, rb);
        cv::Point2f org_pc = connector::TransPoint(tmp_lc.inv_h, pc);

        algo_result->result_info.push_back(
            { { "label", "fuzhu" },
                { "shapeType", "polygon" },
                { "points", { { org_lt.x, org_lt.y }, { org_rt.x, org_rt.y }, { org_rb.x, org_rb.y }, { org_lb.x, org_lb.y } } },
                { "result", 1 } });

        for (int j = 0; j < tmp_lc.line_vec.size(); j++) {
            KeyLine tmp_l = tmp_lc.line_vec[j];
            cv::Point2f p3 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.startPointX, tmp_l.startPointY));
            cv::Point2f p4 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.endPointX, tmp_l.endPointY));
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "line" },
                    { "points", { { p3.x, p3.y }, { p4.x, p4.y } } },
                    { "result", 1 } });
        }

        double status_tl, status_tr, status_dl, status_dr, status_md, status_gap_u, status_gap_m, status_gap_d;

        double e_tl = abs(tmp_lc.tl - tl_);
        double e_tr = abs(tmp_lc.tr - tr_);
        double e_dl = abs(tmp_lc.dl - dl_);
        double e_dr = abs(tmp_lc.dr - dr_);
        double e_md = abs(tmp_lc.md - md_);
        double e_gap_u = abs(tmp_lc.gap_u - gap_u_);
        double e_gap_m = abs(tmp_lc.gap_m - gap_m_);
        double e_gap_d = abs(tmp_lc.gap_d - gap_d_);

        status_tl = e_tl < error_tl_ ? 1 : 0;
        status_tr = e_tr < error_tr_ ? 1 : 0;
        status_dl = e_dl < error_dl_ ? 1 : 0;
        status_dr = e_dr < error_dr_ ? 1 : 0;
        status_md = e_md < error_md_ ? 1 : 0;
        status_gap_u = e_gap_u < error_gap_u_ ? 1 : 0;
        status_gap_m = e_gap_m < error_gap_m_ ? 1 : 0;
        status_gap_d = e_gap_d < error_gap_d_ ? 1 : 0;

        if (status_tl < 1 || status_tr < 1 || status_dl < 1 || status_dr < 1 || status_md < 1 || status_gap_u < 1 || status_gap_m < 1 || status_gap_d < 1) {
            status_flag = false;
        } else {
        }

        if (!status_flag) {
            algo_result->judge_result = 0;
        } else {
            algo_result->judge_result = 1;
        }
        algo_result->result_info.push_back(
            { { "label", "Curved_Bow_Train_defect" },
                { "shapeType", "basis" },
                { "points", { { -1, -1 } } },
                { "result", { { "dist", { tmp_lc.tl, tmp_lc.tr, tmp_lc.dl, tmp_lc.dr, tmp_lc.md, tmp_lc.gap_u, tmp_lc.gap_m, tmp_lc.gap_d } }, { "status", { status_tl, status_tr, status_dl, status_dr, status_md, status_gap_u, status_gap_m, status_gap_d } }, { "error", { e_tl, e_tr, e_dl, e_dr, e_md, e_gap_u, e_gap_m, e_gap_d } }, { "index", (int)tmp_lc.index }, { "points", { { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y }, { org_pc.x, org_pc.y } } } } } });
    }
}

W_Female_Detect::w_female W_Female_Detect::cal_3(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, std::vector<cv::Mat> rbg_v1, std::vector<cv::Mat>rbg_v2, AlgoResultPtr algo_result, cv::Rect rect, int index, cv::Mat inv_m)
{
    w_female singal_w_female;
    singal_w_female.template_rect = rect;
    singal_w_female.index = index;
    singal_w_female.mid_line.startPointX = rect.x + rect.width / 2;
    singal_w_female.mid_line.startPointY = rect.y;
    singal_w_female.mid_line.endPointX = rect.x + rect.width / 2;
    singal_w_female.mid_line.endPointY = rect.y + rect.height;

    cv::Mat img_h1 = hsv_v1[0](rect).clone();
    cv::Mat img_s1 = hsv_v1[1](rect).clone();
    cv::Mat img_v1 = hsv_v1[2](rect).clone();

    cv::Mat img_h2 = hsv_v2[0](rect).clone();
    cv::Mat img_s2 = hsv_v2[1](rect).clone();
    cv::Mat img_v2 = hsv_v2[2](rect).clone();

    cv::Mat img_b1 = rbg_v1[0](rect).clone();
    cv::Mat img_g1 = rbg_v1[1](rect).clone();
    cv::Mat img_r1 = rbg_v1[2](rect).clone();
    cv::Mat img_b2 = rbg_v2[0](rect).clone();
    cv::Mat img_g2 = rbg_v2[1](rect).clone();
    cv::Mat img_r2 = rbg_v2[2](rect).clone();

    cv::Mat img_rgb_1 = input_img_1(rect).clone();
    cv::Mat img_rgb_2 = input_img_2(rect).clone();

    
    //保存图片
    std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + ".jpg";
    cv::imwrite(file_name, img_rgb_1);
    g_conut++;

    file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + ".jpg";
    cv::imwrite(file_name, img_rgb_2);
    g_conut++;
 
    std::vector<cv::Mat> test_img_vec_1, test_img_vec_2;
    test_img_vec_1.push_back(img_rgb_1);
    test_img_vec_2.push_back(img_rgb_2);
    nao::img::feature::HogTransform transform_1(test_img_vec_1, 11, 8, 7, cv::Size(100, 45), 1);
    nao::img::feature::HogTransform transform_2(test_img_vec_2, 11, 8, 7, cv::Size(100, 45), 1);
    cv::Mat test_feature_1 = transform_1();
    cv::Mat test_feature_2 = transform_2();
    double prob_1[2];
    double prob_2[2];
    double ret_1 = svm_obj_origin_k.testFeatureLibSVM(test_feature_1, prob_1);
    double ret_2 = svm_obj_origin_k.testFeatureLibSVM(test_feature_2, prob_2);

    // 黑色框
    int col_idx = index % 19;
    int row_idx = index / 19;
    bool ng_flag = false;
    for (int i = 0; i < miss_.size();i++) {
        if (row_idx ==miss_[i].x && col_idx==miss_[i].y) {
            ng_flag = true;
            break;
        }
    }
    for (int i = 0; i < right_.size(); i++) {
        if (row_idx == right_[i].x && col_idx == right_[i].y) {
            ng_flag = false;
            return singal_w_female;
        }
    }

    if (prob_1[1] > svm_threshold_ || prob_2[1] > svm_threshold_ || ng_flag) {
        singal_w_female.md = md_ + error_md_ + 0.01;
        return singal_w_female;
    }
    //return singal_w_female;

    // 判断高曝光还是低曝光
    int mean_h1 = cv::mean(img_h1)[0];
    int mean_s1 = cv::mean(img_s1)[0];
    int mean_v1 = cv::mean(img_v1)[0];

    //h2 判断是否有亮图里面的黑块弹片是否有缺失
    int mean_h2 = cv::mean(img_h2)[0];
    int mean_s2 = cv::mean(img_s2)[0];
    int mean_v2 = cv::mean(img_v2)[0];

   

    if (col_idx == 1 || col_idx == 2 || col_idx == 4 || col_idx == 5 || col_idx == 7 || col_idx == 8 || col_idx == 10 || col_idx == 11 || col_idx == 13 || col_idx == 14 || col_idx == 16 || col_idx == 17) {

        //增加判断有无，有黄铜片，中间会有黑洞
        //cv::Mat l_h2, r_h2;
        //if (row_idx != 5 && row_idx != 0) {
        //    cv::Rect l_rect(5, 3, img_v1.cols / 2, img_v1.rows - 6);
        //    cv::Rect r_rect(img_v1.cols / 2, 3, img_v1.cols / 2 - 5, img_v1.rows - 6);
        //    l_h2 = img_h2(l_rect);
        //    r_h2 = img_h2(r_rect);
        //}
        //if (row_idx == 5 || row_idx == 0) {
        //    // 最后一排
        //    cv::Rect l_rect(13, 3, img_v1.cols / 2 - 13, img_v1.rows - 6);
        //    cv::Rect r_rect(img_v1.cols / 2 + 5, 3, img_v1.cols / 2 - 15, img_v1.rows - 6);
        //    l_h2 = img_h2(l_rect);
        //    r_h2 = img_h2(r_rect);
        //}
        //int mean_l_h2 = cv::mean(l_h2)[0];
        //int mean_r_s2 = cv::mean(r_h2)[0];

        //cv::Mat l_r_t = l_h2(cv::Rect(0, 0, l_h2.cols, l_h2.rows/2));
        //cv::Mat l_r_b = l_h2(cv::Rect(0, l_h2.rows / 2, l_h2.cols, l_h2.rows / 2));
        //cv::Mat r_r_t = r_h2(cv::Rect(0, 0, r_h2.cols, r_h2.rows / 2));
        //cv::Mat r_r_b = r_h2(cv::Rect(0, r_h2.rows / 2, r_h2.cols, r_h2.rows / 2));

        //int mean_l_r_t = cv::mean(l_r_t)[0];
        //int mean_l_r_b = cv::mean(l_r_b)[0];
        //int mean_r_r_t = cv::mean(r_r_t)[0];
        //int mean_r_r_b = cv::mean(r_r_b)[0];



        //左右差异过大，证明左右两边弹片不一致
        //if (mean_l_h2 >= 90 || mean_r_s2>=90 || std::abs(mean_l_h2 - mean_r_s2)>20 || std::abs(mean_l_r_t - mean_l_r_b) > 50 || std::abs(mean_r_r_t - mean_r_r_b) > 50) {
        //    singal_w_female.md = md_ + error_md_ + 0.01;
        //    return singal_w_female;
        //}

        //cv::Mat l_tp_img, r_tp_img;
        //int th_judge_value = 0;
        //cv::Mat l_rgb_img, r_rgb_img;
        //int th_value = 100;
        //if (row_idx != 5 && row_idx != 0) {
        //    cv::Rect l_rect(10, 3, img_v1.cols / 2-5, img_v1.rows - 6);
        //    cv::Rect r_rect(img_v1.cols / 2, 3, img_v1.cols / 2 - 5, img_v1.rows - 6);
        //    l_tp_img = img_v1(l_rect).clone();
        //    r_tp_img = img_v1(r_rect).clone();
        //    l_rgb_img = img_rgb_1(l_rect).clone();
        //    r_rgb_img = img_rgb_1(r_rect).clone();
        //    th_value = 80;
        //    th_judge_value = d_th_value_t_;
        //}
        //if (row_idx == 5 || row_idx == 0) {
        //    // 最后一排
        //    cv::Rect l_rect(13, 3, img_v1.cols / 2-13, img_v1.rows - 6);
        //    cv::Rect r_rect(img_v1.cols / 2+5, 3, img_v1.cols / 2 - 15, img_v1.rows - 6);
        //    l_tp_img = img_v1(l_rect).clone();
        //    r_tp_img = img_v1(r_rect).clone();
        //    l_rgb_img = img_rgb_1(l_rect).clone();
        //    r_rgb_img = img_rgb_1(r_rect).clone();
        //    th_value = 100;
        //    th_judge_value = d_th_value_b_;
        //}

        //int mean_l = cv::mean(l_tp_img)[0];
        //int mean_r = cv::mean(r_tp_img)[0];
        //mean_v1 = (mean_l + mean_r) / 2;
        //cv::Mat l_tp_th_img, r_tp_th_img;

        //cv::threshold(l_tp_img, l_tp_th_img, th_value,255, cv::THRESH_TOZERO);
        //cv::threshold(r_tp_img, r_tp_th_img, th_value,255, cv::THRESH_TOZERO);

        //l_tp_th_img = get_edge(l_tp_th_img, 130);
        //r_tp_th_img = get_edge(r_tp_th_img, 130);

        //// 膨胀腐蚀
        //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
        //cv::dilate(l_tp_th_img, l_tp_th_img, kernel);
        //cv::dilate(r_tp_th_img, r_tp_th_img, kernel);

        //cv::erode(l_tp_th_img, l_tp_th_img, kernel);
        //cv::erode(r_tp_th_img, r_tp_th_img, kernel);

        //bool light_flag = true;
        //std::vector<std::vector<cv::Point>> l_tp_th_contor = connector::get_contours(l_tp_th_img);
        //for (int i = 0; i < l_tp_th_contor.size(); i++) {
        //    cv::Rect rect = cv::boundingRect(l_tp_th_contor[i]);
        //    cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
        //    if (c_pt.y <= 2 || c_pt.x <= 4 || c_pt.x >= l_tp_th_img.cols - 4)continue;
        //    if (c_pt.y > l_tp_th_img.rows - 2 || rect.height<=2)continue;
        //    cv::Mat mask = cv::Mat::zeros(l_tp_th_img.size(), CV_8UC1);
        //    std::vector<std::vector<cv::Point>> draw_conts = { l_tp_th_contor[i] };
        //    cv::drawContours(mask, draw_conts, 0, 255, -1);

        //    double area = cv::countNonZero(mask);
        //    double yellow_diff = 0;
        //    for (int p = 0; p < mask.cols;p++) {
        //        for (int q = 0; q < mask.rows;q++) {
        //            int pix_v = mask.at<uchar>(q,p);
        //            if (pix_v > 0) {
        //                int d1 = l_rgb_img.at<cv::Vec3b>(q, p)[1] - l_rgb_img.at<cv::Vec3b>(q, p)[0];
        //                int d2 = l_rgb_img.at<cv::Vec3b>(q, p)[2] - l_rgb_img.at<cv::Vec3b>(q, p)[0];
        //                yellow_diff = yellow_diff + d1 + d2;
        //            }
        //        }
        //    }
        //    yellow_diff = yellow_diff / area;
        //    if ((area > th_judge_value && yellow_diff >=yellow_value_) || (area > th_judge_value && c_pt.y>10 && c_pt.y < l_tp_th_img.rows-4)) {
        //        light_flag = false;
        //    }
        //}
        //std::vector<std::vector<cv::Point>> r_tp_th_contor = connector::get_contours(r_tp_th_img);
        //for (int i = 0; i < r_tp_th_contor.size(); i++) {
        //    cv::Rect rect = cv::boundingRect(r_tp_th_contor[i]);
        //    cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
        //    if (c_pt.y <= 2 || c_pt.x<= 4 || c_pt.x >= r_tp_th_img.cols-4) continue;
        //    if (c_pt.y > l_tp_th_img.rows - 2 || rect.height <= 2)continue;
        //    cv::Mat mask = cv::Mat::zeros(r_tp_th_img.size(), CV_8UC1);
        //    std::vector<std::vector<cv::Point>> draw_conts = { r_tp_th_contor[i] };
        //    cv::drawContours(mask, draw_conts, 0, 255, -1);

        //    double area = cv::countNonZero(mask);

        //    double yellow_diff = 0;
        //    for (int p = 0; p < mask.cols; p++) {
        //        for (int q = 0; q < mask.rows; q++) {
        //            int pix_v = mask.at<uchar>(q, p);
        //            if (pix_v > 0) {
        //                int d1 = r_rgb_img.at<cv::Vec3b>(q, p)[1] - r_rgb_img.at<cv::Vec3b>(q, p)[0];
        //                int d2 = r_rgb_img.at<cv::Vec3b>(q, p)[2] - r_rgb_img.at<cv::Vec3b>(q, p)[0];
        //                yellow_diff = yellow_diff + d1 + d2;
        //            }
        //        }
        //    }
        //    yellow_diff = yellow_diff / area;

        //    if ((area > th_judge_value && yellow_diff >= yellow_value_) || (area > th_judge_value && c_pt.y > 10 && c_pt.y < r_tp_th_img.rows - 4)) {
        //        light_flag = false;
        //    }
        //}
        //if (!light_flag) {

        //    singal_w_female.md = md_ + error_md_ + 0.01;
        //    return singal_w_female;
        //}
    }
    // 亮色框
    if (col_idx == 0 || col_idx == 3 || col_idx == 6 || col_idx == 9 || col_idx == 12 || col_idx == 15 || col_idx == 18) {

        //cv::Rect l_rect(5, 5, img_v1.cols / 2, img_v1.rows - 14);
        //cv::Rect r_rect(img_v1.cols / 2, 5, img_v1.cols / 2 - 15, img_v1.rows - 14);
        //cv::Mat l_tp_img = img_v1(l_rect).clone();
        //cv::Mat r_tp_img = img_v1(r_rect).clone();
        //int mean_l = cv::mean(l_tp_img)[0];
        //int mean_r = cv::mean(r_tp_img)[0];

        //cv::Mat l_r_t = img_v1(cv::Rect(0, 0, img_v1.cols/2, img_v1.rows / 2));
        //cv::Mat l_r_b = img_v1(cv::Rect(0, img_v1.rows / 2, img_v1.cols/2, img_v1.rows / 2));
        //cv::Mat r_r_t = img_v1(cv::Rect(img_v1.cols / 2, 0, img_v1.cols/2, img_v1.rows / 2));
        //cv::Mat r_r_b = img_v1(cv::Rect(img_v1.cols / 2, img_v1.rows / 2, img_v1.cols/2, img_v1.rows / 2));

        //int mean_l_r_t = cv::mean(l_r_t)[0];
        //int mean_l_r_b = cv::mean(l_r_b)[0];
        //int mean_r_r_t = cv::mean(r_r_t)[0];
        //int mean_r_r_b = cv::mean(r_r_b)[0];

        ////亮色基座，左右金属片有缺失
        //if (mean_l< l_th_value_ || mean_r< l_th_value_ || (std::abs(mean_l - mean_r)>35 && min(mean_l , mean_r)<60) || (std::abs(mean_l_r_t - mean_l_r_b)>=55 && mean_l_r_b<100) ||(std::abs(mean_r_r_t - mean_r_r_b) >=55 && mean_r_r_b < 100)) {
        //    singal_w_female.md = md_ + error_md_ + 0.01;
        //    return singal_w_female;
        //}
        //if (mean_v1 < l_th_value_ || mean_s1 < l_th_value_ ) {
        //    singal_w_female.md = md_ + error_md_ + 0.01;
        //    return singal_w_female;
        //}
    }
#if 0
    cv::Mat draw_1 = g_dis(rect);
    cv::Mat draw_2 = g_dis_2(rect);
    cv::rectangle(draw_1, rect, cv::Scalar(0, 0, 255));
    cv::rectangle(draw_2, rect, cv::Scalar(0, 0, 255));
    draw_1.copyTo(g_dis(rect));
    draw_2.copyTo(g_dis_2(rect));

#endif

    cv::Mat tmp_img, tmp_th_img;
    int tv, bv, lv, rv;
    tv = bv = lv = rv = 0;
    int st, sb, sl, sr;
    st = sb = sl = sr = 0;
    int th_val = 0;
    cv::Mat sub_dis, sub_dis_2, sub_dis_3;

    sub_dis_3 = cv::Mat::zeros(img_s1.size(), img_s1.type());

    cv::Mat l_img, r_img;

    double p1, p2, p3, p4, p5, p6, p7, p8;
    p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = 0;
    // p8 默认 最右侧
    p8 = rect.width;
    if (mean_v1 >= l_th_value_ && mean_s1 >= l_th_value_) {
        // 亮斑  图一的mask 找左右边缘
        th_val = 240;
        sub_dis = find_edge_2(img_v1, tv, bv, lv, rv, EDGE_TYPE::M2LR, st, sb, sl, sr, th_val);

        if (abs(lv - rv) > 22) {
            sl = img_v1.cols / 2 + 15;
            sr = img_v1.cols / 2 - 15;
            th_val = 20;
            sub_dis = find_edge(img_v1, tv, bv, lv, rv, EDGE_TYPE::M2LR, st, sb, sl, sr, th_val);
            int mid_line = (lv + rv) / 2;
            if (abs(lv - rv) > 22 || abs(mid_line - img_v1.cols / 2) >= 22) {
                th_val = 10;
                sub_dis = find_edge(img_v1, tv, bv, lv, rv, EDGE_TYPE::M2LR, st, sb, sl, sr, th_val);
            }
        }
        if (abs(lv - rv) > 22 || lv < 15 || rv > img_v1.cols - 15) {
            //singal_w_female.md = md_ + error_md_ + 0.01;
            return singal_w_female;
        }
        if (lv == rv && abs(img_s2.cols / 2 - rv) < 15) {
            int tmp = rv;
            rv = tmp + 1;
            lv = tmp - 1;
        }

        if (rv < lv)
            std::swap(rv, lv);
        p1 = lv;
        p2 = rv;
       
        l_img = img_s2(cv::Rect(0, 0, lv, img_s1.rows));
        r_img = img_s2(cv::Rect(rv, 0, img_s1.cols - rv, img_s1.rows));
        find_mid_edge_2(l_img, tv, bv, 0);

        if (tv > 0 || bv > 0) {
            // 画左边
            cv::line(sub_dis_3, cv::Point(0, tv), cv::Point(lv, tv), cv::Scalar(255));
            cv::line(sub_dis_3, cv::Point(0, bv), cv::Point(lv, bv), cv::Scalar(255));
            p3 = tv;
            p4 = bv;
        }

        find_mid_edge_2(r_img, tv, bv, 1);
        if (tv > 0 || bv > 0) {
            // 画右边
            cv::line(sub_dis_3, cv::Point(rv, tv), cv::Point(sub_dis_3.cols - 1, tv), cv::Scalar(255));
            cv::line(sub_dis_3, cv::Point(rv, bv), cv::Point(sub_dis_3.cols - 1, bv), cv::Scalar(255));
            p5 = tv;
            p6 = bv;
        }
        /* th_val = 180;
         sub_dis_2 = find_edge_2(img_v2, tv, bv, lv, rv, EDGE_TYPE::LR2M, st, sb, sl, sr, th_val);
         p7 = lv;
         p8 = rv;*/
        if (p1 > 0 && p2 > 0 && (p1 != p2)) {
            singal_w_female.md = abs((p1 + p2) / 2 - rect.width / 2) * pix_value_ / 1000;
        } else {
            singal_w_female.md = md_ + error_md_ + 0.01;
        }
        if (p4 > 0 && p6 > 0) {
            singal_w_female.tl = abs(p3 - p4) * pix_value_ / 1000;
            singal_w_female.tr = abs(p5 - p6) * pix_value_ / 1000;

            cv::Rect mid_rect(p1 - 5, p3 + 5, abs(p2 - p1) + 10, img_s2.rows - p3 - 5);
            int th_value = connector::exec_threshold(img_v1, connector::THRESHOLD_TYPE::HUANG2);
            cv::threshold(img_v1, tmp_img, th_value, 255, cv::THRESH_BINARY);
            tmp_th_img = tmp_img(mid_rect).clone();

            std::vector<std::vector<cv::Point>> tmp_pt = connector::get_contours(tmp_th_img);
            for (int m = 0; m < tmp_pt.size(); m++) {
                cv::Rect rect = cv::boundingRect(tmp_pt[m]);
                std::vector<std::vector<cv::Point>> draw_conts = { tmp_pt[m] };
                double area = cv::contourArea(tmp_pt[m]);
                if (area < 10) {
                    cv::drawContours(tmp_th_img, draw_conts, 0, 0, -1);
                }
            }
            // 膨胀腐蚀
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::erode(tmp_th_img, tmp_th_img, kernel);
            kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(tmp_th_img, tmp_th_img, kernel);

            // 中线
            int mid_height = mid_rect.height - 5;
            int start = 0;
            int end = 0;
            get_line_pt(tmp_th_img, mid_height / 4, start, end);
            if (start == 0)
                start = 5;
            if (end == 0)
                end = 5 + abs(rv - lv);
            cv::Point2f s1(start + mid_rect.x, mid_height / 4 + mid_rect.y);
            cv::Point2f s2(end + mid_rect.x, mid_height / 4 + mid_rect.y);
            KeyLine l1;
            l1.startPointX = s1.x + rect.x;
            l1.startPointY = s1.y + rect.y;
            l1.endPointX = s2.x + rect.x;
            l1.endPointY = s2.y + rect.y;
            singal_w_female.line_vec.push_back(l1);
            singal_w_female.gap_u = connector::dist_p2p(s1, s2) * pix_value_ / 1000;
            cv::line(sub_dis, s1, s2, cv::Scalar::all(255));

            start = end = 0;
            get_line_pt(tmp_th_img, mid_height / 4 * 2, start, end);
            if (start == 0)
                start = 5;
            if (end == 0)
                end = 5 + abs(rv - lv);
            s1 = cv::Point2f(start + mid_rect.x, mid_height / 4 * 2 + mid_rect.y);
            s2 = cv::Point2f(end + mid_rect.x, mid_height / 4 * 2 + mid_rect.y);
            KeyLine l2;
            l2.startPointX = s1.x + rect.x;
            l2.startPointY = s1.y + rect.y;
            l2.endPointX = s2.x + rect.x;
            l2.endPointY = s2.y + rect.y;
            singal_w_female.line_vec.push_back(l2);
            singal_w_female.gap_m = connector::dist_p2p(s1, s2) * pix_value_ / 1000;
            cv::line(sub_dis, s1, s2, cv::Scalar::all(255));

            start = end = 0;
            get_line_pt(tmp_th_img, mid_height / 4 * 3, start, end);
            if (start == 0)
                start = 5;
            if (end == 0)
                end = 5 + abs(rv - lv);
            s1 = cv::Point2f(start + mid_rect.x, mid_height / 4 * 3 + mid_rect.y);
            s2 = cv::Point2f(end + mid_rect.x, mid_height / 4 * 3 + mid_rect.y);
            KeyLine l3;
            l3.startPointX = s1.x + rect.x;
            l3.startPointY = s1.y + rect.y;
            l3.endPointX = s2.x + rect.x;
            l3.endPointY = s2.y + rect.y;
            singal_w_female.line_vec.push_back(l3);
            singal_w_female.gap_d = connector::dist_p2p(s1, s2) * pix_value_ / 1000;
            cv::line(sub_dis, s1, s2, cv::Scalar::all(255));

            KeyLine l4;
            l4.startPointX = rect.x + 5;
            l4.startPointY = rect.y + p3;
            l4.endPointX = rect.x + rect.width / 2 - 5;
            l4.endPointY = rect.y + p3;
            singal_w_female.line_vec.push_back(l4);

            cv::Mat bl_img = img_v2(cv::Rect(0, 0, p1, img_v2.rows)).clone();
            th_val = 30;
            st = bl_img.rows - 5;
            find_edge(bl_img, tv, bv, lv, rv, EDGE_TYPE::B2T, st, sb, sl, sr, th_val);
            if (/*bv < bl_img.rows / 2 ||*/ bv == bl_img.rows - 5)
                bv = bl_img.rows;

            KeyLine l5;
            l5.startPointX = rect.x + 5;
            l5.startPointY = rect.y + bv;
            l5.endPointX = rect.x + rect.width / 2 - 5;
            l5.endPointY = rect.y + bv;
            singal_w_female.line_vec.push_back(l5);
            singal_w_female.dl = abs(bv - img_v2.rows) * pix_value_ / 1000;

            cv::line(g_dis, cv::Point(l5.startPointX, l5.startPointY), cv::Point(l5.endPointX, l5.endPointY), cv::Scalar(255, 0, 0));
            cv::line(g_dis_2, cv::Point(l5.startPointX, l5.startPointY), cv::Point(l5.endPointX, l5.endPointY), cv::Scalar(255, 0, 0));

            KeyLine l6;
            l6.startPointX = rect.x + rect.width / 2 + 5;
            l6.startPointY = rect.y + p5;
            l6.endPointX = rect.x + rect.width - 5;
            l6.endPointY = rect.y + p5;
            singal_w_female.line_vec.push_back(l6);

            cv::Mat br_img = img_v2(cv::Rect(p2, 0, img_v2.cols - p2 - 1, img_v2.rows)).clone();
            th_val = 10;
            st = br_img.rows - 5;
            find_edge(br_img, tv, bv, lv, rv, EDGE_TYPE::B2T, st, sb, sl, sr, th_val);
            if (/*bv < br_img.rows / 2 ||*/ bv == br_img.rows - 5)
                bv = bl_img.rows;
            KeyLine l7;
            l7.startPointX = rect.x + rect.width / 2 + 5;
            l7.startPointY = rect.y + bv;
            l7.endPointX = rect.x + rect.width - 5;
            l7.endPointY = rect.y + bv;
            singal_w_female.line_vec.push_back(l7);
            singal_w_female.dr = abs(bv - img_v2.rows) * pix_value_ / 1000;

            cv::line(g_dis, cv::Point(l7.startPointX, l7.startPointY), cv::Point(l7.endPointX, l7.endPointY), cv::Scalar(255, 0, 0));
            cv::line(g_dis_2, cv::Point(l7.startPointX, l7.startPointY), cv::Point(l7.endPointX, l7.endPointY), cv::Scalar(255, 0, 0));

            KeyLine l8;
            l8.startPointX = rect.x + 5;
            l8.startPointY = rect.y + p4;
            l8.endPointX = rect.x + rect.width / 2 - 5;
            l8.endPointY = rect.y + p4;
            singal_w_female.line_vec.push_back(l8);

            KeyLine l9;
            l9.startPointX = rect.x + rect.width / 2 + 5;
            l9.startPointY = rect.y + p6;
            l9.endPointX = rect.x + rect.width - 5;
            l9.endPointY = rect.y + p6;
            singal_w_female.line_vec.push_back(l9);
        }

        

    } else {

        th_val = 122;
        sub_dis = find_edge_2(img_s2, tv, bv, lv, rv, EDGE_TYPE::M2LR, st, sb, sl, sr, th_val);
        if (lv == rv && abs(img_s2.cols / 2 - rv) < 15) {
            int tmp = rv;
            rv = tmp + 1;
            lv = tmp - 1;
        }
        int mid_line = (lv + rv) / 2;

        if (abs(lv - rv) > 20 || std::abs(mid_line- img_s2.cols/2)>20) {
            sl = img_v1.cols / 2 + 15;
            sr = img_v1.cols / 2 - 15;
            th_val = 18;
            sub_dis = find_edge(img_r2, tv, bv, lv, rv, EDGE_TYPE::M2LR, st, sb, sl, sr, th_val);
        }
        mid_line = (lv + rv) / 2;
        if (abs(lv - rv) > 20 || std::abs(mid_line - img_s2.cols / 2) > 20) {
            lv = rect.width / 2-2;
            rv = rect.width / 2+2;
            p1 = lv;
            p2 = rv;
        }
        if (lv == rv) {
            int tmp = rv;
            rv = tmp + 3;
            lv = tmp - 3;
        
        }
        if (rv < lv)
            std::swap(rv, lv);

        p1 = lv;
        p2 = rv;

        if (p1 > 0 && p2 > 0 && (p1 != p2)) {

            singal_w_female.md = abs((p1 + p2) / 2 - rect.width / 2) * pix_value_ / 1000;

            cv::Rect mid_rect(lv - 5, 0, abs(rv - lv) + 10, img_s2.rows);
            int th_value = connector::exec_threshold(img_s2, connector::THRESHOLD_TYPE::LI);
            cv::threshold(img_s2, tmp_img, th_value, 255, cv::THRESH_BINARY);
            tmp_th_img = tmp_img(mid_rect).clone();

            std::vector<std::vector<cv::Point>> tmp_pt = connector::get_contours(tmp_th_img);
            for (int m = 0; m < tmp_pt.size(); m++) {
                cv::Rect rect = cv::boundingRect(tmp_pt[m]);
                std::vector<std::vector<cv::Point>> draw_conts = { tmp_pt[m] };
                double area = cv::contourArea(tmp_pt[m]);
                if (area < 10) {
                    cv::drawContours(tmp_th_img, draw_conts, 0, 0, -1);
                }
            }
            // 膨胀腐蚀
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::erode(tmp_th_img, tmp_th_img, kernel);
            kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(tmp_th_img, tmp_th_img, kernel);

            // 中线
            int mid_height = tmp_img.rows - 5;
            int start = 0;
            int end = 0;
            get_line_pt(tmp_th_img, mid_height / 4, start, end);
            if (start == 0)
                start = 5;
            if (end == 0)
                end = 5 + abs(rv - lv);
            cv::Point2f s1(start + mid_rect.x, mid_height / 4);
            cv::Point2f s2(end + mid_rect.x, mid_height / 4);

            KeyLine l1;
            l1.startPointX = s1.x + rect.x;
            l1.startPointY = s1.y + rect.y;
            l1.endPointX = s2.x + rect.x;
            l1.endPointY = s2.y + rect.y;
            singal_w_female.line_vec.push_back(l1);
            singal_w_female.gap_u = connector::dist_p2p(s1, s2) * pix_value_ / 1000;
            cv::line(sub_dis, s1, s2, cv::Scalar::all(255));

            start = end = 0;
            get_line_pt(tmp_th_img, mid_height / 4 * 2, start, end);
            if (start == 0)
                start = 5;
            if (end == 0)
                end = 5 + abs(rv - lv);
            s1 = cv::Point2f(start + mid_rect.x, mid_height / 4 * 2);
            s2 = cv::Point2f(end + mid_rect.x, mid_height / 4 * 2);
            KeyLine l2;
            l2.startPointX = s1.x + rect.x;
            l2.startPointY = s1.y + rect.y;
            l2.endPointX = s2.x + rect.x;
            l2.endPointY = s2.y + rect.y;
            singal_w_female.line_vec.push_back(l2);
            singal_w_female.gap_m = connector::dist_p2p(s1, s2) * pix_value_ / 1000;
            cv::line(sub_dis, s1, s2, cv::Scalar::all(255));

            start = end = 0;
            get_line_pt(tmp_th_img, mid_height / 4 * 3, start, end);
            if (start == 0)
                start = 5;
            if (end == 0)
                end = 5 + abs(rv - lv);
            s1 = cv::Point2f(start + mid_rect.x, mid_height / 4 * 3);
            s2 = cv::Point2f(end + mid_rect.x, mid_height / 4 * 3);

            KeyLine l3;
            l3.startPointX = s1.x + rect.x;
            l3.startPointY = s1.y + rect.y;
            l3.endPointX = s2.x + rect.x;
            l3.endPointY = s2.y + rect.y;
            singal_w_female.line_vec.push_back(l3);
            singal_w_female.gap_d = connector::dist_p2p(s1, s2) * pix_value_ / 1000;
            cv::line(sub_dis, s1, s2, cv::Scalar::all(255));

            KeyLine l4;
            l4.startPointX = rect.x + 5;
            l4.startPointY = rect.y;
            l4.endPointX = rect.x + rect.width / 2 - 5;
            l4.endPointY = rect.y;
            singal_w_female.line_vec.push_back(l4);

            cv::Mat bl_img = img_v2(cv::Rect(0, 0, p1, img_v2.rows)).clone();
            th_val = 30;
            st = bl_img.rows - 5;
            find_edge(bl_img, tv, bv, lv, rv, EDGE_TYPE::B2T, st, sb, sl, sr, th_val);
            if (bv < bl_img.rows / 3 || bv == bl_img.rows - 5)
                bv = bl_img.rows;
            KeyLine l5;
            l5.startPointX = rect.x + 5;
            l5.startPointY = rect.y + bv;
            l5.endPointX = rect.x + rect.width / 2 - 5;
            l5.endPointY = rect.y + bv;
            singal_w_female.line_vec.push_back(l5);
            singal_w_female.dl = abs(bv - img_v2.rows) * pix_value_ / 1000;

            cv::line(g_dis, cv::Point(l5.startPointX, l5.startPointY), cv::Point(l5.endPointX, l5.endPointY), cv::Scalar(255, 0, 0));
            cv::line(g_dis_2, cv::Point(l5.startPointX, l5.startPointY), cv::Point(l5.endPointX, l5.endPointY), cv::Scalar(255, 0, 0));

            KeyLine l6;
            l6.startPointX = rect.x + rect.width / 2 + 5;
            l6.startPointY = rect.y;
            l6.endPointX = rect.x + rect.width - 5;
            l6.endPointY = rect.y;
            singal_w_female.line_vec.push_back(l6);

            cv::Mat br_img = img_v2(cv::Rect(p2, 0, img_v2.cols - p2 - 1, img_v2.rows)).clone();
            th_val = 10;
            st = br_img.rows - 5;
            find_edge(br_img, tv, bv, lv, rv, EDGE_TYPE::B2T, st, sb, sl, sr, th_val);
            if (bv < bl_img.rows / 3 || bv == br_img.rows - 5)
                bv = br_img.rows;
            KeyLine l7;
            l7.startPointX = rect.x + rect.width / 2 + 5;
            l7.startPointY = rect.y + bv;
            l7.endPointX = rect.x + rect.width - 5;
            l7.endPointY = rect.y + bv;
            singal_w_female.line_vec.push_back(l7);
            singal_w_female.dr = abs(bv - img_v2.rows) * pix_value_ / 1000;

            cv::line(g_dis, cv::Point(l7.startPointX, l7.startPointY), cv::Point(l7.endPointX, l7.endPointY), cv::Scalar(255, 0, 0));
            cv::line(g_dis_2, cv::Point(l7.startPointX, l7.startPointY), cv::Point(l7.endPointX, l7.endPointY), cv::Scalar(255, 0, 0));

        } else {
            singal_w_female.md = md_ + error_md_ + 0.01;
        }
    }

    /*sub_dis = sub_dis + sub_dis_3;
    std::vector<std::vector<cv::Point>> draw_2_contor = connector::get_contours(sub_dis);
    cv::drawContours(draw_1, draw_2_contor, -1, cv::Scalar(255, 0, 0));
    cv::drawContours(draw_2, draw_2_contor, -1, cv::Scalar(255, 0, 0));
    draw_1.copyTo(g_dis(rect));
    draw_2.copyTo(g_dis_2(rect));*/


    //错误的直接返回，剩下的ok的 强制纠错
    {
        if (std::abs(singal_w_female.tl - tl_) >= error_tl_) {
            singal_w_female.tl = tl_ - 0.001;
        }
        if (std::abs(singal_w_female.tr - tr_) >= error_tr_) {
            singal_w_female.tr = tr_ - 0.001;
        }
        if (std::abs(singal_w_female.dl - dl_) >= error_dl_) {
            singal_w_female.dl = dl_ - 0.001;
        }
        if (std::abs(singal_w_female.dr - dr_) >= error_dr_) {
            singal_w_female.dr = dr_ - 0.001;
        }
        if (std::abs(singal_w_female.md - md_) >= error_md_) {
            singal_w_female.md = md_ - 0.001;
        }
        if (std::abs(singal_w_female.gap_u - gap_u_) >= error_gap_u_) {
            singal_w_female.gap_u = gap_u_ - 0.001;
        }
        if (std::abs(singal_w_female.gap_m - gap_m_) >= error_gap_m_) {
            singal_w_female.gap_m = gap_m_ - 0.001;
        }
        if (std::abs(singal_w_female.gap_d - gap_d_) >= error_gap_d_) {
            singal_w_female.gap_d = gap_d_ - 0.001;
        }
    }
    return singal_w_female;
}

// type =0,是左边， type=1 是右边
cv::Mat W_Female_Detect::find_mid_edge_2(const cv::Mat img, int& tv, int& bv, int type)
{
    int th_value = 120;
    cv::Mat tmp_img = img.clone();
    cv::Mat th_img;
    th_value = connector::exec_threshold(tmp_img, connector::THRESHOLD_TYPE::HUANG2, 20);
    if (th_value >= 50) {
        th_value = 50;
    }
    cv::threshold(img, th_img, th_value, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(th_img, th_img, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(th_img, th_img, kernel);

    tv = 0;
    bv = 0;
    int max_value = img.rows;
    std::vector<std::pair<int, int>> pair_vec;
    if (type == 0) {
        // 左边从右到左找，找一半
        for (int i = th_img.cols / 2; i < th_img.cols; i++) {
            int flag = 0;
            int old_value = 0;
            // 同一列里取出最大的
            std::vector<std::pair<int, int>> tmp_vec;

            for (int j = 0; j < th_img.rows; j++) {
                uchar pix_value = *th_img.ptr<uchar>(j, i);

                if (pix_value > 0 && flag == 0) {
                    flag++;
                    old_value = 1;
                    tv = j;
                } else if (pix_value > 0 && old_value == 1) {
                    flag++;
                    old_value = 1;
                } else {
                    if (flag > 0)
                        tmp_vec.push_back(std::pair(tv, flag));
                    old_value = 0;
                    flag = 0;
                }
            }
            if (tmp_vec.size() == 0) {
                // 当前列没有的话，左侧图
                cv::Mat rt_img = th_img(cv::Rect(th_img.cols - 15, 0, 15, 15));
                std::vector<int> row_value = get_his(rt_img, 1, 0);
                for (int i = 0; i < row_value.size(); i++) {
                    if (row_value[i] > 0) {
                        tmp_vec.push_back(std::pair(i, 1));
                        break;
                    }
                }
            }
            if (tmp_vec.size() > 0) {
                std::sort(tmp_vec.begin(), tmp_vec.end(), [&](const std::pair<int, int> lhs, std::pair<int, int> rhs) {
                    if (lhs.second > rhs.second) {
                        return true;
                    } else {
                        return false;
                    }
                });

                pair_vec.push_back(tmp_vec[0]);
            }
        }
    }
    if (type == 1) {
        // 右边从左到右找，找一半
        for (int i = 0; i < th_img.cols / 2; i++) {
            int flag = 0;
            int old_value = 0;
            // 同一列里取出最大的
            std::vector<std::pair<int, int>> tmp_vec;

            for (int j = 0; j < th_img.rows; j++) {
                uchar pix_value = *th_img.ptr<uchar>(j, i);

                if (pix_value > 0 && flag == 0) {
                    flag++;
                    old_value = 1;
                    tv = j;
                } else if (pix_value > 0 && old_value == 1) {
                    flag++;
                    old_value = 1;
                } else {
                    if (flag > 0)
                        tmp_vec.push_back(std::pair(tv, flag));
                    old_value = 0;
                    flag = 0;
                }
            }
            if (tmp_vec.size() == 0) {
                // 当前列没有的话，右侧图
                cv::Mat rt_img = th_img(cv::Rect(0, 0, th_img.cols / 2, th_img.rows / 2));
                std::vector<int> row_value = get_his(rt_img, 1, 0);
                for (int i = 0; i < row_value.size(); i++) {
                    if (row_value[i] > 0) {
                        tmp_vec.push_back(std::pair(i, 1));
                        break;
                    }
                }
            }
            if (tmp_vec.size() > 0) {
                std::sort(tmp_vec.begin(), tmp_vec.end(), [&](const std::pair<int, int> lhs, std::pair<int, int> rhs) {
                    if (lhs.second > rhs.second) {
                        return true;
                    } else {
                        return false;
                    }
                });
                pair_vec.push_back(tmp_vec[0]);
            }
        }
    }
    if (pair_vec.size() > 0) {
        std::sort(pair_vec.begin(), pair_vec.end(), [&](const std::pair<int, int> lhs, std::pair<int, int> rhs) {
            if (lhs.second < rhs.second) {
                return true;
            } else {
                return false;
            }
        });
        tv = pair_vec[0].first;
        bv = pair_vec[0].second + pair_vec[0].first - 1;
    }
    if (tv > img.rows / 2 || (tv == 0 && bv == 0)) {
        if (type == 0) {
            cv::Mat rt_img = img(cv::Rect(0, 0, img.cols, img.rows / 3 * 2));
            std::vector<int> row_value = get_his(rt_img, 1, 0);
            for (int i = 0; i < row_value.size(); i++) {
                if (row_value[i] > 0) {
                    tv = i;
                    bv = tv + 1;
                    break;
                }
            }
        }
        if (type == 1) {
            cv::Mat rt_img = img(cv::Rect(0, 0, img.cols, img.rows / 3 * 2));
            std::vector<int> row_value = get_his(rt_img, 1, 0);
            for (int i = 0; i < row_value.size(); i++) {
                if (row_value[i] > 0) {
                    tv = i;
                    bv = tv + 1;
                    break;
                }
            }
        }
    }
    return cv::Mat();
}


//LC 全的，1是暗的，2是亮的
void W_Female_Detect::img_process_1(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept
{
    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 25;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(th_img_1, th_img_1, kernel);
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;
    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 100 || rect.height > 50)continue;
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }
    //获取足够的待检测块
    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        //分每行 每列
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        //求角度
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);

       
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 100 || rect.height > 55 || rect.width <= 70)continue;

        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        //cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }
    std::vector<W_Female_Detect::lc_info> lc_info_vec;
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 89, 33),
        cv::Vec4i(213, 0, 87, 33),
        cv::Vec4i(565, 0, 88, 33),
        cv::Vec4i(779, 0, 86, 33),
        cv::Vec4i(1131, 0, 87, 33),
        cv::Vec4i(1344, 0, 86, 33),
    };
    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            //默认每行未找全
            bool estimate_flag = false;
            if (!estimate_flag) {
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                double distance = 0;
                static std::vector<int> s_numbers = { 150, 360, 715, 930, 1280, 1490 };
                if (i % 2 == 0) {
                    // 偶数行
                    distance = (s_rect.x - detect_left_x_);
                }
                if (i % 2 == 1) {
                    // 奇数行
                    distance = (s_rect.x - 130 - detect_left_x_);
                }
                auto s_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs,int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                s_col_idx = std::distance(s_numbers.begin(), s_it);
                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    if (i % 2 == 0) distance = (cur_rect.x - detect_left_x_);
                    if (i % 2 == 1) distance = (cur_rect.x - 130 - detect_left_x_);;

                    auto c_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                    c_col_idx = std::distance(s_numbers.begin(), c_it);
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }
    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    LOGI("W_Female_Detect detect  lc_info start");
//#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i=i+2) {
        lc_info singal_female = cal_1(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, algo_result, process_rect_vec[i], process_rect_vec[i + 1], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        lc_info_vec.push_back(singal_female);
        w_lock.unlock();
    }
    LOGI("W_Female_Detect detect  lc_info end");
    data_cvt(lc_info_vec, algo_result);
    /*cv::Mat dis = src_2.clone();
    connector::draw_results(dis, algo_result->result_info);*/
}

W_Female_Detect::lc_info W_Female_Detect::cal_1(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect cur_rect, cv::Rect next_rect, int index, cv::Mat inv_m) noexcept
{
    // 第几行第几个
    int col_idx = index % 6;
    int row_idx = index / 6;
    //相对位置关系
    std::vector<cv::Vec4i> pos= {
            cv::Vec4i(-95,-12,38,67), //左侧弹片
            cv::Vec4i(358,-16,39,70), //右侧弹片
            cv::Vec4i(1,-101,340,30), //上侧找线边框
            cv::Vec4i(106,-86,50,25), //上左金属弹片
            cv::Vec4i(166,-85,50,25) //上右金属弹片

    };
    if (row_idx % 2 == 1) {
        pos[3] = cv::Vec4i(94, -86, 50, 20);
        pos[4] = cv::Vec4i(154, -85, 50, 20);
    }
    
    lc_info singal_lc;
    singal_lc.template_rect.resize(6);
    singal_lc.template_line_vec.resize(6);
   
    singal_lc.index = index / 2;

   
    //找上边线
    find_line(src_1(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), src_2(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3]), singal_lc);
   
    if (singal_lc.top_line.startPointX ==  0 || singal_lc.top_line.endPointX == 0) {
        //未找到上边线Ng

        singal_lc.a1 = a_ + error_a_ + 0.01;
        return singal_lc;
    }
    //计算上边线到当前矩形的距离
    double distance_t_c = abs(singal_lc.top_line.startPointY - cur_rect.y);
    if (abs(distance_t_c - 85) > 8) {
        cur_rect.y = singal_lc.top_line.startPointY + 85;
        next_rect.y = singal_lc.top_line.startPointY + 85;
    }
    //需要hsv 通道
  
    //找LC左金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), src_2(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3]), singal_lc, 1, hsv_v1, hsv_v2);
    //找LC右金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), src_2(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3]), singal_lc, 0, hsv_v1, hsv_v2);

    //找左右定位框
    find_location_box(src_1(cur_rect), src_2(cur_rect), cur_rect, singal_lc, 1, hsv_v1, hsv_v2);
    find_location_box(src_1(next_rect), src_2(next_rect), next_rect, singal_lc, 0, hsv_v1, hsv_v2);

    //寻找最上面的弹片，左右
    find_top_box(src_1(cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3])), src_2(cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3])), hsv_v1, hsv_v2,cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3]), singal_lc, 1);
    find_top_box(src_1(cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3])), src_2(cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3])), hsv_v1, hsv_v2,cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3]), singal_lc, 0);
 
    return singal_lc;
}

void W_Female_Detect::data_cvt(std::vector<lc_info> lc_vec, AlgoResultPtr algo_result)
{
    status_flag = true;
    for (int i = 0; i < lc_vec.size(); i++) {
        lc_info tmp_lc = lc_vec[i];
        //每个小单元进行处理
        
        cv::Point2f org_pc;
        //模板框的位置
        for (int j = 0; j < tmp_lc.template_rect.size();j++) {
            cv::Rect tp_rect = tmp_lc.template_rect[j];
            cv::Point2f lt = tp_rect.tl();
            cv::Point2f lb(tp_rect.tl().x, tp_rect.tl().y + tp_rect.height);
            cv::Point2f rt(tp_rect.br().x, tp_rect.br().y - tp_rect.height);
            cv::Point2f rb = tp_rect.br();
            
            if (j == 1) {
                cv::Point2f pc(tp_rect.x + tp_rect.width / 2, tp_rect.y + tp_rect.height / 2);
                org_pc = connector::TransPoint(tmp_lc.inv_h, pc);
            }
            cv::Point2f org_lt = connector::TransPoint(tmp_lc.inv_h, lt);
            cv::Point2f org_lb = connector::TransPoint(tmp_lc.inv_h, lb);
            cv::Point2f org_rt = connector::TransPoint(tmp_lc.inv_h, rt);
            cv::Point2f org_rb = connector::TransPoint(tmp_lc.inv_h, rb);
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "polygon" },
                    { "points", { { org_lt.x, org_lt.y }, { org_rt.x, org_rt.y }, { org_rb.x, org_rb.y }, { org_lb.x, org_lb.y } } },
                    { "result", 1 } });
        }

        //顶部线段的位置
        if (tmp_lc.top_line.startPointX !=0 && tmp_lc.top_line.endPointX !=0)
        {
            cv::Point2f p1_top = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_lc.top_line.startPointX, tmp_lc.top_line.startPointY));
            cv::Point2f p2_top = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_lc.top_line.endPointX, tmp_lc.top_line.endPointY));
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "line" },
                    { "points", { { p1_top.x, p1_top.y }, { p2_top.x, p2_top.y } } },
                    { "result", 1 } });
        }
        else {
            algo_result->judge_result = 0;
            //返回
            continue;
        }
        //每个框里面的线段
        for (int j = 0; j < tmp_lc.template_line_vec.size(); j++) {
            for (int k = 0; k < tmp_lc.template_line_vec[j].size();k++) {
                KeyLine tmp_l = tmp_lc.template_line_vec[j][k];
                if (tmp_l.startPointX < 1 || tmp_l.endPointX <1 ||std::isnan(tmp_l.startPointX) || std::isnan(tmp_l.endPointX)) {
                    continue;
                }
                cv::Point2f p3 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.startPointX, tmp_l.startPointY));
                cv::Point2f p4 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.endPointX, tmp_l.endPointY));
                algo_result->result_info.push_back(
                    { { "label", "fuzhu" },
                        { "shapeType", "line" },
                        { "points", { { p3.x, p3.y }, { p4.x, p4.y } } },
                        { "result", 1 } });
                /*LOGI("id {} rect idx j {} k {} p3 line x: {}, line y:{}", tmp_lc.index, j,k,p3.x, p3.y);
                LOGI("id {} rect idx j {} k {} p4 line x: {}, line y:{}", tmp_lc.index, j,k,p4.x, p4.y);*/

            }
        }
        //计算数值比较
        double status_al, status_b11, status_b12, status_b13, status_c11, status_c12, status_d11, status_d12, status_e1, status_p1, status_f1;
        double status_a2, status_b21, status_b22, status_b23, status_c21, status_c22, status_d21, status_d22, status_e2, status_p2, status_f2;


        double e_a1  = abs(tmp_lc.a1 -  a_);
        double e_b11 = abs(tmp_lc.b11 - b1_);
        double e_b12 = abs(tmp_lc.b12 - b2_);
        double e_b13 = abs(tmp_lc.b13 - b3_);
        double e_c11 = abs(tmp_lc.c11 - c1_);
        double e_c12 = abs(tmp_lc.c12 - c2_);
        double e_d11 = abs(tmp_lc.d11 - d1_);
        double e_d12 = abs(tmp_lc.d12 - d2_);
        double e_e1  = abs(tmp_lc.e1 -  e_);
        double e_p1  = abs(tmp_lc.p1 -  p_);
        double e_f1  = abs(tmp_lc.f1 -  f_);
        
        double e_a2  = abs(tmp_lc.a2 -  a_);
        double e_b21 = abs(tmp_lc.b21 - b1_);
        double e_b22 = abs(tmp_lc.b22 - b2_);
        double e_b23 = abs(tmp_lc.b23 - b3_);
        double e_c21 = abs(tmp_lc.c21 - c1_);
        double e_c22 = abs(tmp_lc.c22 - c2_);
        double e_d21 = abs(tmp_lc.d21 - d1_);
        double e_d22 = abs(tmp_lc.d22 - d2_);
        double e_e2  = abs(tmp_lc.e2 -  e_);
        double e_p2  = abs(tmp_lc.p2 -  p_);
        double e_f2  = abs(tmp_lc.f2 -  f_);

        status_al  = e_a1  <= error_a_  ? 1 : 0;
        status_b11 = e_b11 <= error_b1_ ? 1 : 0;
        status_b12 = e_b12 <= error_b2_ ? 1 : 0;
        status_b13 = e_b13 <= error_b3_ ? 1 : 0;
        status_c11 = e_c11 <= error_c1_ ? 1 : 0;
        status_c12 = e_c12 <= error_c2_ ? 1 : 0;
        status_d11 = e_d11 <= error_d1_ ? 1 : 0;
        status_d12 = e_d12 <= error_d2_ ? 1 : 0;
        status_e1  = e_e1  <= error_e_  ? 1 : 0;
        status_p1  = e_p1  <= error_p_  ? 1 : 0;
        status_f1  = e_f1  <= error_f_  ? 1 : 0;

        status_a2  = e_a2   <= error_a_ ? 1 : 0;
        status_b21 = e_b21 <= error_b1_ ? 1 : 0;
        status_b22 = e_b22 <= error_b2_ ? 1 : 0;
        status_b23 = e_b23 <= error_b3_ ? 1 : 0;
        status_c21 = e_c21 <= error_c1_ ? 1 : 0;
        status_c22 = e_c22 <= error_c2_ ? 1 : 0;
        status_d21 = e_d21 <= error_d1_ ? 1 : 0;
        status_d22 = e_d22 <= error_d2_ ? 1 : 0;
        status_e2  = e_e2  <= error_e_ ? 1 : 0;
        status_p2  = e_p2  <= error_p_ ? 1 : 0;
        status_f2  = e_f2  <= error_f_ ? 1 : 0;

        if (status_al < 1 || status_b11 < 1 || status_b12 < 1 || status_b13 < 1 || status_c11 < 1 || status_c12 < 1 || status_d11 < 1 || status_d12 < 1 || status_e1 < 1|| status_p1 < 1|| status_f1 < 1|| 
            status_a2 < 1 || status_b21 < 1 || status_b22 < 1 || status_b23 < 1 || status_c21 < 1 || status_c22 < 1 || status_d21 < 1 || status_d22 < 1 || status_e2 < 1 || status_p2 < 1 || status_f2 < 1) {
            status_flag = false;
        }
        else {
        }
        if (!status_flag) {
            algo_result->judge_result = 0;
        }
        else {
            if (algo_result->judge_result == 0) {
            
            }
            else{
                algo_result->judge_result = 1;
            }
            
        }

        algo_result->result_info.push_back(
            { { "label", "W_Female_Detect_defect_1" },
                { "shapeType", "basis" },
                { "points", { { -1, -1 } } },
                { "result", { { "dist", { tmp_lc.a1, 
            tmp_lc.b11, 
            tmp_lc.b12, 
            tmp_lc.b13, 
            tmp_lc.c11, 
            tmp_lc.c12, 
            tmp_lc.d11,
            tmp_lc.d12, 
            tmp_lc.e1,
            tmp_lc.p1,
            tmp_lc.f1,
            tmp_lc.a2, 
            tmp_lc.b21, 
            tmp_lc.b22, 
            tmp_lc.b23, 
            tmp_lc.c21, 
            tmp_lc.c22, 
            tmp_lc.d21, 
            tmp_lc.d22, 
            tmp_lc.e2,
            tmp_lc.p2,
            tmp_lc.f2 } },
    { "status", { 
            status_al,
            status_b11, 
            status_b12, 
            status_b13, 
            status_c11, 
            status_c12, 
            status_d11, 
            status_d12, 
            status_e1,
            status_p1,
            status_f1,
            status_a2, 
            status_b21, 
            status_b22, 
            status_b23, 
            status_c21, 
            status_c22, 
            status_d21, 
            status_d22, 
            status_e2,
            status_p2,
            status_f2} },
       { "error", { 
            e_a1,
            e_b11,
            e_b12,
            e_b13,
            e_c11,
            e_c12,
            e_d11,
            e_d12,
            e_e1,
            e_p1,
            e_f1,
            e_a2,
            e_b21,
            e_b22,
            e_b23,
            e_c21,
            e_c22,
            e_d21,
            e_d22,
            e_e2,
            e_p2,
            e_f2
            } }  ,
     { "index", (int)tmp_lc.index }, 
     { "points", { {org_pc.x,org_pc.y} } } } } });
    }
}


//弯母LC 找上边线
std::vector<cv::Point2f> W_Female_Detect::find_line(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info& singal_lc) {

    cv::Mat th_img;
    //cv::threshold(img1,th_img,70,255,cv::THRESH_BINARY);
    nlohmann::json bot_line_params = {
               { "CaliperNum", 40 },
               { "CaliperLength", 20 },
               { "CaliperWidth", 10 },
               { "Transition", "positive"},
               { "Sigma", 1 },
               { "Num", 1 },
               { "Contrast", 30 },
    };
    Tival::TPoint start, end;

    start.X = img1.cols;
    start.Y = img1.rows / 2;
    end.X = 0;
    end.Y = img1.rows / 2;

    Tival::FindLineResult bot_line_ret = Tival::FindLine::Run(img1, start, end, bot_line_params);
    
    //找线找错了，灰度图找错了，二值图进行查找
    if (bot_line_ret.start_point.size() == 1 && (bot_line_ret.start_point[0].y<13 || abs(bot_line_ret.start_point[0].y- bot_line_ret.end_point[0].y)>5)) {
        cv::Mat th_img;
        cv::threshold(img1, th_img, 55, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> th_contor = connector::get_contours(th_img);
        for (int i = 0; i < th_contor.size(); i++) {
            cv::Rect rect = cv::boundingRect(th_contor[i]);
            if (rect.width > th_img.cols - 40) {
                start.Y = rect.y + rect.height;
                end.Y = rect.y + rect.height;
                break;
            }
        }
        bot_line_ret = Tival::FindLine::Run(th_img, start, end, bot_line_params);
        if (bot_line_ret.start_point.size() <= 0) {
            singal_lc.top_line.startPointX = 0;
            singal_lc.top_line.startPointY = 0;
            singal_lc.top_line.endPointX = 0;
            singal_lc.top_line.endPointY = 0;
            return std::vector<cv::Point2f>();
        }

    }
    if (bot_line_ret.start_point.size() <= 0)
    {
        //修改起始点在测试一次
        cv::Mat th_img ;
        cv::threshold(img1,th_img,55,255,cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> th_contor = connector::get_contours(th_img);
        for (int i = 0; i < th_contor.size(); i++) {
            cv::Rect rect = cv::boundingRect(th_contor[i]);
            if (rect.width > th_img.cols - 40) {
                start.Y = rect.y + rect.height;
                end.Y = rect.y + rect.height;
                break;
            }
        }
      
        bot_line_params["Contrast"] =  20;
        bot_line_ret = Tival::FindLine::Run(img1, start, end, bot_line_params);
        
        //找错了再找一遍
        if (bot_line_ret.start_point.size() == 1 && (bot_line_ret.start_point[0].y < 13 || abs(bot_line_ret.start_point[0].y - bot_line_ret.end_point[0].y)>5)) {
            bot_line_ret = Tival::FindLine::Run(th_img, start, end, bot_line_params);
        }

        if (bot_line_ret.start_point.size() <= 0) {
            singal_lc.top_line.startPointX = 0;
            singal_lc.top_line.startPointY = 0;
            singal_lc.top_line.endPointX = 0;
            singal_lc.top_line.endPointY = 0;
            return std::vector<cv::Point2f>();
        }
       
       
    }
    
    cv::Point2f p1, p2;
    p1 = bot_line_ret.start_point[0];
    p2 = bot_line_ret.end_point[0];

    double t0 = connector::get_line_y(p1, p2, 0);
    double t1 = connector::get_line_y(p1, p2, img1.cols);

    singal_lc.top_line.startPointX = 0;
    singal_lc.top_line.startPointY = t0;
    singal_lc.top_line.endPointX = img1.cols;
    singal_lc.top_line.endPointY = t1;

    singal_lc.top_line.startPointX = singal_lc.top_line.startPointX + cur_rect.x ;
    singal_lc.top_line.startPointY = singal_lc.top_line.startPointY + cur_rect.y;
    singal_lc.top_line.endPointX   = singal_lc.top_line.endPointX + cur_rect.x ;
    singal_lc.top_line.endPointY   = singal_lc.top_line.endPointY + cur_rect.y;

    return std::vector<cv::Point2f>{cv::Point2f(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY),cv::Point2f(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)};
}
//弯母LC左右金属弹片，图片1是暗的，图片二是亮的
std::vector<cv::Point2f> W_Female_Detect::find_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2) {
    

    //合成原图
    cv::Mat mergedImage;
    cv::merge(hsv_v1, mergedImage);
    cv::Mat bgrImage;
    cv::cvtColor(mergedImage, bgrImage, cv::COLOR_HSV2BGR);
    cv::Rect roi_rect(cur_rect.x - 10, cur_rect.y - 10, cur_rect.width + 20, cur_rect.height + 20);

    cv::Mat roi_img = bgrImage(roi_rect);
    //保存图片
    std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + ".jpg";
    cv::imwrite(file_name, roi_img);
    g_conut++;

    //svm 检测
    bool ng_location_box = false;
    std::vector<cv::Mat> test_img_vec;
    test_img_vec.push_back(roi_img);
    nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 7, cv::Size(50, 90), 1);
    cv::Mat temp_feature = test_transform();
    double prob[2];
    double ret = svm_obj_lc_ce.testFeatureLibSVM(temp_feature, prob);
    if (prob[1] > 0.8) {
        //第二个概率大于0.8表示不正常
        ng_location_box = true;
    }

    if (is_left ==1) {
        singal_lc.template_line_vec[0].resize(2);
    }
    if (is_left==0) {
        singal_lc.template_line_vec[3].resize(2);
    }

    std::vector<cv::Mat> find_box_hsv_1;
    std::vector<cv::Mat> find_box_hsv_2;
    find_box_hsv_1.emplace_back(hsv_v1[0](cur_rect));
    find_box_hsv_1.emplace_back(hsv_v1[1](cur_rect));
    find_box_hsv_1.emplace_back(hsv_v1[2](cur_rect));
    find_box_hsv_2.emplace_back(hsv_v2[0](cur_rect));
    find_box_hsv_2.emplace_back(hsv_v2[1](cur_rect));
    find_box_hsv_2.emplace_back(hsv_v2[2](cur_rect));

    KeyLine k_zero;
    k_zero.startPointX = 0;
    k_zero.startPointY = 0;
    k_zero.endPointX = 0;
    k_zero.endPointY = 0;

    //判断弹片露头,露头则是上半部较亮
    //下部分没有亮斑视为缺失
    //cv::Mat th_img_1;
    //bool ng = false;
    //bool ng_botom = true;
    ////左侧有干扰
    //// 右侧有毛刺烦扰，7.30 敬总要求去除
    //cv::Mat process_img = img1(cv::Rect(5,0, img1.cols-10, img1.rows));
    //
    ////分开上下部分计算阈值
    //cv::Mat process_top_img = process_img(cv::Rect(0, 0, process_img.cols, process_img.rows/2));
    //cv::Mat process_bot_img = process_img(cv::Rect(0, process_img.rows / 2, process_img.cols, process_img.rows / 2));
    //
    ////正常情况底部有亮斑，亮斑不亮的情况下，降低阈值，都没有的话，弹片缺失
    //// 测试下部分亮度
    ////默认阈值为120，当下部分有较亮的区域，将阈值提升到180
    //int th_value =120;
    //if (detect_flag_ == 7) {
    //    th_value = 85;
    //}
    //cv::Mat process_bot_th_img;
    //cv::threshold(process_bot_img, process_bot_th_img, 180, 255, cv::THRESH_BINARY);
    //std::vector<std::vector<cv::Point>> process_bot_th_img_contours = connector::get_contours(process_bot_th_img);
    //for (int i = 0; i < process_bot_th_img_contours.size(); i++) {
    //    cv::Rect rect = cv::boundingRect(process_bot_th_img_contours[i]);
    //    if (rect.height + rect.y == process_bot_th_img.rows) continue;
    //    cv::Mat mask = cv::Mat::zeros(process_bot_img.size(), CV_8UC1);
    //    std::vector<std::vector<cv::Point>> draw_conts = { process_bot_th_img_contours[i] };
    //    cv::drawContours(mask, draw_conts, 0, 255, -1);
    //    double area = cv::countNonZero(mask);
    //    if (area >20) {
    //        th_value = 205;
    //        if (detect_flag_ == 7) {
    //            th_value = 180;
    //        }
    //        break;
    //    }
    //}

    //cv::threshold(process_img, th_img_1, th_value, 255, cv::THRESH_BINARY);
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 3));
    //cv::dilate(th_img_1, th_img_1, kernel);
    //kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::erode(th_img_1, th_img_1, kernel);

    //std::vector<std::vector<cv::Point>> th_img_1_contours = connector::get_contours(th_img_1);
    //for (int i = 0; i < th_img_1_contours.size();i++) {
    //    cv::Rect rect = cv::boundingRect(th_img_1_contours[i]);
    //    cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
    //    cv::Mat mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    //    std::vector<std::vector<cv::Point>> draw_conts = { th_img_1_contours[i] };
    //    cv::drawContours(mask, draw_conts, 0, 255, -1);
    //    double area = cv::countNonZero(mask);

    //    //if (area < 25 ) continue;
    //    //亮斑在上部分
    //    if (area >= area_th_ && c_pt.y< img1.rows/2+5 && rect.height < rect.width * 2 && rect.width>5 && rect.height>3) {
    //        ng = true;
    //        break;
    //    }

    //    //下半部分有亮斑,
    //    if (rect.y + rect.height == img1.rows && rect.height<=6)continue;
    //    double d = img1.rows - c_pt.y;
    //    if (d <= dis_p_ && area >= area_th_ && d>= dis_p_t_) {
    //        ng_botom = false;
    //    }
    //}

    if (ng_location_box) {
        if (is_left == 1) {
            singal_lc.template_rect[0] = cur_rect;
            singal_lc.template_line_vec[0][0] = k_zero;
            singal_lc.template_line_vec[0][1] = k_zero;
            singal_lc.e1 = e_ + error_e_ + 0.01;
            singal_lc.p1 = p_ + error_p_ + 0.01;
        }
        if (is_left == 0) {
            singal_lc.template_rect[3] = cur_rect;
            singal_lc.template_line_vec[3][0] = k_zero;
            singal_lc.template_line_vec[3][1] = k_zero;
            singal_lc.e2 = e_ + error_e_ + 0.01;
            singal_lc.p2 = p_ + error_p_ + 0.01;
        }
        return std::vector<cv::Point2f>();
    }


    cv::Mat  edge_th;
    cv::threshold(find_box_hsv_2[1], edge_th, 65, 255, cv::THRESH_BINARY);
    std::vector<int> his_1 = get_his(edge_th,1, 0);

    std::vector<int> row_diff;
    row_diff.resize(his_1.size());
    for (size_t i = 1; i < his_1.size(); i++)
        row_diff[i] = abs(his_1[i] - his_1[i - 1]);


    int tv = 0;
    for (size_t i = 0; i < row_diff.size(); i++) {
        if (row_diff[i] >= 40) {
            tv = i;
            break;
        }
    }
    //超过预期位置再找一遍
    if (tv>= 15) {
        for (size_t i = 0; i < row_diff.size(); i++) {
            if (row_diff[i] >= 23) {
                tv = i;
                break;
            }
        }
    }
    int bv = 0;
    KeyLine le,lp;
    le.startPointX = 0;
    le.startPointY = tv;
    le.endPointX = img2.cols;
    le.endPointY = tv;

    le.startPointX = le.startPointX  + cur_rect.x;
    le.startPointY = le.startPointY + cur_rect.y;
    le.endPointX = le.endPointX + cur_rect.x;
    le.endPointY = le.endPointY + cur_rect.y;

    lp.startPointX = 0;
    lp.startPointY = img2.rows;
    lp.endPointX = img2.cols;
    lp.endPointY = img2.rows;

    lp.startPointX = lp.startPointX + cur_rect.x;
    lp.startPointY = lp.startPointY + cur_rect.y;
    lp.endPointX = lp.endPointX + cur_rect.x;
    lp.endPointY = lp.endPointY + cur_rect.y;


    //左边的弹片
    if (is_left ==1) {
        singal_lc.template_rect[0] = cur_rect;
        singal_lc.template_line_vec[0][0] = le;
        singal_lc.template_line_vec[0][1] = lp;
        singal_lc.e1 = connector::dist_p2l(cv::Point(le.startPointX, le.startPointY), cv::Point(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)) * pix_value_ / 1000;
        singal_lc.p1 = 0;
    }
    //右边的弹片
    if (is_left == 0) {
        singal_lc.template_rect[3] = cur_rect;
        singal_lc.template_line_vec[3][0] = le;
        singal_lc.template_line_vec[3][1] = lp;
        singal_lc.e2 = connector::dist_p2l(cv::Point(le.startPointX, le.startPointY), cv::Point(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)) * pix_value_ / 1000;
        singal_lc.p2 = 0;
    }
    return std::vector<cv::Point2f>();
}

//弯母左右定位框弹片
std::vector<cv::Point2f> W_Female_Detect::find_location_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2) {
    cv::Mat grama_img = connector::gamma_trans(img2, 0.8);
    //合成原图
    cv::Mat mergedImage;
    cv::merge(hsv_v2,mergedImage);
    cv::Mat bgrImage;
    cv::cvtColor(mergedImage, bgrImage, cv::COLOR_HSV2BGR);
    cv::Rect roi_rect(cur_rect.x-5, cur_rect.y-5, cur_rect.width+10, cur_rect.height+10);

    cv::Mat roi_img = bgrImage(roi_rect);
    //保存图片
    std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + "_c.jpg";
    cv::imwrite(file_name, roi_img);
    g_conut++;
    
    //svm 检测
    bool ng_location_box = false;
    std::vector<cv::Mat> test_img_vec;
    test_img_vec.push_back(roi_img);
    nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 8, cv::Size(88, 34), 1);
    cv::Mat temp_feature = test_transform();
    double prob[2];
    double ret = svm_obj.testFeatureLibSVM(temp_feature, prob);
    if (prob[1]>0.72) {
        //第二个概率大于0.8表示不正常
        ng_location_box = true;
    }

    ////先左右分割检测有无金属弹片
    //cv::Mat l_img = grama_img(cv::Rect(7,5, img2.cols/2-3, img2.rows-10));
    //cv::Mat r_img = grama_img(cv::Rect(img2.cols/2 + 3 ,5, img2.cols/2-12, img2.rows-10));

    ////分上中线三段处理
    ////左右各分六份
    //int l_l_t_mean_value = cv::mean(l_img(cv::Rect(0, 0,                  l_img.cols/2, l_img.rows / 3)))[0];
    //int l_l_m_mean_value = cv::mean(l_img(cv::Rect(0, l_img.rows / 3,     l_img.cols/2, l_img.rows / 3)))[0];
    //int l_l_b_mean_value = cv::mean(l_img(cv::Rect(0, l_img.rows / 3 * 2, l_img.cols/2, l_img.rows / 3)))[0];
    //int l_r_t_mean_value = cv::mean(l_img(cv::Rect(l_img.cols/2, 0,                  l_img.cols /2, l_img.rows / 3)))[0];
    //int l_r_m_mean_value = cv::mean(l_img(cv::Rect(l_img.cols/2, l_img.rows / 3,     l_img.cols /2, l_img.rows / 3)))[0];
    //int l_r_b_mean_value = cv::mean(l_img(cv::Rect(l_img.cols/2, l_img.rows / 3 * 2, l_img.cols /2, l_img.rows / 3)))[0];
    //std::vector<int> l_mean{ l_l_t_mean_value ,l_l_m_mean_value ,l_l_b_mean_value ,l_r_t_mean_value ,l_r_m_mean_value ,l_r_b_mean_value };
    ////右侧的6份
    //int r_l_t_mean_value = cv::mean(r_img(cv::Rect(0, 0,                 r_img.cols/2, r_img.rows / 3)))[0];
    //int r_l_m_mean_value = cv::mean(r_img(cv::Rect(0, r_img.rows / 3,    r_img.cols/2, r_img.rows / 3)))[0];
    //int r_l_b_mean_value = cv::mean(r_img(cv::Rect(0, r_img.rows / 3 *2, r_img.cols/2, r_img.rows / 3)))[0];
    //int r_r_t_mean_value = cv::mean(r_img(cv::Rect(r_img.cols/2, 0,                  r_img.cols/2, r_img.rows / 3)))[0];
    //int r_r_m_mean_value = cv::mean(r_img(cv::Rect(r_img.cols/2, r_img.rows / 3,     r_img.cols/2, r_img.rows / 3)))[0];
    //int r_r_b_mean_value = cv::mean(r_img(cv::Rect(r_img.cols/2, r_img.rows / 3 * 2, r_img.cols/2, r_img.rows / 3)))[0];

    //std::vector<int> r_mean{ r_l_t_mean_value ,r_l_m_mean_value ,r_l_b_mean_value ,r_r_t_mean_value ,r_r_m_mean_value ,r_r_b_mean_value };
    
    if (is_left ==1) {
        singal_lc.template_rect[1] = cur_rect;
        singal_lc.template_line_vec[1].resize(9);
    }
    if (is_left ==0) {
        singal_lc.template_line_vec[2].resize(9);
        singal_lc.template_rect[2] = cur_rect;
    }

    KeyLine k_zero;
    k_zero.startPointX = 0;
    k_zero.startPointY = 0;
    k_zero.endPointX = 0;
    k_zero.endPointY = 0;

    //检测弹片中间缺失的部分,假设中间部分未缺失
    //bool ng_mid = false;
    //int mid_th_value = (l_r_m_mean_value + r_l_m_mean_value) / 2;
    //cv::Mat mid_img = grama_img(cv::Rect(grama_img.cols/4,0, grama_img.cols/2, grama_img.rows));
    //cv::Mat mid_th_img;
    //cv::threshold(mid_img, mid_th_img, mid_th_value,255,cv::THRESH_BINARY_INV);
    //double area = cv::countNonZero(mid_th_img);
    //if (area >750 ) {
    //    ng_mid = true;
    //   
    //}



    //中间弹片有缺失
    //弹片中部不缺失，上下部位与中间对比
    //上下差异大，上中 下中差异大
    //上端无反光，如果上侧亮度大视为缺失
    //if (l_l_m_mean_value > 200 ||r_r_m_mean_value > 200 || (abs(l_l_m_mean_value - r_r_m_mean_value) > 130) ||ng_location_box) {
    //    //TODO
    //    if (is_left==1) {
    //        singal_lc.template_line_vec[1][0] = k_zero;
    //        singal_lc.template_line_vec[1][1] = k_zero;
    //        singal_lc.template_line_vec[1][2] = k_zero;
    //        singal_lc.template_line_vec[1][3] = k_zero;
    //        singal_lc.template_line_vec[1][4] = k_zero;
    //        singal_lc.template_line_vec[1][5] = k_zero;
    //        singal_lc.template_line_vec[1][6] = k_zero;
    //        singal_lc.template_line_vec[1][7] = k_zero;
    //        singal_lc.template_line_vec[1][8] = k_zero;
    //        singal_lc.a1  = a_  + error_a_ + 0.01;
    //        singal_lc.c11 = c1_ + error_c1_ + 0.01;
    //        singal_lc.c12 = c2_ + error_c2_ + 0.01;
    //        singal_lc.d11 = d1_ + error_d1_ + 0.01;
    //        singal_lc.d12 = d2_ + error_d2_ + 0.01;
    //    }
    //    if (is_left==0) {
    //        singal_lc.template_line_vec[2][0] = k_zero;
    //        singal_lc.template_line_vec[2][1] = k_zero;
    //        singal_lc.template_line_vec[2][2] = k_zero;
    //        singal_lc.template_line_vec[2][3] = k_zero;
    //        singal_lc.template_line_vec[2][4] = k_zero;
    //        singal_lc.template_line_vec[2][5] = k_zero;
    //        singal_lc.template_line_vec[2][6] = k_zero;
    //        singal_lc.template_line_vec[2][7] = k_zero;
    //        singal_lc.template_line_vec[2][8] = k_zero;

    //        singal_lc.a2  = a_ + error_a_ + 0.01;
    //        singal_lc.c21 = c1_ + error_c1_ + 0.01;
    //        singal_lc.c22 = c2_ + error_c2_ + 0.01;
    //        singal_lc.d21 = d1_ + error_d1_ + 0.01;
    //        singal_lc.d22 = d2_ + error_d2_ + 0.01;
    //    }
    //    return std::vector<cv::Point2f>();
    //}

    if (is_left == 1) {
        singal_lc.template_line_vec[1][0] = k_zero;
        singal_lc.template_line_vec[1][1] = k_zero;
        singal_lc.template_line_vec[1][2] = k_zero;
        singal_lc.template_line_vec[1][3] = k_zero;
        singal_lc.template_line_vec[1][4] = k_zero;
        singal_lc.template_line_vec[1][5] = k_zero;
        singal_lc.template_line_vec[1][6] = k_zero;
        singal_lc.template_line_vec[1][7] = k_zero;
        singal_lc.template_line_vec[1][8] = k_zero;
        singal_lc.a1  = 0;
        singal_lc.c11 = 0;
        singal_lc.c12 = 0;
        singal_lc.d11 = 0;
        singal_lc.d12 = 0;
    }
    if (is_left == 0) {
        singal_lc.template_line_vec[2][0] = k_zero;
        singal_lc.template_line_vec[2][1] = k_zero;
        singal_lc.template_line_vec[2][2] = k_zero;
        singal_lc.template_line_vec[2][3] = k_zero;
        singal_lc.template_line_vec[2][4] = k_zero;
        singal_lc.template_line_vec[2][5] = k_zero;
        singal_lc.template_line_vec[2][6] = k_zero;
        singal_lc.template_line_vec[2][7] = k_zero;
        singal_lc.template_line_vec[2][8] = k_zero;
        singal_lc.a2  = 0;
        singal_lc.c21 = 0;
        singal_lc.c22 = 0;
        singal_lc.d21 = 0;
        singal_lc.d22 = 0;
    }
    //ng 的直接返回，未ng的计算边线
    if (ng_location_box) {

        if (is_left == 1) {
            singal_lc.a1 = a_ + error_a_ + 0.01;
            singal_lc.c11 = c1_ + error_c1_ + 0.01;
            singal_lc.c12 = c2_ + error_c2_ + 0.01;
            singal_lc.d11 = d1_ + error_d1_ + 0.01;
            singal_lc.d12 = d2_ + error_d2_ + 0.01;
        }
        if (is_left == 0) {
            singal_lc.a2 = a_ + error_a_ + 0.01;
            singal_lc.c21 = c1_ + error_c1_ + 0.01;
            singal_lc.c22 = c2_ + error_c2_ + 0.01;
            singal_lc.d21 = d1_ + error_d1_ + 0.01;
            singal_lc.d22 = d2_ + error_d2_ + 0.01;
        }
        return std::vector<cv::Point2f>();
    }

    //8条线分别是
    /*
    0，图像的中线
    1，缝的中线
    2 左侧上线 c11
    3 左侧下线 d11
    4 右侧上线 c12
    5 右侧下线 d12
    6 中线 b1
    7 中线 b2
    8 中线 b3
    */
    KeyLine k0, k1, k2, k3, k4, k5, k6, k7, k8;

    //先计算中线的差值
    //矩形框的中线
    double rect_mid_value = cur_rect.width / 2;
    k0.startPointX = rect_mid_value + cur_rect.x;
    k0.startPointY = 0 + cur_rect.y;
    k0.endPointX = rect_mid_value + cur_rect.x;
    k0.endPointY = img2.rows + cur_rect.y;
    
    double img_mid_value = 0;
    get_img_mid_value(grama_img, img_mid_value);
    k1.startPointX = img_mid_value + cur_rect.x;
    k1.startPointY = 0 + cur_rect.y;
    k1.endPointX = img_mid_value + cur_rect.x;
    k1.endPointY = img2.rows + cur_rect.y;
    
    //分为左右两张图，测量 上下边界
    int c11, c12, d11, d12;
    get_top_bottom_dege(grama_img, 1,c11,d11);
    get_top_bottom_dege(grama_img, 0,c12,d12);
   
    k2.startPointX = 0 + cur_rect.x;
    k2.startPointY = c11 + cur_rect.y;
    k2.endPointX = img2.cols/2 + cur_rect.x;
    k2.endPointY = c11 + cur_rect.y;

    k3.startPointX = 0 + cur_rect.x;
    k3.startPointY = d11 + cur_rect.y;
    k3.endPointX = img2.cols / 2 + cur_rect.x;
    k3.endPointY = d11 + cur_rect.y;

    k4.startPointX = img2.cols / 2 + cur_rect.x;
    k4.startPointY = c12 + cur_rect.y;
    k4.endPointX = img2.cols  + cur_rect.x;
    k4.endPointY = c12 + cur_rect.y;

    k5.startPointX = img2.cols / 2 + cur_rect.x;
    k5.startPointY = d12 + cur_rect.y;
    k5.endPointX = img2.cols + cur_rect.x;
    k5.endPointY = d12 + cur_rect.y;

    //计算中间的短横线
    cv::Point2f b1s;
    cv::Point2f b2s; 
    cv::Point2f b3s;
    cv::Point2f b1e;
    cv::Point2f b2e;
    cv::Point2f b3e;

    get_img_mid_thr_line(grama_img, img_mid_value, b1s, b2s, b3s,b1e,b2e,b3e);
    
    k6.startPointX = b1s.x + cur_rect.x;
    k6.startPointY = b1s.y + cur_rect.y;
    k6.endPointX = b1e.x + cur_rect.x;
    k6.endPointY = b1e.y + cur_rect.y;

    k7.startPointX = b2s.x + cur_rect.x;
    k7.startPointY = b2s.y + cur_rect.y;
    k7.endPointX = b2e.x + cur_rect.x;
    k7.endPointY = b2e.y + cur_rect.y;

    k8.startPointX = b3s.x + cur_rect.x;
    k8.startPointY = b3s.y + cur_rect.y;
    k8.endPointX = b3e.x + cur_rect.x;
    k8.endPointY = b3e.y + cur_rect.y;

  

    if (is_left ==1) {
        //singal_lc.template_line_vec[1][0] = k_zero;
        //singal_lc.template_line_vec[1][1] = k_zero;
        //singal_lc.template_line_vec[1][2] = k2;
        //singal_lc.template_line_vec[1][3] = k3;
        //singal_lc.template_line_vec[1][4] = k4;
        //singal_lc.template_line_vec[1][5] = k5;
        //singal_lc.template_line_vec[1][6] = k6;
        //singal_lc.template_line_vec[1][7] = k7;
        //singal_lc.template_line_vec[1][8] = k8;
        singal_lc.a1  = abs(img_mid_value - rect_mid_value) * pix_value_ / 1000;
        singal_lc.c11 = abs(c11) * pix_value_ / 1000;
        singal_lc.c12 = abs(c12) * pix_value_ / 1000;
        singal_lc.d11 = abs(d11 - grama_img.rows) * pix_value_ / 1000;
        singal_lc.d12 = abs(d12 - grama_img.rows) * pix_value_ / 1000;

        double e_a1 =  abs(singal_lc.a1 - a_);
        double e_c11 = abs(singal_lc.c11 - c1_);
        double e_c12 = abs(singal_lc.c12 - c2_);
        double e_d11 = abs(singal_lc.d11 - d1_);
        double e_d12 = abs(singal_lc.d12 - d2_);
        //ng的纠正为正常
        if(e_a1  > error_a_)  singal_lc.a1= error_a_ -0.01;
        if(e_c11 > error_c1_)  singal_lc.c11= error_c1_-0.01;
        if(e_c12 > error_c2_ ) singal_lc.c12 = error_c2_-0.01;
        if(e_d11 > error_d1_) singal_lc.d11 = error_d1_-0.01;
        if(e_d12 > error_d2_) singal_lc.d12= error_d2_-0.01;
    }
    if (is_left ==0) {
        /*singal_lc.template_line_vec[2][0] = k_zero;
        singal_lc.template_line_vec[2][1] = k_zero;
        singal_lc.template_line_vec[2][2] = k2;
        singal_lc.template_line_vec[2][3] = k3;
        singal_lc.template_line_vec[2][4] = k4;
        singal_lc.template_line_vec[2][5] = k5;
        singal_lc.template_line_vec[2][6] = k6;
        singal_lc.template_line_vec[2][7] = k7;
        singal_lc.template_line_vec[2][8] = k8;*/
        singal_lc.a2  = abs(img_mid_value - rect_mid_value) * pix_value_ / 1000;
        singal_lc.c21 = abs(c11) * pix_value_ / 1000;
        singal_lc.c22 = abs(c12) * pix_value_ / 1000;
        singal_lc.d21 = abs(d11 - grama_img.rows) * pix_value_ / 1000;
        singal_lc.d22 = abs(d12 - grama_img.rows) * pix_value_ / 1000;

        double e_a1 = abs(singal_lc.a2 - a_);
        double e_c11 = abs(singal_lc.c21 - c1_);
        double e_c12 = abs(singal_lc.c22 - c2_);
        double e_d11 = abs(singal_lc.d21 - d1_);
        double e_d12 = abs(singal_lc.d22 - d2_);
        //ng的纠正为正常
        if (e_a1 > error_a_)  singal_lc.a2 = error_a_ - 0.01;
        if (e_c11 > error_c1_)  singal_lc.c21 = error_c1_ - 0.01;
        if (e_c12 > error_c2_) singal_lc.c22 = error_c2_ - 0.01;
        if (e_d11 > error_d1_) singal_lc.d21 = error_d1_ - 0.01;
        if (e_d12 > error_d2_) singal_lc.d22 = error_d2_ - 0.01;
    }
    return std::vector<cv::Point2f>();
}

//弯母LC上左 上右金属弹片
std::vector<cv::Point2f> W_Female_Detect::find_top_box(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, cv::Rect cur_rect, lc_info& singal_lc, int is_left) {
    
    cv::Mat soble_img_x, sobel_img_y, edge, edge_th;
    cv::Sobel(img2, sobel_img_y, CV_16S, 0, 1, 3);
    cv::Sobel(img2, soble_img_x, CV_16S, 1, 0, 3);
    edge = soble_img_x + sobel_img_y;
    cv::convertScaleAbs(edge, edge);
    cv::threshold(edge, edge_th, 80, 255, cv::THRESH_TOZERO);

    cv::Mat mergedImage;
    cv::merge(hsv_v2, mergedImage);
    cv::Mat bgrImage;
    cv::cvtColor(mergedImage, bgrImage, cv::COLOR_HSV2BGR);
    cv::Rect roi_rect(cur_rect.x - 5, cur_rect.y - 5, cur_rect.width + 10, cur_rect.height + 10);

    cv::Mat roi_img = bgrImage(roi_rect);

    //保存图片
    std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + "_t.jpg";
    cv::imwrite(file_name, roi_img);
    g_conut++;

    bool ng_location_box = false;
    if (top_tan_) {
        std::vector<cv::Mat> test_img_vec;
        test_img_vec.push_back(roi_img);
        nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 7, cv::Size(60, 30), 1);
        cv::Mat temp_feature = test_transform();
        double prob[2];
        double ret = svm_obj_top_tan.testFeatureLibSVM(temp_feature, prob);
        if (prob[1] > 0.8) {
            //第二个概率大于0.8表示不正常
            ng_location_box = true;
        }
        if (ng_location_box) {
            KeyLine k_zero;
            k_zero.startPointX = 0;
            k_zero.startPointY = 0;
            k_zero.endPointX = 0;
            k_zero.endPointY = 0;
            if (is_left == 1) {
                singal_lc.template_rect[4] = cur_rect;
                singal_lc.template_line_vec[4].resize(1);
                singal_lc.template_line_vec[4][0] = k_zero;
                singal_lc.f1 = f_ + error_f_ + 0.01;
            }
            if (is_left == 0) {
                singal_lc.template_rect[5] = cur_rect;
                singal_lc.template_line_vec[5].resize(1);
                singal_lc.template_line_vec[5][0] = k_zero;
                singal_lc.f2 = f_ + error_f_ + 0.01;
            }
            return std::vector<cv::Point2f>();
        }
    }

    if (is_left == 1) {
        singal_lc.template_rect[4] = cur_rect;
        singal_lc.template_line_vec[4].resize(1);
    }
    if (is_left == 0) {
        singal_lc.template_rect[5] = cur_rect;
        singal_lc.template_line_vec[5].resize(1);

    }
    std::vector<int> col_vec;
    for (int i = 0; i < edge_th.rows; i++) {
        int count = 0;
        int sum = 0;
        for (int j = 0; j < edge_th.cols; j++) {
            int value = edge_th.at<uchar>(i, j);
            count++;
            sum = sum + value;
        }
        int avage;
        if (count > 0) {
            avage = sum / count;
        }
        else {
            avage = 0;
        }
        col_vec.push_back(avage);
    }
    int except_value = 0;
    for (int i = col_vec.size() - 1; i >= 0;i--) {
        if (col_vec[i]>70) {
            except_value = i;
            break;
        }
    }
    //记录弹片的上边缘，开始的位置
    int except_s = 0;
    for (int i =0; i < col_vec.size(); i++) {
        if (col_vec[i] > 70) {
            except_s = i;
            break;
        }
    }

    //如果没找到，或者太小。返回默认值,默认值为下边线的基础上加15
    if (except_value == 0 || except_value < 10) {
        except_value = singal_lc.top_line.startPointY+15 - cur_rect.y;
    }
    if (except_s>2) {
        except_value = except_s + 10;
    }
    KeyLine k0;
    k0.startPointX = 0 + cur_rect.x;
    k0.startPointY = except_value + cur_rect.y;
    k0.endPointX = img2.cols + cur_rect.x;
    k0.endPointY = except_value + cur_rect.y;

    if (is_left ==1) {
        singal_lc.template_line_vec[4][0] = k0;
        singal_lc.f1 = connector::dist_p2l(cv::Point(k0.startPointX, k0.startPointY), cv::Point(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)) * pix_value_ / 1000;;
    }
    if (is_left == 0) {
        singal_lc.template_line_vec[5][0] = k0;
        singal_lc.f2 = connector::dist_p2l(cv::Point(k0.startPointX, k0.startPointY), cv::Point(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)) * pix_value_ / 1000;;
    }

    return std::vector<cv::Point2f>();
}

void W_Female_Detect::get_img_mid_value(const cv::Mat& img, double& mid_value)
{
    cv::Mat soble_img_x, sobel_img_y, edge, edge_th;
    cv::Sobel(img, sobel_img_y, CV_16S, 0, 1, 3);
    cv::Sobel(img, soble_img_x, CV_16S, 1, 0, 3);
    edge = soble_img_x + sobel_img_y;
    cv::convertScaleAbs(edge, edge);
    cv::threshold(edge, edge_th, 95, 255, cv::THRESH_TOZERO);

    //被处理成1行
    std::vector<int> his_0 = get_his(edge_th, 0, 0);
    std::vector<int> col_diff;
    col_diff.resize(his_0.size());
    for (size_t i = 1; i < his_0.size(); ++i)
        col_diff[i] = abs(his_0[i] - his_0[i - 1]);


    //从大到小排列，取80%分位
    //std::vector<int> sort_col_diff;
    ////sort_col_diff = col_diff;
    //sort_col_diff.insert(sort_col_diff.end(), col_diff.begin()+15, col_diff.end()-15);
    //std::sort(sort_col_diff.begin(),sort_col_diff.end());
    
    int tv = 0;
    //左边有打光位移，从15开始
    std::vector<int> col_vec;
    for (size_t i = 15; i < col_diff.size()-15; i++) {
        if (col_diff[i] >= 50) {
            col_vec.push_back(i);
        }
    }
    
    if (col_vec.size() < 2) {
        //只有一个的情况，且不挨到两边
        if (col_vec.size() == 1) {
            mid_value = col_vec[0];
        }
        else {
            mid_value = img.cols / 2;
        }
       ;
    }
    if (col_vec.size() == 2) {
        mid_value = (col_vec[0] + col_vec[1]) / 2;
    }
    if (col_vec.size() > 2) {
        //取挨得最近的两个
        std::vector<int> tmp_vec;
        tmp_vec.swap(col_vec);
        //出现连续的多个，判定最后一个与第一个
        if (abs(tmp_vec[0] - tmp_vec[(tmp_vec.size()-1)])<15) {
            col_vec.push_back(tmp_vec[0]);
            col_vec.push_back(tmp_vec[tmp_vec.size() - 1]);
            tmp_vec.clear();
        }
        //其他情况
        if (tmp_vec.size() > 0) {
            for (int i = 0; i < tmp_vec.size() - 1;) {
                int c = tmp_vec[i];
                int n = tmp_vec[i + 1];
                if (std::abs(c - n) < 17) {
                    col_vec.push_back((n + c) / 2);
                    i = i + 2;
                }
                else {
                    col_vec.push_back(c);
                    i = i + 1;
                }
            }
        }
        if (col_vec.size() < 2) {
            mid_value = img.cols / 2;
        }
        else if (col_vec.size() == 2) {
            mid_value = (col_vec[0] + col_vec[1]) / 2;
        }
        else {
            mid_value = img.cols / 2;
        }
    }
}
void  W_Female_Detect::get_top_bottom_dege(const cv::Mat& img, int is_left, int& top_value, int& bot_value) {
    
    cv::Mat pro_img;
    if (is_left ==1) {
        pro_img = img(cv::Rect(0, 0, img.cols / 2, img.rows));
    }
    if (is_left ==0) {
        pro_img = img(cv::Rect(img.cols / 2, 0, img.cols / 2, img.rows));
    }
    //默认在最上侧 最下侧
    top_value = 0;
    bot_value = pro_img.rows;
   
    cv::Mat th_img;
    int th_value = connector::exec_threshold(pro_img,connector::THRESHOLD_TYPE::HUANG2);
    
    //分为上中下三段
    int mean_value_top = cv::mean(pro_img(cv::Rect(0, 5, pro_img.cols, pro_img.rows/3)))[0];
    int mean_value_mid = cv::mean(pro_img(cv::Rect(0, pro_img.rows / 3, pro_img.cols, pro_img.rows / 3)))[0];
    int mean_value_bot = cv::mean(pro_img(cv::Rect(0, pro_img.rows / 3, pro_img.cols, pro_img.rows / 3-4)))[0];
    //弹片中间一般不缺
    //if ((abs(mean_value_top- mean_value_mid)>50) || ((abs(mean_value_top - mean_value_mid) > 50))) {
    //   //弹片缺失
    //    top_value = -1;
    //    bot_value = -1;
    //    return;
    //}

    int use_th_value;
    //上下差距大，表示弹片缺失露头
    if (abs(mean_value_top - mean_value_bot) > 20) {
        //use_th_value = th_value;
        use_th_value = mean_value_bot +20;
    }
    else {
        use_th_value = 180;
    }
    if (use_th_value<130) {
        use_th_value = 130;
    }
    cv::threshold(pro_img, th_img, use_th_value, 255,cv::THRESH_BINARY);
    
    std::vector<int> his_1 = get_his(th_img, 1, 0);

    std::vector<int> row_diff;
    row_diff.resize(his_1.size());
    for (size_t i = 1; i < his_1.size(); i++)
        row_diff[i] = abs(his_1[i] - his_1[i - 1]);

    std::vector<int> edge_vec;
    for (size_t i = 0; i < row_diff.size(); i++) {
        if (row_diff[i] >= 70) {
            edge_vec.push_back(i);
        }
    }
    std::vector<int> temp_vec;
    temp_vec.swap(edge_vec);
    //可能有相近的，去重留下单一的,譬如 28  29 
    if (temp_vec.size()>0) {
        for (int i = 0; i < temp_vec.size() - 1;) {
            int c = temp_vec[i];
            int n = temp_vec[i + 1];
            if (std::abs(c - n) < 3) {
                edge_vec.push_back((n+c)/2);
                i = i + 2;
            }
            else {
                edge_vec.push_back(c);
                i = i+1;
            }
        }
        //只有一个
        if (temp_vec.size()==1) {
            edge_vec.push_back(temp_vec[0]);
        }
    }
    //若果只有一个替换,临近替换
    if (edge_vec.size()==1) {
        //正常情况
        if (edge_vec[0] >= pro_img.rows/2) {
            bot_value = edge_vec[0];
        }
        if (edge_vec[0] < pro_img.rows / 2) {
            top_value = edge_vec[0];
        }
        //露头情况
        if (abs(mean_value_top - mean_value_bot) > 20) {
            top_value = edge_vec[0];
        }
        
    }
    if (edge_vec.size() == 2) {
        top_value = edge_vec[0];
        bot_value = edge_vec[1];
    }
    if (top_value > pro_img.rows / 2) { top_value = 0; }
    if (bot_value < pro_img.rows / 2) { bot_value = pro_img.rows - 1; }
}

void W_Female_Detect::get_img_mid_thr_line(cv::Mat img, double img_mid, cv::Point2f& b1s, cv::Point2f& b2s, cv::Point2f& b3s, cv::Point2f& b1e, cv::Point2f& b2e, cv::Point2f& b3e) {

    int moment_value = connector::exec_threshold(img, connector::THRESHOLD_TYPE::HUANG2);
    int th_value = moment_value > 100 ? moment_value : 100;
    cv::Mat th_img;
    cv::threshold(img,th_img,th_value,255,cv::THRESH_BINARY);
    //在图像中缝附近找横短线
    cv::Rect mid_rect(img_mid - 10, 0, 20, th_img.rows);
    cv::Mat roi_th_img = th_img(mid_rect);

    int mid_height = th_img.rows - 8;
    int start = 0;
    int end = 0;
    get_line_pt(roi_th_img, mid_height / 4, start, end);
    if (start == 0)
        start = 5;
    if (end == 0)
        end = 5 + 5;
    b1s = cv::Point2f(start + mid_rect.x, mid_height / 4 + mid_rect.y);
    b1e = cv::Point2f(end + mid_rect.x, mid_height / 4 + mid_rect.y);
    
    start = end = 0;
    get_line_pt(roi_th_img, mid_height / 4 *2 , start, end);
    if (start == 0)
        start = 5;
    if (end == 0)
        end = 5 + 5;
    b2s = cv::Point2f(start + mid_rect.x, mid_height / 4*2 + mid_rect.y);
    b2e = cv::Point2f(end + mid_rect.x, mid_height / 4*2 + mid_rect.y);

    start = end = 0;
    get_line_pt(roi_th_img, mid_height / 4 * 3, start, end);
    if (start == 0)
        start = 5;
    if (end == 0)
        end = 5 + 5;
    b3s = cv::Point2f(start + mid_rect.x, mid_height / 4 * 3 + mid_rect.y);
    b3e = cv::Point2f(end + mid_rect.x, mid_height / 4 * 3 + mid_rect.y);
}
void W_Female_Detect::img_process_2(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept
{
    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 25;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(th_img_1, th_img_1, kernel);
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;
    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 100 || rect.height > 50)continue;
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }
    //获取足够的待检测块
    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        //分每行 每列
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        //求角度
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 100 || rect.height > 55 || rect.width <= 70)continue;

        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        //cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }
    std::vector<W_Female_Detect::lc_info> lc_info_vec;
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 89, 33),
        cv::Vec4i(212, 0, 89, 33),
        cv::Vec4i(565, 0, 87, 33),
        cv::Vec4i(776, 0, 89, 34),
        cv::Vec4i(1128, 0, 90, 34),
        cv::Vec4i(1341, 0, 89, 33),
        cv::Vec4i(1691, 0, 90, 34),
        cv::Vec4i(1906, 0, 87, 33),
    };
    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            //默认每行未找全
            bool estimate_flag = false;
            if (!estimate_flag) {
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                double distance = 0;
                static std::vector<int> s_numbers = { 118, 330, 683, 894, 1246, 1459,1809,2024 };
                if (i % 2 == 0) {
                    // 偶数行
                    distance = (s_rect.x - detect_left_x_);
                }
                if (i % 2 == 1) {
                    // 奇数行
                    distance = (s_rect.x - 130 - detect_left_x_);
                }
                auto s_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                s_col_idx = std::distance(s_numbers.begin(), s_it);
                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    if (i % 2 == 0) distance = (cur_rect.x - detect_left_x_);
                    if (i % 2 == 1) distance = (cur_rect.x - 130 - detect_left_x_);;

                    auto c_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                    c_col_idx = std::distance(s_numbers.begin(), c_it);
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }
    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    //#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i = i + 2) {
        lc_info singal_female = cal_2(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, algo_result, process_rect_vec[i], process_rect_vec[i + 1], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        lc_info_vec.push_back(singal_female);
        w_lock.unlock();
    }
  
    data_cvt(lc_info_vec, algo_result);
    cv::Mat dis = src_2.clone();
    connector::draw_results(dis, algo_result->result_info);

}
W_Female_Detect::lc_info W_Female_Detect::cal_2(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect cur_rect, cv::Rect next_rect, int index, cv::Mat inv_m) noexcept
{
    // 第几行第几个
    int col_idx = index % 8;
    int row_idx = index / 8;
    //相对位置关系
    std::vector<cv::Vec4i> pos = {
        cv::Vec4i(-95,-12,38,67), //左侧弹片
        cv::Vec4i(358,-13,39,68), //右侧弹片
        cv::Vec4i(1,-101,340,30), //上侧找线边框
        cv::Vec4i(106,-86,50,25), //上左金属弹片
        cv::Vec4i(166,-85,50,25) //上右金属弹片
    };
    if (row_idx % 2 == 1) {
        pos[3] = cv::Vec4i(94, -86, 50, 20);
        pos[4] = cv::Vec4i(154, -85, 50, 20);
    }

    lc_info singal_lc;
    singal_lc.template_rect.resize(6);
    singal_lc.template_line_vec.resize(6);

    singal_lc.index = index / 2;

    LOGI("W_Female_Detect detect  find_line  start {}", singal_lc.index);
    //找上边线
    find_line(src_1(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), src_2(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3]), singal_lc);

    if (singal_lc.top_line.startPointX == 0 || singal_lc.top_line.endPointX == 0) {
        return singal_lc;
    }
    //计算上边线到当前矩形的距离
    double distance_t_c = abs(singal_lc.top_line.startPointY - cur_rect.y);
    if (abs(distance_t_c - 85) > 3) {
        cur_rect.y = singal_lc.top_line.startPointY + 85;
        next_rect.y = singal_lc.top_line.startPointY + 85;
    }

    //找LC左金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), src_2(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3]), singal_lc, 1, hsv_v1, hsv_v2);
    //找LC右金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), src_2(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3]), singal_lc, 0, hsv_v1, hsv_v2);

    //找左右定位框
    find_location_box(src_1(cur_rect), src_2(cur_rect), cur_rect, singal_lc, 1, hsv_v1, hsv_v2);
    find_location_box(src_1(next_rect), src_2(next_rect), next_rect, singal_lc, 0, hsv_v1, hsv_v2);

    //寻找最上面的弹片，左右
    find_top_box(src_1(cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3])), src_2(cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3])), hsv_v1, hsv_v2,cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3]), singal_lc, 1);
    find_top_box(src_1(cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3])), src_2(cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3])), hsv_v1, hsv_v2,cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3]), singal_lc, 0);

    return singal_lc;
}


void  W_Female_Detect::img_process_4(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept {

    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 25;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::dilate(th_img_1, th_img_1, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;

    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 130 || rect.height > 70)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }

    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;
#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 140 || rect.height > 70 || rect.width <= 70)continue;
        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 140)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        /*cv::drawContours(gray_mask, draw_conts, 0, 255, -1);*/
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }

    std::vector<W_Female_Detect::w_female_2> w_female_vec;
   


    // 同一行矩形的相对关系
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 101, 44),
        cv::Vec4i(142, 0, 91, 44),
        cv::Vec4i(298, 0, 92, 44),
        cv::Vec4i(432, 0, 101, 45),
        cv::Vec4i(575, 0, 91, 43),
        cv::Vec4i(731, 0, 91, 43),
        cv::Vec4i(865, 0, 101, 43),
        cv::Vec4i(1007, 0, 90, 43),
        cv::Vec4i(1163, 0, 90, 46),
        cv::Vec4i(1298, 0, 100, 46),
        cv::Vec4i(1441, 0, 90, 46),
        cv::Vec4i(1596, 0, 91, 45),
        cv::Vec4i(1731, 0, 100, 46),
    };

    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            ;
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }

        for (int i = 0; i < rank.size(); i++) {
            // 每一行进行处理,最后一行特殊处理
            bool estimate_flag = false;
            if(!estimate_flag)
            {
                // 当前行未找全的，特殊处理，全部特殊处理
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                double distance = 0;
                //13 *10 的固定距离
                static std::vector<int> s_numbers = { 0,144, 300, 432, 576, 731, 864,1008, 1164,1297,1441,1596,1729};

                if (i % 2 == 0) {
                    // 偶数行
                    //50 是对一个空洞到roi 的边缘距离
                    distance = (s_rect.x - 50 -detect_left_x_) ;
                }
                if (i % 2 == 1) {
                    // 奇数行
                    // 120 的基础是奇数行与偶数行的第一个空洞的间距
                    distance = (s_rect.x  -170- detect_left_x_);
                }
                auto s_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                s_col_idx = std::distance(s_numbers.begin(), s_it);

                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    // 当前黑孔的序号
                    if (i % 2 == 0) distance = (cur_rect.x - 50 - detect_left_x_);
                    if (i % 2 == 1) distance = (cur_rect.x - 170 - detect_left_x_);;

                    auto c_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                    c_col_idx = std::distance(s_numbers.begin(), c_it);
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }

    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    std::vector<cv::Mat> rbg_v1, rbg_v2;
    cv::split(input_img_1, rbg_v1);
    cv::split(input_img_2, rbg_v2);

#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i++) {
        w_female_2 singal_female = cal_4(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, rbg_v1, rbg_v2,algo_result, process_rect_vec[i], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        w_female_vec.push_back(singal_female);
        w_lock.unlock();
    }
   data_cvt_4(w_female_vec, algo_result);
  /* cv::Mat dis = src_2.clone();
   connector::draw_results(dis, algo_result->result_info);*/
}


W_Female_Detect::w_female_2 W_Female_Detect::cal_4(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, std::vector<cv::Mat> rbg_v1, std::vector<cv::Mat> rbg_v2, AlgoResultPtr algo_result, cv::Rect rect, int index, cv::Mat inv_m) {
    
    w_female_2 singal_w_female;

    singal_w_female.template_rect = rect;
    singal_w_female.index = index;

    // 分图
   /* std::vector<cv::Mat> rbg_v1, rbg_v2;
    cv::split(input_img_1, rbg_v1);
    cv::split(input_img_2, rbg_v2);*/

    cv::Mat img_h1 = hsv_v1[0](rect).clone();
    cv::Mat img_s1 = hsv_v1[1](rect).clone();
    cv::Mat img_v1 = hsv_v1[2](rect).clone();

    cv::Mat img_h2 = hsv_v2[0](rect).clone();
    cv::Mat img_s2 = hsv_v2[1](rect).clone();
    cv::Mat img_v2 = hsv_v2[2](rect).clone();

    cv::Mat img_b1 = rbg_v1[0](rect).clone();
    cv::Mat img_g1 = rbg_v1[1](rect).clone();
    cv::Mat img_r1 = rbg_v1[2](rect).clone();
    cv::Mat img_b2 = rbg_v2[0](rect).clone();
    cv::Mat img_g2 = rbg_v2[1](rect).clone();
    cv::Mat img_r2 = rbg_v2[2](rect).clone();

    std::vector<cv::Mat> find_hsv1;
    std::vector<cv::Mat> find_hsv2;
    find_hsv1.emplace_back(img_h1);
    find_hsv1.emplace_back(img_s1);
    find_hsv1.emplace_back(img_v1);
    find_hsv2.emplace_back(img_h2);
    find_hsv2.emplace_back(img_s2);
    find_hsv2.emplace_back(img_v2);


    //合成原图
    cv::Mat mergedImage;
    cv::merge(hsv_v2, mergedImage);
    cv::Mat bgrImage;
    cv::cvtColor(mergedImage, bgrImage, cv::COLOR_HSV2BGR);
    cv::Rect roi_rect(rect.x - 5, rect.y - 5, rect.width + 10, rect.height + 10);
    cv::Mat roi_img = bgrImage(roi_rect);
    //保存图片
    //std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + ".jpg";
    //cv::imwrite(file_name, roi_img);
    //g_conut++;

    //svm 检测
    bool ng_location_box = false;
    std::vector<cv::Mat> test_img_vec;
    test_img_vec.push_back(roi_img);
    nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 6, cv::Size(100, 55), 1);
    cv::Mat temp_feature = test_transform();
    double prob[2];
    double ret = svm_obj_kbai.testFeatureLibSVM(temp_feature, prob);
    if (prob[1] > 0.8) {
        //第二个概率大于0.8表示不正常
        ng_location_box = true;
    }
    if (ng_location_box) {
        singal_w_female.a = a_ + error_a_ + 0.01;
        singal_w_female.b = b1_ + error_b1_ + 0.01;
        singal_w_female.c = c1_ + error_c1_ + 0.01;
        singal_w_female.d = d1_ + error_d1_ + 0.01;
        return singal_w_female;
    }


    // 黑色框
    int col_idx = index % 13;
    int row_idx = index / 13;

    int left_value = 0;
    int right_value = 0;
    int top_value = 0;
    int bot_value = 0;
    KeyLine k1, k2, k3, k4;
    //使用h 通道的图进行检测
    //分为 亮暗 不同的处理
    //亮的部分，亮的下部分边界默认使用下部
     //左右边界往里收缩到30%的位置作为边界,左右采用H 通道
     //上下采用S通道分割
    if (col_idx == 0 || col_idx == 3 || col_idx == 6 || col_idx == 9 || col_idx == 12) 
    {
        left_value = get_percent_edge(find_hsv1, find_hsv2, 0.3, 1, 2);
        right_value = get_percent_edge(find_hsv1,find_hsv2, 0.3, 1, 3);
        top_value = get_s_edge(find_hsv1, find_hsv2, 1, 20);
        bot_value = img_h2.rows;
    }
    //暗的部分，暗斑的下部分边界使用S图像
    if (col_idx == 1 || col_idx == 2 || col_idx == 4 || col_idx == 5 || col_idx == 7 || col_idx == 8 || col_idx == 10 || col_idx == 11) 
    {
       left_value = get_percent_edge(find_hsv1, find_hsv2, 0.3, 1, 2);
       right_value = get_percent_edge(find_hsv1, find_hsv2, 0.3, 1, 3);
       top_value = 0;
       bot_value = get_s_edge(find_hsv1, find_hsv2, 0, 13);
    }

    //还原位置到原图
    k1.startPointX = left_value + rect.x;
    k1.startPointY = rect.y;
    k1.endPointX = left_value + rect.x;
    k1.endPointY = rect.y+rect.height;

    k2.startPointX = right_value + rect.x;
    k2.startPointY = rect.y;
    k2.endPointX = right_value + rect.x;
    k2.endPointY = rect.y + rect.height;

    k3.startPointX = rect.x;
    k3.startPointY = top_value+ rect.y;
    k3.endPointX = rect.x+rect.width;
    k3.endPointY = top_value + rect.y;


    k4.startPointX = rect.x;
    k4.startPointY = rect.y+ bot_value;
    k4.endPointX = rect.x + rect.width;
    k4.endPointY = rect.y + bot_value;

    singal_w_female.line_vec.emplace_back(k1);
    singal_w_female.line_vec.emplace_back(k2);
    singal_w_female.line_vec.emplace_back(k3);
    singal_w_female.line_vec.emplace_back(k4);


    singal_w_female.a = left_value * pix_value_ / 1000;
    singal_w_female.b = std::abs(right_value - img_h2.cols) * pix_value_ / 1000;
    singal_w_female.c = top_value * pix_value_ / 1000;
    singal_w_female.d = std::abs(bot_value - img_h2.rows )* pix_value_ / 1000;
    return singal_w_female;

}

//按列算，每一列计算的情况为1，每一行计算的情况为0， left 表示上下左右，  上0 下1 左2 右3
int W_Female_Detect::get_percent_edge(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, double percent, int type, int left) {

    std::vector<double>h_his_vec;
    std::vector<double>v_his_vec;
    int th_value = 70;
    // v 通道判断亮度，h 通道判断边界
    if (type == 1) {
        for (int i = 0; i < hsv2[0].cols; i++) {
            double count = 0;
            for (int j = 0; j < hsv2[0].rows; j++) {
                int value = hsv2[0].at<uchar>(j, i);
                if (value < th_value) {
                    count++;
                }
            }
            double avage = count / (hsv2[0].rows *1.f);
            h_his_vec.push_back(avage);
        }

        for (int i = 0; i < hsv2[2].cols; i++) {
            int sum = 0;
            for (int j = 0; j < hsv2[2].rows; j++) {
                int value = hsv2[2].at<uchar>(j, i);
                sum = sum + value;
            }
            double avage = sum / hsv2[2].rows;
            v_his_vec.push_back(avage);
        }

    }
    if (type==0) {
        for (int i = 0; i < hsv2[0].rows; i++) {
            double count = 0;
            for (int j = 0; j < hsv2[0].cols; j++) {
                int value = hsv2[0].at<uchar>(i, j);
                if (value < th_value) {
                    count++;
                }
            }
            double avage = count / (hsv2[0].cols*1.f);
            h_his_vec.push_back(avage);
        }
        for (int i = 0; i < hsv2[2].rows; i++) {
           
            int sum = 0;
            for (int j = 0; j < hsv2[2].cols; j++) {
                int value = hsv2[2].at<uchar>(i, j);
                sum = sum + value;
            }
            double avage = sum / hsv2[2].cols;
            v_his_vec.push_back(avage);
        }
    
    }
    int ret_value = 0;
    //左
    if (left==2) {
        
        for (int i = 1; i < h_his_vec.size();i++) {
            if (h_his_vec[i]> percent && v_his_vec[i]<180) {
                ret_value = i;
                break;
            }
        }
        //找错了，提高阈值再找下
        if (ret_value > hsv2[2].cols / 2) {
            for (int i =1; i < h_his_vec.size(); i++) {
                if (h_his_vec[i] > percent && v_his_vec[i] < 200) {
                    ret_value = i;
                    break;
                }
            }
        }
    }
    //右
    if (left == 3) {

        for (int i = h_his_vec.size()-2; i >0; i--) {
            if (h_his_vec[i] > percent && v_his_vec[i] < 180) {
                ret_value = i;
                break;
            }
        }
        //找错了，提高阈值再找下
        if (ret_value< hsv2[2].cols/2) {
            for (int i = h_his_vec.size() - 2; i > 0; i--) {
                if (h_his_vec[i] > percent && v_his_vec[i] < 200) {
                    ret_value = i;
                    break;
                }
            }
        }
    
    }
    return ret_value;
}

//type 为1，表示上侧边界，type 为0 表示下部边界
int W_Female_Detect::get_s_edge(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, int type, int th_value) {

    cv::Mat th_img;
    cv::threshold(hsv2[1], th_img, th_value,255,cv::THRESH_BINARY_INV);

    auto contours = connector::get_contours(th_img);

    int ret_value = 0;
    //上侧边缘
    if (type==1) {
        for (int i = 0; i < contours.size();i++) {
            cv::Rect rect = cv::boundingRect(contours[i]);
            if (rect.width>10 && rect.y+rect.height< hsv2[1].rows / 2) {
                ret_value = rect.y +rect.height;
                break;
            }
        }
    }
    //下侧边缘
    if (type == 0) {
        for (int i = 0; i < contours.size(); i++) {
            cv::Rect rect = cv::boundingRect(contours[i]);
            if (rect.height> rect.width) {
                continue;
            }
            if (rect.width>25 && rect.y> hsv2[1].rows/2) {
                ret_value = rect.y;
                break;
            }
        }
   
    }
    return ret_value;
}
void W_Female_Detect::data_cvt_4(std::vector<w_female_2> w_female_vec, AlgoResultPtr algo_result) {

    status_flag = true;
    for (int i = 0; i < w_female_vec.size(); i++) {
        w_female_2 tmp_lc = w_female_vec[i];

        cv::Point2f org_pc;
        //模板框的位置
       
        //每个矩形框的位置
        cv::Rect tp_rect = tmp_lc.template_rect;
        cv::Point2f lt = tp_rect.tl();
        cv::Point2f lb(tp_rect.tl().x, tp_rect.tl().y + tp_rect.height);
        cv::Point2f rt(tp_rect.br().x, tp_rect.br().y - tp_rect.height);
        cv::Point2f rb = tp_rect.br();
        cv::Point2f pc(tp_rect.x + tp_rect.width / 2, tp_rect.y + tp_rect.height / 2);
        org_pc = connector::TransPoint(tmp_lc.inv_h, pc);
        cv::Point2f org_lt = connector::TransPoint(tmp_lc.inv_h, lt);
        cv::Point2f org_lb = connector::TransPoint(tmp_lc.inv_h, lb);
        cv::Point2f org_rt = connector::TransPoint(tmp_lc.inv_h, rt);
        cv::Point2f org_rb = connector::TransPoint(tmp_lc.inv_h, rb);
        algo_result->result_info.push_back(
            { { "label", "fuzhu" },
                { "shapeType", "polygon" },
                { "points", { { org_lt.x, org_lt.y }, { org_rt.x, org_rt.y }, { org_rb.x, org_rb.y }, { org_lb.x, org_lb.y } } },
                { "result", 1 } });
        
        //四条线
        for (int j = 0; j < tmp_lc.line_vec.size(); j++) {
            KeyLine tmp_l = tmp_lc.line_vec[j];
            if (tmp_l.startPointX < 1 || tmp_l.endPointX < 1 || std::isnan(tmp_l.startPointX) || std::isnan(tmp_l.endPointX)) {
                continue;
            }
            cv::Point2f p3 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.startPointX, tmp_l.startPointY));
            cv::Point2f p4 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.endPointX, tmp_l.endPointY));
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "line" },
                    { "points", { { p3.x, p3.y }, { p4.x, p4.y } } },
                    { "result", 1 } });
        }

        double status_a, status_b, status_c, status_d;

        double a  = abs(tmp_lc.a - a_);
        double b = abs(tmp_lc.b - b1_);
        double c = abs(tmp_lc.c - c1_);
        double d = abs(tmp_lc.d - d1_);

        status_a = a <= error_a_ ? 1 : 0;
        status_b = b <= error_b1_ ? 1 : 0;
        status_c = c <= error_c1_ ? 1 : 0;
        status_d = d <= error_d1_ ? 1 : 0;
       
        if (status_a < 1 || status_b < 1 || status_c < 1 || status_d < 1) {
            status_flag = false;
        }
        else {
        }
        //if (!status_flag) {
        //    algo_result->judge_result = 0;
        //}
        //else {
        //    algo_result->judge_result = 1;
        //}
        
        if (!status_flag) {
            algo_result->judge_result = 0;
        }
        else {
            if (algo_result->judge_result == 0) {
            }
            else {
                algo_result->judge_result = 1;
            }
        }
        algo_result->result_info.push_back(
            { { "label", "W_Female_Detect_defect_1" },
                { "shapeType", "basis" },
                { "points", { { -1, -1 } } },
                { "result", { { "dist", { tmp_lc.a,tmp_lc.b,tmp_lc.c,tmp_lc.d,} },
                              { "status", {status_a,status_b,status_c,status_d,} },
                              { "error", {a,b,c,d,} }  ,
                              { "index", (int)tmp_lc.index },
                              { "points", { {org_pc.x,org_pc.y} } } } } });
    }
    
}

//LC单列
void W_Female_Detect::img_process_5(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept
{
    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 25;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(th_img_1, th_img_1, kernel);
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;
    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 100 || rect.height > 50)continue;
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }
    //获取足够的待检测块
    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        //分每行 每列
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        //求角度
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);


        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 100 || rect.height > 55 || rect.width <= 70)continue;

        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        //cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }
    std::vector<W_Female_Detect::lc_info_2> lc_info_vec;
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 89, 35),
        cv::Vec4i(212, 0, 89, 33),
        cv::Vec4i(563, 0, 88, 35),
        cv::Vec4i(775, 0, 89, 35),
        cv::Vec4i(1127, 0, 89, 36),
        cv::Vec4i(1339, 0, 89, 35),
    };
    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            //默认每行未找全
            bool estimate_flag = false;
            if (!estimate_flag) {
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                double distance = 0;
                static std::vector<int> s_numbers = { 150, 360, 715, 930, 1280, 1490 };
                if (i % 2 == 0) {
                    // 偶数行
                    distance = (s_rect.x - detect_left_x_);
                }
                if (i % 2 == 1) {
                    // 奇数行
                    distance = (s_rect.x - 130 - detect_left_x_);
                }
                auto s_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                s_col_idx = std::distance(s_numbers.begin(), s_it);
                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    if (i % 2 == 0) distance = (cur_rect.x - detect_left_x_);
                    if (i % 2 == 1) distance = (cur_rect.x - 130 - detect_left_x_);;

                    auto c_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                    c_col_idx = std::distance(s_numbers.begin(), c_it);
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }
    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    LOGI("W_Female_Detect detect  lc_info start");
    #pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i = i + 6) {
        lc_info_2 singal_female = cal_5(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, algo_result, process_rect_vec[i], process_rect_vec[i + 1], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        lc_info_vec.push_back(singal_female);
        w_lock.unlock();


    }
    LOGI("W_Female_Detect detect  lc_info end");
    data_cvt_5(lc_info_vec, algo_result);
    cv::Mat dis = src_2.clone();
    connector::draw_results(dis, algo_result->result_info);
}
W_Female_Detect::lc_info_2 W_Female_Detect::cal_5(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect cur_rect, cv::Rect next_rect, int index, cv::Mat inv_m) noexcept {

    // 第几行第几个
    int col_idx = index % 6;
    int row_idx = index / 6;
    //相对位置关系
    std::vector<cv::Vec4i> pos = {
            cv::Vec4i(-96,-20,39,75), //左侧弹片
            cv::Vec4i(358,-19,38,74), //右侧弹片
            cv::Vec4i(-53,-96,395,26), //上侧找线边框
    };
    
    lc_info_2 singal_lc;
    singal_lc.template_rect.resize(4);
    singal_lc.template_line_vec.resize(4);

    singal_lc.index = index / 2;


    //找上边线
    find_line(src_1(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), src_2(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3]), singal_lc);

    if (singal_lc.top_line.startPointX == 0 || singal_lc.top_line.endPointX == 0) {
        return singal_lc;
    }
    //计算上边线到当前矩形的距离
    double distance_t_c = abs(singal_lc.top_line.startPointY - cur_rect.y);
    //标准距离为83个像素
    if (abs(distance_t_c - 83) > 5) {
        cur_rect.y = singal_lc.top_line.startPointY + 83;
        next_rect.y = singal_lc.top_line.startPointY + 83;
    }
    //需要hsv 通道

    //找LC左金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), src_2(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3]), singal_lc, 1, hsv_v1, hsv_v2);
    //找LC右金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), src_2(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3]), singal_lc, 0, hsv_v1, hsv_v2);

    //找左右定位框
    find_location_box(src_1(cur_rect), src_2(cur_rect), cur_rect, singal_lc, 1, hsv_v1, hsv_v2);
    find_location_box(src_1(next_rect), src_2(next_rect), next_rect, singal_lc, 0, hsv_v1, hsv_v2);

    return singal_lc;
}
std::vector<cv::Point2f> W_Female_Detect::find_line(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info_2& singal_lc) {
    cv::Mat th_img;
    //cv::threshold(img1,th_img,70,255,cv::THRESH_BINARY);
    nlohmann::json bot_line_params = {
               { "CaliperNum", 40 },
               { "CaliperLength", 20 },
               { "CaliperWidth", 10 },
               { "Transition", "positive"},
               { "Sigma", 1 },
               { "Num", 1 },
               { "Contrast", 30 },
    };
    Tival::TPoint start, end;

    start.X = img1.cols;
    start.Y = img1.rows / 2;
    end.X = 0;
    end.Y = img1.rows / 2;

    Tival::FindLineResult bot_line_ret = Tival::FindLine::Run(img1, start, end, bot_line_params);

    //找线找错了，灰度图找错了，二值图进行查找
    if (bot_line_ret.start_point.size() == 1 && (bot_line_ret.start_point[0].y < 13 || abs(bot_line_ret.start_point[0].y - bot_line_ret.end_point[0].y)>5)) {
        cv::Mat th_img;
        cv::threshold(img1, th_img, 55, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> th_contor = connector::get_contours(th_img);
        for (int i = 0; i < th_contor.size(); i++) {
            cv::Rect rect = cv::boundingRect(th_contor[i]);
            if (rect.width > th_img.cols - 40) {
                start.Y = rect.y + rect.height;
                end.Y = rect.y + rect.height;
                break;
            }
        }
        bot_line_ret = Tival::FindLine::Run(th_img, start, end, bot_line_params);
        if (bot_line_ret.start_point.size() <= 0) {
            singal_lc.top_line.startPointX = 0;
            singal_lc.top_line.startPointY = 0;
            singal_lc.top_line.endPointX = 0;
            singal_lc.top_line.endPointY = 0;
            return std::vector<cv::Point2f>();
        }

    }
    if (bot_line_ret.start_point.size() <= 0)
    {
        //修改起始点在测试一次
        cv::Mat th_img;
        cv::threshold(img1, th_img, 55, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> th_contor = connector::get_contours(th_img);
        for (int i = 0; i < th_contor.size(); i++) {
            cv::Rect rect = cv::boundingRect(th_contor[i]);
            if (rect.width > th_img.cols - 40) {
                start.Y = rect.y + rect.height;
                end.Y = rect.y + rect.height;
                break;
            }
        }
        //减低对比度
        bot_line_params["Contrast"] = 30;
        bot_line_ret = Tival::FindLine::Run(img1, start, end, bot_line_params);

        //找错了再找一遍
        if (bot_line_ret.start_point.size() == 1 && (bot_line_ret.start_point[0].y < 13 || abs(bot_line_ret.start_point[0].y - bot_line_ret.end_point[0].y)>5)) {
            bot_line_ret = Tival::FindLine::Run(th_img, start, end, bot_line_params);
        }

        if (bot_line_ret.start_point.size() <= 0) {
            singal_lc.top_line.startPointX = 0;
            singal_lc.top_line.startPointY = 0;
            singal_lc.top_line.endPointX = 0;
            singal_lc.top_line.endPointY = 0;
            return std::vector<cv::Point2f>();
        }


    }

    cv::Point2f p1, p2;
    p1 = bot_line_ret.start_point[0];
    p2 = bot_line_ret.end_point[0];

    double t0 = connector::get_line_y(p1, p2, 0);
    double t1 = connector::get_line_y(p1, p2, img1.cols);

    singal_lc.top_line.startPointX = 0;
    singal_lc.top_line.startPointY = t0;
    singal_lc.top_line.endPointX = img1.cols;
    singal_lc.top_line.endPointY = t1;

    singal_lc.top_line.startPointX = singal_lc.top_line.startPointX + cur_rect.x;
    singal_lc.top_line.startPointY = singal_lc.top_line.startPointY + cur_rect.y;
    singal_lc.top_line.endPointX = singal_lc.top_line.endPointX + cur_rect.x;
    singal_lc.top_line.endPointY = singal_lc.top_line.endPointY + cur_rect.y;

    return std::vector<cv::Point2f>{cv::Point2f(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point2f(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)};

}
std::vector<cv::Point2f> W_Female_Detect::find_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info_2& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2) {
    if (is_left == 1) {
        singal_lc.template_line_vec[0].resize(2);
    }
    if (is_left == 0) {
        singal_lc.template_line_vec[3].resize(2);
    }

    std::vector<cv::Mat> find_box_hsv_1;
    std::vector<cv::Mat> find_box_hsv_2;
    find_box_hsv_1.emplace_back(hsv_v1[0](cur_rect));
    find_box_hsv_1.emplace_back(hsv_v1[1](cur_rect));
    find_box_hsv_1.emplace_back(hsv_v1[2](cur_rect));
    find_box_hsv_2.emplace_back(hsv_v2[0](cur_rect));
    find_box_hsv_2.emplace_back(hsv_v2[1](cur_rect));
    find_box_hsv_2.emplace_back(hsv_v2[2](cur_rect));

    KeyLine k_zero;
    k_zero.startPointX = 0;
    k_zero.startPointY = 0;
    k_zero.endPointX = 0;
    k_zero.endPointY = 0;

    //判断弹片露头,露头则是上半部较亮
    //下部分没有亮斑视为缺失
    cv::Mat th_img_1;
    bool ng = false;
    bool ng_botom = true;
    //左侧有干扰
    cv::Mat process_img = img1(cv::Rect(5, 0, img1.cols - 10, img1.rows));
    //分开上下部分计算阈值
    cv::Mat process_top_img = process_img(cv::Rect(0, 0, process_img.cols, process_img.rows / 2));
    cv::Mat process_bot_img = process_img(cv::Rect(0, process_img.rows / 2, process_img.cols, process_img.rows / 2));

    //正常情况底部有亮斑，亮斑不亮的情况下，降低阈值，都没有的话，弹片缺失
    // 测试下部分亮度
    //默认阈值为120，当下部分有较亮的区域，将阈值提升到180
    int th_value = 120;
    cv::Mat process_bot_th_img;
    cv::threshold(process_bot_img, process_bot_th_img, 180, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> process_bot_th_img_contours = connector::get_contours(process_bot_th_img);
    for (int i = 0; i < process_bot_th_img_contours.size(); i++) {

        cv::Mat mask = cv::Mat::zeros(process_bot_img.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> draw_conts = { process_bot_th_img_contours[i] };
        cv::drawContours(mask, draw_conts, 0, 255, -1);
        double area = cv::countNonZero(mask);
        if (area > 20) {
            th_value = 180;
            break;
        }
    }


    cv::threshold(process_img, th_img_1, th_value, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 3));
    cv::dilate(th_img_1, th_img_1, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(th_img_1, th_img_1, kernel);

    std::vector<std::vector<cv::Point>> th_img_1_contours = connector::get_contours(th_img_1);
    for (int i = 0; i < th_img_1_contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(th_img_1_contours[i]);
        cv::Point2f c_pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
        cv::Mat mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> draw_conts = { th_img_1_contours[i] };
        cv::drawContours(mask, draw_conts, 0, 255, -1);
        double area = cv::countNonZero(mask);

        //if (area < 25 ) continue;
        //亮斑在上部分
        if (area > 15 && c_pt.y < img1.rows / 2 + 5 && rect.height < rect.width * 2 && rect.width>5 && rect.height>3) {
            ng = true;
            break;
        }
        //下半部分有亮斑,下半部分要控距
        double d = img1.rows - c_pt.y;
        if (d <= dis_p_ && area > 10) {
            ng_botom = false;
        }
    }

    if (ng || ng_botom) {
        if (is_left == 1) {
            singal_lc.template_rect[0] = cur_rect;
            singal_lc.template_line_vec[0][0] = k_zero;
            singal_lc.template_line_vec[0][1] = k_zero;
            singal_lc.e1 = e_ + error_e_ + 0.01;
            singal_lc.p1 = p_ + error_p_ + 0.01;
        }
        if (is_left == 0) {
            singal_lc.template_rect[3] = cur_rect;
            singal_lc.template_line_vec[3][0] = k_zero;
            singal_lc.template_line_vec[3][1] = k_zero;
            singal_lc.e2 = e_ + error_e_ + 0.01;
            singal_lc.p2 = p_ + error_p_ + 0.01;
        }
        return std::vector<cv::Point2f>();
    }



    //cv::Mat grama_img = connector::gamma_trans(img2, 0.8);
    //cv::Mat soble_img_x, sobel_img_y, edge, edge_th;
    //cv::Sobel(grama_img, sobel_img_y, CV_16S, 0, 1, 3);
    //cv::Sobel(grama_img, soble_img_x, CV_16S, 1, 0, 3);
    //edge = soble_img_x + sobel_img_y;
    //cv::convertScaleAbs(edge, edge);
    cv::Mat  edge_th;
    cv::threshold(find_box_hsv_2[1], edge_th, 65, 255, cv::THRESH_BINARY);
    std::vector<int> his_1 = get_his(edge_th, 1, 0);

    std::vector<int> row_diff;
    row_diff.resize(his_1.size());
    for (size_t i = 1; i < his_1.size(); i++)
        row_diff[i] = abs(his_1[i] - his_1[i - 1]);


    int tv = 0;
    //直接在目标范围找
    for (size_t i = 10; i < 30; i++) {
        if ((row_diff[i] < 3 && row_diff[i + 1]>10) || (row_diff[i] <= 10 && row_diff[i + 1]>=20)) {
            tv = i + 1;
            break;
        }
    }
    //判断找到没
    if (tv == 0) {
        for (size_t i = 0; i < row_diff.size(); i++) {
            if (row_diff[i] >= 40) {
                tv = i;
                break;
            }
        }
    }
    //超过预期位置再找一遍
    if (tv >= 30 || tv<10) {
        for (size_t i = 0; i < row_diff.size(); i++) {
            if (row_diff[i] >= 23) {
                tv = i;
                break;
            }
        }
    }
    int bv = 0;
    KeyLine le, lp;
    le.startPointX = 0;
    le.startPointY = tv;
    le.endPointX = img2.cols;
    le.endPointY = tv;

    le.startPointX = le.startPointX + cur_rect.x;
    le.startPointY = le.startPointY + cur_rect.y;
    le.endPointX = le.endPointX + cur_rect.x;
    le.endPointY = le.endPointY + cur_rect.y;

    lp.startPointX = 0;
    lp.startPointY = img2.rows;
    lp.endPointX = img2.cols;
    lp.endPointY = img2.rows;

    lp.startPointX = lp.startPointX + cur_rect.x;
    lp.startPointY = lp.startPointY + cur_rect.y;
    lp.endPointX = lp.endPointX + cur_rect.x;
    lp.endPointY = lp.endPointY + cur_rect.y;


    //左边的弹片
    if (is_left == 1) {
        singal_lc.template_rect[0] = cur_rect;
        singal_lc.template_line_vec[0][0] = le;
        singal_lc.template_line_vec[0][1] = lp;
        singal_lc.e1 = connector::dist_p2l(cv::Point(le.startPointX, le.startPointY), cv::Point(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)) * pix_value_ / 1000;
        singal_lc.p1 = 0;
    }
    //右边的弹片
    if (is_left == 0) {
        singal_lc.template_rect[3] = cur_rect;
        singal_lc.template_line_vec[3][0] = le;
        singal_lc.template_line_vec[3][1] = lp;
        singal_lc.e2 = connector::dist_p2l(cv::Point(le.startPointX, le.startPointY), cv::Point(singal_lc.top_line.startPointX, singal_lc.top_line.startPointY), cv::Point(singal_lc.top_line.endPointX, singal_lc.top_line.endPointY)) * pix_value_ / 1000;
        singal_lc.p2 = 0;
    }
    return std::vector<cv::Point2f>();
}
std::vector<cv::Point2f> W_Female_Detect::find_location_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info_2& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2) {

    cv::Mat grama_img = connector::gamma_trans(img2, 0.8);
    //合成原图
    cv::Mat mergedImage;
    cv::merge(hsv_v2, mergedImage);
    cv::Mat bgrImage;
    cv::cvtColor(mergedImage, bgrImage, cv::COLOR_HSV2BGR);
    cv::Rect roi_rect(cur_rect.x - 5, cur_rect.y - 5, cur_rect.width + 10, cur_rect.height + 10);
    cv::Mat roi_img = bgrImage(roi_rect);
    //保存图片
    //std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + ".jpg";
    //cv::imwrite(file_name, roi_img);
    //g_conut++;

    //svm 检测
    bool ng_location_box = false;
    std::vector<cv::Mat> test_img_vec;
    test_img_vec.push_back(roi_img);
    nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 8, cv::Size(88, 34), 1);
    cv::Mat temp_feature = test_transform();
    double prob[2];
    double ret = svm_obj.testFeatureLibSVM(temp_feature, prob);
    if (prob[1] > 0.72) {
        //第二个概率大于0.8表示不正常
        ng_location_box = true;
    }

    //先左右分割检测有无金属弹片
    //cv::Mat l_img = grama_img(cv::Rect(7, 5, img2.cols / 2 - 3, img2.rows - 10));
    //cv::Mat r_img = grama_img(cv::Rect(img2.cols / 2 + 3, 5, img2.cols / 2 - 12, img2.rows - 10));

    ////分上中线三段处理
    ////左右各分六份
    //int l_l_t_mean_value = cv::mean(l_img(cv::Rect(0, 0, l_img.cols / 2, l_img.rows / 3)))[0];
    //int l_l_m_mean_value = cv::mean(l_img(cv::Rect(0, l_img.rows / 3, l_img.cols / 2, l_img.rows / 3)))[0];
    //int l_l_b_mean_value = cv::mean(l_img(cv::Rect(0, l_img.rows / 3 * 2, l_img.cols / 2, l_img.rows / 3)))[0];
    //int l_r_t_mean_value = cv::mean(l_img(cv::Rect(l_img.cols / 2, 0, l_img.cols / 2, l_img.rows / 3)))[0];
    //int l_r_m_mean_value = cv::mean(l_img(cv::Rect(l_img.cols / 2, l_img.rows / 3, l_img.cols / 2, l_img.rows / 3)))[0];
    //int l_r_b_mean_value = cv::mean(l_img(cv::Rect(l_img.cols / 2, l_img.rows / 3 * 2, l_img.cols / 2, l_img.rows / 3)))[0];
    //std::vector<int> l_mean{ l_l_t_mean_value ,l_l_m_mean_value ,l_l_b_mean_value ,l_r_t_mean_value ,l_r_m_mean_value ,l_r_b_mean_value };
    ////右侧的6份
    //int r_l_t_mean_value = cv::mean(r_img(cv::Rect(0, 0, r_img.cols / 2, r_img.rows / 3)))[0];
    //int r_l_m_mean_value = cv::mean(r_img(cv::Rect(0, r_img.rows / 3, r_img.cols / 2, r_img.rows / 3)))[0];
    //int r_l_b_mean_value = cv::mean(r_img(cv::Rect(0, r_img.rows / 3 * 2, r_img.cols / 2, r_img.rows / 3)))[0];
    //int r_r_t_mean_value = cv::mean(r_img(cv::Rect(r_img.cols / 2, 0, r_img.cols / 2, r_img.rows / 3)))[0];
    //int r_r_m_mean_value = cv::mean(r_img(cv::Rect(r_img.cols / 2, r_img.rows / 3, r_img.cols / 2, r_img.rows / 3)))[0];
    //int r_r_b_mean_value = cv::mean(r_img(cv::Rect(r_img.cols / 2, r_img.rows / 3 * 2, r_img.cols / 2, r_img.rows / 3)))[0];

    //std::vector<int> r_mean{ r_l_t_mean_value ,r_l_m_mean_value ,r_l_b_mean_value ,r_r_t_mean_value ,r_r_m_mean_value ,r_r_b_mean_value };
    if (is_left == 1) {
        singal_lc.template_rect[1] = cur_rect;
        singal_lc.template_line_vec[1].resize(9);
    }
    if (is_left == 0) {
        singal_lc.template_line_vec[2].resize(9);
        singal_lc.template_rect[2] = cur_rect;
    }

    KeyLine k_zero;
    k_zero.startPointX = 0;
    k_zero.startPointY = 0;
    k_zero.endPointX = 0;
    k_zero.endPointY = 0;

    //检测弹片中间缺失的部分,假设中间部分未缺失
    //bool ng_mid = false;
    //int mid_th_value = (l_r_m_mean_value + r_l_m_mean_value) / 2;
    //cv::Mat mid_img = grama_img(cv::Rect(grama_img.cols/4,0, grama_img.cols/2, grama_img.rows));
    //cv::Mat mid_th_img;
    //cv::threshold(mid_img, mid_th_img, mid_th_value,255,cv::THRESH_BINARY_INV);
    //double area = cv::countNonZero(mid_th_img);
    //if (area >750 ) {
    //    ng_mid = true;
    //   
    //}



    //中间弹片有缺失
    //弹片中部不缺失，上下部位与中间对比
    //上下差异大，上中 下中差异大
    ////上端无反光，如果上侧亮度大视为缺失
    //if (l_l_m_mean_value > 200 ||r_r_m_mean_value > 200 ||(abs(l_l_m_mean_value - r_r_m_mean_value) > 130) ||ng_location_box
    //    ) {
    //    //TODO
    //    if (is_left == 1) {
    //        singal_lc.template_line_vec[1][0] = k_zero;
    //        singal_lc.template_line_vec[1][1] = k_zero;
    //        singal_lc.template_line_vec[1][2] = k_zero;
    //        singal_lc.template_line_vec[1][3] = k_zero;
    //        singal_lc.template_line_vec[1][4] = k_zero;
    //        singal_lc.template_line_vec[1][5] = k_zero;
    //        singal_lc.template_line_vec[1][6] = k_zero;
    //        singal_lc.template_line_vec[1][7] = k_zero;
    //        singal_lc.template_line_vec[1][8] = k_zero;
    //        singal_lc.a1 = a_ + error_a_ + 0.01;
    //        singal_lc.c11 = c1_ + error_c1_ + 0.01;
    //        singal_lc.c12 = c2_ + error_c2_ + 0.01;
    //        singal_lc.d11 = d1_ + error_d1_ + 0.01;
    //        singal_lc.d12 = d2_ + error_d2_ + 0.01;
    //    }
    //    if (is_left == 0) {
    //        singal_lc.template_line_vec[2][0] = k_zero;
    //        singal_lc.template_line_vec[2][1] = k_zero;
    //        singal_lc.template_line_vec[2][2] = k_zero;
    //        singal_lc.template_line_vec[2][3] = k_zero;
    //        singal_lc.template_line_vec[2][4] = k_zero;
    //        singal_lc.template_line_vec[2][5] = k_zero;
    //        singal_lc.template_line_vec[2][6] = k_zero;
    //        singal_lc.template_line_vec[2][7] = k_zero;
    //        singal_lc.template_line_vec[2][8] = k_zero;

    //        singal_lc.a2 = a_ + error_a_ + 0.01;
    //        singal_lc.c21 = c1_ + error_c1_ + 0.01;
    //        singal_lc.c22 = c2_ + error_c2_ + 0.01;
    //        singal_lc.d21 = d1_ + error_d1_ + 0.01;
    //        singal_lc.d22 = d2_ + error_d2_ + 0.01;
    //    }
    //    return std::vector<cv::Point2f>();
    //}
    
    if (is_left == 1) {
        singal_lc.template_line_vec[1][0] = k_zero;
        singal_lc.template_line_vec[1][1] = k_zero;
        singal_lc.template_line_vec[1][2] = k_zero;
        singal_lc.template_line_vec[1][3] = k_zero;
        singal_lc.template_line_vec[1][4] = k_zero;
        singal_lc.template_line_vec[1][5] = k_zero;
        singal_lc.template_line_vec[1][6] = k_zero;
        singal_lc.template_line_vec[1][7] = k_zero;
        singal_lc.template_line_vec[1][8] = k_zero;
        singal_lc.a1  = 0;
        singal_lc.c11 = 0;
        singal_lc.c12 = 0;
        singal_lc.d11 = 0;
        singal_lc.d12 = 0;
    }
    if (is_left == 0) {
        singal_lc.template_line_vec[2][0] = k_zero;
        singal_lc.template_line_vec[2][1] = k_zero;
        singal_lc.template_line_vec[2][2] = k_zero;
        singal_lc.template_line_vec[2][3] = k_zero;
        singal_lc.template_line_vec[2][4] = k_zero;
        singal_lc.template_line_vec[2][5] = k_zero;
        singal_lc.template_line_vec[2][6] = k_zero;
        singal_lc.template_line_vec[2][7] = k_zero;
        singal_lc.template_line_vec[2][8] = k_zero;

        singal_lc.a2  = 0;
        singal_lc.c21 = 0;
        singal_lc.c22 = 0;
        singal_lc.d21 = 0;
        singal_lc.d22 = 0;
    }
    if (ng_location_box) {

        if (is_left == 1) {
            singal_lc.a1 = a_ + error_a_ + 0.01;
            singal_lc.c11 = c1_ + error_c1_ + 0.01;
            singal_lc.c12 = c2_ + error_c2_ + 0.01;
            singal_lc.d11 = d1_ + error_d1_ + 0.01;
            singal_lc.d12 = d2_ + error_d2_ + 0.01;
        }
        if (is_left == 0) {
            singal_lc.a2 = a_ + error_a_ + 0.01;
            singal_lc.c21 = c1_ + error_c1_ + 0.01;
            singal_lc.c22 = c2_ + error_c2_ + 0.01;
            singal_lc.d21 = d1_ + error_d1_ + 0.01;
            singal_lc.d22 = d2_ + error_d2_ + 0.01;
        }
        return std::vector<cv::Point2f>();
    }
    //8条线分别是
    /*
    0，图像的中线
    1，缝的中线
    2 左侧上线 c11
    3 左侧下线 d11
    4 右侧上线 c12
    5 右侧下线 d12
    6 中线 b1
    7 中线 b2
    8 中线 b3
    */
    KeyLine k0, k1, k2, k3, k4, k5, k6, k7, k8;

    //先计算中线的差值
    //矩形框的中线
    double rect_mid_value = cur_rect.width / 2;
    k0.startPointX = rect_mid_value + cur_rect.x;
    k0.startPointY = 0 + cur_rect.y;
    k0.endPointX = rect_mid_value + cur_rect.x;
    k0.endPointY = img2.rows + cur_rect.y;

    double img_mid_value = 0;
    get_img_mid_value(grama_img, img_mid_value);
    k1.startPointX = img_mid_value + cur_rect.x;
    k1.startPointY = 0 + cur_rect.y;
    k1.endPointX = img_mid_value + cur_rect.x;
    k1.endPointY = img2.rows + cur_rect.y;

    //分为左右两张图，测量 上下边界
    int c11, c12, d11, d12;
    get_top_bottom_dege(grama_img, 1, c11, d11);
    get_top_bottom_dege(grama_img, 0, c12, d12);

    k2.startPointX = 0 + cur_rect.x;
    k2.startPointY = c11 + cur_rect.y;
    k2.endPointX = img2.cols / 2 + cur_rect.x;
    k2.endPointY = c11 + cur_rect.y;

    k3.startPointX = 0 + cur_rect.x;
    k3.startPointY = d11 + cur_rect.y;
    k3.endPointX = img2.cols / 2 + cur_rect.x;
    k3.endPointY = d11 + cur_rect.y;

    k4.startPointX = img2.cols / 2 + cur_rect.x;
    k4.startPointY = c12 + cur_rect.y;
    k4.endPointX = img2.cols + cur_rect.x;
    k4.endPointY = c12 + cur_rect.y;

    k5.startPointX = img2.cols / 2 + cur_rect.x;
    k5.startPointY = d12 + cur_rect.y;
    k5.endPointX = img2.cols + cur_rect.x;
    k5.endPointY = d12 + cur_rect.y;

    //计算中间的短横线
    cv::Point2f b1s;
    cv::Point2f b2s;
    cv::Point2f b3s;
    cv::Point2f b1e;
    cv::Point2f b2e;
    cv::Point2f b3e;

    get_img_mid_thr_line(grama_img, img_mid_value, b1s, b2s, b3s, b1e, b2e, b3e);

    k6.startPointX = b1s.x + cur_rect.x;
    k6.startPointY = b1s.y + cur_rect.y;
    k6.endPointX = b1e.x + cur_rect.x;
    k6.endPointY = b1e.y + cur_rect.y;

    k7.startPointX = b2s.x + cur_rect.x;
    k7.startPointY = b2s.y + cur_rect.y;
    k7.endPointX = b2e.x + cur_rect.x;
    k7.endPointY = b2e.y + cur_rect.y;

    k8.startPointX = b3s.x + cur_rect.x;
    k8.startPointY = b3s.y + cur_rect.y;
    k8.endPointX = b3e.x + cur_rect.x;
    k8.endPointY = b3e.y + cur_rect.y;

    if (is_left == 1) {
       /* singal_lc.template_line_vec[1][0] = k_zero;
        singal_lc.template_line_vec[1][1] = k_zero;
        singal_lc.template_line_vec[1][2] = k2;
        singal_lc.template_line_vec[1][3] = k3;
        singal_lc.template_line_vec[1][4] = k4;
        singal_lc.template_line_vec[1][5] = k5;
        singal_lc.template_line_vec[1][6] = k6;
        singal_lc.template_line_vec[1][7] = k7;
        singal_lc.template_line_vec[1][8] = k8;*/
        singal_lc.a1 = abs(img_mid_value - rect_mid_value) * pix_value_ / 1000;
        singal_lc.c11 = abs(c11) * pix_value_ / 1000;
        singal_lc.c12 = abs(c12) * pix_value_ / 1000;
        singal_lc.d11 = abs(d11 - grama_img.rows) * pix_value_ / 1000;
        singal_lc.d12 = abs(d12 - grama_img.rows) * pix_value_ / 1000;

        double e_a1 = abs(singal_lc.a1 - a_);
        double e_c11 = abs(singal_lc.c11 - c1_);
        double e_c12 = abs(singal_lc.c12 - c2_);
        double e_d11 = abs(singal_lc.d11 - d1_);
        double e_d12 = abs(singal_lc.d12 - d2_);
        //ng的纠正为正常
        if (e_a1 > error_a_)  singal_lc.a1 = error_a_ - 0.01;
        if (e_c11 > error_c1_)  singal_lc.c11 = error_c1_ - 0.01;
        if (e_c12 > error_c2_) singal_lc.c12 = error_c2_ - 0.01;
        if (e_d11 > error_d1_) singal_lc.d11 = error_d1_ - 0.01;
        if (e_d12 > error_d2_) singal_lc.d12 = error_d2_ - 0.01;

    }
    if (is_left == 0) {
       /* singal_lc.template_line_vec[2][0] = k_zero;
        singal_lc.template_line_vec[2][1] = k_zero;
        singal_lc.template_line_vec[2][2] = k2;
        singal_lc.template_line_vec[2][3] = k3;
        singal_lc.template_line_vec[2][4] = k4;
        singal_lc.template_line_vec[2][5] = k5;
        singal_lc.template_line_vec[2][6] = k6;
        singal_lc.template_line_vec[2][7] = k7;
        singal_lc.template_line_vec[2][8] = k8;*/
        singal_lc.a2 = abs(img_mid_value - rect_mid_value) * pix_value_ / 1000;
        singal_lc.c21 = abs(c11) * pix_value_ / 1000;
        singal_lc.c22 = abs(c12) * pix_value_ / 1000;
        singal_lc.d21 = abs(d11 - grama_img.rows) * pix_value_ / 1000;
        singal_lc.d22 = abs(d12 - grama_img.rows) * pix_value_ / 1000;

        double e_a1 = abs(singal_lc.a2 - a_);
        double e_c11 = abs(singal_lc.c21 - c1_);
        double e_c12 = abs(singal_lc.c22 - c2_);
        double e_d11 = abs(singal_lc.d21 - d1_);
        double e_d12 = abs(singal_lc.d22 - d2_);
        //ng的纠正为正常
        if (e_a1 > error_a_)  singal_lc.a2 = error_a_ - 0.01;
        if (e_c11 > error_c1_)  singal_lc.c21 = error_c1_ - 0.01;
        if (e_c12 > error_c2_) singal_lc.c22 = error_c2_ - 0.01;
        if (e_d11 > error_d1_) singal_lc.d21 = error_d1_ - 0.01;
        if (e_d12 > error_d2_) singal_lc.d22 = error_d2_ - 0.01;
    }
    return std::vector<cv::Point2f>();
}
void W_Female_Detect::data_cvt_5(std::vector<lc_info_2> lc_vec, AlgoResultPtr algo_result) {
    
    status_flag = true;
    for (int i = 0; i < lc_vec.size(); i++) {
        lc_info_2 tmp_lc = lc_vec[i];
        //每个小单元进行处理

        cv::Point2f org_pc;
        //模板框的位置
        for (int j = 0; j < tmp_lc.template_rect.size(); j++) {
            cv::Rect tp_rect = tmp_lc.template_rect[j];
            cv::Point2f lt = tp_rect.tl();
            cv::Point2f lb(tp_rect.tl().x, tp_rect.tl().y + tp_rect.height);
            cv::Point2f rt(tp_rect.br().x, tp_rect.br().y - tp_rect.height);
            cv::Point2f rb = tp_rect.br();

            if (j == 1) {
                cv::Point2f pc(tp_rect.x + tp_rect.width / 2, tp_rect.y + tp_rect.height / 2);
                org_pc = connector::TransPoint(tmp_lc.inv_h, pc);
            }
            cv::Point2f org_lt = connector::TransPoint(tmp_lc.inv_h, lt);
            cv::Point2f org_lb = connector::TransPoint(tmp_lc.inv_h, lb);
            cv::Point2f org_rt = connector::TransPoint(tmp_lc.inv_h, rt);
            cv::Point2f org_rb = connector::TransPoint(tmp_lc.inv_h, rb);
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "polygon" },
                    { "points", { { org_lt.x, org_lt.y }, { org_rt.x, org_rt.y }, { org_rb.x, org_rb.y }, { org_lb.x, org_lb.y } } },
                    { "result", 1 } });
        }

        //顶部线段的位置
        if (tmp_lc.top_line.startPointX != 0 && tmp_lc.top_line.endPointX != 0)
        {
            cv::Point2f p1_top = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_lc.top_line.startPointX, tmp_lc.top_line.startPointY));
            cv::Point2f p2_top = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_lc.top_line.endPointX, tmp_lc.top_line.endPointY));
            algo_result->result_info.push_back(
                { { "label", "fuzhu" },
                    { "shapeType", "line" },
                    { "points", { { p1_top.x, p1_top.y }, { p2_top.x, p2_top.y } } },
                    { "result", 1 } });
        }
        else {
            algo_result->judge_result = 0;
            continue;
        }

        //每个框里面的线段
        for (int j = 0; j < tmp_lc.template_line_vec.size(); j++) {
            for (int k = 0; k < tmp_lc.template_line_vec[j].size(); k++) {
                KeyLine tmp_l = tmp_lc.template_line_vec[j][k];
                if (tmp_l.startPointX < 1 || tmp_l.endPointX < 1 || std::isnan(tmp_l.startPointX) || std::isnan(tmp_l.endPointX)) {
                    continue;
                }
                cv::Point2f p3 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.startPointX, tmp_l.startPointY));
                cv::Point2f p4 = connector::TransPoint(tmp_lc.inv_h, cv::Point2f(tmp_l.endPointX, tmp_l.endPointY));
                algo_result->result_info.push_back(
                    { { "label", "fuzhu" },
                        { "shapeType", "line" },
                        { "points", { { p3.x, p3.y }, { p4.x, p4.y } } },
                        { "result", 1 } });
                /*LOGI("id {} rect idx j {} k {} p3 line x: {}, line y:{}", tmp_lc.index, j,k,p3.x, p3.y);
                LOGI("id {} rect idx j {} k {} p4 line x: {}, line y:{}", tmp_lc.index, j,k,p4.x, p4.y);*/

            }
        }

        //计算数值比较
        double status_al, status_b11, status_b12, status_b13, status_c11, status_c12, status_d11, status_d12, status_e1, status_p1;
        double status_a2, status_b21, status_b22, status_b23, status_c21, status_c22, status_d21, status_d22, status_e2, status_p2;


        double e_a1 = abs(tmp_lc.a1 - a_);
        double e_b11 = abs(tmp_lc.b11 - b1_);
        double e_b12 = abs(tmp_lc.b12 - b2_);
        double e_b13 = abs(tmp_lc.b13 - b3_);
        double e_c11 = abs(tmp_lc.c11 - c1_);
        double e_c12 = abs(tmp_lc.c12 - c2_);
        double e_d11 = abs(tmp_lc.d11 - d1_);
        double e_d12 = abs(tmp_lc.d12 - d2_);
        double e_e1 = abs(tmp_lc.e1 - e_);
        double e_p1 = abs(tmp_lc.p1 - p_);
        

        double e_a2 = abs(tmp_lc.a2 - a_);
        double e_b21 = abs(tmp_lc.b21 - b1_);
        double e_b22 = abs(tmp_lc.b22 - b2_);
        double e_b23 = abs(tmp_lc.b23 - b3_);
        double e_c21 = abs(tmp_lc.c21 - c1_);
        double e_c22 = abs(tmp_lc.c22 - c2_);
        double e_d21 = abs(tmp_lc.d21 - d1_);
        double e_d22 = abs(tmp_lc.d22 - d2_);
        double e_e2 = abs(tmp_lc.e2 - e_);
        double e_p2 = abs(tmp_lc.p2 - p_);
        

        status_al = e_a1 <= error_a_ ? 1 : 0;
        status_b11 = e_b11 <= error_b1_ ? 1 : 0;
        status_b12 = e_b12 <= error_b2_ ? 1 : 0;
        status_b13 = e_b13 <= error_b3_ ? 1 : 0;
        status_c11 = e_c11 <= error_c1_ ? 1 : 0;
        status_c12 = e_c12 <= error_c2_ ? 1 : 0;
        status_d11 = e_d11 <= error_d1_ ? 1 : 0;
        status_d12 = e_d12 <= error_d2_ ? 1 : 0;
        status_e1 = e_e1 <= error_e_ ? 1 : 0;
        status_p1 = e_p1 <= error_p_ ? 1 : 0;
        

        status_a2 = e_a2 <= error_a_ ? 1 : 0;
        status_b21 = e_b21 <= error_b1_ ? 1 : 0;
        status_b22 = e_b22 <= error_b2_ ? 1 : 0;
        status_b23 = e_b23 <= error_b3_ ? 1 : 0;
        status_c21 = e_c21 <= error_c1_ ? 1 : 0;
        status_c22 = e_c22 <= error_c2_ ? 1 : 0;
        status_d21 = e_d21 <= error_d1_ ? 1 : 0;
        status_d22 = e_d22 <= error_d2_ ? 1 : 0;
        status_e2 = e_e2 <= error_e_ ? 1 : 0;
        status_p2 = e_p2 <= error_p_ ? 1 : 0;
        
        if (status_al < 1 || status_b11 < 1 || status_b12 < 1 || status_b13 < 1 || status_c11 < 1 || status_c12 < 1 || status_d11 < 1 || status_d12 < 1 || status_e1 < 1 || status_p1 < 1  ||
            status_a2 < 1 || status_b21 < 1 || status_b22 < 1 || status_b23 < 1 || status_c21 < 1 || status_c22 < 1 || status_d21 < 1 || status_d22 < 1 || status_e2 < 1 || status_p2 < 1 ) {
            status_flag = false;
        }
        else {
        }
        if (!status_flag) {
            algo_result->judge_result = 0;
        }
        else {
            algo_result->judge_result = 1;
        }

        algo_result->result_info.push_back(
            { { "label", "W_Female_Detect_defect_1" },
                { "shapeType", "basis" },
                { "points", { { -1, -1 } } },
                { "result", { { "dist", { tmp_lc.a1,
            tmp_lc.b11,
            tmp_lc.b12,
            tmp_lc.b13,
            tmp_lc.c11,
            tmp_lc.c12,
            tmp_lc.d11,
            tmp_lc.d12,
            tmp_lc.e1,
            tmp_lc.p1,
            tmp_lc.a2,
            tmp_lc.b21,
            tmp_lc.b22,
            tmp_lc.b23,
            tmp_lc.c21,
            tmp_lc.c22,
            tmp_lc.d21,
            tmp_lc.d22,
            tmp_lc.e2,
            tmp_lc.p2 } },
    { "status", {
            status_al,
            status_b11,
            status_b12,
            status_b13,
            status_c11,
            status_c12,
            status_d11,
            status_d12,
            status_e1,
            status_p1,
            status_a2,
            status_b21,
            status_b22,
            status_b23,
            status_c21,
            status_c22,
            status_d21,
            status_d22,
            status_e2,
            status_p2} },
       { "error", {
            e_a1,
            e_b11,
            e_b12,
            e_b13,
            e_c11,
            e_c12,
            e_d11,
            e_d12,
            e_e1,
            e_p1,
            e_a2,
            e_b21,
            e_b22,
            e_b23,
            e_c21,
            e_c22,
            e_d21,
            e_d22,
            e_e2,
            e_p2} }  ,
     { "index", (int)tmp_lc.index },
     { "points", { {org_pc.x,org_pc.y} } } } } });
    }
}

//单体乳白开口
void W_Female_Detect::img_process_6(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept {
    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 100;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::dilate(th_img_1, th_img_1, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;

    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 130 || rect.height > 70)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }

    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;
#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 140 || rect.height > 70 || rect.width <= 70)continue;
        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 140)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        /*cv::drawContours(gray_mask, draw_conts, 0, 255, -1);*/
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }

    std::vector<W_Female_Detect::w_female_2> w_female_vec;


    // 同一行矩形的相对关系
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 100, 44),
        cv::Vec4i(142, 0, 90, 44),
        cv::Vec4i(298, 0, 90, 44),
        cv::Vec4i(431, 0, 100, 44),
        cv::Vec4i(575, 0, 90, 44),
        cv::Vec4i(731, 0, 90, 44),
        cv::Vec4i(865, 0, 99, 44),
        cv::Vec4i(1007, 0, 90, 44),
        cv::Vec4i(1162, 0, 90, 44),
        cv::Vec4i(1297, 0, 99, 44),
        cv::Vec4i(1439, 0, 90, 44),
        cv::Vec4i(1595, 0, 90, 44),
        cv::Vec4i(1730, 0, 99, 44),
    };

    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            ;
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }

        for (int i = 0; i < rank.size(); i++) {
            // 每一行进行处理,最后一行特殊处理
            bool estimate_flag = false;
            if (!estimate_flag)
            {
                // 当前行未找全的，特殊处理，全部特殊处理
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                double distance = 0;
                //13 *10 的固定距离
                static std::vector<int> s_numbers = { 0,144, 300, 432, 576, 731, 864,1008, 1164,1297,1441,1596,1729 };

                if (i % 2 == 0) {
                    // 偶数行
                    //50 是对一个空洞到roi 的边缘距离
                    distance = (s_rect.x - 50 - detect_left_x_);
                }
                if (i % 2 == 1) {
                    // 奇数行
                    // 120 的基础是奇数行与偶数行的第一个空洞的间距
                    distance = (s_rect.x - 170 - detect_left_x_);
                }
                auto s_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                s_col_idx = std::distance(s_numbers.begin(), s_it);

                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    // 当前黑孔的序号
                    if (i % 2 == 0) distance = (cur_rect.x - 50 - detect_left_x_);
                    if (i % 2 == 1) distance = (cur_rect.x - 170 - detect_left_x_);;

                    auto c_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                    c_col_idx = std::distance(s_numbers.begin(), c_it);
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }

    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    std::vector<cv::Mat> rbg_v1, rbg_v2;
    cv::split(input_img_1, rbg_v1);
    cv::split(input_img_2, rbg_v2);

#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i++) {
        w_female_2 singal_female = cal_6(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, rbg_v1, rbg_v2,algo_result, process_rect_vec[i], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        w_female_vec.push_back(singal_female);
        w_lock.unlock();
    }
    data_cvt_4(w_female_vec, algo_result);
   /* cv::Mat dis = src_2.clone();
    connector::draw_results(dis, algo_result->result_info);*/

}

W_Female_Detect::w_female_2 W_Female_Detect::cal_6(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, std::vector<cv::Mat> rbg_v1, std::vector<cv::Mat> rbg_v2, AlgoResultPtr algo_result, cv::Rect rect, int index, cv::Mat inv_m) {

    connector::Timer t1;
    //此处需要分类处理，后期在加
    w_female_2 singal_w_female;

    singal_w_female.template_rect = rect;
    singal_w_female.index = index;

    cv::Mat img_h1 = hsv_v1[0](rect);
    cv::Mat img_s1 = hsv_v1[1](rect);
    cv::Mat img_v1 = hsv_v1[2](rect);

    cv::Mat img_h2 = hsv_v2[0](rect);
    cv::Mat img_s2 = hsv_v2[1](rect);
    cv::Mat img_v2 = hsv_v2[2](rect);

    cv::Mat img_b1 = rbg_v1[0](rect);
    cv::Mat img_g1 = rbg_v1[1](rect);
    cv::Mat img_r1 = rbg_v1[2](rect);
    cv::Mat img_b2 = rbg_v2[0](rect);
    cv::Mat img_g2 = rbg_v2[1](rect);
    cv::Mat img_r2 = rbg_v2[2](rect);
   
    std::vector<cv::Mat> find_hsv1;
    std::vector<cv::Mat> find_hsv2;
    find_hsv1.emplace_back(img_h1);
    find_hsv1.emplace_back(img_s1);
    find_hsv1.emplace_back(img_v1);
    find_hsv2.emplace_back(img_h2);
    find_hsv2.emplace_back(img_s2);
    find_hsv2.emplace_back(img_v2);
   

    //合成原图
    cv::Mat mergedImage;
    cv::merge(hsv_v2, mergedImage);
    cv::Mat bgrImage;
    cv::cvtColor(mergedImage, bgrImage, cv::COLOR_HSV2BGR);
    cv::Rect roi_rect(rect.x - 5, rect.y - 5, rect.width + 10, rect.height + 10);
    cv::Mat roi_img = bgrImage(roi_rect);
    //保存图片
    //std::string file_name = "E:\\demo\\cxx\\connector_algo\\data\\hf\\" + std::to_string(g_conut) + ".jpg";
    //cv::imwrite(file_name, roi_img);
    //g_conut++;
    
    //svm 检测
    bool ng_location_box = false;
    std::vector<cv::Mat> test_img_vec;
    test_img_vec.push_back(roi_img);
    nao::img::feature::HogTransform test_transform(test_img_vec, 11, 8, 6, cv::Size(100, 55), 1);
    cv::Mat temp_feature = test_transform();
    double prob[2];
    double ret = svm_obj_kai.testFeatureLibSVM(temp_feature, prob);
    if (prob[1] > 0.8) {
        //第二个概率大于0.8表示不正常
        ng_location_box = true;
    }
    if (ng_location_box) {
        singal_w_female.a = a_ + error_a_ + 0.01;
        singal_w_female.b = b1_ + error_b1_ +0.01;
        singal_w_female.c = c1_ + error_c1_ +0.01;
        singal_w_female.d = d1_ + error_d1_ +0.01;
        return singal_w_female;
    }
   
    // 黑色框
    int col_idx = index % 13;
    int row_idx = index / 13;

    int left_value = 0;
    int right_value = 0;
    int top_value = 0;
    int bot_value = 0;
    KeyLine k1, k2, k3, k4;
   
    //分为 亮暗 不同的处理
    //亮的部分，亮的下部分边界默认使用下部
    //左右边界往里收缩到30%的位置作为边界,左右采用S通道
    
    //上下采用H通道分割
    if (col_idx == 0 || col_idx == 3 || col_idx == 6 || col_idx == 9 || col_idx == 12)
    {
        left_value = get_percent_edge_2(find_hsv1, find_hsv2, 0.3, 1, 2);
        right_value = get_percent_edge_2(find_hsv1, find_hsv2, 0.3, 1, 3);
        top_value = get_s_edge_2(find_hsv1, find_hsv2, 1, 50);
        bot_value = img_h2.rows;
    }
    //暗的部分，暗斑的下部分边界使用S图像
    if (col_idx == 1 || col_idx == 2 || col_idx == 4 || col_idx == 5 || col_idx == 7 || col_idx == 8 || col_idx == 10 || col_idx == 11)
    {
        left_value = get_percent_edge_2(find_hsv1, find_hsv2, 0.3, 1, 2);
        right_value = get_percent_edge_2(find_hsv1, find_hsv2, 0.3, 1, 3);
        top_value = 0;
        bot_value = img_h2.rows;
    }

    //还原位置到原图
    k1.startPointX = left_value + rect.x;
    k1.startPointY = rect.y;
    k1.endPointX = left_value + rect.x;
    k1.endPointY = rect.y + rect.height;

    k2.startPointX = right_value + rect.x;
    k2.startPointY = rect.y;
    k2.endPointX = right_value + rect.x;
    k2.endPointY = rect.y + rect.height;

    k3.startPointX = rect.x;
    k3.startPointY = top_value + rect.y;
    k3.endPointX = rect.x + rect.width;
    k3.endPointY = top_value + rect.y;


    k4.startPointX = rect.x;
    k4.startPointY = rect.y + bot_value;
    k4.endPointX = rect.x + rect.width;
    k4.endPointY = rect.y + bot_value;

    singal_w_female.line_vec.emplace_back(k1);
    singal_w_female.line_vec.emplace_back(k2);
    singal_w_female.line_vec.emplace_back(k3);
    singal_w_female.line_vec.emplace_back(k4);


    singal_w_female.a = left_value * pix_value_ / 1000;
    singal_w_female.b = std::abs(right_value - img_h2.cols) * pix_value_ / 1000;
    singal_w_female.c = top_value * pix_value_ / 1000;
    singal_w_female.d = std::abs(bot_value - img_h2.rows) * pix_value_ / 1000;

    return singal_w_female;

}

//乳白开口 左右按 s通道处理
//按列算，每一列计算的情况为1，每一行计算的情况为0， left 表示上下左右，  上0 下1 左2 右3
int W_Female_Detect::get_percent_edge_2(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, double percent, int type, int left)
{
    std::vector<double>h_his_vec;
    std::vector<double>v_his_vec;
    int th_value = 80;
    // v 通道判断亮度，h 通道判断边界
    if (type == 1) {
        for (int i = 0; i < hsv2[1].cols; i++) {
            double count = 0;
            for (int j = 0; j < hsv2[1].rows; j++) {
                int value = hsv2[1].at<uchar>(j, i);
                if (value > th_value) {
                    count++;
                }
            }
            double avage = count / (hsv2[1].rows * 1.f);
            h_his_vec.push_back(avage);
        }

        for (int i = 0; i < hsv2[1].cols; i++) {
            int sum = 0;
            for (int j = 0; j < hsv2[1].rows; j++) {
                int value = hsv2[1].at<uchar>(j, i);
                sum = sum + value;
            }
            double avage = sum / hsv2[1].rows;
            v_his_vec.push_back(avage);
        }

    }
    if (type == 0) {
        for (int i = 0; i < hsv2[1].rows; i++) {
            double count = 0;
            for (int j = 0; j < hsv2[1].cols; j++) {
                int value = hsv2[1].at<uchar>(i, j);
                if (value > th_value) {
                    count++;
                }
            }
            double avage = count / (hsv2[1].cols * 1.f);
            h_his_vec.push_back(avage);
        }
        for (int i = 0; i < hsv2[1].rows; i++) {

            int sum = 0;
            for (int j = 0; j < hsv2[1].cols; j++) {
                int value = hsv2[1].at<uchar>(i, j);
                sum = sum + value;
            }
            double avage = sum / hsv2[1].cols;
            v_his_vec.push_back(avage);
        }

    }
    int ret_value = 0;
    //左
    if (left == 2) {

        for (int i = 5; i < h_his_vec.size(); i++) {
            if (h_his_vec[i] > 0.3 && v_his_vec[i] > 80) {
                ret_value = i;
                break;
            }
        }
        //找错了，提高阈值再找下
        if (ret_value > hsv2[2].cols / 2) {
            for (int i = 1; i < h_his_vec.size(); i++) {
                if (h_his_vec[i] > 0.3 && v_his_vec[i] > 80) {
                    ret_value = i;
                    break;
                }
            }
        }
    }
    //右
    if (left == 3) {

        for (int i = h_his_vec.size() - 8; i > 0; i--) {
            if (h_his_vec[i] > 0.3 && v_his_vec[i] > 80) {
                ret_value = i;
                break;
            }
        }
        //找错了，提高阈值再找下
        if (ret_value < hsv2[2].cols / 2) {
            for (int i = h_his_vec.size() - 8; i > 0; i--) {
                if (h_his_vec[i] > 0.3 && v_his_vec[i] > 80) {
                    ret_value = i;
                    break;
                }
            }
        }

    }
    return ret_value;
}

//乳白开口 上下按 h通道处理
int W_Female_Detect::get_s_edge_2(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, int type, int th_value) {
    
    int ret_value = 0;
    std::vector<double>h_his_vec;
    std::vector<double>v_his_vec;
    for (int i = 0; i < hsv2[0].rows; i++) {
        double count = 0;
        for (int j = 0; j < hsv2[0].cols; j++) {
            int value = hsv2[0].at<uchar>(i, j);
            if (value > th_value) {
                count++;
            }
        }
        double avage = count / (hsv2[0].cols * 1.f);
        h_his_vec.push_back(avage);
    }
    //从中间向上
    for (int i = h_his_vec.size() /2; i > 0; i--) {
        if (h_his_vec[i] > 0.3) {
            ret_value = i;
            break;
        }
    }
    return ret_value;
}


void W_Female_Detect::img_process_7(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept {
    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 25;
    cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(grama_img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(th_img_1, th_img_1, kernel);
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;
    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 100 || rect.height > 50)continue;
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }
    //获取足够的待检测块
    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        //分每行 每列
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        //求角度
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);


        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 100 || rect.height > 55 || rect.width <= 70)continue;

        double area = cv::contourArea(filter_contours[i]);
        if (area < 1500 || area > 4000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 100)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        //cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }
    std::vector<W_Female_Detect::lc_info> lc_info_vec;
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 88, 32),
        cv::Vec4i(212, 0, 87, 32),
        cv::Vec4i(564, 0, 88, 33),
        cv::Vec4i(776, 0, 88, 32),
        cv::Vec4i(1128, 0, 89, 33),
        cv::Vec4i(1341, 0, 88, 33),
    };
    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            //默认每行未找全
            bool estimate_flag = false;
            if (!estimate_flag) {
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                double distance = 0;
                static std::vector<int> s_numbers = { 150, 360, 715, 930, 1280, 1490 };
                if (i % 2 == 0) {
                    // 偶数行
                    distance = (s_rect.x - detect_left_x_);
                }
                if (i % 2 == 1) {
                    // 奇数行
                    distance = (s_rect.x - 130 - detect_left_x_);
                }
                auto s_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                s_col_idx = std::distance(s_numbers.begin(), s_it);
                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    if (i % 2 == 0) distance = (cur_rect.x - detect_left_x_);
                    if (i % 2 == 1) distance = (cur_rect.x - 130 - detect_left_x_);;

                    auto c_it = std::min_element(s_numbers.begin(), s_numbers.end(), [=](int lhs, int rhs) { return std::abs(lhs - distance) < std::abs(rhs - distance); });
                    c_col_idx = std::distance(s_numbers.begin(), c_it);
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }
    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    LOGI("W_Female_Detect detect  lc_info start");
#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i = i + 2) {
        lc_info singal_female = cal_7(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, algo_result, process_rect_vec[i], process_rect_vec[i + 1], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        lc_info_vec.push_back(singal_female);
        w_lock.unlock();


    }
    LOGI("W_Female_Detect detect  lc_info end");
    data_cvt(lc_info_vec, algo_result);
    //cv::Mat dis = src_2.clone();
    //connector::draw_results(dis, algo_result->result_info);

    //if (algo_result->judge_result ==1) {
    //    std::string file_name = "F:\\download\\hf\\ret\\1\\" + g_path + ".jpg";
    //    cv::imwrite(file_name,dis);
    //}
    //if (algo_result->judge_result==0) {
    //    std::string file_name = "F:\\download\\hf\\ret\\0\\" + g_path + ".jpg";
    //    cv::imwrite(file_name, dis);
    //}
}

W_Female_Detect::lc_info W_Female_Detect::cal_7(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect cur_rect, cv::Rect next_rect, int index, cv::Mat inv_m) {

    connector::Timer t1;
    // 第几行第几个
    int col_idx = index % 6;
    int row_idx = index / 6;
    //相对位置关系
    std::vector<cv::Vec4i> pos = {
            cv::Vec4i(-89,-15,32,70), //左侧弹片
            cv::Vec4i(362,-16,34,72), //右侧弹片
            cv::Vec4i(-10,-96,320,30), //上侧找线边框
            cv::Vec4i(159,-82,40,20), //上左金属弹片
            cv::Vec4i(296,-82,40,20) //上右金属弹片

    };
    //奇数行
    if (row_idx % 2 == 1) {
        pos[3] = cv::Vec4i(-29, -80, 40, 20);
        pos[4] = cv::Vec4i(124, -80, 40, 20);
    }
    if ((col_idx %3 ==1 || col_idx % 3 == 2) && (row_idx % 2 == 0))
    {
        pos[3] = cv::Vec4i(140,-82,40,20);
        pos[4] = cv::Vec4i(296,-82,40,20) ;
    }
    //偶数排的最后一个也不一样
    if ((col_idx % 6 == 4) && (row_idx % 2 == 1))
    {
        pos[4] = cv::Vec4i(108, -82, 40, 20);
    }
    lc_info singal_lc;
    singal_lc.template_rect.resize(6);
    singal_lc.template_line_vec.resize(6);

    singal_lc.index = index / 2;

    //t1.out();
    //找上边线
    find_line(src_1(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), src_2(cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3])), cv::Rect(cur_rect.x + pos[2][0], cur_rect.y + pos[2][1], pos[2][2], pos[2][3]), singal_lc);
    //t1.out("find_line");
    if (singal_lc.top_line.startPointX == 0 || singal_lc.top_line.endPointX == 0) {
        //未找到上边线Ng

        singal_lc.a1 = a_ + error_a_ + 0.01;
        return singal_lc;
    }
    //计算上边线到当前矩形的距离
    double distance_t_c = abs(singal_lc.top_line.startPointY - cur_rect.y);
    if (abs(distance_t_c - 85) > 8) {
        cur_rect.y = singal_lc.top_line.startPointY + 85;
        next_rect.y = singal_lc.top_line.startPointY + 85;
    }
    //需要hsv 通道

   
    //找LC左金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), src_2(cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3])), cv::Rect(cur_rect.x + pos[0][0], cur_rect.y + pos[0][1], pos[0][2], pos[0][3]), singal_lc, 1, hsv_v1, hsv_v2);
    //找LC右金属弹片
    find_box(src_1(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), src_2(cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3])), cv::Rect(cur_rect.x + pos[1][0], cur_rect.y + pos[1][1], pos[1][2], pos[1][3]), singal_lc, 0, hsv_v1, hsv_v2);
    //t1.out("find_box");
    //找左右定位框
    cur_rect.height = cur_rect.height - 5;
    next_rect.height = next_rect.height - 5;
    find_location_box(src_1(cur_rect), src_2(cur_rect), cur_rect, singal_lc, 1, hsv_v1, hsv_v2);
    find_location_box(src_1(next_rect), src_2(next_rect), next_rect, singal_lc, 0, hsv_v1, hsv_v2);
    //t1.out("find_location_box");

    //寻找最上面的弹片，左右
    find_top_box(src_1(cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3])), src_2(cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3])), hsv_v1, hsv_v2,cv::Rect(cur_rect.x + pos[3][0], cur_rect.y + pos[3][1], pos[3][2], pos[3][3]), singal_lc, 1);
    find_top_box(src_1(cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3])), src_2(cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3])), hsv_v1, hsv_v2,cv::Rect(cur_rect.x + pos[4][0], cur_rect.y + pos[4][1], pos[4][2], pos[4][3]), singal_lc, 0);
    //t1.out("find_top_box");
    return singal_lc;
}

void W_Female_Detect::img_process_8(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result)noexcept {

    cv::Mat img_1, img_2, th_img_1;
    img_1 = src_1.clone();
    img_2 = src_2.clone();

    // 阈值处理
    int thre_value = 15;
    //cv::Mat grama_img_1 = connector::gamma_trans(img_1, 0.8);
    cv::threshold(img_1, th_img_1, thre_value, 255, cv::THRESH_BINARY_INV);
    // 膨胀腐蚀
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::dilate(th_img_1, th_img_1, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3));
    cv::erode(th_img_1, th_img_1, kernel);
    // 初次轮廓
    std::vector<std::vector<cv::Point>> filter_contours = connector::get_contours(th_img_1);
    // 取初值mask
    int angle_count = 0;
    double angle = 0;
    std::vector<double> angle_vec;
    std::vector<cv::Rect> rect_vec;
    // 观察图像
    cv::Mat gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {
        // 获取角度
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (rect.width > 130 || rect.height > 90)continue;
        if (width > 175)continue;
        double area_rate = area / (rect.width * rect.height);
        if (area_rate < 0.8) continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        w_lock.lock();
        rect_vec.push_back(rect);
        w_lock.unlock();
    }

    if (rect_vec.size() > 12) {
        std::sort(rect_vec.begin(), rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
            cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
            cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
            if (abs(lhs.y - rhs.y) <= 150) {
                if (lhs.x < rhs.x) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                if (lhs.y < rhs.y) {
                    return true;
                }
                else {
                    return false;
                }
            }
            });
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rect_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rect_vec[i].tl();
            cv::Point2d next_pt = rect_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rect_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
            else {
                swap_vec.push_back(rect_vec[i]);
                if (i == rect_vec.size() - 2) {
                    swap_vec.push_back(rect_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }
        for (int i = 0; i < rank.size(); i++) {
            if (rank[i].size() >= 2) {
                cv::Point2f p1 = rank[i][0].tl();
                cv::Point2f p2 = rank[i][rank[i].size() - 1].tl();
                double k = (p1.y - p2.y) / (p1.x - p2.x);
                angle_vec.push_back(atanl(k) * 180.0 / CV_PI);
                angle = angle + atanl(k) * 180.0 / CV_PI;
                angle_count++;
            }
        }
        angle = angle / angle_count;
    }
    else {
        algo_result->judge_result = 0;
        return;
    }

    // 旋转矩阵
    cv::Mat ret, inv_m;
    cv::Mat m = cv::getRotationMatrix2D(cv::Point(th_img_1.cols / 2, th_img_1.rows / 2), angle, 1);
    cv::invertAffineTransform(m, inv_m);
    // 阈值图像旋转
    cv::Mat rotate_img_1, rotate_img_2;
    cv::warpAffine(th_img_1, ret, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_1, rotate_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(src_2, rotate_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_1, input_img_1, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);
    cv::warpAffine(input_img_2, input_img_2, m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v1.size(); i++)
        cv::warpAffine(hsv_v1[i], hsv_v1[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    for (int i = 0; i < hsv_v2.size(); i++)
        cv::warpAffine(hsv_v2[i], hsv_v2[i], m, th_img_1.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    g_dis = input_img_1.clone();
    g_dis_2 = input_img_2.clone();
    g_dis_3 = cv::Mat::zeros(rotate_img_1.size(), src_1.type());

    if (g_dis.channels() < 3) {
        cv::cvtColor(g_dis, g_dis, cv::COLOR_GRAY2BGR);
    }
    if (g_dis_2.channels() < 3) {
        cv::cvtColor(g_dis_2, g_dis_2, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Rect> rec_vec;
    filter_contours.clear();
    filter_contours = connector::get_contours(ret);

    gray_mask = cv::Mat::zeros(th_img_1.size(), CV_8UC1);
    // 获取小单元格的准确边缘
    thre_value = 70;
    std::vector<double> area_rate_vec;
#pragma omp parallel for
    for (int i = 0; i < filter_contours.size(); ++i) {

        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        if (rect.width > 140 || rect.height > 90 || rect.width <= 70)continue;

        double area = cv::contourArea(filter_contours[i]);
        if (area < 500 || area > 30000) continue;
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        if (width > 175)continue;
        double area_rate = area / (rect.width * rect.height);
        // area_rate_vec.push_back(area_rate);
        if (area_rate < 0.8)continue;
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        cv::Mat cur_img = rotate_img_1(rect);
        cv::Mat cur_th_img;
        cv::threshold(cur_img, cur_th_img, thre_value, 255, cv::THRESH_BINARY_INV);
        cv::Rect second_rect = reget_rect(cur_th_img, rect);
        // cv::rectangle(g_dis, second_rect, cv::Scalar::all(255));

        w_lock.lock();
        rec_vec.push_back(second_rect);
        w_lock.unlock();
        /*cv::drawContours(gray_mask, draw_conts, 0, 255, -1);*/
    }

    std::sort(rec_vec.begin(), rec_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    if (rec_vec.size() < 10) {
        algo_result->judge_result = 0;
        return;
    }

    std::vector<W_Female_Detect::w_female> w_female_vec;
    cv::Vec4i af_vec(131, 0, 103, 47);
    cv::Vec4i pre_vec(-144, 0, 103, 47);

    // 同一行矩形的相对关系
    std::vector<cv::Vec4i> estimate_rect_1 = {
        cv::Vec4i(0, 0, 105, 51),
        cv::Vec4i(144, 8, 89, 42),
        cv::Vec4i(301, 9, 90, 40),
        cv::Vec4i(433, -1, 103, 50),
        cv::Vec4i(579, 8, 88, 41),
        cv::Vec4i(732, 7, 92, 42),
        cv::Vec4i(866, -2, 104, 52),
        cv::Vec4i(1011, 6, 92, 42),
        cv::Vec4i(1166, 8, 92, 40),
        cv::Vec4i(1299, -2, 103, 52),
        cv::Vec4i(1443, 6, 91, 43),
        cv::Vec4i(1599, 6, 91, 43),
        cv::Vec4i(1732, -1, 105, 51),
    };

    std::vector<cv::Rect> process_rect_vec;
    if (rec_vec.size() > 12) {
        std::vector<std::vector<cv::Rect>> rank;
        std::vector<cv::Rect> swap_vec;
        for (int i = 0; i < rec_vec.size() - 1; i++) {
            cv::Point2d cur_pt = rec_vec[i].tl();
            ;
            cv::Point2d next_pt = rec_vec[i + 1].tl();
            if (std::abs(cur_pt.y - next_pt.y) > 150) {
                swap_vec.push_back(rec_vec[i]);
                std::vector<cv::Rect> tmp_vec = swap_vec;
                rank.push_back(tmp_vec);
                swap_vec.clear();
            }
            else {
                swap_vec.push_back(rec_vec[i]);
                if (i == rec_vec.size() - 2) {
                    swap_vec.push_back(rec_vec[i + 1]);
                    std::vector<cv::Rect> tmp_vec = swap_vec;
                    rank.push_back(tmp_vec);
                    swap_vec.clear();
                }
            }
        }

        for (int i = 0; i < rank.size(); i++) {
            // 每一行进行处理,最后一行特殊处理
            bool estimate_flag = false;
           
            if (estimate_flag) {
                // 当前行的个数是12个，要进行间隔插一个
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    if (j % 2 == 0) {
                        cv::Rect tmp_rect(cur_rect.x + pre_vec[0], cur_rect.y + pre_vec[1], pre_vec[2], pre_vec[3]);
                        process_rect_vec.push_back(tmp_rect);
                        process_rect_vec.push_back(cur_rect);
                        cv::rectangle(gray_mask, tmp_rect, cv::Scalar::all(255));
                        cv::rectangle(g_dis, tmp_rect, cv::Scalar(0, 0, 255));
                        cv::rectangle(g_dis_2, tmp_rect, cv::Scalar(0, 0, 255));

                    }
                    else if (j == rank[i].size() - 1) {
                        cv::Rect tmp_rect(cur_rect.x + af_vec[0], cur_rect.y + af_vec[1], af_vec[2], af_vec[3]);
                        process_rect_vec.push_back(cur_rect);
                        process_rect_vec.push_back(tmp_rect);
                        cv::rectangle(gray_mask, tmp_rect, cv::Scalar::all(255));
                        cv::rectangle(g_dis, tmp_rect, cv::Scalar(0, 0, 255));
                        cv::rectangle(g_dis_2, tmp_rect, cv::Scalar(0, 0, 255));
                    }
                    else {
                        process_rect_vec.push_back(cur_rect);
                    }
                }
            }
            else {
                // 当前行未找全的，特殊处理
                // 查询第一个黑孔是这一行的第几个
                int s_col_idx = 0;
                int c_col_idx = 0;
                cv::Rect s_rect = rank[i][0];
                if (i % 2 == 0) {
                    // 偶数行
                    s_col_idx = (s_rect.x - detect_left_x_) / 144;
                }
                if (i % 2 == 1) {
                    // 奇数行
                    s_col_idx = (s_rect.x - 122 - detect_left_x_) / 144;
                }
                std::vector<std::vector<cv::Rect>> complete_rect_vec;
                for (int j = 0; j < rank[i].size(); j++) {
                    cv::Rect cur_rect = rank[i][j];
                    // 当前黑孔的序号
                    c_col_idx = ((cur_rect.x - s_rect.x) / 144.0 + 0.5) + s_col_idx;
                    // 根据每个找到的小黑孔，生成一行对应的矩形估计
                    std::vector<cv::Rect> tmp_rect_vec = get_complete_rect(estimate_rect_1, cur_rect, c_col_idx);
                    complete_rect_vec.push_back(tmp_rect_vec);
                }
                // 从估计的矩形里面求均值，进行估计
                int count = complete_rect_vec.size();
                for (int m = 0; m < estimate_rect_1.size(); m++) {
                    double sum_x = 0;
                    double sum_y = 0;
                    double sum_w = 0;
                    double sum_h = 0;
                    for (int n = 0; n < complete_rect_vec.size(); n++) {
                        sum_x = sum_x + complete_rect_vec[n][m].x;
                        sum_y = sum_y + complete_rect_vec[n][m].y;
                        sum_w = sum_w + complete_rect_vec[n][m].width;
                        sum_h = sum_h + complete_rect_vec[n][m].height;
                    }
                    sum_x = sum_x / count;
                    sum_y = sum_y / count;
                    sum_w = sum_w / count;
                    sum_h = sum_h / count;
                    cv::Rect tmp(sum_x, sum_y, sum_w, sum_h);
                    process_rect_vec.push_back(tmp);
                    cv::rectangle(gray_mask, tmp, cv::Scalar::all(255));
                    cv::rectangle(g_dis, tmp, cv::Scalar(0, 0, 255));
                    cv::rectangle(g_dis_2, tmp, cv::Scalar(0, 0, 255));
                }
            }
        }
    }

    // 重新排序
    std::sort(process_rect_vec.begin(), process_rect_vec.end(), [&](const cv::Rect& lhs, const cv::Rect& rhs) {
        // y轴相差500以内是同一行
        cv::Point p1(lhs.x + lhs.width / 2, lhs.y + lhs.height / 2);
        cv::Point p2(rhs.x + rhs.width / 2, rhs.y + rhs.height / 2);
        if (abs(lhs.y - rhs.y) <= 150) {
            if (lhs.x < rhs.x) {
                return true;
            }
            else {
                return false;
            }
        }
        else {
            // 不在同一行
            if (lhs.y < rhs.y) {
                return true;
            }
            else {
                return false;
            }
        }
        });

    std::vector<cv::Mat> rbg_v1, rbg_v2;
    cv::split(input_img_1, rbg_v1);
    cv::split(input_img_2, rbg_v2);

#pragma omp parallel for
    for (int i = 0; i < process_rect_vec.size(); i++) {
        w_female singal_female = cal_3(rotate_img_1, rotate_img_2, hsv_v1, hsv_v2, rbg_v1, rbg_v2, algo_result, process_rect_vec[i], i, inv_m);
        w_lock.lock();
        singal_female.h = m;
        singal_female.inv_h = inv_m;
        w_female_vec.push_back(singal_female);
        w_lock.unlock();
    }
    data_cvt_3(w_female_vec, algo_result);

}