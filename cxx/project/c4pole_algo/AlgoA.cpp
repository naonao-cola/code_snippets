#include <windows.h>
#include "../framework/InferenceEngine.h"
#include "../utils/logger.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ximgproc.hpp"
#include "./sub_3rdparty/EDLib.h"
#include "./sub_3rdparty/threshold.h"
#include "AlgoA.h"

REGISTER_ALGO(AlgoA)

AlgoA::AlgoA()
{
}

AlgoA::~AlgoA()
{
}

std::tuple<std::string, json> AlgoA::get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map)
{
    std::string task_type_id = task->image_info["type_id"];
    json task_json = param_map[task_type_id];
    return std::make_tuple(task_type_id, task_json);
}

AlgoResultPtr AlgoA::RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results)
{
    LOGI("AlgoA start run!");
    //申请结果
    AlgoResultPtr algo_result = std::make_shared<stAlgoResult>();
    algo_result->status = RunStatus::OK;
    
    //获取参数
    std::tuple<std::string, json> details_info = get_task_info(task, pre_results, m_param_map);
    std::string task_type_id = std::get<0>(details_info);
    json task_param_json = std::get<1>(details_info);
    std::string error_message;
    try {
        std::stringstream ss;
        ss << task_param_json["param"]["inside_dia_min"];
        LOGI("AlgoA param inside_dia_min {}", ss.str());
        inside_dia_min_ = task_param_json["param"]["inside_dia_min"];

        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["inside_dia_max"];
        LOGI("AlgoA param inside_dia_max {}", ss.str());
        inside_dia_max_ = task_param_json["param"]["inside_dia_max"]; // 脏污的面积

        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["outside_dia_min"];
        LOGI("AlgoA param outside_dis_min {}", ss.str());
        outside_dis_min_ = task_param_json["param"]["outside_dia_min"];

        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["outside_dia_max"];
        LOGI("AlgoA param outside_ida_max {}", ss.str());
        outside_dis_max_ = task_param_json["param"]["outside_dia_max"];

        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["circle_cnt"];
        LOGI("AlgoA param circle_cnt {}", ss.str());
        circle_cnt_ = task_param_json["param"]["circle_cnt"];

        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["distance_max"];
        LOGI("AlgoA param distance_max {}", ss.str());
        distance_max_ = task_param_json["param"]["distance_max"];


        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["distance_min"];
        LOGI("AlgoA param diatance_min {}", ss.str());
        diatance_min_ = task_param_json["param"]["distance_min"];

        ss.clear();
        ss.str("");
        ss << task_param_json["param"]["judge_dis"];
        LOGI("AlgoA param diatance_min {}", ss.str());
        judge_dis_ = task_param_json["param"]["judge_dis"];


    } catch (const std::exception& e) {
        error_message = e.what();
        LOGI("AlgoA param error{}", e.what());
        algo_result->status = RunStatus::WRONG_PARAM;
        return algo_result;
    }

    cv::Mat src = task->image.clone();
    cv::Rect circle_rect{0,0,0,0};
    bool ret_pro= img_process(src, circle_rect);
    circle_rect.x = circle_rect.x * 2;
    circle_rect.y = circle_rect.y * 2;
    circle_rect.width = circle_rect.width * 2;
    circle_rect.height = circle_rect.height * 2;
    // algo_result->status = RunStatus::ABNORMAL_IMAGE;
    if (ret_pro) {
        algo_result->result_info = {
            { "label", "ok" },
            { "shapeType", "rectangle" },
            { "points", { { circle_rect.x, circle_rect.y }, { circle_rect.x+ circle_rect.width, circle_rect.y+ circle_rect.height } } },
            { "result", { { "confidence", 1 }, { "area", 0 } } },
        };
    }
    else {
        algo_result->result_info = {
          { "label", "ng" },
          { "shapeType", "rectangle" },
          { "points", { { circle_rect.x, circle_rect.y }, { circle_rect.x + circle_rect.width, circle_rect.y + circle_rect.height } } },
          { "result", { { "confidence", 0 }, { "area", 0 } } },
        };
    
    }
   

    LOGI("AlgoA run finished!");
    return algo_result;
}


cv::Mat gray_stairs(const cv::Mat& img, double sin = 0.0, double hin = 255.0, double mt = 1.0, double sout = 0.0, double hout = 255.0) {
    double Sin = (std::min)((std::max)(sin, 0.0), hin - 2);
    double Hin = (std::min)(hin, 255.0);
    double Mt = (std::min)((std::max)(mt, 0.01), 9.99);
    double Sout = (std::min)((std::max)(sout, 0.0), hout - 2);
    double Hout = (std::min)(hout, 255.0);
    double difin = Hin - Sin;
    double difout = Hout - Sout;
    uchar lutData[256];
    for (int i = 0; i < 256; i++) {
        double v1 = (std::min)((std::max)(255 * (i - Sin) / difin, 0.0), 255.0);
        double v2 = 255 * std::pow(v1 / 255.0, 1.0 / Mt);
        lutData[i] = (int)(std::min)((std::max)(Sout + difout * v2 / 255, 0.0), 255.0);
    }
    cv::Mat lut(1, 256, CV_8UC1, lutData);
    cv::Mat dst;
    cv::LUT(img, lut, dst);
    return dst;
}

bool AlgoA::get_circle_all_pix(cv::Mat img,cv::Point center,int radius) {
    cv::Mat zero_img(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    
    if (radius >= inside_dia_min_ && radius <= inside_dia_max_) {
        radius = radius + 30;
    }

    cv::Size axes(radius, radius);
    cv::ellipse(zero_img, center, axes, 0, 0, 360, cv::Scalar::all(255), 1, cv::LINE_AA);

    std::vector<std::vector<cv::Point> > filter_contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(zero_img, filter_contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //找到圆上轮廓点,统计圆环的亮度
    double inside = 0;
    int inside_count = 0;
    if (filter_contours.size() > 0) {
        std::vector<cv::Point> pts = filter_contours[0];
        for (int i = 0; i < pts.size();i++) {
            uchar pix_value = *img.ptr<uchar>(pts[i].y, pts[i].x);
            inside = pix_value + inside;
            inside_count++;
        }
    }
    inside = inside / inside_count;
    int a = 1;
    return true;
}

bool  AlgoA::img_process(cv::Mat src, cv::Rect& circle_rect)
{
    cv::Mat task_img,dis_img;
    if (src.channels() > 1) {
        cv::cvtColor(src,task_img, cv::COLOR_BGR2GRAY);
    }
    else {
        task_img = src.clone();
    }
    cv::resize(task_img, task_img, cv::Size(task_img.cols / 2, task_img.rows / 2));

    //显示图像
    dis_img = task_img.clone();
    cv::cvtColor(dis_img, dis_img, cv::COLOR_GRAY2BGR);

    //预处理
    cv::Mat gauss_gray;
    //cv::GaussianBlur(task_img,gauss_gray, cv::Size(3, 3), 0);
    //增强对比度 亮度

    gauss_gray = task_img.clone();
    gauss_gray = gray_stairs(gauss_gray,15,170,1.13);
    nao::algorithm::threshold::sauvola(gauss_gray, 0.06, 9);
    std::vector<double> dis_vec;

    //找圆
    EDPF edpf = EDPF(gauss_gray);
    EDCircles edcircle = EDCircles(edpf);
    std::vector<mCircle> found_circles = edcircle.getCircles();

    std::vector<mEllipse> found_mEllipse = edcircle.getEllipses();
    if (found_circles.size()<=0 && found_mEllipse.size()<=0) {
        dis_vec.emplace_back(1000);
    }
    //获取圆与半径
    std::vector<std::pair<cv::Point, int>> find_cricle;
    for (int j = 0; j < found_circles.size(); j++)
    {
        cv::Point center((int)found_circles[j].center.x, (int)found_circles[j].center.y);
        cv::Size axes((int)found_circles[j].r, (int)found_circles[j].r);
        double angle(0.0);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        if (((int)found_circles[j].r >= inside_dia_min_ && (int)found_circles[j].r <= inside_dia_max_) || 
            ((int)found_circles[j].r >= outside_dis_min_ && (int)found_circles[j].r <= outside_dis_max_)) {
            find_cricle.emplace_back(std::make_pair(center, found_circles[j].r));
            ellipse(dis_img, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
        }
    }
    //椭圆
    for (int m = 0; m < found_mEllipse.size();m++) {
        cv::Point center((int)found_mEllipse[m].center.x, (int)found_mEllipse[m].center.y);
        cv::Size axes(found_mEllipse[m].axes);
        //长轴短轴相差小于5，认作是圆
        if (std::abs(axes.width - axes.height) <=8) {
            double r = (axes.width + axes.height) / 2;
            double angle(0.0);
            cv::Scalar color = cv::Scalar(0, 0, 255);
            if (((int)r >= inside_dia_min_ && r <= inside_dia_max_) ||
                ((int)r >= outside_dis_min_ && r <= outside_dis_max_)) {
                find_cricle.emplace_back(std::make_pair(center, r));
                ellipse(dis_img, center, axes, angle, 0, 360, color, 2, cv::LINE_AA);
            }
        }
    }


    //分类 得到 00000111111222222的类型
    std::vector<int> index_used(find_cricle.size(), -1);
    std::set<int> sort_used;
    for (int m = 0; m < find_cricle.size(); m++) {
        if (index_used[m] != -1) continue;
        for (int n = m + 1; n < find_cricle.size(); n++) {
            if (index_used[n] != -1) continue;
            double dis = std::pow((find_cricle[m].first.x - find_cricle[n].first.x), 2) + std::pow((find_cricle[m].first.y - find_cricle[n].first.y), 2);
            dis = std::sqrt(dis);
            if (dis <= distance_max_) {
                if (index_used[m] == -1) {
                    index_used[n] = m;
                    index_used[m] = m;
                }
                else {
                    index_used[n] = index_used[m];
                }
            }
        }
        if (index_used[m] == -1) {
            index_used[m] = m;
        }
        sort_used.insert(index_used[m]);
    }


    int category = 0;
    for (std::set<int>::iterator it = sort_used.begin(); it != sort_used.end(); it++) {
        int current_index = *it;
        std::vector<std::pair<cv::Point, int>> sub_circle;
        //获取每一组同心圆
        for (int q = 0; q < index_used.size(); q++) {
            if (index_used[q] == current_index) {
                sub_circle.emplace_back(std::make_pair(find_cricle[q].first, find_cricle[q].second));
            }
        }


        double dis = 0;
        std::vector<cv::Point> first_pt;
        std::vector<cv::Point> second_pt;

        //get_circle_all_pix(task_img, center, found_circles[j].r);

        if (sub_circle.size() < 2) {
            dis = 10000;
            LOGI("not find circle,center_x {},center_y {}", sub_circle[0].first.x, sub_circle[0].first.y);
            goto savejpg;
        
        }


        cv::Point pt = sub_circle[0].first;
        first_pt.emplace_back(pt);
        double first_r = sub_circle[0].second;
        for (int p = 1; p < sub_circle.size(); p++) {
            cv::Point cur = sub_circle[p].first;
            double cur_r = sub_circle[p].second;
            double dis = std::abs(first_r - cur_r);
            if (dis < diatance_min_) {
                first_pt.emplace_back(cur);
            }
            else {
                second_pt.emplace_back(cur);
            }
        }

        cv::Point f_avg = (0, 0);
        cv::Point s_avg = (0, 0);
        for (int i = 0; i < first_pt.size(); i++) {
            f_avg.x = f_avg.x + first_pt[i].x;
            f_avg.y = f_avg.y + first_pt[i].y;

        }
        if (first_pt.size() > 0) {
            f_avg.x = f_avg.x / first_pt.size();
            f_avg.y = f_avg.y / first_pt.size();
        }
        
        for (int j = 0; j < second_pt.size(); j++) {
            s_avg.x = s_avg.x + second_pt[j].x;
            s_avg.y = s_avg.y + second_pt[j].y;

        }

        if (second_pt.size() > 0) {
            s_avg.x = s_avg.x / second_pt.size();
            s_avg.y = s_avg.y / second_pt.size();
        }
       

        if (f_avg.x != 0 && s_avg.x != 0) {
            dis = std::pow((f_avg.x - s_avg.x), 2) + std::pow((f_avg.y - s_avg.y), 2);
            dis = std::sqrt(dis);
            //缩放图的圆矩形
            circle_rect = cv::Rect(f_avg.x-70, f_avg.y-70,140,140);

        }
        else {
            dis = 1000;
        }
       

    savejpg:
        std::cout << "distance :" << dis << std::endl;
        std::string text = "distance :" + std::to_string(dis);
        cv::Point origin;
        origin.x = 100;
        origin.y = category * 50 + 50;
        cv::putText(dis_img, text, origin, cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2, 8, 0);
        category++;
        LOGI("distance: {}",dis);
        dis_vec.emplace_back(dis);
    }

    int ret_count = 0;

    for (int q = 0; q < dis_vec.size();q++) {
        if (dis_vec[q]<=judge_dis_) {
            ret_count++;
            continue;
        }
    }
     if (ret_count >= circle_cnt_) {
        return true;
    }
    else {
        return false;
    }
    return true;;
}
