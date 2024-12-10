#include "../include/test.h"
#include "../include/contour.h"
/**
 * @FilePath     : /code_snippets/cxx/common/test.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-12-10 17:05:15
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-10 17:05:15
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#include "../include/fs.h"
#include "../include/miscellaneous.h"
#include "../include/threshold.h"
#include "../include/transform.hpp"
#include <fstream>
#include "../include/hessen_light_bar.hpp"


static int img_count = 0;
cv::Mat dis_img;

void test_print_start()
{
    std::cout << "pragram start" << std::endl;
}

void circle_process(cv::Mat src, cv::Mat img, std::vector<std::vector<cv::Point2f>>& circle_process_float, std::vector<std::vector<cv::Point>> circle_process_int)
{
    cv::Mat gauss_img;
    cv::GaussianBlur(img, gauss_img, cv::Size(3, 3), 0);
    nao::algorithm::threshold::select(gauss_img, 9);
    cv::Mat bise;
    cv::bitwise_not(gauss_img, bise);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(gauss_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        cv::Rect rect = cv::boundingRect(contours[i]);
        // std::cout<<"area: "<<area <<"  width: "<<rect.width<<" high: " << rect.height<< std::endl;
        if (area < 3 * 1e2 || 1e7 < area)
            continue;
        if (rect.height < 40 || rect.height > 300)
            continue;
        if (rect.width < 40 || rect.width > 300)
            continue;
        if ((rect.width / rect.height) > 3 || (rect.height / rect.width) > 3)
            continue;
        // 绘制轮廓
        // cv::drawContours(src, contours, i, CV_RGB(255, 0, 0), 3, 8, hierarchy, 0);
        circle_process_int.emplace_back(contours[i]);
    }
    for (int i = 0; i < circle_process_int.size(); i++) {
        std::vector<cv::Point2f> tmp;
        for (int j = 0; j < circle_process_int[i].size(); j++) {
            tmp.emplace_back(circle_process_int[i][j]);
        }
        circle_process_float.emplace_back(tmp);
    }
}

void pca_circle(cv::Mat src, std::vector<std::vector<cv::Point2f>> circle_process_float)
{
    for (int i = 0; i < circle_process_float.size(); i++) {
        cv::Point2f pos;
        double angle = 0;
        angle = nao::algorithm::contours::getOrientation(circle_process_float[i], pos, src);
    }

    return;
}

// 得到图像与直线的交点
void get_cross_pt(cv::Mat src, cv::Point2f p0, cv::Point2f p1, std::vector<cv::Point2f>& cross_pt)
{
    // if (p0.x >= src.cols || p0.x <= 0 || p0.y <= 0 || p0.y >= src.cols ||
    //     p1.x >= src.cols || p1.x <= 0 || p1.y <= 0 || p1.y >= src.cols) {
    //     return;
    // }
    double k_t;
    k_t = p1.x - p0.x;

    if (k_t == 0.) {
        cross_pt.push_back(cv::Point2f(p0.x, 0));
        cross_pt.push_back(cv::Point2f(p0.x, src.rows - 1));
    } else {
        double k, b;
        k = ((double)p1.y - p0.y) / (p1.x - p0.x);
        b = p1.y - k * p1.x;
        if (!k) {
            cross_pt.push_back(cv::Point2f(0, b));
            cross_pt.push_back(cv::Point2f(src.cols - 1, b));
        } else {
            int x1, x2, y1, y2;
            x1 = -b / k; // y=0
            x2 = (src.rows - 1 - b) / k; // y=img->height-1
            y1 = b; // x=0
            y2 = k * (src.cols - 1) + b; // x=img->width-1
            if (y1 >= 0 && y1 < src.rows)
                cross_pt.push_back(cv::Point2f(0, y1));
            if (y2 >= 0 && y2 < src.rows)
                cross_pt.push_back(cv::Point2f(src.cols - 1, y2));
            if (x1 >= 0 && x1 < src.cols)
                cross_pt.push_back(cv::Point2f(x1, 0));
            if (x2 >= 0 && x2 < src.cols)
                cross_pt.push_back(cv::Point2f(x2, src.rows - 1));
        }
    }
}
// 求直线与轮廓的交点,并求出距离
void get_points(cv::Mat src, std::vector<cv::Point2f> circle_point, cv::Point2f p0, cv::Point2f p1, cv::Point2f& f_pt, cv::Point2f& s_pt, double& max_dis)
{

    std::vector<cv::Point2f> cross_pt;
    get_cross_pt(src, p0, p1, cross_pt);
    cv::LineIterator it(src, cross_pt[0], cross_pt[1]);
    std::vector<cv::Point2f> dst;

    for (int i = 0; i < it.count; i++, it++) {
        cv::Point2f pt(it.pos());
        if (std::abs(cv::pointPolygonTest(circle_point, pt, true)) <= 3) {
            // cv::circle(src, pt, 2, CV_RGB(255, 0, 0),1);
            dst.emplace_back(pt);
        }
    }

    if (dst.size() <= 0)
        return;
    for (int i = 0; i < dst.size(); i++) {
        for (int j = i + 1; j < dst.size(); j++) {
            double dis = std::powf((dst[i].x - dst[j].x), 2) + std::powf((dst[i].y - dst[j].y), 2);
            dis = std::sqrtf(dis);
            if (dis >= max_dis) {
                max_dis = dis;
            }
        }
    }

    // std::cout<<" 距离 "<< max_dis<<std::endl;

    // 将轮廓分类
    std::vector<cv::Point2f> first;
    std::vector<cv::Point2f> second;
    first.emplace_back(dst[0]);
    for (int i = 1; i < dst.size(); i++) {
        double dis = std::powf((dst[i].x - first[0].x), 2) + std::powf((dst[i].y - first[0].y), 2);
        dis = std::sqrtf(dis);
        if (dis <= 20) {
            first.emplace_back(dst[i]);
        } else {
            second.emplace_back(dst[i]);
        }
    }

    if (first.size() <= 0 || second.size() <= 0) {
        std::cout << " 交点出错" << std::endl;
        return;
    }

    double sum_x = 0, sum_y = 0;
    for (int i = 0; i < first.size(); i++) {
        sum_x = sum_x + first[i].x;
        sum_y = sum_y + first[i].y;
    }
    f_pt.x = sum_x / first.size();
    f_pt.y = sum_y / first.size();

    sum_x = 0;
    sum_y = 0;
    for (int i = 0; i < second.size(); i++) {
        sum_x = sum_x + second[i].x;
        sum_y = sum_y + second[i].y;
    }
    s_pt.x = sum_x / second.size();
    s_pt.y = sum_y / second.size();
}

// 求旋转直线与轮廓的交点
void get_length(cv::Mat src, std::vector<cv::Point2f> circle_point, cv::Point2f center, cv::Point2f p0, cv::Point2f p1, double rote_angle, double rote_step, cv::Point2f& t_f_pt, cv::Point2f& t_s_pt, double& min_dis)
{
    // 原始点绕中线点旋转之后
    min_dis = 10000;
    for (double i = 0 - rote_angle; i <= 0 + rote_angle; i += rote_step) {
        cv::Point2f rote_p0, rote_p1;
        rote_p0.x = (p0.x - center.x) * cos(i * CV_PI / 180) - (p0.y - center.y) * sin(i * CV_PI / 180) + center.x;
        rote_p0.y = (p0.x - center.x) * sin(i * CV_PI / 180) + (p0.y - center.y) * cos(i * CV_PI / 180) + center.y;
        rote_p1.x = (p1.x - center.x) * cos(i * CV_PI / 180) - (p1.y - center.y) * sin(i * CV_PI / 180) + center.x;
        rote_p1.y = (p1.x - center.x) * sin(i * CV_PI / 180) + (p1.y - center.y) * cos(i * CV_PI / 180) + center.y;

        // cv::line(src, rote_p0, rote_p1, CV_RGB(255, 0, 0));
        cv::Point2f f_pt, s_pt;
        double max_dis = 0;
        get_points(src, circle_point, rote_p0, rote_p1, f_pt, s_pt, max_dis);
        // cv::line(src, f_pt, s_pt, CV_RGB(255, 0, 0));
        if (max_dis <= 5)
            continue;
        if (max_dis < min_dis) {
            min_dis = max_dis;
            t_f_pt = f_pt;
            t_s_pt = s_pt;
        }
    }
}

// 求平移直线与轮廓的交点
void get_length_2(cv::Mat src, std::vector<cv::Point2f> circle_point, cv::Point2f center, cv::Point2f p0, cv::Point2f p1, double trans, double trans_step, cv::Point2f& t_f_pt, cv::Point2f& t_s_pt, double& min_dis)
{
    // 原始点绕中线点旋转之后
    min_dis = 0;
    for (double i = 0 - trans; i <= 0 + trans; i += trans_step) {
        cv::Point2f rote_p0, rote_p1;
        rote_p0.x = p0.x;
        rote_p0.y = p0.y - i;
        rote_p1.x = p1.x;
        rote_p1.y = p1.y - i;
        // cv::line(src, rote_p0, rote_p1, CV_RGB(255, 0, 0));
        cv::Point2f f_pt, s_pt;
        double max_dis = 0;
        get_points(src, circle_point, rote_p0, rote_p1, f_pt, s_pt, max_dis);
        // cv::line(src, f_pt, s_pt, CV_RGB(255, 0, 0));
        if (max_dis > min_dis) {
            min_dis = max_dis;
            t_f_pt = f_pt;
            t_s_pt = s_pt;
        }
    }
}

#include "../include/EDLib.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ximgproc.hpp"

// 求两直线交点
cv::Point2d get2lineIPoint(cv::Point2d lineOnePt1, cv::Point2d lineOnePt2, cv::Point2d lineTwoPt1, cv::Point2d lineTwoPt2)
{
    double x;
    double y;
    cv::Point2d result(-1, -1);
    double k = (lineOnePt1.y - lineOnePt2.y) / (lineOnePt1.x - lineOnePt2.x);
    double b = lineOnePt1.y - k * lineOnePt1.x;
    double k1 = (lineTwoPt1.y - lineTwoPt2.y) / (lineTwoPt1.x - lineTwoPt2.x);
    double b1 = lineTwoPt1.y - k1 * lineTwoPt1.x;

    x = (b1 - b) / (k - k1);
    y = k * x + b;
    result.x = x;
    result.y = y;

    return result;
}

// https://blog.csdn.net/u011754972/article/details/121533505


void y_process(cv::Mat src, cv::Mat& yellow_mask)
{
    cv::Scalar lower_green = cv::Scalar(26, 30, 46);
    cv::Scalar upper_green = cv::Scalar(34, 255, 255);
    //判断是否黄色区域
    std::vector<std::vector<cv::Point>> filter_contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(yellow_mask, filter_contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //cv::Mat tmp_mask = yellow_mask.clone();
    for (size_t i = 0; i < filter_contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        cv::Mat sub_img = src(rect);
        cv::Mat hsv;
        cv::cvtColor(sub_img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat y_mask;
        cv::inRange(hsv, lower_green, upper_green, y_mask);
        double y_mean = cv::mean(y_mask)[0];
        if (y_mean<10) {
            cv::drawContours(yellow_mask, draw_conts, 0,0, -1);
        }
    }
}

void thre_process(cv::Mat src, cv::Mat& gray_mask) {

    cv::Mat gray_img = src.clone();
    int thre_value = nao::algorithm::threshold::exec_threshold(gray_img, nao::algorithm::THRESHOLD_TYPE::YEN);
    thre_value = thre_value + 20;
    cv::threshold(gray_img, gray_img,thre_value,255,cv::THRESH_BINARY);

    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));;//X方向腐蚀
    cv::dilate(gray_img, gray_img, elementX);

    std::vector<std::vector<cv::Point>> filter_contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(gray_img, filter_contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    gray_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (size_t i = 0; i < filter_contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 130)continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        cv::Moments m = cv::moments(filter_contours[i]);
        int center_x = int(m.m10 / m.m00);
        int center_y = int(m.m01 / m.m00);
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        double rate = width / (height * 1.0);
        if (height <= 15 && height >= 4 && width >= 50 && width < 200 && rate >= 2) {

            cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        }
        else {

        }
    }

}

#include "../include/EDLib.h"
void thre_process_2(cv::Mat src, cv::Mat gray_img,cv::Mat& gray_mask) {

    cv::Mat  thre_img;
    int thre_value = nao::algorithm::threshold::exec_threshold(gray_img, nao::algorithm::THRESHOLD_TYPE::YEN);
    thre_value = thre_value + 20;
    cv::threshold(gray_img, thre_img, thre_value, 255, cv::THRESH_BINARY);
    cv::Mat elementX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 3));;//X方向腐蚀
    cv::dilate(thre_img, thre_img, elementX);

    std::vector<std::vector<cv::Point>> filter_contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(thre_img, filter_contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    gray_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (size_t i = 0; i < filter_contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(filter_contours[i]);
        double area = cv::contourArea(filter_contours[i]);
        if (area < 130)continue;
        cv::RotatedRect r_rect = cv::minAreaRect(filter_contours[i]);
        std::vector<std::vector<cv::Point>> draw_conts = { filter_contours[i] };
        int width = (std::max)(r_rect.size.width, r_rect.size.height);
        int height = (std::min)(r_rect.size.width, r_rect.size.height);
        double rate = width / (height * 1.0);

        if (height <= 80 && height >= 45 && width >= 450 && width < 550 && rate >= 4) {

            cv::drawContours(gray_mask, draw_conts, 0, 255, -1);
        }
        else {

        }
    }


    std::vector<cv::Point2f> center_pt_info;

    std::vector<std::vector<cv::Point>> line_pt;
    std::vector<cv::Vec4i> line_hierarchy;
    cv::findContours(gray_mask, line_pt, line_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (size_t j = 0; j < line_pt.size();j++) {
        cv::Rect rect = cv::boundingRect(line_pt[j]);

        cv::RotatedRect rot_rect = cv::minAreaRect(line_pt[j]);
        cv::Point2f box_pts[4];
        rot_rect.points(box_pts);
        std::vector<cv::Point> pt_vec;
        for (int i = 0; i != 4; ++i) pt_vec.emplace_back(cv::Point2i(box_pts[i]));
        std::vector<std::vector<cv::Point>> tmpContours;
        tmpContours.insert(tmpContours.end(), pt_vec);
        //cv::drawContours(dis_img, tmpContours, 0, cv::Scalar(0, 255, 0), 1);



        cv::Rect af_rect;

        double center_x = rect.x + rect.width / 2;
        //左右个去除 1/14，保留上部分整体
        int width = rect.width;
        int step_width = width / 14;

        af_rect.x = rect.x + step_width;
        af_rect.width = rect.width - step_width * 2;
        af_rect.y = rect.y - 15;
        af_rect.height = rect.height;

        //cv::rectangle(dis_img, af_rect, cv::Scalar(0, 255, 0), 1);
        cv::Mat sub_img = gray_img(af_rect);

        int mean_value = cv::mean(sub_img)[0];
        int start_value = mean_value - 20 > 60 ? mean_value - 20 : 60;
        int end_value = mean_value +20 <80 ? 80: mean_value + 20;
        std::map<int, double> center_info;

        for (int cur_th = start_value; cur_th <= end_value; cur_th = cur_th + 20) {
            cv::Mat thre_img;
            cv::threshold(sub_img, thre_img, cur_th, 255, cv::THRESH_BINARY);

            //排除掉干扰
            std::vector<std::vector<cv::Point>> tmp_pt;
            std::vector<cv::Vec4i> tmp_hierarchy;
            cv::findContours(thre_img, tmp_pt, tmp_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            for (int m = 0; m < tmp_pt.size(); m++) {
                cv::Rect rect = cv::boundingRect(tmp_pt[m]);
                std::vector<std::vector<cv::Point>> draw_conts = { tmp_pt[m] };
                if (rect.width < thre_img.cols - 10) {
                    cv::drawContours(thre_img, draw_conts, 0, 0, -1);
                }

            }

            std::vector<cv::Point2f> dst_pt;
            nao::algorithm::hessen_light::StegerLine(thre_img, dst_pt);
            double sumY = 0;
            for (int m = 0; m < dst_pt.size(); m++) {
                sumY = sumY + dst_pt[m].y;
            }
            sumY = sumY / dst_pt.size();
            //此时的center是相对于裁剪之后的，需要还原
            sumY = sumY + af_rect.y;
            center_info.insert(std::pair<int, double>(cur_th, sumY));

            ED testED = ED(thre_img, SOBEL_OPERATOR, 36, 10, 1, 50, 1.0, true);
            EDLines testEDLines = EDLines(testED);
            cv::Mat lineImg0 = testEDLines.getLineImage();
            cv::Mat lineImage = cv::Mat(sub_img.rows,sub_img.cols, CV_8UC1, cv::Scalar(255));

            //求最高点，最低点
            int max_value =0;
            int min_value= 999;
            int max_index = 0;
            int min_index = 0;

            for (int i = 0; i < testEDLines.getLinesNo();i++) {
                double avg_y = (testEDLines.getLines()[i].start.y + testEDLines.getLines()[i].end.y)/2;
                double avg_x = (testEDLines.getLines()[i].start.x - testEDLines.getLines()[i].end.x);
                double abs1 = std::atan2(testEDLines.getLines()[i].start.y - testEDLines.getLines()[i].end.y, testEDLines.getLines()[i].start.x - testEDLines.getLines()[i].end.x) * 180 / CV_PI;
                if (std::abs(abs1) > 20) continue;
                if (avg_y < min_value && std::abs(avg_x)>sub_img.cols/2)
                {
                    min_value = avg_y;
                    min_index = i;
                }
                if (avg_y > max_value&& std::abs(avg_x) > sub_img.cols/2)
                {
                    max_value = avg_y;
                    max_index = i;
                }

                double dis = std::pow((testEDLines.getLines()[i].start.y + testEDLines.getLines()[i].end.y), 2) + std::pow((testEDLines.getLines()[i].start.x + testEDLines.getLines()[i].end.x), 2);
                dis = std::sqrtf(dis);
                if (dis > 100) {

                    cv::line(dis_img, testEDLines.getLines()[i].start, testEDLines.getLines()[i].end, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);
                }

            }
           /* cv::line(dis_img, cv::Point2d(testEDLines.getLines()[min_index].start.x+ af_rect.x, testEDLines.getLines()[min_index].start.y+ af_rect.y),
                cv::Point2d(testEDLines.getLines()[min_index].end.x+ af_rect.x, testEDLines.getLines()[min_index].end.y+ af_rect.y), cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);

            cv::line(dis_img, cv::Point2d(testEDLines.getLines()[max_index].start.x + af_rect.x, testEDLines.getLines()[max_index].start.y + af_rect.y),
                cv::Point2d(testEDLines.getLines()[max_index].end.x + af_rect.x, testEDLines.getLines()[max_index].end.y + af_rect.y), cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0);*/
        }
        double sumY = 0;
        double total_w = 0;
        //权重考虑是否加指数
        for (auto item : center_info) {
            int w = 255 - item.first; //权重
            total_w = total_w + w;
            sumY = sumY + w * item.second;
        }
        cv::Point2f ret_center;
        ret_center.x = center_x;
        ret_center.y = sumY / total_w;
        //原图画出来重点
        cv::circle(dis_img, ret_center, 1, cv::Scalar(0, 255, 0),1);
        center_pt_info.emplace_back(ret_center);
    }

    //TODO 是否需要进行排序，暂时不确定
    cv::Mat rotate_M = fit_line(src, gray_img, center_pt_info);
    //将重点旋转还原
    std::vector<cv::Point2f> rotate_center_pt;
    for (int i = 0; i < center_pt_info.size();i++) {
        cv::Point2f tmp_pt = TransPoint(rotate_M, center_pt_info[i]);
        rotate_center_pt.emplace_back(tmp_pt);
    }
    //TODO 计算误差，与图纸坐标系。左上角点为原点

    cv::imwrite(R"(E:\demo\test\zhigong\ret\)" + std::to_string(img_count) + ".jpg", dis_img);
    img_count++;

}

void  get_center(cv::Mat th_img, cv::Point2f&center) {
    double sumX = 0, sumY = 0;
    double total_gray = 0;
    for (int i = 0; i < th_img.rows;i++) {
        for (int j = 0; j < th_img.cols; j++) {
            double gray_value = static_cast<double>(th_img.at<uchar>(i,j));
            sumX = sumX + j * gray_value;
            sumY = sumY + i * gray_value;
            total_gray = total_gray + gray_value;
        }
    }
    double center_x = sumX / total_gray;
    double center_y = sumY / total_gray;
    center.x = center_x;
    center.y = center_y;
}

//阈值迭代求灰度重心，在阈值附近步进
std::vector<cv::Point2f> post_process(cv::Mat src, cv::Mat gray_src,cv::Mat gray_mask) {

    dis_img = src.clone();
    std::vector<std::vector<cv::Point>> filter_contours;
    std::vector<cv::Vec4i> filter_hierarchy;
    cv::findContours(gray_mask, filter_contours, filter_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    std::vector<cv::Point2f> total_center;
    for (size_t i = 0; i < filter_contours.size(); ++i) {
        cv::Rect rect = cv::boundingRect(filter_contours[i]);


        cv::RotatedRect rot_rect = cv::minAreaRect(filter_contours[i]);

        cv::Point2f box_pts[4];
        rot_rect.points(box_pts);
        std::vector<cv::Point> pt_vec;
        for (int i = 0; i != 4; ++i) pt_vec.emplace_back(cv::Point2i(box_pts[i]));
        std::vector<std::vector<cv::Point>> tmpContours;
        tmpContours.insert(tmpContours.end(), pt_vec);
        cv::drawContours(dis_img, tmpContours, 0, cv::Scalar(0, 0, 255), 1);

        /*cv::Mat boxpts;
        cv::boxPoints(rot_rect, boxpts);
        boxpts.convertTo(boxpts, CV_32S);*/

        //cv::polylines(dis_img, pts,true, cv::Scalar(0, 0, 255),1);
        //cv::rectangle(dis_img, rect, cv::Scalar(0, 0, 255), 1);

        cv::Rect af_rect;
        af_rect.x = rect.x - 5;
        af_rect.y = rect.y - 5;
        af_rect.width = rect.width + 10;
        af_rect.height = rect.height + 10;
        cv::Mat sub_img = gray_src(af_rect);
        std::vector<cv::Point2f> dst_pt;
        //均值在70到110 之间
        int mean_value = cv::mean(sub_img)[0];
        //以均值为基础，向上迭代5次
        int end_value = mean_value + 50;
        std::map<int,cv::Point2f> center_info;
        for (int cur_th = mean_value; cur_th <= end_value;cur_th = cur_th+10) {
            cv::Mat thre_img;
            cv::threshold(sub_img, thre_img, cur_th,255,cv::THRESH_BINARY);
            cv::Point2f center;
            get_center(thre_img, center);
            //此时的center是相对于裁剪之后的，需要还原
            center.x = af_rect.x + center.x;
            center.y = af_rect.y + center.y;
            center_info.insert(std::pair<int, cv::Point2f>(cur_th, center));
        }
        double sumX = 0, sumY = 0;
        double total_w = 0;
        for (auto item: center_info) {
            int w = 255 - item.first; //权重
            total_w = total_w + w;
            sumX = sumX +  w * item.second.x;
            sumY = sumY +  w * item.second.y;
        }
        cv::Point2f ret_center;
        ret_center.x = sumX / total_w;
        ret_center.y = sumY / total_w;
        total_center.emplace_back(ret_center);
        cv::circle(dis_img,ret_center,1,cv::Scalar(0,0,255),1);
    }
   /* cv::imwrite(R"(E:\demo\test\zhigong\ret\)" + std::to_string(img_count) + ".jpg", dis_img);
    img_count++;*/

    return total_center;
}
cv::Mat gray_stairs(const cv::Mat& img, double sin, double hin, double mt, double sout, double hout)
{
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

 // 坐标点仿射变换
 cv::Point2f TransPoint(const cv::Mat& M, const cv::Point2f& point)
 {
     std::vector<double> values = { point.x, point.y };
     cv::Mat mat = cv::Mat(values).clone(); //将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
     cv::Mat dest = mat.reshape(1, 1);

     cv::Mat homogeneousPoint = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
     cv::Mat transformed = M * homogeneousPoint;
     return cv::Point2f(transformed.at<double>(0, 0), transformed.at<double>(0, 1));
 }

 cv::Mat d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5)
 {
     cv::Mat mtx(3, 3, CV_64FC1);
     double* M = mtx.ptr<double>();
     M[0] = d0;
     M[1] = d1;
     M[2] = d2;
     M[3] = d3;
     M[4] = d4;
     M[5] = d5;
     M[6] = 0.0;
     M[7] = 0.0;
     M[8] = 1.0;
     return mtx;
 }

 // 讲OpenCV输出矩阵转换为齐次坐标格式，2x3 => 3x3
 cv::Mat cvMat6_to_cvMat9(const cv::Mat& mtx6)
 {
     cv::Mat mtx9(3, 3, CV_64FC1);
     double* M9 = mtx9.ptr<double>();
     const double* M6 = mtx6.ptr<double>();
     M9[0] = M6[0];
     M9[1] = M6[1];
     M9[2] = M6[2];
     M9[3] = M6[3];
     M9[4] = M6[4];
     M9[5] = M6[5];
     M9[6] = 0.0;
     M9[7] = 0.0;
     M9[8] = 1.0;
     return mtx9;
 }

 cv::Mat vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2)
 {
     cv::Point2f center(x1, y1);
     double angle = d2 - d1;
     cv::Mat rot_M = cv::getRotationMatrix2D(center, angle, 1.0);
     rot_M = cvMat6_to_cvMat9(rot_M);

     cv::Mat trans_M = d6_to_cvMat(1, 0, x2 - x1, 0, 1, y2 - y1);
     cv::Mat M = trans_M * rot_M; // 先旋转在平移（矩阵乘法相反）
     return M;
 }

 //拟合直线，得到旋转矩阵
 cv::Mat fit_line(const cv::Mat& src, const cv::Mat& gray_img, std::vector<cv::Point2f> pts) {
     //先分类，y值相差100以内为一组
     std::vector<std::vector<cv::Point2f>> pt_vec;
     std::vector<int> used(pts.size(),-1);
     for (int i = 0; i < pts.size();i++) {
         if (used[i] != -1) continue;
         double first_y = pts[i].y;
         std::vector<cv::Point2f> tmp_vec;
         tmp_vec.emplace_back(pts[i]);
         for (int j = i + 1; j < pts.size(); j++) {
             if (used[j] != -1) continue;
             double second_y = pts[j].y;
             if (std::abs(first_y - second_y) < 100) {
                 tmp_vec.emplace_back(pts[j]);
                 used[i] = 1;
                 used[j] = 1;
             }
         }
         //排序
         std::sort(tmp_vec.begin(), tmp_vec.end(), [&]( const cv::Point2f& lhs,const cv::Point2f&rhs) {
             if (lhs.x < rhs.x) {
                 return true;
             }
             return false;
             });

         pt_vec.emplace_back(tmp_vec);
     }

     //分成了8组，每组拟合直线，求角度，排序从上到下，从左到右
     std::sort(pt_vec.begin(), pt_vec.end(), [&](const std::vector<cv::Point2f>& lhs,const std::vector<cv::Point2f>& rhs) {
         int l_avg_x = 0;
         int l_avg_y = 0;
         int r_avg_x = 0;
         int r_avg_y = 0;
         for (int i = 0; i < lhs.size();i++) {
             l_avg_x += lhs[i].x;
             l_avg_y += lhs[i].y;
         }
         l_avg_x = l_avg_x / lhs.size();
         l_avg_y = l_avg_y / lhs.size();

         for (int i = 0; i < rhs.size(); i++) {
             r_avg_x += rhs[i].x;
             r_avg_y += rhs[i].y;
         }
         r_avg_x = r_avg_x / rhs.size();
         r_avg_y = r_avg_y / rhs.size();

         if (l_avg_y< r_avg_y) {
             return true;
         }
         return false;

        });
     int count = 0;
     double angel_sum = 0;
     for (int i = 0; i < pt_vec.size(); i++) {
        cv::Vec4f lineParams;
        cv::fitLine(pt_vec[i], lineParams, cv::DIST_HUBER, 0, 0.01, 0.01);
        double angleRad = std::atan(lineParams[1] / lineParams[0]);
        angel_sum += angleRad;
        count += 1;
     }
     double angle = (angel_sum / count) * 180.0 / CV_PI;
     //取左上角点
     cv::Point2f l_t_pt = pt_vec[0][0];
     //cv::Mat rot_M = cv::getRotationMatrix2D(l_t_pt, angle, 1.0);
     //变换矩阵
     cv::Mat M = vector_angle_to_M(l_t_pt.x, l_t_pt.y, 0, 0, 0, angle);
     return M;
 }
void test_zhigong(const std::string& imgs_path)
{
    std::vector<std::string> file_path;
    nao::fl::getAllFormatFiles(imgs_path, file_path);
    for (int i = 0; i < file_path.size(); i++) {
        cv::Mat cur_src_img = cv::imread(file_path[i]);
        cv::Mat gray_img,gray_mask;
        cv::cvtColor(cur_src_img, gray_img, cv::COLOR_RGB2GRAY);
        //阈值处理
        thre_process(gray_img,gray_mask);
        //对得到mask图进行校验，得到修改后的mask图，再原图上进行精确定位查找
        y_process(cur_src_img, gray_mask);
        //阈值迭代，获取中心点
        std::vector<cv::Point2f> total_center = post_process(cur_src_img, gray_img, gray_mask);

        thre_process_2(cur_src_img, gray_img, gray_mask);

        //拟合直线求误差
        //
        //画出结果图

    }
}


void test_chang(const std::string& imgs_path) {

    std::vector<std::string> file_path;
    nao::fl::getAllFormatFiles(imgs_path, file_path);
    for (int i = 0; i < file_path.size(); i++) {
        cv::Mat cur_src_img = cv::imread(file_path[i]);
        cv::Mat gray_img, gray_mask;
        cv::cvtColor(cur_src_img, gray_img, cv::COLOR_RGB2GRAY);
        //阈值处理
        thre_process_2(cur_src_img ,gray_img,gray_mask);
        //对得到mask图进行校验，得到修改后的mask图，再原图上进行精确定位查找
        //y_process(cur_src_img, gray_mask);
        //阈值迭代，获取中心点
        //std::vector<cv::Point2f> total_center = post_process(cur_src_img, gray_img, gray_mask);
        //拟合直线求误差
        //
        //画出结果图

    }
}