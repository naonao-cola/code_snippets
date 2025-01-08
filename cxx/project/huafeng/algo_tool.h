
#pragma once
#include "sub_3rdparty/tival/include/FindLine.h"
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#ifndef __ALGO_TOOL_H__
#define __ALGO_TOOL_H__

// 项目工具函数
namespace connector {

// 计算部分，求两直线交点
cv::Point2d get2lineIPoint(cv::Point2d lineOnePt1, cv::Point2d lineOnePt2, cv::Point2d lineTwoPt1, cv::Point2d lineTwoPt2);
// 求两点距离
double dist_p2p(const cv::Point2f& a, const cv::Point2f& b);
// 点到直线的距离
float dist_p2l(cv::Point pointP, cv::Point pointA, cv::Point pointB);
// 计算点到线的垂足
cv::Point2f calculate_foot_point(cv::Point2f line_pt1, cv::Point2f line_pt2, cv::Point2f src_pt);

double get_line_x(cv::Point2f line_p1, cv::Point2f line_p2, double y);

double get_line_y(cv::Point2f line_p1, cv::Point2f line_p2, double x);

// 获取直线的斜率与截距
cv::Point2f get_lines_fangcheng(const Tival::FindLineResult& ll);
// 求直线的中垂线
Tival::FindLineResult get_med_line(const Tival::FindLineResult& ll, const Tival::FindLineResult& lr);

// 第二种求中线的方式，将基座线向上平移66，70，左右，求与两边的交点
Tival::FindLineResult get_med_line_2(const Tival::FindLineResult& ll, const Tival::FindLineResult& lr, const Tival::FindLineResult& lb);

// 矩阵旋转变换部分
cv::Mat d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5);
cv::Mat cvMat6_to_cvMat9(const cv::Mat& mtx6);
// 根据源向量（带方向的点）和目标向量，生成仿射变换矩阵
cv::Mat vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2);
// 坐标点仿射变换
cv::Point2d TransPoint(const cv::Mat& M, const cv::Point2d& point);
cv::Point2f TransPoint_inv(const cv::Mat& M, const cv::Point2f& point);

// 预处理部分，灰度色阶调整,图像增强
cv::Mat gray_stairs(const cv::Mat& img, double sin, double hin, double mt, double sout, double hout);
cv::Mat gamma_trans(const cv::Mat& img, double gamma, int n_c = 1);
// 阈值处理部分,先用到这些，后面的再补
enum class THRESHOLD_TYPE {
    HUANG2 = 2,
    LI = 5,
    MINIMUM = 9,
    MOMENTS = 10,
    OTSU = 11,
    TRIANGLE = 15,
    YEN = 16,
    SAUVOLA = 18
};
// 获取直方图
static void get_histogram(const cv::Mat& src, int* dst);
int exec_threshold(cv::Mat& src, THRESHOLD_TYPE type, int doIblack = -1, int doIwhite = -1, bool reset = false);
int sauvola(cv::Mat& src, const double k = 0.1, const int wnd_size = 7);
int phansalkar(cv::Mat& src, const double k = 0.25, const int wnd_size = 7, double r = 0.5, double p = 2, double q = 10);
int minimum(std::vector<int> data);
int huang2(std::vector<int> data);
int triangle(std::vector<int> data);
// 海森矩阵求重心
void StegerLine(const cv::Mat src, std::vector<cv::Point2f>& dst_pt);
// 灰度重心法
void get_center(cv::Mat th_img, cv::Point2f& center);

// 寻找轮廓
std::vector<std::vector<cv::Point>> get_contours(const cv::Mat& src);
// 结果绘制部分
void draw_results(cv::Mat& image, const nlohmann::json& result_info);

/**
 * @brief 点排序，按照顺时针方向排序，适用于4个点，不适用于多点
 * @param src_pt，输入的原点
 * @return 排序后的点,顺时针排序的点，第一个点为左上角的点
 */
template <typename T>
std::vector<T> order_pts(const std::vector<T> src_pt)
{
    std::vector<T> tmp, dst;
    if (src_pt.size() != 4) {
        std::cerr << "输入的原点个数错误" << std::endl;
        return dst;
    }
    tmp.assign(src_pt.begin(), src_pt.end());
    // 按照x值大小升序排列，x值小的两个点位左侧的两个点
    std::sort(tmp.begin(), tmp.end(), [=](const T pt1, const T pt2) { return pt1.x < pt2.x; });
    if (tmp[0].y > tmp[1].y) {
        if (tmp[2].y > tmp[3].y) {
            dst.emplace_back(tmp[1]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[0]);
        } else {
            dst.emplace_back(tmp[1]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[0]);
        }
    } else {
        if (tmp[2].y > tmp[3].y) {
            dst.emplace_back(tmp[0]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[1]);
        } else {
            dst.emplace_back(tmp[0]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[1]);
        }
    }
    return dst;
}
class Timer {
public:
    Timer()
        : beg_(clock_::now()) {}
    void reset()
    {
        beg_ = clock_::now();
    }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }
    void out(std::string message = "")
    {
        double t = elapsed();
        std::cout << message << " elasped time:" << t << "s" << std::endl;
        reset();
    }
    double get_t()
    {
        double t = elapsed();
        reset();
        return t;
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

}
#endif