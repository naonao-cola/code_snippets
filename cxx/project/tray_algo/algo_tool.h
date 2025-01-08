/**
 * @FilePath     : /tray_algo/src/custom/algo_tool.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2024-06-20 14:16:43
 **/
#pragma once
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>

#ifndef __ALGO_TOOL_H__
#define __ALGO_TOOL_H__

namespace tool {
/**
 * @brief 直方图匹配，正矩形的框进行匹配
 * @param hist_img
 * @param hist_rect
 * @param template_img
 * @param template_rect
 * @return
 */
std::optional<cv::Mat> get_equal_img(const cv::Mat& hist_img,
                                     const cv::Rect& hist_rect,
                                     const cv::Mat& template_img,
                                     const cv::Rect& template_rect);

/**
 * @brief 直方图匹配，旋转矩形的四个点进行匹配
 * @param hist_img
 * @param hist_pts
 * @param template_img
 * @param template_pts
 * @return
 */
std::optional<cv::Mat> get_equal_img_2(const cv::Mat& hist_img,
                                       const std::vector<cv::Point2f>& hist_pts,
                                       const cv::Mat& template_img,
                                       std::vector<cv::Point2f>& template_pts);

/**
 * @brief 获取旋转矩形的四个点
 * @param r_rect
 * @return
 */
std::optional<std::vector<cv::Point2f>> get_rotated_rect_pts(
    const cv::RotatedRect& r_rect);

/**
 * @brief 根据旋转矩形将图像转正，并返回转正后的四个点
 * @param r_rect
 * @param img
 * @return
 */
std::optional<std::vector<cv::Point2f>> get_rotated_rect_pt_trans_and_rotate(
    const cv::RotatedRect& r_rect,
    cv::Mat& img);

/**
 * @brief 点排序，左上角第一个点，顺时针右上角第二个点
 * @tparam T
 * @param src_pt
 * @return
 */
template <typename T>
std::vector<T> order_pts(const std::vector<T> src_pt)
{
    std::vector<T> tmp;
    std::vector<T> dst;
    if (src_pt.size() != 4) {
        std::cerr << "input point count faile" << std::endl;
        return dst;
    }
    tmp.assign(src_pt.begin(), src_pt.end());
    // 按照x值大小升序排列，x值小的两个点位左侧的两个点
    std::sort(tmp.begin(), tmp.end(), [=](const T pt1, const T pt2) {
        return pt1.x < pt2.x;
    });
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

/**
 * @brief 获取透视矩阵
 * @tparam T
 * @param src_points
 * @param dst_points
 * @param wrap_mat
 */
template <typename T>
void perspective(const std::vector<T> src_points,
                 const std::vector<T> dst_points,
                 cv::Mat& wrap_mat)
{
    if (src_points.size() != 4 || dst_points.size() != 4) return;
    std::vector<T> src_tmp = order_pts(src_points);
    std::vector<T> dst_tmp = order_pts(dst_points);
    std::array<T, 4> src;
    std::array<T, 4> dst;
    for (int i = 0; i < src_tmp.size(); i++) {
        src[i] = T(src_tmp[i].x, src_tmp[i].y);
    }
    for (int i = 0; i < dst_tmp.size(); i++) {
        dst[i] = T(dst_tmp[i].x, dst_tmp[i].y);
    }
    wrap_mat = cv::getPerspectiveTransform(src, dst).clone();
}

/**
 * @brief 点变换，根据透视矩阵进行点变换，将原图的点变换到目标图
 * @tparam T
 * @param src_point
 * @param wrap_mat
 * @param dst_point
 */
template <typename T>
void point_transform(const T src_point, const cv::Mat wrap_mat, T& dst_point)
{
    cv::Mat_<double> pts(3, 1);
    pts(0, 0) = src_point.x;
    pts(0, 1) = src_point.y;
    pts(0, 2) = 1;
    cv::Mat ret = wrap_mat * pts;
    double a1_v = ret.at<double>(0, 0);
    double a2_v = ret.at<double>(1, 0);
    double a3_v = ret.at<double>(2, 0);
    dst_point.x = static_cast<double>(a1_v / a3_v);
    dst_point.y = static_cast<double>(a2_v / a3_v);
}

/**
 * @brief 点反变换，根据透视矩阵反变换，将目标图的点反变换到原图
 * @tparam T
 * @param src_point
 * @param wrap_mat
 * @param dst_point
 */
template <typename T>
void point_inv_transform(const T& src_point, const cv::Mat& wrap_mat, T& dst_point)
{
    cv::Mat M_inv;
    cv::invert(wrap_mat, M_inv, cv::DECOMP_SVD);
    cv::Mat_<double> pt(3, 1);
    pt(0, 0) = src_point.x;
    pt(0, 1) = src_point.y;
    pt(0, 2) = 1;
    cv::Mat ret = M_inv * pt;
    double a1 = ret.at<double>(0, 0);
    double a2 = ret.at<double>(1, 0);
    double a3 = ret.at<double>(2, 0);
    dst_point.x = static_cast<double>(a1 / a3);
    dst_point.y = static_cast<double>(a2 / a3);
}

/**
 * @brief 轮廓提取
 * @param src
 * @return
 */
std::optional<std::vector<std::vector<cv::Point>>> get_contours(
    const cv::Mat& src);

/**
 * @brief 色阶变换，相当于灰度值拉伸或收缩
 * @param img
 * @param sin
 * @param hin
 * @param mt
 * @param sout
 * @param hout
 * @return
 */
cv::Mat gray_stairs(const cv::Mat& img, double sin, double hin, double mt, double sout, double hout);

/**
 * @brief gama变换
 * @param img
 * @param gamma
 * @param n_c
 * @return
 */
cv::Mat gamma_trans(const cv::Mat& img, double gamma, int n_c = 1);

/**
 * @brief 二值化类型
 */
enum class THRESHOLD_TYPE {
    HUANG = 1,
    LI = 5,
    MIN_ERROR = 8,
    MINIMUM = 9,
    OTSU = 11,
    YEN = 16,
    SAUVOLA = 18
};

/**
 * @brief 获取直方图
 * @param src
 * @param dst
 */
static void get_histogram(const cv::Mat& src, int* dst);

/**
 * @brief 执行二值化函数
 * @param src
 * @param type
 * @param doIblack 忽略黑色的下限，低于这个值不进入统计
 * @param doIwhite 忽略白色的上限，高于这个值不进入统计
 * @param reset    是否执行二值化，还是只计算阈值
 * @return
 */
int exec_threshold(cv::Mat& src, THRESHOLD_TYPE type, int doIblack = -1, int doIwhite = -1, bool reset = false);
/**
 * @brief 局部阈值化，利用均值与方差进行比较，逐像素比较
 * @param src
 * @param k
 * @param wnd_size
 * @return
 */
int sauvola(cv::Mat& src, const double& k = 0.1, const int& wnd_size = 7);

/**
 * @brief 点变换，适用于旋转平移的点变换
 * @param M
 * @param point
 * @return
 */
cv::Point2d TransPoint(const cv::Mat& M, const cv::Point2d& point);

/**
 * @brief 计时器类
 */
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
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
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
} // namespace tool
#endif