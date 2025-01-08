/**
 * @FilePath     : /t3cg6/src/project/details.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-12-03 16:29:48
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-03 16:31:30
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#include "nlohmann/json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace xx
{
/**
 * @brief: 获取json 的键
 * @param json_obj
 * @param key
 * @param def_val
 * @return
 * @note :
 **/
template<typename T>
T GetProperty(const nlohmann::json& json_obj, const std::string& key, const T& def_val)
{
    return json_obj.contains(key) ? json_obj[key].get<T>() : def_val;
}

/**
 * @brief: 画出结果图
 * @param dispaly
 * @return
 * @note :
 **/
void DrawImg(cv::Mat dispaly, nlohmann::json json_info);

nlohmann::json pt_json(std::vector<std::vector<cv::Point>> mask);

nlohmann::json pt_json(std::vector<cv::Point2f> mask);

template<typename T>
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
        }
        else {
            dst.emplace_back(tmp[1]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[0]);
        }
    }
    else {
        if (tmp[2].y > tmp[3].y) {
            dst.emplace_back(tmp[0]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[1]);
        }
        else {
            dst.emplace_back(tmp[0]);
            dst.emplace_back(tmp[2]);
            dst.emplace_back(tmp[3]);
            dst.emplace_back(tmp[1]);
        }
    }
    return dst;
}

/**
 * @brief 色阶变换
 * @param img
 * @param sin
 * @param hin
 * @param mt
 * @param sout
 * @param hout
 * @return
 */
cv::Mat gray_stairs(const cv::Mat& img, double sin = 0.0, double hin = 255.0, double mt = 1.0, double sout = 0.0, double hout = 255.0);


/**
 * @brief gama变换
 * @param img
 * @param gamma
 * @param n_c
 * @return
 */
cv::Mat gamma_trans(const cv::Mat& img, double gamma, int n_c = 1);


/**
 * @brief 分段增强
 * @param img
 * @param r1
 * @param r2
 * @param s1
 * @param s2
 * @return
 */
cv::Mat segmented_enhancement(const cv::Mat& img, double r1, double r2, double s1 = 0, double s2 = 255);

/**
 * @brief
 * @param src
 * @param th_low
 * @param th_high
 * @param blur_w
 * @param blur_h
 * @param row_beg
 * @param row_end
 * @param col_beg
 * @param col_end
 * @param type
 */
void get_edge(const cv::Mat& src, int th_low, int th_high, int blur_w, int blur_h, int& row_beg, int& row_end, int& col_beg, int& col_end, int continuous, int th_value, int type);

int getmean(const cv::Mat& img, const std::vector<cv::Point>& countor, const cv::Point& offest);



/**
 * @brief 计时器类
 */
class Timer
{
public:
    Timer()
        : beg_(clock_::now())
    {
    }
    void   reset() { beg_ = clock_::now(); }
    double elapsed() const { return std::chrono::duration_cast<second_>(clock_::now() - beg_).count(); }
    void   out(std::string message = "")
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
    typedef std::chrono::high_resolution_clock           clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_>                      beg_;
};
}   // namespace xx