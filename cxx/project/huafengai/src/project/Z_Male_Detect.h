#pragma once
#include "../../modules/tv_algo_base/src/framework/BaseAlgo.h"
#include <tuple>

class Z_Male_Detect : public BaseAlgo
{
public:
    Z_Male_Detect();
    ~Z_Male_Detect();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    struct point_info
    {
        cv::Point2d img_pt;    // 图上的点
        cv::Point2d tran_pt;   // 变换后的点
        cv::Point2d cal_pt;    // 毫米点计算点
        cv::Point2d org_pt;    // 设计图点
        double      offset_x;
        double      offset_y;
        bool        is_ok;
        int         index;
    };
    // 像素当量
    double pix_value_ = 5.26;
    // 几行几列
    int z_male_x_num_ = 6;
    int z_male_y_num_ = 8;

    // 检测区域
    int detect_left_x_ = 0;
    int detect_left_y_ = 0;
    int detect_width_  = 0;
    int detect_height_ = 0;

    // gamma 参数
    double gamma_value_     = 0.8;
    int    threshold_value_ = 180;
    int    area_range_b_    = 130;
    int    rect_height_t_   = 80;
    int    rect_height_b_   = 45;
    int    rect_width_t_    = 550;
    int    rect_width_b_    = 450;
    double rect_rate_       = 4.0;

    // 误差参数
    double error_value_ = 0.12;

    // 黄铜针的参数
    int    s_threshold_value_ = 200;
    double s_gamma_value_     = 0.8;
    int    s_area_range_b_    = 100;
    int    s_rect_height_t_   = 100;
    int    s_rect_height_b_   = 100;
    int    s_rect_width_t_    = 100;
    int    s_rect_width_b_    = 100;
    double s_rect_rate_       = 3.2;
    double s_angle_           = 88.5;
    double s_angle_max_       = 5.0;
    double s_no_zero_         = 0.4;

    // 图像偏转
    int mode = 90;

    // 原点中心在图中心，计算是在左上角设置偏移量
    double offset_x = 0.0;
    double offset_y = 0.0;

    // 模板点
    std::vector<cv::Point2f> tempImagePoint_1_;   // 黄针
    std::vector<cv::Point2f> tempImagePoint_2_;   // 黄铜片

    cv::Mat                       template_mat_1_, template_mat_2_;
    std::vector<cv::Point2f>      miss_1_;   // 黄针
    std::vector<cv::Point2f>      miss_2_;   // 黄铜片
    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map) const;
    bool                          get_param(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

    std::vector<cv::Point2f> img_process_1(const cv::Mat& src, const cv::Mat& gray_img, AlgoResultPtr algo_result, double& angle) noexcept;
    std::vector<cv::Point2f> img_process_2(const cv::Mat& src, const cv::Mat& gray_img, AlgoResultPtr algo_result, double& angle) noexcept;
    cv::Mat                  fit_line(const cv::Mat& src, const cv::Mat& gray_img, std::vector<cv::Point2f> pts, double angle);
    void        cal_data(const cv::Mat& src, const cv::Mat& gray_img, std::vector<cv::Point2f> pt_vec_1, std::vector<cv::Point2f> pt_vec_2, double angle, AlgoResultPtr algo_result) noexcept;
    int         get_nearest_point_idx(cv::Mat points, cv::Point2f refPoint, double& minDist);
    static bool compare(const point_info& lhs, point_info& rhs);


    cv::Mat          features_1_, features_2_;
    std::atomic_bool status_flag = true;

    cv::Mat get_col_row(std::vector<cv::Point2f>);


    // 二次查找
    point_info refind_point_1(const cv::Mat& img, cv::Point2f pt, int idx, cv::Mat trans_m, double error_pix, int pad = 5);
    DCLEAR_ALGO_GROUP_REGISTER(Z_Male_Detect)
};