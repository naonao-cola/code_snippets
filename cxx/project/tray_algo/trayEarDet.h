/**
 * @FilePath     : /tray_algo/src/custom/trayEarDet.h
 * @Description  :
 * @Author       : weiwei.wang
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2024-06-20 14:42:07
 **/
#pragma once
#include <filesystem> // C++17
#include <time.h>
#include "../framework/BaseAlgo.h"
namespace fs = std::filesystem;
class trayEarDet : public BaseAlgo {
public:
    trayEarDet();
    ~trayEarDet();
    AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    /**
     * @brief 获取参数
     * @param task
     * @return
     */
    bool get_param(InferTaskPtr task);

    /**
     * @brief 获取任务信息
     * @param task
     * @param param_map
     * @return
     */
    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::map<std::string, json> param_map) const;

    /**
     * @brief 获取交点，获取矩形轮廓的4个交点。得到4个点
     * @param img
     * @param pt_vec
     * @param threshold_value
     * @return
     */
    std::vector<cv::Point2f> get_pts(const cv::Mat& img, const std::vector<cv::Point2f>& pt_vec, int threshold_value);

    /**
     * @brief 获取交点，矩形轮廓的每个角的准确交点，得到一个点
     * @param img
     * @param rect
     * @param type
     * @return
     */
    cv::Point2f get_cross_pt(const cv::Mat& img, const cv::Rect& rect, int type);

    /**
     * @brief 3*5 类型图片的检测
     * @param src
     * @param algo_result
     */
    void img_process1(const cv::Mat& src, AlgoResultPtr algo_result);
    /**
     * @brief 2*9类型图片的检测
     * @param src
     * @param algo_result
     */
    void img_process2(const cv::Mat& src, AlgoResultPtr algo_result);

    /**
     * @brief 保存调试图片
     * @param input_img
     * @param input_pts
     */
    void write_debug_img(const cv::Mat& input_img, const std::vector<std::vector<cv::Point2f>>& input_pts, int unknown_flag = 0);

    // 3*5模板图片路径
    std::string template_img_path_1_;
    // 2*9 模板图片路径
    std::string template_img_path_2_;
    // 3*5模板图像
    cv::Mat template_img_1_;
    // 2*9 模板图像
    cv::Mat template_img_2_;
    // 模板一检测的位置点，相对于左上角
    std::vector<cv::Point2f> template_pt_0_;
    // 模板二检测的位置，相对于左上角
    std::vector<cv::Point2f> template_pt_1_;
    // 齿的类型，0表示3*5,1表示2*9
    int tower_type_ = 0;
    // 判定参数
    int area_th_ = 120;
    int img_th_ = 150;
    int img_th_2_ = 180;

    // 3*5 的检测区域
    std::vector<cv::Vec4i> temp_mask_1_{
        cv::Vec4i(-53, 136, 61, 182),
        cv::Vec4i(-53, 342, 60, 187),
        cv::Vec4i(-51, 550, 58, 176),
        cv::Vec4i(-51, 748, 61, 190),
        cv::Vec4i(-48, 958, 60, 171),
        cv::Vec4i(-52, 1158, 64, 192),
        cv::Vec4i(3458, 144, 49, 181),
        cv::Vec4i(3459, 341, 54, 194),
        cv::Vec4i(3453, 553, 47, 187),
        cv::Vec4i(3454, 757, 63, 192),
        cv::Vec4i(3453, 966, 40, 180),
        cv::Vec4i(3452, 1175, 60, 190)};
    // 2*9的检测区域
    std::vector<cv::Vec4i> temp_mask_2_{
        cv::Vec4i(-47, 139, 55, 182),
        cv::Vec4i(-46, 341, 56, 184),
        cv::Vec4i(-46, 550, 55, 178),
        cv::Vec4i(-47, 751, 58, 188),
        cv::Vec4i(-48, 954, 60, 186),
        cv::Vec4i(-47, 1160, 58, 190),
        cv::Vec4i(3452, 134, 60, 189),
        cv::Vec4i(3452, 337, 60, 190),
        cv::Vec4i(3451, 548, 63, 182),
        cv::Vec4i(3449, 754, 65, 193),
        cv::Vec4i(3450, 960, 63, 184),
        cv::Vec4i(3451, 1164, 63, 198)};

    /**
     * @brief 点变换，将点加入到结果中
     * @param input_rect
     * @param wrap_mat
     * @param algo_result
     * @return
     */
    std::vector<cv::Point2f> wrap_point(const cv::Rect& input_rect, const cv::Mat& wrap_mat, AlgoResultPtr algo_result);
    std::string image_file_name_ = "";
    DCLEAR_ALGO_GROUP_REGISTER(trayEarDet)
};