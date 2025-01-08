#pragma once
#include <opencv2/line_descriptor/descriptor.hpp>
#include <tuple>


#include "../framework/BaseAlgo.h"
#include "./sub_3rdparty/tival/include/FindLine.h"
#include "./sub_3rdparty/tival/include/ShapeBasedMatching.h"
#include "svm_train.h"
class W_Female_Detect : public BaseAlgo {
public:
    W_Female_Detect();
    ~W_Female_Detect();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

    typedef cv::line_descriptor::KeyLine KeyLine;

    // 匹配结果,lc 结构体
    struct lc_info {
    public:
        std::vector<cv::Rect> template_rect;
        std::vector<std::vector<KeyLine>> template_line_vec;
        cv::Mat h;
        cv::Mat inv_h;

        //六个矩形框，依次是左侧金属弹片，左侧定位框，右侧定位框 右侧金属弹片，上左金属弹片，上右金属弹片
        std::vector<cv::Rect> org_rect;
        std::vector<std::vector<KeyLine>> org_line_vec;

        KeyLine top_line;
        KeyLine org_top_line;

        double a1 = 0;
        double b11 = 0;
        double b12 = 0;
        double b13 = 0;
        double c11 = 0;
        double c12 = 0;
        double d11 = 0;
        double d12 = 0;
        double e1 = 0;
        double p1 = 0;
        double f1 = 0;

        double a2 = 0;
        double b21 = 0;
        double b22 = 0;
        double b23 = 0;
        double c21 = 0;
        double c22 = 0;
        double d21 = 0;
        double d22 = 0;
        double e2 = 0;
        double p2 = 0;
        double f2 = 0;

        int index;
    };

    struct lc_info_2 {
    public:
        std::vector<cv::Rect> template_rect;
        std::vector<std::vector<KeyLine>> template_line_vec;
        cv::Mat h;
        cv::Mat inv_h;

        //六个矩形框，依次是左侧金属弹片，左侧定位框，右侧定位框 右侧金属弹片，上左金属弹片，上右金属弹片
        std::vector<cv::Rect> org_rect;
        std::vector<std::vector<KeyLine>> org_line_vec;

        KeyLine top_line;
        KeyLine org_top_line;

        double a1 = 0;
        double b11 = 0;
        double b12 = 0;
        double b13 = 0;
        double c11 = 0;
        double c12 = 0;
        double d11 = 0;
        double d12 = 0;
        double e1 = 0;
        double p1 = 0;
       
        double a2 = 0;
        double b21 = 0;
        double b22 = 0;
        double b23 = 0;
        double c21 = 0;
        double c22 = 0;
        double d21 = 0;
        double d22 = 0;
        double e2 = 0;
        double p2 = 0;
       
        int index;
    };

    struct w_female {
    public:
        cv::Rect template_rect;
        std::vector<KeyLine> line_vec;
        KeyLine mid_line;
        cv::Mat h;
        cv::Mat inv_h;
        int index;
        double tl = 0;
        double tr = 0;
        double dl = 0;
        double dr = 0;
        double md = 0;
        double gap_u = 0;
        double gap_m = 0;
        double gap_d = 0;
    };

    struct w_female_2 {
    public:
        std::vector<KeyLine> line_vec;
        cv::Rect template_rect;
        cv::Mat h;
        cv::Mat inv_h;
        int index;
        double a = 0;
        double b = 0;
        double c = 0;
        double d = 0;
    };
    enum class EDGE_TYPE {
        M2LR = 0, // 中间向两边
        M2TB = 1, // 中间向上下
        M2F = 3, // 中间向四周
        B2T = 4, // 下到上
        T2B = 5, // 上到下
        LR2M = 6, // 两边到中间
        L2R = 7, // 左到右
        T2B2 = 8, // 上边两边到下边
        B2T2 = 9, // 下到上，局部最大值
    };

public:
    int detect_flag_ = 1;
    // 基座信息
    int basis_x_num_ = 12;
    int basis_y_num_ = 8;
    // 检测区域
    int detect_left_x_ = 0;
    int detect_left_y_ = 0;
    int detect_width_ = 9344;
    int detect_height_ = 7000;
    double gamma_value_1_ = 0.8;
    double gamma_value_2_ = 0.8;
    // 图像偏转
    int mode = 180;

    std::vector<cv::Vec4i> template_rect_1_ = {
        cv::Vec4i(-99, -24, 50, 90),
        cv::Vec4i(354, -27, 50, 90),
        cv::Vec4i(0, 0, 90, 33),
        cv::Vec4i(0, 0, 90, 33),
        cv::Vec4i(110, -90, 40, 30),
        cv::Vec4i(166, -90, 40, 30),
    };

    std::vector<cv::Vec4i> template_rect_2_ = {
        cv::Vec4i(-99, -24, 50, 90),
        cv::Vec4i(354, -27, 50, 90),
        cv::Vec4i(0, 0, 90, 33),
        cv::Vec4i(0, 0, 90, 33),
    };

    // 缺失基座，理论值误差值
    std::vector<cv::Point> miss_;
    std::vector<double> d_value_;
    std::vector<double> error_value_;
    // 偏移设置
    std::vector<double> ratio_vec_;
    std::vector<double> constant_vec_;
    double pix_value_ = 5.26;


    std::vector<cv::Point> right_;

    nao::svm::SvmTrain svm_obj;


    nao::svm::SvmTrain svm_obj_kai;

    nao::svm::SvmTrain svm_obj_kbai;


    nao::svm::SvmTrain svm_obj_origin_k;

    nao::svm::SvmTrain svm_obj_top_tan;

    nao::svm::SvmTrain svm_obj_lc_ce;
    bool top_tan_ = false;
    double svm_threshold_ = 0.8;


    
    double a_ = 0.1;
    double b1_ = 0.1;
    double b2_ = 0.1;
    double b3_ = 0.1;
    double c1_ = 0.1;
    double c2_ = 0.1;
    double d1_ = 0.1;
    double d2_ = 0.1;
    double e_ = 0.1;
    double p_ = 0.1;
    double f_ = 0.1;

    double error_a_ = 0.1;
    double error_b1_ = 0.1;
    double error_b2_ = 0.1;
    double error_b3_ = 0.1;
    double error_c1_ = 0.1;
    double error_c2_ = 0.1;
    double error_d1_ = 0.1;
    double error_d2_ = 0.1;
    double error_e_ = 0.1;
    double error_p_ = 0.1;
    double error_f_ = 0.1;

    double tl_ = 0;
    double tr_ = 0;
    double dl_ = 0;
    double dr_ = 0;
    double md_ = 0;
    double gap_u_ = 0;
    double gap_m_ = 0;
    double gap_d_ = 0;

    double error_tl_ = 0;
    double error_tr_ = 0;
    double error_dl_ = 0;
    double error_dr_ = 0;
    double error_md_ = 0;
    double error_gap_u_ = 0;
    double error_gap_m_ = 0;
    double error_gap_d_ = 0;

    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map) const;
    bool get_param(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

    void img_process_1(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    lc_info cal_1(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect rect, cv::Rect next_rect,int index, cv::Mat inv_m) noexcept;
    void data_cvt(std::vector<lc_info> lc_vec, AlgoResultPtr algo_result);
    std::vector<cv::Rect> get_complete_rect(std::vector<cv::Vec4i> estimate_rect_1, cv::Rect cur_rect, int c_col_idx);

    void img_process_2(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    lc_info cal_2(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect rect, cv::Rect next_rect, int index, cv::Mat inv_m) noexcept;

    void img_process_3(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    w_female cal_3(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, std::vector<cv::Mat> rbg_v1, std::vector<cv::Mat>rbg_v2,AlgoResultPtr algo_result, cv::Rect rect, int index, cv::Mat inv_m);
    void data_cvt_3(std::vector<w_female> w_female_vec, AlgoResultPtr algo_result);

    int l_th_value_ = 90;

    // 黑色基座前几行的异物面积下限
    int d_th_value_t_ = 20;
    // 黑色基座最后一行的异物面积下限
    int d_th_value_b_ = 60;
    int yellow_value_ = 80;
    
    cv::Mat get_edge(const cv::Mat& src, int th_value);

    cv::Mat find_edge(const cv::Mat& img, int& tv, int& bv, int& lv, int& rv, EDGE_TYPE edge_type, int st, int sb, int sl, int sr, int th_val, int his_val = 0, int step = 1) noexcept;
    cv::Mat find_edge_2(const cv::Mat& img, int& tv, int& bv, int& lv, int& rv, EDGE_TYPE edge_type, int st, int sb, int sl, int sr, int th_val);

    // type =0,是左边， type=1 是右边
    cv::Mat find_mid_edge_2(const cv::Mat img, int& tv, int& bv, int type);

    std::vector<int> get_his(const cv::Mat& img, int type, int th_value);
    cv::Rect reget_rect(const cv::Mat& img, const cv::Rect& rect);
    std::atomic_bool status_flag = true;

    void get_top_line_pt(cv::Mat img, cv::Point2f& l_pt, cv::Point2f& r_pt, cv::Rect rect);



    //单独的参数
    int area_th_ = 15;
    //弯母LC 找上边线
    std::vector<cv::Point2f> find_line(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect,lc_info& singal_lc);
    //弯母LC左右金属弹片
    // is_left 是否是左边
    std::vector<cv::Point2f> find_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info& singal_lc,int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2);
    //弯母左右定位框弹片
    std::vector<cv::Point2f> find_location_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2);
    
    //弯母LC上左 上右金属弹片
    std::vector<cv::Point2f> find_top_box(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2,cv::Rect cur_rect, lc_info& singal_lc, int is_left);
    

    void get_top_bottom_dege(const cv::Mat& img,int is_left,int& top_value,int &bot_value);
    void get_img_mid_value(const cv::Mat& img,double& mid_value);
    void get_img_mid_thr_line(cv::Mat img, double img_mid, cv::Point2f& b1s, cv::Point2f& b2s, cv::Point2f& b3s, cv::Point2f& b1e, cv::Point2f& b2e, cv::Point2f& b3e);
    

    //单体开口
    void img_process_4(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    w_female_2 cal_4(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, std::vector<cv::Mat> rbg_v1, std::vector<cv::Mat> rbg_v2, AlgoResultPtr algo_result, cv::Rect rect, int index, cv::Mat inv_m);
    void data_cvt_4(std::vector<w_female_2> w_female_vec, AlgoResultPtr algo_result);

    int get_percent_edge(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, double percent,int type,int left);
    int get_s_edge(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, int type,int th_value);
    

    //LC 3*8 单列
    void img_process_5(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    lc_info_2 cal_5(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect rect, cv::Rect next_rect, int index, cv::Mat inv_m) noexcept;
    std::vector<cv::Point2f> find_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info_2& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2);
    std::vector<cv::Point2f> find_line(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info_2& singal_lc);
    std::vector<cv::Point2f> find_location_box(const cv::Mat& img1, const cv::Mat& img2, cv::Rect cur_rect, lc_info_2& singal_lc, int is_left, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2);
    void data_cvt_5(std::vector<lc_info_2> lc_vec, AlgoResultPtr algo_result);



    //乳白单体开口
    void img_process_6(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    w_female_2 cal_6(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, std::vector<cv::Mat> rbg_v1, std::vector<cv::Mat> rbg_v2, AlgoResultPtr algo_result, cv::Rect rect, int index, cv::Mat inv_m);
    
    int get_percent_edge_2(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, double percent, int type, int left);
    int get_s_edge_2(std::vector<cv::Mat> hsv1, std::vector<cv::Mat> hsv2, int type, int th_value);


    //灰色LC 上面两个弹片位置不同
    void img_process_7(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;
    lc_info cal_7(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result, cv::Rect rect, cv::Rect next_rect, int index, cv::Mat inv_m);



    //开口 10*13 形态为原始形态
    void img_process_8(const cv::Mat& src_1, const cv::Mat& src_2, std::vector<cv::Mat> hsv_v1, std::vector<cv::Mat> hsv_v2, AlgoResultPtr algo_result) noexcept;


    //弹片距离下边界的距离
    double dis_p_ = 18;
    //弹片距离下边界的下线
    double dis_p_t_ = 5;

    cv::Mat input_img_1, input_img_2;
    DCLEAR_ALGO_GROUP_REGISTER(W_Female_Detect)
};