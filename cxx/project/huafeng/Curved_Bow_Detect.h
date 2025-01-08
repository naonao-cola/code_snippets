/**
 * @FilePath     : /connector/src/custom/Curved_Bow_Detect.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-01-08 11:40:11
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#pragma once
#include "../framework/BaseAlgo.h"
#include "./sub_3rdparty/tival/include/ShapeBasedMatching.h"
#include "./sub_3rdparty/tival/include/FindLine.h"
//卡尺结构体
struct caliper_param {
public:
    int caliper_num = 10;
    int caliper_length = 20;
    int caliper_width = 5;
    int sigma = 1;
    std::string transition = "all";
    int num = 1;
    int contrast = 30;
    //卡尺区域
    int center_x = 0;
    int center_y = 0;
    int box_width = 0;
    int box_height = 0;
    double angle = 0;
    bool sort_by_score = false;
};

struct match_ret {
public:
    int x;
    int y;
    double angle;
    double score;
    double scale;
};

struct basis_info {
public:
    match_ret match_info;
    std::vector<cv::Rect> rect_vec;
    //卡尺的点
    std::vector<std::vector<cv::Point2f>> pt;
    std::vector<Tival::FindLineResult> line_vec;
    int index;
    cv::Point2f trans_point;
    int error_flag = 0;
};

class Curved_Bow_Detect : public BaseAlgo {
public:
    Curved_Bow_Detect();
    ~Curved_Bow_Detect();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
    std::tuple<std::string, json>get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map);


    //全图查找线与模板，并计算距离
    void img_process(const cv::Mat& src, AlgoResultPtr algo_result);
   
private:
    DCLEAR_ALGO_GROUP_REGISTER(Curved_Bow_Detect)

    //获取左右卡尺的中心
    cv::Point2f get_caliper_center(cv::Point2f param[], const cv::Mat& img, std::string direction, cv::Point2f&  pt_center,Tival::FindLineResult line_ret, AlgoResultPtr algo_result);

    //模板匹配参数
    int num_                      = 100;
    int angle_min_                = -10;
    int angle_max_                = 10;
    double scale_min_             = 0.8;
    double scale_max_             = 1.2;
    int contrast_                 = 30;
    double min_score_             = 0.45;
    double strength_              = 0.2;
    double max_overlap_           = 0.5;
    bool sort_by_score_           = false;
    std::string path_             = "";
    int detect_flag_              = 1;
    //基座信息
    int basis_x_num_              = 12;
    int basis_y_num_              = 8;
    //检测区域
    int detect_left_x_            = 0;
    int detect_left_y_            = 0;
    int detect_width_             = 9344;
    int detect_height_            = 7000;
    //屏蔽片参数
    double gamma_value_           = 0.8;
    int  small_film_t_brightness_ = 255;
    int  small_film_b_brightness_ = 220;
    int  small_film_t_area_       = 2000;
    int  small_film_b_area_       = 1000;

     
    //模板信息
    int template_x_               = 0;
    int template_y_               = 0;
    int template_width_           = 0;
    int template_height_          = 0;
    int template_cx_              = 0;
    int template_cy_              = 0;
    /*
    卡尺的顺序，左屏蔽片，右屏蔽片，下屏蔽片。
    左基座，右基座，下基座
    */
    std::vector<std::shared_ptr<caliper_param>> caliper_param_vec_;
    //缺失基座，理论值误差值
    std::vector<cv::Point> miss_;
    std::vector<double> d_value_;
    std::vector<double> error_value_;
    //偏移设置
    std::vector<double> ratio_vec_;
    std::vector<double> constant_vec_;

    double pix_value_ = 5.26;
    



    //相对位置初始化
    std::vector<cv::Vec4i> position_{
        cv::Vec4i(0,0,100,336),//左屏蔽片
        cv::Vec4i(622,-4,100,374),//右屏蔽片
        cv::Vec4i(387,286,90,540),//下屏蔽片
        cv::Vec4i(140,-118,46,126),//左基座
        cv::Vec4i(461,-118,46,126),//右基座
        cv::Vec4i(304,-54,44,410)//下基座
    };

   

    bool cvt_match(const Tival::SbmResults& ret,std::vector<match_ret>& order_ret, AlgoResultPtr algo_result,const cv::Mat&  src, std::vector<basis_info>& basis_info_vec);
    //每一个小基座的检测
    void basis_find_line(const cv::Mat& src,AlgoResultPtr algo_result, basis_info& cur_basis,int index);
    //卡尺参数获取图像与矩形框
    void get_caliper_rect_img(const caliper_param& param,const cv::Mat& src,cv::Rect& caliper_rect,cv::Mat& caliper_img, Tival::TPoint& start, Tival::TPoint& end,int index,const cv::Point2f cur_pt_vec[]);
    //比较排序
    void compare_sbm_results(std::vector<match_ret>& ret);
    bool get_param(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
    
    std::atomic_bool  status_flag = true;


    void basis_cal_line(std::vector<basis_info> root_basis_vec,cv::Mat src, AlgoResultPtr algo_result);

    int get_nearest_point_idx(cv::Mat points, cv::Point2d refPoint, double& minDist);
};