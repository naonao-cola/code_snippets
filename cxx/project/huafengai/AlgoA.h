#pragma once
#include "../../modules/tv_algo_base/src/framework/BaseAlgo.h"
/**
 * @FilePath     : /connector_ai/src/project/AlgoA.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-11-21 09:50:11
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-11-25 15:54:18
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/

#include "ShapeBasedMatching.h"
#include "details.h"
#include <opencv2/opencv.hpp>
#include <string.h>


class AlgoA : public BaseAlgo
{
public:
    AlgoA(){};
    ~AlgoA(){};
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    DCLEAR_ALGO_GROUP_REGISTER(AlgoA)

    struct CropSt
    {
        cv::Mat        img;
        cv::Point      pt;
        float          score;
        cv::Rect       box;
        int            index;
        int            row;
        nlohmann::json rotateRect;
        cv::Mat        invertMat;

        CropSt(cv::Mat limg, cv::Point lpt, float lscore, cv::Rect lbox, nlohmann::json lrotateRect, int lrow, cv::Mat linvertMat, int lindex = 0)
        {
            img        = limg.clone();
            pt         = lpt;
            score      = lscore;
            box        = lbox;
            index      = lindex;
            row        = lrow;
            rotateRect = lrotateRect;
            invertMat  = linvertMat;
        }
    };

    struct lc_info
    {
        double a1  = 0;
        double b11 = 0;
        double b12 = 0;
        double b13 = 0;
        double c11 = 0;
        double c12 = 0;
        double d11 = 0;
        double d12 = 0;
        double e1  = 0;
        double p1  = 0;
        double f1  = 0;

        double a2  = 0;
        double b21 = 0;
        double b22 = 0;
        double b23 = 0;
        double c21 = 0;
        double c22 = 0;
        double d21 = 0;
        double d22 = 0;
        double e2  = 0;
        double p2  = 0;
        double f2  = 0;

        int               index;
        int               row;
        bool              classify_status = true;
        std::vector<bool> classify_statusList;
        bool segment_status = true;
        cv::Point2d         lt;
        nlohmann::json rotateRect;
        cv::Mat             img;
        cv::Mat             invertMat;
    };

    std::vector<CropSt> crop_img_vec_;   // 裁剪的图片的数组
    int                 batch_num_ = 1;

    // 模板匹配参数
    int                       num_levels_    = 3;
    int                       angle_min_     = -10;
    int                       angle_max_     = 10;
    double                    min_score_     = 0.5;
    int                       contrast_      = 30;
    std::string               path_          = "";
    std::string               yml_           = "";
    int                       num_           = 100;
    double                    scale_min_     = 0.8;
    double                    scale_max_     = 1.2;
    bool                      sort_by_score_ = false;
    double                    max_overlap_   = 0.5;
    double                    strength_      = 0.5;
    int                       center_count_  = 4;   // 中间的弹片有几个
    float                     bw;                   // 模板图宽高
    float                     bh;
    bool                      enable_affineImg_;
    Tival::ShapeBasedMatching sbm_;
    // 检测区域
    int detect_left_x_ = 0;
    int detect_left_y_ = 0;
    int detect_width_  = 9344;
    int detect_height_ = 7000;

    // 分类模型
    bool        enable_saveSample_ = false;
    std::string imgName_           = "";
    std::string sampleSavePath_    = "";

    bool        enable_saveNGImg_ = false;
    std::string NGSavePath_       = "";

    std::string maskTempImgPath_ = "D:/work/0_HF/AI/template/lc_crop.jpg";
    cv::Mat     maskTempImg_;
    double      topClsConf_   = 0.5;
    double      lrClsConf_    = 0.5;
    double      inClsConf_    = 0.5;
    double      segthreshold_ = 0.5;
    // 是否分类开关
    bool enable_topClassify_;
    bool enable_lrClassify_;
    bool enable_inClassify_;
    // 是否测量开关
    bool enable_topMeasure_;
    bool enable_lrMeasure_;
    bool enable_inMeasure_;

    json rectLrInTop_0_;
    json rectLrInTop_1_;

    // 产品信息参数
    int product_rows_;
    int product_cols_;

    // 过滤mask
    float iou_;
    float areaRatio_;

    // 各模块相对模板左上角的位置：顺序为top、lr、in

    std::vector<cv::Rect> product_gray_0 = {
        cv::Rect(226, 22, 46, 35), cv::Rect(283, 22, 46, 35), cv::Rect(26, 104, 45, 71), cv::Rect(484, 104, 45, 71), cv::Rect(120, 109, 98, 47), cv::Rect(333, 109, 98, 47)};

    std::vector<cv::Rect> product_gray_1 = {
        cv::Rect(209, 22, 46, 35), cv::Rect(266, 22, 46, 35), cv::Rect(26, 104, 45, 71), cv::Rect(484, 104, 45, 71), cv::Rect(120, 109, 98, 47), cv::Rect(333, 109, 98, 47)};

    std::vector<cv::Rect> product_blue_0 = {
        cv::Rect(229, 25, 52, 32), cv::Rect(381, 25, 52, 32), cv::Rect(21, 98, 44, 66), cv::Rect(456, 98, 44, 66), cv::Rect(114, 106, 87, 35), cv::Rect(317, 106, 87, 35)};

    std::vector<cv::Rect> product_blue_1 = {
        cv::Rect(69, 25, 52, 32), cv::Rect(198, 25, 52, 32), cv::Rect(21, 97, 42, 70), cv::Rect(455, 97, 42, 70), cv::Rect(112, 105, 89, 37), cv::Rect(315, 105, 89, 37)};

    std::vector<cv::Rect> crop_rect_0, crop_rect_1;


    /**
     * @brief: 获取参数
     * @return
     * @note :
     **/
    void getParam(InferTaskPtr task);

    /**
     * @brief: 裁剪图片
     * @return
     * @note :
     **/
    void CropTaskImg(const cv::Mat& input_img);

    /**
     * @brief: 推理图片
     * @return
     * @note :
     **/
    void Infer(InferTaskPtr task, AlgoResultPtr algo_result);

    bool TopComputer(std::vector<std::vector<cv::Point>>& pt, cv::Mat img, lc_info& lc);

    bool SideComputer(std::vector<std::vector<cv::Point>> pt, cv::Mat img, lc_info& lc);

    bool CenterComputer(std::vector<std::vector<cv::Point>> pt, cv::Mat img, lc_info& lc);

    bool FinalJudge(std::vector<lc_info> lc_vec, AlgoResultPtr algo_result);

    std::vector<std::pair<bool, std::vector<bool>>> CheckShapeByClassify(std::vector<cv::Mat>& tmpImgList, std::vector<int>& rowList, std::vector<int>& indexList);

    cv::Mat PerspectTransform(const cv::Mat& image, std::vector<cv::Point2f>& srcPoints, cv::Rect& box, cv::Mat& invertMat);

    float ComputeIoU(const cv::Rect& rect1, const cv::Rect& rect2);

    std::vector<std::vector<cv::Point>> FilterMask(std::vector<std::vector<cv::Point>>& mask, lc_info& lc,  int pos);

    void InvertMask(std::vector<cv::Point>& mask, cv::Mat& invertMat);

    /*
    根据mask 模板的取值，不同mask模板有不同的值
    顶部弹片，高度基准26像素的高度

    左侧弹片的 mask 左 24  右64    下164
    右侧弹片的 mask 左 458 右 497  下164

    中间弹片 左   mask  左116  右 201  上109 下 143
    中间弹片 右   mask  左318  右 404  上109 下 143
    */
    int t_y = 26;

    int ll_x1 = 24;
    int ll_x2 = 64;
    int ll_y1 = 26;
    int ll_y2 = 164;

    int rr_x1 = 458;
    int rr_x2 = 497;
    int rr_y1 = 26;
    int rr_y2 = 164;

    int lc_x1 = 116;
    int lc_x2 = 201;
    int lc_y1 = 109;
    int lc_y2 = 143;

    int rc_x1 = 318;
    int rc_x2 = 404;
    int rc_y1 = 109;
    int rc_y2 = 143;

    double pix_value_ = 5.26;

    // 检测规格
    double a_        = 0.1;
    double b1_       = 0.1;
    double b2_       = 0.1;
    double b3_       = 0.1;
    double c1_       = 0.1;
    double c2_       = 0.1;
    double d1_       = 0.1;
    double d2_       = 0.1;
    double e_        = 0.1;
    double p_        = 0.1;
    double f_        = 0.1;
    double error_a_  = 0.1;
    double error_b1_ = 0.1;
    double error_b2_ = 0.1;
    double error_b3_ = 0.1;
    double error_c1_ = 0.1;
    double error_c2_ = 0.1;
    double error_d1_ = 0.1;
    double error_d2_ = 0.1;
    double error_e_  = 0.1;
    double error_p_  = 0.1;
    double error_f_  = 0.1;
    // 调整公差参数
    json error_index_;
    json error_second_;
    std::vector<int> error_index_list;
    std::vector<double> error_second_list;
};