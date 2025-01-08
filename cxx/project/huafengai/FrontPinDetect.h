
#pragma once
#include "../../modules/tv_algo_base/src/framework/BaseAlgo.h"
//#include <filesystem>

#include "nlohmann/json.hpp"
#include "algo_tool.h"
using json = nlohmann::json;
#include "svm_train.h"

struct BlobInfo
{
    std::vector<cv::Point> contour;
    double area;        // blob 面积
    double mean_gv;     // blob GV均值
    double sum_gv;      // blob GV累加值
    int total_num;      // blob 像素总数
    cv::Point center;   // pin中心点
    bool found;         // 是否已经找到pin针

    BlobInfo(std::vector<cv::Point> cont, double area, double mean_gv, double sum_gv, int total_num, cv::Point center) {
        this->contour = cont;
        this->area = area;
        this->center = center;
        this->mean_gv = mean_gv;
        this->sum_gv = sum_gv;
        this->total_num = total_num;
        this->found = false;
    }
    BlobInfo(std::vector<cv::Point> cont, double area, cv::Point center)
    {
        this->contour = cont;
        this->area = area;
        this->center = center;
    }
};

using BlobInfoPtr = std::shared_ptr<BlobInfo>;

struct PinInfo
{
    cv::Rect bbox;
    cv::Point pin_center;
    cv::Point2f pin_cent_local;     //

    std::vector<cv::Point> cont;
    double mean_gv = 0;
    double area_total = 0;
    double area_init = 0;
    bool found = false;

    int row_idx;
    int col_idx;
    double local_x;
    double  local_y;
    int     index;
    bool    classify_status = true;
    cv::Mat cropPinImg;
};

using PinInfoPtr = std::shared_ptr<PinInfo>;

struct BinResult
{
    double area_sum;
    double threshold;
    double mean_gv;
    std::vector<BlobInfoPtr> blob_list;
    BinResult(double area_sum, double mean_gv, double threshold, std::vector<BlobInfoPtr> blob_list)
    {
        this->area_sum = area_sum;
        this->mean_gv = mean_gv;
        this->threshold = threshold;
        this->blob_list = blob_list;
    }
};

using BinResultPtr = std::shared_ptr<BinResult>;

// 产品型号枚举
enum class productNames
{
    rxwm_notline = 0,
    rxwm_small,
    rxzg_151,
    rxzg_131K,
    rxzg_130A,
    sbzg_1851408151
};

class FrontPinDetect : public BaseAlgo
{
public:
    FrontPinDetect();
    ~FrontPinDetect();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);

private:
    void FirstPassLocate(const cv::Mat& redImage, const cv::Mat& whiteImage, const json& params, cv::Mat& bin_img, std::vector<PinInfoPtr>& pin_infos, std::string image_name, bool draw_contours = false) noexcept;
    void FindPinByImage1(cv::Mat& image1, cv::Mat bin_img, const json& params, std::vector<PinInfoPtr>& pin_infos, cv::Mat& img_draw, int roi_lt_x, int roi_lt_y);
    void FindPinByImage2(cv::Mat& image2, cv::Mat& bin_img, const json& params, std::vector<PinInfoPtr>& pin_infos, cv::Mat& img_draw);


    BinResultPtr LocalThreshold(const cv::Mat& block_img, int threshold, int min_area);
    void PrintBinResult(const std::string& msg, BinResultPtr bin_result, bool detail = false);
    BinResultPtr ChooseBinResult(BinResultPtr last_bin_result, BinResultPtr cur_bin_result, const json& params, const cv::Mat& img_draw);
    int SumBlobArea(std::vector<BlobInfoPtr>& blob_list, std::vector<int>& indices);
    std::vector<std::set<int>> GetBlobPairs(std::vector<BlobInfoPtr>& blob_list, int area_sum, double area_ratio = 0.6);
    bool CalcPinCoordByLocalBlobs(cv::Point& pin_center, BinResultPtr bin_result, const json& params, int offset_x, int offset_y, cv::Mat& img_draw);

    void SumGV(cv::Mat image, cv::Mat mask, double& sum_gv, int& num);
    cv::Rect GetROI(cv::Mat& image, cv::Mat& mask, double& angle, const json& params);
    cv::Mat GetROIImage(const cv::Mat& image, const cv::Mat& roi_mask);
    cv::Mat Cont_delete(const cv::Mat& img2_gray);
    void FilterNonPinBlobs(const cv::Mat& img2_gray, cv::Mat& bin_img, std::vector<std::vector<cv::Point >>& contours, const json& params, std::string image_name);
    //轮廓完整度
    double calculateCompleteness(const std::vector<cv::Point> contour, const cv::Rect boundingRect);
    //LI自适应阈值
    int liThreshold(cv::Mat img);
    // 坐标计算
    // 根据针尖在图像上的坐标，计算出以左上角第一个pin针为原点，第一排方向（拟合直线）为x方向的坐标系，得到图像坐标系和pin坐标系的转换矩阵
    cv::Mat GetPinCSTransMat(const std::vector<PinInfoPtr>& pin_infos, const json& params, double roiAngle, cv::Mat& img_draw);
    // 将param中标准坐标
    json CalcPinResults(std::vector<PinInfoPtr>& pin_infos, const cv::Mat& M, const cv::Mat img1_gray, const cv::Mat img2_gray, const cv::Mat bin_img2, const json& params, double& adj_x, double& adj_y, int roi_lt_x, int roi_lt_y, float& differ_std, float& differ_measure);
    json CalcStdLines(const cv::Mat& M, const json& params, double adj_x, double adj_y, float differ_std);

    //json CalcPinResults(std::vector<PinInfoPtr>& pin_infos, const cv::Mat& M, const json& params, double& adj_x, double& adj_y);
    //json CalcStdLines(const cv::Mat& M, const json& params, double adj_x, double adj_y);


    int GetNearestPointIdx(std::vector<cv::Point2f> points, cv::Point2f refPoint, double& minDist);

    json StdPoint2Box(const cv::Mat& M, const cv::Point2f& pt, double tolerance_px);  // 基于标准点坐标计算标准框在原图上的位置（因为有旋转，多边形四个点）
    cv::Point2f TransPoint(const cv::Mat& M, const cv::Point2f& point);
    bool IsEmptyPos(int x, int y, const json& param);


    void DrawOrgResults(cv::Mat& image, std::vector<PinInfoPtr>& pin_infos);
    void DrawResults(cv::Mat& image, const json& lines, const json& results);


    cv::Mat vector_angle_to_M(double x1, double y1, double d1, double x2, double y2, double d2);
    cv::Mat cvMat6_to_cvMat9(const cv::Mat& mtx6);
    cv::Mat d6_to_cvMat(double d0, double d1, double d2, double d3, double d4, double d5);


    double DistPP(const cv::Point2f& a, const cv::Point2f& b);

    // 像素值px转物理距离mm
    template<typename T>
    inline T Px2Mm(T px, double ppum) { return px * ppum / 1000.0; }
    // 物理距离mm转像素值px
    template<typename T>
    inline T Mm2Px(T mm, double ppum) { return mm * 1000.0 / ppum; }

    float calculateSlope(const cv::Point2f& p1, const cv::Point2f& p2);
    double reget_angle(std::vector<cv::Point2f> pts, int min_points_in_line);

    bool check_folder_state(std::string folderPath);
    void RotateImage(cv::Mat& image, int mode);

    cv::Point2f FrontPinDetect::Refind_pin_fromImg2(cv::Mat& binary_roi, std::vector<std::vector<cv::Point>>& contorus_roi, const json& params);
    cv::Point2f FrontPinDetect::Refind_pin_fromImg1(cv::Mat& white_roi, const json& params);
    bool CheckBlobByShapeOld(cv::Mat& sub_tmp_bin, cv::Mat& sub_whiteImg, cv::Mat& sub_img_bin, std::vector<cv::Point>& sub_con, cv::Mat& labels, int num_labels, const json& params);
    bool CheckBlobByShape(cv::Mat& sub_tmp_bin, cv::Mat& sub_whiteImg, cv::Mat& sub_SvmImg, std::vector<cv::Point>& sub_con, const json& params);
    bool CheckBlobByClassify(std::vector<cv::Mat>& imgList, std::vector<int>& indexList, std::vector<PinInfoPtr>& pin_infos, double confThr);
    cv::Mat AlignTransform(cv::Mat& image1, cv::Mat& pin_img, std::vector<PinInfoPtr>& pin_infos, const json& params);

    cv::Mat alignMatrix;

    double _offset_value_x_ = 0;
    double _offset_value_y_ = 0;

    double _scale_value = 1;
    void reget_ppum(const int& index, double& ppum, const json& params, const bool& enable_offset, int x, int y);

    nao::svm::SvmTrain svm_obj;
    bool isInit = false;
    std::map<std::string, productNames> selectModel;

    int detect_lt_x = 0;
    int detect_lt_y = 0;

private:
    DCLEAR_ALGO_GROUP_REGISTER(FrontPinDetect)

};

