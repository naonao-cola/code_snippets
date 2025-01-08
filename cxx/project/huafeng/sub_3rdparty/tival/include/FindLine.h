#pragma once
#include "ResultBase.h"
#include "Geometry.h"
#include "CommonDefine.h"
#include "JsonHelper.h"

namespace Tival
{
    class ExportAPI FindLineResult : public ResutBase
    {
    public:
        std::vector<cv::Point2f> start_point; // 起点
        std::vector<cv::Point2f> end_point;   // 终点
        std::vector<cv::Point2f> mid_point;   // 中点
        std::vector<double> line_length;           // 线段长度
        std::vector<double> line_angle;            // 角度 (-180°~180°)
        std::vector<cv::Point2f> measure_points; // 测量点
        std::vector<cv::Point2f> used_points;    // 拟合点
        std::vector<std::vector<cv::Point2f>> measures;  // 测量框
  
        virtual json ToJson() const;
        TLine ToTLine();
    };

    /**
     * 通过布置多个卡尺找出边缘点，基于找出的边缘点拟合直线
     * @params：
     * - Num: 搜索实例数(def: 1)
     * - CaliperNum: 卡尺数量 (def:10)
     * - CaliperLength: 卡尺长度（def:20）
     * - CaliperWidth: 卡尺宽度（def:5）
     * - Transition: 极性（def: "all" ["positive", "negative"]）
     * - Sigma: 高斯平滑Sigma(def: 1)
     * - Contrast: 边缘对比度阈值：(deff: 30)
     * - MeasureSelect: 卡尺上测量出多个点时的选择（def: "first", ["all", "first", "last"]）
     * - MinScore: 最小分数（def:0.5）
    */
    class ExportAPI FindLine
    {
    public:
        FindLine() {};
        virtual ~FindLine() {};
        
        static FindLineResult Run(const cv::Mat& image, const TPoint& start, const TPoint& end, const json& params);
        static FindLineResult Run(const cv::Mat& image, const TLine& line, const json& params);
        static FindLineResult Run(void* timage, const TPoint& start, const TPoint& end, const json& params);
        static FindLineResult Run(void* timage, const TLine& line, const json& params);
    };
}


