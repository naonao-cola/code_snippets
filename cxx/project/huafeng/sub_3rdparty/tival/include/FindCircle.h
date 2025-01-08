#pragma once
#include "ResultBase.h"
#include "Geometry.h"
#include "CommonDefine.h"
#include "JsonHelper.h"

namespace Tival
{
    class ExportAPI FindCircleResult : public ResutBase
    {
    public:
        // 圆心、半径、分数
        std::vector<cv::Point2f> center;
        std::vector<double> radius;
        std::vector<double> score;
        
        std::vector<cv::Point2f> unused_points;  // 忽略测量点
        std::vector<cv::Point2f> used_points;    // 拟合点
        std::vector<std::vector<cv::Point2f>> measures;  // 测量框
  
        virtual json ToJson() const;
        TLine ToTLine();
    };

    /**
     * 通过布置多个卡尺找出边缘点，基于找出的边缘点拟合直线
     * @params：
     * - Num: 搜索实例数(def: 1)
     * - CaliperNum: 卡尺数量 (def:20)
     * - CaliperLength: 卡尺长度（def:20）
     * - CaliperWidth: 卡尺宽度（def:5）
     * - Transition: 极性（def: 'all' ['positive', 'negative']）
     * - Sigma: 高斯平滑Sigma(def: 1)
     * - Contrast: 边缘对比度阈值：(deff: 30)
     * - IgnoreDistThresh：忽略点距离阈值（def: 3）
     * - MeasureSelect: 找出多ge时的选择（def: first, ["", "first", "last"]）
     * - MinScore: 最小得分阈值（def: 0.7）
     * - StartAngle: 圆弧起点角度（def: 0）
     * - EndAngle: 圆弧终点角度（def: 360）
    */
    class ExportAPI FindCircle
    {
    public:
        FindCircle(){};
        virtual ~FindCircle(){};

        static FindCircleResult Run(void* timage, const TCircle& refCircle, const json& params);
        static FindCircleResult Run(const cv::Mat& image, const TCircle& refCircle, const json& params);
    };
}


