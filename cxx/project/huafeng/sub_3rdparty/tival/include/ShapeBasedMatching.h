#pragma once
#include <string>
#include <sstream>
#include "ResultBase.h"
#include "CommonDefine.h"

namespace Tival
{
    class ExportAPI SbmResults : public ResutBase
    {
    public:
        void Translate(double ratio, double offsetX, double offsetY)
        {
            for (int i = 0; i < num_instances; i++)
            {
                center[i].x = center[i].x / ratio + offsetX;
                center[i].y = center[i].y / ratio + offsetY;
            }
        }

        json ToJson() const;
        std::vector<cv::Point2f> center;
        std::vector<double> angle;  //角度
        std::vector<double> score;
        std::vector<double> scale;
    };

    using FeaturePoints = std::vector<std::vector<cv::Point2f>>;


    
    class ExportAPI ShapeBasedMatching
    {
    public:
        ShapeBasedMatching();
        ~ShapeBasedMatching();

        /** 加载模板 */
        int Load(const std::string& modelFilePath);
        /** 保存模板 */
        int Save(const std::string& filePath);

        /**
         * 创建模板
         * @params：
         * - NumLevels: 金字塔层数 (def: 3)
         * - AngleMin: 最小角度 (def:-10)
         * - AngleMax: 最大角度（def:10）
         * - ScaleMin: 最小缩放比例（def: 0.95）
         * - ScaleMax: 最大缩放比例 (def: 1.05)
         * - MinScore: 最小得分阈值（def: 0.5）
         * - Contrast: 特征点对比度阈值
         * - MinContrast: 最小对比度
        */
        SbmResults CreateByImage(cv::Mat& image, const json& params);

        // 用屏蔽区域外的特征创建模板
        SbmResults CreateByImageWithMask(cv::Mat& image, cv::Mat& image_mask, const json& params);

        // 通过特征轮廓点创建模板
        SbmResults CreateByFeaturePoints(cv::Mat& image, const FeaturePoints& features, const json& params);

        /** 提取轮廓点
         * @params：
         * - ContrastLow: 最小对比度，梯度小于该值被忽略
         * - ContrastHigh: 边缘点通过对比度阈值，大于该阈值的边缘点直接被选取，ContrastMin~ContrastApprove之间的点按轮廓联通规则选取
         * - DistInterval: 边缘点输出间隔距离, 用于减少输出边缘点的数量
         */
        FeaturePoints GetFeaturePoints(cv::Mat& image, const json& params);

        // 剔除屏蔽区域mask内的特征点
        FeaturePoints FilterFeaturePointsByMask(const FeaturePoints& features, cv::Mat& mask);

        /**
         * 形状模板匹配搜索
         * @params：
         * - Num: 搜索个数（def: 1）
         * - AngleMin: 最小角度 (def:-10)
         * - AngleMax: 最大角度（def:10）
         * - ScaleMin: 最小缩放比例（def: 0.95）
         * - ScaleMax: 最大缩放比例 (def: 1.05)
         * - MinScore: 最小得分阈值（def: 0.5）
         * - Contrast: 边缘对比度阈值 (deff: 30)
         * - SortByScore: 按分数排序（def: false）
         * - MaxOverlap: 最大重叠度（def: 0.5）
         * - Strength: 搜索强度（def: 0.5）
        */
        SbmResults Find(const cv::Mat& image, const json& params);

        /** 释放模板资源 */
        void Destroy();
        /** 判断模板是否已加载 */
        bool IsLoaded() { return mModelHandlePtr != nullptr; }

        // 内部方法
        SbmResults CreateByImage(void* image, const json& params);
        SbmResults CreateByImageWithMask(void* image, void* image_mask, const json& params);
        SbmResults CreateByFeaturePoints(void* image, const FeaturePoints& features, const json& params);
        SbmResults Find(void* image, const json& params);
        FeaturePoints GetFeaturePoints(void* image, const json& params);
        FeaturePoints FilterFeaturePointsByMask(const FeaturePoints& features, void* image);

    private:
        SbmResults CreateByImage_Internal(void* template_img, const json& params, void* search_img=nullptr);
        IntPtr mModelHandlePtr;
    };
} // namespace TivalBase
