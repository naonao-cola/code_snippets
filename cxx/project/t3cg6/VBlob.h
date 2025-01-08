/**
 * @FilePath     : /t3cg6/src/project/VBlob.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-07-31 11:24:32
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-03 19:00:39
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/
#ifndef NAONAO_VBLOB_H
#define NAONAO_VBLOB_H

#include "VBlobDefine.h"

class VBlob
{
public:
    VBlob();
    ~VBlob();
    void Release();

    // 运行Blob算法
    bool DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Mat GrayBuffer = cv::Mat(), int nMaxDefectCount = 99999);
    bool DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Rect rectROI, cv::Mat GrayBuffer = cv::Mat(), int nMaxDefectCount = 99999);
    // 坐标校正
    void                       CoordApply(cv::Rect rectROI, int nTotalLabel);
    bool                       DoFeatureBasic_8bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel);
    bool                       DoFeatureBasic_16bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel);
    std::vector<tBLOB_FEATURE> DoDefectBlobSingleJudgment(const std::vector<STRU_DEFECT_ITEM>& EngineerBlockDefectJudge);
    bool                       DoFiltering(tBLOB_FEATURE& tBlobResult, int nBlobFilter, int nSign, double dValue);
    bool                       Compare(double dFeatureValue, int nSign, double dValue);
    int                        getSignFromSymbol(const std::string& symbol);
    int                        getIndex(const std::string& feature_str);

protected:
    bool                       bComplete_;    // 确认Blob是否已完成。
    std::vector<tBLOB_FEATURE> BlobResult_;   // Blob结果列表
};   // class VBlob


#endif   // NAONAO_VBLOB_H