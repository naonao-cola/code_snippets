/**
 * @FilePath     : /t3cg6/src/project/VBlob.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-07-31 14:02:49
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-03 19:03:00
 * @Copyright (c) 2024 by G, All Rights Reserved.
 **/

#include "VBlob.h"
#include <cmath>

VBlob::VBlob()
{
    bComplete_ = false;
    std::vector<tBLOB_FEATURE>().swap(BlobResult_);
}

VBlob::~VBlob()
{
    Release();
}
void VBlob::Release()
{
    bComplete_ = false;
    // 初始化向量
    if (BlobResult_.size() != 0) {
        for (int i = 0; i < BlobResult_.size(); i++) {
            std::vector<cv::Point>().swap(BlobResult_[i].ptIndexs);
            std::vector<cv::Point>().swap(BlobResult_[i].ptContours);
        }
        std::vector<tBLOB_FEATURE>().swap(BlobResult_);
    }
}

bool VBlob::DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Mat GrayBuffer, int nMaxDefectCount)
{
    Release();
    // 如果没有画面,则返回
    if (ThresholdBuffer.empty()) {
        return false;
    }
    // 如果不是1频道
    if (ThresholdBuffer.channels() != 1) {
        return false;
    }
    if (GrayBuffer.channels() != 1) {
        cv::cvtColor(GrayBuffer, GrayBuffer, cv::COLOR_BGR2GRAY);
    }
    // 如果Gray画面不存在X&1频道
    bool bGrayEmpty = false;
    if (GrayBuffer.empty() || GrayBuffer.channels() != 1) {
        GrayBuffer = ThresholdBuffer.clone();
        bGrayEmpty = true;
    }

    cv::Mat matLabel;
    cv::Mat matStats;
    cv::Mat matCentroid;
    matLabel              = cv::Mat(ThresholdBuffer.size(), CV_32SC1);
    long long nTotalLabel = 0;

    if (ThresholdBuffer.type() == CV_8U) {
        nTotalLabel = cv::connectedComponentsWithStats(ThresholdBuffer, matLabel, matStats, matCentroid, 8, CV_32S, cv::CCL_GRANA) - 1;
    }
    else {
        cv::Mat matSrc8bit = cv::Mat(ThresholdBuffer.size(), CV_8UC1);
        ThresholdBuffer.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);
        nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S, cv::CCL_GRANA) - 1;
        matSrc8bit.release();
    }

    if (nTotalLabel < 0) {
        if (bGrayEmpty) {
            GrayBuffer.release();
        }
        if (!matLabel.empty()) {
            matLabel.release();
        }
        if (!matStats.empty()) {
            matStats.release();
        }
        if (!matCentroid.empty()) {
            matCentroid.release();
        }
        return false;
    }
    // 异常处理
    if (nTotalLabel >= nMaxDefectCount) {
        nTotalLabel = nMaxDefectCount - 1;
    }

    // 默认Feature计算8bit
    if (GrayBuffer.type() == CV_8UC1) {
        DoFeatureBasic_8bit(matLabel, matStats, matCentroid, GrayBuffer, static_cast<int>(nTotalLabel));
    }
    else {
        DoFeatureBasic_16bit(matLabel, matStats, matCentroid, GrayBuffer, static_cast<int>(nTotalLabel));
    }

    if (bGrayEmpty) {
        GrayBuffer.release();
    }
    if (!matLabel.empty()) {
        matLabel.release();
    }
    if (!matStats.empty()) {
        matStats.release();
    }
    if (!matCentroid.empty()) {
        matCentroid.release();
    }
    bComplete_ = true;
    return true;
}

bool VBlob::DoFeatureBasic_8bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel)
{
    if (nTotalLabel <= 0) {
        return true;
    }
    float fVal = 4.f * CV_PI;
    BlobResult_.resize(nTotalLabel);

#ifdef _DEBUG
#else
#    pragma omp parallel for
#endif
    for (int idx = 1; idx <= nTotalLabel; idx++) {
        int nBlobNum                            = idx - 1;
        BlobResult_.at(nBlobNum).rectBox.x      = matStats.at<int>(idx, cv::CC_STAT_LEFT);
        BlobResult_.at(nBlobNum).rectBox.y      = matStats.at<int>(idx, cv::CC_STAT_TOP);
        BlobResult_.at(nBlobNum).rectBox.width  = matStats.at<int>(idx, cv::CC_STAT_WIDTH);
        BlobResult_.at(nBlobNum).rectBox.height = matStats.at<int>(idx, cv::CC_STAT_HEIGHT);


        int cx, cy;
        cx = matStats.at<int>(idx, cv::CC_STAT_LEFT) + (int)(matStats.at<int>(idx, cv::CC_STAT_WIDTH) / 2);
        cy = matStats.at<int>(idx, cv::CC_STAT_TOP) + (int)(matStats.at<int>(idx, cv::CC_STAT_HEIGHT) / 2);

        // int    nDistanceFromEdge = 0;
        // double distanceA = -1, distanceB = -1;
        //// matBKBuf.at<int>(CC_STAT_LEFT);
        // float lineA_00 = matBKBuf.at<float>(0, 0);
        // float lineA_01 = matBKBuf.at<float>(0, 1);
        // float lineB_10 = matBKBuf.at<float>(1, 0);
        // float lineB_11 = matBKBuf.at<float>(1, 1);


        //// distanceA = abs(cx * lineA_00 - cy + lineA_01) / sqrt(lineA_00 * lineA_00 + 1.);
        //// distanceB = abs(cx * lineB_10 - cy + lineB_11) / sqrt(lineB_10 * lineB_10 + 1.);
        // distanceA = calculateMinDistanceToLines(m_BlobResult.at(nBlobNum).rectBox, lineA_00, lineA_01);
        // distanceB = calculateMinDistanceToLines(m_BlobResult.at(nBlobNum).rectBox, lineB_10, lineB_11);
        // if (distanceA > distanceB && distanceB > 0 && distanceA > 0)
        //     nDistanceFromEdge = distanceB;
        // else if (distanceA < distanceB && distanceB > 0 && distanceA > 0)
        //     nDistanceFromEdge = distanceA;


        // m_BlobResult.at(nBlobNum).nDistanceFromEdge = nDistanceFromEdge;

        // 对象周围(用于背景GV)
        int nOffSet = 0;

        int nSX = BlobResult_.at(nBlobNum).rectBox.x - nOffSet;
        int nSY = BlobResult_.at(nBlobNum).rectBox.y - nOffSet;
        int nEX = BlobResult_.at(nBlobNum).rectBox.x + BlobResult_.at(nBlobNum).rectBox.width + nOffSet + nOffSet;
        int nEY = BlobResult_.at(nBlobNum).rectBox.y + BlobResult_.at(nBlobNum).rectBox.height + nOffSet + nOffSet;

        if (nSX < 0) {
            nSX = 0;
        }
        if (nSY < 0) {
            nSY = 0;
        }
        if (nSX >= GrayBuffer.cols) {
            continue;
        }
        if (nSY >= GrayBuffer.rows) {
            continue;
        }
        if (nEX >= GrayBuffer.cols) {
            nEX = GrayBuffer.cols - 1;
        }
        if (nEY >= GrayBuffer.rows) {
            nEY = GrayBuffer.rows - 1;
        }

        cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);

        BlobResult_.at(nBlobNum).FeaRectROI = rectTemp;

        long long nCount_in  = 0;
        long long nCount_out = 0;
        long long nSum_in    = 0;   // defect区域
        long long nSum_out   = 0;   // defect排除区域

        cv::Mat matTmp_src   = GrayBuffer(rectTemp);   // 原始ROI
        cv::Mat matTmp_label = matLabel(rectTemp);     // Label的ROI
        cv::Mat matTemp      = cv::Mat(rectTemp.height, rectTemp.width, CV_8UC1);

        for (int y = 0; y < rectTemp.height; y++) {
            int*   ptrLabel = reinterpret_cast<int*>(matTmp_label.ptr(y));
            uchar* ptrGray  = matTmp_src.ptr(y);
            uchar* ptrTemp  = matTemp.ptr(y);
            for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++) {
                // 对象
                if (*ptrLabel == idx) {
                    nSum_in += *ptrGray;
                    nCount_in++;
                    // 在标签向量中存储像素坐标
                    BlobResult_.at(nBlobNum).ptIndexs.emplace_back(nSX + x, nSY + y);
                    *ptrTemp = static_cast<uchar>(255);
                    BlobResult_.at(nBlobNum).nHist[*ptrGray]++;
                }
                // 其他情况下背景
                else {
                    // 如果标签编号为0...
                    // 因为可以引用其他对象......添加条件
                    if (*ptrLabel == 0) {
                        nSum_out += *ptrGray;
                        nCount_out++;
                    }
                }
            }
        }

        // 亮度累计值
        BlobResult_.at(nBlobNum).nSumGV = nSum_in;
        // 对象面积
        BlobResult_.at(nBlobNum).nArea = nCount_in;   // matStats.at<int>(idx, CC_STAT_AREA);
        // Box Area
        BlobResult_.at(nBlobNum).nBoxArea = BlobResult_.at(nBlobNum).rectBox.width * BlobResult_.at(nBlobNum).rectBox.height;
        // Bounding Box面积比率/对象面积(Rectangulaty(=Extent))
        BlobResult_.at(nBlobNum).fBoxRatio = BlobResult_.at(nBlobNum).nArea / static_cast<float>(BlobResult_.at(nBlobNum).nBoxArea);
        // 拯救Elongation
        BlobResult_.at(nBlobNum).fElongation = BlobResult_.at(nBlobNum).rectBox.width / (float)BlobResult_.at(nBlobNum).rectBox.height;
        // 获取stdDev
        cv::Scalar m;
        cv::Scalar s;
        cv::meanStdDev(matTmp_src, m, s, matTemp);
        BlobResult_.at(nBlobNum).fStdDev = static_cast<float>(s[0]);
        // 拯救Contours
        std::vector<std::vector<cv::Point>> ptContours;
        std::vector<std::vector<cv::Point>>().swap(ptContours);
        cv::findContours(matTemp, ptContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // 获取Perimeter
        if (ptContours.size() != 0) {
            // ROI画面,需要校正
            // 复制轮廓坐标结果
            for (int m = 0; m < ptContours.size(); m++) {
                for (int k = 0; k < ptContours.at(m).size(); k++) {
                    BlobResult_.at(nBlobNum).ptContours.emplace_back(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY);
                }
            }
        }
        else {
            // 因为是原始坐标,所以不用校正。
            // 复制轮廓坐标结果
            BlobResult_.at(nBlobNum).ptContours.resize(static_cast<int>(BlobResult_.at(nBlobNum).ptIndexs.size()));
            std::copy(BlobResult_.at(nBlobNum).ptIndexs.begin(), BlobResult_.at(nBlobNum).ptIndexs.end(), BlobResult_.at(nBlobNum).ptContours.begin());
        }
        BlobResult_.at(nBlobNum).fPerimeter = static_cast<float>(cv::arcLength(BlobResult_.at(nBlobNum).ptContours, true));
        std::vector<std::vector<cv::Point>>().swap(ptContours);
        // 获取Roundness
        BlobResult_.at(nBlobNum).fRoundness = (fVal * BlobResult_.at(nBlobNum).nArea) / (BlobResult_.at(nBlobNum).fPerimeter * BlobResult_.at(nBlobNum).fPerimeter);
        // 对象有多接近原样？(周长^2/4*Pi*面积)
        BlobResult_.at(nBlobNum).fCompactness = (BlobResult_.at(nBlobNum).fPerimeter * BlobResult_.at(nBlobNum).fPerimeter) / (fVal * float(BlobResult_.at(nBlobNum).nArea));
        // 获取Defect GV
        BlobResult_.at(nBlobNum).fMeanGV = nSum_in / static_cast<float>(nCount_in);
        // 获取背景GV
        BlobResult_.at(nBlobNum).fBKGV = nSum_out / static_cast<float>(nCount_out);
        // 求GV差值(背景-对象)
        BlobResult_.at(nBlobNum).fDiffGV = BlobResult_.at(nBlobNum).fBKGV - BlobResult_.at(nBlobNum).fMeanGV;

        // min,获取max GV
        double valMin = NAN;
        double valMax = NAN;

        double ta, tb;
        cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
        cv::minMaxIdx(matTmp_src, &ta, &tb);
        BlobResult_.at(nBlobNum).nMinGV = static_cast<long>(valMin);
        BlobResult_.at(nBlobNum).nMaxGV = static_cast<long>(valMax);
        // 对象最小亮度/对象平均亮度
        BlobResult_.at(nBlobNum).nMinGVRatio = BlobResult_.at(nBlobNum).nMinGV / BlobResult_.at(nBlobNum).fBKGV;
        // 对象最大亮度/对象平均亮度
        BlobResult_.at(nBlobNum).nMaxGVRatio = BlobResult_.at(nBlobNum).nMaxGV / BlobResult_.at(nBlobNum).fBKGV;
        // 背景亮度/对象平均亮度
        BlobResult_.at(nBlobNum).fDiffGVRatio = BlobResult_.at(nBlobNum).fMeanGV / BlobResult_.at(nBlobNum).fBKGV;
        // 获取Center Point
        BlobResult_.at(nBlobNum).ptCenter.x = static_cast<int>(matCentroid.at<double>(idx, 0));
        BlobResult_.at(nBlobNum).ptCenter.y = static_cast<int>(matCentroid.at<double>(idx, 1));
        // 拯救SEMU
        if (BlobResult_.at(nBlobNum).fDiffGV == 0.0) {
            if (BlobResult_.at(nBlobNum).fBKGV == 0) {
                BlobResult_.at(nBlobNum).fSEMU = 1.0 / (1.97F / (cv::pow(static_cast<float>(BlobResult_.at(nBlobNum).nArea), 0.33F) + 0.72F));
            }
            else {
                BlobResult_.at(nBlobNum).fSEMU = (0.000001 / BlobResult_.at(nBlobNum).fBKGV) / (1.97 / (cv::pow(static_cast<float>(BlobResult_.at(nBlobNum).nArea), 0.33F) + 0.72F));
            }
        }
        else {
            if (BlobResult_.at(nBlobNum).fBKGV == 0) {
                BlobResult_.at(nBlobNum).fSEMU =
                    (fabs(BlobResult_.at(nBlobNum).fMeanGV - BlobResult_.at(nBlobNum).fBKGV) / 0.000001) / (1.97 / (cv::pow((float)BlobResult_.at(nBlobNum).nArea, 0.33F) + 0.72F));
            }
            else {
                BlobResult_.at(nBlobNum).fSEMU = (fabs(BlobResult_.at(nBlobNum).fMeanGV - BlobResult_.at(nBlobNum).fBKGV) / BlobResult_.at(nBlobNum).fBKGV) /
                                                 (1.97 / (cv::pow((float)BlobResult_.at(nBlobNum).nArea, 0.33F) + 0.72F));
            }
        }
        cv::RotatedRect BoundingBox = cv::minAreaRect(BlobResult_.at(nBlobNum).ptIndexs);
        // Box width and length
        BlobResult_.at(nBlobNum).BoxSize = BoundingBox.size;
        // Angle between the horizontal axis
        BlobResult_.at(nBlobNum).fAngle = BoundingBox.angle;
        // Minor Axis & Major Axis
        if (BoundingBox.size.width > BoundingBox.size.height) {
            BlobResult_.at(nBlobNum).fMinorAxis = BoundingBox.size.width;
            BlobResult_.at(nBlobNum).fMajorAxis = BoundingBox.size.height;
        }
        else {
            BlobResult_.at(nBlobNum).fMinorAxis = BoundingBox.size.height;
            BlobResult_.at(nBlobNum).fMajorAxis = BoundingBox.size.width;
        }
        // Feret’s area
        BlobResult_.at(nBlobNum).fMinBoxArea = BlobResult_.at(nBlobNum).fMinorAxis * BlobResult_.at(nBlobNum).fMajorAxis;

        // Axis Ratio
        if (BlobResult_.at(nBlobNum).fMajorAxis > 0) {
            BlobResult_.at(nBlobNum).fAxisRatio = BlobResult_.at(nBlobNum).fMinorAxis / BlobResult_.at(nBlobNum).fMajorAxis;
        }
        else {
            BlobResult_.at(nBlobNum).fAxisRatio = 0.F;
        }
        // Min Bounding Box面积比/对象面积(区域孔隙率)
        BlobResult_.at(nBlobNum).fMinBoxRatio = BlobResult_.at(nBlobNum).fMinBoxArea / static_cast<float>(BlobResult_.at(nBlobNum).nArea);
        // choikwangil
        BlobResult_.at(nBlobNum).fMeanAreaRatio = BlobResult_.at(nBlobNum).fMeanGV / static_cast<float>(BlobResult_.at(nBlobNum).nArea);
        BlobResult_.at(nBlobNum).fDefectMeanGV  = (tb - ta);
        // 取消分配
        matTmp_src.release();
        matTmp_label.release();
        matTemp.release();
    }
    return true;
}

bool VBlob::DoFeatureBasic_16bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel)
{
    // 如果有一个结果
    if (nTotalLabel <= 0) {
        return true;
    }
    float fVal = 4.F * CV_PI;
    BlobResult_.resize(nTotalLabel);

#ifdef _DEBUG
#else
#    pragma omp parallel for
#endif
    for (int idx = 1; idx <= nTotalLabel; idx++) {
        int nBlobNum                            = idx - 1;
        BlobResult_.at(nBlobNum).rectBox.x      = matStats.at<int>(idx, cv::CC_STAT_LEFT);
        BlobResult_.at(nBlobNum).rectBox.y      = matStats.at<int>(idx, cv::CC_STAT_TOP);
        BlobResult_.at(nBlobNum).rectBox.width  = matStats.at<int>(idx, cv::CC_STAT_WIDTH);
        BlobResult_.at(nBlobNum).rectBox.height = matStats.at<int>(idx, cv::CC_STAT_HEIGHT);

        // 对象周围(用于背景GV)
        int nOffSet = 20;
        int nSX     = BlobResult_.at(nBlobNum).rectBox.x - nOffSet;
        int nSY     = BlobResult_.at(nBlobNum).rectBox.y - nOffSet;
        int nEX     = BlobResult_.at(nBlobNum).rectBox.x + BlobResult_.at(nBlobNum).rectBox.width + nOffSet + nOffSet;
        int nEY     = BlobResult_.at(nBlobNum).rectBox.y + BlobResult_.at(nBlobNum).rectBox.height + nOffSet + nOffSet;

        if (nSX < 0) {
            nSX = 0;
        }
        if (nSY < 0) {
            nSY = 0;
        }
        if (nEX >= GrayBuffer.cols) {
            nEX = GrayBuffer.cols - 1;
        }
        if (nEY >= GrayBuffer.rows) {
            nEY = GrayBuffer.rows - 1;
        }

        cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);
        BlobResult_.at(nBlobNum).FeaRectROI = rectTemp;

        long long nCount_in  = 0;
        long long nCount_out = 0;
        long long nSum_in    = 0;   // defect区域
        long long nSum_out   = 0;   // defect排除区域

        cv::Mat matTmp_src   = GrayBuffer(rectTemp);   // 原始ROI
        cv::Mat matTmp_label = matLabel(rectTemp);     // Label的ROI
        cv::Mat matTemp      = cv::Mat(rectTemp.height, rectTemp.width, CV_8UC1);

        for (int y = 0; y < rectTemp.height; y++) {
            int*    ptrLabel = reinterpret_cast<int*>(matTmp_label.ptr(y));
            ushort* ptrGray  = reinterpret_cast<ushort*>(matTmp_src.ptr(y));
            uchar*  ptrTemp  = matTemp.ptr(y);
            for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++) {
                // 对象
                if (*ptrLabel == idx) {
                    nSum_in += *ptrGray;
                    nCount_in++;
                    // 在标签向量中存储像素坐标
                    BlobResult_.at(nBlobNum).ptIndexs.emplace_back(nSX + x, nSY + y);
                    *ptrTemp = static_cast<uchar>(255);
                    BlobResult_.at(nBlobNum).nHist[static_cast<int>(*ptrGray)]++;
                }
                // 其他情况下背景
                else {
                    // 如果标签编号为0...
                    // 因为可以引用其他对象......添加条件
                    if (*ptrLabel == 0) {
                        nSum_out += *ptrGray;
                        nCount_out++;
                    }
                }
            }
        }
        // 亮度累计值
        BlobResult_.at(nBlobNum).nSumGV = nSum_in;
        // 对象面积
        BlobResult_.at(nBlobNum).nArea = nCount_in;   // matStats.at<int>(idx, CC_STAT_AREA);
        // Box Area
        BlobResult_.at(nBlobNum).nBoxArea = BlobResult_.at(nBlobNum).rectBox.width * BlobResult_.at(nBlobNum).rectBox.height;
        // Bounding Box面积比率/对象面积(Rectangulaty(=Extent))
        BlobResult_.at(nBlobNum).fBoxRatio = BlobResult_.at(nBlobNum).nArea / static_cast<float>(BlobResult_.at(nBlobNum).nBoxArea);
        // 拯救Elongation
        BlobResult_.at(nBlobNum).fElongation = BlobResult_.at(nBlobNum).rectBox.width / static_cast<float>(BlobResult_.at(nBlobNum).rectBox.height);
        // 获取stdDev
        cv::Scalar m;
        cv::Scalar s;
        cv::meanStdDev(matTmp_src, m, s, matTemp);
        BlobResult_.at(nBlobNum).fStdDev = static_cast<float>(s[0]);
        // 拯救Contours
        std::vector<std::vector<cv::Point>> ptContours;
        std::vector<std::vector<cv::Point>>().swap(ptContours);
        cv::findContours(matTemp, ptContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // 获取Perimeter
        if (ptContours.size() != 0) {
            // ROI画面,需要校正
            // 复制轮廓坐标结果
            for (int m = 0; m < ptContours.size(); m++) {
                for (int k = 0; k < ptContours.at(m).size(); k++) {
                    BlobResult_.at(nBlobNum).ptContours.push_back(cv::Point(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY));
                }
            }
        }
        else {
            // 因为是原始坐标,所以不用校正。
            // 复制轮廓坐标结果
            BlobResult_.at(nBlobNum).ptContours.resize(static_cast<int>(BlobResult_.at(nBlobNum).ptIndexs.size()));
            std::copy(BlobResult_.at(nBlobNum).ptIndexs.begin(), BlobResult_.at(nBlobNum).ptIndexs.end(), BlobResult_.at(nBlobNum).ptContours.begin());
        }
        BlobResult_.at(nBlobNum).fPerimeter = static_cast<float>(cv::arcLength(BlobResult_.at(nBlobNum).ptContours, true));
        std::vector<std::vector<cv::Point>>().swap(ptContours);

        // 获取Roundness
        BlobResult_.at(nBlobNum).fRoundness = (fVal * BlobResult_.at(nBlobNum).nArea) / (BlobResult_.at(nBlobNum).fPerimeter * BlobResult_.at(nBlobNum).fPerimeter);

        // 对象有多接近原样？(周长^2/4*Pi*面积)
        BlobResult_.at(nBlobNum).fCompactness = (BlobResult_.at(nBlobNum).fPerimeter * BlobResult_.at(nBlobNum).fPerimeter) / (fVal * float(BlobResult_.at(nBlobNum).nArea));

        // 获取Defect GV
        BlobResult_.at(nBlobNum).fMeanGV = nSum_in / static_cast<float>(nCount_in);

        // 获取背景GV
        BlobResult_.at(nBlobNum).fBKGV = nSum_out / static_cast<float>(nCount_out);

        // 求GV差值(背景-对象)
        BlobResult_.at(nBlobNum).fDiffGV = BlobResult_.at(nBlobNum).fBKGV - BlobResult_.at(nBlobNum).fMeanGV;

        // min,获取max GV
        double valMin = NAN;
        double valMax = NAN;
        double ta, tb;
        cv::minMaxIdx(matTmp_src, &ta, &tb);
        cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
        BlobResult_.at(nBlobNum).nMinGV = static_cast<long>(valMin);
        BlobResult_.at(nBlobNum).nMaxGV = static_cast<long>(valMax);

        // 对象最小亮度/对象平均亮度
        BlobResult_.at(nBlobNum).nMinGVRatio = BlobResult_.at(nBlobNum).nMinGV / BlobResult_.at(nBlobNum).fBKGV;
        // 对象最大亮度/对象平均亮度
        BlobResult_.at(nBlobNum).nMaxGVRatio = BlobResult_.at(nBlobNum).nMaxGV / BlobResult_.at(nBlobNum).fBKGV;
        // 背景亮度/对象平均亮度
        BlobResult_.at(nBlobNum).fDiffGVRatio = BlobResult_.at(nBlobNum).fMeanGV / BlobResult_.at(nBlobNum).fBKGV;
        // 获取Center Point
        BlobResult_.at(nBlobNum).ptCenter.x = static_cast<int>(matCentroid.at<double>(idx, 0));
        BlobResult_.at(nBlobNum).ptCenter.y = static_cast<int>(matCentroid.at<double>(idx, 1));

        // 拯救SEMU
        if (BlobResult_.at(nBlobNum).fDiffGV == 0.0) {
            if (BlobResult_.at(nBlobNum).fBKGV == 0) {
                BlobResult_.at(nBlobNum).fSEMU = 1.0 / (1.97F / (cv::pow((float)BlobResult_.at(nBlobNum).nArea, 0.33F) + 0.72F));
            }
            else {
                BlobResult_.at(nBlobNum).fSEMU = (0.000001 / BlobResult_.at(nBlobNum).fBKGV) / (1.97 / (cv::pow((float)BlobResult_.at(nBlobNum).nArea, 0.33F) + 0.72F));
            }
        }
        else {
            if (BlobResult_.at(nBlobNum).fBKGV == 0) {
                BlobResult_.at(nBlobNum).fSEMU =
                    (fabs(BlobResult_.at(nBlobNum).fMeanGV - BlobResult_.at(nBlobNum).fBKGV) / 0.000001) / (1.97 / (cv::pow((float)BlobResult_.at(nBlobNum).nArea, 0.33F) + 0.72F));
            }
            else {
                BlobResult_.at(nBlobNum).fSEMU = (fabs(BlobResult_.at(nBlobNum).fMeanGV - BlobResult_.at(nBlobNum).fBKGV) / BlobResult_.at(nBlobNum).fBKGV) /
                                                 (1.97 / (cv::pow((float)BlobResult_.at(nBlobNum).nArea, 0.33F) + 0.72F));
            }
        }

        cv::RotatedRect BoundingBox = cv::minAreaRect(BlobResult_.at(nBlobNum).ptIndexs);
        // Box width and length
        BlobResult_.at(nBlobNum).BoxSize = BoundingBox.size;
        // Angle between the horizontal axis
        BlobResult_.at(nBlobNum).fAngle = BoundingBox.angle;
        // Minor Axis & Major Axis
        if (BoundingBox.size.width > BoundingBox.size.height) {
            BlobResult_.at(nBlobNum).fMinorAxis = BoundingBox.size.width;
            BlobResult_.at(nBlobNum).fMajorAxis = BoundingBox.size.height;
        }
        else {
            BlobResult_.at(nBlobNum).fMinorAxis = BoundingBox.size.height;
            BlobResult_.at(nBlobNum).fMajorAxis = BoundingBox.size.width;
        }

        // Feret’s area
        BlobResult_.at(nBlobNum).fMinBoxArea = BlobResult_.at(nBlobNum).fMinorAxis * BlobResult_.at(nBlobNum).fMajorAxis;

        // Axis Ratio
        if (BlobResult_.at(nBlobNum).fMajorAxis > 0) {
            BlobResult_.at(nBlobNum).fAxisRatio = BlobResult_.at(nBlobNum).fMinorAxis / BlobResult_.at(nBlobNum).fMajorAxis;
        }
        else {
            BlobResult_.at(nBlobNum).fAxisRatio = 0.F;
        }

        // Min Bounding Box面积比/对象面积(区域孔隙率)
        BlobResult_.at(nBlobNum).fMinBoxRatio = BlobResult_.at(nBlobNum).fMinBoxArea / static_cast<float>(BlobResult_.at(nBlobNum).nArea);
        // choikwangil
        BlobResult_.at(nBlobNum).fMeanAreaRatio = BlobResult_.at(nBlobNum).fMeanGV / static_cast<float>(BlobResult_.at(nBlobNum).nArea);
        BlobResult_.at(nBlobNum).fDefectMeanGV  = (tb - ta);
        // 取消分配
        matTmp_src.release();
        matTmp_label.release();
        matTemp.release();
    }
    return true;
}

bool VBlob::DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Rect rectROI, cv::Mat GrayBuffer, int nMaxDefectCount)
{
    Release();
    if (ThresholdBuffer.empty()) {
        return false;
    }
    if (ThresholdBuffer.channels() != 1) {
        return false;
    }
    if (GrayBuffer.channels() != 1) {
        cv::cvtColor(GrayBuffer, GrayBuffer, cv::COLOR_BGR2GRAY);
    }
    bool bGrayEmpty = false;
    if (GrayBuffer.empty() || GrayBuffer.channels() != 1) {
        GrayBuffer = ThresholdBuffer.clone();
        bGrayEmpty = true;
    }

    cv::Mat matLabel;
    cv::Mat matStats;
    cv::Mat matCentroid;
    matLabel              = cv::Mat(ThresholdBuffer.size(), CV_32SC1, false);
    long long nTotalLabel = 0;

    if (ThresholdBuffer.type() == CV_8U) {
        nTotalLabel = cv::connectedComponentsWithStats(ThresholdBuffer, matLabel, matStats, matCentroid, 8, CV_32S, cv::CCL_GRANA) - 1;
    }
    else {
        cv::Mat matSrc8bit = cv::Mat(ThresholdBuffer.size(), CV_8UC1);
        ThresholdBuffer.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);
        nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S, cv::CCL_GRANA) - 1;
        matSrc8bit.release();
    }
    // 异常处理
    if (nTotalLabel < 0) {
        if (bGrayEmpty) {
            GrayBuffer.release();
        }
        if (!matLabel.empty()) {
            matLabel.release();
        }
        if (!matStats.empty()) {
            matStats.release();
        }
        if (!matCentroid.empty()) {
            matCentroid.release();
        }
        return false;
    }
    // 异常处理
    if (nTotalLabel >= nMaxDefectCount) {
        nTotalLabel = nMaxDefectCount - 1;
    }
    // 默认Feature计算8bit
    if (GrayBuffer.type() == CV_8UC1) {
        DoFeatureBasic_8bit(matLabel, matStats, matCentroid, GrayBuffer, static_cast<int>(nTotalLabel));
    }
    else {
        DoFeatureBasic_16bit(matLabel, matStats, matCentroid, GrayBuffer, static_cast<int>(nTotalLabel));
    }

    CoordApply(rectROI, nTotalLabel);

    if (bGrayEmpty) {
        GrayBuffer.release();
    }
    if (!matLabel.empty()) {
        matLabel.release();
    }
    if (!matStats.empty()) {
        matStats.release();
    }
    if (!matCentroid.empty()) {
        matCentroid.release();
    }
    bComplete_ = true;
    return true;
}

// 坐标校正
void VBlob::CoordApply(cv::Rect rectROI, int nTotalLabel)
{
    for (int nBlobNum = 0; nBlobNum < nTotalLabel; nBlobNum++) {
        BlobResult_.at(nBlobNum).rectBox.x += rectROI.tl().x;
        BlobResult_.at(nBlobNum).rectBox.y += rectROI.tl().y;
        BlobResult_.at(nBlobNum).ptCenter.x += rectROI.tl().x;
        BlobResult_.at(nBlobNum).ptCenter.y += rectROI.tl().y;
        for (int idx = 0; idx < BlobResult_.at(nBlobNum).ptIndexs.size(); idx++) {
            BlobResult_.at(nBlobNum).ptIndexs[idx].x += rectROI.tl().x;
            BlobResult_.at(nBlobNum).ptIndexs[idx].y += rectROI.tl().y;
        }
        for (int idx = 0; idx < BlobResult_.at(nBlobNum).ptContours.size(); idx++) {
            BlobResult_.at(nBlobNum).ptContours[idx].x += rectROI.tl().x;
            BlobResult_.at(nBlobNum).ptContours[idx].y += rectROI.tl().y;
        }
    }
}
std::vector<tBLOB_FEATURE> VBlob::DoDefectBlobSingleJudgment(const std::vector<STRU_DEFECT_ITEM>& EngineerBlockDefectJudge)
{
    std::vector<tBLOB_FEATURE> dst_blob;
    for (int nFork = 0; nFork < EngineerBlockDefectJudge.size(); nFork++) {
        if (EngineerBlockDefectJudge[nFork].strItemName.empty()) {
            continue;
        }
        for (int i = 0; i < BlobResult_.size(); i++) {
            bool bFilter = true;
            bool bInit   = false;
            for (int nForj = 0; nForj < EngineerBlockDefectJudge[nFork].Judgment.size(); nForj++) {
                // 特征未使用，或者过滤特征大于特征总数
                if (!EngineerBlockDefectJudge[nFork].Judgment[nForj].bUse || EngineerBlockDefectJudge[nFork].Judgment[nForj].feature_index > E_FEATURE_COUNT) {
                    continue;
                }
                bInit = true;
                if (!DoFiltering(BlobResult_[i],                                                  // Blob结果
                                 EngineerBlockDefectJudge[nFork].Judgment[nForj].feature_index,   // 比较Feature
                                 EngineerBlockDefectJudge[nFork].Judgment[nForj].nSign,           // 运算符（<，>，==，<=，>=）
                                 EngineerBlockDefectJudge[nFork].Judgment[nForj].dValue))         // 值
                {
                    bFilter = false;
                    break;
                }
            }
            // 满足所有设置的条件
            if (bInit && bFilter) {
                BlobResult_[i].bFiltering = true;
                dst_blob.push_back(BlobResult_[i]);
            }
        }
    }
    return dst_blob;
}

bool VBlob::DoFiltering(tBLOB_FEATURE& tBlobResult, int nBlobFilter, int nSign, double dValue)
{
    // 如果已过滤，则排除
    if (tBlobResult.bFiltering) {
        return false;
    }
    bool bRes = false;
    switch (nBlobFilter) {
    case E_FEATURE_AREA:
        bRes = Compare(static_cast<double>(tBlobResult.nArea), nSign, dValue);
        break;
    case E_FEATURE_BOX_AREA:
        bRes = Compare(static_cast<double>(tBlobResult.nBoxArea), nSign, dValue);
        break;
    case E_FEATURE_BOX_RATIO:
        bRes = Compare(static_cast<double>(tBlobResult.fBoxRatio), nSign, dValue);
        break;
    case E_FEATURE_BOX_X:
        bRes = Compare(static_cast<double>(tBlobResult.rectBox.width), nSign, dValue);
        break;
    case E_FEATURE_BOX_Y:
        bRes = Compare(static_cast<double>(tBlobResult.rectBox.height), nSign, dValue);
        break;
    case E_FEATURE_SUM_GV:
        bRes = Compare(static_cast<double>(tBlobResult.nSumGV), nSign, dValue);
        break;
    case E_FEATURE_MIN_GV:
        bRes = Compare(static_cast<double>(tBlobResult.nMinGV), nSign, dValue);
        break;
    case E_FEATURE_MAX_GV:
        bRes = Compare(static_cast<double>(tBlobResult.nMaxGV), nSign, dValue);
        break;
    case E_FEATURE_MEAN_GV:
        bRes = Compare(static_cast<double>(tBlobResult.fMeanGV), nSign, dValue);
        break;
    case E_FEATURE_DIFF_GV:
        bRes = Compare(static_cast<double>(tBlobResult.fDiffGV), nSign, dValue);
        break;
    case E_FEATURE_BK_GV:
        bRes = Compare(static_cast<double>(tBlobResult.fBKGV), nSign, dValue);
        break;
    case E_FEATURE_STD_DEV:
        bRes = Compare(static_cast<double>(tBlobResult.fStdDev), nSign, dValue);
        break;
    case E_FEATURE_SEMU:
        bRes = Compare(static_cast<double>(tBlobResult.fSEMU), nSign, dValue);
        break;
    case E_FEATURE_COMPACTNESS:
        bRes = Compare(static_cast<double>(tBlobResult.fCompactness), nSign, dValue);
        break;
    case E_FEATURE_MIN_GV_RATIO:
        bRes = Compare(static_cast<double>(tBlobResult.nMinGVRatio), nSign, dValue);
        break;
    case E_FEATURE_MAX_GV_RATIO:
        bRes = Compare(static_cast<double>(tBlobResult.nMaxGVRatio), nSign, dValue);
        break;
    case E_FEATURE_DIFF_GV_RATIO:
        bRes = Compare(static_cast<double>(tBlobResult.fDiffGVRatio), nSign, dValue);
        break;
    case E_FEATURE_PERIMETER:
        bRes = Compare(static_cast<double>(tBlobResult.fPerimeter), nSign, dValue);
        break;
    case E_FEATURE_ROUNDNESS:
        bRes = Compare(static_cast<double>(tBlobResult.fRoundness), nSign, dValue);
        break;
    case E_FEATURE_ELONGATION:
        bRes = Compare(static_cast<double>(tBlobResult.fElongation), nSign, dValue);
        break;
    case E_FEATURE_MIN_BOX_AREA:
        bRes = Compare(static_cast<double>(tBlobResult.fMinBoxArea), nSign, dValue);
        break;
    case E_FEATURE_MINOR_AXIS:
        bRes = Compare(static_cast<double>(tBlobResult.fMinorAxis), nSign, dValue);
        break;
    case E_FEATURE_MAJOR_AXIS:
        bRes = Compare(static_cast<double>(tBlobResult.fMajorAxis), nSign, dValue);
        break;
    case E_FEATURE_AXIS_RATIO:
        bRes = Compare(static_cast<double>(tBlobResult.fAxisRatio), nSign, dValue);
        break;
    case E_FEATURE_MIN_BOX_RATIO:
        bRes = Compare(static_cast<double>(tBlobResult.fMinBoxRatio), nSign, dValue);
        break;
    case E_FEATURE_ANGLE:
        bRes = Compare(static_cast<double>(tBlobResult.fAngle), nSign, dValue);
        break;
    case E_FEATURE_GV_UP_COUNT_0:
    case E_FEATURE_GV_UP_COUNT_1:
    case E_FEATURE_GV_UP_COUNT_2:
    {
        int nCount = static_cast<int>(dValue) / 10000;
        int nGV    = static_cast<int>(dValue) % 10000;

        if (nGV < 0) {
            nGV = 0;
        }
        if (nGV > IMAGE_MAX_GV) {
            nGV = IMAGE_MAX_GV - 1;
        }

        long long nHist = 0;
        for (int m = nGV; m < IMAGE_MAX_GV; m++) {
            nHist += tBlobResult.nHist[m];
        }

        bRes = Compare(static_cast<double>(nHist), nSign, static_cast<double>(nCount));
    }; break;

    case E_FEATURE_GV_DOWN_COUNT_0:
    case E_FEATURE_GV_DOWN_COUNT_1:
    case E_FEATURE_GV_DOWN_COUNT_2:
    {
        int nCount = static_cast<int>(dValue) / 10000;
        int nGV    = static_cast<int>(dValue) % 10000;

        if (nGV < 0) {
            nGV = 0;
        }
        if (nGV > IMAGE_MAX_GV) {
            nGV = IMAGE_MAX_GV - 1;
        }
        long long nHist = 0;
        for (int m = 0; m <= nGV; m++) {
            nHist += tBlobResult.nHist[m];
        }
        bRes = Compare(static_cast<double>(nHist), nSign, static_cast<double>(nCount));
    }; break;

    case E_FEATURE_MEANAREA_RATIO:   // choikwangil
        bRes = Compare(static_cast<double>(tBlobResult.fMeanAreaRatio), nSign, dValue);
        break;

    case E_FEATURE_GVAREA_RATIO_TEST:   // 04.20 choi
    {

        int    nTmp    = static_cast<int>(dValue) % 10000;
        double nPer    = (dValue - static_cast<double>(nTmp)) / 10000.0;
        double nRatio  = nTmp / 1000;
        double Mean_GV = tBlobResult.fBKGV * nRatio;
        if (Mean_GV < 0) {
            Mean_GV = 0;
        }
        if (Mean_GV > IMAGE_MAX_GV) {
            Mean_GV = IMAGE_MAX_GV - 1;
        }
        long long nHist = 0;
        for (int m = Mean_GV; m <= 255; m++) {
            nHist += tBlobResult.nHist[m];
        }
        double Area_per = nHist / tBlobResult.nBoxArea;
        Area_per *= 100;
        bRes = Compare(Area_per, nSign, nPer);
    }; break;
    default:
        bRes = false;
        break;
    }
    return bRes;
}

// 运算符(<,>,==,<=,>=)
bool VBlob::Compare(double dFeatureValue, int nSign, double dValue)
{
    bool bRes = false;
    switch (nSign) {
    case E_SIGN_EQUAL:
        bRes = dFeatureValue == dValue;   // x == judgment value
        break;
    case E_SIGN_NOT_EQUAL:
        bRes = dFeatureValue != dValue;   // x != judgment value
        break;
    case E_SIGN_GREATER:
        bRes = dFeatureValue > dValue;   // x >  judgment value
        break;
    case E_SIGN_LESS:
        bRes = dFeatureValue < dValue;   // x <  judgment value
        break;
    case E_SIGN_GREATER_OR_EQUAL:
        bRes = dFeatureValue >= dValue;   // x >= judgment value
        break;
    case E_SIGN_LESS_OR_EQUAL:
        bRes = dFeatureValue <= dValue;   // x <= judgment value
        break;
    default:
        bRes = false;
        break;
    }
    return bRes;
}

int VBlob::getSignFromSymbol(const std::string& symbol)
{
    if (symbol == "=")
        return 0;
    else if (symbol == "<>")
        return 1;
    else if (symbol == ">")
        return 2;
    else if (symbol == "<")
        return 3;
    else if (symbol == ">=")
        return 4;
    else if (symbol == "<=")
        return 5;
    else if (symbol == ">||")
        return 6;
    else if (symbol == "||<")
        return 7;
    else
        return -1;
}

int VBlob::getIndex(const std::string& feature_str)
{
    if (auto it = feature_idx.find(feature_str); it != feature_idx.end()) {
        return it->second;
    }
}