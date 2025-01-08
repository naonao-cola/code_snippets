#pragma once
#include "MatBuf.h"
#include <atltypes.h>

class ExportAPI AlgoBase
{
public:
	static int Test();

	// 背景GV调整
	static long AdjustBkGV(cv::Mat matOri, cv::Mat matBKBuffer, double dblRatio, CMatBuf* cMemSub = NULL);
	
	// 直方图求均值、标准差
	// fullRange:true histSize=256/4096, fullRange:false histSize=pow(256,nBit)
	static long GetHistogram(cv::Mat& matSrcImage, cv::Mat& matHisto, bool fullRange=true);
	static long GetAverage_From_Histo(cv::Mat matHisto, int nMin, int nMax, double& dblAverage);
	static long GetMeanStdDev_From_Histo(cv::Mat matHisto, int nMin, int nMax, double& dblAverage, double& dblStdev);

	// 图像亮度均值
	static double GetAverage(cv::Mat& matSrcImage);
	static double GetAverage(Mat& matSrcImage, double dblAveCutOffCountR_Min, int nAveMinStartGV, int nAveMaxEndGV);
	static double GetAverageForRGB(cv::Mat& matSrcImage);
	static long DistanceTransformThreshold(cv::Mat& matSrcImage, cv::Mat& matDstImage, double dThreshold, CMatBuf* cMemSub = NULL);

	// 二值化
	static long Threshold16(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, int nThreshold, int nMaxGV);
	static long Threshold16_INV(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, int nThreshold, int nMaxGV);
	static long SubThreshold16(cv::Mat& matSrc1Buf, cv::Mat& matSrc2Buf, cv::Mat& matDstBuf, int nThreshold, int nMaxGV);
	static long Binary(cv::Mat& matSrcImage, cv::Mat& matDstImage, double thresh, bool bInv = false, CMatBuf* cMemSub = NULL);
	static long Binary_16bit(cv::Mat& matSrcImage, cv::Mat& matDstImage, double thresh, bool bInv = false, CMatBuf* cMemSub = NULL);
	
	// 图像亮度调整
	static long ApplyMeanGV(Mat& matSrcImage, double dblTargetGV);
	static long ApplyMeanGV(Mat& matSrcImage, double dblTargetGV, CRect rectTemp);
	static long MedianFilter(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, int nKSize, CMatBuf* cMemSub = NULL);

	static long GetRandomSamples(std::vector <cv::Point2i>& ptSrcIndexs, std::vector <cv::Point2i>& ptSamples, int nSampleCount);

	// 直线拟合
	static long calcLineFit(std::vector <cv::Point2i>& ptSamples, long double& dA, long double& dB);
	static long calcLineVerification(std::vector <cv::Point2i>& ptSrcIndexs, std::vector <cv::Point2i>& ptInliers, long double& dA, long double& dB, double distThreshold);
	static long calcRANSAC(std::vector <cv::Point2i>& ptSrcIndexs, long double& dA, long double& dB);
	static long calcRANSAC(std::vector <cv::Point2i>& ptSrcIndexs, long double& dA, long double& dB, int nMinSamples, double distThreshold);

	// ShiftCopy
	static long ShiftCopy(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nShiftX, int nShiftY, int nShiftLoopX, int nShiftLoopY, CMatBuf* cMemSub=NULL);
	static long ShiftCopy16Bit(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nShiftX, int nShiftY, int nShiftLoopX = 1, int nShiftLoopY = 1);

	// 图像求平均
	static long TwoImg_Average(cv::Mat& matSrc1Buf, cv::Mat& matSrc2Buf, cv::Mat& matDstBuf);
	static bool TwoImg_Average2(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf, cv::Mat& matDstBuf); // SVI

	// 对比度调整
	static long Contrast(cv::Mat& matSrc, cv::Mat& matDst, int nMin, int nMax, int nCount);
	static long ContrastColor(cv::Mat& matSrc, cv::Mat& matDst, int nMin, int nMax, int nCount);

	// Pow图像增强
	static long Pow(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, double dPow, int nMaxGV, CMatBuf* cMemSub = NULL);
	static long Image_Pow(int ImgType, double dpow, Mat& InPutImage, Mat& OutPutImage);

	// 最大、最小滤波
	static long MinimumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize);
	static long MaximumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize);

	// Mexican滤波
	static long C_Mexican_filter(cv::Mat& ip, int sz, int nMx_blur_sz, int nMx_blur_sigma);
	static long C_Mexican_filter(cv::Mat& ip, int sz);

	// 投影相关
	static long MakeProfile(int nType, Mat& matSrcImage, Mat& matDstProjection, CMatBuf* cMemSub = NULL);
	static long MakeProfile(cv::Mat& matSrcImage, cv::Mat& MatproYBuf, int width, int height, int nDirection);
	static long MakeProjection(Mat& InPutImage, Mat& MatproYBuf, int width, int height, int Type, int UseNorch, int NorchUnit, std::vector<int> NorchIndex, CPoint OrgIndex);
	static long ProjectionLineDelete(cv::Mat& matSrcImage, int nCutCountX, int nCutCountY, int nThicknessX = 1, int nThicknessY = 1);

	// SVI
	static bool Estimation_X(cv::Mat matSrc, cv::Mat& matDst, int nDimensionX, int nStepX, float fBrightOffset, float fDarkOffset);
	static bool Estimation_Y(cv::Mat matSrc, cv::Mat& matDst, int nDimensionY, int nStepY, float fBrightOffset, float fDarkOffset);
	static bool Estimation_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nDimensionX, int nStepX, float dThBGOffset);
	static bool Estimation_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nDimensionY, int nStepY, float dThBGOffset);
	static bool Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, CRect rectRoi, int nDimensionX, int nDimensionY, int nStepX, int nStepY, float dThBGOffset);

	// 色差计算相关
	static bool Standard_Lab(cv::Mat* matSrcBuf, double* dAvg, cv::Mat& matBKROI);
	static bool CalCIE_Lab2DeltaE2000(cv::Mat* matLABBuf, cv::Mat& matDstBuf, cv::Mat matBKROI, double* dMeanLab, double dLimit_L, double dWeight_L);
	static bool CalCIE_Lab2DeltaE(cv::Mat* matLABBuf, cv::Mat& matDstBuf, cv::Mat matBKROI, double* dMeanLab, double dLimit_L, double dWeight_L);
	static bool CalCIE_Lab2DeltaE2000_Edge(cv::Mat* matLABBuf, cv::Mat& matDstBuf, cv::Mat matBKROI, double* dMeanLab, double dLimit_L, double dWeight_L, cv::Rect matEdge);
	static double DeltaE2000(float L2, float L1, float A2, float A1, float B2, float B1);
	static bool TransLab(cv::Mat& matSrcBuf, cv::Mat* matChannals);
	static bool TransHSV_V(cv::Mat& matSrcBuf, cv::Mat& matBKBuf);

	// 对比度增强
	static bool EnhanceContrast(cv::Mat& matSrcBuf, cv::Mat matBKROI, int nOffSet, double dRatio, int nCase);
	static bool EnhanceContrastColor(cv::Mat& matSrcBuf, cv::Mat matBKROI, int nOffSet, double dRatio, int nCase);
	static long ImageContrast(cv::Mat& matSrcBuf, cv::Mat& matBKBuf, CRect rectROI, double Contrast_Ratio, double Contrast_GVOffset);

	// 缩小、放大投影
	static long StepReduce(cv::Mat& matSrcBuf, cv::Mat& matResBuf, int nStepRows, int nStepCols, int nStepX, int nStepY);
	static long StepExpand(cv::Mat& matSrcBuf, cv::Mat& matResBuf, int nStepRows, int nStepCols, int nStepX, int nStepY);

	// 投影
	static double* LineProfile(cv::Mat& matSrcBuf, bool nAxis = TRUE);

	// 滤波
	static long Filter8(BYTE* InImg, BYTE* OutImg, int nMin, int nMax, int width, int height); // kernelsize=7
	static long Filter8(BYTE* InImg, BYTE* OutImg, int nMin, int nMax, int width, int height, int nMaskSize, int nType = 0);

private:
	static float* diff2Gauss1D(int r);
	static float* computeKernel2D(int r);
	static bool  convolveFloat(cv::Mat& ip, float* kernel, int kw, int kh, int sz);
	static int GetBitFromImageDepth(int nDepth);
	static double getScale(float* kernel, int sz);
	static long Deg2Rad(const double Deg);
	static long Rad2Deg(const double Rad);
	static float TransFormRadDeg(float fValue, int bType);
	static bool FindInSamples(std::vector <cv::Point2i>& ptSamples, cv::Point2i ptIndexs);
	static long CieLab2Hue(double var_a, double var_b);
	static const bool _init;
};
