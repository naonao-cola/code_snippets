/************************************************************************
Mura2算法相关标头
************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"

#define BG_METHOD_BLUR 0
#define BG_METHOD_PROFILE 1

#define PROFILE_ROW 0
#define PROFILE_COL 1

class CInspectMura2
{
public:
	CInspectMura2(void);
	virtual ~CInspectMura2(void);

	//Main检查算法
	long		FindMuraDefect(cv::Mat matSrcBuffer, cv::Mat& matDrawBuffer, cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath,
		stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt = NULL);

	long LogicStart_Gray(cv::Mat& matSrcImage, cv::Mat& matDarkResultImage_01, cv::Mat& matBrightResultImage_01,
		CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge);

protected:

	//按区间平均,将平均GV的特定比率以上的值按为Min,Max值。
	void SetMinMax(Mat& matSrcImage, int nMaxGVAreaPartX, int nMaxGVAreaPartY, double dblMinGVR, double dblMaxGVR);

	//用于分别处理外围和中央的Threshold函数
	void MakeThresholdImageWithCenterAndOut(Mat& matSrcImage, int nOutLine, int nThresholdCen, int nThresholdOut);

	//创建Thresold图像。(生成二进制图像)
	void MakeThresholdImage(Mat& matOri, Mat& matDst, double dblAverage, float fThresholdR, double fThresholdR_Outline, int nOutLineArea);

	//制作背景图像。
	void MakeBGImage(int nMethod, Mat& matSrcImage, Mat& matBGImage, Size szParam01);
	void MakeBGImage_Blur(Mat& matSrcImage, Mat& matBGImage, Size szParam01);
	void MakeBGImage_Profile(Mat& matSrcImage, Mat& matBGImage);
};

