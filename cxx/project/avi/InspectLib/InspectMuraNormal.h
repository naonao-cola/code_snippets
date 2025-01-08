
/************************************************************************/
//Mura不良检测相关标头
//修改日期:18.09.25
/************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include "InspectLibLog.h"
#include "MatBuf.h"				//内存管理

//R,G,B,Gray,White通用
enum ENUM_PARA_AVI_MURA_NORMAL
{
	E_PARA_AVI_MURA_NORMAL_DUST_TEXT = 0,	//显示文本
	E_PARA_AVI_MURA_NORMAL_DUST_BRIGHT_FLAG,	//使用除尘器
	E_PARA_AVI_MURA_NORMAL_DUST_DARK_FLAG,	//使用除尘器
	E_PARA_AVI_MURA_NORMAL_DUST_BIG_AREA,	//只留下大面积不良
	E_PARA_AVI_MURA_NORMAL_DUST_ADJUST_RANGE,	//不良周边有灰尘时清除不良

	E_PARA_AVI_MURA_NORMAL_TYPE4_TEXT,	// Type 4 Parameter
	E_PARA_AVI_MURA_NORMAL_TYPE4_SHIFT_UNIT,	// Shift Copy Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_CUT_UNIT,	// Edge Cut Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_ADD_UNIT,	// Edge Cut Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_EDGECUT_UNIT,	// Shift Copy Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_MEANBLUR,	// Shift Copy Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_RESIZE,	// Shift Copy Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_SIZE,	// Sample Number
	E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_ROI_X,	// ROI_X Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_ROI_Y,	// ROI_Y Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_AVG_UNIT,	// AVG. Filter UNIT
	E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_MINMAX_UNIT,	// Min-Max Filter Unit
	E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_DIFF_AREA,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_DIFF_AREA_LIMITE,	// TH Diff Area

	E_PARA_AVI_MURA_NORMAL_NOTCH_TEXT,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_UNIT,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_RECT_OFFSET,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_RECT_WIDTH,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_RECT_HEIGHT,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_AVG_UNIT,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_CHECK_UNIT,	// TH Diff Area
	E_PARA_AVI_MURA_NORMAL_NOTCH_SLOPE_CALC_UNIT,	// TH Diff Area

	E_PARA_AVI_MURA_NORMAL_TYPE3_TEXT,	// Type 3 Parameter
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_TEXT,	// Type 3 Parameter
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX,	// Seg X Unit
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGY,	// Seg Y Unit
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGEAREA,	// Edge Area
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_DR,	// Active Dark Ratio
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_BR,	// Active Bright Ratio
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_DR,	// Edge Dark Ratio
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_BR,	// Edge Bright Ratio
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MINGV,	// Min Count GV
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MEDIANSIZE,	// Median Size
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MOPHOSIZE,	// Median Size
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_DISTTH,
	E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_DELET_OUTLINE,	// Out Line Delete Bright

	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TEXT,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BGSEG_XY,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_SEG_XY,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BG_BRIGHT_RATIO,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BG_DARK_RATIO,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_BRIGHT_RATIO,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_DARK_RATIO,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_STR,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_MID1,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_MID2,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_WEAK1,	// 
	E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_WEAK2,	//

	E_PARA_AVI_MURA_NORMAL_TYPE2_TEXT,
	E_PARA_AVI_MURA_NORMAL_TYPE2_BLURLOOP,	//
	E_PARA_AVI_MURA_NORMAL_TYPE2_RESIZE,
	E_PARA_AVI_MURA_NORMAL_TYPE2_GAUSSIAN_SIZE,
	E_PARA_AVI_MURA_NORMAL_TYPE2_GAUSSIAN_SIGMA,
	E_PARA_AVI_MURA_NORMAL_TYPE2_SEG_X,
	E_PARA_AVI_MURA_NORMAL_TYPE2_SEG_Y,
	E_PARA_AVI_MURA_NORMAL_TYPE2_BG_BLUESIZE_X,	//
	E_PARA_AVI_MURA_NORMAL_TYPE2_BG_BLUESIZE_Y,	//
	E_PARA_AVI_MURA_NORMAL_TYPE2_PRO_BLURSIZE,
	E_PARA_AVI_MURA_NORMAL_TYPE2_EDGE_AREA_OFFSET,
	E_PARA_AVI_MURA_NORMAL_TYPE2_PRO_CHECK_STD,
	E_PARA_AVI_MURA_NORMAL_TYPE2_THRES_X,
	E_PARA_AVI_MURA_NORMAL_TYPE2_THRES_Y,

	E_PARA_AVI_MURA_NORMAL_TYPE1_TEXT,
	E_PARA_AVI_MURA_NORMAL_TYPE1_PROFILE_BLUR,
	E_PARA_AVI_MURA_NORMAL_TYPE1_RESULT_MORPH,
	E_PARA_AVI_MURA_NORMAL_TYPE1_ACTIVE_TH_X_VALUE,
	E_PARA_AVI_MURA_NORMAL_TYPE1_ACTIVE_TH_Y_VALUE,
	E_PARA_AVI_MURA_NORMAL_TYPE1_EDGE_UNIT,
	E_PARA_AVI_MURA_NORMAL_TYPE1_EDGE_TH_X_VALUE,
	E_PARA_AVI_MURA_NORMAL_TYPE1_EDGE_TH_Y_VALUE,
	E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_MIN_DEFECT_WIDTH,
	E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTH_X,
	E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTH_Y,
	E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTYPE_X,
	E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTYPE_Y,

	E_PARA_AVI_MURA_NORMAL_TOTAL_COUNT							// Total
};

enum ENUM_DEFECT_LINEMURA
{
	E_DEFECT_LINEMURA_X = 0,
	E_DEFECT_LINEMURA_Y,

	E_DEFECT_LINEMURA_TOTAL
};

enum ENUM_MORP_MURANORMAL
{
	EP_MORP_MURANORMAL_INIT = 0,
	EP_MORP_MURANORMAL_ERODE,
	EP_MORP_MURANORMAL_DILATE,
	EP_MORP_MURANORMAL_OPEN,
	EP_MORP_MURANORMAL_CLOSE,
	EP_MORP_MURANORMAL_GRADIENT,
	EP_MORP_MURANORMAL_TOPHAT,
	EP_MORP_MURANORMAL_BLACKHAT,
};

enum ENUM_MURANORMAL_DEFECT_NAME
{

	E_MURANORMAL_TYPE2_X = 0,
	E_MURANORMAL_TYPE2_Y,
	E_MURANORMAL_TYPE3_D,
	E_MURANORMAL_TYPE3_B,
};

enum ENUM_MURANORMAL_TYPE1_TYPE
{
	E_MURANORMAL_TYPE1_DEFECT_X = 0,
	E_MURANORMAL_TYPE1_DEFECT_Y,
};

enum ENUM_MURANORMAL_NOTCH_DEFECT
{
	E_MURANORMAL_NOTCH_DEFECT_NONE = 0,
	E_MURANORMAL_NOTCH_DEFECT_TOP,
	E_MURANORMAL_NOTCH_DEFECT_BOTTOM,
	E_MURANORMAL_NOTCH_DEFECT_BOTH,
};

class CInspectMuraNormal
{
public:
	CInspectMuraNormal(void);
	virtual ~CInspectMuraNormal(void);

	//内存管理
	CMatBuf* cMem[4];
	void		SetMem_Multi(int nCnt, CMatBuf** data)
	{
		for (int i = 0; i < nCnt; i++)
		{
			cMem[i] = data[i];
		}
	};

	//日志管理
	InspectLibLog* m_cInspectLibLog;
	clock_t				m_tInitTime;
	clock_t				m_tBeforeTime;
	wchar_t* m_strAlgLog;

	void		SetLog(InspectLibLog* cLog, clock_t tTimeI, clock_t tTimeB, wchar_t* strLog)
	{
		m_tInitTime = tTimeI;
		m_tBeforeTime = tTimeB;
		m_cInspectLibLog = cLog;
		m_strAlgLog = strLog;
	};

	void		writeInspectLog(int nAlgType, char* strFunc, wchar_t* strTxt)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, m_strAlgLog);
	};

	void		writeInspectLog_Memory(int nAlgType, char* strFunc, wchar_t* strTxt, __int64 nMemory_Use_Value = 0)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, nMemory_Use_Value, m_strAlgLog);
	};
	//////////////////////////////////////////////////////////////////////////

		//Main检查算法
	long		DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDstBright, cv::Mat& matDstDark,
		cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	//删除Dust后,移交结果向量
	long		GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
		double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt);

protected:
	//19.04.23 PNZ-Type1Mura检测
	long		LogicStart_MuraType1(cv::Mat& matSrcImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	// Type2
	long		LogicStart_MuraType2(cv::Mat& matSrcImage, cv::Mat* matdstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath);

	//PNZ-Type3 Mura检测
	long		LogicStart_MuraType3(cv::Mat& matSrcImage, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath);

	// PNZ - Tpye4 Mura Check
	long		LogicStart_MuraType4(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

protected:
	//保存画面(8bit&12bit)
	long		ImageSave(CString strPath, cv::Mat matSrcBuf);

	//清除小面积&附近的不良连接
	long		DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);

	//删除Dust面积较大的环境
	long		DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS);

	long		ShiftCopyParaCheck(int ShiftValue, int& nCpyX, int& nCpyY, int& nLoopX, int& nLoopY);

	// Type2 LineMura Funtion
	long		resizeGaussian(Mat& InPutImage, Mat& OutPutImage, int reSizeValue, int MaskSize, float fSigma, CMatBuf* cMemSub = NULL);

	// Type 4 Inspection Function ( PNZ )
	long		LoopBlur(cv::Mat& matSrcImage, cv::Mat& MatBlur, int nLoop, CMatBuf* cMemSub = NULL);

	long		ProfileJudgment(cv::Mat& matSrcImage, int* nCommonPara, CString strAlgPath, double* dPara, stDefectInfo* pResultBlob, CMatBuf* cMemSub = NULL);

	long		SampleProfileCalc(cv::Mat& matSrcImage, cv::Mat& MatproYBuf, double* dParam, int nDirection);

	long		ProfileUnitMinMax(cv::Mat& matSrcImage, int nUnit, cv::Mat& matDstImageMin, cv::Mat& matDstImageMax);

	long		ProfileSlopeCheck(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nSlopCalceUnite);

	// Tpye 3 Inspection Function
	long		MainProcessingT3(cv::Mat& matSrcImage, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		SBSInspection(cv::Mat& matSrcImage, cv::Mat& matSrcImageBG, cv::Mat& matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		Morphology(Mat& matSrcImage, Mat& matDstImage, long nSizeX, long nSizeY, int nOperation, CMatBuf* cMemSub = NULL, int nIter = 1);

	long		RangeAvgThreshold(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub);

	long		ResultImageReset(cv::Mat* matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	// Type 3 BBS ( Big black Spot)
	long		BBSInspection(cv::Mat& matSrcImageData, cv::Mat& matSrcImageBG, cv::Mat& matDstImage, cv::Mat& matDiffTemp, cv::Mat& matBKBuffer, cv::Rect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	// Type3 RingMura
	//long		RoundConverlution(cv::Mat& matSrcImage, cv::Mat& matDstImage, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		SetRoundKernel(cv::Mat& matKernel_Mid, cv::Mat& matKernel_Center, int nKernelSize, int nRoundWidth);

	long		setKernelConvolution(cv::Mat& matSrcImage, cv::Mat& matKernel, cv::Mat& matDstImage, CMatBuf* cMemSub = NULL);

	long		setSubThresholding(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matDstImage, float nThresholdGV, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	// Feature Extraction
	void		Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset = 0);

	// Type 1 Defect
	long		MainProcessingT1(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		FindDrawResult(cv::Mat* matDstBuf, stDefectInfo* pResultBlob, cv::Rect InputROI, double* dPara, int nResize, int nImageUI, int nPS);

	long		FindnCalContours(cv::Mat& matSrcBuf, cv::Rect InputROI, int nResize, vector<vector<cv::Point>>& ptDefectContours);

	long		AddDefectContours(vector<vector<cv::Point>>& ptDefect_X, vector<vector<cv::Point>>& ptDefect_Y, vector<vector<cv::Point>>& ptDefectALL);

	long		DrawDefectContours(vector<vector<cv::Point>>& ptDefectContours_TOTAL, stDefectInfo* pResultBlob, double* dPara, int nImageUI, int nPS);

	long		CalcDataMinMax(vector<cv::Point>& ptDefectContour, double& nXmax, double& nXmin, double& nYmax, double& nYmin);

	long		ProfileSlopeCalc(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nSlopCalcUnite);

	long		ProfileSlopeSUM(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matSrcImage3, cv::Mat& matSrcImage4, cv::Mat& matSrcImage5, float fSUMTempTH, int nSUMType, cv::Mat& matDstImage);

	long		ProfileDefectCheckLogic(cv::Mat& matSrcImage, float fThresholdValue);

	long		ProfileFindDefect(cv::Mat& matSrcImageDark, float fThresholdValue, cv::Mat& matResult, int nDefectDirection);

	long		ProfileFindDefectEdge(cv::Mat& matSrcImageDark, float fThresholdValue, int nEdgeUnit, cv::Mat& matResult, int nDefectDirection, int* nCommonPara, CString strAlgPath);

	long		PositiveSlopeDataCheck(cv::Mat& matSrcImage, float fThresholdValue, int nCalcLength, int& nDefectStart, cv::Mat& matSlope);

	long		NegativeSlopeDataCheck(cv::Mat& matSrcImage, float fThresholdValue, int nCalcLength, int& nDefectStart, cv::Mat& matSlope);

	// PNZ 19.07.03 Pre-Processing Optimization
	long		PreProOptimization(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	//PNZ 19.07.16 Notch ROI Screen分离Defect Check
	long		NotchROICheck(cv::Mat& matSrcImage, cv::Mat& matSrcNotchReBGBuf, cv::Rect ProROINotchBG, int* nCommonPara, CString strAlgPath, double* dPara, stDefectInfo* pResultBlob, CMatBuf* cMemSub = NULL);

	long		NorchProfileCheck(cv::Mat& matSrcImage, int nCheckUite, double& ProfTop_LeftAvg, double& ProfTop_RightAvg, CMatBuf* cMemSub = NULL);

	long		NorchSlopeCalc(cv::Mat& matSrcImage, int nCheckUite, int nSlopCalceUnite, cv::Mat& matDstImage, CMatBuf* cMemSub = NULL);

	long		DefectJudgement(double dbBlurTopDiff, double dbBlurBottomDiff, double dbMax_PSTopBlur, double dbMax_PSBottomBlur, double* dPara, int& nDefectType);

	long		DrawNotchDefect(cv::Mat& matBGTemp, stDefectInfo* pResultBlob, cv::Rect InputROI, int nResize, int nImageUI, int nPS);

	// PNZ 19.08.29 T3 Test

		//x方向管接头
	long		Estimation_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara);

	//y方向管接头
	long		Estimation_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara);

	//xy方向管接头
	long		Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		RangeFitProcess(cv::Mat matSrcBuf, cv::Mat* matDstBuf, long lSegXUnit, long lSegYUnit, double dEstiBright, double dEstiDark, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		RangeFit_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double dEstiBright, double dEstiDark, double* dPara);

	long		RangeFit_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double dEstiBright, double dEstiDark, double* dPara);

	long		RangeDataCalc(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matHistAvgValue, cv::Mat& matAvgValue, cv::Mat& matMinValue, cv::Mat& matMaxValue, cv::Mat& matStdValue,
		cv::Mat& matPCRValue, cv::Mat& matHISTCount, vector<vector<float>>& matHistData, double* dPara, CMatBuf* cMemSub = NULL);

	long		RangeCheckThreshold(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matHistAvgValue, cv::Mat& matAvgValue, cv::Mat& matMinValue, cv::Mat& matMaxValue, cv::Mat& matStdValue,
		cv::Mat& matPCRValue, cv::Mat& matHISTCount, vector<vector<float>>& matHistData, double* dPara, CMatBuf* cMemSub = NULL);

	long		RangeReFit(cv::Mat& matSrcImageBG, cv::Mat& matSrcImageData, cv::Mat& matDstImage, CMatBuf* cMemSub = NULL);

	long		DistanceTransformThreshold(cv::Mat& matSrcImage, cv::Mat& matDstImage, double dThreshold, CMatBuf* cMemSub = NULL);

	long		RangComProcess(cv::Mat& matSrcImage, cv::Mat& matDstImage, long lSegXUnit, long lSegYUnit, double dEstiBright, double dEstiDark, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		ResultCombination(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matDstImage, int nEdgePara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		RangBGEnhance(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matAvgDefect, cv::Mat& matBGEnhanceResult, cv::Mat& matAvgResult, long lSegXUnit, long lSegYUnit, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	long		RangeFitThreshold(cv::Mat& matSrcImage, cv::Mat* matDstImage, double* dPara, CMatBuf* cMemSub = NULL);

	long		SBSandBBSEdgeReCheck(cv::Mat& matSBSTemp, cv::Mat& matBBSTemp, cv::Mat& matDstImage, cv::Mat& matAvgTemp, cv::Mat& matStdTemp, cv::Mat& matMaxTemp, int nEdgeUnite, CMatBuf* cMemSub = NULL);

};