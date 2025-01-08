
/************************************************************************/
// Mura
// 05.31
/************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include "InspectLibLog.h"
#include "MatBuf.h"

enum ENUM_PARA_AVI_MURA3_GRAY
{

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_RESIZE,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIZE,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIGMA,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_RATIO,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_EDGE_RATIO,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_RATIO,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_EDGE_RATIO,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_MINIMUM_SIZE,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SEG_X,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SEG_Y,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_TEXT,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_FILTER_SIZE,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIZE,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIGMA,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_TEXT,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_FILTER_SIZE,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_BLUR_SIZE,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_BLUR_SIGMA,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_TEXT,	//
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_FLAG,	//
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MORP_OBJ,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MORP_BKG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_THRESHOLD,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_TEXT,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_BRIGHT_RATIO,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC1_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_1,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA1_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA1_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_1,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC2_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_2,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA2_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA2_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_2,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC3_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_3,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA3_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA3_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_3,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC4_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_4,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA4_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA4_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_4,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC5_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_5,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA5_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA5_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_5,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC6_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_6,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA6_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA6_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_6,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC7_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_7,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA7_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA7_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_7,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC8_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_8,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA8_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA8_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_8,

	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_TEXT,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_LEFT,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_TOP,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_RIGHT,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_BOTTOM,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_BRIGHT_RATIO,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC1_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_1,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA1_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA1_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_1,	//
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC2_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_2,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA2_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA2_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_2,	//
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC3_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_3,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA3_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA3_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_3,	//
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC4_FLAG,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_4,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA4_MIN,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA4_MAX,
	E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_4,	//
};

class CInspectMura3
{
public:
	CInspectMura3(void);
	virtual ~CInspectMura3(void);

	CMatBuf* cMem;
	void		SetMem(CMatBuf* data) { cMem = data; };
	CMatBuf* GetMem() { return	cMem; };

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
	long		DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
		cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	long		DoFindMuraDefect2(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
		cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Mat& matDrawBuffer, wchar_t* strContourTxt);

	long		GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
		double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt);

	void		Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset = 0);

	void Filter8(BYTE* InImg, BYTE* OutImg, int nMin, int nMax, int width, int height);

	// 	long		GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	// 		double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt);

protected:
	long		LogicStart_SPOT(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath);

protected:
	long		ImageSave(CString strPath, cv::Mat matSrcBuf);
	bool		OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2);

	//行·hjf
	int CInspectMura3::GetBitFromImageDepth(int nDepth);
	float* CInspectMura3::diff2Gauss1D(int r);

	long CInspectMura3::RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI,
		long nLoop, long nSegX, long nSegY, float fDarkRatio, float fBrightRatio, float fDarkRatio_Edge, float fBrightRatio_Edge, int Defect_Color_mode, CMatBuf* cMemSub);

	//	void CInspectMura3::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset);

	bool CInspectMura3::cMeanFilte(cv::Mat matActiveImage, cv::Mat& matDstImage);

	//	long CInspectMura3::ImageSave(CString strPath, cv::Mat matSrcBuf);

	long CInspectMura3::JudgeWhiteSpot(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stDefectInfo* pResultBlob, CMatBuf* cMemSub);
	//////////////////////////////////////////////////////////////////////////

};
