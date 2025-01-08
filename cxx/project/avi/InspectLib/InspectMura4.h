
/************************************************************************/
// Mura
// 05.31
/************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include "InspectLibLog.h"
#include "MatBuf.h"

enum ENUM_PARA_AVI_MURA4_WHITE
{
	E_PARA_AVI_MURA4_DUST_TEXT,
	E_PARA_AVI_MURA4_DUST_BRIGHT_FLAG,							// 22.02.09
	E_PARA_AVI_MURA4_DUST_DARK_FLAG,							// 22.02.09
	E_PARA_AVI_MURA4_DUST_BIG_AREA,								// 22.02.09
	E_PARA_AVI_MURA4_DUST_ADJUST_RANGE,							// 22.02.09

	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_RESIZE,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIZE,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIGMA,

	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_RATIO,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_EDGE_RATIO,

	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_SEG_X,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_SEG_Y,

	E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_RATIO,
	E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_EDGE_RATIO,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_DARK_MINIMUM_SIZE,

	E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_CONTRAST,
	E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_DARK_INSPECTAREA,
	E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_RIGHT_EDGE_OFFSET,

	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_TEXT,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_FILTER_SIZE,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIZE,
	E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIGMA,
};

class CInspectMura4
{
public:
	CInspectMura4(void);
	virtual ~CInspectMura4(void);

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

protected:
	long		LogicStart_SPOT(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath);

protected:

	long		ImageSave(CString strPath, cv::Mat matSrcBuf);

	bool		OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2);

	////////////////////////////////////////////////////////////////////////// choi 05.01
	float* CInspectMura4::diff2Gauss1D(int r);

	long CInspectMura4::RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI,
		long nLoop, long nSegX, long nSegY, float fDarkRatio, float fBrightRatio, float fDarkRatio_Edge, float fBrightRatio_Edge, int Defect_Color_mode, CMatBuf* cMemSub);

	int CInspectMura4::GetBitFromImageDepth(int nDepth);


	//	void CInspectMura4::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset);

	bool CInspectMura4::cMeanFilte(cv::Mat matActiveImage, cv::Mat& matDstImage);

	//	long CInspectMura4::ImageSave(CString strPath, cv::Mat matSrcBuf);

	long CInspectMura4::JudgeWhiteSpot(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stDefectInfo* pResultBlob, CMatBuf* cMemSub);
	//////////////////////////////////////////////////////////////////////////
	long CInspectMura4::DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);
	long CInspectMura4::DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS);

};
