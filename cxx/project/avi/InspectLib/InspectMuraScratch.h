#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include "InspectLibLog.h"
#include "MatBuf.h"				// ������������ ��u����

enum ENUM_PARA_AVI_MURA_SCRATCH
{
	E_PARA_AVI_MURA_SCRATCH_TEXT,	// A���������� C�ͨ�A
	E_PARA_AVI_MURA_SCRATCH_GAUSSIAN_SIZE,	//
	E_PARA_AVI_MURA_SCRATCH_GAUSSIAN_SIGMA,
	E_PARA_AVI_MURA_SCRATCH_RESIZE_LOOP_COUNT,	//

	E_PARA_AVI_MURA_SCRATCH_CONTRAST_TEXT,	// A���������� C�ͨ�A
	E_PARA_AVI_MURA_SCRATCH_ADJUST1_VALUE,	// 
	E_PARA_AVI_MURA_SCRATCH_CONTRAST_VALUE,	// 

	E_PARA_AVI_MURA_SCRATCH_RANGETH_TEXT,	// A���������� C�ͨ�A
	E_PARA_AVI_MURA_SCRATCH_SEG_X,
	E_PARA_AVI_MURA_SCRATCH_SEG_Y,
	E_PARA_AVI_MURA_SCRATCH_ACTIVE_BRIGHT_RATIO,
	E_PARA_AVI_MURA_SCRATCH_EDGE_BRIGHT_RATIO,

	E_PARA_AVI_MURA_SCRATCH_HOUGH_TEXT,	// A���������� C�ͨ�A
	E_PARA_AVI_MURA_SCRATCH_HOUGH_TH,
	E_PARA_AVI_MURA_SCRATCH_HOUGH_MIN_LINE_LENGHT,
	E_PARA_AVI_MURA_SCRATCH_HOUGH_MAX_LINE_GAP,

	E_PARA_AVI_MURA_SCRATCH_DEGREE_TEXT,	// A���������� C�ͨ�A
	E_PARA_AVI_MURA_SCRATCH_DEGREE_MIN,
	E_PARA_AVI_MURA_SCRATCH_DEGREE_MAX,

	//E_PARA_AVI_MURA_SCRATCH_MORP_TEXT,	// A���������� C�ͨ�A
	//E_PARA_AVI_MURA_SCRATCH_MORP_DLIATE,	// A���������� C�ͨ�A

	E_PARA_AVI_MURA_SCRATCH_TOTAL_COUNT							// Total
};

class CInspectMuraScratch
{
public:
	CInspectMuraScratch(void);
	virtual ~CInspectMuraScratch(void);

	CMatBuf* cMem;
	void		SetMem(CMatBuf* data) { cMem = data; };
	CMatBuf* GetMem() { return	cMem; };

	// ��I���� ��u����
	InspectLibLog* m_cInspectLibLog;
	clock_t				m_tInitTime;
	clock_t				m_tBeforeTime;
	wchar_t* m_strAlgLog;

	//private:
	//static int sz;				 //choi 05.08

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

	long DoFindMuraDefect_Scratch(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
		cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Mat& matDrawBuffer, wchar_t* strContourTxt);

	long LogicStart_Scratch(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath);

	//////////////////////////////////////////////////////////////////////////

	long ImageSave(CString strPath, cv::Mat matSrcBuf);

	long GetAverage_From_Histo(cv::Mat matHisto, int nMin, int nMax, double& dblAverage);

	long GetMeanStdDev_From_Histo(cv::Mat matHisto, int nMin, int nMax, double& dblAverage, double& dblStdev);

	int GetBitFromImageDepth(int nDepth);

	long GetHistogram_th(cv::Mat& matSrcImage, cv::Mat& matHisto);

	long RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI,
		long nLoop, long nSegX, long nSegY, float fDarkRatio, float fBrightRatio, float fDarkRatio_Edge, float fBrightRatio_Edge, int Defect_Color_mode, CMatBuf* cMemSub);

	long FFT(cv::Mat& matSrcImage);
	void getZeroPaddedImage(Mat& src, Mat& dst);
	void getLogMsg(Mat planes[], Mat& magI, Mat& angI);
	void rearrange(Mat magI);
	void myFilter(Mat& magI);
};

