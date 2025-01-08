
/************************************************************************/
// Mura 
// Date : 18.05.31
/************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include "InspectLibLog.h"
#include "MatBuf.h"

// R, G, B, Gray, White 
enum ENUM_PARA_AVI_MURA_COMMON
{
	E_PARA_AVI_MURA_DUST_TEXT = 0,	//
	E_PARA_AVI_MURA_DUST_BRIGHT_FLAG,	//
	E_PARA_AVI_MURA_DUST_DARK_FLAG,	//
	E_PARA_AVI_MURA_DUST_BIG_AREA,	//
	E_PARA_AVI_MURA_DUST_ADJUST_RANGE,	//

	E_PARA_AVI_MURA_JUDGE_EDGE_NUGI_TEXT,	//
	E_PARA_AVI_MURA_JUDGE_EDGE_NUGI_USE,	//

	E_PARA_AVI_MURA_COMMON_TEXT,	//
	E_PARA_AVI_MURA_COMMON_GAUSSIAN_SIZE,	//
	E_PARA_AVI_MURA_COMMON_GAUSSIAN_SIGMA,	//
	E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_X,	//
	E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_Y,	//
	E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_X,	// 
	E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_Y,	//	
	E_PARA_AVI_MURA_COMMON_ESTIMATION_BRIGHT,	//
	E_PARA_AVI_MURA_COMMON_ESTIMATION_DARK,	//

	E_PARA_AVI_MURA_COMMON_TOTAL_COUNT							// Total
};

// R, G, B
enum ENUM_PARA_AVI_MURA_RGB
{
	E_PARA_AVI_MURA_RGB_INSPECT_FLAG_TEXT = E_PARA_AVI_MURA_COMMON_TOTAL_COUNT,
	E_PARA_AVI_MURA_RGB_INSPECT_DARK_FLAG,	//
	E_PARA_AVI_MURA_RGB_INSPECT_LINE_MURA_FLAG,	//

	E_PARA_AVI_MURA_RGB_EMD_DARK_TEXT,
	E_PARA_AVI_MURA_RGB_EMD_DARK_THRESHOLD,	//
	E_PARA_AVI_MURA_RGB_EMD_DARK_MORP,	//
	E_PARA_AVI_MURA_RGB_EMD_DARK_DEL_AREA,	//

	E_PARA_AVI_MURA_RGB_TEXT,
	E_PARA_AVI_MURA_RGB_RESIZE,	//
	E_PARA_AVI_MURA_RGB_GAUSSIAN_SIZE,	//
	E_PARA_AVI_MURA_RGB_GAUSSIAN_SIGMA,	//
	E_PARA_AVI_MURA_RGB_DARK_EDGE_AREA,	//
	E_PARA_AVI_MURA_RGB_DARK_EDGE_THRESHOLD,	//
	E_PARA_AVI_MURA_RGB_DARK_ACTIVE_THRESHOLD,	//
	E_PARA_AVI_MURA_RGB_DARK_MORP,	//
	E_PARA_AVI_MURA_RGB_POW,	//
	E_PARA_AVI_MURA_RGB_BLUR_X,	//
	E_PARA_AVI_MURA_RGB_BLUR_Y1,	//
	E_PARA_AVI_MURA_RGB_BLUR_Y2,	//
	E_PARA_AVI_MURA_RGB_EDGE_AREA,	//
	E_PARA_AVI_MURA_RGB_THRESHOLD,	//
	E_PARA_AVI_MURA_RGB_EDGE_THRESHOLD,	//
	E_PARA_AVI_MURA_RGB_AREA,	//
	E_PARA_AVI_MURA_RGB_INSIDE,	//
	E_PARA_AVI_MURA_RGB_AREA_SEG_X,	//
	E_PARA_AVI_MURA_RGB_AREA_SEG_Y,	//
	E_PARA_AVI_MURA_RGB_AREA_1_RATIO,	//
	E_PARA_AVI_MURA_RGB_AREA_2_COUNT,	//
	E_PARA_AVI_MURA_RGB_AREA_2_RATIO,	//
	E_PARA_AVI_MURA_RGB_AREA_MIN_GV,	//

	E_PARA_AVI_MURA_RGB_TOTAL_COUNT								// Total
};

// Gray, White
enum ENUM_PARA_AVI_MURA_GRAY
{

	E_PARA_AVI_MURA_GRAY_INSPECT_FLAG_TEXT = E_PARA_AVI_MURA_COMMON_TOTAL_COUNT,
	E_PARA_AVI_MURA_GRAY_INSPECT_BRIGHT_FLAG,
	E_PARA_AVI_MURA_GRAY_INSPECT_DARK_FLAG,
	E_PARA_AVI_MURA_GRAY_INSPECT_MID_BRIGHT_FLAG,

	E_PARA_AVI_MURA_GRAY_BRIGHT_TEXT,
	E_PARA_AVI_MURA_GRAY_BRIGHT_THRESHOLD_WHITE_MURA_EDGE_AREA,
	E_PARA_AVI_MURA_GRAY_BRIGHT_THRESHOLD_WHITE_MURA_ACTIVE,
	E_PARA_AVI_MURA_GRAY_BRIGHT_THRESHOLD_WHITE_MURA_EDGE,
	E_PARA_AVI_MURA_GRAY_BRIGHT_MORP,
	E_PARA_AVI_MURA_GRAY_BRIGHT_DEL_AREA,

	E_PARA_AVI_MURA_GRAY_DARK_TEXT,
	E_PARA_AVI_MURA_GRAY_DARK_THRESHOLD,
	E_PARA_AVI_MURA_GRAY_DARK_MORP,
	E_PARA_AVI_MURA_GRAY_DARK_DEL_AREA,

	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_TEXT,
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_ADJUST1_MIN_GV,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_RESIZE_LOOP_CNT,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_CONTRAST_VALUE,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_ADJUST2_MUTI_VALUE,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_CANNY_MIN,	//
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_CANNY_MAX,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_EDGE_DEL_LOOP,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_EDGE_DEL_MORP_SIZE,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_DEL_AREA,	// 
	E_PARA_AVI_MURA_GRAY_MID_BRIGHT_DEFECT_MORP_RESIZE,	// 

	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_TEXT,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_MORP_OBJ,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_MORP_BKG,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_THRESHOLD,	//

	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_TEXT,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_BRIGHT_RATIO,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_3,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA_3,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_3,	//

	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_TEXT,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_LEFT,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_TOP,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_RIGHT,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_BOTTOM,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_BRIGHT_RATIO,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_3,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA_3,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_3,	//

	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_TEXT,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_FLAG,
	//E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_BRIGHT_RATIO,	
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC1_FLAG,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_1,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA1_MIN,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA1_MAX,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_1,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC2_FLAG,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_2,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA2_MIN,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA2_MAX,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_2,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC3_FLAG,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_3,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA3_MIN,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA3_MAX,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_3,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC4_FLAG,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_4,	// 
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA4_MIN,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA4_MAX,
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_4,	// 

	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_TEXT,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_LEFT,	// Edge
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_TOP,	// Edge
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_RIGHT,	// Edge
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_BOTTOM,	// Edge
	//E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_BRIGHT_RATIO,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC1_FLAG, //
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA1_MIN,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA1_MAX,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_1,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC2_FLAG, //
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA2_MIN,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA2_MAX,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_2,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC3_FLAG, //
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_3,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA3_MIN,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA3_MAX,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_3,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC4_FLAG, //
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_4,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA4_MIN,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA4_MAX,	//
	E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_4,	//

	E_PARA_AVI_MURA_TOTAL_COUNT											// Total
};

// Gray, White
enum ENUM_PARA_AVI_RING_MURA
{
	E_PARA_AVI_MURA_RING_TEXT = E_PARA_AVI_MURA_TOTAL_COUNT,
	E_PARA_AVI_MURA_RING_GAUSSIAN_SIZE,
	E_PARA_AVI_MURA_RING_GAUSSIAN_SIGMA,

	E_PARA_AVI_MURA_RING_IMAGE_RESIZE,

	E_PARA_AVI_MURA_RING_CONTRAST_OFFSET,

	E_PARA_AVI_MURA_RING_THRESHOLD_BRIGHT,
	E_PARA_AVI_MURA_RING_THRESHOLD_DARK,

	E_PARA_AVI_MURA_RING_MORPHOLOGY_SIZE,

	E_PARA_AVI_MURA_RING_DELAREA_BRIGHT,
	E_PARA_AVI_MURA_RING_DELAREA_DARK,
	E_PARA_AVI_MURA_RING_DELAREA_SMALLOFFSET,
};

// Gray3(128)
enum ENUM_PARA_AVI_MURA_G3
{
	E_PARA_AVI_MURA_G3_MURA_INSP = E_PARA_AVI_MURA_TOTAL_COUNT,
	E_PARA_AVI_MURA_G3_TEXT,

	E_PARA_AVI_MURA_G3_PREPROCESS_SHIFTCOPY,
	E_PARA_AVI_MURA_G3_PREPROCESS_RESIZEUNIT,
	E_PARA_AVI_MURA_G3_PREPROCESS_LIMITLENGTH,
	E_PARA_AVI_MURA_G3_PREPROCESS_BLUR_L01,
	E_PARA_AVI_MURA_G3_PREPROCESS_BLUR_L02,
	E_PARA_AVI_MURA_G3_PREPROCESS_BLUR_L03,
	E_PARA_AVI_MURA_G3_MAIN_LOWAREA_TH,
	E_PARA_AVI_MURA_G3_MAIN_TOPAREA_TH,
	E_PARA_AVI_MURA_G3_JUDGE_DIFF_GV,

	E_PARA_AVI_MURA_G3_CM2_TEXT,
	E_PARA_AVI_MURA_G3_CM2_ZOOM,
	E_PARA_AVI_MURA_G3_CM2_STDDEV_BRIGHT,
	E_PARA_AVI_MURA_G3_CM2_STDDEV_DARK,

	E_PARA_AVI_MURA_G3_CM3_TEXT,
	E_PARA_AVI_MURA_G3_CM3_ZOOM,
	E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIZE,
	E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIGMA,
	E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIZE2,
	E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIGMA2,
	E_PARA_AVI_MURA_G3_CM3_ESTIMATION_DIM_X,	//
	E_PARA_AVI_MURA_G3_CM3_ESTIMATION_DIM_Y,	//
	E_PARA_AVI_MURA_G3_CM3_ESTIMATION_STEP_X,	//
	E_PARA_AVI_MURA_G3_CM3_ESTIMATION_STEP_Y,	//
	E_PARA_AVI_MURA_G3_CM3_ESTIMATION_BRIGHT,	//
	E_PARA_AVI_MURA_G3_CM3_ESTIMATION_DARK,		//
	E_PARA_AVI_MURA_G3_CM3_THRESHOLD_DARK,
	E_PARA_AVI_MURA_G3_CM3_THRESHOLD_BRIGHT,

	E_PARA_AVI_MURA_G3_CM4_TEXT,
	E_PARA_AVI_MURA_G3_CM4_ZOOM,
	E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIZE,
	E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIGMA,
	E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIZE2,
	E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIGMA2,
	/*E_PARA_AVI_MURA_G3_CM4_EDGE_TB_AREA,
	E_PARA_AVI_MURA_G3_CM4_EDGE_LR_AREA,*/
	E_PARA_AVI_MURA_G3_CM4_EDGE_T_AREA,			//
	E_PARA_AVI_MURA_G3_CM4_EDGE_B_AREA,			//
	E_PARA_AVI_MURA_G3_CM4_EDGE_L_AREA,			//
	E_PARA_AVI_MURA_G3_CM4_EDGE_R_AREA,			//

	E_PARA_AVI_MURA_G3_CM4_CONTRAST_OFFSET,
	E_PARA_AVI_MURA_G3_CM4_CONTRAST_MINIMUM,	//
	E_PARA_AVI_MURA_G3_CM4_CONTRAST_MAXIMUM,	//

	E_PARA_AVI_MURA_G3_CM4_ESTIMATION_DIM_X,	//
	E_PARA_AVI_MURA_G3_CM4_ESTIMATION_DIM_Y,	//
	E_PARA_AVI_MURA_G3_CM4_ESTIMATION_STEP_X,	//
	E_PARA_AVI_MURA_G3_CM4_ESTIMATION_STEP_Y,	//
	E_PARA_AVI_MURA_G3_CM4_ESTIMATION_BRIGHT,	//
	E_PARA_AVI_MURA_G3_CM4_ESTIMATION_DARK,		//
	E_PARA_AVI_MURA_G3_CM4_THRESHOLD_DARK,

	E_PARA_AVI_MURA_G3_TOTAL_COUNT
};

class CInspectMura
{
public:
	CInspectMura(void);
	virtual ~CInspectMura(void);

	CMatBuf* cMem;
	void		SetMem(CMatBuf* data) { cMem = data; };
	CMatBuf* GetMem() { return	cMem; };

	int sz;
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

	long		GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
		double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt);

protected:
	long		LogicStart_SPOT(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath);


	// 2022.04.25
	long		LogicStart_RingMura(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat& matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath);

	long		LogicStart_RGB_LINE_MURA(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	/////////////////////////////////////////////////////////////////////////
	long        LogicStart_MuraG3CM(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara,
		CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool* bFlag);

	//2022.02.28 
	long        LogicStart_MuraG3CM2(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara,
		CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	//2022.04.02 
	long        LogicStart_MuraG3CM3(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, cv::Mat& matDst_Dark, cv::Mat& matDst_Bright, CRect rectROI, double* dPara, int* nCommonPara,
		CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	//2022.04.19 
	long        LogicStart_MuraG3CM4(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, cv::Mat& matDstImage, CRect rectROI, double* dPara, int* nCommonPara,
		CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

	void Flattening(int nFlatteningType, BYTE* pImage, CSize szImage, int nMeanGV = 128);
	void FlattenMeanHorizontal(BYTE* pImage, CSize szImage, int nMeanGV);
	void FlattenMeanVertical(BYTE* pImage, CSize szImage, int nMeanGV);
	void FlattenFillLowGV(BYTE* pImage, CSize szImage, BYTE* LowGVImage);

	//////////////////////////////////////////////////////////////////////////

protected:

	long		ImageSave(CString strPath, cv::Mat matSrcBuf);

	long		DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);

	long		DeleteArea2(cv::Mat& matSrcImage, int nCount, int nLength, CMatBuf* cMemSub = NULL);

	long		DeleteArea3(cv::Mat& matSrcImage, int nCount, int nLength, CMatBuf* cMemSub = NULL);

	long		DeleteArea1_Re(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);
	long		Estimation_X(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, /*double* dPara*/int nDimensionX, int nEstiStepX, double dEstiBright, double dEstiDark);

	long		Estimation_Y(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, /*double* dPara*/int nDimensionY, int nEstiStepY, double dEstiBright, double dEstiDark);

	long		Estimation_Y_N_Average(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf, cv::Mat& matDstBuf, double* dPara);

	long		Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, CMatBuf* cMemSub);

	long		Estimation_XY2(cv::Mat& matSrcBuf, cv::Mat& matDstBuf,/* double* dPara,*/ int nEstiDimX, int nEstiDimY, int nEstiStepX, int nEstiStepY, double dEstiBright, double dEstiDark, CMatBuf* cMemSub);

	long		DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS);

	long		DeleteDarkLine(cv::Mat& matSrcBuffer, float fMajorAxisRatio, CMatBuf* cMemSub);

	long		LimitMaxGV16X(cv::Mat& matSrcBuffer, float fOffset);

	// RGB Line Mura
	long		LimitArea(cv::Mat& matSrcBuffer, double* dPara, CMatBuf* cMemSub);

	// RGB Line Mura
	long		JudgeRGBLineMura(cv::Mat& matSrcBuffer, cv::Mat& matBKBuf16, double* dPara, int* nCommonPara, CRect rectROI, stDefectInfo* pResultBlob, CMatBuf* cMemSub);

	// RGB Line Mura
	long		JudgeRGBLineMuraSave(cv::Mat& matSrcBuffer, cv::Mat& matBKBuf16, double* dPara, int* nCommonPara, CRect rectROI, stDefectInfo* pResultBlob, CString strAlgPath, CMatBuf* cMemSub);

	// RGB Line Mura
	long		AddRGBLineMuraDefect(cv::Mat& matContoursBuf, double* dPara, int* nCommonPara, CRect rectROI, stDefectInfo* pResultBlob);

	// Spot
	long		JudgeWhiteSpot(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
		stDefectInfo* pResultBlob, CMatBuf* cMemSub);

	// White Mura
	long		JudgeWhiteMura(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
		stDefectInfo* pResultBlob, CMatBuf* cMemSub);

	// Nugi
	long		JudgeNugi(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
		stDefectInfo* pResultBlob, CMatBuf* cMemSub);

	bool		OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2);


	////////////////////////////////////////////////////////////////////////// G3
	long  ShiftCopyParaCheck(int ShiftValue, int& nCpyX, int& nCpyY, int& nLoopX, int& nLoopY);
	long  AveragingReducer(cv::Mat& matSrcImage, cv::Mat& matDstImage, CMatBuf* cMemSub);
	long  HistAreaCalc(cv::Mat& matSrcImage, int& nLowerIndex, int& nUpperIndex, int& nLowUpDiff, int& nLowValueArea, int& nTopValueArea, double* dPara);

	void FunFilter(cv::Mat& Intput, cv::Mat& Output, int width, int height);
	void FunWhiteMura(cv::Mat& Intput, cv::Mat& Output, int width, int height, int nThres);
	void FunBlackMura(cv::Mat& Intput, cv::Mat& Output, int width, int height, int nThres);
	void FunImageResize(cv::Mat& Intput, long* lResizeBuff, int widthnew, int heightnew, int width, int height, int m);

	void DarkInBright(cv::Mat& Input_Bright, cv::Mat& Input_Dark, cv::Mat& Output);
	//////////////////////////////////////////////////////////////////////////
};
