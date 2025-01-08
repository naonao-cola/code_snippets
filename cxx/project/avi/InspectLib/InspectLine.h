
/************************************************************************
Line算法相关标题
************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include"MatBuf.h"				//内存管理
#include "InspectLibLog.h"

//////////////////////////////////////////////////////////////////////////

enum ENUM_PARA_LINE_RGB
{
	E_PARA_LINE_RGB_EMPTY_1 = 0,	//空格
	E_PARA_LINE_RGB_EMPTY_2,	//空格
	E_PARA_LINE_RGB_SHIFT_FLAG,	//禁用
	E_PARA_LINE_RGB_SHIFT_X,	//禁用
	E_PARA_LINE_RGB_SHIFT_Y,	//禁用
	E_PARA_LINE_RGB_TARGET_GV,	//满足所需亮度
	E_PARA_LINE_RGB_BLUR_SIZE,	//消除噪音
	E_PARA_LINE_RGB_MEAN_FILTER_SIZE,	//方向性Blur
	E_PARA_LINE_RGB_BG_FILTER_SIZE,	//反向创建背景画面(大尺寸)
	E_PARA_LINE_RGB_IMAGE_FILTER_SIZE,	//方向相反,尺寸稍小(消除噪音)
	E_PARA_LINE_RGB_SEG_X,	//设置二进制区域
	E_PARA_LINE_RGB_SEG_Y,	//设置二进制区域
	E_PARA_LINE_RGB_OUTLINE,	//删除轮廓(仅暗线...)
	E_PARA_LINE_RGB_DARK_RATIO_X,	//比平均值暗
	E_PARA_LINE_RGB_BRIGHT_RATIO_X,	//平均对比亮度
	E_PARA_LINE_RGB_DARK_RATIO_Y,	//禁用
	E_PARA_LINE_RGB_BRIGHT_RATIO_Y,	//禁用
	E_PARA_LINE_RGB_DELETE_AREA,	//删除面积小的
	E_PARA_LINE_RGB_MORP_OPEN,	//只留下线路不良
	E_PARA_LINE_RGB_MORP_CLOSE,	//线路不良连接
	E_PARA_LINE_RGB_MORP_TEMP,	//膨胀,消除钢丝周围过检不良

	E_PARA_LINE_RGB_WEAK_TEXT,	//约线参数
	E_PARA_LINE_RGB_WEAK_FLAG,	//是否使用约线算法
	E_PARA_LINE_RGB_WEAK_RESIZE,	//为Gaussian Blur缩小画面
	E_PARA_LINE_RGB_WEAK_BLUR,	//禁用
	E_PARA_LINE_RGB_WEAK_TARGET_GV2,	//已禁用
	E_PARA_LINE_RGB_WEAK_OUTLINEBRIGHT,	//BRIGHT未检查外围
	E_PARA_LINE_RGB_WEAK_OUTLINEDARK,	//DARK未检查外围
	E_PARA_LINE_RGB_WEAK_TARGET_GV,	//画面平均GV
	E_PARA_LINE_RGB_WEAK_BRIGHT_RATIO_X,	//平均对比亮度X方向
	E_PARA_LINE_RGB_WEAK_DARK_RATIO_X,	//平均对比暗X方向
	E_PARA_LINE_RGB_WEAK_BRIGHT_RATIO_Y,	//平均对比亮度Y方向
	E_PARA_LINE_RGB_WEAK_DARK_RATIO_Y,	//平均对比暗Y方向
	E_PARA_LINE_RGB_WEAK_GAUSSIAN,	//删除Noise
	E_PARA_LINE_RGB_WEAK_PROJ_BLUR1,	//用于背景设置的Projection Blur1
	E_PARA_LINE_RGB_WEAK_PROJ_BLUR2,	//用于背景设置的Projection Blur2
	E_PARA_LINE_RGB_WEAK_PROJ_RANGE,	//设置背景的投影Max-Min范围
	E_PARA_LINE_RGB_WEAK_PROJ_MORPHOLOGY,	//Morphology Size大小
	E_PARA_LINE_RGB_WEAK_PROJ_SIGAM,	// Gaussian blur Sigma

	E_PARA_LINE_RGB_RANSAC_PROCESS_TEXT,	// Text
	E_PARA_LINE_RGB_RANSAC_MAXFILTER_SIZE,	// Max Filter Process Size
	E_PARA_LINE_RGB_RANSAC_AVG_TH_RATIO,	// Max Filter Average Ratio
	E_PARA_LINE_RGB_RANSAC_TH_UNIT,	// RANSAC Process Unit

	E_PARA_LINE_RGB_NORCH_PROCESS_TEXT,	// Text
	E_PARA_LINE_RGB_NORCH_ONOFF,	// On : 1 / Off : 0
	E_PARA_LINE_RGB_NORCH_MAXFILTER_SIZE,//Norch Process中的Max Filter Size
	E_PARA_LINE_RGB_NORCH_AVGFILTER_SIZE,//Norch Process中的Avg Filter Size
	E_PARA_LINE_RGB_NORCH_AVG_TH_RATIO,	// Norch Average Ratio
	E_PARA_LINE_RGB_NORCH_MIN_GV,	// Norch Min Data GV
	E_PARA_LINE_RGB_NORCH_INSP_RATIO_BRIGHT,	// Insp. Ratio Bright
	E_PARA_LINE_RGB_NORCH_INSP_RATIO_DARK,	// Insp. Ratio Dark

	E_PARA_LINE_RGB_TOTAL_COUNT						// Total
};

enum ENUM_PARA_LINE_GRAY
{
	E_PARA_LINE_GRAY_EMPTY_1 = 0,	//空格
	E_PARA_LINE_GRAY_EMPTY_2,	//空格
	E_PARA_LINE_GRAY_SHIFT_FLAG,	//禁用
	E_PARA_LINE_GRAY_SHIFT_X,	//已禁用
	E_PARA_LINE_GRAY_SHIFT_Y,	//已禁用
	E_PARA_LINE_GRAY_TARGET_GV,	//满足所需亮度
	E_PARA_LINE_GRAY_BLUR_SIZE,	//消除噪音
	E_PARA_LINE_GRAY_MEAN_FILTER_SIZE,	//方向性Blur
	E_PARA_LINE_GRAY_BG_FILTER_SIZE,	//反向创建背景画面(尺寸大)
	E_PARA_LINE_GRAY_IMAGE_FILTER_SIZE,	//方向相反,尺寸稍小(除噪)
	E_PARA_LINE_GRAY_SEG_X,	//设置二进制区域
	E_PARA_LINE_GRAY_SEG_Y,	//设置二进制区域
	E_PARA_LINE_GRAY_OUTLINE,	//删除轮廓(仅暗线...)
	E_PARA_LINE_GRAY_DARK_RATIO_X,	//比平均值暗
	E_PARA_LINE_GRAY_BRIGHT_RATIO_X,	//平均对比亮度
	E_PARA_LINE_GRAY_DARK_RATIO_Y,	//已禁用
	E_PARA_LINE_GRAY_BRIGHT_RATIO_Y,	//已禁用
	E_PARA_LINE_GRAY_DELETE_AREA,	//删除面积小的
	E_PARA_LINE_GRAY_MORP_OPEN,	//只留下线路不良
	E_PARA_LINE_GRAY_MORP_CLOSE,	//连接线路不良
	E_PARA_LINE_GRAY_MORP_TEMP,	//膨胀消除钢丝周围过检不良

	E_PARA_LINE_GRAY_WEAK_TEXT,	//约线参数
	E_PARA_LINE_GRAY_WEAK_FLAG,	//是否使用约线算法
	E_PARA_LINE_GRAY_WEAK_RESIZE,	//为Gaussian Blur缩小画面
	E_PARA_LINE_GRAY_WEAK_BLUR,	//已禁用
	E_PARA_LINE_GRAY_WEAK_TARGET_GV2,	//已禁用
	E_PARA_LINE_GRAY_WEAK_OUTLINEBRIGHT,	//BRIGHT未检查外围
	E_PARA_LINE_GRAY_WEAK_OUTLINEDARK,	//DARK未检查外围
	E_PARA_LINE_GRAY_WEAK_TARGET_GV,	//画面平均GV
	E_PARA_LINE_GRAY_WEAK_BRIGHT_RATIO_X,	//平均对比亮度X方向
	E_PARA_LINE_GRAY_WEAK_DARK_RATIO_X,	//平均对比暗X方向
	E_PARA_LINE_GRAY_WEAK_BRIGHT_RATIO_Y,	//平均对比亮度Y方向
	E_PARA_LINE_GRAY_WEAK_DARK_RATIO_Y,	//平均对比暗Y方向
	E_PARA_LINE_GRAY_WEAK_GAUSSIAN,	//删除Noise
	E_PARA_LINE_GRAY_WEAK_PROJ_BLUR1,	//用于背景设置的Projection Blur1
	E_PARA_LINE_GRAY_WEAK_PROJ_BLUR2,	//用于背景设置的Projection Blur2
	E_PARA_LINE_GRAY_WEAK_PROJ_RANGE,	//设置背景的投影Max-Min范围
	E_PARA_LINE_GRAY_WEAK_PROJ_MORPHOLOGY,	//Morphology Size大小
	E_PARA_LINE_GRAY_WEAK_PROJ_SIGAM,	// Gaussian blur Sigma

	E_PARA_LINE_GRAY_TOTAL_COUNT					// Total
};

enum ENUM_PARA_LINE_BLACK
{
	E_PARA_LINE_BLACK_WINDOW_SIZE = 0,	//高斯滤镜尺寸
	E_PARA_LINE_BLACK_SIGMA,	//高斯信号
	E_PARA_LINE_BLACK_RESIZE,//将画面缩小到	//1/RESIZE倍
	E_PARA_LINE_BLACK_THRESHOLD_RATIO,	//标准偏差*RATIO
	E_PARA_LINE_BLACK_OUTLINE,	//删除外围
	E_PARA_LINE_BLACK_ROTATION_FLAG,	//旋转开/关
	E_PARA_LINE_BLACK_OFFSET,	//投影Sub的范围
	E_PARA_LINE_BLACK_THRESHOLD_XY,	//X线向Y方向投影->二进制数值
	E_PARA_LINE_BLACK_THRESHOLD_XX,	//X线向X方向投影->二进制数值
	E_PARA_LINE_BLACK_THRESHOLD_YX,	//将Y线投影到X方向->二进制数值
	E_PARA_LINE_BLACK_THRESHOLD_YY,	//Y线朝Y方向投影->二进制数值
	E_PARA_LINE_BLACK_THICKNESS,	//用于接线(当前固定1)
	E_PARA_LINE_BLACK_PIXEL_DISTANCE,	//距离直线的最大距离(Pixel)

	E_PARA_LINE_BLACK_TOTAL_COUNT				// Total
};

//////////////////////////////////////////////////////////////////////////

enum ENUM_MORP2
{
	EP_MORP_INIT = 0,
	EP_MORP_ERODE,
	EP_MORP_DILATE,
	EP_MORP_OPEN,
	EP_MORP_CLOSE,
	EP_MORP_GRADIENT,
	EP_MORP_TOPHAT,
	EP_MORP_BLACKHAT,
};

enum ENUM_PROFILE
{
	E_PROFILE_ROW = 0,
	E_PROFILE_COL
};

enum ENUM_WEAK_FILTER_DIRECTION
{
	E_WEAK_FILTER_X = 0,
	E_WEAK_FILTER_Y
};

enum ENUM_WEAK_DEFECT_TYPE
{
	E_WEAK_DEFECT_BRIGHT = 0,
	E_WEAK_DEFECT_DARK
};

enum ENUM_BINARY_IMAGE
{
	E_BINARY_BRIGHT_X = 0,
	E_BINARY_BRIGHT_Y,
	E_BINARY_DARK_X,
	E_BINARY_DARK_Y,

	E_BINARY_TOTAL_COUNT
};

enum ENUM_BIT_TYPE
{
	E_BIT_8_IMAGE = 0,
	E_BIT_16_IMAGE
};

#define _MIN_(a,b) ((a) < (b) ? (a) : (b)) 
#define _MAX_(a,b) ((a) > (b) ? (a) : (b)) 
#define DISTANCE1(x1, x2)			fabs(x2-x1)
#define DISTANCE2(x1, y1, x2, y2)	sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

//////////////////////////////////////////////////////////////////////////

class CInspectLine
{
public:
	CInspectLine(void);
	virtual ~CInspectLine(void);

	//内存管理
	CMatBuf* cMem;
	void		SetMem(CMatBuf* data) { cMem = data; };
	CMatBuf* GetMem() { return	cMem; };

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

	//////////////////////////////////////////////////////////////////////////

	void		writeInspectLog_Memory(int nAlgType, char* strFunc, wchar_t* strTxt, __int64 nMemory_Use_Value = 0)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, nMemory_Use_Value, m_strAlgLog);
	};

	///////////////////////////////////////////////////////////////////

			//Main检查算法
	long		FindLineDefect(cv::Mat matSrcBuffer, cv::Mat& matDrawBuffer, cv::Mat& matBKBuffer, vector<int> NorchIndex, CPoint OrgIndex, cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath,
		stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, wchar_t* strContourTxt = NULL);

protected:

	long CInspectLine::LogicStart_BlackWhite3(Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner);

	long CInspectLine::LogicStart_Crack(Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner);

	//RGB模式检测算法
	long CInspectLine::LogicStart_RGB5(Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer);

protected:
	int GetBitFromImageDepth(int nDepth);

	long RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, long nLoop, long nSegX, long nSegY, float fDarkRatio, float fBrightRatio, float fDarkRatio_Edge, float fBrightRatio_Edge, CMatBuf* cMemSub = NULL);

	void LineMeasurement(cv::Mat matSrcImage, cv::Mat& matDstImage, Mat* matProjection, double* dPara, CRect rectROI, int nDirection, int nOutLine, CMatBuf* cMemSub = NULL);

	void Sdtthreshold(cv::Mat matSrcImage, cv::Mat& matDstImage, float fThresholdRatio);

	void CInspectLine::filter8(BYTE* pbyInImg, BYTE* pbyOutImg, int nMin, int nMax, int nWidth, int nHeight); //choikwangil

	//////////////////////////////////////////////////////////////////////////
		//17.09.29-按方向排列
	long	calcLine_BrightX(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath);
	long	calcLine_BrightY(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath);
	long	calcLine_DarkX(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath);
	long	calcLine_DarkY(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath);

	// 17.10.02
	long	calcRGBMain(cv::Mat& matSrcImage, cv::Mat& matThImage, cv::Mat* matDstImage, cv::Mat* matBinaryMorp, double* dPara, int* nCommonPara,
		CRect rectResize, cv::Point* ptCorner, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, CMatBuf* cMemSub = NULL);

	//////////////////////////////////////////////////////////////////////////

	long	LogicStart_Weak(cv::Mat& matSrcImage, cv::Mat* matDstImage, vector<int> NorchIndex, CPoint OrgIndex, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner);

	///////////////////////////////////////////////////////////////////////////
	long	resizeGaussian(Mat& InPutImage, Mat& OutPutImage, int reSizeValue, int MaskSize, double dSigma, CMatBuf* cMemSub = NULL);

	long	calcWeakLine_Y(Mat& InPutImage, Mat& OutPutImage1, Mat& OutPutImage2, Mat* matProjectionY, vector<int> NorchIndex, CPoint OrgIndex, CRect rectROI, double dbThresholdBY, double dbThresholdDY, int nBlur, int nBlur2, int nRange, int nMorp, int nOutLineBright, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara);

	long	calcWeakLine_X(Mat& InPutImage, Mat& OutPutImage1, Mat& OutPutImage2, Mat* matProjectionX, CRect rectROI, double dbThresholdBY, double dbThresholdDY, int nBlur, int nBlur2, int nRange, int nMorp, int nOutLineBright, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara);

	long	calcProjection(Mat& MatproSrc, Mat& MatproYDst, Mat* matProjection, int size, int nBlur, int nBlur2, int nRange, int nMorp, int Type, int* nCommonPara, CString strAlgPath);

	long	calcWeakLine_BrigtY(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionY, Mat& OutPutImage, CRect rectROI, int nNorchUnit, int size, double dbThresholdBY, int nOutLineBright, int* nCommonPara, CString strAlgPath, double* dPara);

	long	calcWeakLine_DarkY(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionY, Mat& OutPutImage, CRect rectROI, int nNorchUnit, int size, double dbThresholdDY, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara);

	long	calcWeakLine_BrigtX(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionX, Mat& OutPutImage, CRect rectROI, int size, double dbThresholdBX, int nOutLineBright, int* nCommonPara, CString strAlgPath, double* dPara);

	long	calcWeakLine_DarkX(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionX, Mat& OutPutImage, CRect rectROI, int size, double dbThresholdDX, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara);

	long	RangeThreshold_Weak(Mat& MatproSub, Mat& OutPutImage1, CRect rectROI, int size, double dbThreshold, int nOutLine, int Type, int* nCommonPara, CString strAlgPath);

	///////////////////////////////////////////////////////////////////////////
	// RANSAC Process

	long	RangRANSACProcess(Mat& MatproSub, Mat& MatSubRANSAC, int size, int* nCommonPara, CString strAlgPath, double* dPara);

	long	ProfileMaxFilter(Mat& MatproSub, Mat& MatSubRANSAC, int size, int* nCommonPara, CString strAlgPath, double* dPara, int nOutLine);

	long	NorchValueProcess(Mat& MatproSub, Mat& MatNorchMaxADD, int size, int* nCommonPara, CString strAlgPath, double* dPara, int nOutLine, int nNorchUnit);

	long	RRM_Thresholding(Mat& MatproSub, Mat& MatSubRANSAC, Mat& OutPutImage, CRect rectROI, int nNorchUnit, int size, double dbThreshold, int nOutLine, int Type, int* nCommonPara, CString strAlgPath, double* dPara, double dInspRatio);

	long	ImageSave(CString strPath, cv::Mat matSrcBuf);
	//////////////////////////////////////////////////////////////////////////
	// 
	void		Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset = 0);

protected:

	// preprocess
	long Morphology(Mat& matSrcImage, Mat& matDstImage, long nSizeX, long nSizeY, int nOperation, CMatBuf* cMemSub = NULL, int nIter = 1);

	bool m_bProcess;
};