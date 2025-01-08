
/************************************************************************/
//Point不良检测相关标头
//修改日期:17.03.08
/************************************************************************/

#pragma once

#include "Define.h"
#include "FeatureExtraction.h"
#include "DefectCCD.h"
#include"MatBuf.h"				//内存管理
#include "InspectLibLog.h"

enum ENUM_PARA_POINT_RGB
{
	E_PARA_POINT_RGB_OUTLINE_TEXT = 0,	//显示文本
	E_PARA_POINT_RGB_DELETE_PIXEL,	//清除轮廓不良

	E_PARA_POINT_RGB_COMMON_TEXT,	//显示文本
	E_PARA_POINT_RGB_COMMON_DARK_DIST,	// 
	E_PARA_POINT_RGB_COMMON_BLUR_LOOP,	//Blur 5x5重复
	E_PARA_POINT_RGB_COMMON_SEG_X,	//分割二进制区域X
	E_PARA_POINT_RGB_COMMON_SEG_Y,	//分割二进制区域Y
	E_PARA_POINT_RGB_COMMON_MIN_THRESHOLD,	//最小二进制值(用于暗亮度)
	E_PARA_POINT_RGB_COMMON_MEDIAN,	//如果是大的不良行为,区域平均值与周围环境不同
	E_PARA_POINT_RGB_COMMON_POW,	// Pow

	E_PARA_POINT_RGB_ACTIVE_TEXT,	//显示文本
	E_PARA_POINT_RGB_ACTIVE_DARK_RATIO,	//与黑暗不良的环境亮度相比
	E_PARA_POINT_RGB_ACTIVE_BRIGHT_RATIO,	//Bright环境亮度不良

	E_PARA_POINT_RGB_EDGE_TEXT,	//显示文本
	E_PARA_POINT_RGB_EDGE_AREA,	//从外围设置范围
	E_PARA_POINT_RGB_EDGE_DARK_RATIO,	//黑暗环境亮度不良
	E_PARA_POINT_RGB_EDGE_BRIGHT_RATIO,	//Bright环境亮度不良

	E_PARA_POINT_RGB_CHOLE_POINT_TEXT,	//显示文本
	E_PARA_POINT_RGB_CHOLE_POINT_FLAG,	//是否使用Chole Point
	E_PARA_POINT_RGB_CHOLE_POINT_TDARK_RATIO,	//黑暗环境亮度不良
	E_PARA_POINT_RGB_CHOLE_POINT_TBRIGHT_RATIO,	//Bright环境亮度不良

	E_PARA_POINT_RGB_DEL_LINE_TEXT,	//显示文本
	E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_X,	//删除行x方向数
	E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_Y,	//删除行y方向数
	E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_X,	//行删除厚度x
	E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_Y,	//行删除厚度y
	E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_X,	//删除行x方向数
	E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_Y,	//删除行y方向计数
	E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_X,	//行删除厚度x
	E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_Y,	//行删除厚度y

	E_PARA_POINT_RGB_DEL_CCD_TEXT,	//显示文本
	E_PARA_POINT_RGB_DEL_CCD_DELETE_FLAG,	//是否启用删除

	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_TEXT,	//使用Gray pattern中的RGB校正Text
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_USE,//是否启用R模式校正
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_ADJUST_RATIO,	//R模式校正Ratio
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_CUT_MINGV,	//R用于图案校正的最小切割GV
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_USE,	//是否启用G模式校正
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_ADJUST_RATIO,	//G模式校正Ratio
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_CUT_MINGV,	//G模式校准时使用的最小切割GV
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_USE,//是否启用B模式校正
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_ADJUST_RATIO,	//B模式校正Ratio
	E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_CUT_MINGV,	//B用于图案校正的最小切割GV

	E_PARA_POINT_RGB_BRIGHT_INSP_TEXT,	//显示亮点检测
	E_PARA_POINT_RGB_BRIGHT_INSP_APPLYGV_WEIGHT,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_RED_SUB_RATIO,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_RED_MLT_RATIO,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_SUB_RATIO,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_MLT_RATIO,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_SUB_RATIO,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_MLT_RATIO,	//明点亮度调整权重比率
	E_PARA_POINT_RGB_BRIGHT_INSP_DARK_RATIO_ACTIVE,	//检测亮点DARK RATIO ACTIVE
	E_PARA_POINT_RGB_BRIGHT_INSP_BRIGHT_RATIO_ACTIVE,	//检测亮点BRIGHT RATIO ACTIVE
	E_PARA_POINT_RGB_BRIGHT_INSP_DARK_RATIO_EDGE,	//检测亮点DARK RATIO EDGE
	E_PARA_POINT_RGB_BRIGHT_INSP_BRIGHT_RATIO_EDGE,	//检测亮点BRIGHT RATIO EDGE

	E_PARA_POINT_RGB_TOTAL_COUNT								// Total
};

enum ENUM_PARA_POINT_BLACK
{
	E_PARA_POINT_BLACK_OUTLINE_TEXT = 0,	//显示文本
	E_PARA_POINT_BLACK_DELETE_PIXEL,	//清除轮廓不良

	E_PARA_POINT_BLACK_ACTIVE_TEXT,	//显示文本
	E_PARA_POINT_BLACK_ACTIVE_BLUR_1,	//消除噪音
	E_PARA_POINT_BLACK_ACTIVE_BLUR_2,	//用于创建背景
	E_PARA_POINT_BLACK_ACTIVE_THRESHOLD,	//二进制

	E_PARA_POINT_BLACK_DEL_LINE_TEXT,	//显示文本
	E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_CNT_X,	//删除行x方向数
	E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_CNT_Y,	//删除行y方向数
	E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_THICK_X,	//行删除厚度x
	E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_THICK_Y,	//行删除厚度y

	E_PARA_POINT_BLACK_BIG_TEXT,	//显示文本
	E_PARA_POINT_BLACK_BIG_FLAG,	//有/无大的不良&低GV不良检测
	E_PARA_POINT_BLACK_BIG_MIN_AREA,	//最大不良最小面积
	E_PARA_POINT_BLACK_BIG_MAX_GV,	//低GV的最大GV

	E_PARA_POINT_BLACK_DEL_CCD_TEXT,	//显示文本
	E_PARA_POINT_BLACK_DEL_CCD_OFFSET_FLAG,	//是否启用校正
	E_PARA_POINT_BLACK_DEL_CCD_DELETE_FLAG,	//是否启用删除
	E_PARA_POINT_BLACK_DEL_CCD_AUTO_FLAG,	//是否启用自动删除
	E_PARA_POINT_BLACK_DEL_CCD_AUTO_GV,	//4个自动删除CCD亮度
	E_PARA_POINT_BLACK_DEL_CCD_AUTO_BKGV,	//自动删除CCD周边背景亮度

	E_PARA_POINT_BLACK_ROI_OUTLINE_TEXT,	//显示文本
	E_PARA_POINT_BLACK_ROI_OUTLINE_OFFSET,	//Black pattern ROI区域offset

	E_PARA_POINT_BLACK_EDGE_TEXT,	//显示文本
	E_PARA_POINT_BLACK_EDGE_AREA,	//黑色模式ROI区域offset

	E_PARA_POINT_BLACK_TOTAL_COUNT						// Total
};

enum ENUM_PARA_POINT_DUST
{
	E_PARA_POINT_DUST_ENHANCMENT_TEXT = 0,	//显示文本
	E_PARA_POINT_DUST_ENHANCMENT_SHIFT_RANGE,	//SHIFT COPY区域
	E_PARA_POINT_DUST_ENHANCMENT_GAUSSIAN_SIZE,	// WINDOW SIZE
	E_PARA_POINT_DUST_ENHANCMENT_GAUSSIAN_SIGMA,	//SIGMA大小
	E_PARA_POINT_DUST_ENHANCMENT_MINMAX_SIZE,	// MINMAX FILTER SIZE

	E_PARA_POINT_DUST_BINARY_TEXT,	//显示文本
	E_PARA_POINT_DUST_BINARY_BLUR_LOOP,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_SEG_X,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_SEG_Y,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_MIN_THRESHOLD_GV,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_MEDIAN,	//Dust放大后删除

	E_PARA_POINT_DUST_BINARY_ACTIVE_TEXT,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_ACTIVE_DARK_RATIO,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_ACTIVE_BRIGHT_RATIO,	//将其放大以删除

	E_PARA_POINT_DUST_BINARY_EDGE_TEXT,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_EDGE_AREA,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_EDGE_DARK_RATIO,	//Dust放大后删除
	E_PARA_POINT_DUST_BINARY_EDGE_BRIGHT_RATIO,	//Dust放大后删除

	E_PARA_POINT_DUST_GROUP_TEXT,	//显示文本
	E_PARA_POINT_DUST_GROUP_USE,	//是否使用E判定
	E_PARA_POINT_DUST_GROUP_EDGE_DEL_OUTLINE,	//设置Edge覆盖区域		
	E_PARA_POINT_DUST_GROUP_EDGE_AREA,	//设置Edge部分区域	
	E_PARA_POINT_DUST_GROUP_MIN_AREA_EDGE,	//Edge局部区域最小大小
	E_PARA_POINT_DUST_GROUP_MIN_AREA_ACTIVE,	//活动局部区域最小大小
	E_PARA_POINT_DUST_GROUP_EDGE_COUNT,	//Edge区域Dust计数
	E_PARA_POINT_DUST_GROUP_ACTIVE_COUNT,	//活动区域Dust计数
	E_PARA_POINT_DUST_GROUP_TOTAL_COUNT,	//总Dust数

	E_PARA_POINT_DUST_LOGIC_TEXT,	//显示文本
	E_PARA_POINT_DUST_LOGIC_DELET_AREA,	//显示文本
	E_PARA_POINT_DUST_LOGIC_MORP_RANGE,	//显示文本
	E_PARA_POINT_DUST_LOGIC_BIG_AREA,	//显示文本

	E_PARA_POINT_DUST_BUBBLE_TEXT,	//用于检测文本显示Bubble
	E_PARA_POINT_DUST_BUBBLE_RESIZE,	//画面重置
	E_PARA_POINT_DUST_BUBBLE_SRC_BLUR_SIZE,	//原画面Blur Size
	E_PARA_POINT_DUST_BUBBLE_BK_BLUR_SIZE,	//背景画面Blur Size
	E_PARA_POINT_DUST_BUBBLE_THRESHOLD,	//背景减除结果画面Threshold_Value
	E_PARA_POINT_DUST_BUBBLE_CLOSE_SIZE,	//防止不良结果中断CLOSE运算大小

	E_PARA_POINT_DUST_TOTAL_COUNT						// Total
};

// G87
enum ENUM_PARA_POINT_G87
{
	E_PARA_POINT_G87_GAUSSIAN_TEXT = 0,	//显示文本

	E_PARA_POINT_G87_GAUSSIAN_SIZE,	//预处理(留空)高斯尺寸
	E_PARA_POINT_G87_GAUSSIAN_SIGMA,	//前处理(留空)高斯信号
	E_PARA_POINT_G87_GAUSSIAN_SIZE2,	//检查用高斯尺寸
	E_PARA_POINT_G87_GAUSSIAN_SIGMA2,	//检查用高斯信号

	E_PARA_POINT_G87_CHOLE_MASK_TEXT,	//显示文本
	E_PARA_POINT_G87_CHOLE_MASK_SIZE_UP,	//增加Chole Mask大小

	E_PARA_POINT_G87_CHOLE_POINT_TEXT,	//显示文本

	E_PARA_POINT_G87_CHOLE_POINT_DARK_RATIO,	//黑暗环境亮度不良
	E_PARA_POINT_G87_CHOLE_POINT_BRIGHT_RATIO,	//Bright环境亮度不良

	E_PARA_POINT_G87_DEL_LINE_TEXT,	//显示文本
	E_PARA_POINT_G87_DEL_LINE_BRIGHT_CNT_X,	//删除行x方向数
	E_PARA_POINT_G87_DEL_LINE_BRIGHT_CNT_Y,	//删除行y方向数
	E_PARA_POINT_G87_DEL_LINE_BRIGHT_THICK_X,	//行删除厚度x
	E_PARA_POINT_G87_DEL_LINE_BRIGHT_THICK_Y,	//行删除厚度y

	E_PARA_POINT_G87_TOTAL_COUNT						// Total

};

enum ENUM_MORP
{
	E_MORP_ERODE = 0,
	E_MORP_DILATE,
	E_MORP_OPEN,
	E_MORP_CLOSE
};

//////////////////////////////////////////////////////////////////////////

class CInspectPoint
{
public:
	CInspectPoint(void);
	virtual ~CInspectPoint(void);

	//内存管理
	CMatBuf* cMem[2];
	void		SetMem_Multi(int nCnt, CMatBuf** data)
	{
		for (int i = 0; i < nCnt; i++)
		{
			cMem[i] = data[i];
		}
	};
	//CMatBuf*	GetMem()				{	return	cMem1	;};

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

	CString		GETDRV()
	{
		return m_cInspectLibLog->GETDRV();
	}

	//////////////////////////////////////////////////////////////////////////

		//Main检查算法
	long		DoFindPointDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
		cv::Point* ptCorner, double* dAlignPara, cv::Rect* rcCHoleROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, CDefectCCD* cCCD, cv::Mat* matCholeBuffer);

	//删除Dust后,移交结果向量
	long		GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer,
		cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer);

	//检测出气泡后,将结果向量移交(灰尘:设置为255GV//气泡:设置为200GV)
	long		GetDefectList_Bubble(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer,
		cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob);

protected:

	//R,G,B模式检测算法
	long		LogicStart_RGB(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, stPanelBlockJudgeInfo* EngineerBlockDefectJudge);

	//R,G,B模式检测算法
	long		LogicStart_RGBTest(cv::Mat** matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, int Type, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer);

	//黑色模式检测算法
	long		LogicStart_Black(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge);

	//Gray模式检测算法
	long		LogicStart_Gray(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer);

	long		LogicStart_Gray2(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge);

	//PNZ DUST集成版
	long		LogicStart_DustALL(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, cv::Rect* rcCHoleROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge);

	// - Chole Point
	long		LogicStart_CholePoint_G87(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
		int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer);

protected:

	//保存画面(8bit&12bit)
	long		ImageSave(CString strPath, cv::Mat matSrcBuf);

	//分区二进制(8bit&12bit)
	long		RangeAvgThreshold(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub = NULL);
	long		RangeAvgThreshold_8bit(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub = NULL);
	long		RangeAvgThreshold_16bit(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub = NULL);
	long		RangeAvgThreshold_RGB(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub = NULL);
	long		RangeAvgThreshold_DUST(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub = NULL);

	//莫波罗吉(8bit&12bit)
	long		Morphology(cv::Mat& matSrcImage, cv::Mat& matDstImage, long nSizeX, long nSizeY, ENUM_MORP nOperation);

	//去除外角(8bit&12bit)
	long		DeleteOutArea(cv::Mat& matSrcImage, cv::Point* ptCorner, CMatBuf* cMemSub = NULL);
	long		DeleteOutArea(cv::Mat& matSrcImage, CRect rectROI, CMatBuf* cMemSub = NULL);

	//删除小面积(8 bit&12 bit)
	long		DeleteArea(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);
	long		DeleteArea_8bit(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);
	long		DeleteArea_16bit(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub = NULL);

	//Dust不好的情况很多时,E级判定检查函数(8bit&12bit)
	long		JudgementCheckE(cv::Mat& matSrcBuf, double* dPara, CRect rectROI, CMatBuf* cMemSub = NULL);

	//在原始图像中加入RGB模式中的一个来提供校正值的功能->在Gray中使用(8bit&12bit)
	long		AdjustImageWithRGB(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matAdjustSrcImage, double dblAdjustRatio, int nCutMinGVForAdjust, CRect rectROI, CMatBuf* cMemSub = NULL);

	// PNZ Pattern Substraction Test
	long		PatternSubstraction(cv::Mat& matSrcImage, cv::Mat* matDstImage, int type, CMatBuf* cMemSub = NULL);

	void		ApplyEnhancement(cv::Mat matSrcImage, cv::Mat matBuff1, cv::Mat matBuff2, cv::Mat& matDstImage1, cv::Mat& matDstImage2, double* dPara, int* nCommonPara, CString strAlgPath, int Type, CMatBuf* cMemSub = NULL);

	void		SubPatternEhancemnt(cv::Mat matSrcImage, cv::Mat& matDstImage, double dSubWeight, double dEnhanceWeight, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub = NULL);

	//在Dust中查找较大的面积(8 bit&12 bit)
	long		FindBigAreaDust(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, long nFindMinArea, CMatBuf* cMemSub = NULL);

	//删除暗点-Dust面积较大的周边(8bit&12bit)
	long		DeleteCompareDarkPoint(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nModePS);

	//只查找暗Dust面积小的情况(8bit&12bit)
	long		DarkDustMaxArea(cv::Mat matSrcBuffer[E_DEFECT_COLOR_COUNT], int nMaxArea, CMatBuf* cMemSub = NULL);

	//删除小面积(8 bit&12 bit)
	long		DeleteMinArea(cv::Mat matSrcBuffer, cv::Mat matThBuffer, int nMinArea, int nMaxGV, CMatBuf* cMemSub = NULL);

	//在Dust画面中查找气泡
	long		FindBubble_DustImage(cv::Mat matSrcbuffer, cv::Mat& matBubbleResult, cv::Rect rtROI, cv::Rect* rcCHoleROI, double* dPara, int* nCommonPara, CString strAlgPath);

	void		Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset = 0);

	// chole Point G87 test
	void		ImageShift(cv::Mat& matSrc, cv::Mat& matDst, int nShift_X, int nShift_Y);
};