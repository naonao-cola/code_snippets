
/************************************************************************/
//Blob相关标头
//修改日期:17.03.08
/************************************************************************/

#pragma once

#include "Define.h"

//过了设置时间,类就结束了
#include "TimeOut.h"

//内存管理
#include "MatBuf.h"
#include "InspectLibLog.h"
#include "AIInspectLib/AIRuntime/AIRuntimeDataStruct.h"

#include <math.h>

//////////////////////////////////////////////////////////////////////////

//绘制Blob
#define BLOB_DRAW_ROTATED_BOX				0x000010
#define BLOB_DRAW_BOUNDING_BOX				0x000020
#define BLOB_DRAW_BLOBS						0x000080
#define BLOB_DRAW_BLOBS_CONTOUR				0x000100

//////////////////////////////////////////////////////////////////////////

const CString gg_strPat[E_IMAGE_CLASSIFY_AVI_COUNT] = {
	_T("R"),			// E_IMAGE_CLASSIFY_SVI_R			,	// 00 R
	_T("G"),			// E_IMAGE_CLASSIFY_SVI_G			,	// 01 G
	_T("B"),			// E_IMAGE_CLASSIFY_SVI_B			,	// 02 B
	_T("BLACK"),		// E_IMAGE_CLASSIFY_SVI_BLACK		,	// 03 BLACK
	_T("WHITE"),		// E_IMAGE_CLASSIFY_SVI_WHITE		,	// 04 WHITE
	_T("GRAY32"),		// E_IMAGE_CLASSIFY_SVI_GRAY_32		,	// 06 GRAY_32
	_T("GRAY64"),		// E_IMAGE_CLASSIFY_SVI_GRAY_64		,	// 07 GRAY_64
	_T("GRAY87"),		// E_IMAGE_CLASSIFY_SVI_GRAY_87		,	// 08 GRAY_87
	_T("GRAY128")		// E_IMAGE_CLASSIFY_SVI_GRAY_128	,	// 09 GRAY_128
};

// Camera String
const CString gg_strCam[2] = {
	_T("Coaxial"),
	_T("Side")
};

//不良
enum ENUM_BOLB_FEATURE
{
	E_FEATURE_AREA = 0,	//对象面积
	E_FEATURE_BOX_AREA,	//Bounding Box面积
	E_FEATURE_BOX_RATIO,	//对象面积/Bounding Box面积比率
	E_FEATURE_SUM_GV,	//累计亮度
	E_FEATURE_MIN_GV,	//最小亮度
	E_FEATURE_MAX_GV,	//最大亮度
	E_FEATURE_MEAN_GV,	//平均亮度
	E_FEATURE_DIFF_GV,	//(背景-对象)亮度差异
	E_FEATURE_BK_GV,	//背景亮度
	E_FEATURE_STD_DEV,	//标准偏差
	E_FEATURE_SEMU,	// SEMU
	E_FEATURE_COMPACTNESS,	//对象有多接近原样？
	E_FEATURE_MIN_GV_RATIO,	//对象最小亮度/背景亮度
	E_FEATURE_MAX_GV_RATIO,	//对象最大亮度/背景亮度
	E_FEATURE_DIFF_GV_RATIO,	//对象平均亮度/背景亮度
	E_FEATURE_PERIMETER,	//外框长度
	E_FEATURE_ROUNDNESS,	// 
	E_FEATURE_ELONGATION,	//Box水平/垂直
	E_FEATURE_BOX_X,	//Bounding Box宽度
	E_FEATURE_BOX_Y,	//Bounding Box垂直长度

	E_FEATURE_MIN_BOX_AREA,	// Feret’s area
	E_FEATURE_MINOR_AXIS,	//长轴
	E_FEATURE_MAJOR_AXIS,	//缩短
	E_FEATURE_AXIS_RATIO,	//长轴/缩短
	E_FEATURE_MIN_BOX_RATIO,	//对象面积/Min Bounding Box面积比(区域孔隙率)

	E_FEATURE_GV_UP_COUNT_0,	//亮度
	E_FEATURE_GV_UP_COUNT_1,	//亮度
	E_FEATURE_GV_UP_COUNT_2,	//亮度
	E_FEATURE_GV_DOWN_COUNT_0,	//亮度
	E_FEATURE_GV_DOWN_COUNT_1,	//亮度
	E_FEATURE_GV_DOWN_COUNT_2,	//亮度

	E_FEATURE_MEANAREA_RATIO,   // MeanAreaRatio choikwangil

	E_FEATURE_GVAREA_RATIO_TEST,   // 04.17 choikwangil

	E_FEATURE_IS_EDGE_C,	//拐角处是否位于外围位置
	E_FEATURE_IS_EDGE_V,	//垂直于外围位置
	E_FEATURE_IS_EDGE_H,	//水平位置
	E_FEATURE_IS_EDGE_CENTER,	//不是外围的位置

	E_FEATURE_COUNT
};

//////////////////////////////////////////////////////////////////////////

//Blob结构体
struct tBLOB_FEATURE
{
	bool				bFiltering;				// 过滤有/无
	int				nBlockNum;
	cv::Rect			rectBox;				// Bounding Box
	long				nArea;					// 对象面积
	long				nBoxArea;				// Bounding Box面积
	float				fBoxRatio;				// Bounding Box面积比/对象面积(Rectangulaty(=Extent)
	cv::Point			ptCenter;				// 中心点
	long				nSumGV;					// 累计亮度
	long				nMinGV;					// 最小亮度
	long				nMaxGV;					// 最大亮度
	float				fMeanGV;				// 平均亮度
	float				fDiffGV;				// (背景-对象)亮度差异
	float				fBKGV;					// 背景亮度
	float				fStdDev;				// 标准偏差
	float				fSEMU;					// SEMU
	float				fCompactness;			// 对象有多接近圆形？
	float				nMinGVRatio;			// 对象最小亮度/背景亮度
	float				nMaxGVRatio;			// 对象最大亮度/背景亮度
	float				fDiffGVRatio;			// 对象平均亮度/背景亮度
	float				fPerimeter;				// 轮廓长度
	float				fRoundness;				// 
	float				fElongation;			// Box宽/深	
	float				fMinBoxArea;			// Feret’s area
	float				fMinorAxis;				// 长轴(Feret's Diameter)
	float				fMajorAxis;				// 缩短(垂直于Feret's diameter的最长轴的长度/Breath)
	float				fAxisRatio;				// 长轴/缩短(ELONGATION)
	float				fAngle;					// Angle between the horizontal axis ( Axis of least second moment )
	float				fMinBoxRatio;			// Min Bounding Box面积比/对象面积(区域孔隙率)
	float				fMeanAreaRatio;			// choikwangil

	float				fAreaPer;			// choikwangil 04.20
	long				nJudge_GV;			// choikwangil 04.20
	long				nIn_Count;			// choikwangil 04.20

	__int64				nHist[IMAGE_MAX_GV];	// 对象直方图

	CvSize2D32f			BoxSize;				// Box width and length

	vector <cv::Point>	ptIndexs;				// Blob像素坐标
	vector <cv::Point>	ptContours;				// Blob轮廓坐标
	bool				fromAI;					// Indicating whether it comes from an AI algorithm.
	double				confidence;				// confidence
	int					AICode;					// the AI result item classfication code

	//初始化结构体
	tBLOB_FEATURE() :
		nArea(0)
	{
		bFiltering = false;
		int nBlockNum = 0; //当前缺陷特征所属分区   默认 0为第一个分区  hjf
		rectBox = cv::Rect(0, 0, 0, 0);
		nArea = 0;
		nBoxArea = 0;
		fBoxRatio = 0.0f;
		nSumGV = 0;
		nMinGV = 0;
		nMaxGV = 0;
		fMeanGV = 0.0f;
		fDiffGV = 0.0f;
		fBKGV = 0.0f;
		fStdDev = 0.0f;
		fSEMU = 0.0f;
		fCompactness = 0.0f;
		nMinGVRatio = 0.0f;
		nMaxGVRatio = 0.0f;
		fDiffGVRatio = 0.0f;
		fPerimeter = 0.0f;
		fRoundness = 0.0f;
		fElongation = 0.0f;
		fMinBoxArea = 0.0f;
		fMinorAxis = 0.0f;
		fMajorAxis = 0.0f;
		fAxisRatio = 0.0f;
		fAngle = 0.0f;
		fMinBoxRatio = 0.0f;
		fMeanAreaRatio = 0.0f;

		fAreaPer = 0.0f;
		nJudge_GV = 0;
		nIn_Count = 0;

		memset(nHist, 0, sizeof(__int64) * IMAGE_MAX_GV);

		ptCenter = CvPoint2D32f(0, 0);
		BoxSize = CvSize2D32f(0, 0);

		//矢量初始化
		vector <cv::Point>().swap(ptIndexs);
		vector <cv::Point>().swap(ptContours);
	}
};

class CFeatureExtraction
{
public:
	CFeatureExtraction(void);
	virtual ~CFeatureExtraction(void);

	//禁用内存
	bool	Release();

public:
	//内存管理
	CMatBuf* cMem;
	void		SetMem(CMatBuf* data) { cMem = data; };
	CMatBuf* GetMem() { return	cMem; };

	//日志管理
	InspectLibLog* m_cInspectLibLog;
	clock_t				m_tInitTime;
	clock_t				m_tBeforeTime;
	wchar_t* m_strAlgLog;
	int					m_nAlgType;

	void		SetLog(InspectLibLog* cLog, int nAlgType, clock_t tTimeI, clock_t tTimeB, wchar_t* strLog)
	{
		m_tInitTime = tTimeI;
		m_tBeforeTime = tTimeB;
		m_cInspectLibLog = cLog;
		m_strAlgLog = strLog;
		m_nAlgType = nAlgType;
	};

	void		writeInspectLog(char* strFunc, wchar_t* strTxt)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(m_nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, m_strAlgLog);
	};

	void		writeInspectLog_Memory(int nAlgType, char* strFunc, wchar_t* strTxt, __int64 nMemory_Use_Value = 0, int nSub_AlgType = NULL)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, nMemory_Use_Value, nSub_AlgType, m_strAlgLog);
	};

	//////////////////////////////////////////////////////////////////////////

		//运行Blob算法
	bool	DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Mat GrayBuffer = cv::Mat(), int nMaxDefectCount = 99999);

	//Blob算法执行ROI
	bool	DoBlobCalculate(cv::Mat ThresholdBuffer, CRect rectROI, cv::Mat GrayBuffer = cv::Mat(), int nMaxDefectCount = 99999);
	//////////////////////////////////////////////////////////////////////////

		//Blob&判定结果(用于Single)
	long	DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, int* nCommonPara,
		long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, int nDefectType, bool bPtRotate = true);

	//Blob&判定结果
	long	DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, int* nCommonPara,
		long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool bPtRotate = true);

	//Blob&判定结果(针对Single)ROI
	long	DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, CRect rectROI, int* nCommonPara,
		long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, int nDefectType, bool bPtRotate = true);

	//Blob&判定结果ROI
	long	DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, CRect rectROI, int* nCommonPara,
		long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool bPtRotate = true);
	//没有报告,只提供信息...
	long	DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, int nDefectType = 0, int nMaxDefectCount = 99999);

	//没有报告,只提供信息...
	long	DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, int nMaxDefectCount = 99999);

	//绘制结果
	bool	DrawBlob(cv::Mat& DrawBuffer, CvScalar DrawColor, long nOption, bool bSelect = false, float fFontSize = 0.4f);

	//保存轮廓坐标文本
	bool	SaveTxt(int* nCommonPara, wchar_t* strContourTxt, bool bUse = false);

	//转交结果
	bool	GetResultblob(vector<tBLOB_FEATURE>& OutBlob);

	//过滤修改轮廓
	long	DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, int* nCommonPara,
		long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool bPtRotate, CRect prerectROI, int offset);

	long	DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
		int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, int nDefectType, bool bPtRotate, CRect prerectROI, int offset);

	long	DoDefectBlobMultiJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
		int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, vector<int> nDefectType, bool bPtRotate, CRect prerectROI, int offset);

	//比较值
	bool	Compare(double dFeatureValue, int nSign, double dValue);

	//坐标校正
	void	CoordApply(CRect rectROI, int nTotalLabel);

	long CFeatureExtraction::DoDefectAIDectectJudgment(
		cv::Mat& matSrcImage,
		cv::Mat& matDrawBuffer,
		int* nCommonPara,
		long nDefectColor,
		CString strTxt,
		stPanelBlockJudgeInfo* EngineerBlockDefectJudge,
		stDefectInfo* pResultBlob,
		std::shared_ptr<vector<tBLOB_FEATURE>> detBlob,
		cv::Rect cutRoi,
		int nDefectType);
	//获取分区块 编号 hjf
	int GetGridNumber(int imageWidth, int imageHeight, int X, int Y, int center_x, int center_y);
protected:
	bool					m_bComplete;	// 确认Blob是否已完成。
	vector<tBLOB_FEATURE>	m_BlobResult;	// Blob结果列表
	//////////////////////////////////////////
	//提取块结果
	void divideBlobResult(int imageWidth, int imageHeight, int X, int Y);
	/// ///////////////////////////////////////////

		//过滤不良颜色
	bool			DoColorFilter(int nDefectName, int nDefectColor);

	//过滤
	bool			DoFiltering(tBLOB_FEATURE& tBlobResult, int nBlobFilter, int nSign, double dValue);

	// Feature Calculate ( 8bit & 12bit )
	bool			DoFeatureBasic_8bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel, CMatBuf* cMemSub = NULL);
	bool			DoFeatureBasic_16bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel, CMatBuf* cMemSub = NULL);

	//17.11.29-外围信息(AVI&SVI其他工具)
	bool			calcContours(int* nContoursX, int* nContoursY, int nDefectIndex, float fAngle, int nCx, int nCy, int nPs);

	//过滤修改轮廓
	bool			is_edge(tBLOB_FEATURE& tBlobResult, int nBlobFilter, CRect prerectROI, int offset);

private:
	//只填写一个面板坐标txt文件。删除并发访问
	CRITICAL_SECTION m_csCoordFile;

	//过了设置时间,类就结束了
	CTimeOut cTimeOut;

	///////AI Detect  
	// 202307
	//HJF
public:
	TaskInfoPtr spTaskInfo;

};