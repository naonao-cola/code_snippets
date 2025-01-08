
/************************************************************************/
//Align集成相关标头
//修改日期:18.02.07
/************************************************************************/

#pragma once

#include "Define.h"
#include "InspectLibLog.h"
#include "InspectMura.h"
#include"MatBuf.h"				//内存管理
#include <algorithm>
#include "DefectCCD.h"
#include <stdlib.h>

enum
{
	TOP = 0, BOTTOM, LINE_END
};

class CInspectAlign
{
public:
	CInspectAlign(void);
	virtual ~CInspectAlign(void);

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
	//////////////////////////////////////////////////////////////////////////

		//查找Top Line角度(用于相机角度校正)(8bit&12bit)
	long	DoFindTheta(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point& ptCenter, wchar_t* strID = NULL);

	//查找AVI扫描区域(8bit&12bit)
	long	DoFindActive(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, cv::Point* ptContCorner, int nRatio, cv::Point& ptCenter, wchar_t* strID = NULL);

	//查找APP检查区域
	long	DoFindActive_APP(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, int nRatio, double dCamResolution, double dPannelSizeX, double dPannelSizeY, cv::Point& ptCenter, wchar_t* strID = NULL);

	//查找SVI检查区域
	long	DoFindActive_SVI(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, int nCameraNum, int nRatio, cv::Point& ptCenter, wchar_t* strID = NULL);

	long	SetFindContour(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath);
	long	SetFindContour_APP(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath, Point* ptAlignCorner, CStringA strImageName, double dAlignTH, bool bImageSave = false);

	//Round/Camera Hole设置&保存文件(8 bit)
	long	SetFindContour_(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nCHoleROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath);

	//Round/Camera Hole设置&保存文件(8 bit)
	long	SetFindContour_2(cv::Mat* matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nCHoleROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath);

	//Round设置&保存文件
	void	SetFindRound(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nContourIdx, int nAlgImg, int nRatio, CString strPath);
	void	SetFindRoundAuto(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], int nContourIdx, int nAlgImg, int nRatio, float dTheta, CString strPath);

	//Chole设置&保存文件
	void	SetFindCHole(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nCHoleROICnt, int nContourIdx, int nAlgImg, int nRatio, CString strPath);

	//CHole设置&保存文件2
	void	SetFindCHole2(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nCHoleROICnt, int nContourIdx, int nAlgImg, int nRatio, CString strPath);

	//G3填充外围test
	double CenterMeanGV(cv::Mat& matSrcBuf, int nMinArea);

	//yuxuefeiforMark
	long DoFindDefectMark(cv::Mat matSrcBuf, double* dPara, cv::Point* ptCorner, double dAngel, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT], int nMarkROICnt);

	//yuxuefeiforLabel
	long DoFindDefectLabel(cv::Mat matSrcBuf, double* dPara, cv::Point* ptCorner, double dAngel, CRect rectLabelArea[MAX_MEM_SIZE_LABEL_COUNT]);

	//yuxuefei
	long DoFindMarkTop(cv::Mat& matSrcBuf, double* dPara, cv::Point* ptCorner, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT]);
	//yuxuefei
	long DoFindMarkBottom(cv::Mat& matSrcBuf, double* dPara, cv::Point* ptCorner, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT]);


	//外围处理(8bit&12bit)
	long	DoFillOutArea(cv::Mat& matSrcBuf, cv::Mat& MatDrawBuffer, cv::Mat& matBKGBuf, cv::Point ptResCornerOrigin[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA], cv::Mat* matCHoleROIBuf, cv::Rect* rcCHoleROI, bool* bCHoleAD,
		double* dPara, int nAlgImg, int nRatio, wchar_t* strID = NULL);

	//SVI外围处理(Color)
	long	DoFillOutArea_SVI(cv::Mat& matSrcBuf, cv::Mat& matBKGBuf, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA],
		double* dPara, int nAlgImg, int nCameraNum, int nRatio, wchar_t* strID = NULL, cv::Point* ptCorner = NULL);

	//APP外围处理(8bit)
	long	DoFillOutArea_APP(cv::Mat& matSrcBuf, cv::Mat& matBKGBuf, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA],
		double* dPara, int nAlgImg, int nCameraNum, int nRatio, wchar_t* strID = NULL, cv::Point* ptCorner = NULL, vector<vector<Point2i>>& ptActive = vector<vector<Point2i>>(), double dAlignTheta = 0, CString strPath = NULL, bool bImageSave = false);

	//Dust Pattern外围处理(8bit&12bit)
	long	DoFillOutAreaDust(cv::Mat& matSrcBuf, cv::Mat& MatDrawBuffer, cv::Point ptResCorner[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, double dAngle, cv::Rect* rcCHoleROI, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA],
		double* dPara, int nAlgImg, int nRatio, wchar_t* strID = NULL);

	//画面旋转
	long	DoRotateImage(cv::Mat ptSrcBuffer, cv::Mat& ptDstBuffer, double dAngle);

	//旋转坐标
	long	DoRotatePoint(cv::Point matSrcPoint, cv::Point& matDstPoint, cv::Point ptCenter, double dAngle);

	//检查AVI AD(8bit&12bit)
	long	DoFindDefectAD(cv::Mat matSrcBuf, double* dPara, double* dResult, int nRatio);

	//SVI AD检查(Color)
	long	DoFindDefectAD_SVI(cv::Mat matSrcBuf, double* dPara, double* dResult, int nCameraNum, int nRatio);

	//检查APP AD(8 bit)
	long	DoFindDefectAD_APP(cv::Mat MatOrgImage, double* dAlgPara, double* dResult, int nRatio);
	long	Check_Abnormal_PADEdge(cv::Mat MatOrgImage, int nThreshold, double dCompare_Theta, Rect rtObject);

	//检查AVI AD GV(8bit&12bit)
	long	DoFindDefectAD_GV(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner, CDefectCCD* cCCD = NULL);

	//AVI PG 压接异常 
	long	DoFindPGAnomal_AVI(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner);

	//检查Dust 4区域GV
	long	DoFindDefectAD_GV_DUST(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner);

	//SVI AD GV检查(Color)
	long	DoFindDefectAD_GV_SVI(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner);

	//卷判定
	long CurlJudge(cv::Mat matSrcBuf, double* dpara, cv::Point* ptCorner, BOOL& bCurl, stMeasureInfo* stCurlMeasure, BOOL bSaveImage, CString strPath);

protected:

	//查找Cell区域
	long	FindCellArea(cv::Mat matThreshBuf, int nMinArea, cv::Rect& rectCell);

	//按方向获取数据
	long	RobustFitLine(cv::Mat& matTempBuf, cv::Rect rectCell, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType, int nSamp = 50);

	//通过Profile的斜度识别物体来导航坐标
	long	ObjectOutAreaGetLine(cv::Mat& matTempBuf, cv::Rect rectCell, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType, int nThreshold_Theta, float fAvgOffset);

	//在二进制画面中从物体的内部到外部导航坐标
	long	ObjectInAreaGetLine(cv::Mat& matTempBuf, cv::Rect rectImgSize, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType);

	//返回最大Blob的APP
	long	FindBiggestBlob_APP(cv::Mat& src, cv::Mat& dst);

	//查找4个转角位置
	long	FindCornerPoint(cv::Point2f ptSrc[E_CORNER_END], cv::Point ptDst[E_CORNER_END], long nWidth, long nHeight);

	//保留点亮区域的外围部分
	long	FindEdgeArea(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nLength, CMatBuf* cMemSub = NULL);

	//保留点亮区域的外围部分(Color)
	long	FindEdgeArea_SVI(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nLength);

	//横向平均填充
	long	FillAreaMeanX(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV);

	//横向平均填充(Color)
	long	FillAreaMeanX_SVI(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV);

	//垂直平均填充
	long	FillAreaMeanY(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV);

	//垂直平均填充(Color)
	long	FillAreaMeanY_SVI(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV);

	//只保留已点亮的部分
	long	FillMerge(cv::Mat& matSrcBuf, cv::Mat matMeanBuf, cv::Mat matMaskBuf, int nAlgImg, CMatBuf* cMemSub = NULL); //choikwangil 04.06b11修改

	//只留下点亮的部分(Color)
	long	FillMerge_SVI(cv::Mat& matSrcBuf, cv::Mat matMeanBuf, cv::Mat matMaskBuf);

	//将曲线部分,点等区域内侧轻轻放入(视情况而定)
	cv::Point	calcRoundIn(ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], int nIndex, int nRoundIn);

	//曲线以外的直线连接
	long	calcLineConnect(cv::Mat& matSrcBuf, cv::Point ptSE[2], cv::Point ptPoly[2], int& nSE, int nSetArea);

	//填充拐角部分区域(Color)
	long	FillCorner(cv::Mat& matSrcROIBuf, cv::Mat& matMaskROIBuf, int nType);

	long Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, CMatBuf* cMemSub);

	vector<tBLOB_FEATURE> m_BlobMark;

	//复制图像Shift
	long	ShiftCopyParaCheck(int ShiftValue, int& nCpyX, int& nCpyY, int& nLoopX, int& nLoopY);

	//查找Cell的Edge以局部地进行移动—以提高速度
	long	CInspectAlign::FindCellEdge_For_Morphology(cv::Mat matSrc, int nThreshold, cv::Rect& rcFindCellROI); // 找到大致的细胞位置。
	long	CInspectAlign::MakeRoI_For_Morphology(cv::Rect rcFindCellROI,
		int nExtROI_Outer, int nExtROI_Inner_L, int nExtROI_Inner_R, int nExtROI_Inner_T, int nExtROI_Inner_B,
		cv::Size rcLimit, cv::Rect* prcMorpROI); //创建要显示的区域

	//部分支持的功能
	long	CInspectAlign::Partial_Morphology(cv::Mat matSrc, cv::Mat matDst, int nMorpType, cv::Mat StructElem, cv::Rect* prcMorpROI);

	//部分拉普拉西亚
	long	CInspectAlign::Partial_Laplacian(cv::Mat matSrc, cv::Mat matDst, cv::Rect* prcMorpROI);

	//限制盒子区域的功能
	void RecalRect(cv::Rect& rcRect, cv::Size szLimit);

	//获取模式名称
	CString		GetPatternString(int nPattern);
	CStringA	GetPatternStringA(int nPattern);

	//获取BM LT,RT
	long GetBMCorner(cv::Mat Src, double* dAlgPara, Point* ptPanelCorner, cv::Rect& rtBMCorner);
};