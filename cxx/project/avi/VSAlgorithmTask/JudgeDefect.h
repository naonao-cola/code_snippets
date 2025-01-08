/*****************************************************************************
  File Name		: JudgeDefect.h
  Version		: ver 1.0
  Create Date	: 2017.03.21
  Description	:检查结果汇总和判定Class
  Abbreviations	:
 *****************************************************************************/

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "InspThrd.h"
#include "AlgorithmInterface.h"

 //	Class功能	:	检查结果汇总和判定Class
 //主要功能		:
 //	创建日期		:2017.03.21
 //	作者		:	CWH
 //	修改历史记录		:	V.1.0初始创建
 //	请参见	:	

class JudgeDefect :
	public CInspThrd, protected InspectAlgorithmInterface
{
public:
	JudgeDefect();
	~JudgeDefect(void);

protected:
	//消除非检查区域不良
	bool			m_fnDefectFiltering(cv::Mat& MatDrawBuffer, int nImageNum, int nCameraNum, stDefectInfo* pResultBlob, tAlignInfo stThrdAlignInfo, int nRatio);

	//多边形内是否有点
	bool			PtInPolygon(cv::Point* ptPolygon, cv::Point ptPoint, int nPolygonCnt = 4);

	//收集不良信息
	void			ConsolidateResult(const CString strPanelID, const CString strDrive,
		CWriteResultInfo WrtResultInfo, ResultBlob_Total* pResultBlob_Total, ResultPanelData& resultPanelData,
		const int nImageNum, const int nCameraNum, int nRatio, ENUM_INSPECT_MODE eInspMode);

	void			RotateRect(CRect rcSrc, cv::Point* pPtDst, tAlignInfo stThrdAlignInfo, bool bRotate = true);

	void			NumberingDefect(const CString strModelID, const CString strPanelID, const CString strLotID, CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, int nRatio);

	//AVI AD不良附加函数
	bool			JudgeADDefect(int nImageNum, int nCameraNum, int nStageNo, int nImageWidth, int nImageHeight, ResultBlob_Total* pResultBlob_Total, int nDefectAD, ENUM_INSPECT_MODE eInspMode, bool bAD = true);

	bool			m_fnCompareValue(int nDefectCount, int nRefCount, int nSign);

	bool			JudgementPanelGrade(ResultPanelData& resultPanelData);

	bool			ApplyReportRule(ResultPanelData& resultPanelData);

	int			UserDefinedFilter(ResultPanelData& resultPanelData, int nTotalDefectCount); //自定义过滤器。N个以上DEPEC设置的数量以上时无条件报告。

	//////////////////////////////////////虚拟函数
	//判定函数
	virtual bool	Judgement(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT], tCHoleAlignInfo& tCHoleAlignData,
		const CString strModelID, const CString strLotID, const CString strPanelID, const CString strDrive, int nRatio,
		ENUM_INSPECT_MODE eInspMode, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], bool bUseInspect, int nStageNo) = 0;

	virtual bool	Judgement_AI(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT], tCHoleAlignInfo& tCHoleAlignData,
		const CString strModelID, const CString strLotID, const CString strPanelID, const CString strDrive, int nRatio,
		ENUM_INSPECT_MODE eInspMode, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], bool bUseInspect, int nStageNo) = 0;

	//修复设备使用的坐标值和代码判定
	virtual bool	JudgementRepair(const CString strPanelID, ResultPanelData& resultPanelData, CWriteResultInfo& WrtResultInfo) = 0;
	virtual bool	JudgementPointToLine(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, const int nImageWidth, const int nImageHeight) { return	false; };
	virtual	bool	JudgeGroup(ResultPanelData& resultPanelData, cv::Mat(*MatDraw)[MAX_CAMERA_COUNT]) { return	false; };
	//过滤函数
	virtual bool	DeleteOverlapDefect(ResultPanelData& resultPanelData) { return	false; };
	//从Casting stDefectInfo中提取所需的部分,并将其装载到ResultPanelData中
	virtual bool	GetDefectInfo(CWriteResultInfo WrtResultInfo, ResultDefectInfo* pResultDefectInfo, stDefectInfo* pResultBlob,
		int nBlobCnt, int nImageNum, int nCameraNum, int nRatio) {
		return	false;
	};

	//添加每个阶段的AD故障计数功能
	bool JudgeDefect::m_fnCountingStageAD(int nImageNum, int nStageNo, int nDefectType);
};