/*****************************************************************************
  File Name		: InspMainSequence.h
  Version		: ver 1.0
  Create Date	: 2017.03.21
  Description	:Area对应检查线程
  Abbreviations	:
 *****************************************************************************/

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "VSAlgorithmTask.h"
#include "JudgeDefect.h"

 //	Class功能	:	
 //主要功能	:
 //	创建日期	:2017/02
 //	作者	:	CWH
 //	修改历史记录	:	V.1.0初始创建
 //	请参见	:	

class InspMainSequence :
	protected JudgeDefect
{
public:
	InspMainSequence();
	virtual ~InspMainSequence(void);

protected:

	// Generated message map functions
	afx_msg	LRESULT OnStartInspection(WPARAM wParam, LPARAM lParam);
	afx_msg	LRESULT OnStartSaveImage(WPARAM wParam, LPARAM lParam);

	BOOL	m_fnMainSequence(const CString strModelID, const CString strPanelID, const CString strLotID, TCHAR* strSaveDrive,
		tAlignInfo stCamAlignInfo[MAX_CAMERA_COUNT], CWriteResultInfo& WrtResultInfo, ResultPanelData& ResultPanelData,
		cv::Mat MatOriginImg[][MAX_CAMERA_COUNT], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT], cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], tCHoleAlignInfo& tCHoleAlignData,
		bool bpInspectEnd[][MAX_CAMERA_COUNT], bool bAlignEnd[MAX_CAMERA_COUNT], bool& bChkDustEnd, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool bUseInspect, bool& bIsHeavyAlarm,
		ENUM_INSPECT_MODE eInspMode, STRU_LabelMarkInfo& labelMarkInfo, STRU_IMAGE_INFO* pStImageInfo = NULL);

	BOOL	CheckImageIsNormal(CString strPanelID, CString strDrive, cv::Mat& MatOrgImage, cv::Mat& MatDrawImage, int nRatio, int nImageNum, int nCameraNum, int nStageNo,
		tAlignInfo stCamAlignInfo[MAX_CAMERA_COUNT], ResultBlob_Total* pResultBlobTotal, double dCamResolution, double dPannelSizeX, double dPannelSizeY,
		bool bAlignEnd[MAX_CAMERA_COUNT], bool& bChkDustEnd, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool& bIsHeavyAlarm,
		ENUM_INSPECT_MODE eInspMode);

	int		InspectImage(const CString strModelID, const CString strPanelID, const CString strDrive,
		cv::Mat MatOriginImg[][MAX_CAMERA_COUNT], cv::Mat& MatDrawBuffer, cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], tCHoleAlignInfo& tCHoleAlignData, STRU_LabelMarkInfo& labelMarkInfo,
		ResultBlob_Total* pResultBlob_Total, const int nImageNum, const int nCameraNum, bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtResultInfo, const double* _mtp);

	int		WriteResultData(const CString strPanelID, const CString strDrive,
		cv::Mat MatResult[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT],
		CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, ENUM_INSPECT_MODE eInspMode);

	int		WriteBlockResultData(const CString strPanelID, const CString strDrive,
		cv::Mat MatResult[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT],
		CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, ENUM_INSPECT_MODE eInspMode);


	void	SaveCropImage(CString strPanelID, CString strDrive, cv::Mat(*MatResult)[MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT], ResultPanelData& resultPanelData, ENUM_INSPECT_MODE eInspMode);

	//保存结果图像
	virtual bool	DrawDefectImage(CString strPanelID,
		cv::Mat MatResult[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT],
		ResultPanelData& resultPanelData) = 0;
};
