/*****************************************************************************
  File Name		: AviInspection.h
  Version		: ver 1.0
  Create Date	: 2017.03.21
  Description	:Area对应检查线程
  Abbreviations	:
 *****************************************************************************/

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "AlgorithmInterface.h"
#include "InspMainSequence.h"
#include "MatBufferResultManager.h"	//内存管理
#include "MatResultBuf.h"			//内存管理
#include "AIReJudge.h"

 //	Class功能	:	
 //主要功能	:
 //	创建日期	:2017/02
 //	作者	:	CWH
 //	修改历史记录		:	V.1.0初始创建
 //	请参见	:	

class AviInspection :
	public InspMainSequence, AIReJudge
{
private:
	DECLARE_DYNCREATE(AviInspection)
public:
	AviInspection();
	virtual ~AviInspection(void);

	//内存管理

// 		for (int i = 0; i < nCnt; i++)

//内存管理
	CMatResultBuf* cMemResult;
	void			SetMem_Result(CMatResultBuf* data) { cMemResult = data; };
	CMatResultBuf* GetMem_Result() { return	cMemResult; };

protected:

private:
	//保存结果图像
	bool	DrawDefectImage(CString strPanelID,
		cv::Mat(*MatResult)[MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT],
		ResultPanelData& resultPanelData);

	//AVI扫描函数
	long	StartLogicAlgorithm(const CString strDrive, const tLogicPara& LogicPara,
		cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat& MatDrawBuffer,
		const int nImageNum, const int nROINumber, const int nAlgorithmNumber,
		tAlignInfo stThrdAlignInfo, ResultBlob_Total* pResultBlob_Total,
		bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtResultInfo, const double* _mtp);

	//根据Align信息进行图像校正,校正后刷新Align信息
	//AVI:Image假设旋转坐标校正
	bool	AdjustAlignInfo(tAlignInfo* pStCamAlignInfo, cv::Point* ptAdjCorner);

	//原始图像校正
	//AVI:启用旋转的情况进行调整
	bool	AdjustOriginImage(cv::Mat& MatOrgImage, cv::Mat& MatDrawImage, tAlignInfo* pStAlignInfo);

	//外围处理
	long	makePolygonCellROI(const tLogicPara& LogicPara, cv::Mat& MatDrawBuffer, tAlignInfo& stThrdAlignInfo, STRU_LabelMarkInfo& labelMarkInfo, int nImageNum, int nCameraNum, double* dAlgPara, int nAlgImg, int nRatio);

	//AVI判定函数
	bool	Judgement(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT], tCHoleAlignInfo& tCHoleAlignData,
		const CString strModelID, const CString strLotID, const CString strPanelID, const CString strDrive, int nRatio,
		ENUM_INSPECT_MODE eInspMode, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], bool bUseInspect, int nStageNo);

	//AI 复判
	bool	Judgement_AI(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT], tCHoleAlignInfo& tCHoleAlignData,
		const CString strModelID, const CString strLotID, const CString strPanelID, const CString strDrive, int nRatio,
		ENUM_INSPECT_MODE eInspMode, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], bool bUseInspect, int nStageNo);

	//修复设备使用的坐标值和代码判定
	bool	JudgementRepair(const CString strPanelID, ResultPanelData& resultPanelData, CWriteResultInfo& WrtResultInfo);

	//如果有多个Point->判定Line
	bool	JudgementPointToLine(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, const int nImageWidth, const int nImageHeight);

	//删除报告
	bool	JudgementDelReport(ResultPanelData& resultPanelData);

	//重新检查Spot分类
	bool	JudgementSpot(ResultPanelData& resultPanelData);

	//删除钢线周围的弱线
	bool	JudgementDeletLineDefect(ResultPanelData& resultPanelData, double* dAlignPara);

	// Line Classification
	bool	JudgementClassifyLineDefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT]);

	// 6.39QHD Notch Y Line Delete
	bool	JudgementNotchDefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara);

	//基于Black Pattern的Merge判定
	bool	JudgementBlackPatternMerge(ResultPanelData& resultPanelData);

	//Crack判定
	bool	JudgementCrack(ResultPanelData& resultPanelData);

	//DGS判定
	bool	JudgementNewDGS(ResultPanelData& resultPanelData);

	// Vth Pattern DGS
	bool	JudgementDGS_Vth(ResultPanelData& resultPanelData);

	// Same Pattern Defect Merge
	bool	JudgementSamePatternDefectMerge(ResultPanelData& resultPanelData);

	// Weak Plan B Test
	bool	JudgementSpecialGrayLineDefect(ResultPanelData& resultPanelData);

	//PCD CRACK判定(非检测区域5.99")
	bool	JudgementPCDCrack(ResultPanelData& resultPanelData, double* dAlignPara);

	//消除PNZ-Camera Tap导致的LINE过检
	bool	JudgementCameraTapOverKill(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara);

	// PNZ - GV Comparison
	bool	DefectLDRCompair(cv::Mat matSrcImage, cv::Rect rectTemp, double& Left_MeanValue, double& Defect_MeanValue, double& Right_MeanValue, double& Defect_MaxGV);

	//获取PNZ-Camera Tap信息
	bool	CameraTapInfo(int CameraType, vector<int>& Direction, vector<int>& Position, vector<double>& GV, vector<double>& LR_Diff_GV);

	//获取PNZ-Modle Norch属性
	bool 	GetModelNorchInfo(ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], vector<int>& NorchIndex, CPoint& OrgIndex);
	//yuxuefei
	bool JudgementZARADefect(ResultPanelData&resultPanelData,cv::Mat MatOrgImage[][MAX_CAMERA_COUNT],double*dAlignPara);
	//yuxuefei
	bool JudgementPSMuraBrightPointDefect(ResultPanelData&resultPanelData,cv::Mat MatOrgImage[][MAX_CAMERA_COUNT],double*dAlignPara);
	//yuxuefei
	bool JudgementDUSTDOWNDefect(ResultPanelData&	resultPanelData,cv::Mat MatOrgImage[][MAX_CAMERA_COUNT],double*dAlignPara);
	//-重复数据删除
	bool	DeleteOverlapDefect(ResultPanelData& resultPanelData, double* dAlignPara);

	//根据CKI-Defect size的Judge
	bool    Judge_DefectSize(ResultPanelData& resultPanelData, double* dAlignPara);

	//PNZ-Dimple删除算法
	bool	DeleteOverlapDefect_DimpleDelet(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara = NULL);

	// PNZ - TOP3 Max GV Calculator
	bool	NewMaxGVMethold(Mat matSrcImage, double OldMaxGV, double& NewMaxGV, int nTopCountGV);

	//PNZ-Dimple删除算法
	bool	DeleteOverlapDefect_SpecInSpotDelet(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara = NULL);

	bool	DeleteOverlapDefect_BlackHole(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara = NULL);

	//PNZ-Dust删除算法
	bool	DeleteOverlapDefect_DustDelet(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara = NULL);

	//PNZ-Black Pattern强明点周围的弱明点删除算法
	bool	DeleteOverlapDefect_BlackSmallDelet(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//-Point-Point重复数据删除
	bool	DeleteOverlapDefect_Point_Point(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//-Point-Line重复数据删除
	bool	DeleteOverlapDefect_Point_Line(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//-Point-Mura重复数据删除
	bool	DeleteOverlapDefect_Point_Mura(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//-Line-Mura重复数据删除
	bool	DeleteOverlapDefect_Line_Mura(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//-Mura-Mura重复数据删除
	bool	DeleteOverlapDefect_Mura_Mura(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//CKI-RGB Bright Point-White Spot Mura重复数据删除//B11名点19.08.20
	bool	DeleteOverlapDefect_White_Spot_Mura_RGBBlk_Point(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//CKI-Black hole重复判定//B11得分20.04.07
	bool    DeleteOverlapDefect_Black_Mura_and_Judge(ResultPanelData& resultPanelData, double* dAlignPara = NULL);

	//在CKI-G64模式中,使用标准偏差来判定明点和白点村

	//17.09.11-混色多->数量多时判定ET
	bool	JudgementET(ResultPanelData& resultPanelData, double* dAlignPara, CString strPanelID);

	//17.09.11-边缘部分亮点->Pad Bright判定
	bool	JudgementEdgePadBright(ResultPanelData& resultPanelData, double* dAlignPara);

	//17.11.29-外围信息(AVI&SVI其他工具)
	bool	JudgeSaveContours(ResultPanelData& resultPanelData, wchar_t* strContourTxt);

	//保存Mura轮廓信息
	bool	JudgeSaveMuraContours(ResultPanelData& resultPanelData, wchar_t* strContourTxt);

	//17.09.25-相邻&群集判定
	bool	JudgeGroup(ResultPanelData& resultPanelData, cv::Mat(*MatDraw)[MAX_CAMERA_COUNT], double* dAlignPara, double dResolution)
	{
		return TRUE;
	}

	//17.11.29-相邻&群集判定
	bool	JudgeGroupTEST(ResultPanelData& resultPanelData, cv::Mat(*MatDraw)[MAX_CAMERA_COUNT], double* dAlignPara, double dResolution);

	//PNZ 19.01.11-组判定的最新版本
	bool	JudgeGroupJudgment(ResultPanelData& resultPanelData, cv::Mat(*MatDraw)[MAX_CAMERA_COUNT], double* dAlignPara, double dResolution);

	//19.04.04-CHole AD判定和二次判定
	bool	JudgeCHoleJudgment(ResultPanelData& resultPanelData, tCHoleAlignInfo tCHoleAlignData, double* dAlignPara);

	//从Casting-stDefectInfo中提取所需的部分,并将其装载到ResultPanelData中
	bool	GetDefectInfo(CWriteResultInfo WrtResultInfo, ResultDefectInfo* pResultDefectInfo, stDefectInfo* pResultBlob, int nBlobCnt, int nImageNum, int nCameraNum, int nRatio);

	//在Align之前检查AD
	long	CheckAD(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nImageNum, int nCameraNum, double* dResult, int nRatio);


	//yuxuefei for Mark
	long	MarkProcess(cv::Mat matSrcBuf, int nImageNum, int  nCameraNum, tAlignInfo& stCamAlignInfo);

	//yuxuefei for Label
	long	LabelProcess(cv::Mat matSrcBuf, int nImageNum, int nCameraNum, tAlignInfo& stCamAlignInfo);
	
	// 检查PG压接导致的异显 hjf
	long	CheckPGConnect(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nImageNum, int nCameraNum, double* dResult, cv::Point* cvPt);
	//ROI GV检查
	long	CheckADGV(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nStageNo, int nImageNum, int nCameraNum, int nRatio, cv::Point* ptCorner, ResultBlob_Total* pResultBlobTotal, double* dMeanResult,
		bool& bChkDustEnd, bool& bNeedRetry, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool& bIsHeavyAlarm, ENUM_INSPECT_MODE eInspMode);

	//17.11.24-Rolution计算
	double	calcResolution(CWriteResultInfo WrtResultInfo);

	//添加17.12.18相同位置错误判定序列	
	bool	JudgementRepeatCount(CString strPanelID, ResultPanelData& resultPanelData);

	bool	m_fnSaveRepeatDefInfo(std::list<RepeatDefectInfo>* pListRepeatInfo);

	//////////////////////////////////////////////////////////////////////////choi goldencell 05.08
	bool JudgementMuraNormalClassification(ResultPanelData& resultPanelData, double* dAlignPara);
	bool JudgementMuraNormalT3Filter(ResultPanelData& resultPanelData, tCHoleAlignInfo tCHoleAlignData, double* dAlignPara);

	// 24.07.05 - 缺陷复合判定（去重）
	bool ApplyMergeRule(ResultPanelData& resultPanelData);

private:
	bool PrepareAITask(cv::Mat dics, double dicsRatio, int cropExpand, stDefectInfo* pResultBlob, AIReJudgeParam& aiParam,
		CString strPanelID, const int nImageNum, const int nAlgNum,
		std::vector<TaskInfoPtr>& taskList);
};
