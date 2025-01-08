

//

#pragma once

#ifndef __AFXWIN_H__
#error"在为PCH包含此文件之前,请先包含'stdafx.h'。"
#endif

#include "resource.h"		//主符号。

#include "TaskInterfacer.h"
#include "Config.h"
#include "CPUTimer.h"
#include "..\..\commonheaders\Structure.h"
#include "InspectPanel.h"
#include "Define.h"
#include "LogWriter.h"
#include "VSAlgorithmTaskDlg.h"
#include "DefineInterface.h"
#include "DiskInfo.h"

#include <map>
#include <list>

#define DRIVE_CHECK_INTERVAL 30000

#include "DefineInterface.h"

//更改为同时删除文件和子目录的函数
inline void DeleteAllFilesWithDirectory(CString dirName, bool bRecurse = FALSE)
{
	CFileFind finder;

	BOOL bWorking = finder.FindFile((CString)dirName + "/*.*");

	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		if (finder.IsDots())
		{
			continue;
		}
		else if (finder.IsDirectory())
		{
			//返回呼叫
			DeleteAllFilesWithDirectory(finder.GetFilePath(), TRUE);
		}
		else
		{
			if (!finder.IsSystem())
				DeleteFile(finder.GetFilePath());
		}
	}
	finder.Close();
	if (bRecurse)
		RemoveDirectory(dirName);
}

//
class CVSAlgorithmTaskApp : public CWinApp
{
public:
	CVSAlgorithmTaskApp();
	virtual ~CVSAlgorithmTaskApp();

	//重定义。
public:
	virtual BOOL InitInstance();

	//实现。
	CInspPanel					InspPanel;

	BOOL						m_bExecIPC;				//IPC Trhead运行Flag
	BOOL						m_bExecDisk;			//Check Disk Trhead运行Flag

	//需要全局访问的变量和函数在此处声明后theApp。使用
// Variable

	CRITICAL_SECTION			m_SVICoordFile;				//	SVI专用关键部分

	CConfig						m_Config;
	TaskInterfacer* m_AlgorithmTask;
	int							m_nDefectCnt;

	STRU_INFO_GRAB_STEP* m_pGrab_Step;				//Grab_step=图像数量				

	STRU_INFO_PAD** m_pPad_Area;

	std::map<std::string, cv::Mat> m_polNumTemplates;
	std::map<std::string, cv::Mat> m_polSignTemplates;;

	CRITICAL_SECTION			m_csCntFileSafe;

	CRITICAL_SECTION			m_csJudgeRepeatCount;
	BOOL						m_bExecAlgoThrBusy;		//检查算法线程是否超时标志 hjf
	// Function
	// Visual Station Thread
	static UINT					ThreadConnectVSServer(LPVOID pParam);

	void						StartInitialize();			//开始初始化,(从头开始)
	bool						InitializeVision();
	void						ExitVision() { InspPanel.ExitVision(); DeletePadAreaInfo(); };

	//写日志的Wrapper函数
	//日志类型,是否发送GUI日志,是否显示自己的UI日志,日志
	void						WriteLog(const ENUM_KIND_OF_LOG eLogID, const ENUM_LOG_LEVEL eLogLevel, const BOOL bSendGUITask, const BOOL bTraceList, TCHAR* pszLog, ...);

	//算法参数加载-XML
	bool						ReadAlgorithmParameter(TCHAR* strModelPath, TCHAR* strCornerEdgePath = NULL);
	//PAD扫描图像&&坐标数据参数加载-XML&&BMP
	bool						ReadPadInspParameter(TCHAR* strModelPath);
	//加载判定参数
	bool						ReadJudgeParameter(TCHAR* strModelPath);
	//加载自定义筛选器
	bool						ReadUserDefinedFilter(TCHAR* strModelPath);
	//代表不良评选排名参数加载
	bool						ReadDefectClassify(TCHAR* strModelPath);
	//读Merge配方
	bool						ReadMergeRules(TCHAR* strModelPath);
	bool						ReadSingleMergeRule(std::string fileName, std::vector<std::vector<std::string>>& vMergeLogic);

	//读PolMark模版
	bool						ReadPolMarkTemplates(TCHAR* strModelPath);

	//检查硬盘,每次在初始Align中运行一次
	void						CheckDrive();
	//加载分区参数
	bool ReadPartitionBlockParameter(TCHAR* strModelPath);
	// Get Func
	//检查算法线程超时退出 hjf
	static UINT CVSAlgorithmTaskApp::AlgoThreadTimeOutCheck(LPVOID pParam);
	BOOL						GetIPCState() { return m_bIPCConnect; };
	int							GetGrabStepCount() { return m_nParamCount; };
	int							GetCameraCount(int nGrabStep) { return m_pGrab_Step[nGrabStep].nCamCnt; };
	int							GetROICnt(int nImageNum, int nCamIndex) { return m_pGrab_Step[nImageNum].stInfoCam[nCamIndex].nROICnt; };
	TCHAR* GetROIName(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].ROI[nROIIndex].strROIName; };

	BOOL						GetUseGrabStep(int nGrabStep) { return m_pGrab_Step[nGrabStep].bUse; };

	CRect						GetFilterROI(int nGrabStep, int nCamIndex, int nROIIndex, int nRatio) {
		CRect rcRet(m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].NonROI[nROIIndex].rectROI.left * nRatio, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].NonROI[nROIIndex].rectROI.top * nRatio, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].NonROI[nROIIndex].rectROI.right * nRatio, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].NonROI[nROIIndex].rectROI.bottom * nRatio); \
			return rcRet;
	};
	BOOL						GetUseFilterROI(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].NonROI[nROIIndex].bUseROI; };
	int							GetFilterROICnt(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nNonROICnt; };
	TCHAR* GetFilterROIName(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].NonROI[nROIIndex].strROIName; };

	CRect						GetRndROI(int nGrabStep, int nCamIndex, int nROIIndex, int nRatio) {
		CRect rcRet(m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].RndROI[nROIIndex].rectROI.left * nRatio, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].RndROI[nROIIndex].rectROI.top * nRatio, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].RndROI[nROIIndex].rectROI.right * nRatio, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].RndROI[nROIIndex].rectROI.bottom * nRatio); \
			return rcRet;
	};
	int							GetRndROICnt(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nRndROICnt; };


	//Mark yuxuefei start
	CRect						GetMarkROI(int nGrabStep, int nCamIndex, int nROIIndex) {
		CRect rcRet(m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].MarkROI[nROIIndex].rectROI.left, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].MarkROI[nROIIndex].rectROI.top, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].MarkROI[nROIIndex].rectROI.right, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].MarkROI[nROIIndex].rectROI.bottom); \
			return rcRet;
	};

	BOOL						GetUseMarkROI(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].MarkROI[nROIIndex].bUseROI; };


	int							GetMarkROICnt(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nMarkROICnt; };

	TCHAR* GetMarkROIName(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].MarkROI[nROIIndex].strROIName; };

	//Mark yuxuefei end


	//Label yuxuefei start
	CRect						GetLabelROI(int nGrabStep, int nCamIndex, int nROIIndex) {
		CRect rcRet(m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].LabelROI[nROIIndex].rectROI.left, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].LabelROI[nROIIndex].rectROI.top, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].LabelROI[nROIIndex].rectROI.right, \
			m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].LabelROI[nROIIndex].rectROI.bottom); \
			return rcRet;
	};

	BOOL						GetUseLabelROI(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].LabelROI[nROIIndex].bUseROI; };


	int							GetLabelROICnt(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nLabelROICnt; };

	TCHAR* GetLabelROIName(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].LabelROI[nROIIndex].strROIName; };

	//Label yuxuefei end



	//17.10.24[Round]-区域设置
//int						GetRoundROICnt(int nGrabStep, int nCamIndex)					{	return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nRndROICnt			;};	
	INSP_AREA* GetRoundROI(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].RndROI; };

	// polmark add
	INSP_AREA* GetPolMarkROI(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].PolMarkROI; };
	int GetPolMarkROICnt(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nPolMarkROICnt; };

	//19.03.05[Camera Hole]-设置区域
	BOOL						GetUseCHoleROI(int nGrabStep, int nCamIndex, int nROIIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].HoleROI->bUseROI; }; //choi 21.11.02

	int							GetCHoleROICnt(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].nHoleROICnt; };

	INSP_AREA* GetCHoleROI(int nGrabStep, int nCamIndex) { return m_pGrab_Step[nGrabStep].stInfoCam[nCamIndex].HoleROI; };

	BOOL						GetUseAlgorithm(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum)
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].bAlgorithmUse;
	};
	BOOL						GetUseDefectFilterParam(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum, int nDefectItemNum)
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].stDefectItem[nDefectItemNum].bDefectItemUse;
	};
	double* GetAlgorithmParameter(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum)					//在Teaching中设置的ROI中验证Defect List检查。
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].dPara;
	};

	STRU_DEFECT_ITEM* GetDefectFilteringParam(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum)
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].stDefectItem;
	};
	////////////////////////////////////////////////
	//获取分区判定参数 [hjf]
	stPanelBlockJudgeInfo* GetBlockDefectFilteringParam(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum)
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].stBlockDefectItem;
	};

	int GetnBlockCountX(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum)
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].nBlockCountX;
	};

	int GetnBlockCountY(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum)
	{
		return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].nBlockCountY;
	};

	int GetnBlockCountX()
	{
		return BlockX;
	};

	int GetnBlockCountY()
	{
		return BlockY;
	};

	void SetBlockCountX(int BlockCountX)
	{
		BlockX = BlockCountX;
		return;
	};

	void SetBlockCountY(int BlockCountY)
	{
		BlockY = BlockCountY;
		return;
	};
	//临时函数，用于测试stPanelBlockJudgeInfo
	// 实现STRU_DEFECT_ITEM 复制到stPanelBlockJudgeInfo->STRU_DEFECT_ITEM,因为stPanelBlockJudgeInfo初始化函数还没实现[hjf]
	void SetBlockDefectFilteringParam(int nImageNum, int nCameraNum, int nROINum, int nAlgorithmNum) {
		stPanelBlockJudgeInfo* stBlockParams = m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].stBlockDefectItem;
		STRU_DEFECT_ITEM* stParams = m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].stDefectItem;
		int BlockX = GetnBlockCountX();
		int BlockY = GetnBlockCountY();
		m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].nBlockCountX = BlockX;
		m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[nAlgorithmNum].nBlockCountY = BlockY;
		for (int i = 0; i < BlockX * BlockY; i++)
		{
			stBlockParams[i].bBlockUse = true;
			stBlockParams[i].nBlockNum = i;
			memcpy(&stBlockParams[i].stDefectItem, &stParams, sizeof(STRU_DEFECT_ITEM) * MAX_MEM_SIZE_E_DEFECT_NAME_COUNT);
		}
	}
	//////////////////////////////////////////////////

	int							GetInsTypeIndex(int nImageNum, int nCameraNum, int nROINum, CString strAlgName);
	int							GetDefectItemCount() { return MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; };
	int							GetDefectFilterCount() { return m_nDefectFilterParamCount; };
	int							GetImageClassify(int nImageNum) { return m_pGrab_Step[nImageNum].eImgClassify; };		// ENUM_IMAGE_CLASSIFY -> int
	int							GetImageNum(int eImgClassify) {
		for (int nImageNum = 0; nImageNum < MAX_GRAB_STEP_COUNT; nImageNum++)	\
		{																	\
			if (m_pGrab_Step[nImageNum].eImgClassify == eImgClassify)		\
				return nImageNum;											\
		}
		return -1
			;
	};

	STRU_INFO_PAD** GetPadAreaInfo() { return m_pPad_Area; };
	TCHAR* GetGrabStepName(int nGrabStepNo) { return m_pGrab_Step[nGrabStepNo].strGrabStepName; };
	//AD Defect&Align Parameter必须断开。目前正在使用Common Parameter。
	BOOL						GetUseFindDefectAD(int nImageNum, int nCameraNum) { return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].bUseAD; };
	double* GetFindDefectADParameter(int nImageNum, int nCameraNum) { return m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].dADPara; };
	double* GetAlignParameter(int nCameraNum) { return m_dAlignParam; };
	ST_COMMON_PARA* GetCommonParameter() { return &m_stCommonPara; };

	int							GetDefectTypeIndex(const CString strDefTypeName) {
		map<CString, UINT>::const_iterator iter = m_MapDefItemList.find(strDefTypeName);	\
			if (iter != m_MapDefItemList.end())	return iter->second;	\
			else	return -1;	\
	};
	CString						GetDefectTypeName(const UINT nDefTypeIndex) {
		map<CString, UINT>::iterator iter;	\
			for (iter = m_MapDefItemList.begin(); iter != m_MapDefItemList.end(); iter++)	\
				if (iter->second == nDefTypeIndex)	return iter->first;
		return _T("NOT_FOUND");	\
	};
	TCHAR* GetDefectSysName(const UINT nDefTypeIndex) { return m_stDefClassify[nDefTypeIndex].strDefectName; };
	TCHAR* GetDefectCode(const UINT nDefTypeIndex) { return m_stDefClassify[nDefTypeIndex].strDefectCode; };
	int							GetAlgorithmIndex(const CString strAlgorithmName) {
		map<CString, UINT>::const_iterator iter = m_MapAlgList.find(strAlgorithmName);	\
			if (iter != m_MapAlgList.end())	return iter->second;	\
			else								return -1;	\
	};
	CString						GetAlgorithmName(const UINT nAlgorithmIndex) {
		map<CString, UINT>::iterator iter;	\
			for (iter = m_MapAlgList.begin(); iter != m_MapAlgList.end(); iter++)	\
				if (iter->second == nAlgorithmIndex)	return iter->first;
		return _T("NOT_FOUND");	\
	};

	TCHAR* GetCurModelID() { return m_strCurModelID; };

	//17.11.03-导入[Round]注册保存路径
	TCHAR* GetRoundPath() { return m_strModelPath; };

	TCHAR* GetCurInspRecipePath() { return m_strCurInspRecipePath; };

	CWriteResultInfo* GetCurWorkCoordInfo() { return &m_WrtResultInfo; };

	CString						GetCurStepFileName(TCHAR* strFindeFolder, TCHAR* strFindFileName);

	std::vector<stPanelJudgePriority> GetPanelJudgeInfo() { return m_vStPanelJudge; };

	std::vector<stUserDefinedFilter> GetUserDefinedFilter() { return m_vStUserDefinedFilter; };
	std::vector<std::vector<std::string>>* GetMergeLogic(int index) { return &m_vMergeLogics[index]; }
	int GetMergeLogicCount() { return m_vMergeLogics.size(); }

	int							GetPanelJudgeIndex(CString strJudge) {
		vector<stPanelJudgePriority>::iterator iter;	int nIndex = 0; \
			for (iter = m_vStPanelJudge.begin(); iter != m_vStPanelJudge.end(); iter++, nIndex++)	\
				if (iter->strGrade == strJudge)	return nIndex;
		return -1;	\
	};
	std::list<RepeatDefectInfo>* GetRepeatDefectInfo() { return m_listRepeatDefInfo; };		//装入现有的连续不良列表

	stPanelJudgeInfo* GetReportFilter(int nJudge) { return m_vStPanelJudge[nJudge].stFilterInfo; };

	int							CheckImageRatio(UINT nCurRatio, int nDstWidth, int nDstHeight, int nSrcWidth, int nSrcHeight);
	int							GetDefectRank(int nDefectTypeNum) { return m_nDefectRank[nDefectTypeNum]; };
	int							GetDefectGroup(int nDefectTypeNum) { return m_nDefectGroup[nDefectTypeNum]; };

	// Set Func
	void						SetIPCState(BOOL bState) { m_bIPCConnect = bState; };
	void						SetGrabStepCount(int nParamCount) { m_nParamCount = nParamCount; };

	void						SetPadAreaInfo(STRU_INFO_PAD** pPad_Area) { m_pPad_Area = pPad_Area/*memcpy(m_pPad_Area, &pPad_Area, sizeof(STRU_INFO_PAD) * 5)*/; };

	//添加UI-System Interface函数
	double						CallFocusValue(Mat matSrc, CRect rect);

	void						SetCurModelID(TCHAR* strModelID) { _tcscpy_s(m_strCurModelID, strModelID); };

	void						SetCurInspRecipePath(TCHAR* strPath) { _tcscpy_s(m_strCurInspRecipePath, strPath); };

	//17.11.03-设置[Round]注册保存路径
	void						SetRoundPath(TCHAR* strPath) { _tcscpy_s(m_strModelPath, strPath); };

	void						SetWorkCoordInfo(double dPanelSizeX, double dPanelSizeY, int nCurWorkDirection, int nCurWorkOrgPos,
		int nCurWorkOffsetX, int nCurWorkOffsetY,
		int nCurDataDrection, int nCurGDOriginPos,
		int nCurGDOffsetX, int nCurGDOffsetY,
		double dCurGatePitch, double dCurDataPitch,
		double* dCurResolution) {
		m_WrtResultInfo.SetWorkCoordInfo(
			dPanelSizeX, dPanelSizeY,
			nCurWorkDirection, nCurWorkOrgPos, nCurWorkOffsetX, nCurWorkOffsetY,
			nCurDataDrection, nCurGDOriginPos,
			nCurGDOffsetX, nCurGDOffsetY,
			dCurGatePitch, dCurDataPitch,
			dCurResolution);
	};
	void						SetAlignParameter(double* dAlignPara) { memcpy(m_dAlignParam, dAlignPara, sizeof(double) * MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT); };
	void						SetCommonParameter(ST_COMMON_PARA* pStCommonPara) { memcpy(&m_stCommonPara, pStCommonPara, sizeof(ST_COMMON_PARA)); };
	void						SetCommonParameter(BOOL bSaveFlag) { m_stCommonPara.bIFImageSaveFlag = bSaveFlag; };	//GVO要求如果是Auto,请修改中间图像的安全记录-190425YWS

	void						SetAlgoritmList(map<CString, UINT> AlgList) { m_MapAlgList = AlgList; };
	void						SetDefectItemList(map<CString, UINT> DefItemList) { m_MapDefItemList = DefItemList; };
	void						SetGrabStepInfo(STRU_INFO_GRAB_STEP* pGrabStepInfo) { memcpy(m_pGrab_Step, pGrabStepInfo, sizeof(STRU_INFO_GRAB_STEP) * MAX_GRAB_STEP_COUNT); };

	void						SetPanelJudgeInfo(std::vector<stPanelJudgePriority> vPanelJudge) { m_vStPanelJudge = vPanelJudge; };
	void						SetUserDefinedFilter(std::vector<stUserDefinedFilter> vUserFilter) { m_vStUserDefinedFilter = vUserFilter; };
	void						SetReportFilter(stReportFilter* pStRF) { memcpy(&m_StReportFilter, pStRF, sizeof(stReportFilter)); };

	void						SetDefectClassify(stDefClassification* stDefectClassify) { memcpy(m_stDefClassify, stDefectClassify, sizeof(stDefClassification) * MAX_MEM_SIZE_E_DEFECT_NAME_COUNT); };
	stDefClassification* GetDefectClassify() { return m_stDefClassify; };
	void						SetDefectRank(int* DefectRank) { memcpy(m_nDefectRank, DefectRank, sizeof(m_nDefectRank)); };
	void						SetDefectGroup(int* DefectGroup) { memcpy(m_nDefectGroup, DefectGroup, sizeof(m_nDefectGroup)); };
	bool WriteResultFile(CString strPanelID, CString strFilePath, CString strFileName, CString strColumn, TCHAR* strResult);

	void						DeletePadAreaInfo() {
		if (m_pPad_Area != NULL)
		{
			for (int nPadInfoCnt = 0; nPadInfoCnt < E_PAD_AREA_COUNT; nPadInfoCnt++)	delete[] m_pPad_Area[nPadInfoCnt];
			delete[] m_pPad_Area; m_pPad_Area = NULL; m_pPad_Area = (NULL);
		}
	};
	//18.03.12-添加Merge Tool get函数
	BOOL GetMergeToolUse();

	// File Read
	void						ReadRepeatDefectInfo();
	DECLARE_MESSAGE_MAP()

private:
	// Variable
	CVSAlgorithmTaskDlg* m_pDlg;
	CWinThread* m_pVSThread;
	CWinThread* m_pDiskCheckThread;
	CWinThread* m_pAlgoCheckTimeOutThread;
	BOOL						m_bIPCConnect;
	HANDLE						m_hEventIPCThreadAlive;
	HANDLE						m_hEventDiskThreadAlive;
	HANDLE						m_hEventAlgoThreadTimeOutAlive;

	double						m_dAlignParam[MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT];			//Align参数最多15个

	ST_COMMON_PARA				m_stCommonPara;

	/// <summary>
	int BlockX;
	int BlockY;
	/// </summary>

	int							m_nInspCamID;

	TCHAR						m_strCurModelID[50];

	//17.11.03-[Round]注册保存路径
	TCHAR						m_strModelPath[200];

	TCHAR						m_strCurInspRecipePath[256];

	// Function
	int							m_fnConectVisualStation();
	BOOL						m_fnInitFunc();

	int							m_nParamCount;				//图像数量
	int							m_nDefectItemCount;
	int							m_nDefectFilterParamCount;
	bool						m_bStopInspect;			//停止检查Flag

	// Log Class
	CAlgLogWriter* m_pLogWriterCam[MAX_CAMERA_COUNT];		//特定于相机的日志
	CAlgLogWriter* m_pLogWriterProc;						//进度日志(主线程和判定/结果填写线程)
	CAlgLogWriter* m_pLogWriterTact;						//Tact日志
	CAlgLogWriter* m_pLogWriterComm;						//通信日志

	map<CString, UINT>			m_MapDefItemList;						//完整Defect Item List
	map<CString, UINT>			m_MapAlgList;							//全部Algorithm List

	CWriteResultInfo			m_WrtResultInfo;						//工作坐标计算/结果数据生成类

	std::vector<stPanelJudgePriority>	m_vStPanelJudge;				//面板判定优先级
	std::vector<stUserDefinedFilter>	m_vStUserDefinedFilter;				//定制过滤器

	std::vector<std::vector<std::vector<std::string>>> m_vMergeLogics;

	stReportFilter						m_StReportFilter;				//过滤父报告
	int									m_nDefectRank[E_DEFECT_JUDGEMENT_COUNT];	///代表不良评选排名,数字越低排名越高。
	int									m_nDefectGroup[E_DEFECT_JUDGEMENT_COUNT];	///各代表群的代表不良选定标准-代表不良群Index(0:不使用代表不良功能,N:同一编号之间设置为不良群,选定代表不良)

	void						m_fnGetJudgeInfo(stPanelJudgeInfo* pStJudgeInfo, CString strVal, int nCount);

	stDefClassification			m_stDefClassify[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];

	CDiskInfo* m_pDiskInfo;
	std::vector<tDriveInfoParam>	m_tDriveInfoParam;
	static UINT					ThreadDiskCheck(LPVOID pParam);

	std::list<RepeatDefectInfo>		m_listRepeatDefInfo[eCOORD_KIND];				//同一位置连续不良Count
	bool							m_fnReadRepeatDefectInfo();						//从文件中检索以前的数据-启动程序仅需一次	
	BOOL							m_fnReadRepeatFile(CString strFilePath, std::list<RepeatDefectInfo>* pList);	//分别读取Pixel和Work坐标

	//18.03.12-Merge Tool添加变量	
	BOOL m_nInspStep;
};

extern CVSAlgorithmTaskApp theApp;
