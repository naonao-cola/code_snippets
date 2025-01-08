#include "stdafx.h"
#include <ppl.h>
#include "InspMainSequence.h"
#include "ImageSave.h"
#include "../../visualstation/CommonHeader/Class/CostTime.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif

using namespace Concurrency;

InspMainSequence::InspMainSequence(void)
{
}

InspMainSequence::~InspMainSequence(void)
{
}

//----------------------------------------------------------------------------------------------------
//	 1. 函数名:	On Start Grab
//	 2. 函数功能:	检查序列(Grab->Inspection->Result)
//	 3. 参数:	lParam:检查方向E_DIR_FORWARD or E_DIR_BACK
//					
//					
//					
//					
//	 4. 返回值:	完成:0
//				失败:非0
//	 5. 创建日期:2015/02/10
//	 6. 作者:	KYT
//	 7. 修改历史记录:	
//	 8. 注意:	
//----------------------------------------------------------------------------------------------------
LRESULT InspMainSequence::OnStartInspection(WPARAM wParam, LPARAM lParam) //2016.11.04 sgkim
{
	m_bBusy = true;
	int nRet = 0;

	tInspectThreadParam* pInspectThreadParam = (tInspectThreadParam*)wParam;		// 通用信息
	STRU_IMAGE_INFO* pStImageInfo = (STRU_IMAGE_INFO*)lParam;		// 特定于映像的信息

	//调用检查-判定-结果填写函数
	nRet = m_fnMainSequence(theApp.GetCurModelID(), pStImageInfo->strPanelID, pStImageInfo->strLotID, pInspectThreadParam->strSaveDrive,
		pInspectThreadParam->stCamAlignInfo, pInspectThreadParam->WrtResultInfo, pInspectThreadParam->ResultPanelData,
		pInspectThreadParam->MatOrgImage, pInspectThreadParam->MatDraw, pInspectThreadParam->MatResult, pInspectThreadParam->tCHoleAlignData,
		pInspectThreadParam->bInspectEnd, pInspectThreadParam->bAlignEnd, pInspectThreadParam->bChkDustEnd, pInspectThreadParam->bIsNormalDust, pInspectThreadParam->bUseDustRetry, pInspectThreadParam->nDustRetryCnt, pInspectThreadParam->bUseInspect, pInspectThreadParam->bHeavyAlarm,
		pInspectThreadParam->eInspMode, pInspectThreadParam->tLabelMarkInfo, pStImageInfo);

	if (pStImageInfo->nImageNo == 0 && pStImageInfo->nCameraNo == 0)
		//SAFE_DELETE(pInspectThreadParam);
		pInspectThreadParam->clear();	//20.07.03

	SAFE_DELETE(pStImageInfo);

	Sleep(30);

	m_bBusy = false;

	return 0;
}

//----------------------------------------------------------------------------------------------------
//	 1. 函数名:	On Start Save Image
//	 6. 作者:
//   7. 功能:保存原始画面
//----------------------------------------------------------------------------------------------------
LRESULT InspMainSequence::OnStartSaveImage(WPARAM wParam, LPARAM lParam) //2016.11.04 sgkim
{
	m_bBusy = true;

	cv::Mat* MatOrgImage = (cv::Mat*)wParam;
	tImageInfo* imageInfo = (tImageInfo*)lParam;
	CostTime ct(true);

	ImageSave(*MatOrgImage, (TCHAR*)(LPCTSTR)*imageInfo->filePath);

	LogSendToUI::getInstance()->SendAlgoLog(EModuleType::ALGO, ELogLevel::INFO_, EAlgoInfoType::IMG_WRITE_END,
		ct.get_cost_time(), 0, imageInfo->panelId, theApp.m_Config.GetPCNum(), imageInfo->imageNo, -1,
		_T("[%s]保存图片结束，耗时:%d(ms) PID:%s"), theApp.GetGrabStepName(imageInfo->imageNo), ct.get_cost_time(), imageInfo->panelId);

	m_bBusy = false;

	return 0;
}

//----------------------------------------------------------------------------------------------------
//	 1. 函数名:	WriteResultData
//	 2. 函数功能:创建向	服务器报告的结果文件。
//	 3. 参数:	
//					
//					
//					
//					
//	 4. 返回值:	TRUE	:Cell存在
//				FALSE	:没有Cell
//	 5. 创建日期:2015/02/10
//	 6. 作者:	KYT
//	 7. 修改历史记录:	
//	 8. 注意:必须从	Vision Server接收信息。
//----------------------------------------------------------------------------------------------------
int InspMainSequence::WriteResultData(const CString strPanelID, const CString strDrive,
	cv::Mat MatResult[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT],
	CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, ENUM_INSPECT_MODE eInspMode)
{
	int					nRet = 0;

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)PNL 结果写入开始."), strPanelID);

	//转到2018.01.04 Judgement()
	// 	NumberingDefect(_T("2"), strPanelID, _T("TEST"), WrtResultInfo, resultPanelData, 1);
	// 
	// 	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Numbering Defect End. Panel ID : %s"), strPanelID);

			//绘制结果图像
	DrawDefectImage(strPanelID, MatResult, MatDrawBuffer, resultPanelData);

	//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Draw Defect Image End. Panel ID : %s"), strPanelID);

	SaveCropImage(strPanelID, strDrive, MatResult, MatDrawBuffer, resultPanelData, eInspMode);

	//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Save Crop Image End. Panel ID : %s"), strPanelID);

	CCPUTimer timerSaveImage;
	timerSaveImage.Start();

	//需要吗？(时间太长了......)
	//如果需要,请转到菜单进行确认(保存为碎片画面)
	//看起来需要flag处理...
/************************************************************************
//#pragma omp parallel for
#pragma omp parallel for schedule(dynamic)
	// Save Result Image
	for (int nImageNum = 0; nImageNum < theApp.GetGrabStepCount(); nImageNum++)
	{
		if(theApp.m_pGrab_Step[nImageNum].bUse)
		{
//#pragma omp parallel for			//当前相机数量为1-2个,效率低下
			for (int nCameraNum = 0; nCameraNum < MAX_CAMERA_COUNT; nCameraNum++)
			{
				if (theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].bUse)
				{
					ImageSave(MatDrawBuffer[nImageNum][nCameraNum], _T("%s\\%s\\%d_%s_CAM%02d.jpg"), INSP_INFO_FILE_PATH, strPanelID, nImageNum, theApp.GetGrabStepName(nImageNum), nCameraNum);
				}
			}
		}
	}
************************************************************************/

	timerSaveImage.End();
	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Save Result Image tact time %.2f"), timerSaveImage.GetTime() / 1000.);

	//生成结果文件
//创建时删除-控制组请求
// 	SYSTEMTIME time;
// 	::GetLocalTime(&time);

// 	CString strTime;
// 	strTime.Format(_T("%02d%02d%02d_%02d%02d%02d"), time.wYear, time.wMonth, time.wDay,time.wHour,time.wMinute,time.wSecond);

	CString strMainPCFileName;
	//strMainPCFileName.Format(_T("E:\\IMTC\\%s_%02d\\%s_%s.txt"), theApp.m_Config.GetEqpTypeName(), theApp.m_Config.GetPCNum(), strPanelID, theApp.m_Config.GetEqpTypeName());

	CString strFinalResultDir, strFinalResultFileName, strResultAIFileName;
	strFinalResultDir.Format(_T("%s_%02d\\"), FINALRESULT_PATH, theApp.m_Config.GetPCNum());
	strFinalResultFileName.Format(_T("%s\\%s_%s_F.txt"), strFinalResultDir, strPanelID, theApp.m_Config.GetEqpTypeName());
	strMainPCFileName.Format(_T("%s\\%s_%s.txt"), strFinalResultDir, strPanelID, theApp.m_Config.GetEqpTypeName());

	CString strServerFileName, strServerFileNameAI;
	strServerFileName.Format(_T("%s\\%s\\%s.txt"), RESULT_PATH, strPanelID, strPanelID);
	strServerFileNameAI.Format(_T("%s\\%s\\%s_AI.txt"), RESULT_PATH, strPanelID, strPanelID);

	//	金亨柱
//更改	//	顺序
	CreateDirectory(strFinalResultDir, NULL);

	//无论Merge Tool flag如何,将概念更改为在每台检查PC上无条件留下结果文件180712YSS
	nRet = WrtResultInfo.WriteResultPanelData_ToMainPC(strMainPCFileName, resultPanelData);//2016.12.29 sgkim H-Project
	CopyFile(strMainPCFileName, strFinalResultFileName, FALSE);

	//	金亨柱
	//	MergeTool相关的临时处理
	//180501YSS-MergeTool相关修改(如果在AVI,SVI中使用MergeTool功能,则不在Alg测试中创建文件。在MergeTool程序中单独创建文件)
// 	if(theApp.m_Config.GetEqpType() == EQP_AVI || theApp.m_Config.GetEqpType() == EQP_SVI)
// 	{
// 		if( !theApp.GetMergeToolUse())
// 		{
// 			nRet = WrtResultInfo.WriteResultPanelData_ToMainPC(strMainPCFileName, resultPanelData);//2016.12.29 sgkim H-Project
// 			CopyFile(strMainPCFileName, strFinalResultFileName, FALSE);
// 		}
// 	}
// 	else
// 	{
// 		nRet = WrtResultInfo.WriteResultPanelData_ToMainPC(strMainPCFileName, resultPanelData);//2016.12.29 sgkim H-Project
// 		CopyFile(strMainPCFileName, strFinalResultFileName, FALSE);
// 	}

	CString strTempPath;
	strTempPath.Format(_T("%s\\%s\\%s_F.txt"), RESULT_PATH, strPanelID, strPanelID);
	CopyFile(strMainPCFileName, strTempPath, FALSE);
	CFileFind find;
	BOOL bRtn = find.FindFile(strTempPath);
	if (bRtn)
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)PNL  结果文件路径: %s"), strPanelID, strTempPath);
	else
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)PNL  创建失败!!! 结果文件路径: %s"), strPanelID, strTempPath);
	nRet = WrtResultInfo.WriteResultPanelData(strServerFileName, resultPanelData, false);

	// 写结果(保留AI复判结果)
	nRet = WrtResultInfo.WriteResultPanelData(strServerFileNameAI, resultPanelData, true);

	SYSTEMTIME time;
	::GetLocalTime(&time);
	//每个Panel的Defect Type数量
	if (GetPrivateProfileInt(_T("DEFECT"), _T("Use_Panel_Trend"), 1, INIT_FILE_PATH) == 1)
	{
		CString strPanelTrendFileName = _T("");
		strPanelTrendFileName.Format(_T("%s\\%02d%02d%02d_Trend.txt"), RESULT_PATH, time.wYear, time.wMonth, time.wDay);
		nRet = WrtResultInfo.WritePanelTrend(strPanelTrendFileName, resultPanelData.m_nDefectTrend, strPanelID, resultPanelData.m_ResultPanel.Judge);
	}

	CString strFinalDefectPath = _T("");
	strFinalDefectPath.Format(_T("%s\\%02d%02d%02d_FinalDefect.txt"), RESULT_PATH, time.wYear, time.wMonth, time.wDay);
	nRet = WrtResultInfo.WriteFinalDefect(strFinalDefectPath, resultPanelData.m_ResultPanel.nFinalDefectNum, strPanelID);
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)PNL 结果写入结束."), strPanelID);

	return nRet;
}

int InspMainSequence::WriteBlockResultData(const CString strPanelID, const CString strDrive,
	cv::Mat MatResult[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT],
	CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, ENUM_INSPECT_MODE eInspMode)
{

	int	nRet = 0;
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Segment BlockResultData Start. PanelID : %s"), strPanelID);

	ResultDefectInfo lstDefVal;
	for (int nIndex = 0; nIndex < resultPanelData.m_ListDefectInfo.GetCount(); nIndex++)
	{
		lstDefVal = resultPanelData.m_ListDefectInfo[nIndex];
		resultPanelData.m_ListDefectInfo[nIndex].nBlockNum = 0;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Segment WriteResultData End. PanelID : %s"), strPanelID);

	return nRet;
}


//----------------------------------------------------------------------------------------------------
//	 1. 函数名:	InspectImage
//	 2. 函数功能:	
//	 3. 参数:	
//	 4. 返回值:	完成:0
//				失败:非0
//	 5. 创建日期:2015/02/10
//	 6. 作者:	KYT
//	 7. 修改历史记录:	
//	 8. 注意:	
//----------------------------------------------------------------------------------------------------
int InspMainSequence::InspectImage(const CString strModelID, const CString strPanelID, const CString strDrive,
	cv::Mat MatOriginImg[][MAX_CAMERA_COUNT], cv::Mat& MatDrawBuffer, cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], tCHoleAlignInfo& tCHoleAlignData, STRU_LabelMarkInfo& labelMarkInfo,
	ResultBlob_Total* pResultBlob_Total, const int nImageNum, const int nCameraNum, bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtResultInfo, const double* _mtp)
{
	long nErrorCode = E_ERROR_CODE_TRUE;

	ENUM_KIND_OF_LOG eLogCamera = (ENUM_KIND_OF_LOG)nCameraNum;

	theApp.WriteLog(eLogCamera, eLOGLEVEL_BASIC, TRUE, FALSE, _T("{%s}(%s)[%02d][%s] 画面检测开始."), strModelID, strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));

	// CHole Align Data
	m_stThrdAlignInfo.tCHoleAlignData = &tCHoleAlignData;

	//为该图像运行Logic(对一个模式/一个相机进行所有检查)
	nErrorCode = StartLogic(strPanelID, strDrive, m_stThrdAlignInfo, MatOriginImg, MatDrawBuffer, MatResultImg,
		pResultBlob_Total, nImageNum, nCameraNum, m_nThrdID, bpInspectEnd, nRatio, eInspMode, WrtResultInfo, _mtp, labelMarkInfo);

	//如果有错误,则输出错误代码&日志
	//18.05.21-正常完成倒圆角设置时返回的值:E_ERROR_CODE_ALIGN_ROUND_SETTING
	if (nErrorCode != E_ERROR_CODE_TRUE &&
		nErrorCode != E_ERROR_CODE_ALIGN_ROUND_SETTING &&
		nErrorCode != E_ERROR_CODE_EMPTY_RGB)	// 18.05.30
	{
		//18.04.03-Err与未检查的情况相同...E级判定
		JudgeADDefect(nImageNum, nCameraNum, m_stThrdAlignInfo.nStageNo, MatOriginImg[nImageNum][nCameraNum].cols, MatOriginImg[nImageNum][nCameraNum].rows, pResultBlob_Total, E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL, eInspMode);

		theApp.WriteLog(eLogCamera, eLOGLEVEL_BASIC, TRUE, FALSE, _T("{%s}(%s)[%02d][%s]画面 开始算法检测."), strModelID, strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));
	}

	return E_ERROR_CODE_TRUE;
}

BOOL InspMainSequence::m_fnMainSequence(CString strModelID, CString strPanelID, CString strLotID, TCHAR* strSaveDrive,
	tAlignInfo stCamAlignInfo[MAX_CAMERA_COUNT], CWriteResultInfo& WrtResultInfo, ResultPanelData& ResultPanelData,
	cv::Mat MatOriginImg[][MAX_CAMERA_COUNT], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT], cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], tCHoleAlignInfo& tCHoleAlignData,
	bool bpInspectEnd[][MAX_CAMERA_COUNT], bool bAlignEnd[MAX_CAMERA_COUNT], bool& bChkDustEnd, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool bUseInspect, bool& bIsHeavyAlarm,
	ENUM_INSPECT_MODE eInspMode, STRU_LabelMarkInfo& labelMarkInfo, STRU_IMAGE_INFO* pStImageInfo)
{
	int	nCameraNum = pStImageInfo->nCameraNo;
	int	nImageNum = pStImageInfo->nImageNo;
	int nRatio = pStImageInfo->nRatio;		// 根据是否使用Pixel Shift缩放图像
	int nStageNo = pStImageInfo->nStageNo;

	//如果是E级判定,结束得太快,等待2秒190314YWS
	bool nSkipFlag = false;

	m_stThrdAlignInfo.nStageNo = nStageNo;

	CString strDrive = _T("");
	strDrive.Format(_T("%s"), strSaveDrive);
	ENUM_KIND_OF_LOG eLogCamera = (ENUM_KIND_OF_LOG)nCameraNum;	// for Camera Log

	if (strPanelID != _T(""))
	{
		//分配要生成的图像缓冲区
		//颜色(SVI)
		if (MatOriginImg[nImageNum][nCameraNum].channels() != 1)
		{
			MatDrawBuffer[nImageNum][nCameraNum] = MatOriginImg[nImageNum][nCameraNum].clone();
		}
		//黑白(AVI,APP)
		else
		{
			//对于12bit图像,转换为8bit后更改颜色
			if (MatOriginImg[nImageNum][nCameraNum].type() != CV_8U)
			{
				cv::Mat matTemp;
				MatOriginImg[nImageNum][nCameraNum].convertTo(matTemp, CV_8U, 1. / 16.);
				cv::cvtColor(matTemp, MatDrawBuffer[nImageNum][nCameraNum], COLOR_GRAY2RGB);
				matTemp.release();
			}
			else
			{
				cv::cvtColor(MatOriginImg[nImageNum][nCameraNum], MatDrawBuffer[nImageNum][nCameraNum], COLOR_GRAY2RGB);	//每个内存分配模式的完整画面
			}
		}
		CostTime ct(true);
		if (bUseInspect)
		{
			
			ResultBlob_Total* pResultBlobTotal = new ResultBlob_Total();
			pResultBlobTotal->RemoveAll_ResultBlob();
			pResultBlobTotal->SetModelID(strModelID);
			pResultBlobTotal->SetPanelID(strPanelID);

			//检查是否为正常点灯图像
			//在线程成员变量中更新线程后的线程信息(m_stThrdAlignInfo)
			long nErrorCode = CheckImageIsNormal(strPanelID, strDrive, MatOriginImg[nImageNum][nCameraNum], MatDrawBuffer[nImageNum][nCameraNum], nRatio, nImageNum, nCameraNum, nStageNo, stCamAlignInfo, pResultBlobTotal, WrtResultInfo.GetCamResolution(0), WrtResultInfo.GetPanelSizeX(), WrtResultInfo.GetPanelSizeY(), bAlignEnd, bChkDustEnd, bIsNormalDust, bUseDustRetry, nDustRetryCnt, bIsHeavyAlarm, eInspMode);

			//仅在正常对齐正常点亮的图像时运行检查序列
			if (m_stThrdAlignInfo.bAlignSuccess == true && bIsHeavyAlarm == false && nErrorCode == E_ERROR_CODE_TRUE)
			{
				double* dAlignPara = theApp.GetAlignParameter(nCameraNum);
				//算法行为
				InspectImage(strModelID, strPanelID, strDrive, MatOriginImg, MatDrawBuffer[nImageNum][nCameraNum], MatResultImg, tCHoleAlignData, labelMarkInfo,
					pResultBlobTotal, nImageNum, nCameraNum, bpInspectEnd, nRatio, eInspMode, WrtResultInfo, pStImageInfo->dPatternCIE);
				theApp.WriteLog(eLogCamera, eLOGLEVEL_BASIC, TRUE, FALSE, _T("{%s}(%s)[%02d][%s]画面 完成算法检测."), strModelID, strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));
			}
			else
			{
				theApp.WriteLog(eLogCamera, eLOGLEVEL_BASIC, TRUE, FALSE, _T("{%s}(%s)[%02d][%s]画面 跳过算法检测."), strModelID, strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));
				//如果是E级判定,结束得太快,等待2秒190314YWS
				nSkipFlag = true;
			}



			//检查结果汇总-模式结果->面板结果
			ConsolidateResult(strPanelID, strDrive, WrtResultInfo, pResultBlobTotal, ResultPanelData, nImageNum, nCameraNum, nRatio, eInspMode);

			//汇总画面缺陷特征到特征表
			////1、创建文件
			//m_FileLOGPROC->m_fnCreateFolder();
			////2、数据写入
			//m_FileLOGPROC->m_fnOnWritefile();
			////3、当前画面写入完成
			SAFE_DELETE(pResultBlobTotal);
		}
		LogSendToUI::getInstance()->SendAlgoLog(EModuleType::ALGO, ELogLevel::INFO_, EAlgoInfoType::IMG_INSP_END,
			ct.get_cost_time(), 0, strPanelID, theApp.m_Config.GetPCNum(), pStImageInfo->nImageNo, -1,
			_T("[%s]画面检测结束，耗时:%d(ms) PID:%s"), theApp.GetGrabStepName(pStImageInfo->nImageNo), ct.get_cost_time(), strPanelID);
		bpInspectEnd[nImageNum][nCameraNum] = true;
		theApp.m_AlgorithmTask->VS_Send_State_To_UI(strPanelID, REPORT_STATE_JUDGE, STATE_END, pStImageInfo->nImageNo);
		//所有相机等待所有线程检查结束
		for (int i = 0; i < MAX_GRAB_STEP_COUNT; i++)
		{
			for (int k = 0; k < MAX_CAMERA_COUNT; k++)
			{
				while (!bpInspectEnd[i][k])
				{
					Sleep(10);
				}
			}
		}
		
		//在单线程中填写判定和结果值
		if (nImageNum == 0 && nCameraNum == 0)
		{
			if (bUseInspect && !bIsHeavyAlarm)
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("{%s}(%s)PNL 完成检测."), strModelID, strPanelID);
			else
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("{%s}(%s)PNL 跳过检测."), strModelID, strPanelID);

			if (!bIsHeavyAlarm)		//发生警报时什么都不做
			{
				
				// test
				CCPUTimer tact;
				tact.Start();
				//18.09.18 Judgement Virture函数修改(App Stage
				//theApp.m_AlgorithmTask->VS_Send_State_To_UI(strPanelID, REPORT_STATE_JUDGE, STATE_START);
				Judgement(WrtResultInfo, ResultPanelData, MatDrawBuffer, tCHoleAlignData, strModelID, strLotID, strPanelID, strDrive, nRatio, eInspMode, MatOriginImg, bUseInspect, nStageNo);
				theApp.m_AlgorithmTask->VS_Send_State_To_UI(strPanelID, REPORT_STATE_GRADE, STATE_END);
				theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tW(1) %.2f"), tact.Stop(false) / 1000.);

				//AI复判
				//Judgement_AI(WrtResultInfo, ResultPanelData, MatDrawBuffer, tCHoleAlignData, strModelID, strLotID, strPanelID, strDrive, nRatio, eInspMode, MatOriginImg, bUseInspect, nStageNo);
				//theApp.m_AlgorithmTask->VS_Send_State_To_UI(strPanelID, REPORT_STATE_JUDGE, STATE_END);

				WriteResultData(strPanelID, strDrive, MatResultImg, MatDrawBuffer, WrtResultInfo, ResultPanelData, eInspMode);
				theApp.m_AlgorithmTask->VS_Send_State_To_UI(strPanelID, REPORT_STATE_WTRESULT, STATE_END);
				theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tW(2) %.2f"), tact.Stop(false) / 1000.);

				//如果是E级判定,结束得太快,等待2秒190314YWS
				if (nSkipFlag == true) Sleep(2000);

				theApp.m_AlgorithmTask->VS_Send_ClassifyEnd_To_Seq(strPanelID, strDrive, (UINT)ResultPanelData.m_ListDefectInfo.GetCount(),
					ResultPanelData.m_ResultPanel.Judge, ResultPanelData.m_ResultPanel.judge_code_1, ResultPanelData.m_ResultPanel.judge_code_2);
			}
		}
	}

	return TRUE;
}

void InspMainSequence::SaveCropImage(CString strPanelID, CString strDrive, cv::Mat(*MatResult)[MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT], ResultPanelData& resultPanelData, ENUM_INSPECT_MODE eInspMode)
{
	//GUI所需的画面大小-不要更改
	//-2018.04.24 Width为固定大小,Height为相机比例
	//-在GUI中,修改为不改变雕刻图像比例,只放大/缩小显示	
	const int	nWantSizeX = 320;
	//const int	nWantSizeY = 240;	// 根据原始图像的比例更改为可变大小
	const int	nOffSet = 2;

	//固定分配使用-更改为可变大小
//cv::Mat				MatDefectImage = cv::Mat::zeros(nWantSizeY, nWantSizeX, CV_8UC3);

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int nCamNum = resultPanelData.m_ListDefectInfo[i].Camera_No;

		int	nTargetWidth = MatDrawBuffer[nImgNum][nCamNum].cols;
		int	nTargetHeight = MatDrawBuffer[nImgNum][nCamNum].rows;

		int nWantSizeY = (nTargetHeight * nWantSizeX) / nTargetWidth;	// 根据原始图像比例调整雕刻图像的高度

		cv::Rect rectCropDefect(0, 0, 0, 0);

		if (resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_X == 0 && resultPanelData.m_ListDefectInfo[i].Pixel_Crop_End_X == 0 &&
			resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_Y == 0 && resultPanelData.m_ListDefectInfo[i].Pixel_Crop_End_Y == 0)
		{
			rectCropDefect = cv::Rect(0, 0, nTargetWidth, nTargetHeight);
		}
		else
		{
			int nCenterX = (int)((resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_Crop_End_X) / 2);
			int nCenterY = (int)((resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_Crop_End_Y) / 2);

			//实际Defect区域
			int nDefStartX, nDefStartY, nDefWidth, nDefHeight;

			nDefStartX = resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_X - nOffSet;
			nDefStartY = resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_Y - nOffSet;
			nDefWidth = resultPanelData.m_ListDefectInfo[i].Pixel_Crop_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_X + (nOffSet * 2) + 1;
			nDefHeight = resultPanelData.m_ListDefectInfo[i].Pixel_Crop_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Crop_Start_Y + (nOffSet * 2) + 1;

			//完美雕刻图像宽度/高度
			int nCropWidth, nCropHeight;

			//如果坏区域小于320*240,则根据中心坐标将其截断为320*240
			if (nDefWidth <= nWantSizeX && nDefHeight <= nWantSizeY)
			{
				nCropWidth = nWantSizeX;
				nCropHeight = nWantSizeY;
			}
			else
			{
				if (nDefWidth > nDefHeight)
				{
					nCropWidth = nDefWidth;
					nCropHeight = (int)(nTargetHeight * (1.0 * nDefWidth / nTargetWidth));
				}
				else
				{
					nCropWidth = (int)(nTargetWidth * (1.0 * nDefHeight / nTargetHeight));
					nCropHeight = nDefHeight;
				}
			}

			//Crop Image剪裁区域
			rectCropDefect = cv::Rect(nCenterX - (nCropWidth / 2), nCenterY - (nCropHeight / 2),
				nCropWidth, nCropHeight);
		}

		//图像坐标异常处理
		if (rectCropDefect.width > nTargetWidth)	rectCropDefect.width = nTargetWidth;
		if (rectCropDefect.height > nTargetHeight)	rectCropDefect.height = nTargetHeight;
		if (rectCropDefect.x + rectCropDefect.width >= nTargetWidth)	rectCropDefect.x = nTargetWidth - 1 - rectCropDefect.width;
		if (rectCropDefect.y + rectCropDefect.height >= nTargetHeight)	rectCropDefect.y = nTargetHeight - 1 - rectCropDefect.height;
		if (rectCropDefect.x < 0)		rectCropDefect.x = 0;
		if (rectCropDefect.y < 0)		rectCropDefect.y = 0;

		//在结果图像不良坐标下320*240 Buffer Copy
		//只创建一个标头,只使用数据
		cv::Mat MatDefectImage = MatDrawBuffer[nImgNum][nCamNum](rectCropDefect);

		//如果图像大小不正确,则将其放入固定内存
//if (matTemBuf.cols != nWantSizeX || matTemBuf.rows != nWantSizeY)
		cv::resize(MatDefectImage, MatDefectImage, cv::Size(nWantSizeX, nWantSizeY));

		//图像存储路径
		CString strImageName = _T("");
		strImageName.Format(_T("%d_%04d.jpg"), resultPanelData.m_ListDefectInfo[i].Defect_Type, i + 1);
		COPY_CSTR2TCH(resultPanelData.m_ListDefectInfo[i].Defect_Img_Name, strImageName, sizeof(resultPanelData.m_ListDefectInfo[i].Defect_Img_Name));

		//保存图像

		ImageSave(MatDefectImage, _T("%s\\%s\\%s"), RESULT_PATH, strPanelID, resultPanelData.m_ListDefectInfo[i].Defect_Img_Name);
	}

	//MatDefectImage.release();
}

///AD检查->Align Image->AD GV检查
BOOL InspMainSequence::CheckImageIsNormal(CString strPanelID, CString strDrive, cv::Mat& MatOrgImage, cv::Mat& MatDrawImage, int nRatio, int nImageNum, int nCameraNum, int nStageNo,
	tAlignInfo stCamAlignInfo[MAX_CAMERA_COUNT], ResultBlob_Total* pResultBlobTotal, double dCamResolution, double dPannelSizeX, double dPannelSizeY,
	bool bAlignEnd[MAX_CAMERA_COUNT], bool& bChkDustEnd, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool& bIsHeavyAlarm,
	ENUM_INSPECT_MODE eInspMode)
{
	//AD判定
	long nErrorCode = E_ERROR_CODE_TRUE;
	//压接异显 hjf
	long nErrorCode1 = E_ERROR_CODE_TRUE;
	double* dAlignPara = theApp.GetAlignParameter(nCameraNum);

	int nAlignImageClassify = 0;
	int nEQType = theApp.m_Config.GetEqpType();
	switch (nEQType)
	{
	case EQP_AVI:
		nAlignImageClassify = (int)dAlignPara[E_PARA_AD_IMAGE_NUM];		// Align Parameter 0号是要Align的Image ID(Classify)
		break;
	case EQP_SVI:
		nAlignImageClassify = (int)dAlignPara[E_PARA_AD_IMAGE_NUM];		// Align Parameter 0号是要Align的Image ID(Classify)
		break;
	case EQP_APP:
		nAlignImageClassify = (int)dAlignPara[E_PARA_APP_AD_IMAGE_NUM];	// Align Parameter 0号是要Align的Image ID(Classify)
		break;
	default:
		return E_ERROR_CODE_FALSE;
	}

	int nCurlImageClassify = (int)dAlignPara[E_PARA_APP_CURL_IMAGE_NUM];

	//if(MatOrgImage.empty())
	//	return E_ERROR_CODE_EMPTY_BUFFER;

	//Scalar m,s;
	//cv::meanStdDev(MatOrgImage, m, s);

	//if( m[0]==0 )
	//	return E_ERROR_CODE_EMPTY_BUFFER;

	int nPat = theApp.GetImageClassify(nImageNum);
	if (nPat == nAlignImageClassify)
	{
		// Alg DLL Start Log
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s] AD检查开始."),
			strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));

		//0:匹配率(Rate)
		//1:平均亮度(Mean GV)
		//2:标准偏差(Std)
		//3:结果判定值(DISPLAY_ABNORMAL,DISPLAY_DARK,DISPLAY_BRIGHT)
		double dResult[4] = { 0, };

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				///AD检查
				//AD算法后,需要区分判定...
		nErrorCode = CheckAD(strPanelID, strDrive, MatOrgImage, nImageNum, nCameraNum, dResult, nRatio);
		/////////////////////////////
		if (nErrorCode != E_ERROR_CODE_TRUE)	//AD不良时停止检查
		{
			// Alg DLL Stop Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE,
				_T("(%s)[%02d][%s] AD检查失败. 载台号: %d  错误码 : %d"),
				strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, nErrorCode);

			//点灯异常/亮度异常/等等...
			if (nErrorCode == E_ERROR_CODE_ALIGN_DISPLAY)
			{
				//报告为AD判定结果
				JudgeADDefect(nImageNum, nCameraNum, nStageNo, MatOrgImage.cols, MatOrgImage.rows, pResultBlobTotal, (int)dResult[3], eInspMode);
			}
			//没有缓冲区/参数异常/等等...
			else
			{
				//需要单独报告为不良的规格—汇总报告为当前AD不良
				JudgeADDefect(nImageNum, nCameraNum, nStageNo, MatOrgImage.cols, MatOrgImage.rows, pResultBlobTotal, E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL, eInspMode);
			}
			CString strOrgFileName = _T("");
			strOrgFileName.Format(_T("%s_CAM%02d_AD_Defect"), theApp.GetGrabStepName(nImageNum), nCameraNum);
			strOrgFileName = strOrgFileName + _T(".bmp");

			CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
			ImageSave(MatOrgImage, _T("%s\\%s\\%02d_%s"),
				ORIGIN_PATH, strPanelID, nImageNum, strOrgFileName);

			// Alg DLL End Log
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE,
			//	_T("AD Algorithm End Fail. PanelID: %s, Stage: %02d, CAM: %02d, Img: %s.\n\t\t\t\t( Rate : %.2f %% / Avg : %.2f GV / Std : %.2f )"),
			//	strPanelID, nStageNo, nCameraNum, theApp.GetGrabStepName(nImageNum), dResult[0], dResult[1], dResult[2]);

			stCamAlignInfo[nCameraNum].bAlignSuccess = false;
		}

		nErrorCode1 = CheckPGConnect(strPanelID, strDrive, MatOrgImage, nImageNum, nCameraNum, dResult, &cv::Point());
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE,
				_T("(%s)[%02d][%s]PG 压接异显- 跳过算法 !!!. 载台号: %d 错误码 : %d"),
				strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, nErrorCode);
			int nAlarmId = (int)eALARMID_PG_DISPLAY + nStageNo;//警报码：警报Id（3100） + stage编号
			theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eInspMode, nAlarmId, eALARMTYPE_HEAVY, bIsHeavyAlarm);
			JudgeADDefect(nImageNum, nCameraNum, nStageNo, MatOrgImage.cols, MatOrgImage.rows, pResultBlobTotal, E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL, eInspMode);
			stCamAlignInfo[nCameraNum].bAlignSuccess = false;
		}

		if (nErrorCode == E_ERROR_CODE_TRUE && nErrorCode1 == E_ERROR_CODE_TRUE)	//仅在AD不坏的情况下运行Align
		{
			// Alg DLL End Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE,
				_T("(%s)[%02d][%s]<AD> 算法检测结束. 载台号: %d ( Rate : %.2f %% / Avg : %.2f GV / Std : %.2f )"),
				strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, dResult[0], dResult[1], dResult[2]);

			// Alg DLL Start Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<Align> 算法检查开始. 载台号: "),
				strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo);

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/// Align Image

						//0:获取计算角度
						//试图获取pStCamAlignInfo->dAlignTheta
						//Align失败时没有放入数据,所以只输出为0值。
						//1:y轴Pixel差异
			double dResult[4] = { 0, };		// SVI错误修复

			//提取Image Align和Auto ROI
			//AVI:原封不动地使用原始画面/SVI:原始画面Cell区域Warping/APP:原始画面Rotate	
			nErrorCode = AcquireAutoRoiAndAlignImage(strPanelID, strDrive, MatOrgImage, nRatio, nImageNum, nCameraNum, stCamAlignInfo[nCameraNum], dResult, dCamResolution,
				dPannelSizeX, dPannelSizeY);

			//用于控制发送的相机中心位置-单元格中心位置-单位um
			int nDiffCenterX = (int)((stCamAlignInfo[nCameraNum].ptAlignCenter.x - stCamAlignInfo[nCameraNum].ptCellCenter.x) * dCamResolution);
			int nDiffCenterY = (int)((stCamAlignInfo[nCameraNum].ptAlignCenter.y - stCamAlignInfo[nCameraNum].ptCellCenter.y) * dCamResolution);

			//17.08.10-如果出现异常,返回值为E_ERROR_CODE_ALIGN_ANGLE_RANGE		//该值为Align参数:AngleError可调
			//超过一定数值时报告警报
			//如果超出角度范围,必须发出警报以确认...
			//手动重新训练...or确认是否继续扭曲
			if (nErrorCode == E_ERROR_CODE_ALIGN_ANGLE_RANGE_ERROR || nErrorCode == E_ERROR_CODE_ALIGN_IMAGE_OVER ||
				nErrorCode == E_ERROR_CODE_ALIGN_LENGTH_RANGE_ERROR)		//如果偏离FOV,添加//因Curl而导致的Align Fail-2018.08.02CJS-必要时进行单独管理
			{
				// Alg DLL End Log
				// Alg DLL End Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<Align> 算法检查结束. 角度异常 : %.4f 载台号: %d"),
					strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), dResult[0], nStageNo);

				//将结果值发送到控制PC
				theApp.m_AlgorithmTask->VS_Send_Align_Result_To_MainPC(eInspMode, (int)(dResult[0] * 1000), nDiffCenterX, nDiffCenterY, nStageNo, theApp.m_Config.GetPCNum());

				//中更改为严重警报
//SendAlarmToMain(eALARMID_ALIGN_WARNING, eALARMTYPE_LIGHT, bIsHeavyAlarm);
				theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eInspMode, eALARMID_ALIGN_ANGLE_ERROR, eALARMTYPE_HEAVY, bIsHeavyAlarm);

				CString strOrgFileName = _T("");
				strOrgFileName.Format(_T("%s_CAM%02d_Abnormal_Angle"), theApp.GetGrabStepName(nImageNum), nCameraNum);
				strOrgFileName = strOrgFileName + _T(".bmp");

				CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
				ImageSave(MatOrgImage, _T("%s\\%s\\%02d_%s"),
					ORIGIN_PATH, strPanelID, nImageNum, strOrgFileName);

				stCamAlignInfo[nCameraNum].bAlignSuccess = false;
			}
			else if (nErrorCode == E_ERROR_CODE_ALIGN_ANGLE_RANGE_WARNING)	//如果Align可以,但检查可能会受到影响
			{
				// Alg DLL End Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<Align> 算法检查结束. 角度警告 : %.4f 载台号: %d"),
					strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), dResult[0], nStageNo);

				//将结果值发送到控制PC
				theApp.m_AlgorithmTask->VS_Send_Align_Result_To_MainPC(eInspMode, (int)(dResult[0] * 1000), nDiffCenterX, nDiffCenterY, nStageNo, theApp.m_Config.GetPCNum());

				//发生警报-删除警报/只填写日志
	//theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eInspMode, eALARMID_ALIGN_ANGLE_WARNING, eALARMTYPE_LIGHT, bIsHeavyAlarm);				

				CString strOrgFileName = _T("");
				strOrgFileName.Format(_T("%s_CAM%02d_Warning_Angle"), theApp.GetGrabStepName(nImageNum), nCameraNum);
				strOrgFileName = strOrgFileName + _T(".bmp");

				CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
				ImageSave(MatOrgImage, _T("%s\\%s\\%02d_%s"),
					ORIGIN_PATH, strPanelID, nImageNum, strOrgFileName);

				stCamAlignInfo[nCameraNum].bAlignSuccess = true;	// 首先检查进行				
			}
			//如果虚线本身失败
			else if (nErrorCode != E_ERROR_CODE_TRUE)
			{
				// Error Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE,
					_T("(%s)[%02d][%s]<Align> 失败 算法检查停止. 载台号 :  错误码 : %d"),
					strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, nErrorCode);

				//更改为严重警报
				theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eInspMode, eALARMID_ALIGN_FAIL, eALARMTYPE_HEAVY, bIsHeavyAlarm);

				CString strOrgFileName = _T("");
				strOrgFileName.Format(_T("%s_CAM%02d_Align_Fail"), theApp.GetGrabStepName(nImageNum), nCameraNum);
				strOrgFileName = strOrgFileName + _T(".bmp");

				CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
				ImageSave(MatOrgImage, _T("%s\\%s\\%02d_%s"),
					ORIGIN_PATH, strPanelID, nImageNum, strOrgFileName);

				//Align Error临时Display Abnormal报告
				//需要单独报告为不良的规格—汇总报告为当前AD不良
	//JudgeADDefect(nImageNum, nCameraNum, nStageNo, MatOrgImage.cols, MatOrgImage.rows, pResultBlobTotal, E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL, eInspMode);

				stCamAlignInfo[nCameraNum].bAlignSuccess = false;

				//删除重复日志。Theta值没有意义
				// 				// Alg DLL End Log
				// 				theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("Align Algorithm End Fail. PanelID: %s, Stage: %02d, CAM: %02d, Img: %s.\n\t\t\t\t( Theta : %.4f' / Diff Pixel : %.0f )"),
				// 					strPanelID, nStageNo, nCameraNum, theApp.GetGrabStepName(nImageNum), dResult[0], dResult[0]);
			}
			//正常情况下
			else
			{
				// Alg DLL End Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<Align> 算法结束. 载台号:%d 角度 : %.4f' / 差异像素 : %.0f )"),
					strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, dResult[0], dResult[1]);

				stCamAlignInfo[nCameraNum].bAlignSuccess = true;

				//将结果值发送到控制PC
				theApp.m_AlgorithmTask->VS_Send_Align_Result_To_MainPC(eInspMode, (int)(dResult[0] * 1000), nDiffCenterX, nDiffCenterY, nStageNo, theApp.m_Config.GetPCNum());
			}
		}

		bAlignEnd[nCameraNum] = true;
	}

	//不Align的图像等待虚线结果
	//直到所有相机连线完成
	for (int i = 0; i < MAX_CAMERA_COUNT; i++)
	{
		while (!bAlignEnd[i])
		{
			Sleep(10);
		}
	}

	bool bNeedRetry = false;

	//设置每个线程的分行信息
	//(根据当前Step的Pixel Shift与否,根据默认分辨率计算并设置Align Info的比例)
	m_stThrdAlignInfo.SetAdjustAlignInfoRatio(&stCamAlignInfo[nCameraNum], nRatio, stCamAlignInfo[0].bAlignSuccess && stCamAlignInfo[nCameraNum].bAlignSuccess);	// 0号相机虚线失败面全部设置为失败

	//在AD检查和Align成功时运行GV检查
	//即使是Align Warning,也会进行检查
	if ((nErrorCode == E_ERROR_CODE_TRUE || nErrorCode == E_ERROR_CODE_ALIGN_ANGLE_RANGE_WARNING) && m_stThrdAlignInfo.bAlignSuccess == true && !bIsHeavyAlarm)
	{
		//0:平均亮度(Mean GV)
		//1:结果判定值(DISPLAY_ABNORMAL,DISPLAY_DARK,DISPLAY_BRIGHT)
		//重新检查点亮区域AD GV值
		double dMeanResult[4] = { 0 , };		// 2->4SVI错误修复

		///AD GV检查		
		nErrorCode = CheckADGV(strPanelID, strDrive, MatOrgImage, nStageNo, nImageNum, nCameraNum, nRatio, m_stThrdAlignInfo.ptCorner, pResultBlobTotal, dMeanResult,
			bChkDustEnd, bNeedRetry, bIsNormalDust, bUseDustRetry, nDustRetryCnt, bIsHeavyAlarm, eInspMode);

		//for SVI-当前必须基于0号相机图像进行警告。
		//添加17.06.08 APP:Rotate
		if (dAlignPara[E_PARA_AVI_Rotate_Image] > 0) {
			AdjustOriginImage(MatOrgImage, MatDrawImage, &stCamAlignInfo[0]);
		}
		//ImageSave(MatOrgImage, _T("E:\\IMTC\\Test\\Origin%d_CAM%d.jpg"), nImageNum, nCameraNum);
	}
	else
	{
		///for AVI-不需要Dust Retry-Dust扫描失败-SVI/APP Align等需要时进行扩展
		// Seq. 结束Dust并将结果发送到Task(AVI)
		bNeedRetry = false;
		bIsNormalDust = false;

		//填充4个临时转角点(用于报告)
		stCamAlignInfo[nCameraNum].ptCorner[E_CORNER_LEFT_TOP] = cv::Point(0, 0);
		stCamAlignInfo[nCameraNum].ptCorner[E_CORNER_RIGHT_TOP] = cv::Point(MatOrgImage.cols - 1, 0);
		stCamAlignInfo[nCameraNum].ptCorner[E_CORNER_RIGHT_BOTTOM] = cv::Point(MatOrgImage.cols - 1, MatOrgImage.rows - 1);
		stCamAlignInfo[nCameraNum].ptCorner[E_CORNER_LEFT_BOTTOM] = cv::Point(0, MatOrgImage.rows - 1);
	}

	// Seq. 结束Dust并将结果发送到Task(目前仅使用AVI)
	if (theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_DUST)
		theApp.m_AlgorithmTask->VS_Send_Dust_Result_To_Seq(eInspMode, bNeedRetry, bIsNormalDust);

	return nErrorCode;
}