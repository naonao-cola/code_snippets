#include "stdafx.h"
#include "WorkManager.h" 
#include "..\..\VisualStation\CommonHeader\Class\LogWriter.h"
#include "VSAlgorithmTask.h"
#include <algorithm>
#include "DefineInterface.h"
#include "InspThrd.h"
//#include "AviDefineInspect.h"
#include "DllInterface.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"

//Image Save相关标题
//#include "ImageSave.h"

void ImageSave(cv::Mat& MatSrcBuffer, TCHAR* strPath, ...);
void ImageAsyncSaveJPG(cv::Mat& MatSrcBuffer, const char* strPath);

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

extern CLogWriter* GlassDataWriter;

#define	SEQUENCE_TABLE(FUNCTION_NO,SEQUENCE_NO,FUNCTION_NAME,ALWAYS_RUNMODE,SEQ_REST_POSSIBLE,CS_LOCK_NAME)					\
	if ( pCmdMsg->uFunID_Dest == FUNCTION_NO && pCmdMsg->uSeqID_Dest == SEQUENCE_NO )										\
	{																														\
		if ((VOID*)CS_LOCK_NAME != NULL)																					\
		{																													\
			EnterCriticalSection(CS_LOCK_NAME);																				\
		}																													\
			if (SEQ_REST_POSSIBLE)																							\
				m_SeqenceCount++;																							\
			isFunctionRuned = true;																							\
			nRet = FUNCTION_NAME((BYTE*)pCmdMsg->cMsgBuf, pCmdMsg->uMsgSize, ALWAYS_RUNMODE, false, SEQ_REST_POSSIBLE);		\
			if (SEQ_REST_POSSIBLE)																							\
				m_SeqenceCount--;																							\
		if ((VOID*)CS_LOCK_NAME != NULL)																					\
		{																													\
			LeaveCriticalSection(CS_LOCK_NAME);																				\
		}																													\
	}

WorkManager::WorkManager()
{
	int nCameraNo = 0;
	TCHAR strVal[2];
	CString strSection, strKey;

	// 	for (int nGrabberCnt=0; nGrabberCnt < MAX_FRAME_GRABBER_COUNT; nGrabberCnt++)	//Frame Grabber Max : 4
	// 	{
	// 		strKey.Format(_T("Frame Grabber_%d"), nGrabberCnt);
	// 		GetPrivateProfileString(_T("Grabber_Board"), strKey ,_T("F"), strVal, 2, DEVICE_FILE_PATH);
	// 		if (!_tcscmp(strVal, _T("T")))
	// 		{
	for (int nCameraCnt = 0; nCameraCnt < MAX_CAMERA_COUNT; nCameraCnt++)//每个Frame Grabber的Camera Max:4
	{
		m_pSharedMemory[nCameraCnt] = NULL;
		strSection.Format(_T("Frame Grabber_%d"), 0);
		strKey.Format(_T("Insp Camera_%d"), nCameraCnt);
		GetPrivateProfileString(strSection, strKey, _T("F"), strVal, 2, DEVICE_FILE_PATH);
		if (!_tcscmp(strVal, _T("T")))
		{
			m_pSharedMemory[nCameraNo] = new CSharedMemWrapper();				// 内存分配
			CString strDrv_CamNo = _T("");
			strDrv_CamNo.Format(_T("%s_%d"), theApp.m_Config.GETDRV(), nCameraNo + 1);
			m_pSharedMemory[nCameraNo]->OpenSharedMem(strDrv_CamNo);			// CAMERA_NUM号摄像头连接
			nCameraNo++;
		}
	}
	// 		}
	// 	}
	m_hSequenceResetEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	m_bSeqResetFlag = 0;
	m_SeqenceCount = 0;

	InitializeCriticalSectionAndSpinCount(&m_csSequenceLock_1, 4000);
	InitializeCriticalSectionAndSpinCount(&m_csSequenceLock_2, 4000);
	InitializeCriticalSectionAndSpinCount(&m_csSequenceLock_3, 4000);
	InitializeCriticalSectionAndSpinCount(&m_csSequenceLock_4, 4000);
	InitializeCriticalSectionAndSpinCount(&m_csSequenceLock_5, 4000);
}

WorkManager::~WorkManager()
{
	for (int nCameraCnt = 0; nCameraCnt < MAX_CAMERA_COUNT; nCameraCnt++)//每个Frame Grabber的Camera Max:4
		SAFE_DELETE(m_pSharedMemory[nCameraCnt]);
	DeleteCriticalSection(&m_csSequenceLock_1);
	DeleteCriticalSection(&m_csSequenceLock_2);
	DeleteCriticalSection(&m_csSequenceLock_3);
	DeleteCriticalSection(&m_csSequenceLock_4);
	DeleteCriticalSection(&m_csSequenceLock_5);
}

int WorkManager::Start()
{
	HANDLE handle;

	//Message接收方单一Thread
	for (int i = 0; i < 30; i++)
	{
		handle = m_fnStartThread();
		if (handle == NULL || handle == (HANDLE)-1)
			return APP_NG;
	}

	return APP_OK;
}

void WorkManager::m_fnRunThread()
{
	int nRet;
	BYTE* pMsg = NULL;
	CMDMSG* pCmdMsg = NULL;

	//Start Status Refresh Thread. 第一次运行的一个线程负责处理。

	while (GetThreadRunFlag())
	{
		EXCEPTION_TRY

			pMsg = m_fnPeekMessage();
		pCmdMsg = (CMDMSG*)pMsg;

		nRet = AnalyzeMsg(pCmdMsg);
		if (nRet != APP_OK)
			throw nRet;

		EndWorkProcess(pMsg);

		pCmdMsg = NULL;
		pMsg = NULL;

		EXCEPTION_CATCH

			if (nRet != APP_OK)
			{
				if (pMsg != NULL)
				{
					EndWorkProcess(pMsg);
					pCmdMsg = NULL;
					pMsg = NULL;
				}
				theApp.WriteLog(eLOGCOMM, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("ERROR WorkManager::AnalyzeMsg. Error Code = %d \n"), nRet);
			}
	}
}

int WorkManager::AnalyzeMsg(CMDMSG* pCmdMsg)
{
	int			nRet = APP_OK;
	bool		isFunctionRuned = false;

	EXCEPTION_TRY

		//	SEQUENCE_TABLE(	FUNNO,	SEQNO,	FUNCTION_NAME,						是否可重复执行,	序列是否可重置	)
			// for test	
		SEQUENCE_TABLE(10, 99, Seq_TEST, false, true, &m_csSequenceLock_1)
		// Alive
		SEQUENCE_TABLE(10, 1, VS_TaskAlive, false, true, &m_csSequenceLock_1)
		// Inspect
		SEQUENCE_TABLE(21, 1, Seq_StartInspection, false, true, &m_csSequenceLock_2)	//开始检查
		SEQUENCE_TABLE(50, 10, Seq_ManualAlign, false, true, &m_csSequenceLock_2)	//运行Manual Align
		SEQUENCE_TABLE(50, 20, Seq_SetParam, false, true, &m_csSequenceLock_2)	//参数设置
		SEQUENCE_TABLE(50, 30, Seq_FocusValue, false, true, &m_csSequenceLock_2)	// Get Focus Value
		SEQUENCE_TABLE(50, 40, Seq_WriteCCDIndex, false, true, &m_csSequenceLock_2)	// Write CCD Defect Index
		SEQUENCE_TABLE(21, 10, Seq_GetAlignPatternNum, false, true, &m_csSequenceLock_3)	//Align PatternNum返回
		SEQUENCE_TABLE(21, 50, Seq_StartAlign, false, true, &m_csSequenceLock_3)	// Align Camera

		if (m_SeqenceCount <= 0)
		{
			m_bSeqResetFlag = 0;
			m_SeqenceCount = 0;
			ResetEvent(m_hSequenceResetEvent);
		}

	if (isFunctionRuned == false)
	{
		theApp.WriteLog(eLOGCOMM, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Function Table is nothing. FuncNo : %d, SeqNo : %d ,From %d Task "), pCmdMsg->uFunID_Dest, pCmdMsg->uSeqID_Dest, pCmdMsg->uTask_Src);
		throw SEQUENCE_TASK_SEQUENCE_IS_NOTHING;
	}

	EXCEPTION_CATCH

		if (pCmdMsg->uMsgType == CMD_TYPE_RES && pCmdMsg->uMsgOrigin == CMD_TYPE_CMD)
		{
			nRet = ResponseSend((USHORT)nRet, pCmdMsg);
			if (nRet != APP_OK)
			{
				theApp.WriteLog(eLOGCOMM, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Response Send  Fail. FuncNo: %d, SeqNo : %d "), pCmdMsg->uFunID_Dest, pCmdMsg->uSeqID_Dest);
				return nRet;
			}
		}

	return nRet;
}

int	WorkManager::Seq_TEST(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode /*= false*/, bool bBusyCheck /*= false*/, bool bSeqResetPossible)
{
	int nRet = APP_OK;
	bool isRunSequence = true;
	int nStepNo = 0;
	static bool isSeqBusy = false;

	//USHORT	usSet;
	//INT		nVal;
	//byte*	tempParam = pParam;

	do
	{
		EXCEPTION_TRY

			if (nStepNo == 0 && isSeqBusy && bAlwaysRunMode == false)	//如果序列为Busy,则bAlwaysRunMode必须为false才能返回错误。
				return SEQUENCE_TASK_SEQUENCE_IS_BUSY;
			else if (nStepNo == 0 && bBusyCheck == true && isSeqBusy == false)
			{
				return SEQUENCE_TASK_SEQUENCE_IS_NOT_BUSY;
			}

		isSeqBusy = true;

		if (m_bSeqResetFlag && bSeqResetPossible)
			throw 9999;

		// Sequence In LOG
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq1099_TEST StepNo=%d, RetVal=%d \n"), nStepNo, nRet);

		nStepNo++;
		switch (nStepNo)
		{
		case 1:

			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("CASE 1"));
			break;

		case 2:

			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("CASE 2"));

			break;

		case 3:
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("CASE 3"));

			break;

		default:
			isRunSequence = false;
			break;
		}

		EXCEPTION_CATCH

			if (nRet != APP_OK)
			{
				// Error Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq1099_TEST Error Occured. StepNo=%d, RetVal=%d \n"), nStepNo, nRet);
				isRunSequence = false;
				int nRetExcept = APP_OK;

				// EQP BIT ALL OFF

			}

	} while (isRunSequence);

	// Sequence Out LOG
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq1099_TEST Sequence END. StepNo=%d, RetVal=%d \n"), nStepNo, nRet);

	isSeqBusy = false;

	return nRet;
}

int WorkManager::VS_TaskAlive(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode /*= false*/, bool bBusyCheck /*= false*/, bool bSeqResetPossible /*= true*/)
{
	int nRet = APP_OK;
	int nStepNo = 0;

	byte* tempParam = pParam;

	EXCEPTION_TRY
		// Do nothing
		EXCEPTION_CATCH

		if (nRet != APP_OK)
		{
			// Error Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq1001_Task_Alive Error Occured. StepNo=%d, RetVal=%d \n"), nStepNo, nRet);
			return nRet;
		}

	// Sequence Out LOG
	//theApp.WriteLog(eLOGCOMM, FALSE, TRUE, _T("Seq1001_Task_Alive Sequence END. StepNo=%d, RetVal=%d \n"), nStepNo, nRet);

	return nRet;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Send Message
int WorkManager::VS_Send_Log_To_UI(TCHAR* cBuffer, int len)
{
	byte* pParam = new byte[len + sizeof(int)];
	byte* pSendParam = pParam;
	int			nRet = APP_OK;

	*(int*)pSendParam = theApp.m_Config.GetPCNum();	pSendParam += sizeof(int);
	memcpy(pSendParam, cBuffer, len);					pSendParam += len;

	EXCEPTION_TRY
		nRet = CmdEditSend(SEND_UI_LOG, 0, (ULONG)(pSendParam - pParam), VS_UI_TASK, pParam, CMD_TYPE_NORES);
	EXCEPTION_CATCH

		if (nRet != APP_OK)
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("VS_Send_Log_To_UI Error Occured. RetVal=%d \n"), nRet);
		}

	SAFE_DELETE_ARR(pParam);

	return nRet;
}

int WorkManager::VS_Send_Notify_Init_To_UI()
{
	byte* pParam = new byte[sizeof(int)];
	byte* pSendParam = pParam;
	int		nRet = APP_OK;

	*(int*)pSendParam = theApp.m_Config.GetPCNum();	pSendParam += sizeof(int);

	EXCEPTION_TRY
		nRet = CmdEditSend(SEND_UI_NOTIFY_INIT, 0, (ULONG)(pSendParam - pParam), VS_UI_TASK, pParam, CMD_TYPE_NORES);
	EXCEPTION_CATCH

		if (nRet != APP_OK)
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("VS_Send_Notify_Init_To_UI Error Occured. RetVal=%d \n"), nRet);
		}

	SAFE_DELETE_ARR(pParam);

	return nRet;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Recieve Message
//2016.10.17
int WorkManager::Seq_StartInspection(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	// Receive //////////////////////////////////////////////////////////////////////////////////////////////////////
	byte* pReceiveParam = pParam;

	PARAM_INSPECT_START* pStParamInsp = new PARAM_INSPECT_START;
	memcpy(pStParamInsp, pReceiveParam, sizeof(PARAM_INSPECT_START));
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection Start. \
											  InspType:%d, StageNo:%d, PanelID:%s, VirtualID:%s, nImageNo:%d, LotID:%s"),
		pStParamInsp->nInspType, pStParamInsp->nStageNo, pStParamInsp->strPanelID, pStParamInsp->strVirtualID, pStParamInsp->nImageNum, pStParamInsp->strLotID);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	BOOL		bReturn = FALSE;

	int			nWidth = 0;
	int			nHeight = 0;
	UINT32		nBufSize = 0;
	int			nBandWidth = 0;
	int			nBitRate = 0;

	//20.07.03-更改线程参数结构的使用方式-S
	//(需要继续为Mat分配内存的不合理修改)
//static tInspectThreadParam *pStInspectThreadParam = NULL;
	static tInspectThreadParam* pStInspectThreadParam[MAX_THREAD_PARAM_COUNT];
	static int nThreadParamIndex = 0;	//20.07.03
	static bool bFirstFlag = true;
	if (bFirstFlag)	//只生成第一次。
	{
		bFirstFlag = false;
		for (int i = 0; i < MAX_THREAD_PARAM_COUNT; i++)
			pStInspectThreadParam[i] = new tInspectThreadParam;
	}
	//20.07.03-更改线程参数结构的使用方式-E

	cv::Mat			MatOrgImage;
	CString			strDrive = _T(""); // 2018.01.18 sggim当前面板中使用的驱动器,已溢出到Thread Parameter。

	// test
	CCPUTimer tact;
	tact.Start();

	for (int nCamIndex = 0; nCamIndex < MAX_CAMERA_COUNT; nCamIndex++)
	{

		//Image Classify Num为负值时不进行检查
		if (pStParamInsp->bUseCamera[nCamIndex] && theApp.m_pGrab_Step[pStParamInsp->nImageNum].eImgClassify >= 0)
		{

			// Send //////////////////////////////////////////////////////////////////////////////////////////////////////
			STRU_IMAGE_INFO* pStImageInfo = new STRU_IMAGE_INFO;

			pStImageInfo->strPanelID = pStParamInsp->strPanelID;
			if (pStImageInfo->strPanelID == _T(""))
				pStImageInfo->strPanelID = _T("Unknown");
			pStImageInfo->strLotID = pStParamInsp->strLotID;
			pStImageInfo->nCameraNo = nCamIndex;
			pStImageInfo->nImageNo = pStParamInsp->nImageNum;
			pStImageInfo->nStageNo = pStParamInsp->nStageNo;
			pStImageInfo->nInspType = pStParamInsp->nInspType;
			pStImageInfo->nRatio = pStParamInsp->nSeqMode[nCamIndex] + 1;
			for (int i = 0; i < 3; i++)
				pStImageInfo->dPatternCIE[i] = pStParamInsp->dPatternCIE[i];		// MTP校正后[0]:X,[1]:Y,[2]:利用L检查
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////// 	

			nWidth = m_pSharedMemory[nCamIndex]->GetImgWidth() * pStImageInfo->nRatio;
			nHeight = m_pSharedMemory[nCamIndex]->GetImgHeight() * pStImageInfo->nRatio;
			nBandWidth = m_pSharedMemory[nCamIndex]->GetImgBandWidth();
			nBitRate = m_pSharedMemory[nCamIndex]->GetImgBitrate();

			CString strManualImagePath = _T("");
			CString strOrgFileName = _T("");	// 基于原始图像文件的名称(Grab Number除外)
			//文件名为Alg。更改为在Task中确定-多摄像头响应
			strOrgFileName.Format(_T("%s_CAM%02d"), theApp.GetGrabStepName(pStImageInfo->nImageNo), nCamIndex);

			if (pStParamInsp->nInspType == (int)eManualInspect)	//检查菜单
			{
				//从路径读取图像			
#if 0
				strDrive = theApp.m_Config.GetSimulationDrive();
#else
				// strDrive = this->GetNewDriveForAlg();
				CString newDir;
				CString strMidName = _T(":\\ALG_");
				newDir = theApp.m_Config.GetSimulationDrive().Left(1) + strMidName;
				newDir += theApp.m_Config.GetEqpType() == EQP_AVI ? _T("AVI") : (theApp.m_Config.GetEqpType() == EQP_SVI ? _T("SVI") : _T("APP"));
				newDir.Format(_T("%s_%s%s"), newDir, theApp.m_Config.GetPCName(), _T("\\"));
				strDrive = newDir;
#endif // 0

				strManualImagePath = theApp.GetCurStepFileName(pStParamInsp->strImagePath[nCamIndex], strOrgFileName.GetBuffer(0));

				char* pStr = NULL;
				pStr = CSTR2PCH(strManualImagePath);
				MatOrgImage = cv::imread(pStr, IMREAD_UNCHANGED);
				SAFE_DELETE_ARR(pStr);

				theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tA(0) %.2f"), tact.Stop(false) / 1000.);
			}
			else	//自动检查
			{		//从共享内存读取
				if (pStImageInfo->nImageNo == 0)
				{
					//日志细分操作。180503 YSS
					theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Drive check Start."));
					//theApp.CheckDrive();
					theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Drive check End."));
				}

#if 0
				strDrive = theApp.m_Config.GetINIDrive();
#else
				strDrive = this->GetNewDriveForAlg();
				//strDrive = theApp.m_Config.GetResultDriveForAlg();
#endif
				if (nBitRate == 8)
				{
					if (nBandWidth == 1)
					{
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC1, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamInsp->nImageNum));
					}
					else
					{
						//根据SVI原始图像的比例删除边距
						double nCropRatio = *(double*)(theApp.GetAlignParameter(nCamIndex) + E_PARA_SVI_CROP_RATIO);
						//double nCropRatio =0.45;
						double StartX = nWidth * (1 - nCropRatio) * (0.5);
						double StartY = nHeight * (1 - nCropRatio) * (0.5);
						double EndX = nWidth * nCropRatio;
						double EndY = nHeight * nCropRatio;

						cv::Rect rect = cv::Rect((int)StartX, (int)StartY, (int)EndX, (int)EndY);
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC3, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamInsp->nImageNum));
						//MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC3, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamInsp->nImageNum));(rect)
					}
				}
				else
				{
					if (nBandWidth == 1)
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_16UC1, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamInsp->nImageNum));
					else
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_16UC3, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamInsp->nImageNum));
				}
			}

			if (MatOrgImage.empty())
			{
				CString strMsg = _T("");
				if (pStParamInsp->nInspType == (int)eManualInspect)
					strMsg.Format(_T("!!! No Image !!! \r\n\t (%s)"), strManualImagePath);
				else
					strMsg.Format(_T("!!! Grab Error !!! \r\n\t (Step : %s)"), theApp.GetGrabStepName(pStImageInfo->nImageNo));
				//AfxMessageBox(strMsg);
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, strMsg.GetBuffer(0));

				//需要临时留下日志,进行其他检查-判定为AD不良-Grab Error时报告的规格
	//pStInspectThreadParam->bInspectEnd[pStImageInfo->nImageNo][nCamIndex] = true;
				//分配空映像
				if (nBitRate == 8)
				{
					if (nBandWidth == 1)
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC1, cv::Scalar(0, 0, 0));
					else
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC3, cv::Scalar(0, 0, 0));
				}
				else
				{
					if (nBandWidth == 1)
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_16UC1, cv::Scalar(0, 0, 0));
					else
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_16UC3, cv::Scalar(0, 0, 0));
				}
			}

			//验证Pixel Shift启用/禁用的一致性
			//可能与之前拍摄的图像的当前设置值不同
			//如果设置值不同,将强制进行校验
			if (pStParamInsp->nInspType == (int)eManualInspect)	//检查菜单
			{

				//临时0处理-需要修改!!!
				int nRet = theApp.CheckImageRatio(pStImageInfo->nRatio, MatOrgImage.cols, MatOrgImage.rows, m_pSharedMemory[nCamIndex]->GetImgWidth(), m_pSharedMemory[nCamIndex]->GetImgHeight());

				//SVI Crop时,转换为nRatio3...需要确认...
//if (nRet != 0)
//{					
//	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Load Image Ratio Incorrect !! Change Ratio %d -> %d \r\n\t (%s) !!!"), pStImageInfo->nRatio, pStImageInfo->nRatio + nRet, strManualImagePath);
//	pStImageInfo->nRatio += nRet;
//}
			}

			//临时
			//保存原始图像-添加Grab Step(按当前模型Grab Step的顺序重新排序并保存)
			if (nBitRate == 8)
				strOrgFileName = strOrgFileName + _T(".bmp");
			else
				strOrgFileName = strOrgFileName + _T(".tiff");

			//将2021.01.07-B11Ph3 17英寸型号的专用图像水平旋转
			//装入线参数
			double dAlignPara[MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT] = { 0.0, };
			memcpy(dAlignPara, theApp.GetAlignParameter(0), sizeof(double) * MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT);
			int nRotate = (int)dAlignPara[E_PARA_IMAGE_ROTATE];

			if (nRotate < 0 || nRotate > 3 || nRotate == NULL)
				nRotate = 0;

			if (nRotate > 0 && !(pStParamInsp->nInspType == (int)eManualInspect))
			{
				if (nRotate == 1)
				{
					cv::rotate(MatOrgImage, MatOrgImage, cv::ROTATE_90_CLOCKWISE);
				}
				else if (nRotate == 2)
				{
					cv::rotate(MatOrgImage, MatOrgImage, cv::ROTATE_180);
				}
				else if (nRotate == 3)
				{
					cv::rotate(MatOrgImage, MatOrgImage, cv::ROTATE_90_COUNTERCLOCKWISE);
				}
			}

			// 			if (GetPrivateProfileInt(_T("Common"), _T("B11 AVI Ph3"), 0, INIT_FILE_PATH) && !(pStParamInsp->nInspType == (int)eManualInspect))
			// 			{
			// 				cv::rotate(MatOrgImage, MatOrgImage, cv::ROTATE_90_CLOCKWISE);
			// 			}

											//2020.07.22-用StartSaveImage替换,theApp。InspPanel.向上移动StartInspection
						/*ImageSave(MatOrgImage, _T("%s\\%s\\%02d_%s"),
							ORIGIN_PATH, pStImageInfo->strPanelID,
							pStParamInsp->nImageNum, strOrgFileName);*/

			theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tA(1) %.2f"), tact.Stop(false) / 1000.);

			//int nThreadParamIndex = 0;	//20.07.03

						//从初始拍摄图像创建Thread Parameter
			if (pStParamInsp->nImageNum == 0)
			{

				if (nCamIndex == 0)
				{
					//创建面板特定的Thread Parameter(仅在初始Pattern/Camera中)
						// pStInspectThreadParam = new tInspectThreadParam;	//20.07.03-删除现有的线程波动
					for (int i = 0; i < MAX_THREAD_PARAM_COUNT; i++)	//20.07.03-获取可用的线程波动索引
					{
						if (pStInspectThreadParam[i]->bParamUse == false)
						{
							nThreadParamIndex = i;
							break;
						}
					}

					pStInspectThreadParam[nThreadParamIndex]->bParamUse = true;	//20.07.03-正在处理线程波动
					pStInspectThreadParam[nThreadParamIndex]->ResultPanelData.m_ResultHeader.SetInspectStartTime();
					if (pStParamInsp->nInspType == (int)eAutoRun)		//仅在进行AutoRun时判断是否进行检查,在菜单操作时无条件进行检查
						pStInspectThreadParam[nThreadParamIndex]->bUseInspect = GetPrivateProfileInt(_T("INSPECT"), _T("Use_Inspect"), 1, INIT_FILE_PATH) ? true : false;
					else
						pStInspectThreadParam[nThreadParamIndex]->bUseInspect = true;
					pStInspectThreadParam[nThreadParamIndex]->eInspMode = (ENUM_INSPECT_MODE)pStParamInsp->nInspType;

					//预览要保存结果值的文件夹。
				//如果文件夹已存在,则清除所有内容
					theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tA(1.1) %.2f"), tact.Stop(false) / 1000.);
					CreateResultDirectory(pStParamInsp->strPanelID, strDrive);
					theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tA(1.2) %.2f"), tact.Stop(false) / 1000.);

					//将当前设置的WorkCoord相关信息传递给所有线程
					memcpy(&pStInspectThreadParam[nThreadParamIndex]->WrtResultInfo, theApp.GetCurWorkCoordInfo(), sizeof(CWriteResultInfo));

					for (int nStepCnt = 0; nStepCnt < theApp.GetGrabStepCount(); nStepCnt++)
					{
						//Image Classify Num为负值时不进行检查
						if (theApp.m_pGrab_Step[nStepCnt].bUse && theApp.m_pGrab_Step[nStepCnt].eImgClassify >= 0)
						{
							for (int nCamCnt = 0; nCamCnt < MAX_CAMERA_COUNT; nCamCnt++)
							{
								if (theApp.m_pGrab_Step[nStepCnt].stInfoCam[nCamCnt].bUse)
								{
									pStInspectThreadParam[nThreadParamIndex]->bInspectEnd[nStepCnt][nCamCnt] = false;
									pStInspectThreadParam[nThreadParamIndex]->bAlignEnd[nCamCnt] = false;
								}
							}
						}
					}

					pStInspectThreadParam[nThreadParamIndex]->tLabelMarkInfo.Reset();
					pStInspectThreadParam[nThreadParamIndex]->bChkDustEnd = false;		// 用于AVI的Dust Check结束标志
					pStInspectThreadParam[nThreadParamIndex]->bUseDustRetry = GetPrivateProfileInt(_T("INSPECT"), _T("Use_Dust_Retry"), 1, INIT_FILE_PATH) ? true : false;
					pStInspectThreadParam[nThreadParamIndex]->nDustRetryCnt = pStParamInsp->nRetryCnt;
					COPY_CSTR2TCH(pStInspectThreadParam[nThreadParamIndex]->strSaveDrive, strDrive, sizeof(pStInspectThreadParam[nThreadParamIndex]->strSaveDrive));
					theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tA(2) %.2f"), tact.Stop(false) / 1000.);
				}
			}

			if (pStInspectThreadParam != NULL)
			{

				//20.07.03-从clone更改为copyTo(copyTo仅在需要分配内存时分配内存)
				//pStInspectThreadParam->MatOrgImage[pStImageInfo->nImageNo][nCamIndex] = MatOrgImage.clone();	//每个内存分配模式的完整画面

				MatOrgImage.copyTo(pStInspectThreadParam[nThreadParamIndex]->MatOrgImage[pStImageInfo->nImageNo][nCamIndex]);
				// 2024.05.07 for develop
				//MatOrgImage.copyTo(pStInspectThreadParam[nThreadParamIndex]->MatOrgImage2[pStImageInfo->nImageNo][nCamIndex]);

								//2020.07.22-保存原始画面
				CString strImagePath;
				CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
				pStInspectThreadParam[nThreadParamIndex]->strImagePath[pStImageInfo->nImageNo][nCamIndex].Format(
					_T("%s\\%s\\%02d_%s"),
					ORIGIN_PATH,
					pStImageInfo->strPanelID,
					pStParamInsp->nImageNum,
					strOrgFileName);

				// 2024.05.07 for develop
				tImageInfo imageInfo(theApp.m_Config.GetPCNum(), pStImageInfo->nImageNo, pStImageInfo->strPanelID,
					pStInspectThreadParam[nThreadParamIndex]->strImagePath[pStImageInfo->nImageNo][nCamIndex]);
				pStInspectThreadParam[nThreadParamIndex]->stImageInfo[pStImageInfo->nImageNo][nCamIndex] = imageInfo;

				//theApp.InspPanel.StartSaveImage(
				//	(WPARAM)&pStInspectThreadParam[nThreadParamIndex]->MatOrgImage[pStImageInfo->nImageNo][nCamIndex],
				//	(LPARAM)&pStInspectThreadParam[nThreadParamIndex]->stImageInfo[pStImageInfo->nImageNo][nCamIndex]);


								//GVO要求修改为Auto时中间映像的安全性-190425YWS
				if (pStParamInsp->nInspType == (int)eAutoRun)
					theApp.SetCommonParameter(false);

				CostTime ct(true);
				int nThreadCount = theApp.InspPanel.StartInspection((WPARAM)pStInspectThreadParam[nThreadParamIndex], (LPARAM)pStImageInfo);
				LogSendToUI::getInstance()->SendAlgoLog(EModuleType::ALGO, ELogLevel::INFO_, EAlgoInfoType::IMG_INSP_START,
					ct.get_cost_time(), 0, pStImageInfo->strPanelID, theApp.m_Config.GetPCNum(), pStImageInfo->nImageNo, -1,
					_T("[%s]画面检测开始，等待:%d(ms) PID:%s"), theApp.GetGrabStepName(pStImageInfo->nImageNo), ct.get_cost_time(), pStImageInfo->strPanelID);
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection End. RetVal:%d, ImageNo:%d, ThreadCount:%d"), bReturn, pStParamInsp->nImageNum, nThreadCount);
			}
		}
	}
	SAFE_DELETE(pStParamInsp);

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tA(3) %.2f"), tact.Stop(false) / 1000.);

	return bReturn;
}
//2016.10.26
int WorkManager::Seq_ManualAlign(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	byte* tempParam = pParam;

	cv::Point	ptCorner[4];		// (通用)Cell最外围转角点
	cv::Point	ptContCorner[4];
	cv::Rect	AlignCellROI;		// Edge ROI
	cv::Rect	AlignPadROI;		// Pad ROI
	cv::Rect	AlignActiveROI;		// Active ROI
	double		AlignTheta = 0;		// 旋转角度
	cv::Point	AlignCenter;		// 旋转时,中心

	// Receive Parameter ////////////////////////////////////////////////////////////////////////////////////////////////////
	double		dAlignPara[MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT] = { 0, };			// Align Parameter-每个相机最多15个
	TCHAR		strOrgImagePath[1000];												// 源映像路径

	memcpy(dAlignPara, tempParam, sizeof(double) * MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT);
	tempParam += sizeof(double) * MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT;

	memcpy(strOrgImagePath, tempParam, sizeof(TCHAR) * 1000);				tempParam += sizeof(TCHAR) * 1000;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	char* pStr = NULL;
	pStr = CSTR2PCH(strOrgImagePath);
	cv::Mat MatAlignImage = cv::imread(pStr, IMREAD_UNCHANGED);
	SAFE_DELETE_ARR(pStr);

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq_ManualAlign Start. ImagePath : %s"), strOrgImagePath);

	CRect rectROI;
	cv::Point ptTemp;

	//运行Align算法(角度结果:dOriginTheta/转角结果:ptCorner)
	int nEQType = theApp.m_Config.GetEqpType();

	//18.02.12-手动Align相机固定0次
	//SCJ 18.08.03-添加手动Align Pannel Size和Pixel resolution

	CWriteResultInfo* WrtResultInfo;
	WrtResultInfo = new CWriteResultInfo;
	long nErrorCode = Align_FindActive(MatAlignImage, dAlignPara, AlignTheta, ptCorner, ptContCorner, 0, nEQType, theApp.GetCurWorkCoordInfo()->GetCamResolution(0),
		theApp.GetCurWorkCoordInfo()->GetPanelSizeX(), theApp.GetCurWorkCoordInfo()->GetPanelSizeY());

	//如果有错误,则输出错误代码&日志
	if (nErrorCode != 0)
	{
		// Error Log
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Error Occured In Seq_ManualAlign (1). ErrorCode : %d"), nErrorCode);
		return nErrorCode;
	}

	//旋转中心
	AlignCenter.x = MatAlignImage.cols / 2;
	AlignCenter.y = MatAlignImage.rows / 2;

	////画面旋转
//nErrorCode = Align_RotateImage(MatAlignImage, MatAlignImage, AlignTheta);

	////如果有错误,则输出错误代码&日志
//if( nErrorCode != 0 )
//{
//	// Error Log
//	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Error Occured In Seq_ManualAlign (2). ErrorCode : %d"), nErrorCode);
//	return nErrorCode;
//}

	//用旋转坐标校正

// LEFT - TOP
	Align_DoRotatePoint(ptCorner[0], ptTemp, AlignCenter, AlignTheta);
	rectROI.left = ptTemp.x;
	rectROI.top = ptTemp.y;

	// RIGHT - BOTTOM
	Align_DoRotatePoint(ptCorner[2], ptTemp, AlignCenter, AlignTheta);
	rectROI.right = ptTemp.x;
	rectROI.bottom = ptTemp.y;

	//AdjustOriginImage(MatOrgImage, MatDrawImage, &stCamAlignInfo[0]);

		//提交临时ROI结果
	AlignCellROI = cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height());
	AlignPadROI = AlignCellROI;
	AlignActiveROI = AlignCellROI;

	// test After Image Save
	//cv::imwrite("E:\\IMTC\\test_After.BMP", MatAlignedImg);

		//在Response Parameter中装载结果值的部分-如果需要更改尺寸,必须先更改Recive Parameter
	*(double*)pParam = AlignTheta;
	pParam += sizeof(double);
	*(cv::Rect*)pParam = AlignCellROI;
	pParam += sizeof(cv::Rect);
	*(cv::Rect*)pParam = AlignPadROI;
	pParam += sizeof(cv::Rect);
	*(cv::Rect*)pParam = AlignActiveROI;
	pParam += sizeof(cv::Rect);

	//12位单声道->8位单声道转换
// 	int nChCnt = MatAlignImage.channels();  int imgDepth = MatAlignImage.depth();
// 	if(nChCnt == 1 && imgDepth == CV_16U) MatAlignImage.convertTo(MatAlignImage, CV_8UC1, 1./16.);
// 	ImageSave(MatAlignImage, ALIGN_IMAGE_PATH);

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq_ManualAlign End (T : %lf). Return AlignActiveROI(%d,%d,%d,%d)"),
		AlignTheta, AlignActiveROI.x, AlignActiveROI.y, AlignActiveROI.width, AlignActiveROI.height);

	return APP_OK;
}

int WorkManager::Seq_SetParam(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	byte* tempParam = pParam;
	TCHAR strModelPath[200];
	TCHAR strModelID[50];
	ST_RECIPE_INFO* pStRecipeInfo = new ST_RECIPE_INFO();
	CS::Rect rcPadArea[MAX_CAMERA_COUNT];

	// Receive ///////////////////////////////////////////////////////////////////////////////////////////////////////

		//如果您有APPLY Model名,则不更新Recipe路径Path,APP Algorithm正在使用路径。
	//用于	//引用装入/保存功能。因为保存功能会暂时保存在APPLY Model Recipe中,所以实现了这些功能,NOME by 2017.11.22
	memcpy(strModelPath, tempParam, sizeof(WCHAR) * 200);			tempParam += sizeof(WCHAR) * 200;
	TCHAR strTest[50];
	TCHAR* strApply = _tcsstr(strModelPath, _T("APPLY"));
	_stprintf_s(strTest, 50, _T("%s"), strApply);
	if (_tcsncmp(strTest, _T("APPLY"), 5) != 0)
		theApp.SetCurInspRecipePath(strModelPath);

	memcpy(strModelID, tempParam, sizeof(WCHAR) * 50);				tempParam += sizeof(WCHAR) * 50;
	theApp.SetCurModelID(strModelID);

	memcpy(pStRecipeInfo, tempParam, sizeof(ST_RECIPE_INFO));		tempParam += sizeof(ST_RECIPE_INFO);

	//设置相机的虚线参数
	theApp.SetAlignParameter((double*)tempParam);
	tempParam += sizeof(double) * MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT;

	theApp.SetCommonParameter((ST_COMMON_PARA*)tempParam);
	tempParam += sizeof(ST_COMMON_PARA);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_SetParam Start. ModelID : %s, Path : %s"), strModelID, strModelPath);

	double dCamResolution[MAX_CAMERA_COUNT] = { 0.0, };
	for (int nCamIndex = 0; nCamIndex < theApp.m_Config.GetUseCamCount(); nCamIndex++)
	{
		//相机分辨率
		dCamResolution[nCamIndex] = pStRecipeInfo->stCamInfo[nCamIndex].dResolution;
		//		//特定于Camera的Pad区域
		// 		rcPadArea[nCamIndex] = pStRecipeInfo->stCamInfo[nCamIndex].rcPad;
		//		//加载特定于Camera的Pad Reference Image
		// 
		//		//2017.06.08 PAD ROI获取
		// 		theApp.SetPadRefROI(rcPadArea[nCamIndex], nCamIndex);
		// 
		//		//更改为17.06.20 Log
		// 				//CString strMsg = _T("");
		// 				if (!theApp.SetPadRefImage(pStRecipeInfo->stCamInfo[nCamIndex].strPadRefPath, nCamIndex))
		// 				{
		// 					theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Pad Reference Image Not Found [Path:%s]"), pStRecipeInfo->stCamInfo[nCamIndex].strPadRefPath);
		// 					strMsg.Format(_T("Pad Reference Image Not Found [Path:%s]"), pStRecipeInfo->stCamInfo[nCamIndex].strPadRefPath);
		// 					AfxMessageBox(strMsg);
		// 				}
		//			//加载相机特定的Mura Reference Image
		// 				if (!theApp.SetMuraRefImage(pStRecipeInfo->stCamInfo[nCamIndex].strMuraRefPath, nCamIndex))
		// 				{
		// 					theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Mura Reference Image Not Found [Path:%s]"), pStRecipeInfo->stCamInfo[nCamIndex].strMuraRefPath);
		// 					strMsg.Format(_T("Mura Reference Image Not Found [Path:%s]"), pStRecipeInfo->stCamInfo[nCamIndex].strMuraRefPath);
		// 					AfxMessageBox(strMsg);
		// 				}
	}
	///设置计算工作坐标所需的信息
	///17.07.07将现有工作坐标原点用作LT/RT/RB/LB的Offset
	theApp.SetWorkCoordInfo(pStRecipeInfo->dPanelSizeX, pStRecipeInfo->dPanelSizeY, pStRecipeInfo->nWorkDirection, pStRecipeInfo->nWorkOriginPosition,
		pStRecipeInfo->nOriginOffsetX, pStRecipeInfo->nOriginOffsetY,
		pStRecipeInfo->nDataDirection, pStRecipeInfo->nGateDataOriginPosition,
		pStRecipeInfo->nGateDataOriginOffsetX, pStRecipeInfo->nGateDataOriginOffsetY, pStRecipeInfo->dGatePitch, pStRecipeInfo->dDataPitch,
		dCamResolution);

	SAFE_DELETE(pStRecipeInfo);

	//17.11.03-[Round]注册存储路径
	bool bApply = false;

	//APPLY除外
	{
		CString strApply;
		strApply.Format(_T("%s"), strModelPath);
		int nRes = strApply.Find(_T("APPLY"));

		//未找到应用
		if (nRes == -1)
			theApp.SetRoundPath(strModelPath);
	}

	//加载算法参数(XML)文件
	if (theApp.ReadAlgorithmParameter(strModelPath, theApp.GetRoundPath()))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[算法参数] 加载成功. 模型路径 : % s"), strModelID);
	}
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[算法参数] 加载失败. 模型路径 : %s"), strModelID);
	}
		

	//加载算法参数(XML)文件
	if (theApp.ReadPadInspParameter(strModelPath))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[PAD区参数] 加载成功. 模型路径 : % s"), strModelID);
	}
		
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[PAD区参数] 加载失败. 模型路径 : %s"), strModelID);
	}

	// 加载缺陷合并规则
	if (theApp.ReadMergeRules(strModelPath))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[合并规则] 加载成功. 模型路径 : % s"), strModelID);
	}
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[合并规则] 加载失败. 模型路径 : % s"), strModelID);
	}

	// 加载PolMark模版
	if (theApp.ReadPolMarkTemplates(strModelPath))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[合并规则] 加载成功. 模型路径 : % s"), strModelID);
	}
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[合并规则] 加载失败. 模型路径 : % s"), strModelID);
	}

	//加载判定参数文件
	if (theApp.ReadJudgeParameter(strModelPath))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[判定规则] 加载成功. 模型路径 : % s"), strModelID);
	}
		
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[判定规则] 加载失败. 模型路径 : % s"), strModelID);
	}
		

	//自定义筛选器
	if (theApp.ReadUserDefinedFilter(strModelPath))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[自定义过滤规则] 加载成功. 模型路径 : % s"), strModelID);
	}
		
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[自定义过滤规则] 加载失败. 模型路径 : % s"), strModelID);
	}
		

	//代表不良评选排名
	if (theApp.ReadDefectClassify(strModelPath))
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[判级规则] 加载成功. 模型路径 : % s"), strModelID);

	}
		
	else
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("[判级规则] 加载失败. 模型路径 : % s"), strModelID);
	}
		

	//	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(Read Judge Tact : %.2f)"), tact.Stop(false)/1000.);

		//////////////////////////////////////////////////////////////////////////

	const int nEqpType = theApp.m_Config.GetEqpType();
	long nErrorCode = E_ERROR_CODE_TRUE;

#if CCD
	switch (nEqpType)
	{
	case EQP_AVI:
		// Load CCD Defect Index
		nErrorCode = CCD_IndexLoad(CCD_DEFECT_FILE_PATH, CCD_DEFECT_FILE_PATH2);

		//如果有错误,则输出错误代码&日志
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			// Error Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, TRUE, _T("Error Occured In Load CCD Defect Index. ErrorCode : %d"), nErrorCode);
		}
		else
		{
			// Successfully Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, TRUE, _T("Load CCD Defect Index Successfully."));
		}
		break;

	case EQP_SVI:
		//SVI色彩校正相关
		nErrorCode = ColorCorrection_Load(COLOR_CORRECTION_FILE_PATH);

		//如果有错误,则输出错误代码&日志
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			// Error Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, TRUE, _T("Error Occured In Load Color Correction. ErrorCode : %d"), nErrorCode);
		}
		else
		{
			// Successfully Log
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, TRUE, _T("Load Color Correction Successfully."));
		}
		break;

	case EQP_APP:
		break;

	default:
		break;
	}
#endif
	//////////////////////////////////////////////////////////////////////////

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_SetParam End. Model ID : %s"), strModelID);

	return APP_OK;
}

int WorkManager::Seq_FocusValue(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	byte* tempParam = pParam;
	UINT		nWidth, nHeight, nSize;

	nWidth = *(int*)tempParam;	tempParam += sizeof(int);
	nHeight = *(int*)tempParam;	tempParam += sizeof(int);
	nSize = nWidth * nHeight;

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq_FocusValue Start. ImageWidth:%d, ImageHeight:%d"), nWidth, nHeight);

	byte* Imagebuf;
	Imagebuf = new byte[nSize];
	Imagebuf = tempParam;			tempParam += nSize;

	Mat MatOrgImage = Mat(nHeight, nWidth, CV_8UC1, Imagebuf);
	CRect rtFocusRect(0, 0, nWidth, nHeight);

	double dFoucsVal = theApp.CallFocusValue(MatOrgImage, rtFocusRect);

	*(double*)pParam = dFoucsVal;
	pParam += sizeof(double);

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq_FocusValue End. Return FocusVal:%.3lf"), dFoucsVal);

	return APP_OK;

}
/**
 * struct TACT_TIME_DATA
{
	TCHAR strPanelID[50];
	TCHAR strTactName[50];  //
	TCHAR strTactState[50]; //S E
	TACT_TIME_DATA()
	{
		memset(strPanelID , 0X00, sizeof(TCHAR) * 50);
		memset(strTactName, 0X00, sizeof(TCHAR) * 50);		
		memset(strTactState, 0X00, sizeof(TCHAR) * 50);		
	}
};.
 * 
 * \param strPanelID
 * \param strDrive
 * \param nDefectCount
 * \param strPanelGrade
 * \param strJudgeCode1
 * \param strJudgeCode2
 * \return 
 */
int WorkManager::VS_Send_State_To_UI(CString strPanelID, CString TactName, CString TactState, int nGrabCount) 
{
	int nRet = APP_OK;
	int nParamSize = 0;
	byte bParam[1000] = { 0, };
	byte* bpTemp = &bParam[0];
	TACT_TIME_DATA* stState = new TACT_TIME_DATA;
	COPY_CSTR2TCH(stState->strTactName, TactName, sizeof(TCHAR) * 50);
	COPY_CSTR2TCH(stState->strTactState, TactState, sizeof(TCHAR) * 50);
	COPY_CSTR2TCH(stState->strPanelID, strPanelID, sizeof(TCHAR) * 50);
	
	*(int*)bpTemp = nGrabCount;
	bpTemp += sizeof(int);
	*(TACT_TIME_DATA*)bpTemp = *stState;
	bpTemp += sizeof(TACT_TIME_DATA);
	nParamSize = int(bpTemp - bParam);
	SAFE_DELETE(stState);
	nRet = CmdEditSend(SEND_UI_TACT_TIME_DATA, 0, USHORT(bpTemp - bParam), VS_UI_TASK, (byte*)&bParam, CMD_TYPE_NORES);

	return nRet;
}


int WorkManager::VS_Send_ClassifyEnd_To_Seq(CString strPanelID, CString strDrive, UINT nDefectCount, CString strPanelGrade, CString strJudgeCode1, CString strJudgeCode2)
{
	int nRet = APP_OK;
	int nParamSize = 0;
	byte bParam[1000] = { 0, };
	byte* bpTemp = &bParam[0];

	PARAM_CLASSIFY_END* stClassifyEnd = new PARAM_CLASSIFY_END;

	COPY_CSTR2TCH(stClassifyEnd->strPanelID, strPanelID, sizeof(stClassifyEnd->strPanelID));
	stClassifyEnd->nPCNum = theApp.m_Config.GetPCNum();
	stClassifyEnd->nDefectCount = nDefectCount;
	//更改为填写面板判定
	// 	if( stClassifyEnd->nDefectCount > 0)
	// 		stClassifyEnd->bIsOK = FALSE;
	// 	else
	// 		stClassifyEnd->bIsOK = TRUE;
	COPY_CSTR2TCH(stClassifyEnd->strPanelJudge, strPanelGrade, sizeof(stClassifyEnd->strPanelJudge));
	COPY_CSTR2TCH(stClassifyEnd->strJudgeCode1, strJudgeCode1, sizeof(stClassifyEnd->strJudgeCode1));
	COPY_CSTR2TCH(stClassifyEnd->strJudgeCode2, strJudgeCode2, sizeof(stClassifyEnd->strJudgeCode2));
	COPY_CSTR2TCH(stClassifyEnd->strSavedDrive, strDrive.Left(7), sizeof(stClassifyEnd->strSavedDrive));

	*(PARAM_CLASSIFY_END*)bpTemp = *stClassifyEnd;
	bpTemp += sizeof(PARAM_CLASSIFY_END);
	SAFE_DELETE(stClassifyEnd);

	nParamSize = (int)(bpTemp - bParam);
	nRet = CmdEditSend(SEND_SEQ_CLASSIFY_END, 0, (USHORT)(bpTemp - bParam), VS_SEQUENCE_TASK, bParam, CMD_TYPE_RES);
	if (nRet != APP_OK)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)PNL 汇总失败! 错误码 : %d"), strPanelID, nRet);
		return APP_NG;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)PNL 结果汇总完成!"), strPanelID);

	return nRet;
}



int WorkManager::VS_Send_Alarm_Occurred_To_MainPC(ENUM_INSPECT_MODE eInspMode, int nAlarmID, int nAlarmType, bool& bIsHeavyAlarm)
{
	//Main PC仅在Auto Mode时运行
	if (eInspMode != eAutoRun)
		return APP_OK;

	int nRet = APP_OK;
	int nParamSize = 0;
	byte bParam[1000] = { 0, };
	byte* bpTemp = &bParam[0];

	//中发生警报时,为其他线程控件标记是否发生警报ON
	if (nAlarmType == eALARMTYPE_HEAVY)
		bIsHeavyAlarm = true;

	*(int*)bpTemp = nAlarmID;			bpTemp += sizeof(int);
	*(int*)bpTemp = nAlarmType;		bpTemp += sizeof(int);

	nParamSize = (int)(bpTemp - bParam);
	nRet = CmdEditSend(SEND_MAINPC_ALARM_OCCURRED, 0, (USHORT)(bpTemp - bParam), VS_MAIN_PC_TASK, bParam, CMD_TYPE_RES);
	if (nRet != APP_OK)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Alarm Occurred Failed. (%d:%d)"), nAlarmID, nAlarmType);
		return APP_NG;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Alarm Occurred. (%d:%d)"), nAlarmID, nAlarmType);

	return nRet;
}

int WorkManager::VS_Send_Align_Result_To_MainPC(ENUM_INSPECT_MODE eInspMode, int nTheta, int nDistX, int nDistY, int nStageNo, int nPCNo)
{
	//Main PC仅在Auto Mode时运行
	if (eInspMode != eAutoRun)
		return APP_OK;

	int nRet = APP_OK;
	int nParamSize = 0;
	byte bParam[1000] = { 0, };
	byte* bpTemp = &bParam[0];

	*(int*)bpTemp = nTheta;			bpTemp += sizeof(int);
	*(int*)bpTemp = nDistX;			bpTemp += sizeof(int);
	*(int*)bpTemp = nDistY;			bpTemp += sizeof(int);
	*(int*)bpTemp = nStageNo;			bpTemp += sizeof(int);
	*(int*)bpTemp = nPCNo;				bpTemp += sizeof(int);

	nParamSize = (int)(bpTemp - bParam);
	nRet = CmdEditSend(SEND_MAINPC_ALIGN_RESULT, 0, (USHORT)(bpTemp - bParam), VS_MAIN_PC_TASK, bParam, CMD_TYPE_RES);
	if (nRet != APP_OK)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Send Align Result Failed. (%d:%d,%d)"), nTheta, nDistX, nDistY);
		return APP_NG;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Send Align Result. (%d:%d,%d)"), nTheta, nDistX, nDistY);

	return nRet;
}

void WorkManager::CreateResultDirectory(TCHAR* strPanelID, CString strDrive)
{
	//生成结果文件夹
	CString strPathfolder = _T("");
	strPathfolder.Format(_T("%s\\%s"), INSP_PATH, strPanelID);
	if (GetFileAttributes(strPathfolder) == -1)
		SHCreateDirectoryEx(NULL, strPathfolder, NULL);
	else
		DeleteAllFilesWithDirectory(strPathfolder);

	strPathfolder.Format(_T("%s\\%s\\ROI"), INSP_PATH, strPanelID);
	if (GetFileAttributes(strPathfolder) == -1)
		SHCreateDirectoryEx(NULL, strPathfolder, NULL);
	else
		DeleteAllFilesWithDirectory(strPathfolder);

	/**
	 * 创建算法特征参数保存路径
	 *
	 * \param strPanelID
	 * \param strDrive
	 */
	strPathfolder.Format(_T("%s\\%s\\AlgoFeatureParams"), INSP_PATH, strPanelID);
	if (GetFileAttributes(strPathfolder) == -1)
		SHCreateDirectoryEx(NULL, strPathfolder, NULL);
	else
		DeleteAllFilesWithDirectory(strPathfolder);

	strPathfolder.Format(_T("%s\\%s"), RESULT_PATH, strPanelID);
	if (GetFileAttributes(strPathfolder) == -1)
		SHCreateDirectoryEx(NULL, strPathfolder, NULL);
	else
		DeleteAllFilesWithDirectory(strPathfolder);

	//创建YHS 18.03.12-Merge_Tool Cell_ID文件夹
	//	金亨柱18.12.06
	//	MergeTool无条件操作,与Falg无关
//if(theApp.GetMergeToolUse())
	{
		strPathfolder.Format(_T("%s\\%s"), MERGETOOL_PATH, strPanelID);
		if (GetFileAttributes(strPathfolder) == -1)
			SHCreateDirectoryEx(NULL, strPathfolder, NULL);
		else
		{
			DeleteAllFilesWithDirectory(strPathfolder);
		}
	}

	//创建保存中间结果值的文件夹	
	if (theApp.GetCommonParameter()->bIFImageSaveFlag)
	{
		strPathfolder.Format(_T("%s\\%s"), ALG_RESULT_PATH, strPanelID);	// ARESULT-Panel ID更改为填写时,通过添加Panel ID修改为查找
		if (GetFileAttributes(strPathfolder) == -1)
			SHCreateDirectoryEx(NULL, strPathfolder, NULL);
		else
			DeleteAllFilesWithDirectory(strPathfolder);
	}
}

//返回Align的Pattern Num
int WorkManager::Seq_GetAlignPatternNum(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_GetAlignPatternNum Start."));
	int nEQPType = theApp.m_Config.GetEqpType();

	double* dAlignPara = theApp.GetAlignParameter(0);

	switch (nEQPType)
	{
	case EQP_AVI:
		*(int*)pParam = theApp.GetImageNum((int)dAlignPara[E_PARA_AD_IMAGE_NUM]);
		break;

	case EQP_SVI:
		*(int*)pParam = theApp.GetImageNum((int)dAlignPara[E_PARA_AD_IMAGE_NUM]);
		break;

	case EQP_APP:
		*(int*)pParam = theApp.GetImageNum((int)dAlignPara[E_PARA_APP_AD_IMAGE_NUM]);
		break;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_GetAlignPatternNum END. (RetVal : %d)"), *(int*)pParam);

	return APP_OK;
}

int WorkManager::Seq_StartAlign(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	// Receive //////////////////////////////////////////////////////////////////////////////////////////////////////
	byte* pReceiveParam = pParam;

	PARAM_ALIGN_START* pStParamAlign = (PARAM_ALIGN_START*)pReceiveParam;
	//memcpy(pStParamAlign, pReceiveParam, sizeof(PARAM_ALIGN_START));
	pReceiveParam += sizeof(PARAM_ALIGN_START);
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartAlign Start. StageNo:%d, PanelID:%s, VirtualID:%s, nImageNo:%d"),
		pStParamAlign->nStageNo, pStParamAlign->strPanelID, pStParamAlign->strVirtualID, pStParamAlign->nImageNum);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	long		nErrorCode = E_ERROR_CODE_TRUE;

	int			nWidth = 0;
	int			nHeight = 0;
	UINT32		nBufSize = 0;
	int			nBandWidth = 0;
	int			nBitRate = 0;

	cv::Mat		MatOrgImage;
	CString		strDrive = _T("");

	double dAlignPara[MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT] = { 0.0, };
	memcpy(dAlignPara, theApp.GetAlignParameter(0), sizeof(double) * MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT);

	//Align Sequence不使用P/S Image,因此将Ratio应用于Parameter
	dAlignPara[E_PARA_CELL_SIZE_X] *= (1.0 / (pStParamAlign->nSeqMode[0] + 1));
	dAlignPara[E_PARA_CELL_SIZE_Y] *= (1.0 / (pStParamAlign->nSeqMode[0] + 1));

	//设备类型
	int nEqpType = theApp.m_Config.GetEqpType();

	// test
	CCPUTimer tact;
	tact.Start();

	for (int nCamIndex = 0; nCamIndex < MAX_CAMERA_COUNT; nCamIndex++)
	{
		if (pStParamAlign->bUseCamera[nCamIndex])
		{
			if (_tcscmp(pStParamAlign->strPanelID, _T("")) == 0)
				_tcscpy(pStParamAlign->strPanelID, _T("Unknown"));

			//Align Sequence必须为Non-P/S Image
			nWidth = m_pSharedMemory[nCamIndex]->GetImgWidth();// * (pStParamAlign->nSeqMode[nCamIndex] + 1);
			nHeight = m_pSharedMemory[nCamIndex]->GetImgHeight();// * (pStParamAlign->nSeqMode[nCamIndex] + 1);
			nBandWidth = m_pSharedMemory[nCamIndex]->GetImgBandWidth();
			nBitRate = m_pSharedMemory[nCamIndex]->GetImgBitrate();

			CString strManualImagePath = _T("");
			CString strOrgFileName = _T("");	// 基于原始图像文件的名称(Grab Number除外)
			//文件名为Alg。更改为在Task中确定-多摄像头响应
			strOrgFileName.Format(_T("ALIGN_CAM%02d"), nCamIndex);

			{		//从共享内存读取
				//添加12位响应
				if (nBitRate == 8)
				{
					if (nBandWidth == 1)
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC1, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamAlign->nImageNum));
					else
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_8UC3, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamAlign->nImageNum));
				}
				else
				{
					if (nBandWidth == 1)
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_16UC1, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamAlign->nImageNum));
					else
						MatOrgImage = cv::Mat(nHeight, nWidth, CV_16UC3, m_pSharedMemory[nCamIndex]->GetImgAddress(pStParamAlign->nImageNum));
				}
			}

			if (MatOrgImage.empty())
			{
				CString strMsg = _T("");
				strMsg.Format(_T("!!! Grab Error !!! \r\n\t (Step : %s)"), theApp.GetGrabStepName(pStParamAlign->nImageNum));
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, strMsg.GetBuffer(0));

				return APP_NG;
			}

			//临时
			//保存原始图像-添加Grab Step(按当前模型Grab Step的顺序重新排序并保存)
			if (nBitRate == 8)
				strOrgFileName = strOrgFileName + _T(".bmp");
			else
				strOrgFileName = strOrgFileName + _T(".tiff");
			theApp.CheckDrive();
#if 0
			strDrive = theApp.m_Config.GetINIDrive();
#else
			strDrive = this->GetNewDriveForAlg();
			//strDrive = theApp.m_Config.GetResultDriveForAlg();
#endif
			CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
			ImageSave(MatOrgImage, _T("%s\\%s\\%s"),
				ORIGIN_PATH, pStParamAlign->strPanelID,
				strOrgFileName);

			theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tB(1) %.2f"), tact.Stop(false) / 1000.);

			double dResult[4] = { 0.0, };

			//图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)
			//根据PS画面,是否需要更改为单镜头坐标,Align Sequence不使用P/S Image。
// nRatio : 1

			nErrorCode = Align_FindDefectAD(MatOrgImage, dAlignPara, dResult, 1, nCamIndex, nEqpType);

			theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tB(2) %.2f"), tact.Stop(false) / 1000.);

			if (nErrorCode != E_ERROR_CODE_TRUE)	//AD不良时不Align-只留下Log,进行正常序列(AD不良报告)
			{
				// Alg DLL Stop Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE,
					_T("AD Defect - InspStop !!!. PanelID: %s, Stage: %02d, CAM: %02d, Img: %s..\n\t\t\t\t( Rate : %.2f %% / Avg : %.2f %% / Std : %.2f %% ) ErrorCode : %d"),
					pStParamAlign->strPanelID, pStParamAlign->nStageNo, nCamIndex, theApp.GetGrabStepName(pStParamAlign->nImageNum), dResult[0], dResult[1], dResult[2], nErrorCode);
				nErrorCode = E_ERROR_CODE_TRUE;		// 在Camera Align Sequence中返回TRUE以报告AD故障
			}
			else
			{
				//17.07.10添加Cell中心坐标
				cv::Point ptCellCenter;
				nErrorCode = Align_FindTheta(MatOrgImage, dAlignPara, pStParamAlign->dAdjTheta[0], ptCellCenter, pStParamAlign->strPanelID);

				//图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)
					//根据PS画面,是否需要更改为单镜头坐标,Align Sequence不使用P/S Image。
	//ptCellCenter.x /= nRatio;
	//ptCellCenter.y /= nRatio;

				//如果有错误,则输出错误代码和日志
				if (nErrorCode != E_ERROR_CODE_TRUE)
				{
					// Error Log
					theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Error Occured In Seq_StartAlign. ErrorCode : %d"), nErrorCode);
					pStParamAlign->dAdjTheta[0] = 0.0;
				}
				else
				{
					theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE,
						_T("Success Seq_StartAlign. PanelID : %s, Stage : %d, Theta : %lf, Cell Center X : %d, Y : %d"),
						pStParamAlign->strPanelID, pStParamAlign->nStageNo, pStParamAlign->dAdjTheta[0], ptCellCenter.x, ptCellCenter.y);
				}
			}
		}
	}
	//SAFE_DELETE(pStParamAlign);

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("\t\tB(3) %.2f"), tact.Stop(false) / 1000.);

	return E_ERROR_CODE_TRUE;	// 即使为了进行物流而Align失败,也无条件TRUE Return
}

int WorkManager::Seq_WriteCCDIndex(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode, bool bBusyCheck, bool bSeqResetPossible)
{
	byte* tempParam = pParam;

	// Receive Parameter ////////////////////////////////////////////////////////////////////////////////////////////////////
	TCHAR		strOrgImagePath[1000];												// 源映像路径
	memcpy(strOrgImagePath, tempParam, sizeof(TCHAR) * 1000);				tempParam += sizeof(TCHAR) * 1000;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq_WriteCCDIndex Start. ImagePath : %s"), strOrgImagePath);

	char* pStr = NULL;
	pStr = CSTR2PCH(strOrgImagePath);
	cv::Mat MatCCDImage = cv::imread(pStr, IMREAD_UNCHANGED);
	SAFE_DELETE_ARR(pStr);

	int nRet = theApp.CheckImageRatio(1, MatCCDImage.cols, MatCCDImage.rows, m_pSharedMemory[0]->GetImgWidth(), m_pSharedMemory[0]->GetImgHeight());
	if (nRet != 0)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Load CCD Image Ratio Incorrect (Path:%s) !!!"), strOrgImagePath);
		return APP_NG;
	}

	long nErrorCode = CCD_IndexSave(MatCCDImage, CCD_DEFECT_FILE_PATH, CCD_DEFECT_FILE_PATH2);

	//如果有错误,则输出错误代码&日志
	if (nErrorCode != 0)
	{
		// Error Log
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Error Occured In Seq_WriteCCDIndex. ErrorCode : %d"), nErrorCode);
		return nErrorCode;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, FALSE, TRUE, _T("Seq_WriteCCDIndex End. (Save File Path : %s"), CCD_DEFECT_FILE_PATH);

	return APP_OK;
}

int WorkManager::VS_Send_Dust_Result_To_Seq(ENUM_INSPECT_MODE eInspMode, bool bNeedRetry, bool bIsNormalDust)
{
	//Main PC仅在Auto Mode时运行
	if (eInspMode != eAutoRun)
		return APP_OK;

	int nRet = APP_OK;
	int nParamSize = 0;
	byte bParam[1000] = { 0, };
	byte* bpTemp = &bParam[0];

	*(bool*)bpTemp = bNeedRetry;				bpTemp += sizeof(bool);
	*(bool*)bpTemp = bIsNormalDust;			bpTemp += sizeof(bool);

	nParamSize = (int)(bpTemp - bParam);
	nRet = CmdEditSend(SEND_SEQ_DUST_RESULT, 0, (USHORT)(bpTemp - bParam), VS_SEQUENCE_TASK, bParam, CMD_TYPE_RES);
	if (nRet != APP_OK)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Send Dust Result Failed !!! result : %s (ErrorCode : %d)"), bIsNormalDust ? _T("OK") : _T("NG"), nRet);
		return APP_NG;
	}

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Send Dust Result. result : %s"), bIsNormalDust ? _T("OK") : _T("NG"));

	return nRet;
}

//LJH配方版本传递函数
int WorkManager::VS_Send_RcpVer_To_UI(CString ver)
{
	byte* pParam = new byte[200];
	byte* pSendParam = pParam;
	int			nRet = APP_OK;

	RCP_VERSION* rcpVer = new RCP_VERSION;

	COPY_CSTR2TCH(rcpVer->Ver, ver, sizeof(rcpVer->Ver));

	*(RCP_VERSION*)pSendParam = *rcpVer;

	pSendParam += sizeof(RCP_VERSION);

	SAFE_DELETE(rcpVer);

	EXCEPTION_TRY
		nRet = CmdEditSend(SEND_UI_RCPVER, 0, (ULONG)(pSendParam - pParam), VS_UI_TASK, pParam, CMD_TYPE_NORES);
	EXCEPTION_CATCH

		if (nRet != APP_OK)
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("VS_Send_RcpVer_To_UI Error Occured. RetVal=%d \n"), nRet);
		}

	SAFE_DELETE_ARR(pParam);

	return nRet;
}

CString  WorkManager::GetNewDriveForAlg()
{
	CString newDir;
	CString strMidName = _T(":\\ALG_");
	//newDir = theApp.m_Config.GetINIDrive().Left(1) + strMidName;
	newDir = theApp.m_Config.GetResultDrive() + strMidName;
	
	newDir += theApp.m_Config.GetEqpType() == EQP_AVI ? _T("AVI") : (theApp.m_Config.GetEqpType() == EQP_SVI ? _T("SVI") : _T("APP"));
	newDir.Format(_T("%s_%s%s"), newDir, theApp.m_Config.GetPCName(), _T("\\"));
	return newDir;
}

