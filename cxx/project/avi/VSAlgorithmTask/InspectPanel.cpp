// InspCam.cpp : implementation file
//

#include "stdafx.h"
#include "InspectPanel.h"
#include "AviInspection.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CInspPanel

CInspPanel::CInspPanel()
{
	for (int nThrdCnt = 0; nThrdCnt < MAX_THREAD_COUNT; nThrdCnt++)
		m_pInspThrd[nThrdCnt] = NULL;
}

CInspPanel::~CInspPanel()
{
}

// BEGIN_MESSAGE_MAP(CInspPanel, CObject)
// 	//{{AFX_MSG_MAP(CInspPanel)
// 	//}}AFX_MSG_MAP
// END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CInspPanel message handlers
bool CInspPanel::InitVision()
{
	m_nCurThrdID = 0;
	theApp.m_nDefectCnt = 0;

	for (int nThrdCnt = 0; nThrdCnt < MAX_THREAD_COUNT; nThrdCnt++)
	{
		if (!m_pInspThrd[nThrdCnt])
		{
			if (theApp.m_Config.GetEqpType() == EQP_AVI)
			{
				m_pInspThrd[nThrdCnt] = (CInspThrd*)
					AfxBeginThread(RUNTIME_CLASS(AviInspection), THREAD_PRIORITY_NORMAL, 0, CREATE_SUSPENDED);
			}
			m_pInspThrd[nThrdCnt]->Initialize(nThrdCnt);
			m_pInspThrd[nThrdCnt]->ResumeThread();
		}
	}
	return true;
}

bool CInspPanel::ExitVision()
{
	for (int nThrdCnt = 0; nThrdCnt < MAX_THREAD_COUNT; nThrdCnt++)
	{
		if (m_pInspThrd[nThrdCnt] != NULL)
		{
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, FALSE, _T("send WM_INSPECT_END_THREAD to Image %d's Camera %d"), i, k);
			m_pInspThrd[nThrdCnt]->ExitInstance();
			delete m_pInspThrd[nThrdCnt];
		}
	}

	return true;
}

int CInspPanel::CheckInsThread()
{
	int nThreadCount = 0;

	while (1)
	{
		for (nThreadCount = 0; nThreadCount < MAX_THREAD_COUNT; nThreadCount++)
		{
			tInspectThreadParam* pInspectThreadParam = (tInspectThreadParam*)m_pInspThrd[nThreadCount]->GetAlgoWParam();
			STRU_IMAGE_INFO* pStImageInfo = (STRU_IMAGE_INFO*)m_pInspThrd[nThreadCount]->GetAlgoLParam();
			//if (m_pInspThrd[nThreadCount]->IsThrdBusy())
				//if (pInspectThreadParam != NULL || pStImageInfo != NULL)
				if (pInspectThreadParam != NULL && pStImageInfo != NULL)
				if (pInspectThreadParam->bInspectEnd[pStImageInfo->nImageNo][pStImageInfo->nCameraNo] == true)
				{
					break;
					//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- ThreadTime %d ms.Restart Thread Num : %d"), m_pInspThrd[nThreadCount]->ThreadTime(), nThreadCount);
				}
		}
	}
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- ThreadTime %d ms.Restart Thread Num : %d"), m_pInspThrd[nThreadCount]->ThreadTime(), nThreadCount);
		for (nThreadCount = 0; nThreadCount < MAX_THREAD_COUNT; nThreadCount++)
		{

			//寻找正在使用的线程
			if (m_pInspThrd[nThreadCount]->IsThrdBusy() && false)
			{
				//超时
				if (m_pInspThrd[nThreadCount]->IsThreadTimeout()) {
					//kill
					if (m_pInspThrd[nThreadCount] != NULL)
					{
						DWORD dwExitCode = 0;
						if (GetExitCodeThread(m_pInspThrd[nThreadCount]->m_hThread, &dwExitCode) && dwExitCode == STILL_ACTIVE)
						{
							theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- TimeOut %d s.Restart Thread Num : %d"), m_pInspThrd[nThreadCount]->GetTimeout(), nThreadCount);
							tInspectThreadParam* pInspectThreadParam = (tInspectThreadParam*)m_pInspThrd[nThreadCount]->GetAlgoWParam();
							STRU_IMAGE_INFO* pStImageInfo = (STRU_IMAGE_INFO*)m_pInspThrd[nThreadCount]->GetAlgoLParam();
							
							if (pStImageInfo->nImageNo == 0 && pStImageInfo->nCameraNo == 0) continue;
							pInspectThreadParam->bHeavyAlarm = TRUE;
							
							pInspectThreadParam->bInspectEnd[pStImageInfo->nImageNo][pStImageInfo->nCameraNo] = true;
							// 终止线程
							//TerminateThread(m_pInspThrd[nThreadCount]->m_hThread, 0);
							WaitForSingleObject(m_pInspThrd[nThreadCount]->m_hThread, INFINITE);
							CloseHandle(m_pInspThrd[nThreadCount]->m_hThread);
							m_pInspThrd[nThreadCount]->m_hThread = NULL;
							delete m_pInspThrd[nThreadCount];
							m_pInspThrd[nThreadCount] = NULL;

							//重新初始化并拉起该线程
							m_pInspThrd[nThreadCount] = (CInspThrd*)AfxBeginThread(RUNTIME_CLASS(AviInspection), THREAD_PRIORITY_NORMAL, 0, CREATE_SUSPENDED);
							m_pInspThrd[nThreadCount]->Initialize(nThreadCount);
							m_pInspThrd[nThreadCount]->ResumeThread();
							//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- TimeOut %d s.Restart Thread Num : %d"), m_pInspThrd[nThreadCount]->GetTimeout(), nThreadCount);
							Sleep(1000);
						}
					}
				}
			}

		}

	return 0;
}



int CInspPanel::StartInspection(WPARAM wParam, LPARAM lParam)
{
	int nThreadCount = 0;
	bool bFindFreeThread = false;

	while (!bFindFreeThread)
	{
		for (nThreadCount = 0; nThreadCount < MAX_THREAD_COUNT; nThreadCount++)
		{
			if (!m_pInspThrd[nThreadCount]->IsThrdBusy())
			{
				bFindFreeThread = true;
				break;
			}
		}
	}
	//MSG msg1;
	//DWORD aaa = GetMessage(&msg1,NULL,0,0);
	// 设置超时时间为10秒
	DWORD dwTimeout = 40000;
	m_pInspThrd[nThreadCount]->SetTimeout(dwTimeout);
	m_pInspThrd[nThreadCount]->SetAlgoParam(wParam, lParam);
	int nRet = m_pInspThrd[nThreadCount]->PostThreadMessage(WM_START_INSPECTION, wParam, lParam);

	//int nRet = m_pInspThrd[nThreadCount]->PostThreadMessage(WM_QUIT, wParam, lParam);
	DWORD errValue = GetLastError();
	CString errTemp;
	errTemp.Format(L"%lu", (LONG)errValue);
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- Start Log. nRet : %d, errValue : %s, Thread Num : %d"), nRet, errTemp, nThreadCount);

	if (!nRet)
	{
		DWORD errValue = GetLastError();
		CString errTemp;
		errTemp.Format(L"%lu", (LONG)errValue);
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- Start Fail. nRet : %d, errValue : %s, Thread Num : %d"), nRet, errTemp, nThreadCount);
		nThreadCount = -1;
	}
	else
	{
		/*int nSleepTime = 0;
		do
		{
			Sleep(10);
			nSleepTime += 10;
		} while (m_pInspThrd[nThreadCount]->IsThrdBusy() == false);*/

		int nSleepTime = 0;

		while (m_pInspThrd[nThreadCount]->IsThrdBusy() == false)
		{
			Sleep(1);
			nSleepTime += 1;

			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Test Log. \nStart Inspect SleepTime %d"), nSleepTime);
		}

		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection -- Start OK. Sleep Time : %d, Thread Num : %d"), nSleepTime, nThreadCount);
	}

	return nThreadCount;
}

int CInspPanel::StartSaveImage(WPARAM wParam, LPARAM lParam)
{
	int nThreadCount = 0;
	bool bFindFreeThread = false;
	CostTime ct(true);

	while (!bFindFreeThread)
	{
		for (nThreadCount = 0; nThreadCount < MAX_THREAD_COUNT; nThreadCount++)
		{
			if (!m_pInspThrd[nThreadCount]->IsThrdBusy())
			{
				bFindFreeThread = true;
				break;
			}
		}
	}

	tImageInfo* imageInfo = (tImageInfo*)lParam;
	LogSendToUI::getInstance()->SendAlgoLog(EModuleType::ALGO, ELogLevel::INFO_, EAlgoInfoType::IMG_WRITE_START,
		ct.get_cost_time(), 0, imageInfo->panelId, theApp.m_Config.GetPCNum(), imageInfo->imageNo, -1,
		_T("[%s]保存图片开始，等待:%d(ms) PID:%s"), theApp.GetGrabStepName(imageInfo->imageNo), ct.get_cost_time(), imageInfo->panelId);

	int nRet = m_pInspThrd[nThreadCount]->PostThreadMessage(WM_START_SAVE_IMAGE, wParam, lParam);

	DWORD errValue = GetLastError();
	CString errTemp;
	errTemp.Format(L"%lu", (LONG)errValue);
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection(Save Image) -- Start Log. nRet : %d, errValue : %s, Thread Num : %d"), nRet, errTemp, nThreadCount);

	if (!nRet)
	{
		DWORD errValue = GetLastError();
		CString errTemp;
		errTemp.Format(L"%lu", (LONG)errValue);
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection(Save Image) -- Start Fail. nRet : %d, errValue : %s, Thread Num : %d"), nRet, errTemp, nThreadCount);
		nThreadCount = -1;
	}
	else
	{
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Test Log. \nStart Sleep")); //////////////////////////////////////////////////////////////////////////////////////////////////////
		/*int nSleepTime = 0;
		do
		{
			Sleep(nSleepTime);  // 10
			//Sleep(10);  // 10
			nSleepTime += 10;
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Test Log. \nStart SleepTime %d"), nSleepTime); //////////////////////////////////////////////////////////////////////////////////////////////////////
		} while (m_pInspThrd[nThreadCount]->IsThrdBusy() == false);*/
		int nSleepTime = 0;

		while (m_pInspThrd[nThreadCount]->IsThrdBusy() == false)
		{
			Sleep(nSleepTime);  // 10
			nSleepTime += 1;

			if (nSleepTime >= 100)
			{
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Waiting -- Message Sending Fail.  Thread Num : %d  --> StartImageSave Error"), nThreadCount);
				break;
			}
		}

		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("Seq_StartInspection(Save Image) -- Start OK. Sleep Time : %d, Thread Num : %d"), nSleepTime, nThreadCount);
	}

	return nThreadCount;
}