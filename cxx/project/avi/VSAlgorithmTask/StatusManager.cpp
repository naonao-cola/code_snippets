#include "stdafx.h"
#include "StatusManager.h"
#include "ErrorProcess.h"
#include "..\..\VisualStation\CommonHeader\ErrorCode\ClientDllErrorCode.h"
//#include "IniProcess.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

StatusManager::StatusManager()
{
}

StatusManager::~StatusManager()
{

}

// Melsec Memory Area を 処理するThreadを開始
int StatusManager::Start()
{
	HANDLE handle;
	StatusManagerQueueParam* param;

		//COMMENT,用于在不使用监视缓冲区的情况下仅使用日志记录功能
	//
		//如果不使用/////PLC始终监视缓冲区,则不需要以下线程。
	//////if (m_nUsePlcBuffer == 0)
	//////{
	//////	return APP_OK;
	//////}

	param = new StatusManagerQueueParam;

	param->data = IN_UPDATE_THREAD;
	m_StatusMessageQueue.m_fnPushMessage(param);

	param = new StatusManagerQueueParam;

	param->data = OUT_UPDATE_THREAD;
	m_StatusMessageQueue.m_fnPushMessage(param);

	param = new StatusManagerQueueParam;

	param->data = USER_DEFINE_UPDATE_THREAD;
	m_StatusMessageQueue.m_fnPushMessage(param);

	handle = m_fnStartThread();
	if ( handle == NULL || handle == (HANDLE)-1 )
		return APP_NG;

	handle = m_fnStartThread();
	if ( handle == NULL || handle == (HANDLE)-1 )
		return APP_NG;

	handle = m_fnStartThread();
	if ( handle == NULL || handle == (HANDLE)-1 )
		return APP_NG;

	return APP_OK;
}

void StatusManager::m_fnRunThread()
{
	StatusManagerQueueParam* param;
	param = m_StatusMessageQueue.m_fnPopMessage();

	switch ( param->data )
	{
	case IN_UPDATE_THREAD:
		delete param;
		InStatusManagerThread();
		break;

	case OUT_UPDATE_THREAD:
		delete param;
		OutStatusManagerThread();
		break;

	case USER_DEFINE_UPDATE_THREAD:
		delete param;
		UserDefineStatusManagerThread();
		break;
	}
}

void StatusManager::InStatusManagerThread()
{
	int nRet = APP_OK;

	return;

	Sleep(3000);

	while (GetThreadRunFlag())
	{

		EXCEPTION_TRY

				//随时检查其他塔斯克的状态值,做自己需要做的事情。

		EXCEPTION_CATCH

		if ( nRet != APP_OK )
		{

			m_fnPrintLog(_T("ERROR OutBitUpdateThread. Error Code = %d \n"), nRet);

		}
	}
}

void StatusManager::OutStatusManagerThread()
{
	int nRet = APP_OK;

	return;

	Sleep(3000);

	while (GetThreadRunFlag())
	{

		EXCEPTION_TRY

				//随时检查自己的状态值,并告知其他塔斯克。

		EXCEPTION_CATCH

		if ( nRet != APP_OK )
		{
			_tprintf(_T("ERROR OutBitUpdateThread. Error Code = %d \n"), nRet);
		}
	}
}

void StatusManager::UserDefineStatusManagerThread()
{

	int		nRet = APP_OK;
	bool	bIsFirstRun	= true;

	return;

	Sleep(3000);

	while (GetThreadRunFlag())
	{
		EXCEPTION_TRY

				//经常与其他清真寺接口时使用的线程

		EXCEPTION_CATCH

		if ( nRet != APP_OK )
		{
			m_fnPrintLog(_T("ERROR UserDefineStatusManagerThread. Error Code = %d \n"), nRet);

		}

		Sleep(300);

	}
}

