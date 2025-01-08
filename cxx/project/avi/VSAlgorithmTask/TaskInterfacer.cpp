#include "stdafx.h"
#include "TaskInterfacer.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

TaskInterfacer::TaskInterfacer(int nTaskNo, int nServerPort, TSTRING szServerIP,
		CString szLogFilePath,			///<日志文件路径
		TSTRING szLogFileName,			///<日志文件名
		bool bAutoExitFlag ,			///<如果注册了Flag,Windows handle,则在服务器关闭时是否自动关闭应用程序,则不能自动关闭。
		int uPeriodDay,					///<日志保留期限
		int uWriteTerm,					///<每隔几个小时生成一次日志文件...时间,0=一个文件生成一天的日志。
	USHORT uLogBlockLevel,
	HWND hPwnd
	)
		//member 変数,member Classの初期化
	: 
	m_bThreadRun(true), m_nTaskNo(nTaskNo), m_nServerPort(nServerPort),
		//Parent Classの初期化
	WorkManager(), 
	StatusManager(), 
	CInterServerInterface((TSTRING)szLogFilePath, szLogFileName, NULL, NULL, bAutoExitFlag, uPeriodDay, uWriteTerm, true, uLogBlockLevel)
{
	memset(m_szServerIP, 0, sizeof(m_szServerIP));
	_tcscpy(m_szServerIP, szServerIP.c_str());
	m_hParentWnd = hPwnd;
	LogSendToUI::getInstance()->RegisterServerInterFace(this);
}

TaskInterfacer::~TaskInterfacer()
{
	m_bThreadRun = false;
	Sleep(1000);
}

int TaskInterfacer::Start()
{
	int nRet;

	CInterServerInterface::Log(LOG_LEVEL_INFO, DEFAULT_LOG_SIZE,  _T("Start Algorithm Task !!\n"));

	// Visual Station Server 登録
	nRet = m_fnRegisterTask(m_nTaskNo, m_szServerIP, m_nServerPort);
	if ( nRet != APP_OK )
		return nRet;

		//Indexerの状態をMonitoringして Event発生。INAF更新
	nRet = StatusManager::Start();
	if ( nRet != APP_OK )
		return nRet;

		//Inder 動作 Sequenceの処理。
	nRet = WorkManager::Start();
	if ( nRet != APP_OK )
		return nRet;	

	return nRet;

}

// Threadを実行するか確認Flag を Get
bool TaskInterfacer::GetThreadRunFlag()
{
	return m_bThreadRun;
}

// Parent ClassのIndexerStatusManager と Indexer Work Manager から 使われる。
int TaskInterfacer::CmdEditSend(USHORT FunID, USHORT SeqNo, USHORT UnitNo, ULONG PrmSize, USHORT TaskNo, byte* Buff, USHORT uMsgType, ULONG lMsgTimeout )
{
	if (!m_fnGetServerConnected())
		return APP_NG;

	int nRet = APP_OK;
	CMDMSG tempCmdMsg;

	tempCmdMsg.uTask_Src	= m_nTaskNo;
	tempCmdMsg.uTask_Dest	= TaskNo;
	tempCmdMsg.uFunID_Dest	= FunID;
	tempCmdMsg.uSeqID_Dest	= SeqNo;
	tempCmdMsg.uUnitID_Dest = UnitNo;
	tempCmdMsg.uMsgSize		= PrmSize;
	tempCmdMsg.cMsgBuf		= (INT64)Buff;
	tempCmdMsg.uMsgType		= uMsgType;
	tempCmdMsg.dwTimeOut	= lMsgTimeout;

	nRet = m_fnSendCommand(&tempCmdMsg, LOG_LEVEL_DEBUG);
	
	return nRet;
}
BYTE* TaskInterfacer::m_fnPeekMessage()
{
	return ::CInterServerInterface::m_fnPeekMessage();
}

// Parent Classの Indexer Work Manager から 使われる。
void TaskInterfacer::EndWorkProcess(BYTE* pMsg)
{
	m_fnFreeMemory(pMsg);
}

// Parent Classの Indexer Work Manager から 使われる。
int TaskInterfacer::ResponseSend(USHORT ReturnVal, CMDMSG* pCmdMsg)
{
	int nRet = APP_OK;
	CMDMSG tempCmdMsg;
	byte* pBuff;

	pBuff = new byte[pCmdMsg->uMsgSize];

	memcpy(pBuff, (void*)pCmdMsg->cMsgBuf, pCmdMsg->uMsgSize);
	memcpy(&tempCmdMsg, pCmdMsg, sizeof(CMDMSG));
	tempCmdMsg.cMsgBuf = (INT64)pBuff;

	tempCmdMsg.uRetStatus = ReturnVal;

	nRet = m_fnSendResponseCommand(&tempCmdMsg);

	delete pBuff;

	return nRet;
}

int	 TaskInterfacer::m_fnPrintLog(const wchar_t* pszLogData, ...)		//bSendUI:是否启用UI Log发送功能
{

	va_list vaList;
	va_start(vaList, pszLogData);
	int len = _vscwprintf( pszLogData, vaList ) + 1;
	//wchar_t *cBuffer = (wchar_t*)malloc(sizeof(wchar_t)*len);
	wchar_t *cBuffer	= new wchar_t[sizeof(wchar_t)*len];
	wchar_t *cTempBuff	= new wchar_t[sizeof(wchar_t)*len];
	memset(cBuffer,0,sizeof(wchar_t)*len);
	memset(cTempBuff,0,sizeof(wchar_t)*len);

	if (cBuffer)
	{
		vswprintf(cBuffer, pszLogData, (va_list)vaList);
		memcpy(cTempBuff, cBuffer, sizeof(wchar_t)*len);
		//vswprintf(cTempBuff, pszLogData, (va_list)vaList);

		::PostMessage(m_hParentWnd, WM_PRINT_UI_LOG_MSG_UNICODE, len, (LPARAM)cTempBuff);	// Print Log GUI

		::CInterServerInterface::Log(LOG_LEVEL_INFO, cBuffer );
		//free(cBuffer);
		SAFE_DELETE(cBuffer);
	}

	va_end(vaList);

	return APP_OK;
}

int	 TaskInterfacer::m_fnPrintLog(const char* pszLogData, ...)
{
	va_list vaList;
	va_start(vaList, pszLogData);
	int len = _vscprintf( pszLogData, vaList ) + 1;
	char	*cBuffer	= new char[sizeof(char)*len];
	char	*cTempBuff	= new char[sizeof(char)*len];
	memset(cBuffer,0,sizeof(char)*len);
	memset(cTempBuff,0,sizeof(char)*len);

	if (cBuffer)
	{
		vsprintf(cBuffer, pszLogData, (va_list)vaList);
		memcpy(cTempBuff, cBuffer, sizeof(char)*len);
		//vsprintf(cTempBuff, pszLogData, (va_list)vaList);

		::PostMessage(m_hParentWnd, WM_PRINT_UI_LOG_MSG_MULTI_BYTE, len, (LPARAM)cTempBuff);	// Print Log GUI
		::CInterServerInterface::Log(LOG_LEVEL_INFO, cBuffer );
		//free(cBuffer);
		delete [] cBuffer;
	}

	va_end(vaList);

	return APP_OK;
}

int	 TaskInterfacer::m_fnSendUILog(const wchar_t* pszLogData, ...)
{
	int nRet = APP_OK;
	va_list vaList;
	va_start(vaList, pszLogData);
	int len = _vscwprintf( pszLogData, vaList ) + 1;
	wchar_t *cBuffer	= new wchar_t[sizeof(wchar_t)*len];
	wchar_t *cTempBuff	= new wchar_t[sizeof(wchar_t)*len];
	memset(cBuffer,0,sizeof(wchar_t)*len);
	memset(cTempBuff,0,sizeof(wchar_t)*len);

	if (cBuffer)
	{
		vswprintf(cBuffer, pszLogData, (va_list)vaList);
		memcpy(cTempBuff, cBuffer, sizeof(wchar_t)*len);
		//vswprintf(cTempBuff, pszLogData, (va_list)vaList);

		//::PostMessage(m_hParentWnd, WM_PRINT_UI_LOG_MSG_UNICODE, len, (LPARAM)cTempBuff);	// Print Log GUI

		nRet = ::WorkManager::VS_Send_Log_To_UI(cBuffer, sizeof(wchar_t)*len);
		//free(cBuffer);
		delete [] cBuffer;
	}

	va_end(vaList);

	return nRet;
}

bool TaskInterfacer::GetVSState()
{
	return m_fnGetServerConnected();
}