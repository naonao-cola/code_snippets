#pragma once

#include "..\..\VisualStation\CommonHeader\Class\interserverinterface.h"
#include "StatusManager.h"
#include "WorkManager.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

class TaskInterfacer :
	private StatusManager, public WorkManager, protected CInterServerInterface
{
public:
	TaskInterfacer(int nTaskNo, int nServerPort, TSTRING szServerIP, 
				CString szLogFilePath,			///<日志文件路径
				TSTRING szLogFileName,			///<日志文件名
				bool bAutoExitFlag = true,		///<如果注册了Flag,Windows handle,则在服务器关闭时是否自动关闭应用程序,则不能自动关闭。
				int uPeriodDay = 30,			///<日志保留期限
				int uWriteTerm = 0,				///<每隔几个小时生成一次日志文件...时间,0=一个文件生成一天的日志。
		USHORT uLogBlockLevel = LOG_LEVEL_INFO,
				HWND hPwnd = NULL				////<Dlg处理程序
		);
	~TaskInterfacer();

	int Start();
	bool GetVSState();
	void SetParentWnd(HWND hParentWnd)			{	m_hParentWnd = hParentWnd;	};

	int		CmdEditSend_TEST(USHORT FunID, USHORT SeqNo, USHORT UnitNo, ULONG PrmSize, USHORT TaskNo, byte* Buff, USHORT uMsgType = CMD_TYPE_NORES, ULONG lMsgTimeout = CMD_TIMEOUT)
			{		return CmdEditSend(FunID, SeqNo, UnitNo, PrmSize, TaskNo, Buff, uMsgType, lMsgTimeout);		};

	// Interface
	int		SendNotifyInitial()					{	return ::WorkManager::VS_Send_Notify_Init_To_UI()	;};

private:
		bool	GetThreadRunFlag();			//Threadを実行するか確認Flag を Get
		//int		VSMessageReceiver();		// Visua Station MessageをGetしてIndexerWorkManagerのMessageQueueにPUT

	int		ResponseSend(USHORT ReturnVal, CMDMSG* pCmdMsg);
	int		CmdEditSend(USHORT FunID, USHORT SeqNo, USHORT UnitNo, ULONG PrmSize, USHORT TaskNo, byte* Buff, USHORT uMsgType = CMD_TYPE_NORES, ULONG lMsgTimeout = CMD_TIMEOUT);
	BYTE*	m_fnPeekMessage();
	void	EndWorkProcess(BYTE* pMsg);
	int		m_fnPrintLog(
				const wchar_t* pszLogData,		///<要实际记录的日志内容
				...								///<可变参数
		);

	int		m_fnPrintLog( 
				const char* pszLogData,			///<要实际记录的日志内容
				...								///<可变参数
		);

	int		m_fnSendUILog( 
				const wchar_t* pszLogData,		///<要发送到UI的日志内容
				...								///<可变参数
		);

private:

		bool			m_bThreadRun;						//Threadを実行するか確認Flag
	int				m_nTaskNo;
	int				m_nServerPort;
	TCHAR			m_szServerIP[100];
	HWND			m_hParentWnd;
};