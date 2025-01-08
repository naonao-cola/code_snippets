#pragma once

#include "..\..\VisualStation\CommonHeader\Class\Thread.h"
#include "..\..\VisualStation\CommonHeader\Class\MessageQueue.h"
#include <tlhelp32.h>
#include "..\..\VisualStation\CommonHeader\Class\LogWriter.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

class StatusManager 
	: protected CThread
{
#define	IN_UPDATE_THREAD			0
#define	OUT_UPDATE_THREAD			1
#define	USER_DEFINE_UPDATE_THREAD	2

	struct StatusManagerQueueParam 
	{
		int data;
	};

public:
	StatusManager();
	~StatusManager();

	int Start();

protected:
	virtual bool GetThreadRunFlag()
	{
		return false;
	}

	virtual int	CmdEditSend(USHORT FunID, USHORT SeqNo, USHORT UnitNo, ULONG PrmSize, USHORT TaskNo, byte* Buff, USHORT uMsgType = CMD_TYPE_NORES, ULONG lMsgTimeout = CMD_TIMEOUT)
	{
		return 0;
	}

	virtual int m_fnPrintLog(const wchar_t* pszLogData, ...){return 0;}
	virtual int m_fnPrintLog(const char* pszLogData, ...){return 0;}

private:

	void	m_fnRunThread();					
	void	InStatusManagerThread();			
	void	OutStatusManagerThread();				
	void	UserDefineStatusManagerThread();	

	CMessageQueue<StatusManagerQueueParam*>	m_StatusMessageQueue;

};
