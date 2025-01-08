#pragma once

#include "..\..\VisualStation\CommonHeader\Class\Thread.h"
#include "..\..\VisualStation\CommonHeader\Class\MessageQueue.h"
#include "..\..\VisualStation\CommonHeader\Class\interserverinterface.h"
#include "..\..\CommonHeaders\Structure.h"

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "SharedMemWrapper.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//其他Task No
#define VS_UI_TASK									10
#define VS_SEQUENCE_TASK							21  + (theApp.m_Config.GetPCNum() * 100)
#define VS_ALGORITHM_TASK							50  + (theApp.m_Config.GetPCNum() * 100)
#define VS_LIGHT_TASK								80  + (theApp.m_Config.GetPCNum() * 100)
#define VS_CAMERA_TASK								90  + (theApp.m_Config.GetPCNum() * 100)
#define VS_MAIN_PC_TASK								60  + (theApp.m_Config.GetPCNum() * 100)

#define SEND_UI_LOG									10, 3
#define SEND_UI_FOCUS_VALUE							10, 30
#define SEND_UI_NOTIFY_INIT							50, 20	
#define SEND_UI_RCPVER								10, 4
#define SEND_UI_TACT_TIME_DATA						10, 101

#define SEND_SEQ_CLASSIFY_END						21, 30
#define SEND_SEQ_DUST_RESULT						21, 90

#define SEND_MAINPC_ALARM_OCCURRED					60, 52
#define SEND_MAINPC_ALIGN_RESULT					60, 61

using namespace std;

class WorkManager 
	: protected CThread
{

public:
	WorkManager();
	~WorkManager();

	int					Start();

	int					VS_Send_Log_To_UI(TCHAR* cBuffer, int len);
	int					VS_Send_Notify_Init_To_UI();	
	int					VS_Send_ClassifyEnd_To_Seq(CString strPanelID, CString strDrive, UINT nDefectCount, CString strPanelGrade, CString strJudgeCode1, CString strJudgeCode2 );
	int					VS_Send_Alarm_Occurred_To_MainPC(ENUM_INSPECT_MODE eInspMode, int nAlarmID, int nAlarmType, bool& bIsHeavyAlarm = *new bool());
	int					VS_Send_Align_Result_To_MainPC(ENUM_INSPECT_MODE eInspMode, int nTheta, int nDistX, int nDistY, int nStageNo, int nPCNo);		//发送控制PC Align结果
	int					VS_Send_Dust_Result_To_Seq(ENUM_INSPECT_MODE eInspMode, bool bNeedRetry, bool bIsNormalDust);		//Seq. 结束Task Dust检查并发送结果
	int					VS_Send_RcpVer_To_UI(CString ver);
	int					VS_Send_State_To_UI(CString strPanelID, CString TactName, CString TactState, int nGrabCount = -1);
protected:
	virtual bool		GetThreadRunFlag()		{	return false;	}

	virtual int			CmdEditSend(USHORT FunID, USHORT SeqNo, USHORT UnitNo, ULONG PrmSize, USHORT TaskNo, byte* Buff, 
								USHORT uMsgType = CMD_TYPE_NORES, ULONG lMsgTimeout = CMD_TIMEOUT){return 0;}

	virtual int			ResponseSend(USHORT ReturnVal, CMDMSG* pCmdMsg){return 0;}
		virtual void		EndWorkProcess(BYTE* pMsg){return ;}					//Memory Pool の Memory 返納
	virtual int			m_fnPrintLog(const wchar_t* pszLogData, ...){return 0;}
	virtual int			m_fnPrintLog(const char* pszLogData, ...){return 0;}
	virtual BYTE*		m_fnPeekMessage(){return 0;}

private:
	void				m_fnRunThread();
	int					AnalyzeMsg(CMDMSG* pCmdMsg);
	HANDLE				m_hSequenceResetEvent;
	bool				m_bSeqResetFlag;
	UINT				m_SeqenceCount;
		CSharedMemWrapper*	m_pSharedMemory[MAX_CAMERA_COUNT];						//声明	//Camera Max 16

	CRITICAL_SECTION	m_csSequenceLock_1;
	CRITICAL_SECTION	m_csSequenceLock_2;
	CRITICAL_SECTION	m_csSequenceLock_3;
	CRITICAL_SECTION	m_csSequenceLock_4;
	CRITICAL_SECTION	m_csSequenceLock_5;

	/************************************************************************/

	/************************************************************************/
	// Test
	int Seq_TEST					(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	// Alive
	int	VS_TaskAlive				(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	// Inspection
	int	Seq_InspAliveChk			(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	int	Seq_StartInspection			(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	int	Seq_ManualAlign				(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	// Etc..
	int	Seq_SetParam				(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	int	Seq_FocusValue				(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	int	Seq_WriteCCDIndex			(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);	

	int	Seq_GetAlignPatternNum		(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);
	int	Seq_StartAlign				(byte* pParam, ULONG& nPrmSize, bool bAlwaysRunMode = false, bool bBusyCheck = false, bool bSeqResetPossible = true);	

	/************************************************************************/

	/************************************************************************/	
	void CreateResultDirectory(TCHAR* strPanelID, CString strDrive);

	CString GetNewDriveForAlg();
};