
//
//////////////////////////////////////////////////////////////////////

#ifndef AFX_CONFIG_H__89A008AB_6C8F_4C25_9509_8F34D97F8EEE__INCLUDED_
#define	AFX_CONFIG_H__89A008AB_6C8F_4C25_9509_8F34D97F8EEE__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "..\..\commonheaders\Structure.h"
#include "InspResultInfo.h"

//不要使用分支语句来区分字符串
enum EQP_TYPE
{
	EQP_AVI = 0,
	EQP_SVI,
	EQP_APP,
	MAX_EQP_TYPE
};

const CString strEqpType[MAX_EQP_TYPE] = {_T("AVI"), _T("SVI"), _T("APP")};

class CConfig  
{
public:
	CConfig();
	virtual ~CConfig();

	bool			Load();
//禁用文件格式-利用IPC Message
// 	bool			LoadModel(CString strModel = _T(""));
	int				GetPCNum()												{	return m_nPCNum														;};
	CString        GetPCName()												 { return m_strPCName; };
		//17.06.23如果是Manual操作,请将结果保存路径固定为D:\->修改为遵循设置值

	CString			GetOriginPath()											{	return m_strOriginPath 												;};
	CString			GetResultPath()											{	return m_strResultPath												;};
	CString			GetOriginDrive() { return m_strOriginDrive; };

	CString			GetLogFilePath()										{	return m_strLogFilePath												;};

	CString			GetInspPath()											{	return m_strInspPath												;};
	CString			GetEqpTypeName()										{	return strEqpType[m_nEqpType]										;};
	CString			GetEqpName()										    {	return m_strEQPName													;};
	EQP_TYPE		GetEqpType()											{	return (EQP_TYPE)m_nEqpType											;};
	int				GetLogLevel()											{	return m_nLogLevel													;};	
	int				GetUseCamCount()										{	return m_nCamCount													;};
	CString			GetNetworkDrivePath()									{	return m_strNetworkDrive											;};
	BOOL			GetSimualtionMode()										{	return m_nSimualtionMode											;};
	CString			GetUseDrive()											{	return m_strUseDrive												;};		
	CString			GetSimulationDrive()									{	return m_strSimulationDrive											;};
	CString			GetCurrentDrive()										{	return m_strCurrentDrive											;}; 
		CString			GetINIDrive()											{	return m_strINIDrive												;}; //ini中设置的驱动器Last Used Drive
	int				GetDriveLimitSize()										{	return m_nDriveLimitSize											;};
		//CCD[0]/工程[1]不良发生判断标准
	BOOL*			GetUseCCDAlarm()										{	return m_bUseRepeatAlarm											;};
	int*			GetCCDOffset()											{	return m_nRepeatCompareOffset										;};
	int*			GetCCDLightAlarmCount()									{	return m_nRepeatLightAlarmCount										;};
	int*			GetCCDHeavyAlarmCount()									{	return m_nRepeatHeavyAlarmCount										;};

	// Set Func
	void			SetCurrentDrive(CString strDrive)						{	m_strCurrentDrive = strDrive										;};
		void			SetINIDrive(CString strDrive)							{	m_strINIDrive = strDrive											;}; //ini中设置的驱动器Last Used Drive
	void			SetDriveLimitSize(int nSize)							{	m_nDriveLimitSize = nSize											;};

	void			Write(TCHAR* sec, TCHAR* key, UINT val);
	void			Write(TCHAR* sec, TCHAR* key, int val);
	void			Write(TCHAR* sec, TCHAR* key, double val);
	void			Write(TCHAR* sec, TCHAR* key, CString val);

	void			SetPath(CString str);	
	bool			OpenFile(CString sPath);

	CString			GETDRV();		//获取当前exe运行盘符 hjf
	CString			GETCmdDRVPath();//获取工站目录 hjf
	CString			GETCmdDRV();	//获取工站盘符 hjf
	void SetINIPath(const CString& path); //设置工作目录 hjf
	CString  GetOriginDriveForAlg();
	CString  GetResultDriveForAlg();
	CString  GetResultDrive();

private:
	TCHAR*			GetItem(TCHAR* sec, TCHAR* key, TCHAR* def);
	UINT			GetUINT(TCHAR* Sec, TCHAR* key, UINT def);
	int				GetInt(TCHAR* Sec, TCHAR* key, int def);
	double			GetDbl(TCHAR* Sec, TCHAR* key, double def);
	CString			GetStr(TCHAR* Sec, TCHAR* key, CString def);
	CString			GetPath()	{ return m_strCfgPath; }


	CString			m_strINIDrive;
	CString			m_strCfgPath;
	TCHAR			buf[128];	

	// Device Info
	int				m_nCamCount;
	int				m_nLightCount;

	// Initialize Info
	CString			m_strModelPath;
	int				m_nPCNum;
	int				m_stationNum;//按工位区分 用于划分共享内存和workstation字段  hjf
	CString        m_strPCName;

	// Model Info
	ST_MODEL_INFO	m_stModelInfo;

	CString			m_strNetworkDrive;
	CString			m_strOriginPath;
	CString			m_strResultPath;
	CString			m_strLogFilePath;		// Log File Path
	CString			m_strInspPath;
	CString			m_strCurrentDrive;
	int				m_nEqpType;
	int				m_nLogLevel;
	BOOL			m_nSimualtionMode;
	CString			m_strUseDrive;
	CString			m_strSimulationDrive;
	int				m_nDriveLimitSize;
	CString			m_strEQPName;
	CString			m_strOriginDrive;
	CString        m_strResultDrive;

		//相同坐标重复不良(CCD/工艺不良)
	BOOL			m_bUseRepeatAlarm[eCOORD_KIND];
	int				m_nRepeatCompareOffset[eCOORD_KIND];
	int				m_nRepeatLightAlarmCount[eCOORD_KIND];
	int				m_nRepeatHeavyAlarmCount[eCOORD_KIND];
};

#endif // !defined(AFX_CONFIG_H__89A008AB_6C8F_4C25_9509_8F34D97F8EEE__INCLUDED_)
