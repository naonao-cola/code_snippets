
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Config.h"
#include "Define.h"
#include "VSAlgorithmTask.h"
#include <direct.h>
#include<Shlwapi.h>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

CConfig::CConfig()
{

}

CConfig::~CConfig()
{
}

TCHAR* CConfig::GetItem(TCHAR *sec, TCHAR *key, TCHAR *def)
{
	GetPrivateProfileString(sec,key,def,buf,_MAX_PATH, m_strCfgPath);
	return buf;
}

CString CConfig::GetStr(TCHAR* Sec, TCHAR* key, CString def)
{
	return CString(GetItem(Sec,key,(TCHAR*)(LPCTSTR)(def)));
}

UINT CConfig::GetUINT(TCHAR *Sec, TCHAR *key, UINT def)
{
	CString ret = GetStr(Sec,key,_T("--"));
	return (ret==_T("--"))? def :  (UINT)_ttoi((TCHAR*)(LPCTSTR)(ret));
}

int CConfig::GetInt(TCHAR* Sec, TCHAR* key, int def)
{
	CString ret = GetStr(Sec,key,_T("--"));
	return (ret==_T("--"))? def :  (int)_ttoi((TCHAR*)(LPCTSTR)(ret));
}

double CConfig::GetDbl(TCHAR* Sec, TCHAR* key, double def)
{
	CString ret = GetStr(Sec,key,_T("--"));
	return (ret==_T("--"))? def :  (double)_ttof((TCHAR*)(LPCTSTR)(ret));
}

void CConfig::Write(TCHAR *sec, TCHAR *key, UINT val)
{
	CString str; str.Format(_T("%d"),val);
	WritePrivateProfileString(sec,key,(TCHAR*)(LPCTSTR)(str), m_strCfgPath);
}

void CConfig::Write(TCHAR *sec, TCHAR *key, int val)
{
	CString str; str.Format(_T("%d"),val);
	WritePrivateProfileString(sec,key,(TCHAR*)(LPCTSTR)(str), m_strCfgPath);
}

void CConfig::Write(TCHAR *sec, TCHAR *key, double val)
{
	CString str; str.Format(_T("%10.3f"),val);
	WritePrivateProfileString(sec,key,(TCHAR*)(LPCTSTR)(str), m_strCfgPath);
}

void CConfig::Write(TCHAR *sec, TCHAR *key, CString val)
{
	int n = WritePrivateProfileString(sec, key, (TCHAR*)(LPCTSTR)(val), m_strCfgPath);
}

bool CConfig::Load()
{
	CString strSection, strKey;
	CString strModel;

#pragma region >> Loading Device.cfg

#pragma endregion

#pragma region >> Loading Initialize.ini
	if(OpenFile(INIT_FILE_PATH))
	{		
		m_nPCNum			= GetInt(_T("Common"), _T("VISION PC NUM"), 1);		
		CString m_strPCNameAll         = GetStr(_T("Common"), _T("VISION PC NAME"), _T("LEFT,RIGHT")).Trim() ;
		int index = m_strPCNameAll.Find(',');
		m_strPCName = index == -1 ? _T("LEFT") : m_strPCNameAll.Left(index);

		CString sSection, sKey;
		sSection.Format(_T("NETWORK_DRIVE_PATH_%d"), m_nPCNum);
		if (GetStr(_T("Common"), _T("SIMULATION Mode"), _T("FALSE")).CompareNoCase(_T("FALSE")) == 0)
		{
			sKey = _T("DRIVE");
			m_nSimualtionMode = FALSE;
		}
		else
		{
			sKey = _T("DRIVE_SIMUL");
			m_nSimualtionMode = TRUE;
		}
		m_strOriginDrive = GetStr(sSection.GetBuffer(0), _T("DRIVE_ORIGIN"), _T(""));
		m_strResultDrive = GetStr(sSection.GetBuffer(0), _T("DRIVE_RESULT"), _T(""));
		m_strNetworkDrive	= GetStr(sSection.GetBuffer(0), sKey.GetBuffer(0), _T("111"));
		m_strOriginPath		= GetStr(sSection.GetBuffer(0), _T("ORIGIN_PATH"), _T(""));
		m_strResultPath		= GetStr(sSection.GetBuffer(0), _T("RESULT_PATH"), _T(""));
		m_strLogFilePath	= GetStr(_T("LogFilePath"), _T("LogFilePath"), _T(""));
		m_strInspPath		= GetStr(sSection.GetBuffer(0), _T("INSPDATA_PATH"), _T(""));

		m_nDriveLimitSize   = GetInt(_T("Diskinformation"), _T("DriveLimitSize"), 80);	
		m_strUseDrive		= GetStr(_T("Diskinformation"), _T("Use Drive"), _T("C"));
		m_strSimulationDrive = GetStr(_T("Diskinformation"), _T("Simulation Drive"), _T("D_Drive")).Left(1) + _T("_Drive\\");
		m_strEQPName		= GetStr(_T("Common"), _T("EQP"), _T("UNKNOWN"));

		m_nEqpType			= GetInt(_T("Common"), _T("TYPE"), 0);

				//出现新的CCD故障
		m_bUseRepeatAlarm[ePIXEL]			= GetInt(_T("Repeat Defect Alarm - CCD"), _T("Use"), 0);
		m_nRepeatCompareOffset[ePIXEL]		= GetInt(_T("Repeat Defect Alarm - CCD"), _T("Compare Offset(Pixel)"), 0);
		m_nRepeatLightAlarmCount[ePIXEL]	= GetInt(_T("Repeat Defect Alarm - CCD"), _T("Light Alarm"), 5);
		m_nRepeatHeavyAlarmCount[ePIXEL]	= GetInt(_T("Repeat Defect Alarm - CCD"), _T("Heavy Alarm"), 10);

				//发生工艺不良
		m_bUseRepeatAlarm[eWORK]			= GetInt(_T("Repeat Defect Alarm - Work"), _T("Use"), 0);
		m_nRepeatCompareOffset[eWORK]		= GetInt(_T("Repeat Defect Alarm - Work"), _T("Compare Offset(um)"), 0);
		m_nRepeatLightAlarmCount[eWORK]		= GetInt(_T("Repeat Defect Alarm - Work"), _T("Light Alarm"), 5);
		m_nRepeatHeavyAlarmCount[eWORK]		= GetInt(_T("Repeat Defect Alarm - Work"), _T("Heavy Alarm"), 10);

				//APP Repeat Defect设备出现故障
		m_bUseRepeatAlarm[eMACHINE] = GetInt(_T("Repeat Defect Alarm - Machine"), _T("Use"), 0);
		m_nRepeatCompareOffset[eMACHINE] = GetInt(_T("Repeat Defect Alarm - Machine"), _T("Compare Offset"), 0);
		m_nRepeatLightAlarmCount[eMACHINE] = GetInt(_T("Repeat Defect Alarm - Machine"), _T("Light Alarm"), 5);
		m_nRepeatHeavyAlarmCount[eMACHINE] = GetInt(_T("Repeat Defect Alarm - Machine"), _T("Heavy Alarm"), 10);
	}
#pragma endregion

#pragma region >> Loading Crux_Algorithm.ini
	if(OpenFile(VS_ALGORITHM_TASK_INI_FILE))
	{				
		m_nLogLevel			= GetInt(_T("General"),  _T("LogLevel"), 1);
	}
#pragma endregion

#pragma region >> Loading Device.cfg
	if(OpenFile(DEVICE_FILE_PATH))
	{
		m_nCamCount = 0;
		for (int nGrabberCnt=0; nGrabberCnt<MAX_FRAME_GRABBER_COUNT; nGrabberCnt++)
		{
			strSection.Format(_T("Frame Grabber_%d"), nGrabberCnt);
			for (int nCameraCnt=0; nCameraCnt<MAX_CAMERA_COUNT; nCameraCnt++)
			{
				strKey.Format(_T("Insp Camera_%d"), nCameraCnt);
				if (GetStr(strSection.GetBuffer(0), strKey.GetBuffer(0), _T("F"))==_T("T"))
					m_nCamCount++;
			}
		}
	}
#pragma endregion
	return true;
}

void CConfig::SetPath(CString str)
{
	m_strCfgPath = str;
}

bool CConfig::OpenFile(CString sPath)
{
	SetPath(sPath);
	//CFile conf;
	if (!PathFileExists(sPath))
	//if(!conf.Open(sPath.GetBuffer(0), CFile::modeRead)) 
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Not exist file (%s)"), sPath);
		return false ;
	}
	//conf.Close();
	return true;
}

// 181001

CString CConfig::GETDRV() 
{ 
	TCHAR buff[MAX_PATH];
	memset(buff, 0, MAX_PATH);
	::GetModuleFileName(NULL, buff, sizeof(buff));
	CString strFolder = buff;
	CString strRet = strFolder.Left(1);
		strRet.MakeUpper();		//180919 YSS	//181001 Lower->更改为Upper
	return strRet;

	//int nDrive = 0;   nDrive = _getdrive();   
	//CString str;	
	//str.Format(_T("%c"), char(nDrive) + 96);
	//return str;   
};


/**
 * 获取自定义盘符 hjf
 * 
 * \return 
 */
CString CConfig::GETCmdDRVPath()
{
	return m_strINIDrive;
};

CString CConfig::GETCmdDRV()
{
	return m_strINIDrive.Left(1);
};


/**
 * 获取cmd命令 解析出Path并设置到 m_strINIDrive hjf
 * 
 * \param path
 */
void CConfig::SetINIPath(const CString& path)
{
	CString newPath = path;

	CFileFind fileFind;
	if (fileFind.FindFile(newPath))
	{
		int delimiterIndex = path.Find(_T("\\Config"));
		if (delimiterIndex != -1)
		{
			newPath = newPath.Left(delimiterIndex);
		}
	}
	
	m_strINIDrive = newPath;
	return;
}

CString  CConfig::GetOriginDriveForAlg()
{
	CString originDir;
	CString strMidName = _T(":\\ALG_");
	originDir = m_strOriginDrive.Left(1) + strMidName;
	originDir +=GetEqpType() == EQP_AVI ? _T("AVI") : (theApp.m_Config.GetEqpType() == EQP_SVI ? _T("SVI") : _T("APP"));

	CString strDate;
	SYSTEMTIME time;
	::GetLocalTime(&time);
	strDate.Format(_T("%04d%02d%02d"), time.wYear, time.wMonth, time.wDay);
	originDir.Format(_T("%s_%s%s%s%s"), originDir, GetPCName(), _T("\\"), strDate, _T("\\"));
	return originDir;
}

CString  CConfig::GetResultDriveForAlg()
{
	CString resultDir;
	CString strMidName = _T(":\\ALG_");
	resultDir = m_strResultDrive.Left(1) + strMidName;
	resultDir += GetEqpType() == EQP_AVI ? _T("AVI") : (theApp.m_Config.GetEqpType() == EQP_SVI ? _T("SVI") : _T("APP"));

	CString strDate;
	SYSTEMTIME time;
	::GetLocalTime(&time);
	strDate.Format(_T("%04d%02d%02d"), time.wYear, time.wMonth, time.wDay);
	resultDir.Format(_T("%s_%s%s%s%s"), resultDir, GetPCName(), _T("\\"), strDate, _T("\\"));
	return resultDir;
}

CString  CConfig::GetResultDrive()
{
	return m_strResultDrive.Left(1);
}
