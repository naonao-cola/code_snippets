
/************************************************************************/
//算法日志相关标头
//修改日期:18.03.28
/************************************************************************/

#pragma once

#include <iostream>
#include <fstream>

#include "time.h"
#include <string>

#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>

#include <direct.h>

#pragma comment(lib, "pdh.lib")

using namespace std;

//日志存储路径
#define LOG_PATH_ALG		GETDRV() + L":\\IMTC\\DATA\\LOG\\Algorithm"

// 1GB = 1024 * 1024 * 1024
#define GIGA_BYTE			1073741824.0

//是否使用Tact Time Log
#define USE_LOG_COMMON_LIB		TRUE
#define USE_LOG_COMMON_ALIGN	TRUE
#define USE_LOG_AVI_POINT		FALSE
#define USE_LOG_AVI_LINE		FALSE
#define USE_LOG_AVI_MURA		FALSE
#define USE_LOG_AVI_MURA_NORMAL	 FALSE
#define USE_LOG_SVI_MURA		FALSE
#define USE_LOG_AVI_MURA3		FALSE
#define USE_LOG_AVI_MURA4		FALSE
#define USE_LOG_AVI_MURA_CHOLE	 FALSE
#define	USE_LOG_AVI_MURA_SCRATCH FALSE
#define USE_LOG_AVI_MEMORY		TRUE//使用时出现Alg Tact日志错误,仅在检查Memory Test时使用21.05.27

#define USE_LOG_APP				FALSE

//日志类型
enum ENUM_ALG_TYPE
{
	E_ALG_TYPE_COMMON_LIB = 0,
	E_ALG_TYPE_COMMON_ALIGN,
	E_ALG_TYPE_AVI_POINT,
	E_ALG_TYPE_AVI_LINE,
	E_ALG_TYPE_AVI_MURA,
	E_ALG_TYPE_SVI_MURA,

	E_ALG_TYPE_AVI_MURA_NORMAL,
	E_ALG_TYPE_AVI_MURA_3,	//21.01.29崔光日
	E_ALG_TYPE_AVI_MURA_4,   //
	E_ALG_TYPE_AVI_MURA_CHOLE,   //
	E_ALG_TYPE_AVI_MURA_SCRATCH,   //
	E_ALG_TYPE_AVI_MEMORY,
	E_ALG_TYPE_APP,

	E_ALG_TYPE_CPU,
	E_ALG_TYPE_TOTAL
};

//////////////////////////////////////////////////////////////////////////

class InspectLibLog
{
public:
	InspectLibLog(void)
	{
		for (int i = 0; i < E_ALG_TYPE_TOTAL; i++)
			InitializeCriticalSection(&safeFile[i]);

		m_bCPU = initCPU();
	}

	virtual ~InspectLibLog(void)
	{
		for (int i = 0; i < E_ALG_TYPE_TOTAL; i++)
			DeleteCriticalSection(&safeFile[i]);

		destroyCPU();
	}

	static const bool Use_AVI_Memory_Log = USE_LOG_AVI_MEMORY; //choi 21.05.27

protected:
	//	Algorithm Log专用critical section
	CRITICAL_SECTION	safeFile[E_ALG_TYPE_TOTAL];

	// CPU
	PDH_HCOUNTER		m_hCounter;
	PDH_HQUERY			m_hQuery;
	bool				m_bCPU;

public:

	//获取日志路径
	void getLogPath(int nAlgType, wchar_t* strPath)
	{
		SYSTEMTIME time;
		::GetLocalTime(&time);
		CString strAlgLogPath = LOG_PATH_ALG;

		switch (nAlgType)
		{
		case E_ALG_TYPE_COMMON_LIB:
			if (USE_LOG_COMMON_LIB)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibInspect_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_COMMON_ALIGN:
			if (USE_LOG_COMMON_ALIGN)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAlign_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_AVI_POINT:
			if (USE_LOG_AVI_POINT)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIPoint_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_AVI_LINE:
			if (USE_LOG_AVI_LINE)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVILine_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_AVI_MURA:
			if (USE_LOG_AVI_MURA)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMura_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;
		case E_ALG_TYPE_AVI_MURA_NORMAL:
			if (USE_LOG_AVI_MURA_NORMAL)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMuraNormal_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;
		case E_ALG_TYPE_SVI_MURA:
			if (USE_LOG_SVI_MURA)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibSVIMura_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

			//////////////////////////////////////////////////////////////////////////////////////
		case E_ALG_TYPE_AVI_MURA_3:
			if (USE_LOG_AVI_MURA3)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMura3_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;
		case E_ALG_TYPE_AVI_MURA_4:
			if (USE_LOG_AVI_MURA4)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMura4_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_AVI_MURA_CHOLE:
			if (USE_LOG_AVI_MURA_CHOLE)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMuraChole_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_AVI_MURA_SCRATCH:
			if (USE_LOG_AVI_MURA_CHOLE)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMuraScratch_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		case E_ALG_TYPE_AVI_MEMORY:
			if (USE_LOG_AVI_MEMORY)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAVIMemory_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			//////////////////////////////////////////////////////////////////////////

		case E_ALG_TYPE_APP:
			if (USE_LOG_APP)
				swprintf_s(strPath, MAX_PATH, L"%s\\LibAPP_%04d%02d%02d.Log", (LPCWSTR)strAlgLogPath, time.wYear, time.wMonth, time.wDay);
			break;

		default:
			break;
		}
	};

	void get_SubType(int nAlgType, string& sub_AlgType)
	{
		SYSTEMTIME time;
		::GetLocalTime(&time);
		CString strAlgLogPath = LOG_PATH_ALG;

		switch (nAlgType)
		{
		case E_ALG_TYPE_COMMON_LIB:
			if (USE_LOG_COMMON_LIB)
				sub_AlgType = "COMMON_LIB";

			break;

		case E_ALG_TYPE_COMMON_ALIGN:
			if (USE_LOG_COMMON_ALIGN)
				sub_AlgType = "LibAlign";

			break;

		case E_ALG_TYPE_AVI_POINT:
			if (USE_LOG_AVI_POINT)
				sub_AlgType = "LibAVIPoint";

			break;

		case E_ALG_TYPE_AVI_LINE:
			if (USE_LOG_AVI_LINE)
				sub_AlgType = "LibAVILine";

			break;

		case E_ALG_TYPE_AVI_MURA:
			if (USE_LOG_AVI_MURA)
				sub_AlgType = "LibAVIMura";

			break;
		case E_ALG_TYPE_AVI_MURA_NORMAL:
			if (USE_LOG_AVI_MURA_NORMAL)
				sub_AlgType = "LibAVIMuraNormal";

			break;
		case E_ALG_TYPE_SVI_MURA:
			if (USE_LOG_SVI_MURA)
				sub_AlgType = "LibSVIMura";

			break;

			//////////////////////////////////////////////////////////////////////////////////////
		case E_ALG_TYPE_AVI_MURA_3:
			if (USE_LOG_AVI_MURA3)
				sub_AlgType = "LibAVIMura3";

			break;
		case E_ALG_TYPE_AVI_MURA_4:
			if (USE_LOG_AVI_MURA4)
				sub_AlgType = "LibAVIMura4";

			break;

		case E_ALG_TYPE_AVI_MURA_CHOLE:
			if (USE_LOG_AVI_MURA_CHOLE)
				sub_AlgType = "LibAVIMuraChole";

			break;

		case E_ALG_TYPE_AVI_MURA_SCRATCH:
			if (USE_LOG_AVI_MURA_SCRATCH)
				sub_AlgType = "LibAVIMuraScratch";

			break;

		case E_ALG_TYPE_AVI_MEMORY:
			if (USE_LOG_AVI_MEMORY)
				sub_AlgType = "LibAVIMemory";

			break;
			//////////////////////////////////////////////////////////////////////////

		case E_ALG_TYPE_APP:
			if (USE_LOG_APP)
				sub_AlgType = "LibAPP";

			break;

		default:
			break;
		}
	};

	//初始化CPU
	bool initCPU()
	{
		PDH_STATUS status = PdhOpenQuery(NULL, NULL, &m_hQuery);
		if (status != ERROR_SUCCESS)	return false;

		status = PdhAddCounter(m_hQuery, _T("\\Processor(_TOTAL)\\% Processor Time"), NULL, &m_hCounter);
		if (status != ERROR_SUCCESS)	return false;

		status = PdhCollectQueryData(m_hQuery);
		if (status != ERROR_SUCCESS)	return false;

		return true;
	}

	//禁用CPU
	void destroyCPU()
	{
		if (m_hQuery)
			PdhCloseQuery(m_hQuery);

		m_hQuery = NULL;
	}

	//获取CPU内存
	double getUseMemoryCPU()
	{
		//如果未初始化
		if (!m_bCPU)	return 0.0;

		//EnterCriticalSection(&safeFile[E_ALG_TYPE_CPU]);

		PDH_STATUS status = PdhCollectQueryData(m_hQuery);
		if (status != ERROR_SUCCESS)
		{
			//LeaveCriticalSection(&safeFile[E_ALG_TYPE_CPU]);
			return 0.0;
		}

		PDH_FMT_COUNTERVALUE    value;
		status = PdhGetFormattedCounterValue(m_hCounter, PDH_FMT_DOUBLE, 0, &value);
		if (status != ERROR_SUCCESS)
		{
			//LeaveCriticalSection(&safeFile[E_ALG_TYPE_CPU]);
			return 0.0;
		}

		//LeaveCriticalSection(&safeFile[E_ALG_TYPE_CPU]);

		return (double)value.doubleValue;
	}

	//导入RAM内存
	double getUseMemoryRAM()
	{
		//获取内存信息
		MEMORYSTATUS memoryInfo;
		memoryInfo.dwLength = sizeof(MEMORYSTATUS);
		GlobalMemoryStatus(&memoryInfo);

		//剩余(可用)物理内存
		double dEmptyMemory = memoryInfo.dwAvailPhys / GIGA_BYTE;

		//物理内存总大小(GB)
		double dTotalMemory = memoryInfo.dwTotalPhys / GIGA_BYTE;

		//使用中的内存
		return dTotalMemory - dEmptyMemory;
	};

	//仅写日志
	clock_t writeInspectLog(int nAlgType, char* strFunc, wchar_t* strTxt, wchar_t* strPat = NULL)
	{
		//获取日志路径
		wchar_t strPath[MAX_PATH] = { 0, };
		getLogPath(nAlgType, strPath);

		//如果没有日志路径,请退出
		if (strPath[0] == NULL)		return 0;

		clock_t tBeforeTime = clock();
		int		nClock = (int)tBeforeTime;

		SYSTEMTIME lst;
		GetLocalTime(&lst);
		EnterCriticalSection(&safeFile[nAlgType]);

		// File Open
		std::ofstream outFile(strPath, std::ofstream::ios_base::app);

		// Function
		string swsFunc(strFunc);

		CString strTime;
		if (strPat != NULL)
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]\t%s"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, nClock, getUseMemoryRAM(), strPat);
		else
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, nClock, getUseMemoryRAM());

		wstring wsTimeTxt(strTime);
		string	swsTimeTxt(wsTimeTxt.begin(), wsTimeTxt.end());

		// Txt
		if (strTxt == NULL)
		{
			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\n";
		}
		else
		{
			wstring wsTxt(strTxt);
			string swsTxt(wsTxt.begin(), wsTxt.end());

			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << swsTxt << "\n";
		}

		// File Close
		outFile.close();

		LeaveCriticalSection(&safeFile[nAlgType]);

		return tBeforeTime;
	};

	//时间测量日志
	clock_t writeInspectLogTime(int nAlgType, clock_t tBeforeTime, char* strFunc, wchar_t* strTxt, wchar_t* strPat = NULL)
	{
		//获取日志路径
		wchar_t strPath[MAX_PATH] = { 0, };
		getLogPath(nAlgType, strPath);

		//如果没有日志路径,请退出
		if (strPath[0] == NULL)		return 0;

		int		nClock = (int)tBeforeTime;

		SYSTEMTIME lst;
		GetLocalTime(&lst);
		EnterCriticalSection(&safeFile[nAlgType]);

		// File Open
		std::ofstream outFile(strPath, std::ofstream::ios_base::app);

		// Function
		string swsFunc(strFunc);

		// Time
		clock_t tAfterTime = clock();
		double dAfterTime = (double)(tAfterTime - tBeforeTime) / CLOCKS_PER_SEC;

		CString strTime;
		if (strPat != NULL)
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]\t%s"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tBeforeTime, getUseMemoryRAM(), strPat);
		else
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tBeforeTime, getUseMemoryRAM());

		wstring wsTimeTxt(strTime);
		string	swsTimeTxt(wsTimeTxt.begin(), wsTimeTxt.end());

		if (strTxt == NULL)
		{
			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << dAfterTime << "\n";
		}
		else
		{
			wstring wsTxt(strTxt);
			string swsTxt(wsTxt.begin(), wsTxt.end());

			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << swsTxt << "\t" << dAfterTime << "\n";
		}

		// File Close
		outFile.close();

		LeaveCriticalSection(&safeFile[nAlgType]);

		return tAfterTime;
	};

	//时间测量日志
	clock_t writeInspectLogTime(int nAlgType, clock_t tInitTime, clock_t tBeforeTime, char* strFunc, wchar_t* strTxt, wchar_t* strPat = NULL)
	{
		//获取日志路径
		wchar_t strPath[MAX_PATH] = { 0, };
		getLogPath(nAlgType, strPath);

		//如果没有日志路径,请退出
		if (strPath[0] == NULL)		return 0;

		int		nClock = (int)tBeforeTime;

		SYSTEMTIME lst;
		GetLocalTime(&lst);
		EnterCriticalSection(&safeFile[nAlgType]);

		// File Open
		std::ofstream outFile(strPath, std::ofstream::ios_base::app);

		// Function
		string swsFunc(strFunc);

		// Time
		clock_t tAfterTime = clock();
		double dAfterTime = (double)(tAfterTime - tBeforeTime) / CLOCKS_PER_SEC;
		double dTotalTime = (double)(tAfterTime - tInitTime) / CLOCKS_PER_SEC;

		CString strTime;
		if (strPat != NULL)
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]\t%s"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tInitTime, getUseMemoryRAM(), strPat);
		else
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tInitTime, getUseMemoryRAM());

		wstring wsTimeTxt(strTime);
		string	swsTimeTxt(wsTimeTxt.begin(), wsTimeTxt.end());

		if (strTxt == NULL)
		{
			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << dAfterTime << "\t(" << dTotalTime << ")" << "\n";
		}
		else
		{
			wstring wsTxt(strTxt);
			string swsTxt(wsTxt.begin(), wsTxt.end());

			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << swsTxt << "\t" << dAfterTime << "\t(" << dTotalTime << ")" << "\t" << "\n";
		}

		// File Close
		outFile.close();

		LeaveCriticalSection(&safeFile[nAlgType]);

		return tAfterTime;
	};
	//////////////////////////////////////////////////////////////////////////choi
	clock_t writeInspectLogTime(int nAlgType, clock_t tInitTime, clock_t tBeforeTime, char* strFunc, wchar_t* strTxt, __int64 nMemory_Use_Value = 0, wchar_t* strPat = NULL)
	{
		//获取日志路径
		wchar_t strPath[MAX_PATH] = { 0, };
		wchar_t AlgType_Sub[MAX_PATH] = { 0, };

		getLogPath(nAlgType, strPath);

		//如果没有日志路径,请退出
		if (strPath[0] == NULL)		return 0;

		int		nClock = (int)tBeforeTime;

		SYSTEMTIME lst;
		GetLocalTime(&lst);
		EnterCriticalSection(&safeFile[nAlgType]);

		// File Open
		std::ofstream outFile(strPath, std::ofstream::ios_base::app);

		// Function
		string swsFunc(strFunc);

		// Time
		clock_t tAfterTime = clock();
		double dAfterTime = (double)(tAfterTime - tBeforeTime) / CLOCKS_PER_SEC;
		double dTotalTime = (double)(tAfterTime - tInitTime) / CLOCKS_PER_SEC;

		CString strTime;
		if (strPat != NULL)
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]\t%s"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tInitTime, getUseMemoryRAM(), strPat);
		else
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tInitTime, getUseMemoryRAM());

		wstring wsTimeTxt(strTime);
		string	swsTimeTxt(wsTimeTxt.begin(), wsTimeTxt.end());

		if (strTxt == NULL)
		{
			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << nMemory_Use_Value << "\n";
		}
		else
		{
			wstring wsTxt(strTxt);
			string swsTxt(wsTxt.begin(), wsTxt.end());

			// write
			outFile << swsTimeTxt << "\t" << swsFunc << "\t" << swsTxt << "\t" << nMemory_Use_Value << "\n";
		}

		// File Close
		outFile.close();

		LeaveCriticalSection(&safeFile[nAlgType]);

		return tAfterTime;
	};

	clock_t writeInspectLogTime(int nAlgType, clock_t tInitTime, clock_t tBeforeTime, char* strFunc, wchar_t* strTxt, __int64 nMemory_Use_Value = 0, int nSub_AlgType = NULL, wchar_t* strPat = NULL)
	{
		//获取日志路径
		wchar_t strPath[MAX_PATH] = { 0, };
		string AlgType_Sub = "";

		getLogPath(nAlgType, strPath);

		if (nSub_AlgType != NULL) {
			get_SubType(nSub_AlgType, AlgType_Sub);
		}
		//如果没有日志路径,请退出
		if (strPath[0] == NULL)		return 0;

		int		nClock = (int)tBeforeTime;

		SYSTEMTIME lst;
		GetLocalTime(&lst);
		EnterCriticalSection(&safeFile[nAlgType]);

		// File Open
		std::ofstream outFile(strPath, std::ofstream::ios_base::app);

		// Function
		string swsFunc(strFunc);

		// Time
		clock_t tAfterTime = clock();
		double dAfterTime = (double)(tAfterTime - tBeforeTime) / CLOCKS_PER_SEC;
		double dTotalTime = (double)(tAfterTime - tInitTime) / CLOCKS_PER_SEC;

		CString strTime;
		if (strPat != NULL)
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]\t%s"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tInitTime, getUseMemoryRAM(), strPat);
		else
			strTime.Format(_T("[%02d:%02d:%02d:%03d]\t%09d\t[%.4fGB]"), lst.wHour, lst.wMinute, lst.wSecond, lst.wMilliseconds, tInitTime, getUseMemoryRAM());

		wstring wsTimeTxt(strTime);
		string	swsTimeTxt(wsTimeTxt.begin(), wsTimeTxt.end());

		if (nSub_AlgType != NULL) {
			if (strTxt == NULL)
			{
				// write
				outFile << swsTimeTxt << "\t" << AlgType_Sub << "::" << swsFunc << "\t" << nMemory_Use_Value << "\n";
			}
			else
			{
				wstring wsTxt(strTxt);
				string swsTxt(wsTxt.begin(), wsTxt.end());

				// write
				outFile << swsTimeTxt << "\t" << AlgType_Sub << "::" << swsFunc << "\t" << swsTxt << "\t" << nMemory_Use_Value << "\n";
			}
		}
		else {
			if (strTxt == NULL)
			{
				// write
				outFile << swsTimeTxt << "\t" << swsFunc << "\t" << nMemory_Use_Value << "\n";
			}
			else
			{
				wstring wsTxt(strTxt);
				string swsTxt(wsTxt.begin(), wsTxt.end());

				// write
				outFile << swsTimeTxt << "\t" << swsFunc << "\t" << swsTxt << "\t" << nMemory_Use_Value << "\n";
			}
		}

		// File Close
		outFile.close();

		LeaveCriticalSection(&safeFile[nAlgType]);

		return tAfterTime;
	};

	//////////////////////////////////////////////////////////////////////////
		//添加对当前解决方案运行的驱动器的判断
	CString GETDRV()
	{
		TCHAR buff[MAX_PATH];
		memset(buff, 0, MAX_PATH);
		::GetModuleFileName(NULL, buff, sizeof(buff));
		CString strFolder = buff;
		CString strRet = strFolder.Left(1);
		strRet.MakeUpper();
		return strRet;
	};
};