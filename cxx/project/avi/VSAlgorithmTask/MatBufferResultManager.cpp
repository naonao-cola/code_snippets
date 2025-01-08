
/************************************************************************/
//Mat内存管理相关源
//修改日期:17.08.02
/************************************************************************/

//

#include "stdafx.h"

#include "AviInspection.h"
#include "MatBufferResultManager.h"

// 1GB = 1024 * 1024 * 1024
#define GIGA_BYTE			1073741824.0

IMPLEMENT_DYNAMIC(CMatBufferResultManager, CWnd)

CMatBufferResultManager::CMatBufferResultManager()
{
	//////////////////////////////////////////////////////////////////////////
		//17.09.29-重复运行时出现警报
	//////////////////////////////////////////////////////////////////////////

		//获取内存信息
	MEMORYSTATUS memoryInfo;
	memoryInfo.dwLength = sizeof(MEMORYSTATUS);
	GlobalMemoryStatus(&memoryInfo);

	//剩余(可用)物理内存
	double dEmptyMemory = memoryInfo.dwAvailPhys / GIGA_BYTE;

	//物理内存总大小(GB)
	double dTotalMemory = memoryInfo.dwTotalPhys / GIGA_BYTE;

	//使用中的内存
	double dUseMemory = dTotalMemory - dEmptyMemory;

	//大于100GB(设备计算机)
	if (dTotalMemory > 60) // 2024.05.07 for develop
	{
		//如果使用的内存超过50 GB,请确认为重复运行(提醒)
		if (dUseMemory > 160)
		{
			AfxMessageBox(_T("Overlap VSAlgorithmTask !"));

			//程序强制终止语法
			DWORD dwExitCode;
			DWORD dwPID = GetCurrentProcessId();

			HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, 0, dwPID);

			if (NULL != hProcess)
			{
				GetExitCodeProcess(hProcess, &dwExitCode);
				TerminateProcess(hProcess, dwExitCode);
				WaitForSingleObject(hProcess, INFINITE);
				CloseHandle(hProcess);
			}
		}
		//如果不是重复运行,则分配正常内存
		else
		{
			AllocMem_Result(1 * 1024);
		}
	}
	//笔记本电脑和内存不足的PC,分配4M
	else
	{
		// 4 MB
		AllocMem_Result(4);
	}
}

CMatBufferResultManager::~CMatBufferResultManager()
{
	DeleteMem_Result();
}

BEGIN_MESSAGE_MAP(CMatBufferResultManager, CWnd)
END_MESSAGE_MAP()

CMatResultBuf* CMatBufferResultManager::FindFreeBuf_Result()
{
	m_MemManager.Lock();

	int nIndex = -1;

	while (nIndex == -1)
	{
		for (int i = 0; i < MAX_MEM_COUNT; i++)
		{
			if (!m_Data[i].GetUse_Result())
			{
				m_Data[i].SetUse_Result(true);
				nIndex = i;
				break;
			}
		}

		Sleep(50);
	}

	m_MemManager.Unlock();

	return &m_Data[nIndex];
}

void CMatBufferResultManager::ReleaseFreeBuf_Result(CMatResultBuf* data)
{
	for (int i = 0; i < MAX_MEM_COUNT; i++)
	{
		if (data == &m_Data[i])
		{
			m_Data[i].SetUse_Result(false);
		}
	}
}

void CMatBufferResultManager::AllocMem_Result(__int64 nSizeMB)
{
	for (int i = 0; i < MAX_MEM_COUNT; i++)
	{
		// MB
		m_Data[i].AllocMem_Result(nSizeMB * 1024 * 1024);
	}
}

void CMatBufferResultManager::DeleteMem_Result()
{
	for (int i = 0; i < MAX_MEM_COUNT; i++)
	{
		m_Data[i].DeleteMem_Result();
	}
}
