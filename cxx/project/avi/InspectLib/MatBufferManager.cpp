
/************************************************************************/
//Mat内存管理相关源
//修改日期:17.08.02
/************************************************************************/

//

#include "stdafx.h"
#include "InspectLib.h"
#include "MatBufferManager.h"

// 1GB = 1024 * 1024 * 1024
#define GIGA_BYTE			1073741824.0

IMPLEMENT_DYNAMIC(CMatBufferManager, CWnd)

CMatBufferManager::CMatBufferManager()
{
	m_Data = NULL;
	m_Data_Low = NULL;
	m_Data_High = NULL;
}

CMatBufferManager::~CMatBufferManager()
{
	if (m_Data != NULL) {
		DeleteMem();
		delete(m_Data);
	}
	
	if (m_Data_Low != NULL) {
		DeleteMem_Low();
		delete(m_Data_Low);
	}

	if (m_Data_High != NULL) {
		DeleteMem_High();
		delete(m_Data_High);
	}
}

BEGIN_MESSAGE_MAP(CMatBufferManager, CWnd)
END_MESSAGE_MAP()

void CMatBufferManager::Initialize(CString initFilePath)
{
	//////////////////////////////////////////////////////////////////////////
	//17.09.29-重复运行时出现警报
	//////////////////////////////////////////////////////////////////////////
	SetLodeData(initFilePath);
	SetLodeData_Low(initFilePath);
	SetLodeData_High(initFilePath);

	m_Data = new CMatBuf[m_nMax_Men_Count_ini];
	m_Data_Low = new CMatBuf[m_nMax_Men_Count_ini_Low];
	m_Data_High = new CMatBuf[m_nMax_Men_Count_ini_High];

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
	if (dTotalMemory > 60)
	{
		//如果使用的内存超过50 GB,请确认为重复运行(提醒)
		if (dUseMemory > 120)
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
			AllocMem(m_nMaxAllocCount * 1024);
			AllocMem_Low(m_nMaxAllocCount_Low * 1024);
			AllocMem_High(m_nMaxAllocCount_High * 1024);
		}
	}
	//笔记本电脑和内存不足的PC,分配4M
	else
	{
		// 4 MB
		AllocMem(m_nMenAllocCount);
		AllocMem_Low(m_nMenAllocCount_Low);
		AllocMem_High(m_nMenAllocCount_High);
	}
}

CMatBuf* CMatBufferManager::FindFreeBuf()
{
	m_MemManager.Lock();

	int nIndex = -1;

	while (nIndex == -1)
	{
		for (int i = 0; i < m_nMax_Men_Count_ini; i++)
		{
			if (!m_Data[i].GetUse())
			{
				m_Data[i].SetUse(true);
				nIndex = i;
				break;
			}
		}

		Sleep(50);
	}

	m_MemManager.Unlock();

	return &m_Data[nIndex];
}

void CMatBufferManager::FindFreeBuf_Multi(const int nCount, CMatBuf** cMatBuf_Multi)
{
	m_MemManager.Lock();

	int* nIndex = new int[nCount];
	int nCnt = 0;
	while (true)
	{
		for (int i = 0; i < m_nMax_Men_Count_ini; i++)
		{
			if (!m_Data[i].GetUse())
			{
				m_Data[i].SetUse(true);
				nIndex[nCnt++] = i;
				break;
			}
		}
		Sleep(50);
		if (nCnt == nCount) break;
	}
	m_MemManager.Unlock();

	for (int i = 0; i < nCount; i++)
	{
		cMatBuf_Multi[i] = &m_Data[nIndex[i]];
	}
}

void CMatBufferManager::ReleaseFreeBuf(CMatBuf* data)
{
	for (int i = 0; i < m_nMax_Men_Count_ini; i++)
	{
		if (data == &m_Data[i])
		{
			m_Data[i].SetUse(false);
		}
	}
}

void CMatBufferManager::ReleaseFreeBuf_Multi(const int nCount, CMatBuf** data)
{
	for (int i = 0; i < m_nMax_Men_Count_ini; i++)
	{
		for (int j = 0; j < nCount; j++)
		{
			if (data[j] == &m_Data[i])
			{
				m_Data[i].SetUse(false);
			}
		}
	}
}

void CMatBufferManager::AllocMem(__int64 nSizeMB)
{
	for (int i = 0; i < m_nMax_Men_Count_ini; i++)
	{
		// MB
		m_Data[i].AllocMem(nSizeMB * 1024 * 1024);
	}
}

void CMatBufferManager::DeleteMem()
{
	for (int i = 0; i < m_nMax_Men_Count_ini; i++)
	{
		m_Data[i].DeleteMem();
	}
}

//在ini中临时添加数据读取190218YWS
void CMatBufferManager::SetLodeData(CString initFilePath)
{
	CFileFind find;
	BOOL bFindFile = FALSE;
	bFindFile = find.FindFile(initFilePath);
	find.Close();

	CString message;
	message.Format(_T("CMatBufferManager init fail, INI file not exist: %s"), initFilePath);
	if (!bFindFile) AfxMessageBox(message);
	m_nMax_Men_Count_ini = GetPrivateProfileInt(_T("InspInfo"), _T("MaxMen Count"), 1, initFilePath);
	m_nMaxAllocCount = GetPrivateProfileInt(_T("InspInfo"), _T("MaxAlloc Count"), 1, initFilePath);
	m_nMenAllocCount = GetPrivateProfileInt(_T("InspInfo"), _T("MenAlloc Count"), 1, initFilePath);
}

//在ini中临时添加数据读取190218YWS
CString CMatBufferManager::GETDRV()
{
	TCHAR buff[MAX_PATH];
	memset(buff, 0, MAX_PATH);
	::GetModuleFileName(NULL, buff, sizeof(buff));
	CString strFolder = buff;
	CString strRet = strFolder.Left(1);
	strRet.MakeUpper();
	return strRet;
}

//////////////////////////////////////////////////////////////////////////Low

CMatBuf* CMatBufferManager::FindFreeBuf_Low()
{
	m_MemManager_Low.Lock();

	int nIndex = -1;

	while (nIndex == -1)
	{
		for (int i = 0; i < m_nMax_Men_Count_ini_Low; i++)
		{
			if (!m_Data_Low[i].GetUse())
			{
				m_Data_Low[i].SetUse(true);
				nIndex = i;
				break;
			}
		}

		Sleep(50);
	}

	m_MemManager_Low.Unlock();

	return &m_Data_Low[nIndex];
}

void CMatBufferManager::FindFreeBuf_Multi_Low(const int nCount, CMatBuf** cMatBuf_Multi)
{
	m_MemManager_Low.Lock();

	int* nIndex = new int[nCount];
	int nCnt = 0;
	while (true)
	{
		for (int i = 0; i < m_nMax_Men_Count_ini_Low; i++)
		{
			if (!m_Data_Low[i].GetUse())
			{
				m_Data_Low[i].SetUse(true);
				nIndex[nCnt++] = i;
				break;
			}
		}
		Sleep(50);
		if (nCnt == nCount) break;
	}
	m_MemManager_Low.Unlock();

	for (int i = 0; i < nCount; i++)
	{
		cMatBuf_Multi[i] = &m_Data_Low[nIndex[i]];
	}
}

void CMatBufferManager::ReleaseFreeBuf_Low(CMatBuf* data)
{
	for (int i = 0; i < m_nMax_Men_Count_ini_Low; i++)
	{
		if (data == &m_Data_Low[i])
		{
			m_Data_Low[i].SetUse(false);
		}
	}
}

void CMatBufferManager::ReleaseFreeBuf_Multi_Low(const int nCount, CMatBuf** data)
{
	for (int i = 0; i < m_nMax_Men_Count_ini_Low; i++)
	{
		for (int j = 0; j < nCount; j++)
		{
			if (data[j] == &m_Data_Low[i])
			{
				m_Data_Low[i].SetUse(false);
			}
		}
	}
}

void CMatBufferManager::AllocMem_Low(__int64 nSizeMB)
{
	for (int i = 0; i < m_nMax_Men_Count_ini_Low; i++)
	{
		// MB
		m_Data_Low[i].AllocMem(nSizeMB * 1024 * 1024);
	}
}

void CMatBufferManager::DeleteMem_Low()
{
	for (int i = 0; i < m_nMax_Men_Count_ini_Low; i++)
	{
		m_Data_Low[i].DeleteMem();
	}
}

//在ini中临时添加数据读取190218YWS
void CMatBufferManager::SetLodeData_Low(CString initFilePath)
{
	CFileFind find;
	BOOL bFindFile = FALSE;
	bFindFile = find.FindFile(initFilePath);
	find.Close();

	CString message;
	message.Format(_T("CMatBufferManager init fail, INI file not exist: %s"), initFilePath);
	if (!bFindFile) AfxMessageBox(message);
	m_nMax_Men_Count_ini_Low = GetPrivateProfileInt(_T("InspInfo"), _T("MaxMen Count Low"), 1, initFilePath);
	m_nMaxAllocCount_Low = GetPrivateProfileInt(_T("InspInfo"), _T("MaxAlloc Count Low"), 1, initFilePath);
	m_nMenAllocCount_Low = GetPrivateProfileInt(_T("InspInfo"), _T("MenAlloc Count Low"), 1, initFilePath);
}

//////////////////////////////////////////////////////////////////////////HIgh

CMatBuf* CMatBufferManager::FindFreeBuf_High()
{
	m_MemManager_High.Lock();

	int nIndex = -1;

	while (nIndex == -1)
	{
		for (int i = 0; i < m_nMax_Men_Count_ini_High; i++)
		{
			if (!m_Data_High[i].GetUse())
			{
				m_Data_High[i].SetUse(true);
				nIndex = i;
				break;
			}
		}

		Sleep(50);
	}

	m_MemManager_High.Unlock();

	return &m_Data_High[nIndex];
}

void CMatBufferManager::FindFreeBuf_Multi_High(const int nCount, CMatBuf** cMatBuf_Multi)
{
	m_MemManager_High.Lock();

	int* nIndex = new int[nCount];
	int nCnt = 0;
	while (true)
	{
		for (int i = 0; i < m_nMax_Men_Count_ini_High; i++)
		{
			if (!m_Data_High[i].GetUse())
			{
				m_Data_High[i].SetUse(true);
				nIndex[nCnt++] = i;
				break;
			}
		}
		Sleep(50);
		if (nCnt == nCount) break;
	}
	m_MemManager_High.Unlock();

	for (int i = 0; i < nCount; i++)
	{
		cMatBuf_Multi[i] = &m_Data_High[nIndex[i]];
	}
}

void CMatBufferManager::ReleaseFreeBuf_High(CMatBuf* data)
{
	for (int i = 0; i < m_nMax_Men_Count_ini_High; i++)
	{
		if (data == &m_Data_High[i])
		{
			m_Data_High[i].SetUse(false);
		}
	}
}

void CMatBufferManager::ReleaseFreeBuf_Multi_High(const int nCount, CMatBuf** data)
{
	for (int i = 0; i < m_nMax_Men_Count_ini_High; i++)
	{
		for (int j = 0; j < nCount; j++)
		{
			if (data[j] == &m_Data_High[i])
			{
				m_Data_High[i].SetUse(false);
			}
		}
	}
}

void CMatBufferManager::AllocMem_High(__int64 nSizeMB)
{
	for (int i = 0; i < m_nMax_Men_Count_ini_High; i++)
	{
		// MB
		m_Data_High[i].AllocMem(nSizeMB * 1024 * 1024);
	}
}

void CMatBufferManager::DeleteMem_High()
{
	for (int i = 0; i < m_nMax_Men_Count_ini_High; i++)
	{
		m_Data_High[i].DeleteMem();
	}
}

//在ini中临时添加数据读取190218YWS
void CMatBufferManager::SetLodeData_High(CString initFilePath)
{
	CFileFind find;
	BOOL bFindFile = FALSE;
	bFindFile = find.FindFile(initFilePath);
	find.Close();

	CString message;
	message.Format(_T("CMatBufferManager init fail, INI file not exist: %s"), initFilePath);
	if (!bFindFile) AfxMessageBox(message);
	m_nMax_Men_Count_ini_High = GetPrivateProfileInt(_T("InspInfo"), _T("MaxMen Count High"), 1, initFilePath);
	m_nMaxAllocCount_High = GetPrivateProfileInt(_T("InspInfo"), _T("MaxAlloc Count High"), 1, initFilePath);
	m_nMenAllocCount_High = GetPrivateProfileInt(_T("InspInfo"), _T("MenAlloc Count High"), 1, initFilePath);
}