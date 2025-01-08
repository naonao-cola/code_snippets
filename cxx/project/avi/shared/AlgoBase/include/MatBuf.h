
#pragma once
#include "BaseDefine.h"


enum ExportAPI ENUM_MEMORY_HALF
{
	E_MEMORY_UP = 0,
	E_MEMORY_DOWN
};

class ExportAPI CMatBuf
{
public:
	CMatBuf();
	virtual ~CMatBuf();

public:
	void				AllocMem(__int64 nSize);
	void				DeleteMem();

	cv::Mat				GetMat(int rows, int cols, int type, bool bSetFlag = true, ENUM_MEMORY_HALF memory = E_MEMORY_UP);
	cv::Mat				GetMat(cv::Size size, int type, bool bSetFlag = true, ENUM_MEMORY_HALF  memory = E_MEMORY_UP);

	void				SetMem(CMatBuf* cMatBuf);

	bool				GetUse();
	void				SetUse(bool bUse);
	__int64				Get_AutoMemory();
	__int64				Get_FixMemory();

protected:
	int					GetTypeSize(int type);

	void				MemIndexSet();

protected:
	//CCriticalSection	m_MemLock;
	CRITICAL_SECTION    m_MemLock;
	BYTE*				m_Data;
	bool				m_bUse;
	bool				m_bMemAlloc;
	__int64				m_nSizeIndex;
	__int64				m_nMaxSizeToTal;
	__int64				m_nStartIndex;

	__int64				m_nMem_Fix;
	__int64				m_nMem_Auto;
	//int					m_nMemoryIndex;

	bool				m_bNULL;
};
