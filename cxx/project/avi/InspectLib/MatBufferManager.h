
/************************************************************************/
//Mat内存管理相关标头
//修改日期:17.08.02
/************************************************************************/

#pragma once

#include "afxmt.h"
#include "Define.h"
#include "MatBuf.h"

class CMatBufferManager : public CWnd
{
	DECLARE_DYNAMIC(CMatBufferManager)

public:
	CMatBufferManager();
	virtual ~CMatBufferManager();

protected:
	DECLARE_MESSAGE_MAP()

public:
	void Initialize(CString initFilePath);
	CMatBuf* FindFreeBuf();
	void				ReleaseFreeBuf(CMatBuf* data);

	void				FindFreeBuf_Multi(const int nCount, CMatBuf** cMatBuf_Multi);
	void				ReleaseFreeBuf_Multi(const int nCount, CMatBuf** data);

	void				AllocMem(__int64 nSizeMB);
	void				DeleteMem();
	void				SetLodeData(CString initFilePath);
	CString				GETDRV();

	////////////////////////////////////////////////////////////////////////// Low

	CMatBuf* FindFreeBuf_Low();
	void				ReleaseFreeBuf_Low(CMatBuf* data);

	void				FindFreeBuf_Multi_Low(const int nCount, CMatBuf** cMatBuf_Multi);
	void				ReleaseFreeBuf_Multi_Low(const int nCount, CMatBuf** data);

	void				AllocMem_Low(__int64 nSizeMB);
	void				DeleteMem_Low();
	void				SetLodeData_Low(CString initFilePath);

	////////////////////////////////////////////////////////////////////////// High

	CMatBuf* FindFreeBuf_High();
	void				ReleaseFreeBuf_High(CMatBuf* data);

	void				FindFreeBuf_Multi_High(const int nCount, CMatBuf** cMatBuf_Multi);
	void				ReleaseFreeBuf_Multi_High(const int nCount, CMatBuf** data);

	void				AllocMem_High(__int64 nSizeMB);
	void				DeleteMem_High();
	void				SetLodeData_High(CString initFilePath);

protected:
	CCriticalSection	m_MemManager;
	CCriticalSection	m_MemManager_Low;
	CCriticalSection	m_MemManager_High;

	CMatBuf* m_Data;
	CMatBuf* m_Data_Low;
	CMatBuf* m_Data_High;

	// 190218 YWS
	int m_nMax_Men_Count_ini = 0;
	int m_nMaxAllocCount = 0;
	int m_nMenAllocCount = 0;

	//////////////////////////////////////////////////////////////////////////Low

	int m_nMax_Men_Count_ini_Low = 0;
	int m_nMaxAllocCount_Low = 0;
	int m_nMenAllocCount_Low = 0;

	//////////////////////////////////////////////////////////////////////////High
	int m_nMax_Men_Count_ini_High = 0;
	int m_nMaxAllocCount_High = 0;
	int m_nMenAllocCount_High = 0;
};

