
/************************************************************************/
//Mat内存管理相关标头
//修改日期:17.08.02
/************************************************************************/

#pragma once

#include "afxmt.h"
#include "Define.h"
#include "MatResultBuf.h"

#define MAX_MEM_COUNT	10

class CMatBufferResultManager : public CWnd
{
	DECLARE_DYNAMIC(CMatBufferResultManager)

public:
	CMatBufferResultManager();
	virtual ~CMatBufferResultManager();

protected:
	DECLARE_MESSAGE_MAP()

public:
	CMatResultBuf* FindFreeBuf_Result();
	void					ReleaseFreeBuf_Result(CMatResultBuf* data);

	void					AllocMem_Result(__int64 nSizeMB);
	void					DeleteMem_Result();

protected:
	CCriticalSection		m_MemManager;
	CMatResultBuf			m_Data[MAX_MEM_COUNT];
};

