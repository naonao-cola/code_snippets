
/************************************************************************/
//Mat内存相关标头
//修改日期:17.08.02
/************************************************************************/

#pragma once

#include "afxmt.h"
#include "Define.h"

//内存减半:使用父,子
enum ENUM_MEMORY_HALF
{
	E_MEMORY_UP		= 0,
	E_MEMORY_DOWN
};

class CMatResultBuf : public CWnd
{
	DECLARE_DYNAMIC(CMatResultBuf)

public:
	CMatResultBuf();
	virtual ~CMatResultBuf();

protected:
	DECLARE_MESSAGE_MAP()

public:
		//分配和禁用全部内存
	void				AllocMem_Result(__int64 nSize);
	void				DeleteMem_Result();

		//使用固定内存
	cv::Mat				GetMat_Result(int rows, int cols, int type, bool bSetFlag = true, ENUM_MEMORY_HALF memory = E_MEMORY_UP);
	cv::Mat				GetMat_Result(cv::Size size, int type, bool bSetFlag = true, ENUM_MEMORY_HALF  memory = E_MEMORY_UP);

		//用Sub的时候...
	void				SetMem_Result(CMatResultBuf* cMatBuf);

		//正在使用
	bool				GetUse_Result();
	void				SetUse_Result(bool bUse);

protected:
		//Mat Type特定的内存大小
	int					GetTypeSize_Result(int type);

		//初始化和设置内存Index
	void				MemIndexSet_Result();

protected:
		CCriticalSection	m_MemLock;				//防止并发访问
		BYTE*				m_Data;					//总内存
		bool				m_bUse;					//确认当前使用
		bool				m_bMemAlloc;			//验证是否已分配内存
		__int64				m_nSizeIndex;			//内存Index
		__int64				m_nMaxSizeToTal;		//分配的总内存
		__int64				m_nStartIndex;			//内存开始Index

	//int					m_nMemoryIndex;

		//如果为NULL
	bool				m_bNULL;
};

