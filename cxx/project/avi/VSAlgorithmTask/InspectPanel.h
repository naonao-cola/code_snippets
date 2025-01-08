#if !defined(AFX_STAGE_H__E8D3E32F_6DAD_4C46_A743_6707FF28740F__INCLUDED_)
#define AFX_STAGE_H__E8D3E32F_6DAD_4C46_A743_6707FF28740F__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

//
#include "InspThrd.h"

/////////////////////////////////////////////////////////////////////////////

class CInspPanel : public CObject
{
public:
	CInspPanel();           // protected constructor used by dynamic creation

public:

public:
	bool ExitVision();
	bool InitVision();

	//如果没有空线程,请等待出现空线程,然后开始检查
	int StartInspection(WPARAM wParam, LPARAM lParam);
	int StartSaveImage(WPARAM wParam, LPARAM lParam);	//2020.07.22原始影像存储也使用Thread操作

	int CheckInsThread();

	virtual ~CInspPanel();

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CInspPanel)
public:
	//}}AFX_VIRTUAL

private:
	CInspThrd* m_pInspThrd[MAX_THREAD_COUNT];
	
	int			m_nCurThrdID;

};

/////////////////////////////////////////////////////////////////////////////

#endif // !defined(AFX_STAGE_H__E8D3E32F_6DAD_4C46_A743_6707FF28740F__INCLUDED_)
