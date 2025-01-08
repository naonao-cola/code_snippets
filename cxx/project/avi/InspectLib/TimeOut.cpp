
/************************************************************************/
//Time Out相关源
//修改日期:18.02.26
/************************************************************************/

#include "StdAfx.h"
#include "TimeOut.h"

CTimeOut::CTimeOut(void)
{
	//设置时间(默认为40秒)
	m_nSetTime = 40000;
}

CTimeOut::~CTimeOut(void)
{
}

long CTimeOut::TimeCheckStart()
{
	//设置时间
	m_nEndTime = clock() + m_nSetTime;

	return 0;
}

//超时确认标志
//调试时,无条件返回False(实际调试时,需要确认...)
BOOL CTimeOut::GetTimeOutFlag()
{
	int nTime = clock();

#ifdef _DEBUG	//Debug
	return	FALSE;
#else			//Release
	return ( nTime >= m_nEndTime ) ? TRUE : FALSE;
#endif	
}

//设置超时
void CTimeOut::SetTimeOut(int nTime_ms)
{
	m_nSetTime = nTime_ms;
}