
/************************************************************************/
//Time Out相关标头
//修改日期:18.02.26
/************************************************************************/

#pragma once

class CTimeOut
{
public:
	CTimeOut(void);
	virtual ~CTimeOut(void);

public:
	long				TimeCheckStart();			//执行动作
	BOOL				GetTimeOutFlag();			//超时确认标志
	void				SetTimeOut(int nTime_ms);	//设置超时

protected:
	int					m_nSetTime;				//设置时间
	int					m_nEndTime;				//结束时间
};

