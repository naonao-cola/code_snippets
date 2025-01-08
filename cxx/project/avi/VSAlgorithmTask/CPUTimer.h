// CCPUTimer.h: interface for the CCPUTimer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CCPUTimer_H__816DDD41_1BE5_11D4_835D_00105A83EED2__INCLUDED_)
#define AFX_CCPUTimer_H__816DDD41_1BE5_11D4_835D_00105A83EED2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

//#define rdtsc(x) \
//{__asm __emit 0fh __asm __emit 031h __asm mov x, eax}

//#define rdtscEx(low, high) \
//{__asm __emit 0fh __asm __emit 031h __asm mov low, eax __asm mov high, edx}

//#define cpuid {__asm __emit 0fh __asm __emit 0a2h}

class CCPUTimer
{
private:
//	__int64 freq,diff;
//	double  cpu_clock;
//	LARGE_INTEGER start,end;
	__int64 start,end, freq;
public:
	void WaitFromStart(double msec);
	//CCPUTimer(double cpu_clock);
	CCPUTimer();
	virtual ~CCPUTimer();

	void Start(void);
	void End(void);
	void Wait(double msec);
	double GetTime();
	double Stop(bool StartAgain);
	BOOL MoreThanTime(double dTime);
};

#endif // !defined(AFX_CCPUTimer_H__816DDD41_1BE5_11D4_835D_00105A83EED2__INCLUDED_)
