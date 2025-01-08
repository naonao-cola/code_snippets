// CCPUTimer.cpp: implementation of the CCPUTimer class.
//
//////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "CPUTimer.h"
//#include <windows.h>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
CCPUTimer::CCPUTimer()
{
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//cpu_clock=_cpu_clock;
}

CCPUTimer::~CCPUTimer()
{

}

void CCPUTimer::Start(void)
{
	start = 0;
	//rdtscEx(start.LowPart, start.HighPart);
	QueryPerformanceCounter((LARGE_INTEGER*)&start);
}

void CCPUTimer::End(void)
{
	end = 0;
//	rdtscEx(end.LowPart, end.HighPart);
	QueryPerformanceCounter((LARGE_INTEGER*)&end);
}

double CCPUTimer::GetTime()
{
	//diff=*(__int64*)&end - *(__int64*)&start;
	//time=(double)diff/cpu_clock/1000.0;
	double msec;
	msec=(double)(end-start)/freq*1000.0;
	return msec;
}

void CCPUTimer::Wait(double msec)
{
	double ctime;
	Start();
	while(1){
		// relinquish the remainder of its time slice 
		// to any other thread of equal priority 
		Sleep(0);

		End();
		ctime=GetTime();
		if(ctime>msec)
			return;
		// if start>end 
		if(ctime<0.0)
			Start();
	}
}

void CCPUTimer::WaitFromStart(double msec)
{
	double ctime;
	__int64 curTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&curTime);
	if ((double)(curTime-start) < 0.0) {
		start = curTime;
	}
	while(1){
		// relinquish the remainder of its time slice 
		// to any other thread of equal priority 
		Sleep(0);

		End();
		ctime=GetTime();
		if(ctime>msec)
			return;
		// if start>end 
		if(ctime<0.0)
			Start();
	}
}

double CCPUTimer::Stop(bool StartAgain)
{
	end = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&end);
	double theElapsedTime = GetTime();
	if(StartAgain)
		start = end; 
	return theElapsedTime;
}

BOOL CCPUTimer::MoreThanTime(double dTime)
{
	end = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&end);

	double dCompareTime=(double)(end-start)/freq*1000.0;

	if( dCompareTime > dTime ) return TRUE;
	else return FALSE;
}
