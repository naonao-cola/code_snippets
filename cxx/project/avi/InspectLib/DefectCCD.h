
/************************************************************************/
//与CCD Defect相关的标头
//修改日期:17.07.10
/************************************************************************/

#pragma once

#include "Define.h"
#include "MatBuf.h"				//内存管理
#include "InspectLibLog.h"

//CCD不良结构
struct tCCD_DEFECT
{
	int x;		//x坐标
	int y;		//y坐标
	int gv;		//GV值

	//初始化结构体
	tCCD_DEFECT()
	{
		x = 0;
		y = 0;
		gv = 0;
	}
};

class CDefectCCD
{
public:
	CDefectCCD(void);
	virtual ~CDefectCCD(void);

	//内存管理
	CMatBuf* cMem;
	void		SetMem(CMatBuf* data) { cMem = data; };
	CMatBuf* GetMem() { return	cMem; };

	//////////////////////////////////////////////////////////////////////////
	InspectLibLog		cInspectLibLog;
	InspectLibLog* m_cInspectLibLog;
	clock_t				m_tInitTime;
	clock_t				m_tBeforeTime;
	wchar_t* m_strAlgLog;

	void		SetLog(InspectLibLog* cLog, clock_t tTimeI, clock_t tTimeB, wchar_t* strLog)
	{
		m_tInitTime = tTimeI;
		m_tBeforeTime = tTimeB;
		m_cInspectLibLog = cLog;
		m_strAlgLog = strLog;
	};

	void		writeInspectLog(int nAlgType, char* strFunc, wchar_t* strTxt)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, m_strAlgLog);
	};

	void		writeInspectLog_Memory(int nAlgType, char* strFunc, wchar_t* strTxt, __int64 nMemory_Use_Value = 0)
	{
		if (m_cInspectLibLog == NULL)
			return;

		m_tBeforeTime = m_cInspectLibLog->writeInspectLogTime(nAlgType, m_tInitTime, m_tBeforeTime, strFunc, strTxt, nMemory_Use_Value, m_strAlgLog);
	};

	CString		GETDRV()
	{
		return m_cInspectLibLog->GETDRV();
	}
	//////////////////////////////////////////////////////////////////////////
	// CCD Defect Load
	long	DefectCCDLoad(CString strFileName, CString strFileName2);

	// CCD Defect Save
	long	DefectCCDSave(cv::Mat& matSrcBuffer, CString strFileName, CString strFileName2);

	//获取CCD不良删除数量
	int		GetDefectCCDDeleteCount();

	//获取CCD不良校正数量
	int		GetDefectCCDOffsetCount();

	//删除CCD故障
	long	DeleteDefectCCD(cv::Mat& matSrcBuffer, int nSize = 0, int nPS = 1);

	//自动删除CCD故障
	long	DeleteAutoDefectCCD(cv::Mat& matSrcBuffer, float fGV, float fBkGV, int nPS = 1, CMatBuf* cMem = NULL);

	//CCD不良校正
	long	OffsetDefectCCD(cv::Mat& matSrcBuffer, int nSize = 0, int nPS = 1);

protected:
	//CCD Defect Delete(删除)
	vector <tCCD_DEFECT>	ptIndexsDelete;

	//CCD Defect Offset(用于校准)
	vector <tCCD_DEFECT>	ptIndexsOffset;

	//确认是否加载了
	bool	bLoad;
};