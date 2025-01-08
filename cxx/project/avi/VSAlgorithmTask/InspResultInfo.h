#pragma once

#if !defined(AFX_FTPCLIENT_H__F196D430_806C_4A00_B5BE_04AC559B59A2__INCLUDED_)
#define AFX_FTPCLIENT_H__F196D430_806C_4A00_B5BE_04AC559B59A2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <vector>		
#include <list>
using namespace std;	

#define DEFAULT_DATE  _T("YYYY/MM/DD")
#define DEFAULT_TIME  _T("HH:MM:SS")

enum ENUM_KIND_OF_REPEAT_COORD
{
	ePIXEL = 0	,
	eWORK		,
	eMACHINE	,					// APP Repeat Defect
	eCOORD_KIND
};

#include "DefineInterface.h"

struct ResultHeaderInfo 
{   
		//	金亨柱18.12.06
		//无论	MergeTool Falg如何,无条件操作
	int				MERGE_nRatio;
	cv::Rect		MERGE_rcAlignCellROI;
	float MERGE_dPanelSizeX;
	float MERGE_dPanelSizeY;
	int	MERGE_nWorkDirection;
	int	MERGE_nWorkOriginPosition;
	int	MERGE_nWorkOffsetX;
	int	MERGE_nWorkOffsetY;
	int	MERGE_nDataDirection;
	int	MERGE_nGateDataOriginPosition;
	int	MERGE_nGateDataOffsetX;
	int	MERGE_nGateDataOffsetY;
	int	MERGE_dGatePitch;
	int	MERGE_dDataPitch;
	/////////////

	/*/////////////////////////////////////////////////////////////////////////////////////////////
		ITEM									FORTMAT来源			必需		COMMENT
	/////////////////////////////////////////////////////////////////////////////////////////////*/
	CString	EQP_ID;							/// 999			EQP			Y			Equipment ID
		CString Insp_start_date;				///	YYYY/MM/DD	EQP			Y			检查开始日期
		CString Insp_start_time;				///HH:MM:SS	EQP			Y			检查开始时间
		CString Insp_end_date;					///YYYY/MM/DD	EQP			Y			检查结束日期
		CString Insp_end_time;					///HH:MM:SS	EQP			Y			检查结束时间

	ResultHeaderInfo()
	{
		EQP_ID		= _T("");

		SYSTEMTIME time;
		::GetLocalTime(&time);
		Insp_start_date.Format(_T("%04d/%02d/%02d"), time.wYear, time.wMonth, time.wDay);
		Insp_start_time.Format(_T("%02d:%02d:%02d"), time.wHour, time.wMinute, time.wSecond);

	}

	void SetInspectStartTime()
	{
		SYSTEMTIME time;
		::GetLocalTime(&time);
		Insp_start_date.Format(_T("%04d/%02d/%02d"), time.wYear, time.wMonth, time.wDay);
		Insp_start_time.Format(_T("%02d:%02d:%02d"), time.wHour, time.wMinute, time.wSecond);
	}

	void SetInspectEndTime()
	{
		SYSTEMTIME time;
		::GetLocalTime(&time);
		Insp_end_date.Format(_T("%04d/%02d/%02d"), time.wYear, time.wMonth, time.wDay);
		Insp_end_time.Format(_T("%02d:%02d:%02d"), time.wHour, time.wMinute, time.wSecond);
	}	
};

struct ResultPanelInfo 
{   /*/////////////////////////////////////////////////////////////////////////////////////////////
		ITEM									FORTMAT来源			必需		COMMENT
	/////////////////////////////////////////////////////////////////////////////////////////////*/
	CString EQP_ID;							//	Add By KYH EQP ID Write
	CString LOT_ID;							//  Add By KYH LOT ID Write
		CString Panel_ID;						///	A-18-A	EQP			Y		CIM提供																	
	CString Judge;							///		 A		EQP			Y	
	CString judge_code_1;					///	  AAAA		EQP			Y	
	CString judge_code_2;					///	  AAAA		EQP			Y
	CString	Recipe_ID;						/// A-16-A		EQP			Y
	int nFinalDefectNum;
	cv::Rect rcAlignCellROI;				/// Auto ROI
	ResultPanelInfo()
	{	
		EQP_ID			=  _T("EQP_ID");
		LOT_ID			= _T("LOTID");
		Panel_ID		= _T("PNID");
		Judge			= _T("JUDGE");
		judge_code_1	= _T("CODE1");
		judge_code_2	= _T("CODE2");
		Recipe_ID		= _T("RCPID");
		nFinalDefectNum = 0;
		rcAlignCellROI	= cv::Rect(0, 0, 0, 0);
	}
	void SetAlignCellROI(cv::Rect* rcInput, double dRatio)
	{
		rcAlignCellROI.x		= (int)(rcInput->x		* dRatio);
		rcAlignCellROI.y		= (int)(rcInput->y		* dRatio);
		rcAlignCellROI.width	= (int)(rcInput->width	* dRatio);
		rcAlignCellROI.height	= (int)(rcInput->height	* dRatio);
	}
};

struct ResultDefectInfo 
{   /*///////////////////////////////////////////////////////////////////////////
		ITEM										FORTMAT单位			说明
	///////////////////////////////////////////////////////////////////////////*/
	int			Defect_No;						///	999						0 ~ 999	
	TCHAR		Defect_Code[50];				///	AAAAA					K+4桁（ex：KSoRh）
		cv::Point	Defect_Rect_Point[4];			///9999999999				原始不良Rect转角
		int			Repair_Gate;					///99999					使用repair设备Gate Line的故障位置
		int			Repair_Data;					///99999					使用repair设备Data Line的故障位置
		DOUBLE		Repair_Coord_X;					///±9999.99	um			repair设备使用缺陷开始位置X
		DOUBLE		Repair_Coord_Y;					///±9999.99	um			repair设备使用缺陷起始位置Y
		int			Pixel_Start_X;					///9999					Image坐标X
		int			Pixel_Start_Y;					///99999					图像坐标Y
		DOUBLE		Pixel_Center_X;					///99999.999			Image坐标X
		DOUBLE		Pixel_Center_Y;					///99999.999			Image坐标Y
		int			Pixel_End_X;					///9999					Image坐标X
		int			Pixel_End_Y;					///99999					图像坐标Y
		DOUBLE		Pixel_Repair_X;					///99999.999				Repair坐标X	//中心坐标可用,DOUBLE
		DOUBLE		Pixel_Repair_Y;					///99999.999				Repair坐标Y
		int			Pixel_Crop_Start_X;				///99999					要Crop的坏区域坐标X
		int			Pixel_Crop_Start_Y;				///99999					要Crop的坏区域坐标Y
		int			Pixel_Crop_End_X;				///99999					要Crop的坏区域坐标X
		int			Pixel_Crop_End_Y;				///99999					要Crop的坏区域坐标Y
		int			Gate_Start_No;					///99999					Gate Line的故障起始位置
		int			Data_Start_No;					///99999					Data Line的故障起始位置
		int			Gate_End_No;					///99999					Gate Line的故障结束位置
		int			Data_End_No;					///99999					Data Line的故障结束位置
		DOUBLE		Coord_Start_X;					///±9999.99	um			缺陷起始位置X
		DOUBLE		Coord_Start_Y;					///±9999.99	um			缺陷起始位置Y
		DOUBLE		Coord_End_X;					///±9999.99	um			缺陷结束位置X
		DOUBLE		Coord_End_Y;					///±9999.99	um			缺陷结束位置Y
		int			Defect_Size;					///	999		um			缺陷强度
		int			Img_Number;						///	999						模式号
		TCHAR		Defect_Img_Name[50];			///9-12-9.AAA				缺陷图像文件名
		DOUBLE		Img_Size_X;						///9999.99	um				缺陷图像大小X
		DOUBLE		Img_Size_Y;						///9999.99	um				缺陷图像大小Y
		int			Defect_Type;					///99					缺陷类型
		int			Pattern_Type;					///99					模式类型
	int			Camera_No;						/// 9					
		double		Defect_BKGV;					///不良的平均GV
		double		Defect_MeanGV;					///不良的平均GV
		int			Defect_MinGV;					///不良的最大GV
		int			Defect_MaxGV;					///不良的最大GV
		int			Defect_Size_Pixel;				///大小不良pixel
		bool		Draw_Defect_Rect;				///是否在迷你地图上绘制不良Rect-使用单独文件生成的坐标绘制迷你地图存在不良(ex:MURA)
		int			nRatio;							///使用Pixel Shift Flag-0:Non/1:4-Shot/2:9-Shot
																								///图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)
		bool		bUseReport;//是否报告父项

		double		dDimension;///17.12.04-(长轴+缩短)/2->添加规格(客户要求)

#if USE_ALG_CONTOURS
		int			nContoursX[MAX_CONTOURS];///17.11.29-外围信息(AVI&SVI其他工具)
		int			nContoursY[MAX_CONTOURS];///17.11.29-外围信息(AVI&SVI其他工具)
		int			nContoursCount;
#endif
		int			nReJudgeCode;//AI文本代码
		double      dReJudgeConf;//AI字母零嘀嗒
		int			nReJudgeResult; // AI咯噔咯噔(0:布角)
		bool		bReJudge;//AI瑞特角瑞德(锦江CV)
	//新增分区编号 hjf
	int			nBlockNum;

	ResultDefectInfo()
	{
		Defect_No		= 0;
		_tcscpy_s(Defect_Code, _T("CODE"));
		for (int i=0; i<4; i++)
			Defect_Rect_Point[i] = cv::Point(0, 0);
		Repair_Gate			= 0;
		Repair_Data			= 0;
		Repair_Coord_X		= 0;
		Repair_Coord_Y		= 0;
		Pixel_Start_X		= 0;
		Pixel_Start_Y		= 0;
		Pixel_Center_X		= 0;
		Pixel_Center_Y		= 0;
		Pixel_End_X			= 0;
		Pixel_End_Y			= 0;
		Pixel_Repair_X		= 0;
		Pixel_Repair_Y		= 0;
		Gate_Start_No		= 0;
		Data_Start_No		= 0;
		Gate_End_No			= 0;
		Data_End_No			= 0;
		Coord_Start_X		= 0;
		Coord_Start_Y		= 0;
		Coord_End_X			= 0;
		Coord_End_Y			= 0;
		Defect_Size			= 0;
		Img_Number			= 0;
		_tcscpy_s(Defect_Img_Name, _T("IMG_NAME"));
		Img_Size_X			= 0;
		Img_Size_Y			= 0;
		Defect_Type			= 0;
		Pattern_Type		= 0;
		Camera_No			= 0;
		Defect_BKGV			= 0.0;					
		Defect_MeanGV		= 0.0;					
		Defect_MinGV		= 0;					
		Defect_MaxGV		= 0;					
		Defect_Size_Pixel	= 0;
		Draw_Defect_Rect	= true;
		nRatio				= 0;
		bUseReport			= true;
		dDimension			= 0.0;
		nReJudgeCode		= -1;
		dReJudgeConf		= 0.0;
		nReJudgeResult		= -1;
		bReJudge			= false;
		//新增分区编号初始化 hjf
		nBlockNum			= 0;

#if USE_ALG_CONTOURS
		memset(nContoursX, 0, sizeof(int) * MAX_CONTOURS );
		memset(nContoursY, 0, sizeof(int) * MAX_CONTOURS );
		nContoursCount = 0;
#endif
	}
};

struct STRU_DEFECT_INFO
{
	ResultHeaderInfo Hinfo;
	ResultPanelInfo  Pinfo;
	ResultDefectInfo Dinfo;
};

struct ResultPanelData
{
	ResultHeaderInfo							m_ResultHeader;
	ResultPanelInfo								m_ResultPanel;
	CArray<ResultDefectInfo, ResultDefectInfo&>	m_ListDefectInfo;
	CPoint										CornerPt;
		//按不良类别计数
		int											m_nDefectTrend[E_PANEL_DEFECT_TREND_COUNT];//总数量E_PANEL_DEFECT_TREND_COUNT->BP+DP等判定所需的不良计数

	CRITICAL_SECTION	m_cs;

	INT_PTR Add_DefectInfo(ResultDefectInfo& DefectInfo)
	{
		EnterCriticalSection(&m_cs);
		INT_PTR ps = m_ListDefectInfo.Add(DefectInfo);
		LeaveCriticalSection(&m_cs);

		return ps;
	}

	ResultPanelData()
	{
		m_ListDefectInfo.RemoveAll();
		InitializeCriticalSection(&m_cs);		

		CornerPt.x= 0;
		CornerPt.y= 0;

		memset(m_nDefectTrend, 0, sizeof(int) * E_PANEL_DEFECT_TREND_COUNT);
	}
	~ResultPanelData()
	{
		m_ListDefectInfo.RemoveAll();
	}
};

struct Coord
{   
	DOUBLE X ;
	DOUBLE Y ;
	Coord()
	{
		X = 0;
		Y = 0;		
	}
	Coord(DOUBLE dX, DOUBLE dY)
	{
		X = dX;
		Y = dY;		
	}
};

struct WorkCoordCrt
{
	double	dPanelSizeX;
	double	dPanelSizeY;

	// for Work Coordinate
	int		nWorkDirection;					// 0 : X = Width, 1 : Y = Width
	int		nWorkOriginPosition;			// 0 : LT, 1 : RT, 2 : RB, 3 : LB
	int     nWorkOffsetX;
	int     nWorkOffsetY;

	// for Gate/Data Coordinate
	int		nDataDirection;						// 0 : Data = X, 1 : Data = Y
	int     nGateDataOriginPosition;			// 0 : LT, 1 : RT, 2 : RB, 3 : LB
		int     nGateDataOffsetX;//原点Pixel坐标Offset
	int     nGateDataOffsetY;
	double  dGatePitch;							// Gate Pitch (um)
	double  dDataPitch;							// Data Pitch (um)	

	// Common
		double	dResolution[MAX_CAMERA_COUNT];//Camera星Resolution

	WorkCoordCrt()
	{
		dPanelSizeX = 0.0;
		dPanelSizeY = 0.0;
		nWorkDirection = 0;
		nWorkOriginPosition = 0;
		nWorkOffsetX = 0;
		nWorkOffsetY = 0;		
		nDataDirection = 0;
		nGateDataOriginPosition = 0;
		nGateDataOffsetX = 0;
		nGateDataOffsetY = 0;
		dGatePitch = 0.0;
		dDataPitch = 0.0;
		for (int nCamIndex = 0; nCamIndex < MAX_CAMERA_COUNT; nCamIndex++)
			dResolution[nCamIndex] = 0.0;
	}	
};

struct GD_POINT///Gate/Data的值结构
{   
	int Gate ;
	int Data ;
		//enum DataDirection{Xdirection,	Ydirertion} Direction; //  决定GATE/DATA是X轴还是Y轴。

	GD_POINT()
	{
		Gate = 0;
		Data = 0;		
	}
	GD_POINT(int nGate, int nData)
	{
		Gate = nGate;
		Data = nData;		
	}
};

class CFileProcess 
{	
public:	
	void    m_fnCreateFolder(CString szPath);									
	int		m_fnOnWritefile(CString DstFileName, CString DstFileInfo );

	CFileProcess(); 
	virtual ~CFileProcess();

private:
	CString m_fnLastToken(CString strSplitVal, CString strSplit);
	int     m_fnLastTokenCnt(CString strSplitVal, CString strSplit);
	int		m_fnCutToken(CString strSplitVal, CString strSplit);

protected:	

public:
};

///在GUI中保存Recipe时,将其值用于计算
class CWriteResultInfo 
{	

private:	
	CFileProcess m_FileLOGPROC;
	WorkCoordCrt m_stWorkCoordCrt;

public:
	CWriteResultInfo(); 
	virtual ~CWriteResultInfo();

		//金亨柱18.12.06
		//无条件操作,与MergeTool Falg无关
	WorkCoordCrt GetWorkCoordCrt(){	return m_stWorkCoordCrt;	};

	// 17.11.24 - Panel Size
	double		GetPanelSizeX(){ return m_stWorkCoordCrt.dPanelSizeX; };
	double		GetPanelSizeY(){ return m_stWorkCoordCrt.dPanelSizeY; };

	// Get
		//17.07.07 Ratio删除修改为全部转交Corner而不是LT-改变坐标系的计算方式
		Coord		CalcWorkCoord(cv::Rect rcAlignedCell, Coord cpPixelCoord, int nDefectRatio, int nCurRatio);//将像素中心坐标更改为CPoint->Coord

		//利用缺陷坐标和起始坐标以及GATE/DATA的尺寸导出GATE/DATA LINE值。
		GD_POINT	CalcGateDataCoord(cv::Rect rcAlignedCell, Coord CoordPixel, int nDefectRatio, int nCurRatio);//将像素中心坐标更改为CPoint->Coord
	double		CalcDistancePixelToUm(double dPixelDistance, int nCameraNum, int nRatio);
	double		GetCamResolution(int nCameraNum)	{	return m_stWorkCoordCrt.dResolution[nCameraNum]		;};
	void		GetCalcResolution(cv::Rect rcAlignedCell, double& dHResolution, double& dVResolution)	{	dHResolution = m_stWorkCoordCrt.dPanelSizeX / (rcAlignedCell.width * 1.0) * 1000; 	\
																											dVResolution = m_stWorkCoordCrt.dPanelSizeY / (rcAlignedCell.height * 1.0) * 1000;	};
	//CIH 2017.07.14
		//从JudgementRepair获取值
		//m_stWorkCoordCtrt变量为private
	//
	void		GetWorkCoordUsingRepair(int &nWorkOriginPosition, int &nWorkDirection )	{	nWorkOriginPosition = m_stWorkCoordCrt.nWorkOriginPosition;			 
																							nWorkDirection		= m_stWorkCoordCrt.nWorkDirection;				};

	// Set
	void		SetWorkCoordInfo(double dPanelSizeX, double dPanelSizeY, int nCurWorkDirection, int nCurWorkOrgPos, int nCurWorkOriginX, int nCurWorkOriginY, 
							int nCurDataDrection, int nCurGDOriginPos, int nCurGDOrgX, int nCurGDOrgY, double dCurGatePitch, double dCurDataPitch, 
							double* dCurResolution);

		//生成结果文件
	int			WriteResultPanelData(CString DstFileName, ResultPanelData& resultPanelData, bool includeAI=false);
	int			WriteResultPanelData_ToMainPC(CString DstFileName, ResultPanelData& resultPanelData);

	int			WritePanelTrend(CString DstFileName, int* pDefectTrend, CString strPanelID, CString strPanelGrade);
	int			WriteFinalDefect(CString DstFileName, int nDefectNum, CString strPanelID);
	//获取分区格位置 hjf
	int			GetGridNumber(int imageWidth, int imageHeight, int X, int Y, int center_x, int center_y);

private:
	int			m_fnWriteHeaderInfo(CString DstFileName, ResultHeaderInfo& HeaderInfo);	
	int			m_fnWritePanelInfo(CString DstFileName, ResultPanelInfo& PanelInfo);
	int			m_fnWriteDefectInfo(CString DstFileName, CArray<ResultDefectInfo, ResultDefectInfo&> &DefectInfo, bool includeAI = false);
	CString		m_fnGetHeaderString(ResultHeaderInfo HeaderInfo);

private:
	CString		m_fnDivisionPoint(CString  strSplitVal, CString strSplit);
	CString		m_fnIntToCstr(int iConvert,CString strDivision=_T(""));
	CString		m_fnDblToCstr(double iConvert,CString strDivision);
	CString		m_fnBoolToCstr(bool iConvert);
	CString		m_fnConvertDateFormat(CString strConvert, CString strDivision);		

protected:	
public:
private:
};

struct RepeatDefectInfo
{
	ENUM_DEFECT_JUDGEMENT	eDefType;
	CPoint					ptCenterPos;
	int						nRepeatCount;
	int						nStageNo;		// APP Repeat Defect

	RepeatDefectInfo()
	{
		eDefType = (ENUM_DEFECT_JUDGEMENT)0;
		ptCenterPos = CPoint(0, 0);
		nRepeatCount = 0;
		nStageNo = 0;
	}
	void SetRepeatInfo(ENUM_DEFECT_JUDGEMENT eDefectJudge, DOUBLE dX, DOUBLE dY)
	{
			//当前不良代表代码
		eDefType = eDefectJudge;
			//CCD不良-仅获取Pixel中心坐标
		ptCenterPos.x = (LONG)dX;
		ptCenterPos.y = (LONG)dY;
		nRepeatCount = 1;
	}
	// APP Repeat Defect
	void SetRepeatInfo(ENUM_DEFECT_JUDGEMENT eDefectJudge, DOUBLE dX, DOUBLE dY, INT StageNo)
	{
			//当前不良代表代码
		eDefType = eDefectJudge;
			//CCD不良-仅获取Pixel中心坐标
		ptCenterPos.x = (LONG)dX;
		ptCenterPos.y = (LONG)dY;
		nRepeatCount = 1;
		nStageNo = StageNo;
	}

		//List Merge按Pixel坐标排序
	bool operator < (const RepeatDefectInfo& p)
	{
		return (this->ptCenterPos.x + this->ptCenterPos.y < p.ptCenterPos.x + p.ptCenterPos.y);
	}
	bool operator > (const RepeatDefectInfo& p)
	{
		return (this->ptCenterPos.x + this->ptCenterPos.y > p.ptCenterPos.x + p.ptCenterPos.y);
	}	
		//速度改善测试-Functor
	void CheckVal(const std::list<RepeatDefectInfo>* nVal)
	{
		struct DoSetVal1
		{
			DoSetVal1(RepeatDefectInfo* val) : m_val(val) {}
			bool operator()(RepeatDefectInfo This)
			{
					//if(NULL==This)//避开NULL值。
				// 				return false;
				//			This.CheckVal1(m_val);
				return true;
			}
			const RepeatDefectInfo* m_val;
		};

			//当前不良列表
		//for_each(nVal->begin(), nVal->end(), DoSetVal1(this));
		//for (iterSrc = nVal->begin(); iterSrc != nVal->end(); )
		{

		}

	////如果没有任何重叠,则删除现有的错误

// 			this->erase();

// 			iterDst++;

	}
};

struct ListCurDefect
{
		std::list<RepeatDefectInfo> listCurDefInfo[eCOORD_KIND];//当前不良列表
		BOOL bUseChkRptDefect[eCOORD_KIND];//Pixel是否启用工作坐标重复检查
	ListCurDefect()
	{
		for (int i=0; i<eCOORD_KIND; i++)
		{
			listCurDefInfo[i].clear();
			bUseChkRptDefect[i] = false;
		}
	}
 	void Add_Tail(ENUM_KIND_OF_REPEAT_COORD eKind, ENUM_DEFECT_JUDGEMENT eJudge, ResultDefectInfo* pDefect)
 	{
 		if (bUseChkRptDefect[eKind])
 		{
 			RepeatDefectInfo stRepeatDefectInfo;
 			DOUBLE dX = 0., dY = 0.;

 			if (eKind ==ePIXEL)
 			{
 				dX = pDefect->Pixel_Center_X;
 				dY = pDefect->Pixel_Center_Y;
 			}
 			else
 			{
 				dX = pDefect->Repair_Coord_X;
 				dY = pDefect->Repair_Coord_Y;
 			}
 			stRepeatDefectInfo.SetRepeatInfo(eJudge, dX, dY);
 			listCurDefInfo[eKind].push_back(stRepeatDefectInfo);
 		}
 	}
	// APP Repeat Defect
	void Add_Tail(ENUM_KIND_OF_REPEAT_COORD eKind, ENUM_DEFECT_JUDGEMENT eJudge, ResultDefectInfo* pDefect, INT nStageNo)
	{
		if (bUseChkRptDefect[eKind])
		{
			RepeatDefectInfo stRepeatDefectInfo;
			DOUBLE dX = 0., dY = 0.;

			if (eKind ==eMACHINE)
			{
				dX = pDefect->Pixel_Center_X;
				dY = pDefect->Pixel_Center_Y;
			}
			else
			{
				dX = pDefect->Repair_Coord_X;
				dY = pDefect->Repair_Coord_Y;
			}
			stRepeatDefectInfo.SetRepeatInfo(eJudge, dX, dY, nStageNo);
			listCurDefInfo[eKind].push_back(stRepeatDefectInfo);
		}
	}
};

#endif // !defined(AFX_FTPCLIENT_H__F196D430_806C_4A00_B5BE_04AC559B59A2__INCLUDED_)
