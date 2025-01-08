#ifndef DEFINE_H
#define DEFINE_H

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#define DLLAPI __declspec(dllimport)
/*
//添加当前解决方案驱动程序判断
#define VS_ALGORITHM_TASK_INI_FILE			theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\INI\\Algorithm.ini")
#define ALGORITHM_LOG_PATH					theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\LOG\\Algorithm\\")
#define DEVICE_FILE_PATH					theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\INI\\Device.cfg")
#define INIT_FILE_PATH						theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\INI\\Initialize.ini")
#define MODEL_FILE_PATH						theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\MODEL")
#define CCD_DEFECT_FILE_PATH				theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\INI\\CCD\\CCD.Index")
#define CCD_DEFECT_FILE_PATH2				theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\INI\\CCD\\DEAD_CCD.Index")
#define COLOR_CORRECTION_FILE_PATH			theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\INI\\CCD\\ColorCorrection.ini")
#define ALIGN_IMAGE_PATH					theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\IMAGE\\Align.bmp")
#define DEFECT_INFO_PATH					theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\Defect Info")
#define REPEAT_DEFECT_PIXEL_INFO_PATH		theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\Defect Info\\CountingCCDDefect.txt")
#define REPEAT_DEFECT_WORK_INFO_PATH		theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\Defect Info\\CountingWorkDefect.txt")
#define REPEAT_DEFECT_MACHINE_INFO_PATH		theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\Defect Info\\CountingMachineDefect.txt")	// 2018.09.21 MDJ APP Repeat Defect
#define REPEAT_DEFECT_MACHINE_INFO_PATH2	theApp.m_Config.GETDRV() + _T(":\\IMTC\\DATA\\Defect Info\\CountingMachineDefect")		//2018.10.09为MDJ Test生成文件

////////202018.01.18 sggim修改GetCurrentDrive()->strDrive-ThreadParameter获取当前驱动器并设置路径
#define ORIGIN_PATH							strOriginDrive + theApp.m_Config.GetOriginPath()
#define RESULT_PATH							 strDrive + theApp.m_Config.GetResultPath()
#define INSP_PATH							strDrive + theApp.m_Config.GetInspPath()
#define ALG_RESULT_PATH						 strDrive + _T("ARESULT\\")
#define INSP_INFO_FILE_PATH					 strDrive + _T("Defect Info\\Insp Data")
//////////

// MergeTool 
//添加当前解决方案驱动程序判断
#define MERGETOOL_PATH						strDrive  +_T(":\\MergeTool")

//result for Gui
#define FINALRESULT_PATH						strDrive  + _T("\\FinalResult")

//分区存储文件夹
#define BLOCKRESULT_PATH						strDrive  + _T("\\BlockResult")
*/

#define VS_ALGORITHM_TASK_INI_FILE			theApp.m_Config.GETCmdDRVPath() + _T("\\Config\\Algorithm.ini")
#define ALGORITHM_LOG_PATH					theApp.m_Config.GETCmdDRVPath() + _T("\\LOG\\Algorithm\\")//返回上层目录写日志 hjf
#define DEVICE_FILE_PATH					theApp.m_Config.GETCmdDRVPath() + _T("\\Config\\Device.cfg")
#define INIT_FILE_PATH						theApp.m_Config.GETCmdDRVPath() + _T("\\Config\\Initialize.ini")

#define CCD_DEFECT_FILE_PATH				theApp.m_Config.GETCmdDRVPath() + _T("\\CCD\\CCD.Index")
#define CCD_DEFECT_FILE_PATH2				theApp.m_Config.GETCmdDRVPath() + _T("\\CCD\\DEAD_CCD.Index")
#define COLOR_CORRECTION_FILE_PATH			theApp.m_Config.GETCmdDRVPath() + _T("\\CCD\\ColorCorrection.ini")

#define DEFECT_INFO_PATH					theApp.m_Config.GETCmdDRVPath() + _T("\\Defect Info")
#define REPEAT_DEFECT_PIXEL_INFO_PATH		theApp.m_Config.GETCmdDRVPath() + _T("\\Defect Info\\CountingCCDDefect.txt")
#define REPEAT_DEFECT_WORK_INFO_PATH		theApp.m_Config.GETCmdDRVPath() + _T("\\Defect Info\\CountingWorkDefect.txt")
#define REPEAT_DEFECT_MACHINE_INFO_PATH		theApp.m_Config.GETCmdDRVPath() + _T("\\Defect Info\\CountingMachineDefect.txt")	
#define REPEAT_DEFECT_MACHINE_INFO_PATH2	theApp.m_Config.GETCmdDRVPath() + _T("\\Defect Info\\CountingMachineDefect")		


#define ORIGIN_PATH							 strOriginDrive + theApp.m_Config.GetOriginPath()
#define RESULT_PATH							 strDrive + theApp.m_Config.GetResultPath()
#define INSP_PATH							 strDrive + theApp.m_Config.GetInspPath()
#define ALG_RESULT_PATH						 strDrive + _T("ARESULT\\")
#define INSP_INFO_FILE_PATH					 strDrive + _T("Defect Info\\Insp Data")
//////////

#define MERGETOOL_PATH						strDrive  +_T(":\\MergeTool")

//result for Gui
#define FINALRESULT_PATH						strDrive  + _T("\\FinalResult")

//分区存储文件夹
#define BLOCKRESULT_PATH						strDrive  + _T("\\BlockResult")



#include "..\..\commonheaders\Structure.h"

//////////////////////////////////////////////////////////////////////////
// OpenCV 3.1
//////////////////////////////////////////////////////////////////////////
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\cudafilters.hpp>
//#include <opencv2\cudaimgproc.hpp>
//#include <opencv2\cudaarithm.hpp>
//#include <opencv2\cudabgsegm.hpp>
//#include <opencv2\cudacodec.hpp>
//#include <opencv2\cudafeatures2d.hpp>
//#include <opencv2\cudaobjdetect.hpp>
//#include <opencv2\cudawarping.hpp>

using namespace cv;
using namespace cv::ml;
using namespace cv::cuda;

//要声明为全局的变量定义在这里
//如果算法参数发生变化,为了应对,提前确保较大的参考内存空间。以下为曲面结构尺寸的常数
const int MAX_MEM_SIZE_CAM_COUNT				= 4;
const int MAX_MEM_SIZE_ROI_COUNT				= 1;
const int MAX_MEM_SIZE_E_INSPECT_AREA			= 20;		//无检查区域,倒圆角单元区域
const int MAX_MEM_SIZE_E_ALGORITHM_NUMBER		= 20;
const int MAX_MEM_SIZE_E_DEFECT_NAME_COUNT		= 150;		//18.05.31(不良数量增加)
const int MAX_MEM_SIZE_ALG_PARA_TOTAL_COUNT		= 200;		//Mura Normal Parameter增加
const int MAX_MEM_SIZE_GRAB_STEP_COUNT			= 20;
const int MAX_MEM_SIZE_E_DEFECT_JUDGMENT_COUNT	= 70;
const int MAX_MEM_SIZE_E_MAX_INSP_TYPE			= 10;
const int MAX_MEM_SIZE_AD_PARA_TOTAL_COUNT		= 15;
const int MAX_MEM_SIZE_ALIGN_PARA_TOTAL_COUNT	= 200;		//pwj 20.09.10(参数增加)
const int MAX_MEM_SIZE_ROUND_COUNT				= 20000;	// 17.07.20 [Round]
///新增分区数量， 限制3*3 hjf
const int MAX_MEM_SIZE_BLOCK_COUNT = 9;

const int MAX_MEM_SIZE_LABEL_COUNT = 5;    //yuxuefei 2023.06.16
const int MAX_MEM_SIZE_MARK_COUNT = 5;    //yuxuefei 2023.06.16
//添加2015.08.06 Defect Filter ROI
enum ENUM_ROI_DRAW_MODE
{
	eBasicROI	=	0,
	eFilterROI,
	eRndROI,
};

//2017.06.06 NDH:GUI的顺序应该相同。
enum ENUM_PAD
{
	E_PAD_LEFT		= 0	,
	E_PAD_RIGHT			,
	E_PAD_TOP 			,
	E_PAD_BOTTOM				
};

enum ENUM_PANEL_DIRECTION
{
	E_PANEL_LEFT	= 0	,
	E_PANEL_TOP			,
	E_PANEL_RIGHT		,
	E_PANEL_BOTTOM		,
	E_PANEL_DIRECTION_MAX
};
// [CInspectAlign] MARK
enum ENUM_MARK
{
	E_MARK_RIGHT_TOP = 0,
	E_MARK_RIGHT_BOTTOM,
	E_MARK_END
};
// [CInspectAlign] Corner
enum ENUM_CORNER
{
	E_CORNER_LEFT_TOP		= 0	,
	E_CORNER_RIGHT_TOP			,
	E_CORNER_RIGHT_BOTTOM		,
	E_CORNER_LEFT_BOTTOM		,
	E_CORNER_END
};

//日志类型
enum ENUM_KIND_OF_LOG
{
	eLOGCAM0 = 0,
	eLOGCAM1,
	eLOGCAM2,
	eLOGCAM3,
		eRESERVE0,		//可增加Camera数量
	eRESERVE1,
	eRESERVE2,
	eRESERVE3,
	eLOGPROC,
	eLOGTACT,
	eLOGCOMM,
	eLOGTEST
};

enum ENUM_INSPECT_MODE
{
	eAutoRun = 0,
	eManualGrabInspect,
	eManualInspect
};

enum ENUM_ABNORMAL_PAD_EDGE
{
	E_ABNORMAL_PAD_EDGE_TOTAL = 0,
	E_ABNORMAL_PAD_EDGE_TOP,
	E_ABNORMAL_PAD_EDGE_BOTTOM,
	E_ABNORMAL_PAD_EDGE_MIDDLE,
	E_ABNORMAL_PAD_EDGE

};

#define MAX_THREAD_COUNT	100									//可同时检查的线程数

// for Tray Icon - User Message
#define	WM_TRAYICON_MSG					WM_USER + 1
#define	WM_PRINT_UI_LOG_MSG_UNICODE		WM_USER + 2
#define	WM_PRINT_UI_LOG_MSG_MULTI_BYTE	WM_USER + 3

#define MAX_GRID_LOG					1000

#define	WM_START_INSPECTION				WM_USER + 108
#define	WM_START_SAVE_IMAGE				WM_USER + 109

// enum ---------------------------------------------------------
// LOG LEVEL
enum ENUM_LOG_LEVEL
{
	eLOGLEVEL_DETAIL	= 1,
	eLOGLEVEL_BASIC,
	eLOGLEVEL_SIMPLE
};

#define ALG_LOG_LEVEL eLOGLEVEL_DETAIL

//定义Alarm ID
enum ENUM_ALARM_ID
{
	eALARMID_LIGHT_ABNORMAL		= 1000,
	eALARMID_DIABLE_CHECK_LIGHT,
	eALARMID_DUST_ABNORMAL,
	eALARMID_DIABLE_CHECK_DUST,

	eALARMID_ALIGN_ANGLE_ERROR	= 1100,
	eALARMID_ALIGN_ANGLE_WARNING,
	eALARMID_ALIGN_FAIL,

	eALARMID_CCD_DEFECT_ERROR	= 3000,
	eALARMID_WORK_DEFECT_ERROR,
	eALARMID_MACHINE_DEFECT_ERROR,				// 2018.09.25 MDJ APP Repeat Defect
	eALARMID_CCD_DEFECT_WARNING,
	eALARMID_WORK_DEFECT_WARNING,
	eALARMID_MACHINE_DEFCT_WARNING,				// 2018.09.25 MDJ APP Repeat Defect

	//异常警报 ：索引起始 3100 hjf
	eALARMID_PG_DISPLAY = 3100,//PG压接异常，警报处理
	eALARMID_COLOUR_DISPLAY,//颜色显示异常，警报处理（适用于MTP未正确或未校正情况） 
	eALARMID_DARK_DISPLAY,//黑屏，警报处理 （正常情况会走E级流程，同一个stage连续三次E级之后，UI会通知物流向PLC警报，所以一次黑屏暂不警报处理，判E级）
	eALARMID_MEANSTD_DISPLAY,//均值方差异常，同黑屏处理方式
	/// 50000 ~ 59999 AVI
	//eALARMID_AVI_OFFSET			= 50000,

	/// 60000 ~ 69999 AVI
	//eALARMID_SVI_OFFSET			= 60000,

	/// 70000 ~ 79999 AVI
	//eALARMID_APP_OFFSET			= 70000,
};

//定义Alarm Type
enum ENUM_ALARM_TYPE
{
		eALARMTYPE_LIGHT			= 1,		//警报
		eALARMTYPE_HEAVY						//严重警报
};

//使用返回值后,删除动态分配的内存
__inline char* CSTR2PCH(CString strInput)
{
	char * tmpch;
	int sLen = WideCharToMultiByte(CP_ACP, 0, strInput, -1, NULL, 0, NULL, NULL);       
	tmpch = new char[sLen + 1];
	WideCharToMultiByte(CP_ACP, 0, strInput, -1, tmpch, sLen, NULL, NULL);
	return tmpch;
}

__inline bool COPY_CSTR2TCH(TCHAR* strDst, CString strSrc, size_t sizeDst)
{
	memset(strDst, 0, sizeDst);
	if (strSrc.GetLength() * sizeof(TCHAR) >= sizeDst)
		strSrc = strSrc.Left((int)(sizeDst / sizeof(TCHAR)) - 1);
	try{
		memcpy(strDst, (LPCTSTR)strSrc, strSrc.GetLength() * sizeof(TCHAR));
	}
	catch(...){
		return false;
	}
	return true;
}

//2017.06.06 NDH:添加ROI over检查函数
inline void GetCheckROIOver(cv::Rect rtInput, cv::Rect rtCheck, cv::Rect &rtOutput)
{
	rtOutput = rtInput;

	if ( rtInput.x		<  rtCheck.x	)		rtOutput.x	=	rtCheck.x;
	if ( rtInput.y		<  rtCheck.y	)		rtOutput.y	=	rtCheck.y;

	int nCheck = ( rtCheck.x + rtCheck.width ) - ( rtOutput.x + rtInput.width ) ;
	if ( nCheck <  0	)		rtOutput.width += nCheck;

		nCheck = ( rtCheck.y + rtCheck.height) - ( rtOutput.y + rtInput.height );
	if ( nCheck <  0	)		rtOutput.height += nCheck;

	if( rtInput.width <= 0)
		rtInput.width = 1;
	if(rtInput.height <= 0)
		rtInput.height = 1;
};

// 2018.03.03
/*////////////////////////////////////////////////////////////////
函数功能:检查正方矩阵Mask Size是偶数还是小于3
返回值:返回大于3的奇数MaskSize,
*/////////////////////////////////////////////////////////////////
inline void GetCheckMaskSize(int& nInput)
{
	if(nInput < 3)
		nInput = 3;

	if(nInput % 2 == 0)
	{
		nInput += 1;
	}
};

#endif
