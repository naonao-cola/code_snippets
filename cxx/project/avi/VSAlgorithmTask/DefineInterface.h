////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef DEFINE_INTERFACE_H
#define DEFINE_INTERFACE_H

/***************************************************************************************************
 需要在Algorithm DLL和Algorithm Task之间共享的ENUM,结构体等在此处定义-AVI/SVI/APP共同点
***************************************************************************************************/

#pragma once

#include "Define.h"
#include "DefineInterface_SITE.h"

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
enum ENUM_DEFECT_COLOR
{
	E_DEFECT_COLOR_DARK = 0,	//00黑暗不良
	E_DEFECT_COLOR_BRIGHT,	//01明亮的不良

	E_DEFECT_COLOR_COUNT				// 02 Total
};

//不等号运算符(<,>,==,<=,>=)
enum ENUM_SIGN_OF_INEQUALITY
{
	E_SIGN_EQUAL = 0,	// x == judgment value
	E_SIGN_NOT_EQUAL,	// x != judgment value
	E_SIGN_GREATER,	// x >  judgment value
	E_SIGN_LESS,	// x <  judgment value
	E_SIGN_GREATER_OR_EQUAL,	// x >= judgment value
	E_SIGN_LESS_OR_EQUAL						// x <= judgment value
};

//2017.06.08错误代码更改为AVI/SVI/APP公共
//错误代码返回值
enum ENUM_ERROR_CODE
{
	E_ERROR_CODE_FALSE = -1,	//异常-无实际实现函数

	//通用:000号台
	E_ERROR_CODE_TRUE = 000,	//正常

	E_ERROR_CODE_EMPTY_BUFFER = 001,	//如果没有画面缓冲区
	E_ERROR_CODE_EMPTY_PATH,	//如果没有保存路径。
	E_ERROR_CODE_EMPTY_PARA,	//如果没有检测参数。
	E_ERROR_CODE_EMPTY_SET_ROI,	//没有ROI设置。
	E_ERROR_CODE_IMAGE_CLASSIFY_OVER,	//如果画面号码不是0~9。
	E_ERROR_CODE_ROI_OVER,	//如果检查区域超出画面大小。
	E_ERROR_CODE_INCOMPATIBLE_BUFFER_TYPE,	//Mat缓冲区类型不同。
	E_ERROR_CODE_TIME_OUT,	//设置超时时
	E_ERROR_CODE_IMAGE_GRAY,	//如果不是Gray画面
	E_ERROR_CODE_IMAGE_COLOR,	//如果不是Color画面
	E_ERROR_CODE_IMAGE_SIZE,	//如果画面大小不同
	E_ERROR_CODE_EMPTY_RGB,	//R,G,B画面AD或无

	//Align:100号台
	E_ERROR_CODE_ALIGN_NO_DATA = 101,	//数据不足&如果没有数据。
	E_ERROR_CODE_ALIGN_WARNING_PARA,	//设置了无效参数。
	E_ERROR_CODE_ALIGN_NO_FIND_CELL,	//如果找不到Cell。
	E_ERROR_CODE_ALIGN_ANGLE_RANGE_ERROR,	//角度变大的情况。(ERROR-无法检查)
	E_ERROR_CODE_ALIGN_ANGLE_RANGE_WARNING,	//角度变大的情况。(WARNING-检查失败)
	E_ERROR_CODE_ALIGN_CAN_NOT_CALC,	//无法计算角度。
	E_ERROR_CODE_ALIGN_DISPLAY,	//显示以上。
	E_ERROR_CODE_ALIGN_IMAGE_OVER,	//影像坐标偏离。
	E_ERROR_CODE_ALIGN_WRONG_SLOPE,	//倾斜提取无效值(SVI)
	E_ERROR_CODE_ALIGN_ROUND_SETTING,	//不检查Round设置标志On
	E_ERROR_CODE_ALIGN_LENGTH_RANGE_ERROR,	//Align后不检查Pannel Size Range。

	//Point:200号台
	E_ERROR_CODE_POINT_DEFECT_NO = 201,	//不是积分不良的情况。
	E_ERROR_CODE_POINT_JUDEGEMENT_E,	//Judgement E级判定返回用
	E_ERROR_CODE_POINT_RES_VECTOR_NULL,	//积分不良结果向量为NULL。
	E_ERROR_CODE_POINT_NOT_USE_DUST_ALG,	//如果没有使用Dust算法。
	E_ERROR_CODE_POINT_WARNING_PARA,	//设置了无效参数。
	E_ERROR_CODE_POINT_TIME_OVER,	//如果计算时间和等待时间很长。(当前设置为10秒)

	//Line:300号台
	E_ERROR_CODE_LINE_HISTO = 301,	// 

	//Mura:400号台
	E_ERROR_CODE_MURA_WRONG_PARA = 401,	//Mura参数异常
	E_ERROR_CODE_MURA_HISTO,	//
	E_ERROR_CODE_MURA_RGB_LINE_DEFECT,	//检测出RGB Line Mura不良

	//SVI:500号台
	E_ERROR_CODE_SVI_EMD_ESTMATION = 501,	//找不到Cell点灯区域
	E_ERROR_CODE_SVI_WRONG_PARAM,

	// Empty
	E_ERROR_CODE_EMPTY = 600,	// 

	//APP:700号台
	E_ERROR_CODE_CUTTING_MARK_TYPE_PARA = 701,
	E_ERROR_CODE_PAD_TEACHING_ABNORMAL,	//PAD教学不正常。
	E_ERROR_CODE_PAD_MASK_REF_EMPTY,	//没有PAD MASK标准信息或画面。
	E_ERROR_CODE_PAD_REF_IMG_EMPTY_BUFFER,	//没有PAD Reference画面
	E_ERROR_CODE_PAD_MASK_SIZE,//PAD MASK Size或更高版本
	E_ERROR_CODE_MEASURE_BM_FITTING_FAIL,//进行测量时BM Line Fitting失败
	E_ERROR_CODE_MEASURE_PANEL_FITTING_FAIL,//进行测量时Panel Line Fitting失败
	E_ERROR_CODE_EMPTY_CROSS_MARK_BUFFER,//无十字标记图像
	E_ERROR_CODE_CROSS_MARK_SEARCH_FAIL,//十字标记查找失败
	E_ERROR_CODE_APP_HISTO,
	E_ERROR_CODE_CCUT_REF_IMG_BUFFER,//没有CCut Reference画面
	E_ERROR_CODE_CCUT_ROI_MATCHING_FAIL,//CCut ROI匹配失败(面板损坏或画面损坏)
	E_ERROR_CODE_PAD_PARAM_IS_NULL,//Pad Area中的参数不存在
	E_ERROR_CODE_PAD_INSP_ROI_OVER,//PAD INSP ROI Size问题
	E_ERROR_CODE_ACTIVE_MASK,//Active Mask有问题
	E_ERROR_CODE_ACTIVE_FEATURE_SIZE,//Active Feature Size问题

	//CCD Defect:800号台
	E_ERROR_CODE_CCD_EMPTY_BUFFER = 801,//没有画面缓冲区
	E_ERROR_CODE_CCD_PS_BUFFER,//如果是PS画面缓冲区
	E_ERROR_CODE_CCD_WARNING_PARA,//设置了错误的参数。
	E_ERROR_CODE_CCD_NOT_LOAD,//如果加载失败
	E_ERROR_CODE_CCD_EMPTY_FILE,//如果文件不存在


};

enum ENUM_PAD_AREA_NAME
{
	E_PAD_REF = 0,//垫片参考图像&坐标
	E_PAD_FIDUCIAL,//Fiducial Mark&定点图像和坐标
	E_PAD_ALIGN_MARK,//垫片十字标记图像和坐标
	E_PAD_INSP_AREA,//垫片扫描区域图像和坐标
	E_PAD_NONE_AREA,//垫片非扫描区域图像和坐标
	E_PAD_AREA_COUNT
};

//////////////////////////////////////////////////////////////////////////
//结构体
//////////////////////////////////////////////////////////////////////////

//要使用测量数据的结构2017.09.20
struct stMeasureInfo
{
	int		nMeasureValueSize;
	CString strPanelID;
	bool	bJudge;

	double* dMeasureValue;

	double  dCameraResolution;

	stMeasureInfo(int nMaxSize)
	{
		dMeasureValue = new double[nMaxSize];
		memset(dMeasureValue, 0, sizeof(double) * nMaxSize);
		nMeasureValueSize = nMaxSize;
		bJudge = true;
	}
	~stMeasureInfo()
	{
		delete[] dMeasureValue;
	}
};

struct stActiveBurntROI
{
	int	nMaxROINum;
	Rect* rtROI;

	stActiveBurntROI(int nROINum)
	{
		rtROI = new Rect[nROINum];
		nMaxROINum = nROINum;
	}
	~stActiveBurntROI()
	{
		delete[] rtROI;
	}
};

#pragma pack(push)
#pragma pack(1)

struct STRU_JUDGEMENT
{
	BOOL bUse;//是否选择判定项目
	int nSign;//运算符(<,>,==,<=,>=)
	double dValue;      // 价钱

	//初始化结构体
	struct STRU_JUDGEMENT()
	{
		memset(this, 0, sizeof(STRU_JUDGEMENT));
	}
};

struct STRU_DEFECT_ITEM
{

	BOOL bDefectItemUse;   // 是否使用算法
	TCHAR strItemName[50];

	STRU_JUDGEMENT Judgment[2 * MAX_MEM_SIZE_E_DEFECT_JUDGMENT_COUNT]; // 每个Defect判定项目2个范围

	STRU_DEFECT_ITEM()
	{
		memset(this, 0, sizeof(STRU_DEFECT_ITEM));
	}
};
//////////////////////////////////////////////
//st分区判定参数 [hjf]
struct stPanelBlockJudgeInfo {

	BOOL bBlockUse;
	UINT nBlockNum;

	STRU_DEFECT_ITEM stDefectItem[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];

	stPanelBlockJudgeInfo()
	{
		memset(this, 0, sizeof(stPanelBlockJudgeInfo));
	}

	~stPanelBlockJudgeInfo() {

	}
};


/////////////////////////////////////////////
struct STRU_PARAM_ALG
{
	BOOL bAlgorithmUse;   // 是否使用算法
	TCHAR strAlgName[50];
	double dPara[MAX_MEM_SIZE_ALG_PARA_TOTAL_COUNT];//算法参数
	UINT nBlockCountX;
	UINT nBlockCountY;
	UINT nDefectItemCount;
	STRU_DEFECT_ITEM stDefectItem[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];	// Defect Item
	stPanelBlockJudgeInfo stBlockDefectItem[MAX_MEM_SIZE_BLOCK_COUNT];//[hjf] Block Defect Item
	//vector<STRU_DEFECT_ITEM> stBlockDefectItem[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];
	STRU_PARAM_ALG()
	{
		nBlockCountX = 0;
		nBlockCountY = 0;
		memset(this, 0, sizeof(STRU_PARAM_ALG));
	}

};

struct STRU_INFO_ROI
{
	BOOL bUseROI;
	CRect rectROI;
	TCHAR strROIName[50];

	STRU_PARAM_ALG AlgorithmList[MAX_MEM_SIZE_E_ALGORITHM_NUMBER];   // 算法参数

	STRU_INFO_ROI()
	{
		memset(this, 0, sizeof(STRU_INFO_ROI));
	}
};

//17.10.30[Round]-Alg设置
struct ROUND_SET
{
	//17.10.24[Round]-拐角部分区域与Grab Step一样存在
	//基于最多30个Cell转角使用Offset
	cv::Point2i ptContours[MAX_MEM_SIZE_ROUND_COUNT];//设置区域的曲线坐标
	int			nContourCount;//设置区域的曲线坐标数
	int			nCornerInside[E_CORNER_END];//在设置区域的4个顶点中,是否存在于Cell区域内
	int			nCornerMinLength;//距离设置区域最近的拐角点[E_CORNER_END]
	//试图以E_CORNER_LEFT_TOP为基准,但越远越存在误差
	ROUND_SET()
	{
		//17.10.30[Round]-初始化
		memset(ptContours, 0, sizeof(cv::Point2i) * MAX_MEM_SIZE_ROUND_COUNT);
		memset(nCornerInside, 0, sizeof(int) * E_CORNER_END);
		nContourCount = 0;
		nCornerMinLength = 0;
	}
};

struct INSP_AREA
{
	BOOL bUseROI;
	CRect rectROI;
	TCHAR strROIName[50];

	INSP_AREA()
	{
		memset(this, 0, sizeof(INSP_AREA));
	}
};

struct STRU_INFO_CAMERA
{
	BOOL bUse;

	int nROICnt;
	STRU_INFO_ROI ROI[MAX_MEM_SIZE_ROI_COUNT];

	int nNonROICnt;
	INSP_AREA NonROI[MAX_MEM_SIZE_E_INSPECT_AREA];

	int nRndROICnt;
	INSP_AREA RndROI[MAX_MEM_SIZE_E_INSPECT_AREA];

	int nAutoRndROICnt;
	INSP_AREA AutoRndROI[MAX_MEM_SIZE_E_INSPECT_AREA];

	// PolMark ROI
	int nPolMarkROICnt;
	INSP_AREA PolMarkROI[MAX_MEM_SIZE_E_INSPECT_AREA];

	// Hole ROI
	int nHoleROICnt;
	INSP_AREA HoleROI[MAX_MEM_SIZE_E_INSPECT_AREA];

	BOOL bUseAD;
	double dADPara[MAX_MEM_SIZE_AD_PARA_TOTAL_COUNT];
	//Label ROI
	int nLabelROICnt;
	INSP_AREA LabelROI[MAX_MEM_SIZE_LABEL_COUNT];

	//Mark ROI
	int nMarkROICnt;
	INSP_AREA MarkROI[MAX_MEM_SIZE_MARK_COUNT];
	STRU_INFO_CAMERA()
	{
		memset(this, 0, sizeof(STRU_INFO_CAMERA));
	}
};

struct STRU_INFO_GRAB_STEP
{
	BOOL	bUse;
	TCHAR	strGrabStepName[50];
	int		eImgClassify;			// ENUM_IMAGE_CLASSIFY -> int
	int		nCamCnt;
	STRU_INFO_CAMERA stInfoCam[MAX_MEM_SIZE_CAM_COUNT];

	//17.10.30[Round]-Alg设置
	ROUND_SET	tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA];

	ROUND_SET	tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA];

	STRU_INFO_GRAB_STEP()
	{
		memset(this, 0, sizeof(STRU_INFO_GRAB_STEP));
	}
};

struct STRU_IMAGE_INFO
{
	CString	strPanelID;
	CString strLotID;
	int		nCameraNo;
	int		nImageNo;
	UINT	nRatio;//Pixel Shift与源的比率
	UINT	nInspType;
	int		nStageNo;
	double dPatternCIE[3];//MTP校正后[0]:X,[1]:Y,[2]:利用L检查

	STRU_IMAGE_INFO()
	{
		strPanelID = _T("");
		strLotID = _T("");
		nCameraNo = 0;
		nImageNo = 0;
		nRatio = 0;//Pixel Shift与源的比率
		nInspType = 0;
		nStageNo = 0;
		for (int i = 0; i < 3; i++)
			dPatternCIE[i] = 0.0;
	}
};

//公共结果结构体
struct stDefectInfo
{
	int			nImageNumber;//检测到的模式画面编号[ENUM_IMAGE_CLASSIFY]
	int			nCamNumber;//检测到的摄像头编号

	int			nDefectCount;//总不良数量
	int			nMaxDefect;//可存储的不良信息数量
	//新增缺陷分区
	int* nBlockNum;

	//不良分类
	int* nDefectJudge;//不良判定结果[ENUM_DEFECT_JUDGEMENT]
	int* nDefectColor;//背景对比不良颜色[ENUM_DEFECT_COLOR]
	int* nPatternClassify;//模式类型(R,G,B...)

	int* nArea;//不良面积
	POINT* ptLT;//错误rect Left-Top
	POINT* ptRT;//错误rect Right-Top
	POINT* ptRB;//错误rect Right-Bottom
	POINT* ptLB;//错误rect Left-Bottom
	double* dMeanGV;//平均亮度不良
	double* dSigma;//标准偏差
	int* nMinGV;//最低亮度不良
	int* nMaxGV;//最大亮度不良
	double* dBackGroundGV;//背景平均亮度
	int* nCenterx;//中心定x
	int* nCentery;//中心精y

	double* dBreadth;//不良厚度
	double* dCompactness;//不良原型
	double* dRoundness;//接近圆形的程度1是完全圆形
	double* dF_Elongation;//章,缩短率
	double* dF_Min;//缩短
	double* dF_Max;//长轴

	double* dMuraObj;//18.09.04-Mura对象亮度
	double* dMuraBk;//18.09.04-Mura背景亮度
	bool* bMuraActive;//18.09.04-Mura Active位置？
	bool* bMuraBright;//18.09.04-Mura里面有亮GV吗？

	double* dF_MeanAreaRatio;	// choikwangil

	double* dF_AreaPer;			// choikwangil 04.20
	int* nJudge_GV;		// choikwangil 04.20
	int* nIn_Count;			// choikwangil 04.20

	// Color ( SVI )
	double* Lab_avg_L;			// 	
	double* Lab_avg_a;			// 
	double* Lab_avg_b;			// 
	double* Lab_diff_L;			// 
	double* Lab_diff_a;			// 
	double* Lab_diff_b;			// 

	long		nMallocCount;
	//AI Re-Judge xb 2023/08/03
	int* AI_ReJudge_Code;
	double* AI_ReJudge_Conf;
	int* AI_ReJudge_Result;
	bool* AI_ReJudge;

	//AI Feature
	std::vector<double>		AI_Confidence;
	std::vector<int>		AI_CODE;
	bool		From_AI{ false };
	int			AI_Object_Nums{ 0 };

#if USE_ALG_HIST
	//17.06.24对象直方图
	__int64** nHist;				//
#endif

#if USE_ALG_CONTOURS
	//17.11.29-外围信息(AVI&SVI其他工具)
	int** nContoursX;
	int** nContoursY;
#endif

	bool* bUseResult;//使用最终不良结果有/无

	//初始化结构体
	stDefectInfo(int MaxDefect, int ImageNumber = -1, int CamNumber = -1)
	{
		nImageNumber = ImageNumber;
		nCamNumber = CamNumber;

		nMaxDefect = MaxDefect;
		nDefectCount = 0;

		nBlockNum = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nBlockNum, 0, sizeof(int) * MaxDefect);

		nDefectJudge = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nDefectJudge, 0, sizeof(int) * MaxDefect);

		nDefectColor = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nDefectColor, 0, sizeof(int) * MaxDefect);

		nPatternClassify = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nPatternClassify, 0, sizeof(int) * MaxDefect);

		nArea = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nArea, 0, sizeof(int) * MaxDefect);

		ptLT = (POINT*)malloc(sizeof(POINT) * MaxDefect);
		ptRT = (POINT*)malloc(sizeof(POINT) * MaxDefect);
		ptRB = (POINT*)malloc(sizeof(POINT) * MaxDefect);
		ptLB = (POINT*)malloc(sizeof(POINT) * MaxDefect);

		dMeanGV = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dMeanGV, 0, sizeof(double) * MaxDefect);

		dSigma = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dSigma, 0, sizeof(double) * MaxDefect);

		nMinGV = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nMinGV, 0, sizeof(int) * MaxDefect);

		nMaxGV = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nMaxGV, 0, sizeof(int) * MaxDefect);

		dBackGroundGV = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dBackGroundGV, 0, sizeof(double) * MaxDefect);

		nCenterx = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nCenterx, 0, sizeof(int) * MaxDefect);

		nCentery = (int*)malloc(sizeof(int) * MaxDefect);
		memset(nCentery, 0, sizeof(int) * MaxDefect);

		dBreadth = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dBreadth, 0, sizeof(double) * MaxDefect);

		dCompactness = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dCompactness, 0, sizeof(double) * MaxDefect);

		dRoundness = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dRoundness, 0, sizeof(double) * MaxDefect);

		dF_Elongation = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dF_Elongation, 0, sizeof(double) * MaxDefect);

		dF_Min = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dF_Min, 0, sizeof(double) * MaxDefect);

		dF_Max = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dF_Max, 0, sizeof(double) * MaxDefect);

		dMuraObj = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dMuraObj, 0, sizeof(double) * MaxDefect);

		dMuraBk = (double*)malloc(sizeof(double) * MaxDefect);
		memset(dMuraBk, 0, sizeof(double) * MaxDefect);

		bMuraActive = (bool*)malloc(sizeof(bool) * MaxDefect);
		memset(bMuraActive, 0, sizeof(bool) * MaxDefect);

		bMuraBright = (bool*)malloc(sizeof(bool) * MaxDefect);
		memset(bMuraBright, 0, sizeof(bool) * MaxDefect);

		Lab_avg_L = (double*)malloc(sizeof(double) * MaxDefect);
		memset(Lab_avg_L, 0, sizeof(double) * MaxDefect);

		Lab_avg_a = (double*)malloc(sizeof(double) * MaxDefect);
		memset(Lab_avg_a, 0, sizeof(double) * MaxDefect);

		Lab_avg_b = (double*)malloc(sizeof(double) * MaxDefect);
		memset(Lab_avg_b, 0, sizeof(double) * MaxDefect);

		Lab_diff_L = (double*)malloc(sizeof(double) * MaxDefect);
		memset(Lab_diff_L, 0, sizeof(double) * MaxDefect);

		Lab_diff_a = (double*)malloc(sizeof(double) * MaxDefect);
		memset(Lab_diff_a, 0, sizeof(double) * MaxDefect);

		Lab_diff_b = (double*)malloc(sizeof(double) * MaxDefect);
		memset(Lab_diff_b, 0, sizeof(double) * MaxDefect);

		dF_MeanAreaRatio = (double*)malloc(sizeof(double) * MaxDefect); //choikwangil
		memset(dF_MeanAreaRatio, 0, sizeof(double) * MaxDefect);

		dF_AreaPer = (double*)malloc(sizeof(double) * MaxDefect); //choikwangil 04.20
		memset(dF_AreaPer, 0, sizeof(double) * MaxDefect);

		nJudge_GV = (int*)malloc(sizeof(int) * MaxDefect); //choikwangil 04.20
		memset(nJudge_GV, 0, sizeof(int) * MaxDefect);

		nIn_Count = (int*)malloc(sizeof(int) * MaxDefect); //choikwangil 04.20
		memset(nIn_Count, 0, sizeof(int) * MaxDefect);

		//AI_ReJudge xb 2023/08/03
		AI_ReJudge_Code = (int*)malloc(sizeof(int) * MaxDefect);
		memset(AI_ReJudge_Code, 0, sizeof(int) * MaxDefect);

		AI_ReJudge_Conf = (double*)malloc(sizeof(double) * MaxDefect);
		memset(AI_ReJudge_Conf, 0, sizeof(double) * MaxDefect);

		AI_ReJudge_Result = (int*)malloc(sizeof(int) * MaxDefect);
		memset(AI_ReJudge_Result, 0, sizeof(int) * MaxDefect);

		AI_ReJudge = (bool*)malloc(sizeof(bool) * MaxDefect);
		memset(AI_ReJudge, 0, sizeof(bool) * MaxDefect);

		nMallocCount = MaxDefect;

#if USE_ALG_HIST
		//17.06.24对象直方图
		nHist = (__int64**)malloc(sizeof(__int64*) * MaxDefect);
		for (int m = 0; m < nMallocCount; m++)
		{
			nHist[m] = (__int64*)malloc(sizeof(__int64) * IMAGE_MAX_GV);
			memset(nHist[m], 0, sizeof(__int64) * IMAGE_MAX_GV);
		}
#endif

#if USE_ALG_CONTOURS
		//17.11.29-外围信息(AVI&SVI其他工具)
		nContoursX = (int**)malloc(sizeof(int*) * MaxDefect);
		nContoursY = (int**)malloc(sizeof(int*) * MaxDefect);
		for (int m = 0; m < nMallocCount; m++)
		{
			nContoursX[m] = (int*)malloc(sizeof(int) * MAX_CONTOURS);
			nContoursY[m] = (int*)malloc(sizeof(int) * MAX_CONTOURS);

			memset(nContoursX[m], 0, sizeof(int) * MAX_CONTOURS);
			memset(nContoursY[m], 0, sizeof(int) * MAX_CONTOURS);
		}
#endif

		bUseResult = (bool*)malloc(sizeof(bool) * MaxDefect);
		for (int nIndex = 0; nIndex < MaxDefect; nIndex++) {
			bUseResult[nIndex] = true;
			AI_ReJudge_Code[nIndex] = -1;
			AI_ReJudge_Result[nIndex] = -1;
		}
	}

	//禁用内存	
	~stDefectInfo()
	{
		free(nBlockNum);	nBlockNum = NULL;
		free(nDefectJudge);	nDefectJudge = NULL;
		free(nDefectColor);	nDefectColor = NULL;
		free(nPatternClassify);	nPatternClassify = NULL;

		free(nArea);	nArea = NULL;
		free(ptLT);	ptLT = NULL;
		free(ptRT);	ptRT = NULL;
		free(ptRB);	ptRB = NULL;
		free(ptLB);	ptLB = NULL;
		free(dMeanGV);	dMeanGV = NULL;
		free(dSigma);	dSigma = NULL;
		free(nMinGV);	nMinGV = NULL;
		free(nMaxGV);	nMaxGV = NULL;
		free(dBackGroundGV);	dBackGroundGV = NULL;
		free(nCenterx);	nCenterx = NULL;
		free(nCentery);	nCentery = NULL;

		free(dBreadth);	dBreadth = NULL;
		free(dCompactness);	dCompactness = NULL;
		free(dRoundness);  dRoundness = NULL;
		free(dF_Elongation);	dF_Elongation = NULL;
		free(dF_Min);	dF_Min = NULL;
		free(dF_Max);	dF_Max = NULL;

		free(dMuraObj);				dMuraObj = NULL;
		free(dMuraBk);				dMuraBk = NULL;
		free(bMuraActive);			bMuraActive = NULL;
		free(bMuraBright);			bMuraBright = NULL;

		free(Lab_avg_L);	Lab_avg_L = NULL;
		free(Lab_avg_a);	Lab_avg_a = NULL;
		free(Lab_avg_b);	Lab_avg_b = NULL;
		free(Lab_diff_L);	Lab_diff_L = NULL;
		free(Lab_diff_a);	Lab_diff_a = NULL;
		free(Lab_diff_b);	Lab_diff_b = NULL;

		free(dF_MeanAreaRatio);	dF_MeanAreaRatio = NULL; //choikwangil

		free(dF_AreaPer);	dF_AreaPer = NULL; //choikwangil 04.20
		free(nJudge_GV);	nJudge_GV = NULL; //choikwangil 04.20
		free(nIn_Count);	nIn_Count = NULL; //choikwangil 04.20

		free(AI_ReJudge_Code);		AI_ReJudge_Code = NULL; //xb 2023/08/03
		free(AI_ReJudge_Conf);		AI_ReJudge_Conf = NULL;
		free(AI_ReJudge_Result);	AI_ReJudge_Result = NULL;
		free(AI_ReJudge);			AI_ReJudge = NULL;

		free(bUseResult);	bUseResult = NULL;

#if USE_ALG_HIST
		//17.06.24对象直方图
		for (int m = 0; m < nMallocCount; m++)
		{
			free(nHist[m]);
			nHist[m] = NULL;
		}
		free(nHist);
		nHist = NULL;
#endif

#if USE_ALG_CONTOURS
		//17.11.29-外围信息(AVI&SVI其他工具)
		for (int m = 0; m < nMallocCount; m++)
		{
			free(nContoursX[m]);
			free(nContoursY[m]);

			nContoursX[m] = NULL;
			nContoursY[m] = NULL;
		}
		free(nContoursX);
		free(nContoursY);
		nContoursX = NULL;
		nContoursY = NULL;
#endif
	}
};

//面板判定标准值/符号
struct stPanelJudgeInfo
{
	int nRefVal;
	int nSign;

	stPanelJudgeInfo()
	{
		nRefVal = 0;
		nSign = 0;
	}
};

//判定优先级。索引越低,优先级越高
struct stPanelJudgePriority
{
	TCHAR strGrade[50];
	stPanelJudgeInfo stJudgeInfo[E_PANEL_DEFECT_TREND_COUNT];
	stPanelJudgeInfo stFilterInfo[E_PANEL_DEFECT_TREND_COUNT];
	stPanelJudgePriority()
	{
		memset(this, 0, sizeof(stPanelJudgePriority));
	}
};

//父报告筛选条件
struct stReportFilter
{
	BOOL bUse;//是否启用功能
	stPanelJudgeInfo stJudgeInfo[E_PANEL_DEFECT_TREND_COUNT];

	stReportFilter()
	{
		bUse = false;
		memset(stJudgeInfo, 0, sizeof(stPanelJudgeInfo) * E_PANEL_DEFECT_TREND_COUNT);
	}
};


//父报告筛选条件
struct stUserDefinedFilter
{
	BOOL bUse;//是否启用功能
	TCHAR strGrade[10];//应用过滤器的Grade
	int nFilterItemsCount;//Defect Num个ex)1+4个面
	int nFilterItems[100];//使用过滤器的DEFECT_NUM,ex)5+6

	stPanelJudgeInfo stFilterInfo;

	stUserDefinedFilter()
	{
		bUse = false;
		nFilterItemsCount = 0;
		memset(nFilterItems, 0, sizeof(int) * 100);
		memset(this, 0, sizeof(stUserDefinedFilter));
	}
};

struct stDefClassification
{
	TCHAR strDefectName[50];
	TCHAR strDefectCode[10];

	stDefClassification()
	{
		memset(this, 0, sizeof(stDefClassification));
	}
};

struct STRU_PAD_AREA
{
	int			 nPolygonCnt;
	IplImage* ipImg;
	cv::Rect	 cvRect;
	cv::Point* cvPoint;

	STRU_PAD_AREA()
	{
		nPolygonCnt = 0;
		cvPoint = new Point[1];
	}
	void Point_malloc(int nPntCnt = 1000)
	{
		nPolygonCnt = nPntCnt;
		cvPoint = new Point[nPntCnt];
	}
	IplImage* GetImage()
	{
		return ipImg;
	}
	Rect GetRectCoord()
	{
		return cvRect;
	}
	Point* GetPolygonCoord()
	{
		return cvPoint;
	}
	int GetPolygonCount()
	{
		return nPolygonCnt;
		return (int)_msize(cvPoint) / sizeof(*cvPoint);
	}
	~STRU_PAD_AREA()
	{
		delete[] cvPoint; cvPoint = NULL;
	}
};

struct STRU_INFO_PAD
{
	int nRectCnt;
	STRU_PAD_AREA* tPadInfo;

	STRU_INFO_PAD()
	{
		tPadInfo = new STRU_PAD_AREA[1];
	}
	void _malloc(int nMaxCnt = 100)
	{
		nRectCnt = nMaxCnt;
		tPadInfo = new STRU_PAD_AREA[nMaxCnt];
	}
	int GetRoiCount()
	{
		return nRectCnt;
		return (int)_msize(this->tPadInfo) / sizeof(*this->tPadInfo);
	}
	Rect GetRectCoord(int nRoiCnt)
	{
		return tPadInfo[nRoiCnt].GetRectCoord();
	}
	Point* GetPolygonCoord(int nRoiCnt)
	{
		return this->tPadInfo[nRoiCnt].GetPolygonCoord();
	}
	int GetPolygonCount(int nRoiCnt)
	{
		return this->tPadInfo[nRoiCnt].GetPolygonCount();
	}
	IplImage* GetImage(int nRoiCnt)
	{
		return tPadInfo[nRoiCnt].GetImage();
	}
	void ReleaseImage()
	{
		for (int nRoiCnt = 0; nRoiCnt < GetRoiCount(); nRoiCnt++)
			cvReleaseImage(&tPadInfo[nRoiCnt].ipImg);
	}
	~STRU_INFO_PAD()
	{
		ReleaseImage();
		delete[] tPadInfo; tPadInfo = NULL;
	}
};

struct STRU_LabelMarkInfo
{
	bool bFindEnd;
	cv::Mat labelMask;		// 标签掩码图
	cv::Mat polNumMask;
	cv::Mat polSignMask;

	cv::Rect labelMaskBBox;
	cv::Rect polNumBBox;
	cv::Rect polSignBBox;

	void Reset() {
		bFindEnd = false;
		labelMask = cv::Mat();
		polNumMask = cv::Mat();
		polSignMask = cv::Mat();
	}
};

struct STRU_LabelMarkParams
{
	bool bUseLabel = false;
	bool bUsePolNum = false;
	bool bUsePolSign = false;
	bool bSaveTemplate = false;

	int polNumID = 0;
	int polSignID = 0;

	double labelThreshold = 50;
	double polNumThreshold = 45;
	double polSignThreshold = 54;

	double labelWidth = 900;
	double labelHeight = 2000;

	double polNumMinWidth = 220;
	double polNumMaxWidth = 300;
	double polNumMinHeight = 18;
	double polNumMaxHeight = 250;

	double polSignMinWidth = 150;
	double polSignMaxWidth = 250;
	double polSignMinHeight = 150;
	double polSignMaxHeight = 250;

	double polCellSize = 4;

	CString modelPath;

	cv::Rect cellBBox;
	cv::Rect polNumROI;		// 前端设置的polNum区域，在该区域内查找PolNum
	cv::Rect polSignROI;	// 前端设置的polSign区域，在该区域内查找PolSign

	std::map<std::string, cv::Mat>* polNumTemplates;
	std::map<std::string, cv::Mat>* polSignTemplates;;

	STRU_LabelMarkParams(double* alignParams, TCHAR* modelPth) {
		bUseLabel = alignParams[E_PARA_AVI_Label_Flag];
		bUsePolNum = alignParams[E_PARA_AVI_PolNum_Flag];
		bUsePolSign = alignParams[E_PARA_AVI_PolSign_Flag];
		bSaveTemplate = alignParams[E_PARA_AVI_Pol_Save_Template];
		polNumID = alignParams[E_PARA_AVI_PolNum_ID];
		polSignID = alignParams[E_PARA_AVI_PolSign_ID];

		labelWidth = alignParams[E_PARA_AVI_Label_Width];
		labelHeight = alignParams[E_PARA_AVI_Label_Height];
		polNumMinWidth = alignParams[E_PARA_AVI_PolNum_MinWidth];
		polNumMaxWidth = alignParams[E_PARA_AVI_PolNum_MaxWidth];
		polNumMinHeight = alignParams[E_PARA_AVI_PolNum_MinHeight];
		polNumMaxHeight = alignParams[E_PARA_AVI_PolNum_MaxHeight];

		polSignMinWidth = alignParams[E_PARA_AVI_PolSign_MinWidth];
		polSignMaxWidth = alignParams[E_PARA_AVI_PolSign_MaxWidth];
		polSignMinHeight = alignParams[E_PARA_AVI_PolSign_MinHeight];
		polSignMaxHeight = alignParams[E_PARA_AVI_PolSign_MaxHeight];

		labelThreshold = alignParams[E_PARA_AVI_Label_Threshold];
		polNumThreshold = alignParams[E_PARA_AVI_PolNum_Threshold];
		polSignThreshold = alignParams[E_PARA_AVI_PolSign_Threshold];

		polCellSize = alignParams[E_PARA_AVI_PolCell_Size];
		modelPath = modelPth;
	}
};

#pragma pack(pop)

#endif