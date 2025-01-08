#if !defined(AFX_INSPTHRD_H__1C81645D_D41C_4CB4_AD97_DDAAEAB90337__INCLUDED_)
#define AFX_INSPTHRD_H__1C81645D_D41C_4CB4_AD97_DDAAEAB90337__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// InspThrd.h : header file
//

#include "InspResultInfo.h"

/**
 * 保存图片信息.
 */
struct tImageInfo
{
	int stationNo;
	int imageNo;
	CString panelId;
	CString filePath;

	tImageInfo(int _stationNo, int _imageNo, const CString& _panelId, const CString& _filePath)
	{
		stationNo = _stationNo;
		imageNo = _imageNo;
		panelId = _panelId;
		filePath = _filePath;
	}
	tImageInfo() {}
};

//关于CHole Align
struct tCHoleAlignInfo
{
	bool			bCHoleAD[E_IMAGE_CLASSIFY_AVI_COUNT][MAX_MEM_SIZE_E_INSPECT_AREA];					//CHole区域AD解析
	cv::Mat			matCHoleROIBuf[E_IMAGE_CLASSIFY_AVI_COUNT][MAX_MEM_SIZE_E_INSPECT_AREA];			//White Pattern基准坐标影像(非AD时的影像)
	cv::Rect		rcCHoleROI[E_IMAGE_CLASSIFY_AVI_COUNT][MAX_MEM_SIZE_E_INSPECT_AREA];				//CHole ROI(AD时为AD区域,非AD时为CHole坐标区域)
};

//关于Panel Align
struct tAlignInfo
{
	cv::Point		ptCorner[4];				//(通用)原始Cell最外围的转角点
	cv::Point       ptContCorner[4];
	int				nRatio;						//(通用)Align有无画面P/S模式/17.08.29
	cv::Rect		rcAlignCellROI;				//(通用)Edge ROI
	cv::Rect		rcAlignPadROI;				//(通用)Pad ROI
	cv::Rect		rcAlignActiveROI;			//(通用)主动ROI
	double			dAlignTheta;				//旋转角度
	cv::Point		ptAlignCenter;				//旋转时,中心
	cv::Point		ptCellCenter;				//画面中以Cell为中心
	cv::Point		ptStandard;					//Left-Top基点
	bool			bAlignSuccess;				//AD-Align成功与否
	int				nStageNo;					//舞台编号
	vector<vector<vector<Point2i>>> ptRoundContour;
	vector<vector<Point2i>> ptActiveContour;
	cv::Rect		rcActive;					//APP Active ROI通用Align Active ROI值因时间点问题不断更改(由UI管理)。
	cv::Rect        rcMarkROI[MAX_MEM_SIZE_MARK_COUNT]; //Mark
	//关于CHole Align
	tCHoleAlignInfo* tCHoleAlignData;

	tAlignInfo()
	{
		// Init Align Parameter
		for (int i = 0; i < 4; i++)
		{
			ptCorner[i] = cv::Point(0, 0);
			ptContCorner[i] = cv::Point(0, 0);
		}
		rcAlignCellROI = cv::Rect(0, 0, 0, 0);
		rcAlignPadROI = cv::Rect(0, 0, 0, 0);
		rcAlignActiveROI = cv::Rect(0, 0, 0, 0);
		dAlignTheta = 0.0;
		ptAlignCenter = cv::Point(0, 0);
		ptCellCenter = cv::Point(0, 0);
		ptStandard = cv::Point(0, 0);
		bAlignSuccess = false;
		nRatio = 0;
		nStageNo = 0;
		vector<vector<vector<Point2i>>>().swap(ptRoundContour);
		vector<vector<Point2i>>().swap(ptActiveContour);
		rcActive = cv::Rect(0, 0, 0, 0);

		tCHoleAlignData = NULL;
	}

	void SetAdjustAlignInfoRatio(tAlignInfo* pStAlignInfo, double dRatio, bool bSuccess = true)
	{
		ptAlignCenter.x = (int)(pStAlignInfo->ptAlignCenter.x * dRatio);
		ptAlignCenter.y = (int)(pStAlignInfo->ptAlignCenter.y * dRatio);
		ptCellCenter.x = (int)(pStAlignInfo->ptCellCenter.x * dRatio);
		ptCellCenter.y = (int)(pStAlignInfo->ptCellCenter.y * dRatio);
		ptStandard.x = (int)(pStAlignInfo->ptStandard.x * dRatio);
		ptStandard.y = (int)(pStAlignInfo->ptStandard.y * dRatio);

		nRatio = (int)dRatio;

		dAlignTheta = pStAlignInfo->dAlignTheta;

		for (int nPoint = E_CORNER_LEFT_TOP; nPoint <= E_CORNER_LEFT_BOTTOM; nPoint++)
		{
			ptCorner[nPoint].x = (int)(pStAlignInfo->ptCorner[nPoint].x * dRatio);
			ptCorner[nPoint].y = (int)(pStAlignInfo->ptCorner[nPoint].y * dRatio);
		}

		// Cell区轮廓四个角点 2024.07
		for (int nPoint = E_CORNER_LEFT_TOP; nPoint <= E_CORNER_LEFT_BOTTOM; nPoint++)
		{
			ptContCorner[nPoint].x = (int)(pStAlignInfo->ptContCorner[nPoint].x * dRatio);
			ptContCorner[nPoint].y = (int)(pStAlignInfo->ptContCorner[nPoint].y * dRatio);
		}

		rcAlignCellROI.x = (int)(pStAlignInfo->rcAlignCellROI.x * dRatio);
		rcAlignCellROI.y = (int)(pStAlignInfo->rcAlignCellROI.y * dRatio);
		rcAlignCellROI.width = (int)(pStAlignInfo->rcAlignCellROI.width * dRatio);
		rcAlignCellROI.height = (int)(pStAlignInfo->rcAlignCellROI.height * dRatio);

		rcAlignActiveROI.x = (int)(pStAlignInfo->rcAlignActiveROI.x * dRatio);
		rcAlignActiveROI.y = (int)(pStAlignInfo->rcAlignActiveROI.y * dRatio);
		rcAlignActiveROI.width = (int)(pStAlignInfo->rcAlignActiveROI.width * dRatio);
		rcAlignActiveROI.height = (int)(pStAlignInfo->rcAlignActiveROI.height * dRatio);

		rcAlignPadROI.x = (int)(pStAlignInfo->rcAlignPadROI.x * dRatio);
		rcAlignPadROI.y = (int)(pStAlignInfo->rcAlignPadROI.y * dRatio);
		rcAlignPadROI.width = (int)(pStAlignInfo->rcAlignPadROI.width * dRatio);
		rcAlignPadROI.height = (int)(pStAlignInfo->rcAlignPadROI.height * dRatio);

		bAlignSuccess = bSuccess;
		ptActiveContour = pStAlignInfo->ptActiveContour;
		rcActive = pStAlignInfo->rcActive;
	}

	void SetAdjustAlignInfo(cv::Point* ptAdjCorner, double* dAlignPara, int nMaxWidth, int nMaxHeight)
	{
		ptStandard.x = ptAdjCorner[E_CORNER_LEFT_TOP].x;
		ptStandard.y = ptAdjCorner[E_CORNER_LEFT_TOP].y;

		// 2017.06.06 NDH : APP ROI Setting
		cv::Rect rtCheck(0, 0, nMaxWidth, nMaxHeight);		//图像区域
		cv::Rect rtEdge, rtActive, rtPAD;

		rtEdge.x = ptAdjCorner[E_CORNER_LEFT_TOP].x;
		rtEdge.y = ptAdjCorner[E_CORNER_LEFT_TOP].y;

		rtEdge.width = ptAdjCorner[E_CORNER_RIGHT_BOTTOM].x - ptAdjCorner[E_CORNER_LEFT_TOP].x;
		rtEdge.height = ptAdjCorner[E_CORNER_RIGHT_BOTTOM].y - ptAdjCorner[E_CORNER_LEFT_TOP].y;

		GetCheckROIOver(rtEdge, rtCheck, rcAlignCellROI);

		// active Setting
		int nActiveLeft = (int)dAlignPara[E_PARA_ALIGN_AREA_ACTIVE_L];
		int nActiveRight = (int)dAlignPara[E_PARA_ALIGN_AREA_ACTIVE_R];
		int nActiveTop = (int)dAlignPara[E_PARA_ALIGN_AREA_ACTIVE_T];
		int nActiveBottom = (int)dAlignPara[E_PARA_ALIGN_AREA_ACTIVE_B];

		rtActive.x = (ptAdjCorner[E_CORNER_LEFT_TOP].x + ptAdjCorner[E_CORNER_LEFT_BOTTOM].x) / 2 + nActiveLeft;
		rtActive.width = (((ptAdjCorner[E_CORNER_RIGHT_TOP].x + ptAdjCorner[E_CORNER_RIGHT_BOTTOM].x) / 2 - nActiveRight) - rtActive.x);

		rtActive.y = (ptAdjCorner[E_CORNER_LEFT_TOP].y + ptAdjCorner[E_CORNER_RIGHT_TOP].y) / 2 + nActiveTop;
		rtActive.height = (((ptAdjCorner[E_CORNER_LEFT_BOTTOM].y + ptAdjCorner[E_CORNER_RIGHT_BOTTOM].y) / 2 - nActiveBottom) - rtActive.y);

		GetCheckROIOver(rtActive, rtCheck, rcAlignActiveROI);

		// Pad Setting
		int nPadWidth = (int)dAlignPara[E_PARA_ALIGN_AREA_PAD_WIDTH];
		int nPadLocation = (int)dAlignPara[E_PARA_ALIGN_AREA_PAD_LOCATION];

		switch (nPadLocation)
		{
		case E_PAD_LEFT:
		{
			rtPAD.x = (ptAdjCorner[E_CORNER_LEFT_TOP].x + ptAdjCorner[E_CORNER_LEFT_BOTTOM].x) / 2;
			rtPAD.width = nPadWidth;
			rtPAD.y = ptAdjCorner[E_CORNER_LEFT_TOP].y;
			rtPAD.height = ptAdjCorner[E_CORNER_LEFT_BOTTOM].y - rtPAD.y;
		}
		break;

		case E_PAD_TOP:
		{
			rtPAD.x = ptAdjCorner[E_CORNER_LEFT_TOP].x;
			rtPAD.width = ptAdjCorner[E_CORNER_RIGHT_TOP].x - rtPAD.x;
			rtPAD.y = (ptAdjCorner[E_CORNER_LEFT_TOP].y + ptAdjCorner[E_CORNER_RIGHT_TOP].y) / 2;
			rtPAD.height = nPadWidth;
		}
		break;

		case E_PAD_RIGHT:
		{
			rtPAD.x = (ptAdjCorner[E_CORNER_RIGHT_TOP].x + ptAdjCorner[E_CORNER_RIGHT_BOTTOM].x) / 2 - nPadWidth;
			rtPAD.width = nPadWidth;
			rtPAD.y = ptAdjCorner[E_CORNER_RIGHT_TOP].y;
			rtPAD.height = ptAdjCorner[E_CORNER_RIGHT_BOTTOM].y - rtPAD.y;
		}
		break;

		case E_PAD_BOTTOM:
		{
			rtPAD.x = ptAdjCorner[E_CORNER_LEFT_BOTTOM].x;
			rtPAD.width = ptAdjCorner[E_CORNER_RIGHT_BOTTOM].x - rtPAD.x;
			rtPAD.y = (ptAdjCorner[E_CORNER_LEFT_BOTTOM].y + ptAdjCorner[E_CORNER_RIGHT_BOTTOM].y) / 2 - nPadWidth;
			rtPAD.height = nPadWidth;
		}
		break;
		}

		GetCheckROIOver(rtPAD, rtCheck, rcAlignPadROI);
	}

	void SetRoundContour(vector<vector<Point2i>> ptContour)
	{
		ptRoundContour.push_back(vector<vector<Point2i>>(ptContour));
	}

	void SetRoundContour(tAlignInfo* pStAlignInfo)
	{
		ptRoundContour = pStAlignInfo->ptRoundContour;
	}
};

struct tInspectThreadParam
{
	///公用
	cv::Mat				MatOrgImage[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT];		//用于保存原始图像		
	cv::Mat				MatOrgImage2[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT];
	cv::Mat				MatDraw[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT];			//最终结果图像
	cv::Mat				MatResult[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE];		// [2:Dark/Bright]	
	bool				bInspectEnd[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT];		//检查结束标志
	bool				bAlignEnd[MAX_CAMERA_COUNT];							//虚线结束标志
	ResultPanelData		ResultPanelData;										//面板数据
	tAlignInfo			stCamAlignInfo[MAX_CAMERA_COUNT];						//关于每个相机的虚线
	CWriteResultInfo	WrtResultInfo;											//工作坐标计算/结果数据生成类
	bool				bUseInspect;											//检查是否正在进行(Grab Only功能)
	ENUM_INSPECT_MODE	eInspMode;
	bool				bHeavyAlarm;											//中是否出现警报。中出现提醒时,所有检查Skip
	///AVI专用
	bool				bChkDustEnd;											//Dust灯光状态检查结束标志
	bool				bIsNormalDust;											//Dust照明状态(true:正常点灯)
	bool				bUseDustRetry;											//是否使用Dust Retry
	int					nDustRetryCnt;											//Dust Retry次数
	TCHAR				strSaveDrive[20];										//image,result存储驱动器
	CString				strImagePath[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT];	//原始画面路径,2020.07.22,为原始画面异步处理而创建
	tImageInfo          stImageInfo[MAX_GRAB_STEP_COUNT][MAX_CAMERA_COUNT];

	tCHoleAlignInfo		tCHoleAlignData;										//关于CHole Align
	STRU_LabelMarkInfo  tLabelMarkInfo;											// LabelMark 信息 2024.7

	bool				bParamUse;												//是否使用参数

	tInspectThreadParam()
	{
		clear();	//20.07.03
	}

	void clear()	//20.07.03
	{

		////////////////////////////////////////////////////////////////////////// 21.05.27 choi
// 		for (int i = 0; i < MAX_GRAB_STEP_COUNT; i++)
// 		{
// 			for (int j = 0; j < MAX_CAMERA_COUNT; j++)
// 			{
// 				MatOrgImage[i][j].release();
// 				MatDraw[i][j].release();
// 
// 				for (int u = 0; u < MAX_MEM_SIZE_E_ALGORITHM_NUMBER; u++)
// 				{
// 					for (int k = 0; k < MAX_MEM_SIZE_E_MAX_INSP_TYPE; k++)
// 					{
// 						MatResult[i][j][u][k].release();
// 					}
// 				}
// 			}
// 		}

		//////////////////////////////////////////////////////////////////////////
				//禁用使用参数标志...
		bParamUse = false;

		//bInspectEnd初始化
		memset(bInspectEnd, true, sizeof(bInspectEnd));

		//bAlignEnd初始化
		memset(bAlignEnd, true, sizeof(bAlignEnd));

		bChkDustEnd = true;
		bIsNormalDust = false;
		bUseInspect = true;
		bHeavyAlarm = false;
		eInspMode = eAutoRun;
		bUseDustRetry = false;
		nDustRetryCnt = 0;
		memset(strSaveDrive, 0, sizeof(strSaveDrive));

		//ResultPanelData初始化
		ResultPanelData.m_ResultHeader = ResultHeaderInfo();
		ResultPanelData.m_ResultPanel = ResultPanelInfo();
		memset(&ResultPanelData.CornerPt, 0, sizeof(struct CPoint));
		memset(ResultPanelData.m_nDefectTrend, 0, sizeof(ResultPanelData.m_nDefectTrend));
		ResultPanelData.m_ListDefectInfo.RemoveAll();

		//tAlignInfo初始化
		for (int i = 0; i < MAX_CAMERA_COUNT; i++)
			stCamAlignInfo[i] = tAlignInfo();

		//WrtResultInfo初始化
		WrtResultInfo = CWriteResultInfo();

		//初始化tCHoleAlignData
		for (int i = 0; i < E_IMAGE_CLASSIFY_AVI_COUNT; i++)
		{
			for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
			{
				tCHoleAlignData.bCHoleAD[i][j] = false;
				tCHoleAlignData.rcCHoleROI[i][j] = cv::Rect(0, 0, 0, 0);
			}
		}
	}
};

/////////////////////////////////////////////////////////////////////////////
// CInspThrd thread
class CInspThrd : public CWinThread
{
	DECLARE_DYNCREATE(CInspThrd)

protected:
	CInspThrd();           // protected constructor used by dynamic creation

	bool		m_bBusy;				//验证线程是否正在进行检查序列
	tAlignInfo	m_stThrdAlignInfo;
	WPARAM wp;
	LPARAM lp;
	// Attributes
public:
	// thread info
	int m_nThrdID;

	// Operations
public:
	////////////////
	virtual void Initialize(int nThreadCount);

	////////////////

	bool IsThrdBusy();

public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();
	//}}AFX_VIRTUAL

public:
	virtual ~CInspThrd();

	// Generated message map functions
	//{{AFX_MSG(CInspThrd)
	//}}AFX_MSG
	virtual afx_msg LRESULT OnStartInspection(WPARAM wp, LPARAM lp);	//2016.10.17

	virtual afx_msg LRESULT OnStartSaveImage(WPARAM wp, LPARAM lp);		//2020.07.22
	//检查算法线程超时 hjf
public:
	virtual void SetTimeout(DWORD timeout); // 设置超时时间
	virtual int GetTimeout(); // 获取超时时间

	virtual void SetAlgoParam(WPARAM wp, LPARAM lp);
	virtual WPARAM GetAlgoWParam();

	virtual LPARAM GetAlgoLParam();

	virtual bool IsThreadTimeout() const;   // 检查线程是否超时
	virtual int ThreadTime() const;
	bool m_bExitThread; // 用于表示线程是否需要退出
private:
	DWORD m_dwStartTime;  // 线程开始时间
	DWORD m_dwTimeout;    // 超时时间
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_INSPTHRD_H__1C81645D_D41C_4CB4_AD97_DDAAEAB90337__INCLUDED_)
