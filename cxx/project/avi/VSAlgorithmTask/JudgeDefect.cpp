#include "stdafx.h"
#include "JudgeDefect.h"
#include "VSAlgorithmTask.h"
#include "DllInterface.h"
#include "Markup.h"
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif

JudgeDefect::JudgeDefect()
{
}

JudgeDefect::~JudgeDefect(void)
{
}

bool JudgeDefect::m_fnDefectFiltering(cv::Mat& MatDrawBuffer, int nImageNum, int nCameraNum, stDefectInfo* pResultBlob, tAlignInfo stThrdAlignInfo, int nRatio)
{
	TRACE("\n[%s][%s] - Start\n", __FILE__, __FUNCTION__);

	//异常处理
	if (pResultBlob == NULL)	return false;

	//不良数量
	int nDefectCount = pResultBlob->nDefectCount;

	//获取未检查区域ROI数量
	int			nFilterROICnt = theApp.GetFilterROICnt(nImageNum, nCameraNum);
	cv::Point	ptDefectCenter;
	CRect		rectFilterArea;

	//非检查区域ROI数量
	for (int nROICnt = 0; nROICnt < nFilterROICnt; nROICnt++)
	{
		if (theApp.GetUseFilterROI(nImageNum, nCameraNum, nROICnt))
		{
			//获取无检查区域ROI
			rectFilterArea = theApp.GetFilterROI(nImageNum, nCameraNum, nROICnt, nRatio);

			//非检查区域-以Left-Top坐标为原点具有坐标值
			rectFilterArea.OffsetRect(CPoint(stThrdAlignInfo.ptStandard.x, stThrdAlignInfo.ptStandard.y));

			//Cell顶点坐标
			cv::Point ptPoint[4];

			//必须反向旋转到原始坐标
			if (theApp.m_Config.GetEqpType() != EQP_APP)
				RotateRect(rectFilterArea, ptPoint, stThrdAlignInfo);
			//APP不旋转2017.09.23
			else
				RotateRect(rectFilterArea, ptPoint, stThrdAlignInfo, false);

			//绘制无检查ROI
			DrawAdjustROI(MatDrawBuffer, ptPoint, theApp.GetFilterROIName(nImageNum, nCameraNum, nROICnt), nROICnt, eFilterROI);

			//不良数量
			for (int nForDefect = 0; nForDefect < nDefectCount; nForDefect++)
			{
				//			//以不良为中心
// 			if( m_stThrdAlignInfo.ptCorner )
// 			{
				//			//更改为原始基准坐标
// 				ptDefectCenter.x = (LONG)pResultBlob->nCenterx[nForDefect] + m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_TOP].x;
// 				ptDefectCenter.y = (LONG)pResultBlob->nCentery[nForDefect] + m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_TOP].y;
// 			}
// 			else
// 			{
			//不良坐标-基于原始画面
				ptDefectCenter.x = (pResultBlob->ptLT[nForDefect].x + pResultBlob->ptRB[nForDefect].x) / 2;
				ptDefectCenter.y = (pResultBlob->ptLT[nForDefect].y + pResultBlob->ptRB[nForDefect].y) / 2;
				// 			}			

							//中心在非检查区域内吗？
				if (PtInPolygon(ptPoint, ptDefectCenter, 4))
				{
					pResultBlob->bUseResult[nForDefect] = false;
				}
			}
		}
	}

	TRACE("\n[%s][%s] - End\n", __FILE__, __FUNCTION__);

	return true;
}

//多边形区域内是否有点
bool JudgeDefect::PtInPolygon(cv::Point* ptPolygon, cv::Point ptPoint, int nPolygonCnt)
{
	bool flag = false;

	// step 1.
	for (int i = 0, j = nPolygonCnt - 1; i < nPolygonCnt; j = i++)
	{
		// step 2.
		if (((ptPolygon[i].y > ptPoint.y) != (ptPolygon[j].y > ptPoint.y)) &&
			// step 3.
			(ptPoint.x < (ptPoint.y - ptPolygon[i].y) * (ptPolygon[j].x - ptPolygon[i].x) / (ptPolygon[j].y - ptPolygon[i].y) + ptPolygon[i].x))
			flag = !flag;
	}

	return flag;
}

void JudgeDefect::ConsolidateResult(const CString strPanelID, const CString strDrive, CWriteResultInfo WrtResultInfo, ResultBlob_Total* pResultBlob_Total, ResultPanelData& resultPanelData, const int nImageNum, const int nCameraNum, int nRatio, ENUM_INSPECT_MODE eInspMode)
{
	//聚合结果值
	stDefectInfo* pRb;

	//导出特定于Image的检测结果(Alg)文件
	int nImageDefectCount = 0;

	//根据图像收集结果
	for (POSITION pos = pResultBlob_Total->GetHeadPosition_ResultBlob(); pos != NULL;)
	{
		pRb = pResultBlob_Total->GetNext_ResultBlob(pos);

		//17.09.07-必要时使用
		if (theApp.GetCommonParameter()->bIFImageSaveFlag)
		{
			CString strResultPath = _T("");
			strResultPath.Format(_T("%s\\%s\\%d_%s_CAM%02d.csv"), INSP_PATH, strPanelID, pRb->nImageNumber, theApp.GetGrabStepName(nImageNum), nCameraNum);

			//保存模式结果文件
			BlobFeatureSave(pRb, strResultPath, &nImageDefectCount);
		}

		for (int nListCnt = 0; nListCnt < pRb->nDefectCount; nListCnt++)
		{
			//不报告的不良过滤
			if (!pRb->bUseResult[nListCnt])	continue;

			//17.03.10需要添加在哪些模式画面中检测到的内容
			ResultDefectInfo* pResultDefectInfo = new ResultDefectInfo;

			//将来自AVI/SVI/APP算法的结果值(pRb)作为综合表单(pResultDefectInfo)进行铸造
			GetDefectInfo(WrtResultInfo, pResultDefectInfo, pRb, nListCnt, nImageNum, nCameraNum, nRatio);

			resultPanelData.Add_DefectInfo(*pResultDefectInfo);

			SAFE_DELETE(pResultDefectInfo);
		}
	}
}

void JudgeDefect::RotateRect(CRect rcSrc, cv::Point* pPtDst, tAlignInfo stThrdAlignInfo, bool bRotate)
{
	pPtDst[E_CORNER_LEFT_TOP].x = rcSrc.left;
	pPtDst[E_CORNER_LEFT_TOP].y = rcSrc.top;
	pPtDst[E_CORNER_RIGHT_TOP].x = rcSrc.right;
	pPtDst[E_CORNER_RIGHT_TOP].y = rcSrc.top;
	pPtDst[E_CORNER_RIGHT_BOTTOM].x = rcSrc.right;
	pPtDst[E_CORNER_RIGHT_BOTTOM].y = rcSrc.bottom;
	pPtDst[E_CORNER_LEFT_BOTTOM].x = rcSrc.left;
	pPtDst[E_CORNER_LEFT_BOTTOM].y = rcSrc.bottom;

	if (bRotate)
	{
		//反向旋转
		Align_DoRotatePoint(pPtDst[E_CORNER_LEFT_TOP], pPtDst[E_CORNER_LEFT_TOP], stThrdAlignInfo.ptAlignCenter, -stThrdAlignInfo.dAlignTheta);
		Align_DoRotatePoint(pPtDst[E_CORNER_RIGHT_TOP], pPtDst[E_CORNER_RIGHT_TOP], stThrdAlignInfo.ptAlignCenter, -stThrdAlignInfo.dAlignTheta);
		Align_DoRotatePoint(pPtDst[E_CORNER_RIGHT_BOTTOM], pPtDst[E_CORNER_RIGHT_BOTTOM], stThrdAlignInfo.ptAlignCenter, -stThrdAlignInfo.dAlignTheta);
		Align_DoRotatePoint(pPtDst[E_CORNER_LEFT_BOTTOM], pPtDst[E_CORNER_LEFT_BOTTOM], stThrdAlignInfo.ptAlignCenter, -stThrdAlignInfo.dAlignTheta);
	}
}

void JudgeDefect::NumberingDefect(const CString strModelID, const CString strPanelID, const CString strLotID, CWriteResultInfo WrtResultInfo, ResultPanelData& ResultPanelData, int nRatio)
{
	//参考设置
	ResultPanelData.m_ResultPanel.Recipe_ID = strModelID;
	ResultPanelData.m_ResultPanel.Panel_ID = strPanelID;
	ResultPanelData.m_ResultPanel.LOT_ID = strLotID;
	ResultPanelData.m_ResultPanel.SetAlignCellROI(&m_stThrdAlignInfo.rcAlignCellROI, 1.0 / nRatio);

	double dHResolution = 0.0, dVResolution = 0.0;
	WrtResultInfo.GetCalcResolution(m_stThrdAlignInfo.rcAlignCellROI, dHResolution, dVResolution);

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, FALSE, FALSE,
		_T("Saved Resolution = %.3f -> Calc Resolution = H : %.3f, V : %.3f (Panel Size = Width : %d, Height : %d / Ratio : %d)"),
		WrtResultInfo.GetCamResolution(0), dHResolution, dVResolution, m_stThrdAlignInfo.rcAlignCellROI.width, m_stThrdAlignInfo.rcAlignCellROI.height, nRatio);

	for (int nListCnt = 0; nListCnt < ResultPanelData.m_ListDefectInfo.GetCount(); nListCnt++)
	{
		//错误顺序(UI必须从1开始发送)
		ResultPanelData.m_ListDefectInfo[nListCnt].Defect_No = nListCnt + 1;

		Coord coordStart(0, 0), coordEnd(0, 0), coordCenter(0, 0), coordRepair(0, 0);
		GD_POINT lineStart(0, 0), lineEnd(0, 0), lineCenter(0, 0), lineRepair(0, 0);
		Coord CoordPixel(0, 0);
		int nCameraNum = 0, nDefectRatio = 0;

		nCameraNum = ResultPanelData.m_ListDefectInfo[nListCnt].Camera_No;
		nDefectRatio = ResultPanelData.m_ListDefectInfo[nListCnt].nRatio;

		//错误大小为0时进行异常处理
		if (ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Start_X - ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_End_X == 0 &&
			ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Start_Y - ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_End_Y == 0)
		{
			coordStart = Coord(0, 0);
			coordCenter = Coord(0, 0);
			coordEnd = Coord(0, 0);
			coordRepair = Coord(0, 0);

			lineStart = GD_POINT(0, 0);
			lineCenter = GD_POINT(0, 0);
			lineEnd = GD_POINT(0, 0);
			lineRepair = GD_POINT(0, 0);
		}
		else
		{
			// Defect Start X, Y
			CoordPixel.X = (DOUBLE)ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Start_X;
			CoordPixel.Y = (DOUBLE)ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Start_Y;
			///17.07.07更改为在整个Panel设计中工作的计算方式
			coordStart = WrtResultInfo.CalcWorkCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);
			lineStart = WrtResultInfo.CalcGateDataCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);

			// Defect End X, Y
			CoordPixel.X = (DOUBLE)ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_End_X;
			CoordPixel.Y = (DOUBLE)ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_End_Y;
			coordEnd = WrtResultInfo.CalcWorkCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);
			lineEnd = WrtResultInfo.CalcGateDataCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);

			//Defect Center X,Y-当前未使用
			CoordPixel.X = ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Center_X;
			CoordPixel.Y = ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Center_Y;
			coordCenter = WrtResultInfo.CalcWorkCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);
			lineCenter = WrtResultInfo.CalcGateDataCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);

			//转交给Repair的坐标
			CoordPixel.X = ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Repair_X;
			CoordPixel.Y = ResultPanelData.m_ListDefectInfo[nListCnt].Pixel_Repair_Y;
			coordRepair = WrtResultInfo.CalcWorkCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);
			lineRepair = WrtResultInfo.CalcGateDataCoord(m_stThrdAlignInfo.rcAlignCellROI, CoordPixel, nDefectRatio, nRatio);
		}

		ResultPanelData.m_ListDefectInfo[nListCnt].Gate_Start_No = min(lineStart.Gate, lineEnd.Gate);
		ResultPanelData.m_ListDefectInfo[nListCnt].Data_Start_No = min(lineStart.Data, lineEnd.Data);
		ResultPanelData.m_ListDefectInfo[nListCnt].Gate_End_No = max(lineStart.Gate, lineEnd.Gate);
		ResultPanelData.m_ListDefectInfo[nListCnt].Data_End_No = max(lineStart.Data, lineEnd.Data);
		ResultPanelData.m_ListDefectInfo[nListCnt].Coord_Start_X = min(coordStart.X, coordEnd.X);
		ResultPanelData.m_ListDefectInfo[nListCnt].Coord_Start_Y = min(coordStart.Y, coordEnd.Y);
		ResultPanelData.m_ListDefectInfo[nListCnt].Coord_End_X = max(coordStart.X, coordEnd.X);
		ResultPanelData.m_ListDefectInfo[nListCnt].Coord_End_Y = max(coordStart.Y, coordEnd.Y);
		ResultPanelData.m_ListDefectInfo[nListCnt].Repair_Gate = lineRepair.Gate;
		ResultPanelData.m_ListDefectInfo[nListCnt].Repair_Data = lineRepair.Data;
		ResultPanelData.m_ListDefectInfo[nListCnt].Repair_Coord_X = coordRepair.X;
		ResultPanelData.m_ListDefectInfo[nListCnt].Repair_Coord_Y = coordRepair.Y;
	}
}

//添加AD不良列表(添加时为true/未添加时为false)
bool JudgeDefect::JudgeADDefect(int nImageNum, int nCameraNum, int nStageNo, int nImageWidth, int nImageHeight, ResultBlob_Total* pResultBlob_Total, int nDefectAD, ENUM_INSPECT_MODE eInspMode, bool bAD)
{
	//如果AD不好,添加列表后退出(不检查其他算法)

	//检查结果(不良)信息结构体
	stDefectInfo* pAD = new stDefectInfo(2, nImageNum);

	//放入要传递给UI的结果
	pAD->nArea[0] = 0;	//nImageWidth * nImageHeight;	// 17.06.23负坐标修改
	pAD->nMaxGV[0] = 255;
	pAD->nMinGV[0] = 0;
	pAD->dMeanGV[0] = 0;

	pAD->ptLT[0].x = 0;
	pAD->ptLT[0].y = 0;
	pAD->ptRT[0].x = 0;	//nImageWidth-1;	// 17.06.23负坐标修改
	pAD->ptRT[0].y = 0;
	pAD->ptRB[0].x = 0;	//nImageWidth-1;	// 17.06.23负坐标修改
	pAD->ptRB[0].y = 0;	//nImageHeight-1;	// 17.06.23负坐标修改
	pAD->ptLB[0].x = 0;
	pAD->ptLB[0].y = 0;	//nImageHeight-1;	// 17.06.23负坐标修改

	pAD->dBackGroundGV[0] = 0;

	pAD->dCompactness[0] = 0;
	pAD->dSigma[0] = 0;
	pAD->dBreadth[0] = 0;	//nImageHeight;	// 17.06.23负坐标修改
	pAD->dF_Min[0] = 0;	//nImageHeight;	// 17.06.23负坐标修改
	pAD->dF_Max[0] = 0;	//nImageWidth;	// 17.06.23负坐标修改
	pAD->dF_Elongation[0] = 0;
	pAD->dCompactness[0] = 0;

	//亮度
	pAD->nDefectColor[0] = E_DEFECT_COLOR_DARK;

	pAD->nDefectJudge[0] = nDefectAD;
	pAD->nPatternClassify[0] = nImageNum;

	//计数增加
	pAD->nDefectCount = 1;

	//合并不良信息
	pResultBlob_Total->AddTail_ResultBlobAndAddOffset(pAD, NULL);

	//如果是AutoRun,则AD错误计数
	if (eInspMode == eAutoRun && bAD)
		m_fnCountingStageAD(nImageNum, nStageNo, nDefectAD);

	//AD不良时为true
	return true;
}

//添加每个Stage的AD不良计数功能
bool JudgeDefect::m_fnCountingStageAD(int nImageNum, int nStageNo, int nDefectType)
{
	//指定文件名/节/键
	CString strLogPath = _T(""), strSection = _T(""), strKey = _T("");
	strLogPath.Format(_T("%s\\CountingStageAD_PC%02d.INI"), DEFECT_INFO_PATH, theApp.m_Config.GetPCNum());
	strSection.Format(_T("Stage_%d_%d"), nStageNo, theApp.m_Config.GetPCNum());
	if (nDefectType == E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL || nDefectType == E_DEFECT_JUDGEMENT_DISPLAY_OFF)
		strKey.Format(_T("AD"));
	else
		strKey.Format(_T("%s_GV"), theApp.GetGrabStepName(nImageNum));

	TRY
	{
		//读取/增加当前Count后写入
EnterCriticalSection(&theApp.m_csCntFileSafe);
int nCurCount = GetPrivateProfileInt(strSection, strKey, 0, strLogPath);
nCurCount++;
CString strCount = _T("");
strCount.Format(_T("%d"), nCurCount);
WritePrivateProfileString(strSection, strKey, strCount, strLogPath);

LeaveCriticalSection(&theApp.m_csCntFileSafe);
	}
		CATCH(CException, e)
	{
		e->Delete();
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Exception m_fnCountingStageAD()"));
		return false;
	}
	END_CATCH

		return true;
}

bool JudgeDefect::JudgementPanelGrade(ResultPanelData& resultPanelData)
{
	int nDefectTypeNum = 0;
	memset(resultPanelData.m_nDefectTrend, 0, sizeof(int) * E_PANEL_DEFECT_TREND_COUNT);

	//用于不良的Counting-Panel统计操作
	for (int nIndex = 0; nIndex < resultPanelData.m_ListDefectInfo.GetCount(); nIndex++)
	{
		nDefectTypeNum = resultPanelData.m_ListDefectInfo[nIndex].Defect_Type;
		resultPanelData.m_nDefectTrend[nDefectTypeNum]++;

		//用于修复的BP+DP计数Count
		if (nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_DARK)
			resultPanelData.m_nDefectTrend[E_DEFECT_BP_PLUS_DP]++;

		//BP+DP+WD+BD+GD+GB计数Count//lxq--2023/06/19
		if (nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_DARK ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			resultPanelData.m_nDefectTrend[E_DEFECT_BP_DP_WB_BD_GD_GB_PLUS]++;

		/*if (nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_DARK ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK )
			resultPanelData.m_nDefectTrend[E_DEFECT_BP_DP_WB_BD_GD_GB_PLUS]++;*/

			//Re-test计数
		if (nDefectTypeNum >= E_DEFECT_JUDGEMENT_RETEST_POINT_DARK &&
			nDefectTypeNum <= E_DEFECT_JUDGEMENT_RETEST_MURA_BRIGHT)
			resultPanelData.m_nDefectTrend[E_DEFECT_RETEST]++;

		// BP + WB + GB
		if (nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT ||
			nDefectTypeNum == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			resultPanelData.m_nDefectTrend[E_DEFECT_BP_WB_GB]++;
	}

	//根据面板判定优先级进行比较
	std::vector<stPanelJudgePriority> vPanelJudge = theApp.GetPanelJudgeInfo();
	bool bNextOrder = false;
	int nPriority = 0;
	for (; nPriority < vPanelJudge.size(); nPriority++)
	{
		bNextOrder = false;
		for (nDefectTypeNum = 0; nDefectTypeNum < E_PANEL_DEFECT_TREND_COUNT; nDefectTypeNum++)
		{
			if (!m_fnCompareValue(resultPanelData.m_nDefectTrend[nDefectTypeNum], vPanelJudge[nPriority].stJudgeInfo[nDefectTypeNum].nRefVal, vPanelJudge[nPriority].stJudgeInfo[nDefectTypeNum].nSign))
			{
				bNextOrder = true;
				break;
			}
		}
		if (!bNextOrder)
			break;
	}
	//如果没有匹配类型,则NG
	if (nPriority == vPanelJudge.size())
	{
		//如果没有比较语法
		if (resultPanelData.m_ListDefectInfo.GetCount() == 0)
			resultPanelData.m_ResultPanel.Judge = _T("S");		// 没有不良的话V
		else
			resultPanelData.m_ResultPanel.Judge = _T("F");		// S - Scrap (NG)
	}
	//如果有,请填写相应的Grade
	else
	{
		resultPanelData.m_ResultPanel.Judge.Format(_T("%s"), vPanelJudge[nPriority].strGrade);
	}

	return true;
}

//适用不良上位报告过滤条件后选定代表不良
bool JudgeDefect::ApplyReportRule(ResultPanelData& resultPanelData)
{
	int nDefectTypeNum = 0;

	int nTotalDefectCount = (int)resultPanelData.m_ListDefectInfo.GetCount();
	int* pDefectTrend = new int[E_PANEL_DEFECT_TREND_COUNT];
	memcpy(pDefectTrend, resultPanelData.m_nDefectTrend, sizeof(int) * E_PANEL_DEFECT_TREND_COUNT);		// 保留现有趋势文件,创建基于父报告的趋势数组

	int nJudgeIndex = theApp.GetPanelJudgeIndex(resultPanelData.m_ResultPanel.Judge);

	stPanelJudgeInfo stFilterInfo[E_PANEL_DEFECT_TREND_COUNT];
	memcpy(stFilterInfo, theApp.GetReportFilter(nJudgeIndex), sizeof(stPanelJudgeInfo) * E_PANEL_DEFECT_TREND_COUNT);
	//int nOverlapDefectNum[E_DEFECT_JUDGEMENT_COUNT] = {0,}; // 如果在代表性不良评选中有重复的排名,请保存相应的Defect Num
	vector<int>::iterator iter;
	if (TRUE)
	{
		//for (nDefectTypeNum = 0; nDefectTypeNum < E_DEFECT_JUDGEMENT_COUNT; nDefectTypeNum++)	//YWS Defect Count 190510
		for (nDefectTypeNum = 0; nDefectTypeNum < E_PANEL_DEFECT_TREND_COUNT; nDefectTypeNum++)
		{
			if (pDefectTrend[nDefectTypeNum] == 0)	continue;		// 如果没有任何不良,则无需过滤Skip

			//符合过滤条件的类型的错误显示为不报告父项(bUseReport=false)
			if (m_fnCompareValue(pDefectTrend[nDefectTypeNum], stFilterInfo[nDefectTypeNum].nRefVal, stFilterInfo[nDefectTypeNum].nSign))
			{//父报告无不良

				pDefectTrend[nDefectTypeNum] = 0;
				for (int nIndex = 0; nIndex < resultPanelData.m_ListDefectInfo.GetCount(); nIndex++)
				{
					if (resultPanelData.m_ListDefectInfo[nIndex].Defect_Type == nDefectTypeNum)
					{
						resultPanelData.m_ListDefectInfo[nIndex].bUseReport = false;
						nTotalDefectCount--;
					}
				}
			}
		}
	}

	nTotalDefectCount = UserDefinedFilter(resultPanelData, nTotalDefectCount); // 自定义过滤器。N个以上DEPEC设置的数量以上时无条件报告。

	//选定代表不良	
	if (nTotalDefectCount != 0)		//如果有任何不良情况
	{
		//选定代表不良	%
		int nHighestDefectRank = 0;
		int nCurDefType = 0;
		int nMostDefectNum = -1;

		for (int nIndex = 0; nIndex < resultPanelData.m_ListDefectInfo.GetCount(); nIndex++)
		{
			if (resultPanelData.m_ListDefectInfo[nIndex].bUseReport == true)
			{
				nCurDefType = resultPanelData.m_ListDefectInfo[nIndex].Defect_Type;

				if (nMostDefectNum == -1)	//初始不良
				{
					nMostDefectNum = nCurDefType;
					nHighestDefectRank = theApp.GetDefectRank(nCurDefType);
					continue;
				}
				//如果当前Defect Type的排名更高,请选择当前Defect Type
				if (nHighestDefectRank > theApp.GetDefectRank(nCurDefType))
				{
					nHighestDefectRank = theApp.GetDefectRank(nCurDefType);
					nMostDefectNum = nCurDefType;
				}
				//如果排名相同,则选择不良次数更多的Type
				else if (nHighestDefectRank == theApp.GetDefectRank(nCurDefType))
				{
					if (pDefectTrend[nMostDefectNum] < pDefectTrend[nCurDefType])
					{
						nMostDefectNum = nCurDefType;
					}
				}
			}
		}
		resultPanelData.m_ResultPanel.nFinalDefectNum = nMostDefectNum;

		///各不良人群代表不良选定功能//////////////////////////////////////////////////////////////////////////////////////
		//根据工程师的要求,设置为只报告一个百分点/漏电/混色不良
		//必要时需要UI操作来设置各不良群体的代表性不良功能
		//目前只需要对一个不良(==不良群)进行代表性不良报告的语法。
		//为了扩展性,将多个不良识别为一个不良群,没有将是否使用代表性不良区分为TRUE/FALSE,而是赋予Index(相同Index为相同不良群)
		//最多可以生成E_DEFECT_JUDGEMENT_COUNT大小的坏群。在不良人群中选定代表不良的话,为了不重复检查的标志
		bool bFindDefect[E_DEFECT_JUDGEMENT_COUNT] = { false, };

		for (nDefectTypeNum = 0; nDefectTypeNum < E_DEFECT_JUDGEMENT_COUNT; nDefectTypeNum++)
		{
			//使用不良人群代表性不良功能(0则不使用不良人群代表性不良功能)
			//如果需要将多个不良识别为一个不良群体,则需要单独操作。完成任务
			int nGroupIndex = theApp.GetDefectGroup(nDefectTypeNum);
			if (nGroupIndex != 0)		//如果Defect Type被选为代表性不良报告的不良人群
			{
				//有一个或多个不良行为时
				if (pDefectTrend[nDefectTypeNum] >= 1)
				{
					for (int nIndex = 0; nIndex < resultPanelData.m_ListDefectInfo.GetCount(); nIndex++)
					{
						//除了最初的一个缺陷之外,没有报告
						if (resultPanelData.m_ListDefectInfo[nIndex].Defect_Type == nDefectTypeNum)
						{
							if (!bFindDefect[nGroupIndex])
							{
								bFindDefect[nGroupIndex] = true;	// 目前在不良人群中选定代表不良
								continue;
							}
							resultPanelData.m_ListDefectInfo[nIndex].bUseReport = false;
						}
					}
				}
			}
		}
		////////////////////////////////////////////////////////////////////////////////

		try
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("读取上报规则缺陷表."));
			//////////////////////////////////////////////////////////////////////////////////////////////
			CString strMsg = _T("");
			CFileFind find;
			BOOL bFindFile = FALSE;
			CString strDefItemListXMLPath;

			strDefItemListXMLPath.Format(_T("%s:\\IMTC\\Text\\DEFITEM_LIST.xml"), theApp.m_Config.GETCmdDRV());

			bFindFile = find.FindFile(strDefItemListXMLPath);
			find.Close();

			if (!bFindFile)
			{
				// 			strMsg.Format(_T("Not found defect item list xml file. (%s)"), strDefItemListXMLPath);		
				// 			AfxMessageBox(strMsg);		
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("读取上报规则缺陷表失败. (%s)"), strDefItemListXMLPath);
				return false;
			}

			//加载XML文件
			CMarkup xmlDefectItem;
			if (!xmlDefectItem.Load(strDefItemListXMLPath))
			{
				// 			strMsg.Format(_T("Model xml load fail. (%s)"), strDefItemListXMLPath);
				// 			AfxMessageBox(strMsg);
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("上报规则缺陷表加载失败. (%s)"), strDefItemListXMLPath);
				return false;
			}

			xmlDefectItem.FindElem();		// DEF_ITEM
			xmlDefectItem.IntoElem();		// inside DEF_ITEM

			CString strDefSysName = _T(""), strDefCode = _T("");
			CMarkup* xmlDefItemList = new CMarkup(xmlDefectItem);
			stDefClassification* stDefClass = new stDefClassification[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Create stDefClassification"));
			for (int nDefItemIndex = 0; nDefItemIndex < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; nDefItemIndex++)
			{
				if (xmlDefItemList->FindElem(_T("DefType_%d"), nDefItemIndex))
				{
					strDefSysName = xmlDefItemList->GetAttrib(_T("SysName"));
					//memcpy(stDefClass[nDefItemIndex].strDefectName, strDefSysName.GetBuffer(0), sizeof(stDefClass[nDefItemIndex].strDefectName) - sizeof(TCHAR));
					//_tcscat(stDefClass[nDefItemIndex].strDefectName, _T("\0"));
					COPY_CSTR2TCH(stDefClass[nDefItemIndex].strDefectName, strDefSysName, sizeof(stDefClass[nDefItemIndex].strDefectName));
					strDefCode = xmlDefItemList->GetAttrib(_T("DefCode"));
					//memcpy(stDefClass[nDefItemIndex].strDefectCode, strDefCode.GetBuffer(0), sizeof(stDefClass[nDefItemIndex].strDefectCode) - sizeof(TCHAR));
					//_tcscat(stDefClass[nDefItemIndex].strDefectCode, _T("\0"));
					COPY_CSTR2TCH(stDefClass[nDefItemIndex].strDefectCode, strDefCode, sizeof(stDefClass[nDefItemIndex].strDefectCode));
					xmlDefItemList->ResetMainPos();
				}
			}
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("SetDefectClassify set data"));
			theApp.SetDefectClassify(stDefClass);

			SAFE_DELETE_ARR(stDefClass);
			SAFE_DELETE(xmlDefItemList);

			if (pDefectTrend[E_DEFECT_JUDGEMENT_POINT_BRIGHT] + pDefectTrend[E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT] +
				pDefectTrend[E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK] + pDefectTrend[E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT] >= 2)
			{
				stDefClassification* stDefClass = new stDefClassification[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];
				memcpy(stDefClass, theApp.GetDefectClassify(), sizeof(stDefClassification) * MAX_MEM_SIZE_E_DEFECT_NAME_COUNT);
				//stDefClass[5].strDefectCode
				_tcscpy(stDefClass[E_DEFECT_JUDGEMENT_POINT_BRIGHT].strDefectCode, _T("PB1000"));
				_tcscpy(stDefClass[E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT].strDefectCode, _T("PB1000"));
				_tcscpy(stDefClass[E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK].strDefectCode, _T("PB1000"));
				_tcscpy(stDefClass[E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT].strDefectCode, _T("PB2000"));
				theApp.SetDefectClassify(stDefClass);
				//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Change SetDefectClassify"));
			}
			resultPanelData.m_ResultPanel.judge_code_1.Format(_T("%s"), theApp.GetDefectSysName(nMostDefectNum));
			resultPanelData.m_ResultPanel.judge_code_2.Format(_T("%s"), theApp.GetDefectCode(nMostDefectNum));
		}
		catch (CMemoryException* e)
		{
			TCHAR strErr[256];
			e->GetErrorMessage(strErr, 256);
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Apply report rule memory Exception : %s"), strErr);
			e->Delete();
			return false;
		}
		catch (CFileException* e)
		{
			TCHAR strErr[256];
			e->GetErrorMessage(strErr, 256);
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Apply report rule file Exception : %s"), strErr);
			e->Delete();
			return false;
		}
		catch (CException* e)
		{
			TCHAR strErr[256];
			e->GetErrorMessage(strErr, 256);
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Apply report rule Exception : %s"), strErr);
			e->Delete();
			return false;
		}
	}
	else
	{
		resultPanelData.m_ResultPanel.nFinalDefectNum = -1;
		resultPanelData.m_ResultPanel.judge_code_1 = _T("NOTHING");
		resultPanelData.m_ResultPanel.judge_code_2 = _T("NOTHING");
	}
	SAFE_DELETE_ARR(pDefectTrend);

	return true;
}

int JudgeDefect::UserDefinedFilter(ResultPanelData& resultPanelData, int nTotalDefectCount)
{
	int* pDefectTrendCount = new int[E_DEFECT_JUDGEMENT_COUNT];
	//memcpy(pDefectTrend, resultPanelData.m_nDefectTrend, sizeof(int) * E_DEFECT_JUDGEMENT_COUNT);		// 保留现有趋势文件,创建基于父报告的趋势数组
	for (int nIndex = 0; nIndex < E_DEFECT_JUDGEMENT_COUNT; nIndex++)
		pDefectTrendCount[nIndex] = resultPanelData.m_nDefectTrend[nIndex];
	CString strGrade;

	resultPanelData.m_ResultPanel.Judge;

	//resultPanelData.m_ResultPanel.Judge; 与strGrade相比,对于目前推出的GRADE
	//使用pDefectTrend获取完美项目数
	//如果每个DEPEC项目的数量超过设置的值,则报告标志TRUE

	int nDefectCount = 0;
	std::vector<stUserDefinedFilter> vUserFilter = theApp.GetUserDefinedFilter();
	for (int i = 0; i < vUserFilter.size(); i++)
	{
		strGrade.Format(_T("%s"), vUserFilter[i].strGrade);
		if (strGrade == resultPanelData.m_ResultPanel.Judge)
		{
			for (int j = 0; j < vUserFilter[i].nFilterItemsCount; j++)
				nDefectCount += pDefectTrendCount[vUserFilter[i].nFilterItems[j]]; // 在定制过滤器中设置的Defect之和

			if (m_fnCompareValue(nDefectCount, vUserFilter[i].stFilterInfo.nRefVal, vUserFilter[i].stFilterInfo.nSign))//如果Defect的和符合过滤器中定义的条件
			{
				for (int j = 0; j < vUserFilter[i].nFilterItemsCount; j++)  // 
				{
					for (int nIndex = 0; nIndex < resultPanelData.m_ListDefectInfo.GetCount(); nIndex++)
					{
						if (resultPanelData.m_ListDefectInfo[nIndex].Defect_Type == vUserFilter[i].nFilterItems[j])
						{
							resultPanelData.m_ListDefectInfo[nIndex].bUseReport = true; // 看着DEPEC重新生活。
							nTotalDefectCount++;
						}
					}
				}
			}
		}
	}
	SAFE_DELETE_ARR(pDefectTrendCount);
	return nTotalDefectCount;
}

bool JudgeDefect::m_fnCompareValue(int nDefectCount, int nRefCount, int nSign)
{
	bool bRet = false;
	switch (nSign)
	{
	case 0:
		if (nDefectCount == nRefCount)	bRet = true;
		break;
	case 1:
		if (nDefectCount != nRefCount)	bRet = true;
		break;
	case 2:
		if (nDefectCount > nRefCount)	bRet = true;
		break;
	case 3:
		if (nDefectCount < nRefCount)	bRet = true;
		break;
	case 4:
		if (nDefectCount >= nRefCount)	bRet = true;
		break;
	case 5:
		if (nDefectCount <= nRefCount)	bRet = true;
		break;
	}
	return bRet;
}