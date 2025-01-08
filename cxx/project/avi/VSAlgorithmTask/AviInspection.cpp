#include "stdafx.h"
#include "AviInspection.h"
#include "AviDefineInspect.h"
#include "DllInterface.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"
#include <codecvt>
#include <locale>

CMatBufferResultManager	cMemResultBuff;

void ImageSave(cv::Mat& MatSrcBuffer, TCHAR* strPath, ...);
void ImageAsyncSaveJPG(cv::Mat& MatSrcBuffer, const char* strPath);

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif

using namespace Concurrency;

IMPLEMENT_DYNCREATE(AviInspection, CWinThread)

AviInspection::AviInspection(void)
{

	//cMemResult[0] = NULL;
}

AviInspection::~AviInspection(void)
{
}

bool AviInspection::DrawDefectImage(CString strPanelID, cv::Mat(*MatResult)[MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat MatDrawBuffer[][MAX_CAMERA_COUNT], ResultPanelData& resultPanelData)
{
	CCPUTimer timerDrawRect;
	timerDrawRect.Start();

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//[Draw&Save]显示错误
		cv::Scalar color;

		//只需添加Point&Line(Mura外角线图)
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
		case E_DEFECT_JUDGEMENT_POINT_DARK:
		case E_DEFECT_JUDGEMENT_POINT_GROUP_DARK:
			color = cv::Scalar(255, 0, 0);		// Red
			break;

		case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
		case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
		case E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT:
		case E_DEFECT_JUDGEMENT_MURA_MULT_BP:	//接近Point不良Gray图案(100分相似)
			color = cv::Scalar(0, 255, 0);		// Green
			break;

		case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:
		case E_DEFECT_JUDGEMENT_RETEST_POINT_DARK:
		case E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT:
			color = cv::Scalar(0, 0, 255);		// Blue
			break;

		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT_MULT:
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT_MULT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK_MULT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK_MULT:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
		case E_DEFECT_JUDGEMENT_LINE_DGS:
		case E_DEFECT_JUDGEMENT_LINE_DGS_X:
		case E_DEFECT_JUDGEMENT_LINE_DGS_Y:
		case E_DEFECT_JUDGEMENT_XLINE_SPOT:
		case E_DEFECT_JUDGEMENT_YLINE_SPOT:
		case E_DEFECT_JUDGEMENT_LINE_X_EDGE_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_EDGE_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_CRACK_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_CRACK_LEFT:
		case E_DEFECT_JUDGEMENT_LINE_CRACK_BOTH:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_RETEST_LINE_BRIGHT:
		case E_DEFECT_JUDGEMENT_RETEST_LINE_DARK:
			color = cv::Scalar(255, 105, 180);	// Hot Pink
			break;

		case E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL:
		case E_DEFECT_JUDGEMENT_DISPLAY_DARK:
		case E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT:
		case E_DEFECT_JUDGEMENT_DUST_GROUP:
			color = cv::Scalar(210, 105, 30);		// Chocolate
			break;

		default:
			color = cv::Scalar(0, 0, 0);			// Black
			break;
		}

		int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int nCamNum = resultPanelData.m_ListDefectInfo[i].Camera_No;

		//17.03.29不绘制Mura&SVI矩形(在Alg中单独绘制轮廓)
		//画面剪切部分必须完成
		int nOffSet = 5, nThickness = 1;
		//		//实际Defect区域

		//		//临时使用绝对值(by CWH)

		cv::Point ptDefectArea[4] = { (0, 0), };
		ptDefectArea[E_CORNER_LEFT_TOP].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_TOP].x - nOffSet;
		ptDefectArea[E_CORNER_LEFT_TOP].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_TOP].y - nOffSet;
		ptDefectArea[E_CORNER_RIGHT_TOP].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_TOP].x + nOffSet;
		ptDefectArea[E_CORNER_RIGHT_TOP].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_TOP].y - nOffSet;
		ptDefectArea[E_CORNER_RIGHT_BOTTOM].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM].x + nOffSet;
		ptDefectArea[E_CORNER_RIGHT_BOTTOM].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM].y + nOffSet;
		ptDefectArea[E_CORNER_LEFT_BOTTOM].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_BOTTOM].x - nOffSet;
		ptDefectArea[E_CORNER_LEFT_BOTTOM].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_BOTTOM].y + nOffSet;

		//Mura-在迷你地图上不画Rect
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_MURA_START &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_MURA_END)
		{
			//在迷你地图上不画Rect
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP)
				resultPanelData.m_ListDefectInfo[i].Draw_Defect_Rect = false;
		}
		//Mura Retest-在迷你地图上不画Rect
		else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_MURA ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_MURA_BRIGHT)
		{
			//在迷你地图上不画Rect
			resultPanelData.m_ListDefectInfo[i].Draw_Defect_Rect = false;
		}
		//APP-在迷你地图上不显示Rect
		else if (resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_APP_START &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_APP_END)
		{
			//在迷你地图上不画Rect
			resultPanelData.m_ListDefectInfo[i].Draw_Defect_Rect = false;
		}

		//绘制除村外的不良/矩形
		if (resultPanelData.m_ListDefectInfo[i].Draw_Defect_Rect)
		{
			//绘制矩形
//cv::rectangle(MatDrawBuffer[nImgNum][nCamNum], cv::Rect(nDefStartX, nDefStartY, nDefWidth, nDefHeight), color);

			cv::line(MatDrawBuffer[nImgNum][nCamNum], ptDefectArea[E_CORNER_LEFT_TOP], ptDefectArea[E_CORNER_RIGHT_TOP], color, nThickness);
			cv::line(MatDrawBuffer[nImgNum][nCamNum], ptDefectArea[E_CORNER_RIGHT_TOP], ptDefectArea[E_CORNER_RIGHT_BOTTOM], color, nThickness);
			cv::line(MatDrawBuffer[nImgNum][nCamNum], ptDefectArea[E_CORNER_RIGHT_BOTTOM], ptDefectArea[E_CORNER_LEFT_BOTTOM], color, nThickness);
			cv::line(MatDrawBuffer[nImgNum][nCamNum], ptDefectArea[E_CORNER_LEFT_BOTTOM], ptDefectArea[E_CORNER_LEFT_TOP], color, nThickness);
		}
	}
	timerDrawRect.End();
	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Draw Rect tact time %.2f"), timerDrawRect.GetTime() / 1000.);

	return true;
}

//外围处理
long AviInspection::makePolygonCellROI(const tLogicPara& LogicPara, cv::Mat& MatDrawBuffer, tAlignInfo& stThrdAlignInfo, STRU_LabelMarkInfo& labelMarkInfo, int nImageNum, int nCameraNum, double* dAlgPara, int nAlgImg, int nRatio)
{
	//异常处理
	if (dAlgPara == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	wchar_t wstrID[MAX_PATH] = { 0, };
	swprintf(wstrID, _T("%s"), LogicPara.tszPanelID);

	//设备类型
	int nEqpType = theApp.m_Config.GetEqpType();

	//yuxuefei for Label ROI. start
	//int nWhiteNumer = (int)dAlgPara[E_PARA_AVI_White_Number] - 1;
	//CRect		recLabelArea = { 0,0,0,0 };
	//if (theApp.GetUseLabelROI(nWhiteNumer, nCameraNum, 0))
	//{
	//	recLabelArea = theApp.GetLabelROI(nWhiteNumer, nCameraNum, 0);
	//}
	//yuxuefei. end
	//如果设置标志为On
	if (dAlgPara[E_PARA_ROUND_SETTING] > 0 || dAlgPara[E_PARA_CHOLE_SETTING] > 0 || dAlgPara[E_PARA_CHOLE_POINT_SETTING] > 0)
	{
		if (dAlgPara[E_PARA_CHOLE_POINT_SETTING] > 0)
		{
			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST)
			{
				int TimeCount = 0;
				while (LogicPara.MatOrgRGBAdd[0]->empty())
				{
					TimeCount++;
					Sleep(10);
					if (TimeCount >= 1000)
					{
						//Sleep(10);
						return E_ERROR_CODE_EMPTY_BUFFER;
					}
				}

				cv::Mat matSrcBuff[2];
				matSrcBuff[0] = LogicPara.MatOrgImage.clone();
				matSrcBuff[1] = LogicPara.MatOrgRGBAdd[0]->clone();

				nErrorCode = Align_SetFindContour_2(matSrcBuff, theApp.GetRoundROI(nImageNum, nCameraNum), theApp.GetCHoleROI(nImageNum, nCameraNum),
					theApp.GetRndROICnt(nImageNum, nCameraNum), theApp.GetCHoleROICnt(nImageNum, nCameraNum), dAlgPara, nAlgImg, nCameraNum, nRatio, nEqpType, theApp.GetRoundPath());
			}

		}

		if (dAlgPara[E_PARA_ROUND_SETTING] > 0 || dAlgPara[E_PARA_CHOLE_SETTING] > 0)
		{
			//注册拐角
			nErrorCode = Align_SetFindContour_(LogicPara.MatOrgImage, theApp.GetRoundROI(nImageNum, nCameraNum), theApp.GetCHoleROI(nImageNum, nCameraNum),
				theApp.GetRndROICnt(nImageNum, nCameraNum), theApp.GetCHoleROICnt(nImageNum, nCameraNum), dAlgPara, nAlgImg, nCameraNum, nRatio, nEqpType, theApp.GetRoundPath());
		}

		//如果正常工作,请更改
		//如果出现错误,请转交错误代码
		if (nErrorCode == E_ERROR_CODE_TRUE)
			nErrorCode = E_ERROR_CODE_ALIGN_ROUND_SETTING;
	}

	//如果设置标志为Off
	else
	{
		//用于Alg日志
		wchar_t strAlgLog[MAX_PATH] = { 0, };
		swprintf(strAlgLog, _T("ID:%s\tPat:%s"), LogicPara.tszPanelID, theApp.GetGrabStepName(nImageNum));

		// polmark add +
		STRU_LabelMarkParams labelMarkParams(dAlgPara, theApp.GetRoundPath());
		INSP_AREA* polRois = theApp.GetPolMarkROI(nImageNum, nCameraNum);
		int polmarkCnt = theApp.GetPolMarkROICnt(nImageNum, nCameraNum);

		for (int pi = 0; pi < polmarkCnt; pi++) {
			INSP_AREA polROI = polRois[pi];
			if (_tcsicmp(polROI.strROIName, _T("pol_num")) == 0) {
				labelMarkParams.polNumROI = cv::Rect(polROI.rectROI.left, polROI.rectROI.top, polROI.rectROI.Width(), polROI.rectROI.Height());
			}
			else if (_tcsicmp(polROI.strROIName, _T("pol_sign")) == 0) {
				labelMarkParams.polSignROI = cv::Rect(polROI.rectROI.left, polROI.rectROI.top, polROI.rectROI.Width(), polROI.rectROI.Height());
			}
		}
		labelMarkParams.cellBBox = stThrdAlignInfo.rcAlignCellROI;
		labelMarkParams.polNumTemplates = &theApp.m_polNumTemplates;
		labelMarkParams.polSignTemplates = &theApp.m_polSignTemplates;
		// polmark add -

		//如果不是Dust和背光模式
		if (nAlgImg != E_IMAGE_CLASSIFY_AVI_DUST && nAlgImg != E_IMAGE_CLASSIFY_AVI_DUSTDOWN)
		{
			nErrorCode = Align_FindFillOutArea(LogicPara.MatOrgImage, MatDrawBuffer, LogicPara.MatBKG, stThrdAlignInfo.ptCorner, labelMarkParams, labelMarkInfo, theApp.m_pGrab_Step[nImageNum].tRoundSet, theApp.m_pGrab_Step[nImageNum].tCHoleSet, stThrdAlignInfo.tCHoleAlignData->matCHoleROIBuf[nAlgImg], stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nAlgImg], stThrdAlignInfo.tCHoleAlignData->bCHoleAD[nAlgImg],
				dAlgPara, nAlgImg, nCameraNum, nRatio, nEqpType, strAlgLog, wstrID);
		}

		//如果是Dust和背光模式
		else
		{
			nErrorCode = Align_FindFillOutAreaDust(LogicPara.MatOrgImage, MatDrawBuffer, stThrdAlignInfo.ptCorner, labelMarkParams, labelMarkInfo, stThrdAlignInfo.dAlignTheta, stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nAlgImg],
				theApp.m_pGrab_Step[nImageNum].tRoundSet, theApp.m_pGrab_Step[nImageNum].tCHoleSet, dAlgPara, nAlgImg, nRatio, strAlgLog, wstrID);
		}
	}

	return nErrorCode;
}

bool AviInspection::Judgement(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT], tCHoleAlignInfo& tCHoleAlignData,
	const CString strModelID, const CString strLotID, const CString strPanelID, const CString strDrive, int nRatio,
	ENUM_INSPECT_MODE eInspMode, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], bool bUseInspect, int nStageNo)
{
	resultPanelData.m_ResultHeader.SetInspectEndTime();

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)<Judge> 流程开始."), strPanelID);

	//UI-检查运行检查
	//仅在执行检查操作时运行
	//仅Grab不起作用(无不良)
	if (bUseInspect)
	{
		//Point->Line判定
		//以后,稳定Line算法时,删除函数
//JudgementPointToLine(WrtResultInfo, resultPanelData, nImageW, nImageH);
//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge Point To Line End. Panel ID : %s"), strPanelID);

		//删除报告
// 
// 
// //2023/09/21  hjf  refix report 
//JudgementDelReport(resultPanelData);

/////////////////////////////////////hjf 09/21
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Del Report. Panel ID : %s"), strPanelID);

		//JudgementSpot(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Spot. Panel ID : %s"), strPanelID);

				//PNZ:17.11.27先删除重复部分,然后进行Line Merge
				//获取参数
		double* dAlignPara = theApp.GetAlignParameter(0);

		//Delet Some Line Defect判定
		//删除钢线周围的弱线
		JudgementDeletLineDefect(resultPanelData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Del Line Defect End. Panel ID : %s - %d"), strPanelID, resultPanelData.m_ListDefectInfo.GetCount());

		// Same Pattern Defect Merge
		JudgementSamePatternDefectMerge(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Same Pattern Defect Merge End. Panel ID : %s"), strPanelID);

		// Line Classification
		JudgementClassifyLineDefect(resultPanelData, MatOrgImage);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Line Defect Classification End. Panel ID : %s"), strPanelID);

		// 6.39QHD Notch Y Line Delete
		JudgementNotchDefect(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - NotchLine Defect End. Panel ID : %s - %d"), strPanelID, resultPanelData.m_ListDefectInfo.GetCount());

		//Crack判定
		JudgementCrack(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Crack Pattern Judge End. Panel ID : %s"), strPanelID);

		//PCD CRACK判定(非检查区域5.99")
		JudgementPCDCrack(resultPanelData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - PCD Crack End. Panel ID : %s"), strPanelID);

		//清除2018-07-31 Camera Tap中的过检
		JudgementCameraTapOverKill(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Camera Tap OverKill Delet End. Panel ID : %s"), strPanelID);

		//DGS判定
		JudgementNewDGS(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - DGS End. Panel ID : %s"), strPanelID);

		JudgementDGS_Vth(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - DGS Vth End. Panel ID : %s"), strPanelID);

		// Weak Plan B Test
		//JudgementSpecialGrayLineDefect(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Plan B End. Panel ID : %s"), strPanelID);

		// 2018-03-16 ver. 0.0.3 PNZ
				//气泡消除算法
		DeleteOverlapDefect_DimpleDelet(resultPanelData, MatOrgImage, dAlignPara);		//choikwangil boe 11临时主席19 08 09
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete Dimple Overkill Defect End. Panel ID : %s"), strPanelID);

		// 2018-04-19 ver. 0.0.3 PNZ
				//限制内Spot删除算法
				//重复数据删除后退

				//正在开发
		//DeleteOverlapDefect_BlackHole(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete Black Hole Spot Overkill Defect End. Panel ID : %s"), strPanelID);

		// 2018-05-17 ver. 0.0.1 PNZ
				//暗点确认算法
		DeleteOverlapDefect_DustDelet(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete Dust Defect End. Panel ID : %s"), strPanelID);

		// 2018-04-19 ver. 0.0.1 PNZ
				//限制内Spot删除算法
		DeleteOverlapDefect_BlackSmallDelet(resultPanelData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete Black Pattern Small Point Defect End. Panel ID : %s"), strPanelID);

		//清除white spot前判断
//Judge_Defect_BP_WSpot(resultPanelData, dAlignPara);

		DeleteOverlapDefect_SpecInSpotDelet(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete Spot Overkill Defect End. Panel ID : %s"), strPanelID); //05_26 choi white mura转为重复判定前



		JudgementDUSTDOWNDefect(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete DustDown Defect End. Panel ID : %s"), strPanelID);

		JudgementZARADefect(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete ZARA Defect End. Panel ID : %s"), strPanelID);

		JudgementPSMuraBrightPointDefect(resultPanelData, MatOrgImage, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete PSMura BrightPoint Defect End. Panel ID : %s"), strPanelID);


		//删除重复坐标
		DeleteOverlapDefect(resultPanelData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Delete Overlap Defect End. Panel ID : %s"), strPanelID);

		//choikwangil
		//Judge_DefectSize(resultPanelData, dAlignPara);
				//基于Black Pattern的Merge判定
		JudgementBlackPatternMerge(resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Black Pattern Merge End. Panel ID : %s"), strPanelID);

		//17.09.11-混色较多->数量较多时判定ET
		//在内部生成End日志
		JudgementET(resultPanelData, dAlignPara, strPanelID);

		//17.09.11-边缘部分亮点->Pad Bright判定
//JudgementEdgePadBright(resultPanelData, dAlignPara);
//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Edge Pad Bright End. Panel ID : %s"), strPanelID);

		JudgeCHoleJudgment(resultPanelData, tCHoleAlignData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - CHole Judgement End. Panel ID : %s"), strPanelID);

		//计算设置的Panel Size-Resolution
		double dResolution = calcResolution(WrtResultInfo);

		//17.09.25-判定相邻&群集
//JudgeGroup(resultPanelData, MatDrawBuffer, dAlignPara, dResolution);
//JudgeGroupTEST(resultPanelData, MatDrawBuffer, dAlignPara, dResolution);
		JudgeGroupJudgment(resultPanelData, MatDrawBuffer, dAlignPara, dResolution);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Group End. Panel ID : %s"), strPanelID);

		// PNZ 20.04.09 Point G64 GDS Judge
		//JudgePointGDStoFgarde(resultPanelData, dAlignPara);

				//PNZ 19.04.15 Mura Normal分类
		//JudgementMuraNormalClassification(resultPanelData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - MuraNormal Classification End. Panel ID : %s"), strPanelID);

				//添加PNZ 19.08.15 Mura Normal Type 3 Filtering+PNZ 19.09.20 CHole周围的T3 Filtering
		JudgementMuraNormalT3Filter(resultPanelData, tCHoleAlignData, dAlignPara);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - MuraNormal T3 Filtering End. Panel ID : %s"), strPanelID);

		//	金亨柱18.12.06
		//	MergeTool无条件操作,与Falg无关
		{
			resultPanelData.m_ResultHeader.MERGE_nRatio = nRatio;
			resultPanelData.m_ResultHeader.MERGE_rcAlignCellROI = m_stThrdAlignInfo.rcAlignCellROI;
			WorkCoordCrt tmpWorkCoordCrt = WrtResultInfo.GetWorkCoordCrt();
			resultPanelData.m_ResultHeader.MERGE_dPanelSizeX = (float)tmpWorkCoordCrt.dPanelSizeX;
			resultPanelData.m_ResultHeader.MERGE_dPanelSizeY = (float)tmpWorkCoordCrt.dPanelSizeY;
			resultPanelData.m_ResultHeader.MERGE_nWorkDirection = tmpWorkCoordCrt.nWorkDirection;
			resultPanelData.m_ResultHeader.MERGE_nWorkOriginPosition = tmpWorkCoordCrt.nWorkOriginPosition;
			resultPanelData.m_ResultHeader.MERGE_nWorkOffsetX = tmpWorkCoordCrt.nWorkOffsetX;
			resultPanelData.m_ResultHeader.MERGE_nWorkOffsetY = tmpWorkCoordCrt.nWorkOffsetY;
			resultPanelData.m_ResultHeader.MERGE_nDataDirection = tmpWorkCoordCrt.nDataDirection;
			resultPanelData.m_ResultHeader.MERGE_nGateDataOriginPosition = tmpWorkCoordCrt.nGateDataOriginPosition;
			resultPanelData.m_ResultHeader.MERGE_nGateDataOffsetX = tmpWorkCoordCrt.nGateDataOffsetX;
			resultPanelData.m_ResultHeader.MERGE_nGateDataOffsetY = tmpWorkCoordCrt.nGateDataOffsetY;
			resultPanelData.m_ResultHeader.MERGE_dGatePitch = (int)tmpWorkCoordCrt.dGatePitch;
			resultPanelData.m_ResultHeader.MERGE_dDataPitch = (int)tmpWorkCoordCrt.dDataPitch;
		}

		//17.11.30-AVI&SVI外围信息存储路径	
		wchar_t strContoursPath[MAX_PATH] = { 0 };
		CString strResultPath = RESULT_PATH;
		CString strMergeToolPath = MERGETOOL_PATH;
		wchar_t strContoursPath_Folder[MAX_PATH] = { 0 }; // 创建JSO-Result文件夹

		// Contours.Merge文件路径设置180322YSS
		swprintf(strContoursPath, _T("%s\\%s\\Contours.Merge"), (LPCWSTR)strResultPath, (LPCWSTR)strPanelID);
		swprintf(strContoursPath_Folder, _T("%s\\%s\\"), (LPCWSTR)strResultPath, (LPCWSTR)strPanelID);

		//17.11.29-保存AVI&SVI外围信息(AVI&SVI其他工具)
		//if (GetFileAttributes(strContoursPath_Folder) == -1)
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Can't find Folder : %s"), strContoursPath_Folder);
		//else
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Found Folder : %s"), strContoursPath_Folder);

		if (GetFileAttributes(strContoursPath_Folder) == -1)//JSO-再次创建文件夹
		{
			SHCreateDirectoryEx(NULL, strContoursPath_Folder, NULL);
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Created Folder : %s"), strContoursPath_Folder);
		}
		//else
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Exist Folder : %s"), strContoursPath_Folder);

		JudgeSaveContours(resultPanelData, strContoursPath);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Save Contours End. Panel ID : %s"), strPanelID);

		// 需要在JudgeSaveContours()之后调用，merge依赖缺陷的轮廓信息
		//ApplyMergeRule(resultPanelData);

		//if (eInspMode == eAutoRun && theApp.GetMergeToolUse())
		//{
		//	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("eInspMode = %d, MergeToolFlag = %d, Panel ID : %s"), eInspMode, theApp.GetMergeToolUse(), strPanelID); // JSO0604-复制其余工具文件
		//}
		//CString strDest;
		//strDest.Format(_T("%s\\%s\\"), MERGETOOL_PATH, strPanelID);

		//if (GetFileAttributes(strDest) == -1)//JSO-再次创建文件夹
		//{
		//	SHCreateDirectoryEx(NULL, strDest, NULL);
		//	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Created Folder : %s"), strDest);
		//}
		//else
		//	theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Exist Folder : %s"), strDest);

		//CString strContoursDest = strDest + _T("Contours.Merge");
		//bool nRet = CopyFile(strContoursPath, strContoursDest, FALSE);

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("MergeTool Data Copy Result = %d. Panel ID : %s"), nRet, strPanelID); // JSO0604-复制其余工具文件

		//存储Mura轮廓信息的路径
		swprintf(strContoursPath, _T("%s\\%s\\mura.coord"), (LPCWSTR)strResultPath, (LPCWSTR)strPanelID);

		//保存Mura轮廓信息
		JudgeSaveMuraContours(resultPanelData, strContoursPath);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Save Mura Contours End. Panel ID : %s"), strPanelID);
	}
	//	金亨柱18.12.05
//else
//{
//	wchar_t strContoursPath[MAX_PATH] = { 0 };

	//	// Contours.Merge文件路径设置180322YSS
//	swprintf(strContoursPath, _T("%s\\%s\\Contours.Merge"), MERGETOOL_PATH, strPanelID);

//	JudgeSaveContours(resultPanelData, strContoursPath);
//}

	//修复报告的坐标？
	JudgementRepair(strPanelID, resultPanelData, WrtResultInfo);
	//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Repair End. Panel ID : %s"), strPanelID);


	//17.07.12添加Panel Grade判定
	JudgementPanelGrade(resultPanelData);
	//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Judge - PanelGrade End. Panel ID : %s"), strPanelID);

	//按工作坐标计算和不良顺序编号
	NumberingDefect(strModelID, strPanelID, strLotID, WrtResultInfo, resultPanelData, nRatio);

	//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - Numbering Defect End. Panel ID : %s"), strPanelID);

	//17.12.19同一位置重复检测故障仅在轻/重警报-CCD故障-Auto-Run时有效
	if (eInspMode == eAutoRun)
	{
		//JudgementRepeatCount(strPanelID, resultPanelData);
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Judge - RepeatCount End. Panel ID : %s"), strPanelID);
	}

	//适用上位报告规则(特定不良过滤,选定代表不良)
	ApplyReportRule(resultPanelData);

	//	//根据结果判定CELL:NG/OK

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)<Judge> 流程结束. PNL等级 : %s"), strPanelID, resultPanelData.m_ResultPanel.Judge);

	return true;
}

bool AviInspection::Judgement_AI(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, cv::Mat(*MatDrawBuffer)[MAX_CAMERA_COUNT], tCHoleAlignInfo& tCHoleAlignData,
	const CString strModelID, const CString strLotID, const CString strPanelID, const CString strDrive, int nRatio,
	ENUM_INSPECT_MODE eInspMode, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], bool bUseInspect, int nStageNo)
{

	int nTotalblob = resultPanelData.m_ListDefectInfo.GetCount();
	/**
	 * 获取当前的AI模型ID、模型参数、需要复判的画面以及模型Code对应的缺陷Judge
	 * 
	 * \param WrtResultInfo
	 * \param resultPanelData
	 * \param MatDrawBuffer
	 * \param tCHoleAlignData
	 * \param strModelID
	 * \param strLotID
	 * \param strPanelID
	 * \param strDrive
	 * \param nRatio
	 * \param eInspMode
	 * \param MatOrgImage
	 * \param bUseInspect
	 * \param nStageNo
	 * \return 
	 */
	int nImageNum{0}, nCameraNum_{0}, nROINumber{0}, nAlgorithmNumber_{0};
	double* dAlgPara = theApp.GetAlgorithmParameter(nImageNum, nCameraNum_, nROINumber, nAlgorithmNumber_);
	double* dAlignPara = theApp.GetAlignParameter(nCameraNum_);
	stDefectInfo* pResultBlob = new stDefectInfo(MAX_DEFECT_COUNT, nImageNum);
	//AI
	AIReJudgeParam AIParam = GetAIParam(dAlgPara, nAlgorithmNumber_, theApp.GetImageClassify(nImageNum));
	//2024.05.07 for develop
	if (AIParam.AIEnable && nTotalblob > 0 && nTotalblob < 30 && false) {
		CString strAIResultFile = _T("");
		strAIResultFile.Format(_T("%s\\%s\\AI_Result\\%s_AI.csv"),
			INSP_PATH, strPanelID, strPanelID);

		CString strAIResultPath = _T("");
		strAIResultPath.Format(_T("%s\\%s\\AI_Result"), INSP_PATH, strPanelID);
		wchar_t wstrAIPath[MAX_PATH] = { 0, };
		swprintf(wstrAIPath, _T("%s\\"), (LPCWSTR)strAIResultPath);
		CreateDirectory(wstrAIPath, NULL);


		std::vector<TaskInfoPtr> taskList;
		int dicsRatio = (int)dAlignPara[E_PARA_AVI_DICS_IMAGE_RATIO];
		int cropExpand = (int)dAlignPara[E_PARA_AVI_DICS_AI_CROP_EXPAND];
		cv::Mat MatJudge = cv::imread("./01_WHITE_CAM00.bmp");
		PrepareAITask(MatJudge, dicsRatio, cropExpand, pResultBlob, AIParam, strPanelID, nImageNum, nAlgorithmNumber_, taskList);


		for (auto task : taskList)
		{
			GetAIRuntime()->CommitInferTask(task);
		}


		for (auto task : taskList)
		{
			std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
			std::future<ModelResultPtr> futureRst = promiseResult->get_future();
			ModelResultPtr rst = futureRst.get();

			AIInfoPtr aiInfoPtr = std::static_pointer_cast<STRU_AI_INFO>(task->inspParam);
			//写下ResultBlob
			for (int i = 0; i < aiInfoPtr->defectNoList.size(); i++)
			{
				int di = aiInfoPtr->defectNoList[i];
				pResultBlob->AI_ReJudge_Code[di] = rst->itemList[i][0].code;
				pResultBlob->AI_ReJudge_Conf[di] = rst->itemList[i][0].confidence;
				pResultBlob->AI_ReJudge[i] = AIParam.rejudge;
				//
				if (pResultBlob->AI_ReJudge_Code[di] == 0 && pResultBlob->AI_ReJudge_Conf[di] >= AIParam.confidence)
				{
					pResultBlob->AI_ReJudge_Result[di] = 0; // 
				}
				else if (pResultBlob->AI_ReJudge_Code[di] == 1 && pResultBlob->AI_ReJudge_Conf[di] >= AIParam.confidence)
				{
					pResultBlob->AI_ReJudge_Result[di] = 1; // 
				}
				else {
					pResultBlob->AI_ReJudge_Result[di] = 2;	// 
				}
			}

			//
			SaveInferenceResultMult(strAIResultFile, strAIResultPath, AIParam, rst->itemList, rst->taskInfo);
			delete promiseResult;
		}

		////////AI Detect for After filter log
		//theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("%s AI Algorithm filter. PanelID: %s, CAM: %02d ROI: %02d, Img: %s.\n\t\t\t\t( After AI FilterNum: %d )"),
		//	theApp.GetAlgorithmName(nAlgorithmNumber_), strPanelID_, nCameraNum_, nROINumber, theApp.GetGrabStepName(nImageNum), nTotalblob);
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("(%s)<Judge> AI 复判流程结束. PNL等级 : %s"), strPanelID, resultPanelData.m_ResultPanel.Judge);
	}

	return true;
}

bool AviInspection::JudgementPointToLine(CWriteResultInfo WrtResultInfo, ResultPanelData& resultPanelData, const int nImageWidth, const int nImageHeight)
{
	int nPointCount = 20;   // 在同一坐标下判定Point不良,设置数以上的Line
	int nOffSet = 9;        // 在同一坐标下确认左右(nOffSet/2)(奇数设置)

	//ex)9:左右4 Pixel
	int nHalfnOffSet = nOffSet / 2;

	//水平,垂直投影
	int* nProjectionX = new int[nImageWidth];
	int* nProjectionY = new int[nImageHeight];

	//初始化数组
	memset(nProjectionX, 0, sizeof(int) * nImageWidth);
	memset(nProjectionY, 0, sizeof(int) * nImageHeight);

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//不包括Point不良
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type < E_DEFECT_JUDGEMENT_POINT_DARK)
			continue;

		if (resultPanelData.m_ListDefectInfo[i].Defect_Type > E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			continue;

		//增加对应的坐标1
		nProjectionX[(int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X]++;
		nProjectionY[(int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y]++;
	}

	CArray<CRect> listTempX;
	listTempX.RemoveAll();

	//如果有多个相同的x坐标nPointCount
	for (int i = 0; i < nImageWidth; i++)
	{
		//添加不良列表编号
		if (nProjectionX[i] > nPointCount)
		{
			//首次添加
			if (listTempX.GetCount() == 0)
			{
				listTempX.Add(CRect(i - nHalfnOffSet, 0, i + nHalfnOffSet, nOffSet));
			}
			else
			{
				//导入列表
				CRect rectTemp = listTempX[listTempX.GetCount() - 1];

				//修改连续
				if (i - rectTemp.right < nHalfnOffSet)
				{
					//更改X长度
					rectTemp.right = i + nHalfnOffSet;

					//放入结果
					listTempX[listTempX.GetCount() - 1] = rectTemp;
				}
				//添加非连续
				else
				{
					listTempX.Add(CRect(i - nHalfnOffSet, 0, i + nHalfnOffSet, nOffSet));
				}
			}
		}
	}

	CArray<CRect> listTempY;
	listTempY.RemoveAll();

	//如果有多个相同的y坐标nPointCount
	for (int i = 0; i < nImageHeight; i++)
	{
		//添加不良列表编号
		if (nProjectionY[i] > nPointCount)
		{
			//首次添加
			if (listTempY.GetCount() == 0)
			{
				listTempY.Add(CRect(0, i - nHalfnOffSet, nOffSet, i + nHalfnOffSet));
			}
			else
			{
				//导入列表
				CRect rectTemp = listTempY[listTempY.GetCount() - 1];

				//修改连续
				if (i - rectTemp.bottom < nHalfnOffSet)
				{
					//更改Y长度
					rectTemp.bottom = i + nHalfnOffSet;

					//放入结果
					listTempY[listTempY.GetCount() - 1] = rectTemp;
				}
				//添加非连续
				else
				{
					listTempY.Add(CRect(0, i - nHalfnOffSet, nOffSet, i + nHalfnOffSet));
				}
			}
		}
	}

	//如果nPointCount或更高版本有一个相同的坐标
	if (listTempX.GetCount() != 0 || listTempY.GetCount() != 0)
	{
		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//不包括Point不良
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type < E_DEFECT_JUDGEMENT_POINT_DARK)
			{
				i++;		continue;
			}	//修改无限循环

			if (resultPanelData.m_ListDefectInfo[i].Defect_Type > E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			{
				i++;		continue;
			}	//修改无限循环

			bool bDelete = false;

			//相同的x坐标
			for (int j = 0; j < listTempX.GetCount(); j++)
			{
				if (listTempX[j].left > resultPanelData.m_ListDefectInfo[i].Pixel_Center_X ||
					listTempX[j].right < resultPanelData.m_ListDefectInfo[i].Pixel_Center_X)
					continue;

				//导入列表
				CRect rectTemp = listTempX[j];

				//初始设置
				if (rectTemp.top == 0)
				{
					rectTemp.top = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					rectTemp.bottom = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;
				}
				//设置长度
				else
				{
					if (rectTemp.top > resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y)
						rectTemp.top = (LONG)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

					else if (rectTemp.bottom < resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y)
						rectTemp.bottom = (LONG)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;
				}

				//放入结果
				listTempX[j] = rectTemp;

				bDelete = true;
			}

			//相同的y坐标
			for (int j = 0; j < listTempY.GetCount(); j++)
			{
				if (listTempY[j].top > resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y ||
					listTempY[j].bottom < resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y)
					continue;

				//导入列表
				CRect rectTemp = listTempY[j];

				//初始设置
				if (rectTemp.left == 0)
				{
					rectTemp.left = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					rectTemp.right = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
				}
				//设置长度
				else
				{
					if (rectTemp.left > resultPanelData.m_ListDefectInfo[i].Pixel_Center_X)
						rectTemp.left = (LONG)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;

					else if (rectTemp.right < resultPanelData.m_ListDefectInfo[i].Pixel_Center_X)
						rectTemp.right = (LONG)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				}

				//放入结果
				listTempY[j] = rectTemp;

				bDelete = true;
			}

			//Point->如果Line出现故障
			if (bDelete)
			{
				//删除相应的错误
				resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}
			//下一个不良
			else
			{
				i++;
			}
		}

		ResultDefectInfo* pResultData;

		//相同的x坐标
		for (int j = 0; j < listTempX.GetCount(); j++)
		{
			pResultData = new ResultDefectInfo;
			//在Judgement中Defect Number没有意义
//pResultData.Defect_No			= 0;	// ??
			_tcscpy_s(pResultData->Defect_Code, _T("CODE"));
			pResultData->Pixel_Start_X = (int)listTempX[j].left;
			pResultData->Pixel_Start_Y = (int)listTempX[j].top;
			pResultData->Pixel_End_X = (int)listTempX[j].right;
			pResultData->Pixel_End_Y = (int)listTempX[j].bottom;
			pResultData->Pixel_Center_X = (int)listTempX[j].left + listTempX[j].Width() / 2;
			pResultData->Pixel_Center_Y = (int)listTempX[j].top + listTempX[j].Height() / 2;
			pResultData->Defect_Size_Pixel = (int)(listTempX[j].Width() * listTempX[j].Height());
			pResultData->Defect_Size = (int)(pResultData->Defect_Size_Pixel * WrtResultInfo.GetCamResolution(0));
			pResultData->Img_Number = theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_GRAY_128);		// 临时使用G128映像。如果没有图像编号,则在剪切碎片图像时出现错误
			pResultData->Defect_MeanGV = 0;	// 合并后平均无法获得GV。			
			//pResultData->Defect_Img_Name		= _T("");
			pResultData->Img_Size_X = (DOUBLE)listTempX[j].Width();		// 临时不良画面宽度大小
			pResultData->Img_Size_Y = (DOUBLE)listTempX[j].Height();	// 临时不良影像垂直大小
			pResultData->Defect_Type = E_DEFECT_JUDGEMENT_LINE_X_OPEN;
			//在Defect结果图像上绘制时,必须选择使用哪个图像。0应用时出错
			pResultData->Camera_No = 0;

			resultPanelData.Add_DefectInfo(*pResultData);
			SAFE_DELETE(pResultData);
		}

		//相同的y坐标
		for (int j = 0; j < listTempY.GetCount(); j++)
		{
			pResultData = new ResultDefectInfo;
			//在Judgement中Defect Number没有意义
//pResultData.Defect_No			= 0;	// ??
			_tcscpy_s(pResultData->Defect_Code, _T("CODE"));
			pResultData->Pixel_Start_X = (int)listTempY[j].left;
			pResultData->Pixel_Start_Y = (int)listTempY[j].top;
			pResultData->Pixel_End_X = (int)listTempY[j].right;
			pResultData->Pixel_End_Y = (int)listTempY[j].bottom;
			pResultData->Pixel_Center_X = (int)listTempY[j].left + listTempY[j].Width() / 2;
			pResultData->Pixel_Center_Y = (int)listTempY[j].top + listTempY[j].Height() / 2;
			pResultData->Defect_Size_Pixel = (int)(listTempY[j].Width() * listTempY[j].Height());
			pResultData->Defect_Size = (int)(pResultData->Defect_Size_Pixel * WrtResultInfo.GetCamResolution(0));
			pResultData->Img_Number = theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_GRAY_128);		// 临时使用G128映像。如果没有图像编号,则在剪切碎片图像时出现错误
			pResultData->Defect_MeanGV = 0;	// 合并后平均无法获得GV。
			//pResultData->Defect_Img_Name	= _T("");
			pResultData->Img_Size_X = 0;	// 当前无替代变量
			pResultData->Img_Size_Y = 0;	// 当前无替代变量
			pResultData->Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT;
			//在Defect结果图像上绘制时,必须选择使用哪个图像。0应用时出错
			pResultData->Camera_No = 0;

			resultPanelData.Add_DefectInfo(*pResultData);
			SAFE_DELETE(pResultData);
		}

		listTempX.RemoveAll();
		listTempY.RemoveAll();
	}
	return true;
}

bool AviInspection::JudgeGroupJudgment(ResultPanelData& resultPanelData, cv::Mat(*MatDraw)[MAX_CAMERA_COUNT], double* dAlignPara, double dResolution)
{
	//////////////////////////////////////////////////////////////////////////////////////////////////
		//PNZ-2019.01.11最新规格。
		//Spect说明
		// 1. 按RGB Pattern将POINT Dark不良按Size(Area)分为1,2,3,检测不良统一为Dark Point。
		// 2. 案例分为3种
		// 3. Parameter们
		//		-Count			:Pattern中的不良数量管理,超过时均判定为组
		//		-Distance		:邻近不良范围(半径)
		//		-Delete On/Off:删除不属于该Pattern Group的错误
	//////////////////////////////////////////////////////////////////////////////////////////////////

		//异常处理
	if (dAlignPara == NULL)	return false;
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)	return true;
	if (dAlignPara[E_PARA_GROUP_FLAG] <= 0)	return true;

	//初始化Common Parameter
	int		nDefect1_Count = 0;
	int		nDefect2_Count = 0;
	int		nDefect3_Count = 0;
	int		nDefect_PD_Count = 0;
	//添加	////////////////////////////////////////////choikwangil
		//所有group judger后使用
	int		nAll_Count = 0;
	int		nAll_Dark_Count = 0;
	int		nAll_Bright_Count = 0;

	////////////////////////////////////
	float	nGroupDistance_1st = 0;
	float	nGroupDistance_2nd = 0;

	// Case Parameter
	int		nCount_SP_1 = (int)dAlignPara[E_PARA_GROUP1_COUNT];
	float	nDistance_SP_1 = (float)dAlignPara[E_PARA_GROUP1_DIAMETER];
	int		nDeleteOnOff_SP_1 = (int)dAlignPara[E_PARA_GROUP1_DELETE];
	int		nCount_SP_2 = (int)dAlignPara[E_PARA_GROUP2_COUNT];
	float	nDistance_SP_2 = (float)dAlignPara[E_PARA_GROUP2_DIAMETER];
	int		nDeleteOnOff_SP_2 = (int)dAlignPara[E_PARA_GROUP2_DELETE];
	int		nCount_SP_3 = (int)dAlignPara[E_PARA_GROUP3_COUNT];
	float	nDistance_SP_3 = (float)dAlignPara[E_PARA_GROUP3_DIAMETER];
	int		nDeleteOnOff_SP_3 = (int)dAlignPara[E_PARA_GROUP3_DELETE];

	//添加	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//int		nAll_Dark_Judge = (int)dAlignPara[E_PARA_LASTGROUP_JUDGE_DARK_COUNT];
	int		nCount_All_Judge = (int)dAlignPara[E_PARA_LASTGROUP_JUDGE_ALL_COUNT];
	int		nAll_Bright_Judge = (int)dAlignPara[E_PARA_LASTGROUP_JUDGE_BRIGHT_COUNT];
	int		nAll_Dark_Judge = (int)dAlignPara[E_PARA_LASTGROUP_JUDGE_DARK_COUNT];
	/////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Count Group Judgment

	// Defect Number Count
	bool	bGroupJudgment_All_Count = false;
	bool	bGroupJudgment_Count = false;
	bool	bGroupJudgment_DarkBright_Count = false;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

		switch (nDefectType)
		{
		case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1: nDefect1_Count++; break;
		case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2: nDefect2_Count++; break;
		case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3: nDefect3_Count++; break;
		case E_DEFECT_JUDGEMENT_POINT_DARK:		 nDefect_PD_Count++; break;

		case E_DEFECT_JUDGEMENT_POINT_BRIGHT:		nAll_Bright_Count++; break;
		case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:	nAll_Bright_Count++; break;
		case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:	nAll_Bright_Count++; break;
		}
	}

	nAll_Dark_Count = nDefect1_Count + nDefect2_Count + nDefect3_Count + nDefect_PD_Count;
	nAll_Count = nDefect1_Count + nDefect2_Count + nDefect3_Count + nAll_Bright_Count + nDefect_PD_Count;

	if (nAll_Count >= nCount_All_Judge)		bGroupJudgment_All_Count = true;
	else								bGroupJudgment_All_Count = false;

	if (bGroupJudgment_All_Count == true)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			switch (nDefectType)
			{
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:
			case E_DEFECT_JUDGEMENT_POINT_DARK:

				nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK;
				break;

			case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
			case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
			case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:

				nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT;
				break;
			}
			resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;
		}
	}
	else {

		if (nAll_Dark_Count >= nAll_Dark_Judge) {
			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{
				int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

				switch (nDefectType)
				{
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:
				case E_DEFECT_JUDGEMENT_POINT_DARK:

					nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK;
					break;

				case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:

					nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT;
					break;
				}
				resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;
			}
			bGroupJudgment_DarkBright_Count = true;
		}
		else if (nAll_Bright_Count >= nAll_Bright_Judge) {

			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{
				int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

				switch (nDefectType)
				{
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:
				case E_DEFECT_JUDGEMENT_POINT_DARK:

					nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK;
					break;

				case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:

					nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT;
					break;
				}
				resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;
			}
			bGroupJudgment_DarkBright_Count = true;
		}

	}

	///////////////////////////////////////////////////////////////////////////////////////
		//检查是否大于设置数量
	if (nDefect1_Count >= nCount_SP_1 || nDefect2_Count >= nCount_SP_2 || nDefect3_Count >= nCount_SP_3)	bGroupJudgment_Count = true;
	else																									bGroupJudgment_Count = false;

	if (bGroupJudgment_Count == true && bGroupJudgment_DarkBright_Count == false && bGroupJudgment_All_Count == false)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			switch (nDefectType)
			{
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:

				nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK;
				break;

			}
			resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Distance Group Judgement

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

		//不包括非Point不良
		if (nDefectType < E_DEFECT_JUDGEMENT_POINT_DARK || nDefectType > E_DEFECT_JUDGEMENT_POINT_DARK_SP_3) continue;

		//参数初始化
		nGroupDistance_1st = 0;

		//如果POINT Dark有问题,请选择相应的Distance
		if (nDefectType == E_DEFECT_JUDGEMENT_POINT_DARK_SP_1) nGroupDistance_1st = nDistance_SP_1;
		if (nDefectType == E_DEFECT_JUDGEMENT_POINT_DARK_SP_2) nGroupDistance_1st = nDistance_SP_2;
		if (nDefectType == E_DEFECT_JUDGEMENT_POINT_DARK_SP_3) nGroupDistance_1st = nDistance_SP_3;

		// Temp Defect Center in MM
		CPoint ptTemp((int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio,
			(int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); j++)
		{
			if (j == i) continue;

			int nDefectType2 = resultPanelData.m_ListDefectInfo[j].Defect_Type;

			//不包括非Point故障
			if (nDefectType2 < E_DEFECT_JUDGEMENT_POINT_DARK || nDefectType2 > E_DEFECT_JUDGEMENT_POINT_DARK_SP_3) continue;

			//参数初始化
			nGroupDistance_2nd = 0;

			//如果POINT Dark不好,请选择相应的Distance
			if (nDefectType2 == E_DEFECT_JUDGEMENT_POINT_DARK_SP_1) nGroupDistance_2nd = nDistance_SP_1;
			if (nDefectType2 == E_DEFECT_JUDGEMENT_POINT_DARK_SP_2) nGroupDistance_2nd = nDistance_SP_2;
			if (nDefectType2 == E_DEFECT_JUDGEMENT_POINT_DARK_SP_3) nGroupDistance_2nd = nDistance_SP_3;

			// Compair Defect Center in MM
			CPoint ptTemp2((int)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio,
				(int)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

			// dX, dY in MM
			float fTempX = (float)(ptTemp2.x - ptTemp.x);
			float fTempY = (float)(ptTemp2.y - ptTemp.y);

			//以坏组为中心的长度(MM)
			float fTempLength = (float)(sqrt(fTempX * fTempX + fTempY * fTempY) * dResolution);

			//组判定条件
			bool bGroupJudgment_Distance = false;

			//比较不良之一是Dark SP123时
			if (nGroupDistance_1st + nGroupDistance_2nd > 0)
			{
				if (fTempLength >= nGroupDistance_1st && fTempLength >= nGroupDistance_2nd)	bGroupJudgment_Distance = false;
				else if (fTempLength >= nGroupDistance_1st && fTempLength < nGroupDistance_2nd)	bGroupJudgment_Distance = true;
				else if (fTempLength < nGroupDistance_1st && fTempLength >= nGroupDistance_2nd)	bGroupJudgment_Distance = true;
				else if (fTempLength < nGroupDistance_1st && fTempLength < nGroupDistance_2nd)	bGroupJudgment_Distance = true;
			}

			//比较不良均为Dark SP123以外的不良
			else if (nGroupDistance_1st + nGroupDistance_2nd == 0)
			{
				//基于最大距离3
				if (fTempLength <= nDistance_SP_3) bGroupJudgment_Distance = true;
			}

			else
				continue;

			if (bGroupJudgment_Distance == true && nGroupDistance_1st + nGroupDistance_2nd > 0)
			{
				//更改为不良名称组
				switch (nDefectType)
				{
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:

					nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK;
					break;
				}
				resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;

				switch (nDefectType2)
				{
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
				case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:

					nDefectType2 = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK;
					break;
				}
				resultPanelData.m_ListDefectInfo[j].Defect_Type = nDefectType2;
			}

			else if (bGroupJudgment_Distance == true && nGroupDistance_1st + nGroupDistance_2nd == 0)
			{
				//更改为不良名称组
				switch (nDefectType)
				{
				case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:

					nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT;
					break;
				}
				resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;

				switch (nDefectType2)
				{
				case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
				case E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK:

					nDefectType2 = E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT;
					break;
				}
				resultPanelData.m_ListDefectInfo[j].Defect_Type = nDefectType2;
			}

			else
				continue;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Delete Dark POINT 123

		//如果不全部删除,即使是名字
	//if (nDeleteOnOff_SP_1 + nDeleteOnOff_SP_2 + nDeleteOnOff_SP_3 == 0) return true;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount();)
	{
		int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

		// Only Point Dark SP1,2,3
		if (nDefectType != E_DEFECT_JUDGEMENT_POINT_DARK_SP_1 && nDefectType != E_DEFECT_JUDGEMENT_POINT_DARK_SP_2 && nDefectType != E_DEFECT_JUDGEMENT_POINT_DARK_SP_3)
		{
			i++;
			continue;
		}

		// Delete On/Off
		int nDeleteDefect = 0;

		if (nDefectType == E_DEFECT_JUDGEMENT_POINT_DARK_SP_1) { nDeleteDefect = nDeleteOnOff_SP_1;	if (nDeleteDefect == 0)	nDefectType = E_DEFECT_JUDGEMENT_POINT_DARK; }		// Case 1 Defect
		else if (nDefectType == E_DEFECT_JUDGEMENT_POINT_DARK_SP_2) { nDeleteDefect = nDeleteOnOff_SP_2;							nDefectType = E_DEFECT_JUDGEMENT_POINT_DARK; }		// Case 2 Defect
		else if (nDefectType == E_DEFECT_JUDGEMENT_POINT_DARK_SP_3) { nDeleteDefect = nDeleteOnOff_SP_3;							nDefectType = E_DEFECT_JUDGEMENT_POINT_GROUP_DARK; }		// Case 3 Defect
		else	 nDeleteDefect = 0;

		//重命名Defect
		resultPanelData.m_ListDefectInfo[i].Defect_Type = nDefectType;

		//清除不良
		if (nDeleteDefect > 0) { resultPanelData.m_ListDefectInfo.RemoveAt(i); }
		else { i++; continue; }

	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	return true;
}

// // 		case E_DEFECT_JUDGEMENT_POINT_DARK:		 nAll_Dark_Count++; break;
// // 		case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1: nAll_Dark_Count++; break;
// // 		case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2: nAll_Dark_Count++; break;
// // 		case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3: nAll_Dark_Count++; break;

//////////////////////////////////////////////////////////////////////////
// PNZ 2019-04-15
//Mura Normal不良分类
//////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementMuraNormalClassification(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//异常处理
	if (dAlignPara == NULL)									return false;
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)	return true;

	//////////////////////////////////////////////////////////////////////////
		//按类型分类Classify Parameters
	int		nType2ClassPara = (int)dAlignPara[E_PARA_MURANORMAL_CLASSPARA_TYPE2]; // Count 5
	float	nType3ClassPara = (float)dAlignPara[E_PARA_MURANORMAL_CLASSPARA_TYPE3]; // Diff GV 23.0

	//////////////////////////////////////////////////////////////////////////
		//异常处理:检查各类型的错误数量
	int nMuraType1_Count = 0;
	int nMuraType2_Count = 0;
	int nMuraType3_Count = 0;
	int nMuraType4_Count = 0;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

		switch (nDefectType)
		{
		case E_DEFECT_JUDGEMENT_MURA_LINEMURA_X:	nMuraType2_Count++; break;
		case E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y:	nMuraType2_Count++; break;
		case E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG:		nMuraType3_Count++; break;
		}
	}

	if (nMuraType1_Count == 0 && nMuraType2_Count == 0 && nMuraType3_Count == 0 && nMuraType4_Count == 0) return true;

	//////////////////////////////////////////////////////////////////////////
		//Type 2不良分类:判定大于设置的条件
	if ((nMuraType2_Count >= nType2ClassPara) && nType2ClassPara != 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			//如果是Type 2 Mura,则Change Name
			if (nDefectType == E_DEFECT_JUDGEMENT_MURA_LINEMURA_X || nDefectType == E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y)
			{
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURANORMAL_TYPE2_F_GRADE;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//Type 3不良分类:判定大于设置的条件
	if (nMuraType3_Count != 0 && nType3ClassPara != 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int	nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			if (nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG)
			{
				bool bFgradMura = false;

				// Check Diff GV of Defect
				double	dbDefectMeanGV = resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
				int		nDefectMinGV = resultPanelData.m_ListDefectInfo[i].Defect_MinGV;

				if (dbDefectMeanGV - nDefectMinGV > nType3ClassPara) bFgradMura = true;

				//如果是Type 3 Mura,则ChangeName
				if (bFgradMura == true)	resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURANORMAL_TYPE3_F_GRADE;
				else                        continue;
			}

			else
				continue;
		}
	}

	return true;

}

//////////////////////////////////////////////////////////////////////////
// PNZ 2019-08-15
//Type 3过检改善方案:限量Cell检测Merge
//
//////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementMuraNormalT3Filter(ResultPanelData& resultPanelData, tCHoleAlignInfo tCHoleAlignData, double* dAlignPara)
{
	//异常处理
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)	return true;

	double	dbTHDiffGV_BBS = (double)dAlignPara[E_PARA_MURANORMAL_OR_TYPE3_BBS_DIFFGV];
	double	dbTHDiffGV_SBS = (double)dAlignPara[E_PARA_MURANORMAL_OR_TYPE3_SBS_DIFFGV];
	double	dbTHDiffGV_CH = (double)dAlignPara[E_PARA_MURANORMAL_OR_TYPE3_CH_DIFFGV];
	int		nBubbleMergePara = (int)dAlignPara[E_PARA_MURANORMAL_OR_TYPE3_BUBBLEMERGE];

	int		nOffSet = 200;

	//禁用功能Reture
	if (dbTHDiffGV_BBS == 0 && dbTHDiffGV_SBS == 0 && dbTHDiffGV_CH == 0 && nBubbleMergePara == 0) return true;

	//////////////////////////////////////////////////////////////////////////
		//移除Camera Hole周围的T3过检
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount();)
	{
		//重新获取值
		dbTHDiffGV_CH = (double)dAlignPara[E_PARA_MURANORMAL_OR_TYPE3_CH_DIFFGV];

		//验证是否使用了CHole Para
		if (dbTHDiffGV_CH <= 0) break;

		int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

		// Check Targut Defect
		if (nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_RING ||
			nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG ||
			nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL ||
			nDefectType == E_DEFECT_JUDGEMENT_MURANORMAL_TYPE3_F_GRADE)
		{
			bool bOverKill_Locatoin = false;
			bool bOverKill_GVCheck = false;

			// Check GV
			double dbDefectAvgGV = resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
			double dbDefectBKGV = resultPanelData.m_ListDefectInfo[i].Defect_BKGV;
			double dbDiffGV_CHole = abs(dbDefectAvgGV - dbDefectBKGV);

			//19.11.14添加CHole周边不良条件,小Size不良Diff设置高
			double dbDefectArea = resultPanelData.m_ListDefectInfo[i].Defect_Size_Pixel;

			if (dbDefectArea <= 3000) dbTHDiffGV_CH = 5;

			if (dbDefectArea >= 50000) { i++; continue; }

			if (dbDiffGV_CHole <= dbTHDiffGV_CH) bOverKill_GVCheck = true;

			//检查是否在CHole周围
			for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
			{
				// Defect Info
				int nDefect_ImgNo = resultPanelData.m_ListDefectInfo[i].Img_Number;
				int nDefect_AlgNo = theApp.GetImageClassify(nDefect_ImgNo);

				if (tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].empty())		continue;
				if (tCHoleAlignData.matCHoleROIBuf[nDefect_AlgNo][j].empty())	continue;

				int nX_Start = tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].x;
				int nX_End = tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].x + tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].width;
				int nY_Start = tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].y;
				int nY_End = tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].y + tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].height;

				int nDefect_CenterX = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nDefect_CenterY = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;
				//int nOffSet			= (int)(max(tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].width, tCHoleAlignData.rcCHoleROI[nDefect_AlgNo][j].height) / 2);

				int nXDCheck = 0;
				int nYDCheck = 0;

				if (nDefect_CenterX >= nX_Start - nOffSet && nDefect_CenterX <= nX_End + nOffSet) nXDCheck = 1;
				if (nDefect_CenterY >= nY_Start - nOffSet && nDefect_CenterY <= nY_End + nOffSet) nYDCheck = 1;

				//在CHole周围
				if (nXDCheck == 1 && nYDCheck == 1) { bOverKill_Locatoin = true;  break; }
			}

			//清除故障
			if (bOverKill_GVCheck == true && bOverKill_Locatoin == true)
				resultPanelData.m_ListDefectInfo.RemoveAt(i);

			else { i++; continue; }
		}

		else { i++; continue; }
	}

	//////////////////////////////////////////////////////////////////////////
	// Bubble Merge Logic

	if (nBubbleMergePara != 0)//必须使用
	{
		int nBubbleDefectCount = 0;

		// Check Defect Active Bubble
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			switch (nDefectType)
			{
			case E_DEFECT_JUDGEMENT_APP_ACTIVE_BUBBLE:
				nBubbleDefectCount++;
				break;
			}
		}

		if (nBubbleDefectCount != 0)//必须存在Bubble错误
		{
			// Merge
			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{

				int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

				if (nDefectType != E_DEFECT_JUDGEMENT_APP_ACTIVE_BUBBLE) continue;

				// Check Location
				int nBubbleX = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nBubbleY = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				int nBubbleWidth = (int)resultPanelData.m_ListDefectInfo[i].Pixel_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
				int nBubbleHeight = (int)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;

				int nBubbleRadian = (int)max(nBubbleWidth, nBubbleHeight) / 2;

				//检查是否在CHole周围
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
				{
					int nDefectType = resultPanelData.m_ListDefectInfo[j].Defect_Type;

					// Check Targut Defect
					if (nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_RING ||
						nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG ||
						nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL ||
						nDefectType == E_DEFECT_JUDGEMENT_MURANORMAL_TYPE3_F_GRADE)
					{
						int nT3CenterX = (int)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X;
						int nT3CenterY = (int)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y;

						int dbDiffX = (int)abs(nBubbleX - nT3CenterX);
						int dbDiffY = (int)abs(nBubbleY - nT3CenterY);

						double	dbDistance = (double)sqrt(dbDiffX * dbDiffX + dbDiffY * dbDiffY);

						if (dbDistance - nBubbleRadian <= nBubbleMergePara)
							resultPanelData.m_ListDefectInfo.RemoveAt(j);

						else { j++; continue; }
					}
					else { j++; continue; }
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// T3 SBS + BBS Diff GV Check Logic

		//限制判定
	bool bJudgeLimiteSample = false;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

		switch (nDefectType)
		{
		case E_DEFECT_JUDGEMENT_MURA_TYPE3_RING:
			bJudgeLimiteSample = true;
			break;
		}
	}

	//Type 3 BBS不良过滤(用于限制Cell)
	if (bJudgeLimiteSample == true)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount();)
		{
			double dbDefectAvgGV = resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
			double dbDefectBKGV = resultPanelData.m_ListDefectInfo[i].Defect_BKGV;

			double dbDiffGV = abs(dbDefectAvgGV - dbDefectBKGV);

			int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			if (nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_RING)
				resultPanelData.m_ListDefectInfo.RemoveAt(i);

			// BBS Judge On/Off
			if (dbTHDiffGV_BBS > 0)
			{
				if (nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG && dbDiffGV <= dbTHDiffGV_BBS)
					resultPanelData.m_ListDefectInfo.RemoveAt(i);

				else if (nDefectType == E_DEFECT_JUDGEMENT_MURANORMAL_TYPE3_F_GRADE && dbDiffGV <= dbTHDiffGV_BBS)
					resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}

			// SBS Judge On/Off
			if (dbTHDiffGV_SBS > 0)
			{
				if (nDefectType == E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL && dbDiffGV <= dbTHDiffGV_SBS)
					resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}
			else
			{
				i++;
				continue;
			}
		}
	}

	return true;
}

//	 2. 函数功能:	Algorithm DLL
//	 3. 参数:	
//	 4. 返回值:	完成:true
//				失败:false
//	 5. 创建日期:	16.02.29
//	 6. 作者:	
//	 7. 修改历史记录:
//	 8. 注意:算法集成

long AviInspection::StartLogicAlgorithm(const CString strDrive, const tLogicPara& LogicPara, cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE],
	cv::Mat& MatDrawBuffer, const int nImageNum, const int nROINumber, const int nAlgorithmNumber, tAlignInfo stThrdAlignInfo, ResultBlob_Total* pResultBlob_Total, bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtResultInfo, const double* _mtp)
{
	//setlocale(LC_ALL, ""); //使用当前计算机首选项中的语言信息 hjf
	//设置参数
	cv::Mat				MatOriginImage_ = LogicPara.MatOrgImage;
	int					nAlgorithmNumber_ = nAlgorithmNumber;					// 当前算法	
	int					nCameraNum_ = LogicPara.nCameraNum;				// 检查照相机
	int					nThrdIndex_ = LogicPara.nThreadLog;
	cv::Mat* MatOriImageRGB_[3];
	cv::Mat				MatBKG_ = LogicPara.MatBKG;					// 在Mura中使用(非Cell区域)/Dust,无黑色图案

	MatOriImageRGB_[0] = LogicPara.MatOrgRGBAdd[0];
	MatOriImageRGB_[1] = LogicPara.MatOrgRGBAdd[1];
	MatOriImageRGB_[2] = LogicPara.MatOrgRGBAdd[2];

	ENUM_KIND_OF_LOG	eLogCamera = (ENUM_KIND_OF_LOG)nCameraNum_;

	//当前面板ID
	CString				strPanelID_;
	strPanelID_.Format(_T("%s"), LogicPara.tszPanelID);

	long lErrorCode = E_ERROR_CODE_TRUE;

	//生成保存算法结果的文件夹
	SYSTEMTIME time;
	::GetLocalTime(&time);

	//获取单个算法检查参数
	double* dAlgPara = theApp.GetAlgorithmParameter(nImageNum, nCameraNum_, nROINumber, nAlgorithmNumber_);
	double* dAlignPara = theApp.GetAlignParameter(nCameraNum_);
	//公共参数(需要中间结果画面的参数)
	int nCommonPara[] = {
				MAX_DEFECT_COUNT											,	//00:最大不良数量(小于MAX_DEFECT_COUNT)
				theApp.GetCommonParameter()->bIFImageSaveFlag	,	//01:算法中间结果Image Save flag
				0													,	//02:画面存储顺序计数(Point的函数分为两个)
				theApp.GetImageClassify(nImageNum)				,	//03:当前Alg画面编号
				nCameraNum_																						,	//04:Cam Number(相机编号/SVI)
				nROINumber																				,	//05:ROI Number(ROI编号)
				nAlgorithmNumber_													,	//06:算法编号(nAlgorithmNumber_)
				nThrdIndex_										,	// 07 : Thread ID
				theApp.GetCommonParameter()->bDrawDefectNum		,	//08:Draw Defect Num显示
				theApp.GetCommonParameter()->bDrawDust			,	//09:Draw Dust显示(Point算法必须运行)
				nImageNum																				,	//10:UI上的模式顺序画面号码
				(int)(stThrdAlignInfo.dAlignTheta * 1000)		,	//11:Cell旋转角度(Align计算值,小数点仅为3位...)
				(int)stThrdAlignInfo.ptAlignCenter.x			,	//12:Cell旋转中心x坐标
				(int)stThrdAlignInfo.ptAlignCenter.y			,	//13:Cell旋转中心y坐标
				nRatio														,	//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)
				theApp.GetnBlockCountX(),//[hjf] 分区大小 X 块
				theApp.GetnBlockCountY()//[hjf]	分区大小 Y 块
	};

	//////////////////////////////////////////////////////////////////////////
		//用于验证临时Round Cell
	//CString strTem;
	//strTem.Format(_T("E:\\IMTC\\%02d.bmp"), theApp.GetImageClassify(nImageNum));
	//cv::imwrite((cv::String)(CStringA)strTem, MatOriginImage_);
	//return true;
	//////////////////////////////////////////////////////////////////////////

	CString strResultTime;
	// Fix wrong path for ARESULT. 230313.xb
	//strResultTime.Format(_T("%s%sARESULT"), theApp.m_Config.GetNetworkDrivePath(),strDrive);
	strResultTime.Format(_T("%sARESULT"), strDrive);
	if (nCommonPara[1] > 0) CreateDirectory(strResultTime, NULL);

	//检查结果(不良)信息结构体(在Dust中可以做一点...)
	stDefectInfo* pResultBlob = new stDefectInfo(MAX_DEFECT_COUNT, nImageNum);
	/////////////////////////////////////////
	//st分区判定参数 [hjf]
	stPanelBlockJudgeInfo* EngineerBlockDefectJudgment = theApp.GetBlockDefectFilteringParam(nImageNum, nCameraNum_, nROINumber, nAlgorithmNumber_);
	// ///////////////////////
	// Alg DLL Start Log
	theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<%s> 算法检测开始."),
		strPanelID_, nCameraNum_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_));

	//轮廓存储路径
	wchar_t fPath[MAX_PATH] = { 0 };
	CString strResultPath = RESULT_PATH;
	swprintf(fPath, _T("%s\\%s\\mura.coord"), (LPCWSTR)strResultPath, (LPCWSTR)strPanelID_);

	//创建中间画面存储路径和文件夹
	wchar_t wstrAlgPath[MAX_PATH] = { 0, };
	swprintf(wstrAlgPath, _T("%s\\%s\\"), (LPCWSTR)strResultTime, (LPCWSTR)strPanelID_);
	if (nCommonPara[1] > 0) CreateDirectory(wstrAlgPath, NULL);

	//用于Alg日志
	wchar_t strAlgLog[MAX_PATH] = { 0, };
	swprintf(strAlgLog, _T("ID:%s\tPat:%s"), (LPCWSTR)strPanelID_, theApp.GetGrabStepName(nImageNum));

	//////////////////////////////////////////////////////////////////////////
		//生成Result Buff:仅在非LINE时使用
		//缓冲区分配和初始化

	cv::Mat matReusltImage[E_DEFECT_COLOR_COUNT];
	cv::Mat matDustReusltImage[E_DEFECT_COLOR_COUNT];
	CMatResultBuf* ResultBuf;

	if (nAlgorithmNumber_ != E_ALGORITHM_LINE)
	{
		ResultBuf = cMemResultBuff.FindFreeBuf_Result();

		SetMem_Result(ResultBuf);

		//缓冲区分配
		matReusltImage[E_DEFECT_COLOR_DARK] = cMemResult->GetMat_Result(MatOriginImage_.size(), MatOriginImage_.type(), false);
		matReusltImage[E_DEFECT_COLOR_BRIGHT] = cMemResult->GetMat_Result(MatOriginImage_.size(), MatOriginImage_.type(), false);

		// Buff Set Start Log
		theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<%s> 内存分配."),
			strPanelID_, nCameraNum_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_));
	}

	ST_ALGO_INFO algoInfo(LogicPara.strPanelID, theApp.m_Config.GetPCNum(), nImageNum, nAlgorithmNumber_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_));

	//////////////////////////////////////////////////////////////////////////
		//运行算法DLL
	//////////////////////////////////////////////////////////////////////////
	switch (nAlgorithmNumber_)
	{
	case E_ALGORITHM_POINT:
	{
		//在C#中,由于C++和cv::Mat结构不同,缓冲区和大小分别传递了参数。
		//使用MFC,因此不妨直接跳过cv::Mat
		lErrorCode = Point_FindDefect(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlignPara, stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, strAlgLog, stThrdAlignInfo.tCHoleAlignData->matCHoleROIBuf[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]],
			&algoInfo, LogSendToUI::getInstance(), EngineerBlockDefectJudgment);

		if (theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_DUST)
		{
			MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nAlgorithmNumber_][0] = matReusltImage[E_DEFECT_COLOR_DARK].clone();
			MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nAlgorithmNumber_][1] = matReusltImage[E_DEFECT_COLOR_BRIGHT].clone();
		}

		//Dust不好的情况下,E级判定检查函数
		if (lErrorCode == E_ERROR_CODE_POINT_JUDEGEMENT_E)
		{
			//添加1个不良E级判定不良(E_DEFECT_JUDGEMENT_DUST_GROUP)

			//将结果转交给UI(暂定为Stage编号0)				
//JudgeADDefect(nImageNum, nCameraNum_, 0, MatOriginImage_.cols, MatOriginImage_.rows, pResultBlob_Total, E_DEFECT_JUDGEMENT_DUST_GROUP, eInspMode, false);

				//添加nStageNo
			JudgeADDefect(nImageNum, nCameraNum_, m_stThrdAlignInfo.nStageNo, MatOriginImage_.cols, MatOriginImage_.rows, pResultBlob_Total, E_DEFECT_JUDGEMENT_DUST_GROUP, eInspMode, false);

			//错误代码改为True(直接运行现有检查逻辑)
			lErrorCode = E_ERROR_CODE_TRUE;
		}

		//如果不是错误
		if (lErrorCode == E_ERROR_CODE_TRUE)
		{
			//如果是Dust模式,气泡结果
			if (theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_DUST)
			{
				lErrorCode = Point_GetDefectList(MatOriginImage_, matReusltImage, MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nAlgorithmNumber_], MatDrawBuffer, stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], stThrdAlignInfo.tCHoleAlignData->matCHoleROIBuf[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], false);
			}
			//如果不是Dust模式
			else
			{
				//等待Dust检查结束
				while (!bpInspectEnd[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_])
				{
					Sleep(10);
				}

				/////////////////////////////////////////////////////////////////////////////////////////////
									//为了获得Point_GetDefectList函数...
								//以dust Pattern结束Point_FindDefect()后,必须运行Point_GetDefectList()函数
				/////////////////////////////////////////////////////////////////////////////////////////////

				if (!matReusltImage[E_DEFECT_COLOR_DARK].empty() &&
					!matReusltImage[E_DEFECT_COLOR_BRIGHT].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nAlgorithmNumber_][0].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nAlgorithmNumber_][1].empty())
				{
					//Dust图像和检查Point的Pattern图像的分辨率不同。
					//在DLL中放大/缩小Dust图像进行检查

					lErrorCode = Point_GetDefectList(MatOriginImage_, matReusltImage, MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nAlgorithmNumber_], MatDrawBuffer, stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], stThrdAlignInfo.tCHoleAlignData->matCHoleROIBuf[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]]);
				}
			}
		}
	}
	break;

	case E_ALGORITHM_LINE:
	{
		vector<int> NorchIndex;
		CPoint OrgIndex;

		// UI ( list )
		if (theApp.m_pGrab_Step[nImageNum].tRoundSet != NULL)
		{
			//获取Norch Round部分信息
			GetModelNorchInfo(theApp.m_pGrab_Step[nImageNum].tRoundSet, NorchIndex, OrgIndex);
		}

		lErrorCode = Line_FindDefect(MatOriginImage_, MatDrawBuffer, MatBKG_, NorchIndex, OrgIndex, stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], stThrdAlignInfo.tCHoleAlignData->matCHoleROIBuf[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], &algoInfo, LogSendToUI::getInstance());
	}
	break;

	case E_ALGORITHM_MURA:
	{
		lErrorCode = Mura_FindDefect(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, &algoInfo, LogSendToUI::getInstance());

		//如果不是错误
		if (lErrorCode == E_ERROR_CODE_TRUE)
		{
			//如果不是Dust模式
			if (theApp.GetImageClassify(nImageNum) != E_IMAGE_CLASSIFY_AVI_DUST)
			{
				//等待Point Dust检查(常用)结束
				while (!bpInspectEnd[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_])
				{
					Sleep(10);
				}

				/////////////////////////////////////////////////////////////////////////////////////////////
									//要获得Mura_GetDefectList函数...
								//以Dust Pattern结束Point_FindDefect()后,必须运行Mura_GetDefectList()函数
				/////////////////////////////////////////////////////////////////////////////////////////////

								//获取Point算法编号
				int nPointAlgIndex = theApp.GetAlgorithmIndex(_T("POINT"));

				if (!matReusltImage[E_DEFECT_COLOR_DARK].empty() &&
					!matReusltImage[E_DEFECT_COLOR_BRIGHT].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex][0].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex][1].empty())
				{
					lErrorCode = Mura_GetDefectList(MatOriginImage_, matReusltImage, MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex], MatDrawBuffer, stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, fPath, strAlgLog);
				}
			}
		}
		//18.09.21-RGB检测出不良,也需要进行漏气检测
		//检测到R,G,B Line故障
//else if ( lErrorCode == E_ERROR_CODE_MURA_RGB_LINE_DEFECT )
//{
					//在18.09.19-Alg中添加(请求显示Defect-Map)
					//添加nStageNo
	//JudgeADDefect(nImageNum, nCameraNum_, m_stThrdAlignInfo.nStageNo, MatOriginImage_.cols, MatOriginImage_.rows, pResultBlob_Total, E_DEFECT_JUDGEMENT_MURA_LINE_X, eInspMode, false);				

				//错误代码改为True(直接运行现有检查逻辑)
	//lErrorCode = E_ERROR_CODE_TRUE;
//}
	}
	break;
	case E_ALGORITHM_MURA_NORMAL:
	{
		lErrorCode = MuraNormal_FindDefect(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, &algoInfo, LogSendToUI::getInstance());

		//如果不是错误
		if (lErrorCode == E_ERROR_CODE_TRUE)
		{
			//如果不是Dust模式
			if (theApp.GetImageClassify(nImageNum) != E_IMAGE_CLASSIFY_AVI_DUST)
			{
				//等待Point Dust检查(常用)结束
				while (!bpInspectEnd[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_])
				{
					Sleep(10);
				}

				/////////////////////////////////////////////////////////////////////////////////////////////
									//要获得Mura_GetDefectList函数...
								//以Dust Pattern结束Point_FindDefect()后,必须运行Mura_GetDefectList()函数
				/////////////////////////////////////////////////////////////////////////////////////////////

								//获取Point算法编号
				int nPointAlgIndex = theApp.GetAlgorithmIndex(_T("POINT"));

				if (!matReusltImage[E_DEFECT_COLOR_DARK].empty() &&
					!matReusltImage[E_DEFECT_COLOR_BRIGHT].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex][0].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex][1].empty())
				{
					lErrorCode = MuraNormal_GetDefectList(MatOriginImage_, matReusltImage, MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex], MatDrawBuffer, stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, fPath, strAlgLog);
				}
			}
		}
	}
	break;
	case E_ALGORITHM_MURA3:
	{

		lErrorCode = Mura_FindDefect3(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, MatDrawBuffer, &algoInfo, LogSendToUI::getInstance(), fPath);

		//如果不是Dust模式

			//等待Point Dust检查(通用)结束

// 						Sleep(10);

//					//获取Point算法编号

	}
	break;
	case E_ALGORITHM_MURA4:
	{
		lErrorCode = Mura_FindDefect4(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, MatDrawBuffer, &algoInfo, LogSendToUI::getInstance(), fPath);

		//-客户请求
		//白色模式Amorp Dark检测Dust模式中有异物时删除
		//确认结果是用Amorph Dark检测出了异物

		if (lErrorCode == E_ERROR_CODE_TRUE)
		{
			//如果不是Dust模式
			if (theApp.GetImageClassify(nImageNum) != E_IMAGE_CLASSIFY_AVI_DUST)
			{
				//等待Point Dust检查(通用)结束
				while (!bpInspectEnd[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_])
				{
					Sleep(10);
				}

				/////////////////////////////////////////////////////////////////////////////////////////////
							///要获得Mura_GetDefectList函数...
							//以Dust Pattern结束Point_FindDefect()后,必须运行Mura_GetDefectList()函数
				/////////////////////////////////////////////////////////////////////////////////////////////

							//获取Point算法编号
				int nPointAlgIndex = theApp.GetAlgorithmIndex(_T("POINT"));

				if (!matReusltImage[E_DEFECT_COLOR_DARK].empty() &&
					!matReusltImage[E_DEFECT_COLOR_BRIGHT].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex][0].empty() &&
					!MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex][1].empty())
				{
					lErrorCode = Mura_GetDefectList3(MatOriginImage_, matReusltImage, MatResultImg[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][nCameraNum_][nPointAlgIndex], MatDrawBuffer, stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, fPath, strAlgLog);
				}
			}
		}
	}
	break;
	case E_ALGORITHM_MURA_CHOLE:
	{
		lErrorCode = Mura_FindDefect_Chole(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], stThrdAlignInfo.tCHoleAlignData->matCHoleROIBuf[nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER]], dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, MatDrawBuffer, &algoInfo, LogSendToUI::getInstance(), fPath);
	}
	break;
	case E_ALGORITHM_MURA_SCRATCH:
	{

		lErrorCode = Mura_FindDefect_Scratch(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, MatDrawBuffer, fPath);
	}
	break;
	case E_ALGORITHM_DUST:
	{
		lErrorCode = Dust_FindDefect_PS(MatOriginImage_, MatOriImageRGB_, MatBKG_, matReusltImage[E_DEFECT_COLOR_DARK], matReusltImage[E_DEFECT_COLOR_BRIGHT],
			stThrdAlignInfo.ptCorner, dAlgPara, nCommonPara, wstrAlgPath, EngineerBlockDefectJudgment, pResultBlob, strAlgLog, MatDrawBuffer, fPath);
	}
	break;

	default:
		break;

	}

	// Memory Release
	if (nAlgorithmNumber_ != E_ALGORITHM_LINE)
	{
		cMemResultBuff.ReleaseFreeBuf_Result(ResultBuf);
	}

	// Buff Release Start Log
	//theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<%s> 内存释放."),
	//	strPanelID_, nCameraNum_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_));

	//如果有错误,则输出错误代码&日志
	if (lErrorCode != E_ERROR_CODE_TRUE)
	{
		// Alg DLL Error Log
		theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<%s>算法检测失败. 错误码: %d"),
			strPanelID_, nCameraNum_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_), lErrorCode);
		return lErrorCode;
	}

	//检测不良数量
	int nTotalblob = 0;

	//禁用Dust
	if (pResultBlob != NULL)
	{
		//在删除非检查区域之前,错误的数量(记录在哪个模式,哪个算法中检测出了几个错误)
		nTotalblob = pResultBlob->nDefectCount;

		//在删除未检查区域之前,Alg DLL End Log
		theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<%s>算法检测结束. 原始缺陷数量: %d "),
			strPanelID_, nCameraNum_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_), nTotalblob);

		//////////////////////////////////////////////////////////////////////////

				//获取参数
		double* dAlignPara = theApp.GetAlignParameter(0);

		//如果没有收到结果,则跳过
		if (dAlignPara)
		{
			//如果只操作线路不良
			if (dAlignPara[E_PARA_NON_INSP_AREA_ONLY_LINE] > 0)
			{
				//仅在行中操作/不执行其他不良操作
				if (nAlgorithmNumber_ == E_ALGORITHM_LINE)
				{
					//删除非检查区域(按原始画面计算,无需Left-Top校正:NULL)
					m_fnDefectFiltering(MatDrawBuffer, nImageNum, nCameraNum_, pResultBlob, stThrdAlignInfo, nRatio);
				}
			}
			//如果所有不良行为
			else
			{
				//删除非检查区域(按原始画面计算,无需Left-Top校正:NULL)
				m_fnDefectFiltering(MatDrawBuffer, nImageNum, nCameraNum_, pResultBlob, stThrdAlignInfo, nRatio);
			}
		}
		//如果收不到,所有不良行为
		else
		{
			//删除非检查区域(按原始画面计算,无需Left-Top校正:NULL)
			m_fnDefectFiltering(MatDrawBuffer, nImageNum, nCameraNum_, pResultBlob, stThrdAlignInfo, nRatio);
		}

		//删除非检查区域后,错误的数量(记录在哪个模式,哪个算法中检测出了几个错误)
		nTotalblob = pResultBlob->nDefectCount;

		//17.09.07-必要时使用
		if (theApp.GetCommonParameter()->bIFImageSaveFlag)
		{
			//保存算法不良结果信息
			//需要在MatDrawBuffer画面中显示数字才能知道。

			CString strResultROIFile = _T("");
			strResultROIFile.Format(_T("%s\\%s\\ROI\\IMG-%02d_CAM-%02d_ROI-%02d %s Result.csv"),
				INSP_PATH, strPanelID_, nImageNum, nCameraNum_, nROINumber,
				theApp.GetAlgorithmName(nAlgorithmNumber_));

			//保存不良信息
			BlobFeatureSave(pResultBlob, strResultROIFile);
			/**
			 * 保存算法缺陷参数 hjf
			 *
			 * \param strDrive
			 * \param LogicPara
			 * \param MatResultImg
			 * \param MatDrawBuffer
			 * \param nImageNum
			 * \param nROINumber
			 * \param nAlgorithmNumber
			 * \param stThrdAlignInfo
			 * \param pResultBlob_Total
			 * \param bpInspectEnd
			 * \param nRatio
			 * \param eInspMode
			 * \param WrtResultInfo
			 * \param _mtp
			 * \return
			 */
			CString strResultAlgoFeatureFile = _T("");
			strResultAlgoFeatureFile.Format(_T("%s\\%s\\AlgoFeatureParams\\%s.csv"),
				INSP_PATH, strPanelID_, strPanelID_);
			AlgoFeatureSave(pResultBlob, strResultAlgoFeatureFile, strPanelID_, nImageNum, theApp.GetAlgorithmName(nAlgorithmNumber_), m_stThrdAlignInfo.nStageNo);
			//////////////////////////////////////////////////////////////
		}

		//合并不良信息
		pResultBlob_Total->AddTail_ResultBlobAndAddOffset(pResultBlob, NULL);
	}

	//AI
	AIReJudgeParam AIParam = GetAIParam(dAlgPara, nAlgorithmNumber_, theApp.GetImageClassify(nImageNum));
	//2024.05.07 for develop
	if (AIParam.AIEnable && nTotalblob > 0 && nTotalblob < 30 && false) {
		CString strAIResultFile = _T("");
		strAIResultFile.Format(_T("%s\\%s\\AI_Result\\%s_AI.csv"),
			INSP_PATH, strPanelID_, strPanelID_);

		CString strAIResultPath = _T("");
		strAIResultPath.Format(_T("%s\\%s\\AI_Result"), INSP_PATH, strPanelID_);
		wchar_t wstrAIPath[MAX_PATH] = { 0, };
		swprintf(wstrAIPath, _T("%s\\"), (LPCWSTR)strAIResultPath);
		CreateDirectory(wstrAIPath, NULL);


		while (LogicPara.MatDics.empty()) {
			Sleep(10);
		}

		std::vector<TaskInfoPtr> taskList;
		int dicsRatio = (int)dAlignPara[E_PARA_AVI_DICS_IMAGE_RATIO];
		int cropExpand = (int)dAlignPara[E_PARA_AVI_DICS_AI_CROP_EXPAND];
		PrepareAITask(LogicPara.MatDics, dicsRatio, cropExpand, pResultBlob, AIParam, LogicPara.strPanelID, nImageNum, nAlgorithmNumber_, taskList);


		for (auto task : taskList)
		{
			GetAIRuntime()->CommitInferTask(task);
		}


		for (auto task : taskList)
		{
			std::promise<ModelResultPtr>* promiseResult = static_cast<std::promise<ModelResultPtr>*>(task->promiseResult);
			std::future<ModelResultPtr> futureRst = promiseResult->get_future();
			ModelResultPtr rst = futureRst.get();

			AIInfoPtr aiInfoPtr = std::static_pointer_cast<STRU_AI_INFO>(task->inspParam);
			//写下ResultBlob
			for (int i = 0; i < aiInfoPtr->defectNoList.size(); i++)
			{
				int di = aiInfoPtr->defectNoList[i];
				pResultBlob->AI_ReJudge_Code[di] = rst->itemList[i][0].code;
				pResultBlob->AI_ReJudge_Conf[di] = rst->itemList[i][0].confidence;
				pResultBlob->AI_ReJudge[i] = AIParam.rejudge;
				//
				if (pResultBlob->AI_ReJudge_Code[di] == 0 && pResultBlob->AI_ReJudge_Conf[di] >= AIParam.confidence)
				{
					pResultBlob->AI_ReJudge_Result[di] = 0; // 
				}
				else if (pResultBlob->AI_ReJudge_Code[di] == 1 && pResultBlob->AI_ReJudge_Conf[di] >= AIParam.confidence)
				{
					pResultBlob->AI_ReJudge_Result[di] = 1; // 
				}
				else {
					pResultBlob->AI_ReJudge_Result[di] = 2;	// 
				}
			}

			//
			SaveInferenceResultMult(strAIResultFile, strAIResultPath, AIParam, rst->itemList, rst->taskInfo);
			delete promiseResult;
		}

		//////AI Detect for After filter log
		theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("%s AI Algorithm filter. PanelID: %s, CAM: %02d ROI: %02d, Img: %s.\n\t\t\t\t( After AI FilterNum: %d )"),
			theApp.GetAlgorithmName(nAlgorithmNumber_), strPanelID_, nCameraNum_, nROINumber, theApp.GetGrabStepName(nImageNum), nTotalblob);
	}

	// Alg DLL End Log
	theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<%s>算法检测结束.过滤后缺陷数量: %d "),
		strPanelID_, nCameraNum_, theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgorithmNumber_), nTotalblob);

	return true;
}

bool AviInspection::JudgementET(ResultPanelData& resultPanelData, double* dAlignPara, CString strPanelID)
{
	//异常处理
	if (dAlignPara == NULL)	return false;

	//验证设置
	if (dAlignPara[E_PARA_HS_JUDGMENT_COUNT_BRIGHT] > 0)
	{
		int nCountBright = 0;
		int nIndexBright = 0;

		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;

			if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_32 &&
				theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_64 &&
				theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_87)
				continue;

			//名牌束
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			{
				nCountBright++;

				if (nCountBright == 1)
					nIndexBright = i;	// 拥有第一个索引
			}
		}

		//超过一定数量时,只报告一个
		if (dAlignPara[E_PARA_HS_JUDGMENT_COUNT_BRIGHT] < nCountBright)
		{
			//修改第一个报告的判定->RETEST_POINT_BRIGHT
			resultPanelData.m_ListDefectInfo[nIndexBright].Defect_Type = E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT;

			//不良数量
			for (int i = nIndexBright + 1; i < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;

				//G32名店铺
				if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_32 &&
					theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_64 &&
					theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_87)
				{
					i++;  // 下一个不良...
					continue;
				}

				//名牌束
				if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
				{
					//删除错误
					resultPanelData.m_ListDefectInfo.RemoveAt(i);
				}
				else i++;  // 下一个不良...
			}
		}
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - ET Bright End. Panel ID : %s (Count : %d )"), strPanelID, nCountBright);
	}

	//验证设置
	if (dAlignPara[E_PARA_HS_JUDGMENT_COUNT_DARK] > 0)
	{
		int nCountDark = 0;
		int nIndexDark = 0;

		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;

			//R,G,B多发癌症
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
			{
				nCountDark++;

				if (nCountDark == 1)
					nIndexDark = i;	// 拥有第一个索引
			}
		}

		//超过一定数量时,只报告一个
		if (dAlignPara[E_PARA_HS_JUDGMENT_COUNT_DARK] < nCountDark)
		{
			//修改第一个报告的判定->RETEST_POINT_DARK
			resultPanelData.m_ListDefectInfo[nIndexDark].Defect_Type = E_DEFECT_JUDGEMENT_RETEST_POINT_DARK;

			//不良数量
			for (int i = nIndexDark + 1; i < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;

				
				if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
				{
					//删除错误
					resultPanelData.m_ListDefectInfo.RemoveAt(i);
				}
				else i++;  // 下一个不良...
			}
		}
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("Judge - ET Dark End. Panel ID : %s (Count : %d )"), strPanelID, nCountDark);
	}

	return true;
}

//	//异常处理

//		//Pad部分,点亮区域边(P/S校正)

//		//Pad簿,区域设置

//		//按照所有不良的数量...

//			//比较P/S模式下的坐标

//			//范围内有不良存在吗？

// 				tTemp.Defect_Type = E_DEFECT_JUDGEMENT_POINT_PAD_BRIGHT;

//			//范围内有不良存在吗？

// 				tTemp.Defect_Type = E_DEFECT_JUDGEMENT_POINT_PAD_BRIGHT;

//17.11.29-外围信息(AVI&SVI其他工具)
bool AviInspection::JudgeSaveContours(ResultPanelData& resultPanelData, wchar_t* strContourTxt)
{
#if USE_ALG_CONTOURS

	CStdioFile	fileWriter;
	CString		strContourIndex;
	bool bIsOpen = false;

	//如果没有外框存储路径
	if (strContourTxt != NULL) {
		bIsOpen = fileWriter.Open(strContourTxt, CFile::modeCreate | CFile::modeNoTruncate | CFile::modeWrite);
		if (!bIsOpen) {
			return false;
		}
	}

	//计算旋转坐标时,使用
	double dTheta = -m_stThrdAlignInfo.dAlignTheta * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	int nCx = m_stThrdAlignInfo.ptAlignCenter.x;
	int nCy = m_stThrdAlignInfo.ptAlignCenter.y;

	//旋转的点灯区域
	cv::Point ptCorner[E_CORNER_END];

	//旋转时,计算预测坐标
	for (int m = 0; m < E_CORNER_END; m++)
	{
		ptCorner[m].x = (int)(dCos * (m_stThrdAlignInfo.ptCorner[m].x - nCx) - dSin * (m_stThrdAlignInfo.ptCorner[m].y - nCy) + nCx);
		ptCorner[m].y = (int)(dSin * (m_stThrdAlignInfo.ptCorner[m].x - nCx) + dCos * (m_stThrdAlignInfo.ptCorner[m].y - nCy) + nCy);
	}

	//已点亮的范围
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	//根据P/S模式修改坐标(更改为单杆坐标)
	rectROI.left /= m_stThrdAlignInfo.nRatio;
	rectROI.top /= m_stThrdAlignInfo.nRatio;
	rectROI.right /= m_stThrdAlignInfo.nRatio;
	rectROI.bottom /= m_stThrdAlignInfo.nRatio;

	if (bIsOpen) {
		//创建顶级标头(Cell大小)
		strContourIndex.Format(_T("Cell_X=%d, Cell_Y=%d\n"), rectROI.Width(), rectROI.Height());
		fileWriter.SeekToEnd();
		fileWriter.WriteString(strContourIndex);
	}

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//UI模式顺序
		int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;

		if (bIsOpen) {
			//写入头文件(No.||模式信息||相机编号||不良名称)
			strContourIndex.Format(_T("No=%d, Pattern=%02d, Camera=%02d, Defect=%02d\n"), i + 1, theApp.GetImageClassify(nImgNum), resultPanelData.m_ListDefectInfo[i].Camera_No, resultPanelData.m_ListDefectInfo[i].Defect_Type);
			fileWriter.SeekToEnd();
			fileWriter.WriteString(strContourIndex);
		}

		//如果AD有问题
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_DUST_GROUP)
		{
			if (bIsOpen) {
				//写入矩形E_CORNER_LEFT_TOP信息文件
				strContourIndex.Format(_T("%d, %d\n"), 0, 0);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);

				//写入矩形E_CORNER_RIGHT_TOP信息文件
				strContourIndex.Format(_T("%d, %d\n"), rectROI.Width() - 1, 0);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);

				//写入矩形E_CORNER_RIGHT_BOTTOM信息文件
				strContourIndex.Format(_T("%d, %d\n"), rectROI.Width() - 1, rectROI.Height() - 1);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);

				//写入矩形E_CORNER_LEFT_BOTTOM信息文件
				strContourIndex.Format(_T("%d, %d\n"), 0, rectROI.Height() - 1);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);
			}

			resultPanelData.m_ListDefectInfo[i].nContoursX[0] = 0;
			resultPanelData.m_ListDefectInfo[i].nContoursY[0] = 0;
			resultPanelData.m_ListDefectInfo[i].nContoursX[1] = rectROI.Width() - 1;
			resultPanelData.m_ListDefectInfo[i].nContoursY[1] = 0;
			resultPanelData.m_ListDefectInfo[i].nContoursX[2] = rectROI.Width() - 1;
			resultPanelData.m_ListDefectInfo[i].nContoursY[2] = rectROI.Height() - 1;
			resultPanelData.m_ListDefectInfo[i].nContoursX[3] = 0;
			resultPanelData.m_ListDefectInfo[i].nContoursY[3] = rectROI.Height() - 1;
			resultPanelData.m_ListDefectInfo[i].nContoursCount = 4;
			continue;
		}

		//村里是不良品吗？
		bool bMura = false;

		// Mura
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_MURA_START &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_MURA_END)
		{
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP)
				bMura = true;
		}
		// Mura Retest
		else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_MURA ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_MURA_BRIGHT)
		{
			bMura = true;
		}
		// APP
		else if (resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_APP_START &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_APP_END)
		{
			bMura = true;
		}

		//转移轮廓信息
		if (bMura)
		{
			resultPanelData.m_ListDefectInfo[i].nContoursCount = 0;
			//关于轮廓
			for (int j = 0; j < MAX_CONTOURS; j++)
			{
				//根据P/S模式修改坐标(更改为单杆坐标)
					//CFeatureExtraction::calcContours()已更改为一杆坐标
				int x = resultPanelData.m_ListDefectInfo[i].nContoursX[j];
				int y = resultPanelData.m_ListDefectInfo[i].nContoursY[j];

				//如果没有值,请退出
				if (x == 0 && y == 0)	break;

				//使用E_CORNER_LEFT_TOP坐标作为原点
				x -= rectROI.left;
				y -= rectROI.top;

				//异常处理
				if (x < 0)						x = 0;
				if (y < 0)						y = 0;
				if (x >= rectROI.Width())		x = rectROI.Width() - 1;
				if (y >= rectROI.Height())		y = rectROI.Height() - 1;

				if (bIsOpen) {
					//写入轮廓信息文件
					strContourIndex.Format(_T("%d, %d\n"), x, y);
					fileWriter.SeekToEnd();
					fileWriter.WriteString(strContourIndex);
				}

				resultPanelData.m_ListDefectInfo[i].nContoursX[j] = x;
				resultPanelData.m_ListDefectInfo[i].nContoursY[j] = y;
				resultPanelData.m_ListDefectInfo[i].nContoursCount++;
			}
		}
		//转交除村外的不良(Point&Line)/矩形信息
		else
		{
			int nOffSet = 5;
			cv::Point ptDefectArea[4] = { (0, 0), };

			//关于无效的矩形转角点
			ptDefectArea[E_CORNER_LEFT_TOP].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_TOP].x - nOffSet;
			ptDefectArea[E_CORNER_LEFT_TOP].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_TOP].y - nOffSet;
			ptDefectArea[E_CORNER_RIGHT_TOP].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_TOP].x + nOffSet;
			ptDefectArea[E_CORNER_RIGHT_TOP].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_TOP].y - nOffSet;
			ptDefectArea[E_CORNER_RIGHT_BOTTOM].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM].x + nOffSet;
			ptDefectArea[E_CORNER_RIGHT_BOTTOM].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM].y + nOffSet;
			ptDefectArea[E_CORNER_LEFT_BOTTOM].x = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_BOTTOM].x - nOffSet;
			ptDefectArea[E_CORNER_LEFT_BOTTOM].y = resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_BOTTOM].y + nOffSet;

			//根据P/S模式修改坐标(更改为单杆坐标)
			int nRatio = resultPanelData.m_ListDefectInfo[i].nRatio;
			ptDefectArea[E_CORNER_LEFT_TOP].x /= nRatio;
			ptDefectArea[E_CORNER_LEFT_TOP].y /= nRatio;
			ptDefectArea[E_CORNER_RIGHT_TOP].x /= nRatio;
			ptDefectArea[E_CORNER_RIGHT_TOP].y /= nRatio;
			ptDefectArea[E_CORNER_RIGHT_BOTTOM].x /= nRatio;
			ptDefectArea[E_CORNER_RIGHT_BOTTOM].y /= nRatio;
			ptDefectArea[E_CORNER_LEFT_BOTTOM].x /= nRatio;
			ptDefectArea[E_CORNER_LEFT_BOTTOM].y /= nRatio;

			//使用E_CORNER_LEFT_TOP坐标作为原点
			ptDefectArea[E_CORNER_LEFT_TOP].x -= rectROI.left;
			ptDefectArea[E_CORNER_LEFT_TOP].y -= rectROI.top;
			ptDefectArea[E_CORNER_RIGHT_TOP].x -= rectROI.left;
			ptDefectArea[E_CORNER_RIGHT_TOP].y -= rectROI.top;
			ptDefectArea[E_CORNER_RIGHT_BOTTOM].x -= rectROI.left;
			ptDefectArea[E_CORNER_RIGHT_BOTTOM].y -= rectROI.top;
			ptDefectArea[E_CORNER_LEFT_BOTTOM].x -= rectROI.left;
			ptDefectArea[E_CORNER_LEFT_BOTTOM].y -= rectROI.top;

			//异常处理
			for (int m = E_CORNER_LEFT_TOP; m <= E_CORNER_LEFT_BOTTOM; m++)
			{
				if (ptDefectArea[m].x < 0)						ptDefectArea[m].x = 0;
				if (ptDefectArea[m].y < 0)						ptDefectArea[m].y = 0;
				if (ptDefectArea[m].x >= rectROI.Width())		ptDefectArea[m].x = rectROI.Width() - 1;
				if (ptDefectArea[m].y >= rectROI.Height())		ptDefectArea[m].y = rectROI.Height() - 1;
			}

			if (bIsOpen) {
				//写入矩形E_CORNER_LEFT_TOP信息文件
				strContourIndex.Format(_T("%d, %d\n"), ptDefectArea[E_CORNER_LEFT_TOP].x, ptDefectArea[E_CORNER_LEFT_TOP].y);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);

				//写入矩形E_CORNER_RIGHT_TOP信息文件
				strContourIndex.Format(_T("%d, %d\n"), ptDefectArea[E_CORNER_RIGHT_TOP].x, ptDefectArea[E_CORNER_RIGHT_TOP].y);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);

				//写入矩形E_CORNER_RIGHT_BOTTOM信息文件
				strContourIndex.Format(_T("%d, %d\n"), ptDefectArea[E_CORNER_RIGHT_BOTTOM].x, ptDefectArea[E_CORNER_RIGHT_BOTTOM].y);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);

				//写入矩形E_CORNER_LEFT_BOTTOM信息文件
				strContourIndex.Format(_T("%d, %d\n"), ptDefectArea[E_CORNER_LEFT_BOTTOM].x, ptDefectArea[E_CORNER_LEFT_BOTTOM].y);
				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);
			}

			resultPanelData.m_ListDefectInfo[i].nContoursX[0] = ptDefectArea[E_CORNER_LEFT_TOP].x;
			resultPanelData.m_ListDefectInfo[i].nContoursY[0] = ptDefectArea[E_CORNER_LEFT_TOP].y;
			resultPanelData.m_ListDefectInfo[i].nContoursX[1] = ptDefectArea[E_CORNER_RIGHT_TOP].x;
			resultPanelData.m_ListDefectInfo[i].nContoursY[1] = ptDefectArea[E_CORNER_RIGHT_TOP].y;
			resultPanelData.m_ListDefectInfo[i].nContoursX[2] = ptDefectArea[E_CORNER_RIGHT_BOTTOM].x;
			resultPanelData.m_ListDefectInfo[i].nContoursY[2] = ptDefectArea[E_CORNER_RIGHT_BOTTOM].y;
			resultPanelData.m_ListDefectInfo[i].nContoursX[3] = ptDefectArea[E_CORNER_LEFT_BOTTOM].x;
			resultPanelData.m_ListDefectInfo[i].nContoursY[3] = ptDefectArea[E_CORNER_LEFT_BOTTOM].y;
			resultPanelData.m_ListDefectInfo[i].nContoursCount = 4;
		}
	}

	if (bIsOpen) {
		//仅在文件打开时关闭
		fileWriter.Close();
	}
#endif

	return true;
}

//保存Mura轮廓信息
bool AviInspection::JudgeSaveMuraContours(ResultPanelData& resultPanelData, wchar_t* strContourTxt)
{
	//如果没有外框存储路径
	if (strContourTxt == NULL)			return false;

	//不检查或不检查时Skip
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)	return true;

	//保存TXT
	CStdioFile	fileWriter;
	CString		strContourIndex;

	//打开文件
	if (fileWriter.Open(strContourTxt, CFile::modeCreate | CFile::modeNoTruncate | CFile::modeWrite))
	{
		//计算旋转坐标时,使用
		double dTheta = -m_stThrdAlignInfo.dAlignTheta * PI / 180.;
		double	dSin = sin(dTheta);
		double	dCos = cos(dTheta);
		int nCx = m_stThrdAlignInfo.ptAlignCenter.x;
		int nCy = m_stThrdAlignInfo.ptAlignCenter.y;

		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//如果AD有问题
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_DUST_GROUP)
				continue;

			bool bMura = false;

			//错误判定
			int nDefectType = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			// Mura
			if (nDefectType >= E_DEFECT_JUDGEMENT_MURA_START &&
				nDefectType <= E_DEFECT_JUDGEMENT_MURA_END)
			{
				if (nDefectType != E_DEFECT_JUDGEMENT_MURA_MULT_BP)
					bMura = true;
			}
			// Mura Retest
			else if (nDefectType == E_DEFECT_JUDGEMENT_RETEST_MURA ||
				nDefectType == E_DEFECT_JUDGEMENT_RETEST_MURA_BRIGHT)
			{
				bMura = true;
			}
			// APP
			else if (nDefectType >= E_DEFECT_JUDGEMENT_APP_START &&
				nDefectType <= E_DEFECT_JUDGEMENT_APP_END)
			{
				bMura = true;
			}

			//不包括村上
			if (!bMura) continue;

			//关于轮廓
			for (int j = 0; j < MAX_CONTOURS; j++)
			{
				//根据P/S模式修改坐标(更改为单杆坐标)
					//CFeatureExtraction::calcContours()已更改为一杆坐标
				int x = resultPanelData.m_ListDefectInfo[i].nContoursX[j];
				int y = resultPanelData.m_ListDefectInfo[i].nContoursY[j];

				//如果没有值,请退出
				if (x == 0 && y == 0)	break;

				//写入轮廓信息文件
				strContourIndex.Format(_T("%d, %d, %s\n"), x, y, theApp.GetDefectTypeName(nDefectType));

				fileWriter.SeekToEnd();
				fileWriter.WriteString(strContourIndex);
			}
		}

		//仅在文件打开时关闭
		fileWriter.Close();
	}

	return true;
}

bool AviInspection::DeleteOverlapDefect(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//17.07.14-P/S模式下的坐标比较
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//////////////////////////////////////////////////////////////////////////
		//18.09.21-6.39"耳朵部分没有点亮时,判定为E级
		//如果是EDGE_NUGI,则删除除EDGE_NUGI以外的所有错误
		//SCJ20.02.18-B11曹文请求EDGE区域//ACTIVE区域NUGI请求不良分类
	//////////////////////////////////////////////////////////////////////////
	//
	//bool bNugiJudgeE = false;
	//
	//for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	//{
		//	//查找EDGE_NUGI
	//	if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI)
	//	{
		//		//首次复制EDGE_NUGI列表
	//		resultPanelData.m_ListDefectInfo[0] = resultPanelData.m_ListDefectInfo[i];
	//
		//		//E级判定
	//		bNugiJudgeE = true;
	//
	//		break;
	//	}
	//}
	//
		////如果有EDGE_NUGI...
	//if (bNugiJudgeE)
	//{
		//	//清除所有故障
	//	for (int i = 1; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	//		resultPanelData.m_ListDefectInfo.RemoveAt(i);
	//
		//	//删除后退出
	//	return true;
	//}

	//////////////////////////////////////////////////////////////////////////
		//如果是AD,则删除除AD以外的所有错误
		//没有EDGE_NUGI,如果是AD
	//////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//如果AD不坏(忽略E_DEFECT_JUDGEMENT_DUST_GROUP)
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type > E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT)
			continue;

		//如果第一个模式AD有问题
		if (resultPanelData.m_ListDefectInfo[i].Img_Number == 0)
		{
			//覆盖坏的第一个列表
			ResultDefectInfo tTempAD = resultPanelData.m_ListDefectInfo[i];
			resultPanelData.m_ListDefectInfo[0] = tTempAD;

			//除了第一个列表(AD不良),全部删除。
			for (int j = 1; j < resultPanelData.m_ListDefectInfo.GetCount();)
				resultPanelData.m_ListDefectInfo.RemoveAt(j);

			//删除后退出
			return true;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//如果不是AD,重复数据删除
	//////////////////////////////////////////////////////////////////////////
		//RGBBlack Bright Point-White Spot Mura重复数据删除
		//DeleteOverlapDefect_White_Spot_Mura_RGBBlk_Point(resultPanelData, dAlignPara); //临时主席choi05.07看起来不需要

		//Black mura重复判定
	DeleteOverlapDefect_Black_Mura_and_Judge(resultPanelData, dAlignPara);

	//Point-Point重复数据删除
	DeleteOverlapDefect_Point_Point(resultPanelData, dAlignPara);

	//Mura-Mura重复数据删除
	DeleteOverlapDefect_Mura_Mura(resultPanelData, dAlignPara);

	//Point-Line重复数据删除
	DeleteOverlapDefect_Point_Line(resultPanelData, dAlignPara);

	//Point-Mura重复数据删除
	DeleteOverlapDefect_Point_Mura(resultPanelData, dAlignPara);

	//Line-Mura重复数据删除
	DeleteOverlapDefect_Line_Mura(resultPanelData, dAlignPara);

	//RGBBlack Bright Point-White Spot Mura重复数据删除
//DeleteOverlapDefect_White_Spot_Mura_RGBBlk_Point(resultPanelData, dAlignPara);

	return true;
}

//	//如果没有不良列表或Flag,请退出

// 	//SP1

// 	//SP2

//		//查找EDGE_NUGI

bool AviInspection::DeleteOverlapDefect_DimpleDelet(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	//如果没有异常参数
	if (dAlignPara == NULL)
		return false;

	// Dimple Ratio
	double dDimpleRatio_GrayPattern_Active = (double)(dAlignPara[E_PARA_DIMPLE_RATIO_GRAY_Active]);
	double dDimpleRatio_GrayPattern_Edge = (double)(dAlignPara[E_PARA_DIMPLE_RATIO_GRAY_Edge]);
	double dDimpleRatio_RGBPattern = (double)(dAlignPara[E_PARA_DIMPLE_RATIO_RGB]);

	//异常处理
	if (dDimpleRatio_GrayPattern_Active <= 0 || dDimpleRatio_GrayPattern_Edge <= 0 || dDimpleRatio_RGBPattern <= 0)
		return false;

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//Point不良Count
	int nPointDefect = 0;
	int nRGBDefect = 0;
	int nGreenDefect = 0;
	int nGrayDefect = 0;
	int nMultiBP = 0;

	//剔除不良列表->Point不良则出去/White or G64 Pattern不良则直接以RGB不良为基准获取数据
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{

		//仅在Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
		{
			//Point不良Count
			nPointDefect++;
		}

		//关于Pattern
		int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int	nImgNum1 = theApp.GetImageClassify(nImgNum);

		//仅在White Gray Pattern中发现Point不良
		if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_64)
		{
			//G64 Pattern中的Point不良Count
			nGrayDefect++;
		}

		//仅在White Gray Pattern中发现Point不良
		if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_R || nImgNum1 == E_IMAGE_CLASSIFY_AVI_G || nImgNum1 == E_IMAGE_CLASSIFY_AVI_B)
		{
			//White Pattern和G64 Pattern中的Point错误计数
			nRGBDefect++;
		}

	}

	//检查Green Pattern是否出现Retest故障
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//仅在Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT)
		{
			//关于Pattern
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅在White Gray Pattern中发现Point不良
			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_G)
			{
				//White Pattern和G64 Pattern中的Point错误计数
				nGreenDefect++;
			}
		}
	}

	//G64 Pattern中的Multi-BP计数Count
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//仅在Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_MULT_BP)
		{
			//关于Pattern
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅在White Gray Pattern中发现Point不良
			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				//White Pattern和G64 Pattern中的Point错误计数
				nMultiBP++;
			}
		}
	}

	//Point没有不良的话就算了
	if (nPointDefect <= 0 && nGreenDefect <= 0 && nMultiBP <= 0)
		return true;

	//Parameter
	int	nOffSet = 4;
	int	nRangeOffset = 10;

	//坐标
	cv::Rect rectTemp;

	//初始化坐标
	int	PixelStartX = 0;
	int	PixelStartY = 0;
	int	PixelEndX = 0;
	int	PixelEndY = 0;

	// ROI Region
	int nXStart = m_stThrdAlignInfo.rcAlignCellROI.x;
	int nYStart = m_stThrdAlignInfo.rcAlignCellROI.y;
	int nXEnd = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
	int nYEnd = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

	//White&G64 Pattern原始画面
	cv::Mat mWhiteImage = MatOrgImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_WHITE)][0];
	cv::Mat mG64Image = MatOrgImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_GRAY_64)][0];

	// Gaussian Blur
	cv::GaussianBlur(mWhiteImage, mWhiteImage, cv::Size(5, 5), 3.0);
	cv::GaussianBlur(mG64Image, mG64Image, cv::Size(5, 5), 3.0);

	//////////////////////////////////////////////////////////////////////////
	//Gray Pattern名分不良Count
	//////////////////////////////////////////////////////////////////////////

	if (nGrayDefect > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			{
				continue;
			}

			//找到不良模式
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅查找RGB中的不良内容
			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				continue;
			}

			////////////////////////////////
			// Max-Mean Ratio Calculation //
			int nDefectMaxGV = (int)resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
			int nDefectMeanGV = (int)resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
			float MaxMeanRatio = (float)nDefectMaxGV / nDefectMeanGV;

			//亮点/Dimple较弱,临时分类为Max/Mean Ratio&&最大亮度值
			if (MaxMeanRatio > 1.83 && nDefectMaxGV > 120)
				continue;

			//坐标
			PixelStartX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X - nRangeOffset);
			PixelStartY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y - 2);
			PixelEndX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_End_X + nRangeOffset);
			PixelEndY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y + 2);

			//选择Ratio初始化Active Ratio
			double dDimpleRatio_GrayPattern = dDimpleRatio_GrayPattern_Active;

			//Edge附近100 Pixel以内的Edge区域
			int nEdgeOffset_X = 90;
			int nEdgeOffset_Y = 100;

			//选择X方向Edge部分Dimple删除Ratio
			if ((PixelStartX - nEdgeOffset_X <= nXStart) || (PixelEndX + nEdgeOffset_X >= nXEnd))
			{
				//垂直转换水平配置文件
				PixelStartX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X - 2);
				PixelStartY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y - nRangeOffset);
				PixelEndX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X + 2);
				PixelEndY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y + nRangeOffset);

				dDimpleRatio_GrayPattern = dDimpleRatio_GrayPattern_Edge;
			}

			//选择Y方向的Edge部分Dimple删除Ratio
			if ((PixelStartY - nEdgeOffset_Y <= nYStart) || (PixelEndY + nEdgeOffset_Y >= nYEnd))
			{
				dDimpleRatio_GrayPattern = dDimpleRatio_GrayPattern_Edge;
			}

			//绘制Rect
			rectTemp.x = PixelStartX;
			rectTemp.y = PixelStartY;
			rectTemp.width = PixelEndX - PixelStartX + 1;
			rectTemp.height = PixelEndY - PixelStartY + 1;

			//以不良为中心,在White/G64 Pattern上获取雕塑画面
			cv::Mat matTempBufWhite = mWhiteImage(rectTemp);
			cv::Mat matTempBufG64 = mG64Image(rectTemp);

			//求区域最大值的平均值
			cv::Scalar tempMean_W, tempStd_W, tempMean_G64, tempStd_G64;

			double minvalue_W = 0;
			double maxvlaue_W = 0;
			double minvalue_G64 = 0;
			double maxvlaue_G64 = 0;

			//获取White Pattern信息
			cv::meanStdDev(matTempBufWhite, tempMean_W, tempStd_W);
			cv::minMaxIdx(matTempBufWhite, &minvalue_W, &maxvlaue_W, NULL, NULL);
			double dImageavg_W = tempMean_W[0];
			double dRatio1 = maxvlaue_W / dImageavg_W;

			//获取G64 Pattern信息
			cv::meanStdDev(matTempBufG64, tempMean_G64, tempStd_G64);
			cv::minMaxIdx(matTempBufG64, &minvalue_G64, &maxvlaue_G64, NULL, NULL);
			double dImageavg_G64 = tempMean_G64[0];
			double dRatio2 = maxvlaue_G64 / dImageavg_G64;

			double dRatio = dRatio2 / dRatio1;

			matTempBufWhite.release();
			matTempBufG64.release();

			//使用Top 3GV重新获得Max GV

				//////////////////////////////////////////////////////////////////////////
							//在G64和White上非常亮的话,会持续出现强视不良
			if (maxvlaue_W >= 200 && maxvlaue_G64 >= 200)
				continue;

			if (dRatio <= dDimpleRatio_GrayPattern)
				//if (dNewRatio <= dDimpleRatio_GrayPattern)
			{
				//将Defect_Type更改为疑似Dimple的画面
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_POINT_DIMPLE;
				continue;
			}
			else
				continue;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//Case2:White-X G64-X->RGB Pattern Dimple Check功能
	// 1. 在White+G64 Pattern中存在无法判定为Dimple的位置
	// 2. 根据RGB不良位置到White&G64 Pattern求Ratio进行判定
	// 3. RGB在状态上很清晰,所以Ratio大也行,不提供Active或Edge。
	//////////////////////////////////////////////////////////////////////////

	if (nRGBDefect > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			{
				continue;
			}

			//找到不良模式
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅查找RGB中的不良内容
			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_R && nImgNum1 != E_IMAGE_CLASSIFY_AVI_G && nImgNum1 != E_IMAGE_CLASSIFY_AVI_B)
			{
				continue;
			}

			////////////////////////////////
			// Max-Mean Ratio Calculation //
			int nDefectMaxGV = (int)resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_G && nDefectMaxGV > 250) continue;

			//基于坐标->RGB Pattern的坐标
			PixelStartX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X - nRangeOffset);
			PixelStartY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y - 2);
			PixelEndX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_End_X + nRangeOffset);
			PixelEndY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y + 2);

			//设置范围
			rectTemp.x = PixelStartX;
			rectTemp.y = PixelStartY;
			rectTemp.width = PixelEndX - PixelStartX + 1;
			rectTemp.height = PixelEndY - PixelStartY + 1;

			//基于RGB查找White/G64 Pattern故障
			cv::Mat matTempBufWhite = mWhiteImage(rectTemp);
			cv::Mat matTempBufG64 = mG64Image(rectTemp);

			//求区域最大值的平均值
			cv::Scalar tempMean_W, tempStd_W, tempMean_G64, tempStd_G64;

			double minvalue_W = 0;
			double maxvlaue_W = 0;
			double minvalue_G64 = 0;
			double maxvlaue_G64 = 0;

			//获取White Pattern信息
			cv::meanStdDev(matTempBufWhite, tempMean_W, tempStd_W);
			cv::minMaxIdx(matTempBufWhite, &minvalue_W, &maxvlaue_W, NULL, NULL);
			double dImageavg_W = tempMean_W[0];
			double dRatio1 = maxvlaue_W / dImageavg_W;

			//获取G64 Pattern信息
			cv::meanStdDev(matTempBufG64, tempMean_G64, tempStd_G64);
			cv::minMaxIdx(matTempBufG64, &minvalue_G64, &maxvlaue_G64, NULL, NULL);
			double dImageavg_G64 = tempMean_G64[0];
			double dRatio2 = maxvlaue_G64 / dImageavg_G64;

			double dRatio = dRatio2 / dRatio1;

			matTempBufWhite.release();
			matTempBufG64.release();

			//使用Top 3GV重新获得Max GV

			//////////////////////////////////////////////////////////////////////////

			if (maxvlaue_W >= 200 && maxvlaue_G64 >= 200)
				continue;

			if (dRatio <= dDimpleRatio_RGBPattern)
				//if (dNewRatio <= dDimpleRatio_RGBPattern)
			{
				//将Defect_Type更改为疑似Dimple的画面
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_POINT_DIMPLE;
				continue;
			}
			else
				continue;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//Case3:White-X G64-X->RGB Pattern Dimple Check功能
		// 1. 在White+G64 Pattern中存在无法判定为Dimple的位置
		// 2. 根据RGB不良位置到White&G64 Pattern求Ratio进行判定
		// 3. RGB在状态上很清晰,所以Ratio大也行,不提供Active或Edge。
	//////////////////////////////////////////////////////////////////////////

	if (nGreenDefect > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT)
			{
				continue;
			}

			//找到不良模式
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅查找RGB中的不良内容
			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_G)
			{
				continue;
			}

			//基于坐标->RGB Pattern的坐标
			PixelStartX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X - nRangeOffset);
			PixelStartY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y - 2);
			PixelEndX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_End_X + nRangeOffset);
			PixelEndY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y + 2);

			//设置范围
			rectTemp.x = PixelStartX;
			rectTemp.y = PixelStartY;
			rectTemp.width = PixelEndX - PixelStartX + 1;
			rectTemp.height = PixelEndY - PixelStartY + 1;

			//基于RGB查找White/G64 Pattern故障
			cv::Mat matTempBufWhite = mWhiteImage(rectTemp);
			cv::Mat matTempBufG64 = mG64Image(rectTemp);

			//求区域最大值的平均值
			cv::Scalar tempMean_W, tempStd_W, tempMean_G64, tempStd_G64;

			double minvalue_W = 0;
			double maxvlaue_W = 0;
			double minvalue_G64 = 0;
			double maxvlaue_G64 = 0;

			//获取White Pattern信息
			cv::meanStdDev(matTempBufWhite, tempMean_W, tempStd_W);
			cv::minMaxIdx(matTempBufWhite, &minvalue_W, &maxvlaue_W, NULL, NULL);
			double dImageavg_W = tempMean_W[0];
			double dRatio1 = maxvlaue_W / dImageavg_W;

			//获取G64 Pattern信息
			cv::meanStdDev(matTempBufG64, tempMean_G64, tempStd_G64);
			cv::minMaxIdx(matTempBufG64, &minvalue_G64, &maxvlaue_G64, NULL, NULL);
			double dImageavg_G64 = tempMean_G64[0];
			double dRatio2 = maxvlaue_G64 / dImageavg_G64;

			double dRatio = dRatio2 / dRatio1;

			matTempBufWhite.release();
			matTempBufG64.release();

			//使用Top 3GV重新获得Max GV

				//////////////////////////////////////////////////////////////////////////

			if (maxvlaue_W >= 200 && maxvlaue_G64 >= 200)
				continue;

			if (dRatio <= dDimpleRatio_RGBPattern)
				//if (dNewRatio <= dDimpleRatio_RGBPattern)
			{
				//将Defect_Type更改为疑似Dimple的画面
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_POINT_DIMPLE;
				continue;
			}
			else
				continue;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//如果DIMPLE判定为Muti BP,则添加
	//////////////////////////////////////////////////////////////////////////
	if (nMultiBP > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP)
			{
				continue;
			}

			//找到不良模式
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅查找RGB中的不良内容
			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				continue;
			}

			////////////////////////////////
			// Max-Mean Ratio Calculation //
			int nDefectMaxGV = (int)resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
			int nDefectMeanGV = (int)resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
			float MaxMeanRatio = (float)nDefectMaxGV / nDefectMeanGV;

			//亮点/Dimple较弱,临时分类为Max/Mean Ratio&&最大亮度值
			if (MaxMeanRatio > 1.83 && nDefectMaxGV > 120)
				continue;

			//坐标
			PixelStartX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X - nRangeOffset);
			PixelStartY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y - 2);
			PixelEndX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_End_X + nRangeOffset);
			PixelEndY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y + 2);

			//选择Ratio初始化Active Ratio
			double dDimpleRatio_GrayPattern = dDimpleRatio_GrayPattern_Active;

			//Edge附近100 Pixel以内的Edge区域
			int nEdgeOffset_X = 90;
			int nEdgeOffset_Y = 100;

			//选择X方向Edge部分Dimple删除Ratio
			if ((PixelStartX - nEdgeOffset_X <= nXStart) || (PixelEndX + nEdgeOffset_X >= nXEnd))
			{
				//垂直转换水平配置文件
				PixelStartX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X - 2);
				PixelStartY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y - nRangeOffset);
				PixelEndX = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X + 2);
				PixelEndY = (int)(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y + nRangeOffset);

				dDimpleRatio_GrayPattern = dDimpleRatio_GrayPattern_Edge;
			}

			//选择Y方向的Edge部分Dimple删除Ratio
			if ((PixelStartY - nEdgeOffset_Y <= nYStart) || (PixelEndY + nEdgeOffset_Y >= nYEnd))
			{
				dDimpleRatio_GrayPattern = dDimpleRatio_GrayPattern_Edge;
			}

			//绘制Rect
			rectTemp.x = PixelStartX;
			rectTemp.y = PixelStartY;
			rectTemp.width = PixelEndX - PixelStartX + 1;
			rectTemp.height = PixelEndY - PixelStartY + 1;

			//以不良为中心,在White/G64 Pattern上获取雕塑画面
			cv::Mat matTempBufWhite = mWhiteImage(rectTemp);
			cv::Mat matTempBufG64 = mG64Image(rectTemp);

			//求区域最大值的平均值
			cv::Scalar tempMean_W, tempStd_W, tempMean_G64, tempStd_G64;

			double minvalue_W = 0;
			double maxvlaue_W = 0;
			double minvalue_G64 = 0;
			double maxvlaue_G64 = 0;

			//获取White Pattern信息
			cv::meanStdDev(matTempBufWhite, tempMean_W, tempStd_W);
			cv::minMaxIdx(matTempBufWhite, &minvalue_W, &maxvlaue_W, NULL, NULL);
			double dImageavg_W = tempMean_W[0];
			double dRatio1 = maxvlaue_W / dImageavg_W;

			//获取G64 Pattern信息
			cv::meanStdDev(matTempBufG64, tempMean_G64, tempStd_G64);
			cv::minMaxIdx(matTempBufG64, &minvalue_G64, &maxvlaue_G64, NULL, NULL);
			double dImageavg_G64 = tempMean_G64[0];
			double dRatio2 = maxvlaue_G64 / dImageavg_G64;

			double dRatio = dRatio2 / dRatio1;

			matTempBufWhite.release();
			matTempBufG64.release();

			//使用Top 3GV重新获得Max GV

				//////////////////////////////////////////////////////////////////////////

							//在G64和White上非常亮的话,会持续出现强视不良
			if (maxvlaue_W >= 200 && maxvlaue_G64 >= 200)
				continue;

			if (dRatio <= dDimpleRatio_GrayPattern)
				//if (dNewRatio <= dDimpleRatio_GrayPattern)
			{
				//将Defect_Type更改为疑似Dimple的画面
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_POINT_DIMPLE;
				continue;
			}
			else
				continue;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//删除与Dimple中心坐标相同的错误
	//////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{

		//不是Point不良的话就转
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_DIMPLE)
		{
			continue;
		}

		//中心坐标
		CPoint ptSrc1;
		ptSrc1.x = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X);
		ptSrc1.y = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y);

		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//如果是的话就转
			if (i == j)
			{
				j++;
				continue;
			}

			//如果Point不坏就转
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP)
			{
				j++;
				continue;
			}

			//找到不良模式
			int	nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//Black Pattern不碰坏东西
			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_BLACK)
			{
				j++;
				continue;
			}

			//求不良中心坐标
			CPoint ptSrc2;
			ptSrc2.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X);
			ptSrc2.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y);

			//如果不良中心点相同的话...
			if (abs(ptSrc1.x - ptSrc2.x) < nOffSet && abs(ptSrc1.y - ptSrc2.y) < nOffSet)
			{
				//删除小列表时
				if (i > j)	 i--;

				//删除其他模式的行
				resultPanelData.m_ListDefectInfo.RemoveAt(j);
			}
			else j++;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//清除Dimple中的错误
	//////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		//非Dimple旋转
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DIMPLE)
		{
			//如果是Dimple,则删除
			resultPanelData.m_ListDefectInfo.RemoveAt(i);
		}
		else i++;
	}

	//////////////////////////////////////////////////////////////////////////
	//删除Green Pattern Retest Point
	//////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		//确认是否是Retest Point
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT)
		{
			//找到不良模式
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//删除Green Pattern中的Retest Point面
			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_G)
			{
				resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}
			else i++;

		}
		else i++;
	}

	mWhiteImage.release();
	mG64Image.release();

	return true;
}

bool AviInspection::DeleteOverlapDefect_SpecInSpotDelet(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	//如果没有异常参数
	if (dAlignPara == NULL)
		return false;

	//约明店-S级白店村PARAMETER
	double	dblSpotThRatio_Active = (double)(dAlignPara[E_PARA_SISPOT_RATIO_ACTIVE]);	//设置为当前1
	double	dblSpotThRatio_Edge = (double)(dAlignPara[E_PARA_SISPOT_RATIO_EDGE]);		//当前设置为1.04
	int		nGVCount = (int)(dAlignPara[E_PARA_SISPOT_NUMBER_GVCOUNT]); //GV Count: 7000
	int		nPoint_Area = (int)(dAlignPara[E_PARA_SISPOT_AREA]);  // Point Area不良面积设置

	//康明店-F级白点村PARAMETER
	double	dStrong_blSpotThRatio_Active = (double)(dAlignPara[E_PARA_STRONG_SISPOT_RATIO_ACTIVE]);	//设置为当前1
	double	dStrong_blSpotThRatio_Edge = (double)(dAlignPara[E_PARA_STRONG_SISPOT_RATIO_EDGE]);		//当前设置为1.04
	int		nStrong_GVCount = (int)(dAlignPara[E_PARA_STRONG_SISPOT_NUMBER_GVCOUNT]); //GV Count: 7000
	int		nStrong_Point_Area = (int)(dAlignPara[E_PARA_STRONG_SISPOT_AREA]);  // Point Area不良面积设置

	//如果没有异常Dimpe Ratio
	if (dblSpotThRatio_Active <= 0 || dblSpotThRatio_Edge <= 0 || nGVCount <= 0)
		return false;

	//如果没有异常Defect
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	// Gray Pattern Point Count
	int nGrayDefect = 0;
	int nStrong_GrayDefect = 0;

	//
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int	nImgNum1 = theApp.GetImageClassify(nImgNum);

		//仅在无名点Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT && nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_64 && resultPanelData.m_ListDefectInfo[i].Defect_Size_Pixel >= nPoint_Area)
		{
			nGrayDefect++;
		}
		//康明店
		if ((resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT || resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT) &&
			nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_64 && resultPanelData.m_ListDefectInfo[i].Defect_Size_Pixel >= nStrong_Point_Area)
		{
			nStrong_GrayDefect++;
		}

	}

	//////////////////////////////////////////////////////////////////////////
	//名点不良确认作业ver.0.0ver
	//信息:PNZ/208/04/19
	//内容:检查不良的真实性,不良坐标区域的
	//////////////////////////////////////////////////////////////////////////
	if (nGrayDefect > 0 && nPoint_Area >= 0)
	{
		//G64 Pattern原始画面Load
		cv::Mat mG64Image = MatOrgImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_GRAY_64)][0];

		// ROI Region
		int nXStart = m_stThrdAlignInfo.rcAlignCellROI.x;
		int nYStart = m_stThrdAlignInfo.rcAlignCellROI.y;
		int nXEnd = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
		int nYEnd = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//模式验证操作
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅在Point出现故障时查找
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT && resultPanelData.m_ListDefectInfo[i].Defect_Size_Pixel >= nPoint_Area
				//|| resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT	
				)
			{
				if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_64)
				{
					i++;
					continue;
				}

				////////////////////////////////
				// Max-Mean Ratio Calculation //

//			//分类为临时Max/Mean Ratio的最大值

// 					i++;

				/////////////////////////////////
							//选择验证位置Threshold Ratio//
				double	dblSpotThRatio = 0;
				int		nEdgeOffset = 100;

				//中心坐标
				int nDefectCenter_X = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nDefectCenter_Y = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				//从Edge到100 Pixel使用Edge Ratio
				if ((nXStart <= nDefectCenter_X - nEdgeOffset) ||
					(nXEnd >= nDefectCenter_X + nEdgeOffset) ||
					(nYStart <= nDefectCenter_Y - nEdgeOffset) ||
					(nYEnd >= nDefectCenter_Y + nEdgeOffset))
				{
					dblSpotThRatio = dblSpotThRatio_Edge;
				}

				else
					dblSpotThRatio = dblSpotThRatio_Active;

				////////////////////////////
				// Region Image Selection //

				// Image Offset
				int nOffset = 100;

				//起始坐标
				int nCal_X = nDefectCenter_X - nOffset;
				int nCal_Y = nDefectCenter_Y - nOffset;

				//区域设置200x200
				cv::Rect rectTemp;

				rectTemp.x = nCal_X;
				rectTemp.y = nCal_Y;
				rectTemp.width = nOffset * 2;
				rectTemp.height = nOffset * 2;

				cv::Mat matRegion = mG64Image(rectTemp);

				////////////////////////////
				// Shift Copy Enhancement //

				int nShiftUnit = 5;

				cv::Mat matBuff, matDstBuff, matTempBuff1, matTempBuff2;
				matRegion.copyTo(matBuff);
				matRegion.copyTo(matDstBuff);

				// size
				int nImageSizeX = matRegion.cols;
				int nImageSizeY = matRegion.rows;

				matTempBuff1 = matDstBuff(cv::Rect(0, 0, nImageSizeX - nShiftUnit, nImageSizeY));
				matTempBuff2 = matBuff(cv::Rect(nShiftUnit, 0, nImageSizeX - nShiftUnit, nImageSizeY));

				cv::add(matTempBuff1, matTempBuff2, matTempBuff1);

				nShiftUnit /= 2;

				cv::Mat matDstImage = cv::Mat::zeros(matRegion.size(), matRegion.type());
				matDstBuff(cv::Rect(0, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)).copyTo(matDstImage(cv::Rect(nShiftUnit, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)));

				/////////////////////////////////////////
				// Region Image Mean Value Calculation //

				int nRegionUnit = 50;

				cv::Mat matSubRegion = matDstImage(cv::Rect(nRegionUnit, nRegionUnit, matDstImage.cols - nRegionUnit * 2, matDstImage.rows - nRegionUnit * 2));
				cv::GaussianBlur(matSubRegion, matSubRegion, cv::Size(19, 19), 3.0, 3.0);

				cv::Scalar m, s;
				cv::meanStdDev(matSubRegion, m, s);
				double Imagemean = m[0];

				//提取碎片画面
				nRegionUnit = 70;
				cv::Mat matResult = matDstImage(cv::Rect(nRegionUnit, nRegionUnit, matDstImage.cols - nRegionUnit * 2, matDstImage.rows - nRegionUnit * 2));

				///////////////////////////
				// Histogram Calculation //

				cv::Mat matHisto;
				int nHistSize = 256;
				float fHistRange[] = { 0.0f, (float)(nHistSize - 1) };
				const float* ranges[] = { fHistRange };
				cv::calcHist(&matResult, 1, 0, Mat(), matHisto, 1, &nHistSize, ranges, true, false);
				int ImageImageMean = (int)(Imagemean * dblSpotThRatio);
				float* pVal = (float*)matHisto.data;

				// Diff x GV Calculation
				__int64 nPixelSum = 0;
				__int64 nPixelCount = 0;

				pVal = (float*)matHisto.ptr(0) + ImageImageMean;

				for (int m = ImageImageMean; m <= 255; m++, pVal++)
				{
					int nDiff = m - ImageImageMean;
					nPixelSum += (long)(nDiff * *pVal);
				}

				double dblDiffxGVCount = (double)nPixelSum;

				// Memory Release
				matBuff.release();
				matDstBuff.release();
				matTempBuff1.release();
				matTempBuff2.release();

				matDstImage.release();
				matSubRegion.release();
				matResult.release();

				matRegion.release();

				//清除大于设置值的错误
				if (dblDiffxGVCount > nGVCount)
				{
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT;
				}

				//resultPanelData.m_ListDefectInfo.RemoveAt(i);

				else i++;
			}

			else i++;
		}

		mG64Image.release();
	}

	//姜诗人
	if (nStrong_GrayDefect > 0 && nStrong_Point_Area >= 0)
	{
		//G64 Pattern原始画面Load
		cv::Mat mG64Image = MatOrgImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_GRAY_64)][0];

		// ROI Region
		int nXStart = m_stThrdAlignInfo.rcAlignCellROI.x;
		int nYStart = m_stThrdAlignInfo.rcAlignCellROI.y;
		int nXEnd = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
		int nYEnd = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//模式验证操作
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅在Point出现故障时查找
			if ((resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT || resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT) &&
				resultPanelData.m_ListDefectInfo[i].Defect_Size_Pixel >= nStrong_Point_Area)
			{
				if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_64)
				{
					i++;
					continue;
				}

				////////////////////////////////
				// Max-Mean Ratio Calculation //
				// 				int nDefectMaxGV	= (int)	resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
				// 				int nDefectMeanGV	= (int) resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
				// 				float MaxMeanRatio	= (float) nDefectMaxGV / nDefectMeanGV;
				// 
								//			//分类为临时Max/Mean Ratio的最大值
				// 				if ( MaxMeanRatio > 1.83 && nDefectMaxGV > 120)
				// 				{
				// 					i++;
				// 					continue;
				// 				}

				/////////////////////////////////
							//选择验证位置Threshold Ratio//
				double	dblSpotThRatio = 0;
				int		nEdgeOffset = 100;

				//中心坐标
				int nDefectCenter_X = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nDefectCenter_Y = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				//从Edge到100 Pixel使用Edge Ratio
				if ((nXStart <= nDefectCenter_X - nEdgeOffset) ||
					(nXEnd >= nDefectCenter_X + nEdgeOffset) ||
					(nYStart <= nDefectCenter_Y - nEdgeOffset) ||
					(nYEnd >= nDefectCenter_Y + nEdgeOffset))
				{
					dblSpotThRatio = dStrong_blSpotThRatio_Edge;
				}

				else
					dblSpotThRatio = dStrong_blSpotThRatio_Active;

				////////////////////////////
				// Region Image Selection //

				// Image Offset
				int nOffset = 100;

				//起始坐标
				int nCal_X = nDefectCenter_X - nOffset;
				int nCal_Y = nDefectCenter_Y - nOffset;

				//区域设置200x200
				cv::Rect rectTemp;

				rectTemp.x = nCal_X;
				rectTemp.y = nCal_Y;
				rectTemp.width = nOffset * 2;
				rectTemp.height = nOffset * 2;

				cv::Mat matRegion = mG64Image(rectTemp);

				////////////////////////////
				// Shift Copy Enhancement //

				int nShiftUnit = 5;

				cv::Mat matBuff, matDstBuff, matTempBuff1, matTempBuff2;
				matRegion.copyTo(matBuff);
				matRegion.copyTo(matDstBuff);

				// size
				int nImageSizeX = matRegion.cols;
				int nImageSizeY = matRegion.rows;

				matTempBuff1 = matDstBuff(cv::Rect(0, 0, nImageSizeX - nShiftUnit, nImageSizeY));
				matTempBuff2 = matBuff(cv::Rect(nShiftUnit, 0, nImageSizeX - nShiftUnit, nImageSizeY));

				cv::add(matTempBuff1, matTempBuff2, matTempBuff1);

				nShiftUnit /= 2;

				cv::Mat matDstImage = cv::Mat::zeros(matRegion.size(), matRegion.type());
				matDstBuff(cv::Rect(0, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)).copyTo(matDstImage(cv::Rect(nShiftUnit, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)));

				/////////////////////////////////////////
				// Region Image Mean Value Calculation //

				int nRegionUnit = 50;

				cv::Mat matSubRegion = matDstImage(cv::Rect(nRegionUnit, nRegionUnit, matDstImage.cols - nRegionUnit * 2, matDstImage.rows - nRegionUnit * 2));
				cv::GaussianBlur(matSubRegion, matSubRegion, cv::Size(19, 19), 3.0, 3.0);

				cv::Scalar m, s;
				cv::meanStdDev(matSubRegion, m, s);
				double Imagemean = m[0];

				//提取碎片画面
				nRegionUnit = 70;
				cv::Mat matResult = matDstImage(cv::Rect(nRegionUnit, nRegionUnit, matDstImage.cols - nRegionUnit * 2, matDstImage.rows - nRegionUnit * 2));

				///////////////////////////
				// Histogram Calculation //

				cv::Mat matHisto;
				int nHistSize = 256;
				float fHistRange[] = { 0.0f, (float)(nHistSize - 1) };
				const float* ranges[] = { fHistRange };
				cv::calcHist(&matResult, 1, 0, Mat(), matHisto, 1, &nHistSize, ranges, true, false);
				int ImageImageMean = (int)(Imagemean * dblSpotThRatio);
				float* pVal = (float*)matHisto.data;

				// Diff x GV Calculation
				__int64 nPixelSum = 0;
				__int64 nPixelCount = 0;

				pVal = (float*)matHisto.ptr(0) + ImageImageMean;

				for (int m = ImageImageMean; m <= 255; m++, pVal++)
				{
					int nDiff = m - ImageImageMean;
					nPixelSum += (long)(nDiff * *pVal);
				}

				double dblDiffxGVCount = (double)nPixelSum;

				// Memory Release
				matBuff.release();
				matDstBuff.release();
				matTempBuff1.release();
				matTempBuff2.release();

				matDstImage.release();
				matSubRegion.release();
				matResult.release();

				matRegion.release();

				//清除大于设置值的错误
				if (dblDiffxGVCount > nStrong_GVCount)
				{
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}

				else i++;
			}

			else i++;
		}

		mG64Image.release();
	}

	//white pattern(黑洞Mura判定)

	return true;
}

bool AviInspection::DeleteOverlapDefect_BlackHole(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	//如果没有异常参数
	if (dAlignPara == NULL)
		return false;

	// PARAMETER
	double	dblSpotThRatio_Active = (double)(dAlignPara[E_PARA_SISPOT_RATIO_ACTIVE]);	//设置为当前1
	double	dblSpotThRatio_Edge = (double)(dAlignPara[E_PARA_SISPOT_RATIO_EDGE]);		//当前设置为1.04
	int		nGVCount = (int)(dAlignPara[E_PARA_SISPOT_NUMBER_GVCOUNT]); //GV Count: 7000

	//如果没有异常处理Dimpe Ratio
	if (dblSpotThRatio_Active <= 0 || dblSpotThRatio_Edge <= 0 || nGVCount <= 0)
		return false;

	//如果没有异常Defect
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	// Gray Pattern Point Count
	int nGrayDefect = 0;

	//
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int	nImgNum1 = theApp.GetImageClassify(nImgNum);

		//仅在Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT
			)
		{
			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				nGrayDefect++;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//明细不良检查作业ver.0.0ver
		//信息:PNZ/208/04/19
		//内容:检查不良的真实性,不良坐标区域的
	//////////////////////////////////////////////////////////////////////////
	if (nGrayDefect > 0)
	{
		//G64 Pattern原始画面Load
		cv::Mat mWhiteImage = MatOrgImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_WHITE)][0];

		// ROI Region
		int nXStart = m_stThrdAlignInfo.rcAlignCellROI.x;
		int nYStart = m_stThrdAlignInfo.rcAlignCellROI.y;
		int nXEnd = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
		int nYEnd = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//模式验证操作
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			//仅在Point出现故障时查找
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT
				)
			{
				if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_64)
				{
					i++;
					continue;
				}

				////////////////////////////////
				// Max-Mean Ratio Calculation //
				// 				int nDefectMaxGV	= (int)	resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
				// 				int nDefectMeanGV	= (int) resultPanelData.m_ListDefectInfo[i].Defect_MeanGV;
				// 				float MaxMeanRatio	= (float) nDefectMaxGV / nDefectMeanGV;
				// 
								//			//分类为临时Max/Mean Ratio的最大值
				// 				if ( MaxMeanRatio > 1.83 && nDefectMaxGV > 120)
				// 				{
				// 					i++;
				// 					continue;
				// 				}

				/////////////////////////////////
							//选择验证位置Threshold Ratio//
				double	dblSpotThRatio = 0;
				int		nEdgeOffset = 100;

				//中心坐标
				int nDefectCenter_X = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nDefectCenter_Y = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				//从Edge到100 Pixel使用Edge Ratio
				if ((nXStart <= nDefectCenter_X - nEdgeOffset) ||
					(nXEnd >= nDefectCenter_X + nEdgeOffset) ||
					(nYStart <= nDefectCenter_Y - nEdgeOffset) ||
					(nYEnd >= nDefectCenter_Y + nEdgeOffset))
				{
					dblSpotThRatio = dblSpotThRatio_Edge;
				}

				else
					dblSpotThRatio = dblSpotThRatio_Active;

				////////////////////////////
				// Region Image Selection //

				// Image Offset
				int nOffset = 100;

				//起始坐标
				int nCal_X = nDefectCenter_X - nOffset;
				int nCal_Y = nDefectCenter_Y - nOffset;

				//区域设置200x200
				cv::Rect rectTemp;

				rectTemp.x = nCal_X;
				rectTemp.y = nCal_Y;
				rectTemp.width = nOffset * 2;
				rectTemp.height = nOffset * 2;

				cv::Mat matRegion = mWhiteImage(rectTemp);

				////////////////////////////
				// Shift Copy Enhancement //

				int nShiftUnit = 5;

				cv::Mat matBuff, matDstBuff, matTempBuff1, matTempBuff2;
				matRegion.copyTo(matBuff);
				matRegion.copyTo(matDstBuff);

				// size
				int nImageSizeX = matRegion.cols;
				int nImageSizeY = matRegion.rows;

				matTempBuff1 = matDstBuff(cv::Rect(0, 0, nImageSizeX - nShiftUnit, nImageSizeY));
				matTempBuff2 = matBuff(cv::Rect(nShiftUnit, 0, nImageSizeX - nShiftUnit, nImageSizeY));

				cv::add(matTempBuff1, matTempBuff2, matTempBuff1);

				nShiftUnit /= 2;

				cv::Mat matDstImage = cv::Mat::zeros(matRegion.size(), matRegion.type());
				matDstBuff(cv::Rect(0, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)).copyTo(matDstImage(cv::Rect(nShiftUnit, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)));

				/////////////////////////////////////////
				// Region Image Mean Value Calculation //

				int nRegionUnit = 50;

				cv::Mat matSubRegion = matDstImage(cv::Rect(nRegionUnit, nRegionUnit, matDstImage.cols - nRegionUnit * 2, matDstImage.rows - nRegionUnit * 2));
				cv::GaussianBlur(matSubRegion, matSubRegion, cv::Size(19, 19), 3.0, 3.0);

				cv::Scalar m, s;
				cv::meanStdDev(matSubRegion, m, s);
				double Imagemean = m[0];

				//提取碎片画面
				nRegionUnit = 70;
				cv::Mat matResult = matDstImage(cv::Rect(nRegionUnit, nRegionUnit, matDstImage.cols - nRegionUnit * 2, matDstImage.rows - nRegionUnit * 2));

				///////////////////////////
				// Histogram Calculation //

				cv::Mat matHisto;
				int nHistSize = 256;
				float fHistRange[] = { 0.0f, (float)(nHistSize - 1) };
				const float* ranges[] = { fHistRange };
				cv::calcHist(&matResult, 1, 0, Mat(), matHisto, 1, &nHistSize, ranges, true, false);
				int ImageImageMean = (int)(Imagemean * dblSpotThRatio);
				float* pVal = (float*)matHisto.data;

				// Diff x GV Calculation
				__int64 nPixelSum = 0;
				__int64 nPixelCount = 0;

				pVal = (float*)matHisto.ptr(0) + ImageImageMean;

				for (int m = ImageImageMean; m <= 255; m++, pVal++)
				{
					int nDiff = m - ImageImageMean;
					nPixelSum += (long)(nDiff * *pVal);
				}

				double dblDiffxGVCount = (double)nPixelSum;

				// Memory Release
				matBuff.release();
				matDstBuff.release();
				matTempBuff1.release();
				matTempBuff2.release();

				matDstImage.release();
				matSubRegion.release();
				matResult.release();

				matRegion.release();

				//清除大于设置值的错误
				if (dblDiffxGVCount > nGVCount)
					//resultPanelData.m_ListDefectInfo.RemoveAt(i);

					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT;

				else i++;
			}

			else i++;
		}

		mWhiteImage.release();
	}

	//white pattern(黑洞Mura判定)

	return true;
}

//使用Dust Pattern检查暗点不良是否为不良
bool AviInspection::DeleteOverlapDefect_DustDelet(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	//如果没有异常参数
	if (dAlignPara == NULL)
		return false;

	// PARAMETER
	int nOnOffKey = (int)(dAlignPara[E_PARA_DUST_DARKPOINT_ONOFF]);	//设置为当前1

	//0则禁用
	if (nOnOffKey == 0)
		return true;

	//如果没有异常Defect
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	// ALL Pattern Dark Point Count
	int nDarkDefect = 0;

	//
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int	nImgNum1 = theApp.GetImageClassify(nImgNum);

		//仅在Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
		{
			nDarkDefect++;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//检查暗点不良的真实性工作
		//信息:PNZ/208/05/17

	if (nDarkDefect > 0)
	{
		//G64 Pattern原始画面Load
		cv::Mat mDustImage = MatOrgImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_DUST)][0];

		// ROI Region
		int nXStart = m_stThrdAlignInfo.rcAlignCellROI.x;
		int nYStart = m_stThrdAlignInfo.rcAlignCellROI.y;
		int nXEnd = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
		int nYEnd = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{

			//仅在Point出现故障时查找
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
			{

				////////////////////////////
				// Region Image Selection //

				// Image Offset
				int nOffset = 100;

				//中心坐标
				int nDefectCenter_X = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nDefectCenter_Y = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				//起始坐标
				int nCal_X = nDefectCenter_X - nOffset;
				int nCal_Y = nDefectCenter_Y - nOffset;

				//区域设置200x200
				cv::Rect rectTemp;

				rectTemp.x = nCal_X;
				rectTemp.y = nCal_Y;
				rectTemp.width = nOffset * 2;
				rectTemp.height = nOffset * 2;

				cv::Mat matRegion = mDustImage(rectTemp);

				////////////////////////////
				// Shift Copy Enhancement //

				int nShiftUnit = 5;

				cv::Mat matBuff, matDstBuff, matTempBuff1, matTempBuff2;
				matRegion.copyTo(matBuff);
				matRegion.copyTo(matDstBuff);

				// size
				int nImageSizeX = matRegion.cols;
				int nImageSizeY = matRegion.rows;

				matTempBuff1 = matDstBuff(cv::Rect(0, 0, nImageSizeX - nShiftUnit, nImageSizeY));
				matTempBuff2 = matBuff(cv::Rect(nShiftUnit, 0, nImageSizeX - nShiftUnit, nImageSizeY));

				cv::add(matTempBuff1, matTempBuff2, matTempBuff1);

				nShiftUnit /= 2;

				cv::Mat matDstImage = cv::Mat::zeros(matRegion.size(), matRegion.type());
				matDstBuff(cv::Rect(0, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)).copyTo(matDstImage(cv::Rect(nShiftUnit, 0, matDstBuff.cols - nShiftUnit, matDstBuff.rows)));

				//////////////////////////////////////////////////////////////////////////
				// Big Region Image MIN/Mean/MAX Value Calculation

							//删除边缘区域
				int nBigRegionUnit = 10;

				//获取增强画面
				cv::Mat matBigSubRegion = matDstImage(cv::Rect(nBigRegionUnit, nBigRegionUnit, matDstImage.cols - nBigRegionUnit * 2, matDstImage.rows - nBigRegionUnit * 2));

				cv::GaussianBlur(matBigSubRegion, matBigSubRegion, cv::Size(11, 11), 3.0, 3.0);

				cv::Scalar sBigmean, sBigSdt;
				double dbBigminvalue;
				double dbBigmaxvalue;

				cv::meanStdDev(matBigSubRegion, sBigmean, sBigSdt);
				cv::minMaxIdx(matBigSubRegion, &dbBigminvalue, &dbBigmaxvalue, NULL, NULL);

				double dbBigImagemean = sBigmean[0];
				double dbBigImageStdDev = sBigSdt[0];
				double dbBigImageSub = dbBigmaxvalue - dbBigminvalue;

				//////////////////////////////////////////////////////////////////////////
				// Small Region Image MIN/Mean/MAX Value Calculation

							//删除边缘区域
				int nSmallRegionUnit = 70;

				//获取增强画面
				cv::Mat matSmallSubRegion = matBigSubRegion(cv::Rect(nSmallRegionUnit, nSmallRegionUnit, matBigSubRegion.cols - nSmallRegionUnit * 2, matBigSubRegion.rows - nSmallRegionUnit * 2));

				cv::Scalar sSmallmean, sSmallSdt;
				double dbSmallminvalue;
				double dbSmallmaxvalue;

				cv::meanStdDev(matSmallSubRegion, sSmallmean, sSmallSdt);
				cv::minMaxIdx(matSmallSubRegion, &dbSmallminvalue, &dbSmallmaxvalue, NULL, NULL);

				double dbSmallImagemean = sSmallmean[0];
				double dbSmallImageStdDev = sSmallSdt[0];
				double dbSmallImageSub = dbSmallmaxvalue - dbSmallminvalue;

				//////////////////////////////////////////////////////////////////////////
							//Dust判定Logic

							//初始化判定结果
				bool	bDustDefect = false;

				double	dbIniStdDev = 1.0;
				double	dbSubStdDev = 0.35;

				if ((dbBigImageStdDev <= dbIniStdDev) && (dbSmallImageStdDev <= dbIniStdDev))
				{
					if ((dbBigImageStdDev < dbSmallImageStdDev) && (dbSmallImageStdDev - dbBigImageStdDev > dbSubStdDev)) bDustDefect = true;
					else bDustDefect = false;
				}

				else if ((dbBigImageStdDev > dbIniStdDev) && (dbSmallImageStdDev > dbIniStdDev))
					bDustDefect = true;

				else if ((dbBigImageStdDev <= dbIniStdDev) && (dbSmallImageStdDev > dbIniStdDev))
				{
					if (abs(dbBigImageStdDev - dbSmallImageStdDev) <= dbSubStdDev) bDustDefect = false;
					else bDustDefect = true;
				}

				else if ((dbBigImageStdDev > dbIniStdDev) && (dbSmallImageStdDev < dbIniStdDev))
					bDustDefect = false;

				// Memory Release
				matRegion.release();
				matBuff.release();
				matDstBuff.release();
				matTempBuff1.release();
				matTempBuff2.release();
				matDstImage.release();
				matBigSubRegion.release();
				matSmallSubRegion.release();

				//清除大于设置值的错误
				if (bDustDefect == true)
					resultPanelData.m_ListDefectInfo.RemoveAt(i);

				else i++;

			}

			else i++;

		}
		mDustImage.release();
	}

	return true;
}

bool AviInspection::DeleteOverlapDefect_BlackSmallDelet(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//如果没有异常参数
	if (dAlignPara == NULL)
		return false;

	//如果没有异常Defect
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	// Black Pattern Point Count
	int nBlackPointDefect = 0;

	//设置水平/垂直方向Offset
	int nBig_Offset = 40;
	int nSmall_Offset = 5;

	//初始化判定
	bool bSmallWeakDefect = false;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int	nImgNum1 = theApp.GetImageClassify(nImgNum);

		//仅在Point不好的情况下查找
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
		{
			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_BLACK)
			{
				nBlackPointDefect++;
			}
		}
	}

	if (nBlackPointDefect > 1)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//仅在Point出现故障时查找
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
			{
				//获取Pattern信息仅查找Black Pattern中的命名错误
				int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
				int	nImgNum1 = theApp.GetImageClassify(nImgNum);

				if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_BLACK) continue;

				//获取亮度不良信息
				int nDefectMaxGV = (int)resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
				int nDefectArea = (int)resultPanelData.m_ListDefectInfo[i].Defect_Size_Pixel;

				//不强,小不良除外
				if ((nDefectMaxGV <= 200) && (nDefectArea <= 100)) continue;

				//强不良的中心坐标
				int nBig_DefectCenter_X = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nBig_DefectCenter_Y = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j)
					{
						j++;
						continue;
					}

					//仅在Point出现故障时查找
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
					{
						j++;
						continue;
					}

					//获取Pattern信息仅查找Black Pattern中的命名错误
					nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;
					nImgNum1 = theApp.GetImageClassify(nImgNum);

					if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_BLACK)
					{
						j++;
						continue;
					}

					//获取亮度不良信息
					nDefectMaxGV = (int)resultPanelData.m_ListDefectInfo[j].Defect_MaxGV;
					nDefectArea = (int)resultPanelData.m_ListDefectInfo[j].Defect_Size_Pixel;

					//不弱,排除大的不良
					if ((nDefectMaxGV >= 20) && (nDefectArea >= 10))
					{
						j++;
						continue;
					}

					//初始化判定
					bSmallWeakDefect = false;

					//中心坐标弱
					int nSmall_DefectCenter_X = (int)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X;
					int nSmall_DefectCenter_Y = (int)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y;

					//X/Y方向存在相同的现象,如果满足条件,则判定为Small Weak Defect
					if ((abs(nBig_DefectCenter_X - nSmall_DefectCenter_X) <= 5) && (abs(nBig_DefectCenter_Y - nSmall_DefectCenter_Y) <= 40) ||
						(abs(nBig_DefectCenter_X - nSmall_DefectCenter_X) <= 40) && (abs(nBig_DefectCenter_Y - nSmall_DefectCenter_Y) <= 5))
					{
						bSmallWeakDefect = true;
					}

					//如果是Small Weak名点,则删除
					if (bSmallWeakDefect == true)
					{
						if (i > j)	 i--;
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					else  j++;
				}
			}
		}
	}

	return true;
}

bool AviInspection::JudgementPSMuraBrightPointDefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;
	//Judge the numer of PSMura and BrightPoint and 
	int nPSMura = 0;
	int nBrightPoint = 0;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_PS_MURA ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_PS_MURA_1)
		{
			nPSMura++;
		}
	}
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT_1 ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT_2 ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
		{
			nBrightPoint++;
		}
	}

	int nDeleteDistance = 20;

	if (nPSMura > 0 && nBrightPoint > 0)
	{


		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{

			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_1 &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_2 &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)

			{
				continue;
			}

			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_PS_MURA &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_PS_MURA_1)
				{
					j++;
					continue;
				}
				if (abs(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X - resultPanelData.m_ListDefectInfo[j].Pixel_Center_X) < nDeleteDistance &&
					abs(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y - resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y) < nDeleteDistance)
				{
					resultPanelData.m_ListDefectInfo.RemoveAt(j);
				}
				else
				{
					j++;
				}
			}
		}
	}
	return true;
}
bool AviInspection::JudgementDUSTDOWNDefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;
	int nOffSet = 4;
	int nDustdownNum = 0;

	CRect rectDUSTDOWN;
	CRect rectMura;
	CRect rectRes;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_DUST_DOWN ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DUSTDOWN)
		{
			nDustdownNum++;
		}
	}
	if (nDustdownNum > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_DUST_DOWN &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_DUSTDOWN) {
				continue;
			}
			{

				rectDUSTDOWN.SetRect(
					(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
					(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
					(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
					(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);


				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_BLACK_GAP)
					{
						j++;
						continue;
					}

					rectMura.SetRect(
						(resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio) + nOffSet,
						(resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio) + nOffSet);



					IntersectRect(rectRes, rectDUSTDOWN, rectMura);

					if ((rectRes.Width() + rectRes.Height()) > 0) // 判断是否存在公共区域
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					else
					{
						j++;
					}

				}
			}

		}
	}
	return true;
}
bool AviInspection::JudgementZARADefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{

	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	int nZARAnum = (int)dAlignPara[E_PARA_AVI_ZARA_Number];
	int nDenseZARAnum = (int)dAlignPara[E_PARA_AVI_DENSE_ZARA_Number];
	int nDenseZARADistance = (int)dAlignPara[E_PARA_AVI_DENSE_ZARA_Distance];


	//先判断有无PS_MURA
	int nPSMura = 0;
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_PS_MURA ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_PS_MURA_1)
		{
			nPSMura++;
		}
	}

	if (nPSMura > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount();)
		{
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_ZARA)
			{
				i++;
				continue;
			}
			resultPanelData.m_ListDefectInfo.RemoveAt(i);
		}
		return true;
	}

	//判断亮点的数量是否足够多

	int nBrightNum = 0;
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
		case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
		case E_DEFECT_JUDGEMENT_POINT_BRIGHT_1:
		case E_DEFECT_JUDGEMENT_POINT_BRIGHT_2:
		case E_DEFECT_JUDGEMENT_POINT_CELL_PARTICLE:
		case E_DEFECT_JUDGEMENT_POINT_CELL_PARTICLE_1:
		case E_DEFECT_JUDGEMENT_POINT_CELL_PARTICLE_2:
		case E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT:
		case E_DEFECT_JUDGEMENT_ZARA:

			nBrightNum++;
			break;
		}
	}

	int nZaraNum = 0;
	if (nBrightNum >= nZARAnum)
	{
		int nStart_X = 0;
		int nStart_Y = 0;
		int nEnd_X = 0;
		int nEnd_Y = 0;


		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_ZARA)
			{
				continue;
			}

			nStart_X = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
			nStart_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
			nEnd_X = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
			nEnd_Y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				// 鞍篮 阂樊 亲格 厚背 陛瘤

				if (i == j)
				{
					j++;
					continue;
				}
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_ZARA)
				{
					j++;
					continue;
				}

				if (resultPanelData.m_ListDefectInfo[j].Pixel_Start_X < nStart_X)
				{
					nStart_X = resultPanelData.m_ListDefectInfo[j].Pixel_Start_X;
				}
				if (resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y < nStart_Y)
				{
					nStart_Y = resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y;
				}
				if (resultPanelData.m_ListDefectInfo[j].Pixel_End_X > nEnd_X)
				{
					nEnd_X = resultPanelData.m_ListDefectInfo[j].Pixel_End_X;
				}
				if (resultPanelData.m_ListDefectInfo[j].Pixel_End_Y > nEnd_Y)
				{
					nEnd_Y = resultPanelData.m_ListDefectInfo[j].Pixel_End_Y;
				}

				resultPanelData.m_ListDefectInfo.RemoveAt(j);
			}
			resultPanelData.m_ListDefectInfo[i].Draw_Defect_Rect = true;
			resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_ZARA;
			resultPanelData.m_ListDefectInfo[i].Pixel_Start_X = nStart_X;
			resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y = nStart_Y;
			resultPanelData.m_ListDefectInfo[i].Pixel_End_X = nEnd_X;
			resultPanelData.m_ListDefectInfo[i].Pixel_End_Y = nEnd_Y;
			resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_TOP].x = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
			resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_LEFT_TOP].y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
			resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM].x = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
			resultPanelData.m_ListDefectInfo[i].Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM].y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;

		}

	}

	else
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount();)
		{

			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_ZARA) {
				i++; continue;
			}
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
			{
				if (i == j) { j++; continue; }
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_ZARA) { j++; continue; }
				if (abs(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X - resultPanelData.m_ListDefectInfo[j].Pixel_Center_X) < nDenseZARADistance &&
					abs(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y - resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y) < nDenseZARADistance)
				{
					nZaraNum++;
					if (nZaraNum > 1)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					else { j++; }

				}
				else { j++; }
			}
			if (nZaraNum >= int(nDenseZARAnum - 1))
			{
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_DENSE_ZARA;
				i++;
			}
			else
			{
				resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}
		}
	}

	return true;
}

bool AviInspection::DeleteOverlapDefect_Point_Point(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//17.07.14-P/S模式下的坐标比较
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	int nOffSet = (int)(dAlignPara[E_PARA_Duplicate_offset]);

	//int nOffSet = 6;

	//////////////////////////////////////////////////////////////////////////
		//17.10.24-White中的明点诗人/Black中的明点未诗人=>气泡
	//////////////////////////////////////////////////////////////////////////

		//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		//仅在Point不良的情况下...
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
		{
			i++;
			continue;
		}

		//只有在White模式下...
		int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_WHITE)
		{
			i++;
			continue;
		}

		//不良中心坐标
		CPoint ptSrc1;
		ptSrc1.x = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio);
		ptSrc1.y = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

		//White亮度不良
		int nWhiteGV = resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;

		//确认气泡判定
		bool bBubble = true;

		//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT)
			{
				j++;
				continue;
			}

			//			//仅在Black模式下...

			// 				j++;

									//不良中心坐标
			CPoint ptSrc2;
			ptSrc2.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
			ptSrc2.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

			//如果不良中心点相同的话...
			if (abs(ptSrc1.x - ptSrc2.x) < nOffSet &&
				abs(ptSrc1.y - ptSrc2.y) < nOffSet)
			{
				//判定为亮点(白黑重复)
				bBubble = false;
				/*
							2023/10/19hjf锦织美奂宅混合美奂美奂
				*/
				///////////////////////////////////////
				if (!bBubble &&
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT)
				{
					resultPanelData.m_ListDefectInfo.GetAt(j) = resultPanelData.m_ListDefectInfo.GetAt(i);
				}
				/////////////////////////end

								//退出
				break;
			}
			//如果不能重复的话...下一个不良...
			else j++;
		}

		//如果White&Black重复,则判定明点
		if (!bBubble)
		{
			//删除White模式错误

			resultPanelData.m_ListDefectInfo.RemoveAt(i);

			continue;
		}
		//如果不是重复,则判定气泡
		else
		{
			//////////////////////////////////////////////////////////////////////////
						//17.12.14-第二次气泡检查start
						//如果是气泡引起的,光线聚集在一起确认的话
						//点灯光线暗的话看不清
						//如果不良GV值为White<Gray,则很有可能是真的不良

						//无法比较
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				//禁止比较同一不良项目
				if (i == j)
				{
					j++;
					continue;
				}

				//仅在Point不良的情况下...
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT)
				{
					j++;
					continue;
				}

				//仅在灰色模式下...
				nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;
				if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_32 &&
					theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_64 &&
					theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_87 &&
					theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_128)
				{
					j++;
					continue;
				}

				//不良中心坐标
				CPoint ptSrc2;
				ptSrc2.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
				ptSrc2.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

				//如果不良中心点相同的话...
				if (abs(ptSrc1.x - ptSrc2.x) < nOffSet &&
					abs(ptSrc1.y - ptSrc2.y) < nOffSet)
				{
					//灰色亮度不良
					int nGrayGV = resultPanelData.m_ListDefectInfo[j].Defect_MaxGV;

					//如果灰色亮度比White亮...
					if (nWhiteGV <= nGrayGV)
					{
						//判定为亮点(重复White Black)
						bBubble = false;

						//退出
						break;
					}
					//如果灰色亮度比White暗...下一个不良...
					else j++;
				}
				//如果不能重复的话...下一个不良...
				else j++;
			}

			//白色和黑色重复,判定明点
			if (!bBubble)
			{
				//删除White模式错误
				resultPanelData.m_ListDefectInfo.RemoveAt(i);

				continue;
			}

			//17.12.14-检查第二次气泡end
//////////////////////////////////////////////////////////////////////////
		}

		//删除气泡等坐标
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT)
			{
				j++;
				continue;
			}

			//不良中心坐标
			CPoint ptSrc2;
			ptSrc2.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
			ptSrc2.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

			//如果不良中心点相同的话...
			if (abs(ptSrc1.x - ptSrc2.x) < nOffSet &&
				abs(ptSrc1.y - ptSrc2.y) < nOffSet)
			{
				if (i > j)   i--;

				//删除相应的错误
				resultPanelData.m_ListDefectInfo.RemoveAt(j);
			}
			//如果不能重复的话...下一个不良...
			else j++;
		}

		//删除White模式错误
		resultPanelData.m_ListDefectInfo.RemoveAt(i);
	}

	//////////////////////////////////////////////////////////////////////////
		//18.01.15-White中的暗点诗人/暗点大小比较不包括过检因素
	//////////////////////////////////////////////////////////////////////////

		//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		//仅在Point不良的情况下...
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_DARK &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
		{
			i++;
			continue;
		}

		//只有在White模式下...
		int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_WHITE)
		{
			i++;
			continue;
		}

		//White不良范围
		CRect rectTemp = CRect(
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - 40,
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - 40,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + 40,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + 40);

		//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			//仅在Point不良的情况下...
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_DARK &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_DARK &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_RETEST_POINT_DARK &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK)
			{
				j++;
				continue;
			}

			//不良中心坐标
			CPoint ptSrc;
			ptSrc.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
			ptSrc.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

			//范围内有不良存在吗？
			if (rectTemp.PtInRect(ptSrc))
			{
				if (i > j)   i--;

				//删除相应的错误
				resultPanelData.m_ListDefectInfo.RemoveAt(j);
			}
			//如果不能重复的话...下一个不良...
			else j++;
		}

		//删除White模式错误
		resultPanelData.m_ListDefectInfo.RemoveAt(i);
	}

	//////////////////////////////////////////////////////////////////////////
		//明点&暗点&派对比较
	//////////////////////////////////////////////////////////////////////////

		//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//仅在Point不良的情况下...
		if ((resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_POINT_DARK &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_POINT_DARK_SP_3) ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_POINT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_MULT_BP)	//17.09.28添加
		{
			//不良中心坐标
			//17.07.14-P/S模式下的坐标比较
			CPoint ptSrc1;
			ptSrc1.x = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio);
			ptSrc1.y = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

			//无法比较
			for (int j = i + 1; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{

				//不包括Point不良
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type < E_DEFECT_JUDGEMENT_POINT_DARK ||
					(resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type < E_DEFECT_JUDGEMENT_MURA_MULT_BP) |	//17.09.28添加
					(resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_MURA_MULT_BP &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type <= E_DEFECT_JUDGEMENT_RETEST_POINT_DARK) /*||
										resultPanelData.m_ListDefectInfo[j].Defect_Type>E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT*/)
										//B11客户要求->Retest Point即使存在重复故障,也不会移除
				{
					j++;
					continue;
				}

				if ((resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK) ||
					(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK))
				{
					j++;
					continue;
				}

				//不良中心坐标
					//17.07.14-P/S模式下的坐标比较
				CPoint ptSrc2;
				ptSrc2.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
				ptSrc2.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

				//如果不良中心点相同的话...
				if (abs(ptSrc1.x - ptSrc2.x) < nOffSet && abs(ptSrc1.y - ptSrc2.y) < nOffSet)
				{
					int nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;

					//17.09.11-如果重复,请先报告黑色模式,而不是其他模式(客户请求)
					if (theApp.GetImageClassify(nImgNum) == E_IMAGE_CLASSIFY_AVI_BLACK)
					{
						ResultDefectInfo tTemp1 = resultPanelData.m_ListDefectInfo[i];
						ResultDefectInfo tTemp2 = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp2;
						resultPanelData.m_ListDefectInfo[j] = tTemp1;
					}

					//17.12.14-重复时,Gray模式优先于其他模式报告

										//17.10.12-多重优先报告(客户请求)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_MULT_BP)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					//如果是明/暗点,则修改判定

					//如果是明/暗点,则修改判定

						//////////////////////////////////////////////////////////////////////////
											//PNZ2018-05-08客户要求:Point_Bright_Dark不要出现不良

					else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK)
					{
					}
					//如果是名/暗点,则修改判定
					else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}

					////如果是小名/暗点,则修改判定
//else if(	resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT	&&
//			resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK		)
//{
//	resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK;
//}
					////如果是小名/暗点,则修改判定
//else if(resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK		&&
//		resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT	)
//{
//	resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK;
//}
					//名/暗点时,清除其他不良
					else if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					//删除Retest
					else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_POINT_DARK)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					//删除Retest
					else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					// choi 20.06.23
					else if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_DARK_SP_3)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					//17.07.19-群集优先级
					else if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					//17.07.19-群集优先级
					else if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT)
					{
						ResultDefectInfo tTemp = resultPanelData.m_ListDefectInfo[j];
						resultPanelData.m_ListDefectInfo[i] = tTemp;
					}
					//如果是Particle,就留下,清除其他不良
					//如果是暗影-队伍:删除暗点

											//删除错误
					resultPanelData.m_ListDefectInfo.RemoveAt(j);
				}
				else j++;  // 如果不良中心点不一样的话...下一个不良...
			}
		}
	}

	//		//派对-明点&暗点比较

	//		//不良数量

	//			//由于Particle不良大小较大,仅中心点无法删除时发生
	//			//如果以Particle内部为中心,则删除

			//17.07.14-P/S模式下的坐标比较
	// 				CRect rectTemp = CRect(

			//无法比较

					//禁止比较劣质项目
	// 					if (i == j)

	// 						j++;

					//17.07.14-P/S模式下的坐标比较
					//不良中心坐标

					//范围内是否存在不良行为？

	// 						if (i > j)   i--;

					//删除相应的错误

	//					//如果不能重复的话...下一个不良...
	// 					else j++;

	//		//派对-明点&暗点比较

	//		//不良数量

	//			//由于Particle不良大小较大,仅中心点无法删除时发生
	//			//如果以Particle内部为中心,则删除

			//17.07.14-P/S模式下的坐标比较
	// 				CRect rectTemp = CRect(

			//无法比较

					//禁止比较劣质项目
	// 					if (i == j)

	// 						j++;

					//17.07.14-P/S模式下的坐标比较
					//不良中心坐标

					//范围内是否存在不良行为？

	// 						if (i > j)   i--;

					//删除相应的错误

	//					//如果不能重复的话...下一个不良...
	// 					else j++;

	return true;
}

bool AviInspection::DeleteOverlapDefect_Point_Line(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//17.07.14-P/S模式下的坐标比较
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	int nOffSet = 8; //choi 05.21 4->8

	//////////////////////////////////////////////////////////////////////////
		//线条-明点&暗点&派对比较
	//////////////////////////////////////////////////////////////////////////

		//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//17.07.17-Line不良内有Point不良时->Line Alg决定
		//17.05.09线路不良内有Point不良时,应报告
		//如果Line有问题
		if ((resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_LINE_Y_EDGE_BRIGHT) ||
			(resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_RETEST_LINE_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_RETEST_LINE_DARK) ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_PCD_CRACK)
		{
			CRect rectTemp = CRect(
				(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
				(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
				(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
				(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);

			//无法比较
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				//禁止比较同一不良项目
				if (i == j)
				{
					j++;
					continue;
				}

				//不包括Point不良
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type < E_DEFECT_JUDGEMENT_POINT_DARK ||
					(resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type < E_DEFECT_JUDGEMENT_MURA_MULT_BP) |	//17.09.28添加
					(resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_MURA_MULT_BP &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type < E_DEFECT_JUDGEMENT_RETEST_POINT_DARK) ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT)
				{
					j++;
					continue;
				}

				//不良中心坐标
				CPoint ptSrc;
				ptSrc.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
				ptSrc.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

				//仅重复数据删除Line-Point故障
				//范围内有不良存在吗？
				if (rectTemp.PtInRect(ptSrc))
				{
					//删除低于当前列表的列表时...
					if (i > j)	i--;

					//删除错误
					resultPanelData.m_ListDefectInfo.RemoveAt(j);
				}
				//如果不能重复的话...下一个不良...
				else j++;
			}
		}
	}

	return true;
}

bool AviInspection::DeleteOverlapDefect_Point_Mura(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//比较P/S模式下的坐标
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//////////////////////////////////////////////////////////////////////////
		//KJY 17.09.18-消除漏气附近的积分
		//如果有很多不良情况,速度可能会变慢,在删除上面的点群相关后运行
		//反正漏气和EMD等都是明显的不良,比该不良更能去除积分。
	//////////////////////////////////////////////////////////////////////////

	if (dAlignPara != NULL)
	{
		int		nDefectTypeM, nDefectTypeS;
		int		nAdjustRange = (int)dAlignPara[E_PARA_STRONG_DEFECT_NOISE_ADJUST_RANGE];
		CRect	rcDeleteArea;
		CPoint	ptCenterSub;

		//像素不良时删除村上
		if (nAdjustRange > 0)
		{
			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)//lxq--20230605
			{
				nDefectTypeM = resultPanelData.m_ListDefectInfo[i].Defect_Type;

				if (nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_AMORPH_DARK)
				{
					rcDeleteArea.SetRect(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio,
						resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio,
						resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio,
						resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

					rcDeleteArea.NormalizeRect();

					//rcDeleteArea.InflateRect(nAdjustRange, nAdjustRange);

					for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
					{
						if (i == j)
						{
							j++;
							continue;
						}

						nDefectTypeS = resultPanelData.m_ListDefectInfo[j].Defect_Type;

						if (nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_DARK ||
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK ||
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_DARK_SP_1 ||
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_DARK_SP_2 ||
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_DARK_SP_3)
						{
							ptCenterSub.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
							ptCenterSub.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

							if (rcDeleteArea.PtInRect(ptCenterSub))
							{
								ResultDefectInfo tTemp1 = resultPanelData.m_ListDefectInfo[i];
								ResultDefectInfo tTemp2 = resultPanelData.m_ListDefectInfo[j];
								resultPanelData.m_ListDefectInfo[i] = tTemp2;
								resultPanelData.m_ListDefectInfo[j] = tTemp1;

								rcDeleteArea.SetRect(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio,
									resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio,
									resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio,
									resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

								resultPanelData.m_ListDefectInfo.RemoveAt(j);

								if (i > j)	i--;
							}
							else
							{
								j++;
								continue;
							}
						}
						else j++;
					}
				}
			}
			//不良数量
			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{
				nDefectTypeM = resultPanelData.m_ListDefectInfo[i].Defect_Type;

				//查找对象不良。
				if (nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_NUGI ||		//点击
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI ||		//外圈垂直水平漏电
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI_ |		//外圈垂直水平漏电
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_EMP ||		// EMP
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_EMD_BRIGHT ||		// EMD
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_EMD_DARK ||		// EMD					
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_CORNER_CM |		//混色
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_UP_CM |		//混色
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_FINGER_CM |		//混色
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_BOX_SCRATCH |		//剪裁(F级...)
					nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT)		//2018-05-08 PNZ Mura不良
				{
					//P/S模式坐标校正
				//大小设置为盒子-虽然有圆形的漏气,但也存在很多吉利的形态
					rcDeleteArea.SetRect(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio,
						resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio,
						resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio,
						resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

					//坐标系规范化
					rcDeleteArea.NormalizeRect();

					//范围扩展
					rcDeleteArea.InflateRect(nAdjustRange, nAdjustRange);

					//不良数量
					for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
					{
						//禁止比较同一不良项目
						if (i == j)
						{
							j++;
							continue;
						}

						nDefectTypeS = resultPanelData.m_ListDefectInfo[j].Defect_Type;

						//确认该不良是积分不良之一
						if (nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_DARK ||		//暗点
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||		//亮点
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT ||		//要点
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK |		//对比度
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_GROUP_DARK |		//组暗点
							nDefectTypeS == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT ||		//组名称
							nDefectTypeS == E_DEFECT_JUDGEMENT_RETEST_POINT_DARK ||		//重新检查暗点
							nDefectTypeS == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT ||		//重新检查名点
							nDefectTypeS == E_DEFECT_JUDGEMENT_MURA_MULT_BP)
						{
							//如果该不良行为在black pattern中是亮点,则不删除。和Mura一起看pwj
							int nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;
							if (theApp.GetImageClassify(nImgNum) == E_IMAGE_CLASSIFY_AVI_BLACK)
							{
								j++;
								continue;
							}
							//P/S模式坐标校正
							ptCenterSub.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
							ptCenterSub.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

							//进入范围后删除
							if (rcDeleteArea.PtInRect(ptCenterSub))
							{
								//删除Point错误
								resultPanelData.m_ListDefectInfo.RemoveAt(j);

								//删除低于当前列表的列表时...
								if (i > j)	i--;
							}
							else
							{
								j++; // 只有在没有清除的情况下,才会出现以下不良情况:
								continue;
							}
						}
						else j++; // 如果不是积分不良,则为下一个不良
					}
				}
			}
		}
	}

	//19-08-20 choikwangil注释->choi05.01注释解除
//////////////////////////////////////////////////////////////////////////
//	//Mura Spot&如果是亮点->删除亮点&Mura报告
// 	//////////////////////////////////////////////////////////////////////////
// 
	int nOffSet = 5;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//Mura除非是亮Spot不良
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT)//改为删除mura boe11choikwangil->05.01choi原装

			continue;

		CRect rectTemp = CRect(
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);

		//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			//不包括Point不良
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_MULT_BP)
				//			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT)
			{
				j++;
				continue;
			}
			// 

						//不良中心坐标
			CPoint ptSrc;
			ptSrc.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
			ptSrc.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

			//范围内有不良存在吗？
			if (rectTemp.PtInRect(ptSrc))
			{

				//同时发现黑色图案名牌和S级木乃伊时不进行重复数据删除
				int nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;
				if (theApp.GetImageClassify(nImgNum) == E_IMAGE_CLASSIFY_AVI_BLACK && resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT)//如果是s级点,将其视为黑色模式明点,pwj 20.08.25(强点判定正常)
				{
					j++;
					continue;
				}

				double meanRatio = resultPanelData.m_ListDefectInfo[j].Defect_MeanGV / (double)resultPanelData.m_ListDefectInfo[j].Defect_Size_Pixel;

				//删除低于当前列表的列表时...
				if (i > j)	i--;

				//删除错误
				resultPanelData.m_ListDefectInfo.RemoveAt(j);

			}

			//如果不能重复的话...下一个不良...
			else j++;
		}
	}

	return true;
}

bool AviInspection::DeleteOverlapDefect_White_Spot_Mura_RGBBlk_Point(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//比较P/S模式下的坐标
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//////////////////////////////////////////////////////////////////////////
	// B11
	// CKI 19.08.20
		//为了区分百分村和名店,将G64百分村从R,G,B,Black中删除为名店
	//////////////////////////////////////////////////////////////////////////

	if (dAlignPara != NULL)
	{
		int		nDefectTypeM, nDefectTypeS;
		int		nAdjustRange = (int)dAlignPara[E_PARA_STRONG_DEFECT_NOISE_ADJUST_RANGE];
		CRect	rcDeleteArea;
		CPoint	ptCenterSub;

		int	nImgNum = 0;
		int	nImgNum1 = 0;

		//像素不良时删除村上
		if (nAdjustRange > 0)
		{
			//不良数量
			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{
				//关于Pattern
				nImgNum = 0;
				nImgNum1 = 0;
				nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
				nImgNum1 = theApp.GetImageClassify(nImgNum);

				//仅当Point故障位于R,G,B
				if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_R | nImgNum1 == E_IMAGE_CLASSIFY_AVI_G | nImgNum1 == E_IMAGE_CLASSIFY_AVI_B)//04.09 choikwangil黑色模式排除
				{

					//Defect_Type
					nDefectTypeM = resultPanelData.m_ListDefectInfo[i].Defect_Type;

					if (		//暗点
						nDefectTypeM == E_DEFECT_JUDGEMENT_POINT_BRIGHT ||		//亮点
						nDefectTypeM == E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT |		//要点
						nDefectTypeM == E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT ||		//组名称
						nDefectTypeM == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT ||		//重新检查名点
						nDefectTypeM == E_DEFECT_JUDGEMENT_MURA_MULT_BP)
					{

						//P/S模式坐标校正
						//大小设置为盒子-虽然有圆形的湿气,但也存在很多吉利的形态

//坐标系规范化

//范围扩展

						//P/S模式坐标校正
						ptCenterSub.x = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio); //choikwangil04.10修改
						ptCenterSub.y = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

						//不良数量
						for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
						{
							//禁止比较劣质项目
							if (i == j)
							{
								j++;
								continue;
							}

							nDefectTypeS = resultPanelData.m_ListDefectInfo[j].Defect_Type;

							if (nDefectTypeS == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT | nDefectTypeS == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT)	//白点无拉				//04.16 choi		
							{
								//P/S模式坐标校正

								rcDeleteArea.SetRect(resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio,	//修改choikwangil04.10
									resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio,
									resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio,
									resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

								//坐标系规范化
								rcDeleteArea.NormalizeRect();

								//范围扩展
								rcDeleteArea.InflateRect(nAdjustRange, nAdjustRange);

								//进入范围后删除
								if (rcDeleteArea.PtInRect(ptCenterSub))
								{
									//删除White Spot错误
									resultPanelData.m_ListDefectInfo.RemoveAt(j);

									//删除低于当前列表的列表时...
									if (i > j)	i--;
								}
								else
								{
									j++; // 只有在没有清除的情况下,才会出现以下不良情况:
									continue;
								}
							}
							else j++; // 如果不是积分不良,则为下一个不良
						}
					}
				}
			}
		}
	}

	return true;
}

//	//比较P/S模式下的坐标
//	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

// 	// B11
// 	// CKI 19.08.20
//	//在G64模式下,使用MaxGV/MeanGV裁判明点和白点村

// 		int	nImgNum = 0;
// 		int	nImgNum1 = 0;

// 		double MaxGV_Ratio = 0.0;

// 		if (MaxGV_Judge > 0)

//			//不良数量

//			//关于Pattern
// 			nImgNum = 0;
// 			nImgNum1 = 0;
// 			MaxGV_Ratio = 0.0;

// 				if (MaxGV_Ratio <= MaxGV_Judge) {

bool AviInspection::DeleteOverlapDefect_Line_Mura(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	int nOffSet = 4;

	//声明删除行村的变量(范围扩展)
	int nAdjustRange = 15;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//查找对象不良
		if ((resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_LINE_Y_EDGE_BRIGHT) ||
			(resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_RETEST_LINE_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_RETEST_LINE_DARK) ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
		{
			CRect rectLine = CRect(
				(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
				(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
				(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
				(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);

			//坐标系规范化
			rectLine.NormalizeRect();

			rectLine.InflateRect(nAdjustRange, nAdjustRange);

			//无法比较
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				//禁止比较同一不良项目
				if (i == j)
				{
					j++;
					continue;
				}

				//确认该不良是积分不良之一
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT || //04.16 choi

					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_EMD_BRIGHT ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_EMD_DARK ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_UP_CM ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_CORNER_CM ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_FINGER_CM ||
					//resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_NUGI		|	//18.05.21-客户要求:单独报告漏气和线路故障
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BN_CORNER ||
					//resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_STAMPED		|	//18.05.21-客户要求:单独报告漏气和线路不良
					//resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI		|	//18.05.21-客户要求:单独报告漏气和线路故障
					//resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI_		|	//18.05.21-客户要求:单独报告漏气和线路不良
//resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT	||  // choi 06.04
resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_AMORPH_DARK ||
resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BOX_SCRATCH ||
resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_RETEST_MURA)
				{
					CRect rectMura = CRect(
						(resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio) + nOffSet,
						(resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio) + nOffSet);

					CRect rectRes;
					IntersectRect(rectRes, rectLine, rectMura);

					if ((rectRes.Width() + rectRes.Height() > 0))//进入范围后删除
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(j);

						//删除小列表时...
						if (i > j)	 i--;
					}
					else
					{
						j++; // 只有在没有清除的情况下,才会出现以下不良情况:
						continue;
					}
				}
				//如果不能重复的话...下一个不良...
				else j++;
			}
		}
	}

	return true;
}

bool AviInspection::DeleteOverlapDefect_Black_Mura_and_Judge(ResultPanelData& resultPanelData, double* dAlignPara)//choikwangil04.07函数
{
	//比较P/S模式下的坐标
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//pwj
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//不良数量

	int	nImgNum = 0;

	int nImgNum1 = 0;

	int		nAdjustRange = (int)dAlignPara[E_PARA_STRONG_DEFECT_NOISE_ADJUST_RANGE];

	CRect	rcDeleteArea;
	CPoint	ptCenterSub;

	int nOffSet = 20; // choi 05.07

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//排除Mura不不良(不包括多个BP)
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT);
		else continue;

		CRect rectM = CRect(
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);

		//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT || resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT) { //04.16 choi

				CRect rectS = CRect(
					(resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio),
					(resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio),
					(resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio),
					(resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio));

				//检查重复区域
				CRect rectRes;
				IntersectRect(rectRes, rectM, rectS);

				//进入范围后临时更改名称
				if ((rectRes.Width() + rectRes.Height()) > 0)
				{

					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_CM;

					//删除低于当前列表的列表时...
					if (i > j)	i--;

					//删除错误
					resultPanelData.m_ListDefectInfo.RemoveAt(j);
				}
				//如果不能重复的话...下一个不良...
				else
				{
					j++;
					continue;
				}
			}

			///////////////////////////////////////////////////////////////////////////
			//Point
			///////////////////////////////////////////////////////////////////////////

// 				nImgNum = 0;
// 				nImgNum1 = 0;

// 				if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_64) {
// 					j++;

				//P/S模式坐标校正
			//大小设置为盒子-虽然有圆形的漏气,但也存在很多吉利的形态

				//坐标系规范化

				//范围扩展

				//进入范围后删除

				//删除低于当前列表的列表时...
// 						if (i > j)	i--;

				//删除相应的错误

//					//如果不能重复的话...下一个不良...

// 						j++;

			else
			{
				j++;
				continue;
			}

		}
	}

	//删除black mura和白点mura不重叠的所有内容
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{

		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT)
		{
			resultPanelData.m_ListDefectInfo.RemoveAt(i);
			i--;
		}
	}

	//恢复临时更改的名称
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{

		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_CM)
		{
			resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT;
		}
	}

	return true;

}

bool AviInspection::DeleteOverlapDefect_Mura_Mura(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//比较P/S模式下的坐标
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//////////////////////////////////////////////////////////////////////////
		//消除重复的Mura错误
	//////////////////////////////////////////////////////////////////////////

	int nOffSet = 0;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//排除Mura不不良(不包括多个BP)
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_MURA_START ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type > E_DEFECT_JUDGEMENT_MURA_END ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_X ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_X ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_Y ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BOX_SCRATCH) //choi 06.04
			continue;

		CRect rectM = CRect(
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);

		//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			//不包括Mura不良(不包括多个BP)
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type <= E_DEFECT_JUDGEMENT_MURA_START ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_MURA_END ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_X ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_X ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_Y ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BOX_SCRATCH) //choi 06.04
			{
				j++;
				continue;
			}

			//choikwangil 12.03 black hole
			if ((resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT && resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT) ||
				(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT && resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT) ||
				(resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT && resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT) || //04.16 choi
				(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT && resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT) ||
				(resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT && resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT) ||
				(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT && resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT) ||
				(resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT && resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT) || //05.26 choi
				(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT && resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT)
				)
			{
				j++;
				continue;
			}

			/////////////////////////////////////

// 				j++;

			/////////////////////////////////////

			CRect rectS = CRect(
				(resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio),
				(resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio),
				(resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio),
				(resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio));

			//检查重复区域
			CRect rectRes;
			IntersectRect(rectRes, rectM, rectS);

			//进入范围后删除
			if ((rectRes.Width() + rectRes.Height()) > 0)
			{
				//百分&享受&留下下部拍摄的优先顺序
				if (//resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT || //choi 06.04
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT ||
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT |	//满分
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT ||	// 04.16 choi
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_NUGI ||	//漏电
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI ||	//保持边缘
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI_ |	//边缘拉伸
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_STAMPED)	//下部拍摄
				{

					ResultDefectInfo tTemp1 = resultPanelData.m_ListDefectInfo[i];
					ResultDefectInfo tTemp2 = resultPanelData.m_ListDefectInfo[j];
					resultPanelData.m_ListDefectInfo[i] = tTemp2;
					resultPanelData.m_ListDefectInfo[j] = tTemp1;

					//重新修改范围
					rectM = CRect(
						(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
						(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);
				}

				//删除低于当前列表的列表时...
				if (i > j)	i--;

				//删除相应的错误
				resultPanelData.m_ListDefectInfo.RemoveAt(j);
			}
			//如果不能重复的话...下一个不良...
			else j++;
		}
	}

	//pwj
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//不良数量
	/*for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
				//排除Mura不不良(不包括多个BP)
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_MURA_START ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type > E_DEFECT_JUDGEMENT_MURA_END ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_X ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_X ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_Y)
			continue;

		CRect rectM = CRect(
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
			(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);

				//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
						//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

						//不包括Mura不良(不包括多个BP)
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type <= E_DEFECT_JUDGEMENT_MURA_START ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type > E_DEFECT_JUDGEMENT_MURA_END ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_X ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_X ||
				resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_Y)
			{
				j++;
				continue;
			}

			//choikwangil 12.03 black hole
			// 			if ((resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT&&resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT) ||
			// 				(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT&&resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT))
			// 			{
			// 				j++;
			// 				continue;
			// 			}
			/////////////////////////////////////

			CRect rectS = CRect(
				(resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio),
				(resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio),
				(resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio),
				(resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio));

						//检查重复区域
			CRect rectRes;
			IntersectRect(rectRes, rectM, rectS);

						//进入范围后删除
			if ((rectRes.Width() + rectRes.Height()) > 0)
			{
							//百分&享受&留下下部拍摄的优先顺序
				// 				if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT ||
								// 					//resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT|	//满分
								// 					resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_NUGI||	//漏电
								// 					resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI||	//保持边缘
								// 					resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI_|	//边缘拉伸
								// 					resultPanelData.m_ListDefectInfo[j].Defect_Type==E_DEFECT_JUDGEMENT_MURA_STAMPED)	//下部拍摄
				// 				{

				if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT && resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT)
				{
										resultPanelData.m_ListDefectInfo[j].Defect_Type = E_DEFECT_JUDGEMENT_MURA_CM; // 临时将真不良更改为其他名称

					ResultDefectInfo tTemp1 = resultPanelData.m_ListDefectInfo[i];
					ResultDefectInfo tTemp2 = resultPanelData.m_ListDefectInfo[j];
					resultPanelData.m_ListDefectInfo[i] = tTemp2;
					resultPanelData.m_ListDefectInfo[j] = tTemp1;

									//重新修改范围
					rectM = CRect(
						(resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio) - nOffSet,
						(resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet,
						(resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio) + nOffSet);
				}
				//}

				if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT && resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT)
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_CM;

							//删除低于当前列表的列表时...
				if (i > j)	i--;

							//删除相应的错误
				resultPanelData.m_ListDefectInfo.RemoveAt(j);
			}
						//如果不能重复的话...下一个不良...
			else j++;
		}
	}

		//删除black mura和白点mura不重叠的所有内容
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{

		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT)
		{
			resultPanelData.m_ListDefectInfo.RemoveAt(i);
			i--;
		}
	}

		//恢复临时更改的名称
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{

		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_MURA_CM)
		{
			resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT;
		}
	}*/

	return true;
}

bool AviInspection::GetDefectInfo(CWriteResultInfo WrtResultInfo, ResultDefectInfo* pResultDefectInfo, stDefectInfo* pResultBlob, int nBlobCnt, int nImageNum, int nCameraNum, int nRatio)
{
	//AVI使用未旋转的原始画面坐标,因此在向上报告时,必须旋转坐标以计算工作坐标等。
	_tcscpy_s(pResultDefectInfo->Defect_Code, _T("CODE"));
	pResultDefectInfo->Defect_Rect_Point[E_CORNER_LEFT_TOP] = cv::Point(pResultBlob->ptLT[nBlobCnt].x, pResultBlob->ptLT[nBlobCnt].y);
	pResultDefectInfo->Defect_Rect_Point[E_CORNER_RIGHT_TOP] = cv::Point(pResultBlob->ptRT[nBlobCnt].x, pResultBlob->ptRT[nBlobCnt].y);
	pResultDefectInfo->Defect_Rect_Point[E_CORNER_RIGHT_BOTTOM] = cv::Point(pResultBlob->ptRB[nBlobCnt].x, pResultBlob->ptRB[nBlobCnt].y);
	pResultDefectInfo->Defect_Rect_Point[E_CORNER_LEFT_BOTTOM] = cv::Point(pResultBlob->ptLB[nBlobCnt].x, pResultBlob->ptLB[nBlobCnt].y);
	pResultDefectInfo->nBlockNum = pResultBlob->nBlockNum[nBlobCnt];//分区编号标识  hjf
	cv::Point ptRotate(0, 0);

	//不是AD的不良...
	if (pResultBlob->nDefectJudge[nBlobCnt] > E_DEFECT_JUDGEMENT_DUST_GROUP)
	{
		cv::Point ptTemp(pResultBlob->ptLT[nBlobCnt].x, pResultBlob->ptLT[nBlobCnt].y);
		Align_DoRotatePoint(ptTemp, ptRotate, m_stThrdAlignInfo.ptAlignCenter, m_stThrdAlignInfo.dAlignTheta);
		pResultDefectInfo->Pixel_Start_X = ptRotate.x;
		pResultDefectInfo->Pixel_Start_Y = ptRotate.y;

		ptTemp.x = pResultBlob->ptRB[nBlobCnt].x;
		ptTemp.y = pResultBlob->ptRB[nBlobCnt].y;
		Align_DoRotatePoint(ptTemp, ptRotate, m_stThrdAlignInfo.ptAlignCenter, m_stThrdAlignInfo.dAlignTheta);
		pResultDefectInfo->Pixel_End_X = ptRotate.x;
		pResultDefectInfo->Pixel_End_Y = ptRotate.y;
	}
	//如果AD不正确,则返回完整坐标而不旋转
	else
	{
		pResultDefectInfo->Pixel_Start_X = pResultBlob->ptLT[nBlobCnt].x;
		pResultDefectInfo->Pixel_Start_Y = pResultBlob->ptLT[nBlobCnt].y;
		pResultDefectInfo->Pixel_End_X = pResultBlob->ptRB[nBlobCnt].x;
		pResultDefectInfo->Pixel_End_Y = pResultBlob->ptRB[nBlobCnt].y;
	}

	pResultDefectInfo->Pixel_Center_X = (int)(pResultDefectInfo->Pixel_Start_X + pResultDefectInfo->Pixel_End_X) / 2;
	pResultDefectInfo->Pixel_Center_Y = (int)(pResultDefectInfo->Pixel_Start_Y + pResultDefectInfo->Pixel_End_Y) / 2;

	//指定倾斜的源图像中要Crop的坏区域
	pResultDefectInfo->Pixel_Crop_Start_X = min(pResultBlob->ptLT[nBlobCnt].x, pResultBlob->ptLB[nBlobCnt].x);
	pResultDefectInfo->Pixel_Crop_Start_Y = min(pResultBlob->ptLT[nBlobCnt].y, pResultBlob->ptRT[nBlobCnt].y);
	pResultDefectInfo->Pixel_Crop_End_X = max(pResultBlob->ptRT[nBlobCnt].x, pResultBlob->ptRB[nBlobCnt].x);
	pResultDefectInfo->Pixel_Crop_End_Y = max(pResultBlob->ptLB[nBlobCnt].y, pResultBlob->ptRB[nBlobCnt].y);

	//BOE Eng'r请求Defect Size
//pResultDefectInfo->Defect_Size			= (int)(pResultBlob->nArea[nBlobCnt] * WrtResultInfo.GetCamResolution(nCameraNum));
	pResultDefectInfo->Defect_Size = (int)WrtResultInfo.CalcDistancePixelToUm(sqrt(pow((double)(pResultDefectInfo->Pixel_End_X - pResultDefectInfo->Pixel_Start_X), 2) +
		pow((double)(pResultDefectInfo->Pixel_End_Y - pResultDefectInfo->Pixel_Start_Y), 2)),
		nCameraNum, nRatio);
	pResultDefectInfo->Defect_BKGV = pResultBlob->dBackGroundGV[nBlobCnt];
	pResultDefectInfo->Defect_MeanGV = pResultBlob->dMeanGV[nBlobCnt];
	pResultDefectInfo->Defect_MinGV = pResultBlob->nMinGV[nBlobCnt];
	pResultDefectInfo->Defect_MaxGV = pResultBlob->nMaxGV[nBlobCnt];
	pResultDefectInfo->Defect_Size_Pixel = (int)pResultBlob->nArea[nBlobCnt];
	pResultDefectInfo->Img_Number = nImageNum;
	pResultDefectInfo->Img_Size_X = (DOUBLE)(pResultDefectInfo->Pixel_Crop_End_X - pResultDefectInfo->Pixel_Crop_Start_X);		// 临时不良画面宽度大小
	pResultDefectInfo->Img_Size_Y = (DOUBLE)(pResultDefectInfo->Pixel_Crop_End_Y - pResultDefectInfo->Pixel_Crop_Start_Y);		// 临时不良影像垂直大小
	pResultDefectInfo->Defect_Type = pResultBlob->nDefectJudge[nBlobCnt];
	pResultDefectInfo->Pattern_Type = pResultBlob->nPatternClassify[nBlobCnt];

	// AI_ReJudge xb 2023/08/03
	pResultDefectInfo->nReJudgeCode = pResultBlob->AI_ReJudge_Code[nBlobCnt];
	pResultDefectInfo->dReJudgeConf = pResultBlob->AI_ReJudge_Conf[nBlobCnt];
	pResultDefectInfo->nReJudgeResult = pResultBlob->AI_ReJudge_Result[nBlobCnt];
	pResultDefectInfo->bReJudge = pResultBlob->AI_ReJudge[nBlobCnt];

	pResultDefectInfo->Camera_No = nCameraNum;
	pResultDefectInfo->nRatio = nRatio;

	//17.12.04-(长轴+缩短)/2->添加规格(客户要求)
	pResultDefectInfo->dDimension = (pResultBlob->dF_Max[nBlobCnt] + pResultBlob->dF_Min[nBlobCnt]) / 2.0;

#if USE_ALG_CONTOURS
	//17.11.29-外围信息(AVI&SVI其他工具)
	memcpy(pResultDefectInfo->nContoursX, pResultBlob->nContoursX[nBlobCnt], sizeof(int) * MAX_CONTOURS);
	memcpy(pResultDefectInfo->nContoursY, pResultBlob->nContoursY[nBlobCnt], sizeof(int) * MAX_CONTOURS);
#endif

	return true;
}

bool AviInspection::AdjustAlignInfo(tAlignInfo* pStCamAlignInfo, cv::Point* ptAdjCorner)
{
	//旋转时校正Left-Top基点坐标
	Align_DoRotatePoint(pStCamAlignInfo->ptCorner[E_CORNER_LEFT_TOP], ptAdjCorner[E_CORNER_LEFT_TOP], pStCamAlignInfo->ptAlignCenter, pStCamAlignInfo->dAlignTheta);
	Align_DoRotatePoint(pStCamAlignInfo->ptCorner[E_CORNER_RIGHT_TOP], ptAdjCorner[E_CORNER_RIGHT_TOP], pStCamAlignInfo->ptAlignCenter, pStCamAlignInfo->dAlignTheta);
	Align_DoRotatePoint(pStCamAlignInfo->ptCorner[E_CORNER_RIGHT_BOTTOM], ptAdjCorner[E_CORNER_RIGHT_BOTTOM], pStCamAlignInfo->ptAlignCenter, pStCamAlignInfo->dAlignTheta);
	Align_DoRotatePoint(pStCamAlignInfo->ptCorner[E_CORNER_LEFT_BOTTOM], ptAdjCorner[E_CORNER_LEFT_BOTTOM], pStCamAlignInfo->ptAlignCenter, pStCamAlignInfo->dAlignTheta);

	return true;
}

bool AviInspection::AdjustOriginImage(cv::Mat& MatOriginImage, cv::Mat& MatDrawImage, tAlignInfo* pStAlignInfo)
{
	cv::Point* ptCorner = pStAlignInfo->ptCorner;
	cv::Point* ptContCorner = pStAlignInfo->ptContCorner;

	cv::Point2f ptSrc[] = { cv::Point2f((float)ptContCorner[E_CORNER_LEFT_TOP].x,(float)ptContCorner[E_CORNER_LEFT_TOP].y),
							cv::Point2f((float)ptContCorner[E_CORNER_RIGHT_TOP].x,(float)ptContCorner[E_CORNER_RIGHT_TOP].y),
							cv::Point2f((float)ptContCorner[E_CORNER_RIGHT_BOTTOM].x,(float)ptContCorner[E_CORNER_RIGHT_BOTTOM].y),
							cv::Point2f((float)ptContCorner[E_CORNER_LEFT_BOTTOM].x,(float)ptContCorner[E_CORNER_LEFT_BOTTOM].y) };



	cv::Point2f ptDst[] = { cv::Point2f((float)ptCorner[E_CORNER_LEFT_TOP].x,(float)ptCorner[E_CORNER_LEFT_TOP].y),
							cv::Point2f((float)ptCorner[E_CORNER_RIGHT_TOP].x,(float)ptCorner[E_CORNER_RIGHT_TOP].y),
							cv::Point2f((float)ptCorner[E_CORNER_RIGHT_BOTTOM].x,(float)ptCorner[E_CORNER_RIGHT_BOTTOM].y),
							cv::Point2f((float)ptCorner[E_CORNER_LEFT_BOTTOM].x,(float)ptCorner[E_CORNER_LEFT_BOTTOM].y) };

	cv::Mat matWarp = cv::getPerspectiveTransform(ptSrc, ptDst);
	cv::warpPerspective(MatOriginImage, MatOriginImage, matWarp, MatOriginImage.size(), CV_INTER_AREA);
	cv::warpPerspective(MatDrawImage, MatDrawImage, matWarp, MatDrawImage.size(), CV_INTER_AREA);
	return true;
}

//////////////////////////////////////////////////////////////////////////
//在Align之前,检查AVI和SVI点亮异常
//////////////////////////////////////////////////////////////////////////
long AviInspection::CheckAD(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nImageNum, int nCameraNum, double* dResult, int nRatio)
{
	// test
	CCPUTimer tact;
	tact.Start();

	//获取单个算法检查参数
	double* dAlgPara = theApp.GetAlignParameter(nCameraNum);

	//设备类型
	int nEqpType = theApp.m_Config.GetEqpType();

	//如果与正常亮度不符,或者没有点亮,则AD不良,可以不检查。(无法Align)
	long nErrorCode = Align_FindDefectAD(MatOrgImage, dAlgPara, dResult, nRatio, nCameraNum, nEqpType);
	//long nErrorCode = E_ERROR_CODE_TRUE;

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Inspect AD : %.2f"), tact.Stop(false) / 1000.);

	return nErrorCode;
}

long AviInspection::CheckPGConnect(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nImageNum, int nCameraNum, double* dResult, cv::Point* cvPt)
{
	// test
	CCPUTimer tact;
	tact.Start();

	//获取单个算法检查参数
	double* dAlgPara = theApp.GetAlignParameter(nCameraNum);

	//设备类型
	int nEqpType = theApp.m_Config.GetEqpType();

	//如果与正常亮度不符,或者没有点亮,则AD不良,可以不检查。(无法Align)
	long nErrorCode = Align_FindDefectPGConnect(MatOrgImage, dAlgPara, dResult, cvPt, nCameraNum, nEqpType);
	//long nErrorCode = E_ERROR_CODE_TRUE;

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Inspect AD : %.2f"), tact.Stop(false) / 1000.);

	return nErrorCode;
}

//修复设备使用的坐标值和代码判定
bool AviInspection::JudgementRepair(const CString strPanelID, ResultPanelData& resultPanelData, CWriteResultInfo& WrtResultInfo)
{
	//检查Line是否贴在外壳上
	//获取Cell Corner
//[0] = LeftTop
	//[1]=RightTop			处理未写注释
//[2] = RightBottom
	//[3]=LeftBottom			处理未使用的注释

	int		nWorkOriginPosition = -1;			// 0 : LT, 1 : RT, 2 : RB, 3 : LB
	int		nWorkDirection = -1;				// 0 : X = Width, 1 : Y = Width

	WrtResultInfo.GetWorkCoordUsingRepair(nWorkOriginPosition, nWorkDirection);

	cv::Point ptCornerTemp[4];  // 用于保存Cell Corner的Align前
	cv::Point ptRotate(0, 0);   // Align结果的变量
	cv::Point ptAlignCorner[4]; // Align后Cell Corner

	ptAlignCorner[0].x = m_stThrdAlignInfo.rcAlignCellROI.x;
	ptAlignCorner[0].y = m_stThrdAlignInfo.rcAlignCellROI.y;

	ptAlignCorner[2].x = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
	ptAlignCorner[2].y = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

	int nDefectCount = (int)resultPanelData.m_ListDefectInfo.GetCount();

	//用于误差校正
	int nRepairOffSet = 20;

	//线路缺陷大于实际缺陷
	//需要修改算法
	//临时offset
	int nDefectRectOffSet = 6;

	//按顺序传阅for语句defect
	for (int i = 0; i < nDefectCount; i++)
	{
		int nDefect_Type = resultPanelData.m_ListDefectInfo[i].Defect_Type;
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_POINT_DARK &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_POINT_BRIGHT_DARK)	//如果是POINT
		{
			resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
			resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
			resultPanelData.m_ListDefectInfo[i].bUseReport = true;
		}

		//检查Line Defect repair条件
		//如果是Line Defect
		else if (resultPanelData.m_ListDefectInfo[i].Defect_Type >= E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type <= E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
		{
			int nStart_X = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
			int nStart_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
			int nEnd_X = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
			int nEnd_Y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;

			//如果超出Cell大小,则限制所有最大最小值
			if (nStart_X < ptAlignCorner[0].x)
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_Start_X = ptAlignCorner[0].x;
			}
			if (nStart_Y < ptAlignCorner[0].y)
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y = ptAlignCorner[0].y;
			}
			if (nEnd_X > ptAlignCorner[2].x)
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_End_X = ptAlignCorner[2].x;
			}
			if (nEnd_Y > ptAlignCorner[2].y)
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_End_Y = ptAlignCorner[2].y;
			}

			//如果不是DGS和Line Point
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_XLINE_SPOT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_YLINE_SPOT)
			{
				//X Line Defect为		
				if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN)
				{
					//如果只贴在一边,DO,L-GO,R-GO
						//如果是X Line,哪怕是一边也贴在Cell的末端
					if (ptAlignCorner[0].x - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_Start_X && resultPanelData.m_ListDefectInfo[i].Pixel_Start_X <= ptAlignCorner[0].x + nRepairOffSet ||
						ptAlignCorner[2].x - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_End_X && resultPanelData.m_ListDefectInfo[i].Pixel_End_X <= ptAlignCorner[2].x + nRepairOffSet)
					{
						// X-Line, Y-Line
											//如果是X Line,两边都贴在Cell的末端
						if (ptAlignCorner[0].x - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_Start_X && resultPanelData.m_ListDefectInfo[i].Pixel_Start_X <= (ptAlignCorner[0].x + nRepairOffSet) &&
							ptAlignCorner[2].x - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_End_X && resultPanelData.m_ListDefectInfo[i].Pixel_End_X <= ptAlignCorner[2].x + nRepairOffSet)
						{
							//工作原点更改时,需要报告的位置将发生变化
	// 
							switch (nWorkOriginPosition)
							{
							case E_CORNER_LEFT_TOP:
							case E_CORNER_LEFT_BOTTOM:
								//输入Defect左坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = ptAlignCorner[0].x; // 线的第一个x坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
								break;
							case E_CORNER_RIGHT_TOP:
							case E_CORNER_RIGHT_BOTTOM:
								//输入Defect右坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = ptAlignCorner[2].x; // 线端点x坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
								break;
							}
						}
						else
						{
							// DO, L-GO
													///X Line时,Line位于Cell左侧
														//在Left Corner X坐标内时
							if (ptAlignCorner[0].x - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_Start_X &&
								resultPanelData.m_ListDefectInfo[i].Pixel_Start_X <= ptAlignCorner[0].x + nRepairOffSet)
							{
								//输入Defect右坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = resultPanelData.m_ListDefectInfo[i].Pixel_End_X - nDefectRectOffSet; // 线端点x坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0; // y中间坐标
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
							}
							// DO, R-GO
													///X Line时,Line位于Cell右侧
													//在Right Corner X坐标中,OffSet内时
							if (ptAlignCorner[2].x - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_End_X &&
								resultPanelData.m_ListDefectInfo[i].Pixel_End_X <= ptAlignCorner[2].x + nRepairOffSet)
							{
								//输入Defect左坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + nDefectRectOffSet; // 线起始坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0; // y中间坐标
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
							}
						}
					}
					//如果Cell末端没有任何位置,则Defect中心坐标
					else
					{
						resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
						resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
						resultPanelData.m_ListDefectInfo[i].bUseReport = true;

					}
				}
				//如果是Y Line
				else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT ||
					resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT)
				{
					//如果只贴在一边,DO,L-GO,R-GO
						//Y Line,哪怕是一边贴在Cell的末端
					if (ptAlignCorner[0].y - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y && resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y <= ptAlignCorner[0].y + nRepairOffSet ||
						ptAlignCorner[2].y - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_End_Y && resultPanelData.m_ListDefectInfo[i].Pixel_End_Y <= ptAlignCorner[2].y + nRepairOffSet)
					{
						// Y-Line 
											//如果是Y Line,两边都贴在Cell的末端
						if (ptAlignCorner[0].y - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y && resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y <= ptAlignCorner[0].y + nRepairOffSet &&
							ptAlignCorner[2].y - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_End_Y && resultPanelData.m_ListDefectInfo[i].Pixel_End_Y <= ptAlignCorner[2].y + nRepairOffSet)
						{
							//工作原点更改时,需要报告的位置将发生变化
	// 
							switch (nWorkOriginPosition)
							{
							case E_CORNER_LEFT_TOP:
							case E_CORNER_RIGHT_TOP:
								//输入Defect的上坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = ptAlignCorner[0].y; // 线末端y坐标
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
								break;
							case E_CORNER_LEFT_BOTTOM:
							case E_CORNER_RIGHT_BOTTOM:
								//输入Defect的下坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;;
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = ptAlignCorner[2].y; // 线末端y坐标
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
								break;
							}
						}
						else
						{
							// DO, L-GO
													///Y Line,如果Line贴在Cell上方
							if (ptAlignCorner[0].y - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y &&
								resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y <= ptAlignCorner[0].y + nRepairOffSet)
							{
								//输入Defect的下坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0; // x中间坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - nDefectRectOffSet; // 线末端y坐标
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
							}
							// DO, R-GO
													///Y Line时,Line紧贴Cell底部
							if (ptAlignCorner[2].y - nRepairOffSet <= resultPanelData.m_ListDefectInfo[i].Pixel_End_Y &&
								resultPanelData.m_ListDefectInfo[i].Pixel_End_Y <= ptAlignCorner[2].y + nRepairOffSet)
							{
								//输入Cell的上标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0; // x中间坐标
								resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + nDefectRectOffSet; // 线起点Y坐标
								resultPanelData.m_ListDefectInfo[i].bUseReport = true;
							}
						}
					}
					else//如果Line的外角没有连接线端点
					{
						resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
						resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
						resultPanelData.m_ListDefectInfo[i].bUseReport = true;
					}
				}
				else//如果不是Line的话//确实没有可以进来的情况。
				{
					resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
					resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
					resultPanelData.m_ListDefectInfo[i].bUseReport = true;
				}
			}
			// LinePoint
			else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_XLINE_SPOT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_YLINE_SPOT)
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
				resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
				resultPanelData.m_ListDefectInfo[i].bUseReport = true;
			}
			//如果是DGS
			else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS)
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
				resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
				resultPanelData.m_ListDefectInfo[i].bUseReport = true;
			}

			//Sub判定
			else
			{
				resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
				resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
				resultPanelData.m_ListDefectInfo[i].bUseReport = true;
			}
		}//Line Defect Repair结束

		//MURA和其他不良
		else
		{
			resultPanelData.m_ListDefectInfo[i].Pixel_Repair_X = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_X + resultPanelData.m_ListDefectInfo[i].Pixel_End_X) / 2.0;
			resultPanelData.m_ListDefectInfo[i].Pixel_Repair_Y = (resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y + resultPanelData.m_ListDefectInfo[i].Pixel_End_Y) / 2.0;
			resultPanelData.m_ListDefectInfo[i].bUseReport = true;
		}
	}
	return true;
}

bool AviInspection::JudgementSpot(ResultPanelData& resultPanelData)
{
	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//周边比较
	int nOffSet = 4;

	//////////////////////////////////////////////////////////////////////////
		//18.03.30-G64判定为100分,重新确认后应用判定
	//////////////////////////////////////////////////////////////////////////

		//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//只有在名分&百分不良的情况下...
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT && //04.16 choi
			resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_POINT_BRIGHT)
			continue;

		//仅在G87模式下...百分检查
		int nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_87)
			continue;

		//不良中心坐标
		CPoint ptSrc1;
		ptSrc1.x = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio);
		ptSrc1.y = (LONG)(resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio);

		//无法比较
		for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//禁止比较同一不良项目
			if (i == j)
			{
				j++;
				continue;
			}

			//仅在G64模式下...
			nImgNum = resultPanelData.m_ListDefectInfo[j].Img_Number;
			if (theApp.GetImageClassify(nImgNum) != E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				j++;
				continue;
			}

			//只有在Mura检查不良的情况下...
			if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL &&
				resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG)
			{
				j++;
				continue;
			}

			//不良中心坐标
			CPoint ptSrc2;
			ptSrc2.x = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio);
			ptSrc2.y = (LONG)(resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio);

			//如果不良中心点相同的话...
			if (abs(ptSrc1.x - ptSrc2.x) < nOffSet &&
				abs(ptSrc1.y - ptSrc2.y) < nOffSet)
			{
				//如果是100分,则为100分
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL)
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				//删除明示点
				else
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL;

				break;
			}
			//如果不能重复的话...下一个不良...
			else j++;
		}
	}

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		//只有在Mura检查不良的情况下...
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_TYPE3_SMALL &&
			resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_MURA_TYPE3_BIG)
		{
			i++;
			continue;
		}

		//删除不良内容
		resultPanelData.m_ListDefectInfo.RemoveAt(i);
	}

	return true;
}

bool AviInspection::JudgementDelReport(ResultPanelData& resultPanelData)
{
	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		if (resultPanelData.m_ListDefectInfo[i].bUseReport)
		{
			i++;
			continue;
		}

		resultPanelData.m_ListDefectInfo.RemoveAt(i);
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	周四:删除同一Pattern内Line周围的Line Defect																//
//工作日:2017-09-18																																											//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementDeletLineDefect(ResultPanelData& resultPanelData, double* dAlignPara)
{
	int nLineNumber = 0;	// 真正的线条不良
	int nLineDefect = 0;	// 弱线
	int nOffSet = 30;	// 中心点距离差
	int nWidthofLine = 45;	// 如果弱线条中的线条比其粗细高,则删除

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在Line性能真的不好的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
			nLineNumber++;	// 真正的线条
			break;

			//如果Line不好,请更改
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
			nLineDefect++;	// 弱线
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//删除钢线周围的弱线
	//////////////////////////////////////////////////////////////////////////

	// Delete the line defect around the Strong Line
		//如果两者都存在
	if (nLineNumber > 0 && nLineDefect > 0)
	{
		//////////////////////////////////////////////////////////////////////////
				//x方向
		//////////////////////////////////////////////////////////////////////////

				//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向(钢线)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN)
			{
				CPoint	X1St, X1End, X1Center;
				int		X1DefectImageType;

				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					//排除X方向不不良(仅限弱线...)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
					{
						j++; continue;
					}

					CPoint	X2St, X2End, X2Center;
					int		X2DefectImageType;

					X2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//不是相同的模式就是不同的模式
					if (X1DefectImageType != X2DefectImageType) { j++; continue; }

					//如果钢丝绳周围有钢丝绳
					if (abs(X1Center.y - X2Center.y) <= nOffSet)
					{
						//删除小列表时...
						if (i > j)	 i--;

						//删除缩写
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					//如果钢线周围没有弱线
					else  j++;
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////
				//y方向
		//////////////////////////////////////////////////////////////////////////

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//Y方向(钢线)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT)
			{
				CPoint Y1St, Y1End, Y1Center;
				int Y1DefectImageType;
				Y1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1Center.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;

				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///Y方向不不良除外(仅限弱线...)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int		Y2DefectImageType;
					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;

					//不是相同的模式就是不同的模式
					if (Y1DefectImageType != Y2DefectImageType) { j++; continue; }

					//如果钢丝绳周围有钢丝绳
					if (abs(Y1Center.x - Y2Center.x) <= nOffSet)
					{
						//删除小列表时...
						if (i > j) i--;

						//删除缩写
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					//如果钢线周围没有弱线
					else  j++;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//B行周围的D行两侧删除功能(x方向)
		//D行周围的B行两侧的删除功能(x方向)
	//////////////////////////////////////////////////////////////////////////

		//需要y方向的实现

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//X方向
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK)
		{
			CPoint	X1St, X1End, X1Center;
			int		X1DefectImageType;
			X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
			X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
			X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
			X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				//避免类似的不良比较
				if (i == j)
				{
					j++; continue;
				}

				//不是X方向
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DARK)
				{
					j++; continue;
				}

				if ((resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK) ||	 					//i:亮的
					//j:暗
					(resultPanelData.m_ListDefectInfo[j].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&	 					//j:亮
						resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK))	 					//i:暗的
				{
					CPoint	X2St, X2End, X2Center;
					int		X2DefectImageType;
					X2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;

					//不是相同的模式就是不同的模式
					if (X1DefectImageType != X2DefectImageType) { j++; continue; }

					if ((abs(X2St.y - X2End.y) >= nWidthofLine) &&		//大于线厚度设置
						(abs(X1Center.y - X2Center.y) >= nOffSet))		//中心坐标相似时
					{
						//删除小列表时...
						if (i > j) i--;

						//删除行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					else j++;
				}
				else j++;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//强线&弱线重新计数
	//////////////////////////////////////////////////////////////////////////

		//初始化旧数据
	nLineNumber = 0;
	nLineDefect = 0;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在真的Line性不良的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
			nLineNumber++;
			break;

			//如果Line不好,请更改
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
		{
			//验证是否使用了Norch
			int NORCH_ONOFF = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_ONOFF]);

			if ((resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT) &&
				NORCH_ONOFF == 1)
			{
				//获取设置的Norch编号值
				int NORCH_UNIT = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_UNIT_X]);

				if (NORCH_UNIT == NULL) NORCH_UNIT = 400;

				if (abs(resultPanelData.m_ListDefectInfo[i].Pixel_Center_X - m_stThrdAlignInfo.rcAlignCellROI.x) < NORCH_UNIT) continue;
				else nLineDefect++;
			}

			else
				nLineDefect++;
		}
		break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//如果有15个或更多的线不良:Retest判定
	//////////////////////////////////////////////////////////////////////////

		//约15行以上
	if (nLineDefect > 15)
	{
		int nBuff = 0;

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//约线时
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
			{
				//约行数
				nBuff++;

				//第一行缩写:修改Retest判定
				if (nBuff == 1) { resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_RETEST_LINE_BRIGHT; }

				//删除其余的缩写
				if (nBuff > 1) { resultPanelData.m_ListDefectInfo.RemoveAt(i); }
			}
			else i++;
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:同一Pattern内同一方向的不良行为Merge															//
//工作日:2017-11-14																																										//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementSamePatternDefectMerge(ResultPanelData& resultPanelData)
{
	int nLineNumber = 0;	// 真正的线条不良
	int nSSLineNumber = 0;	// 相同Pattern,在同一方向上无效
	int nOffSet = 10;	// 中心点距离差

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在真的Line性不良的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
			nLineNumber++;	// 真正的线条
			break;
		}
	}

	//在同一方向上有很多不良行为时
	if (nLineNumber > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK)
			{
				CPoint	X1Center, X1Length;
				int		X1DefectImageType, X1DefectLength;

				X1Length.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;	// 缺陷起点
				X1Length.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X;	// 缺陷端点
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y; // 中心位置
				X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式
				X1DefectLength = abs(X1Length.x - X1Length.y);								// 缺陷长度

				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_OPEN &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DARK)
					{
						j++; continue;
					}

					CPoint	X2Center, X2Length;
					int		X2DefectImageType, X2DefectLength;

					X2Length.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X;	// 缺陷起点
					X2Length.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X;	// 缺陷端点
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y; // 中心位置
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式
					X2DefectLength = abs(X2Length.x - X2Length.y);								// 缺陷长度

					//相同模式的缺陷
					if (X1DefectImageType != X2DefectImageType) { j++; continue; }

					//相同的方向
					if (abs(X1Center.y - X2Center.y) <= nOffSet)
					{

						//删除小列表时
// 							if( i < j )	 j--;

						//删除其他模式的行

						resultPanelData.m_ListDefectInfo[i].Pixel_Start_X = min(X1Length.x, X2Length.x);
						resultPanelData.m_ListDefectInfo[i].Pixel_End_X = max(X1Length.y, X2Length.y);

						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式的行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
						//}
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:以Black Pattern为基准,周围的Bright Line Merge																	//
//工作日:2017-10-21																																									//
//17.10.30协议结果:根据Black Pattern宽度删除Dark Line											//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementBlackPatternMerge(ResultPanelData& resultPanelData)
{
	int nXLineNumber = 0;	// 真正的线条不良
	int nYLineNumber = 0;	// 真正的线条不良
	int nOffSet = 20;	// 中心点距离差

	//计算不良数量的部分
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在X Line真的不好的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_DGS_X:
		case E_DEFECT_JUDGEMENT_LINE_PCD_CRACK:
			nXLineNumber++;	// 真正的线条
			break;

			//只有在Y Line真的不好的情况下才会计数
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
		case E_DEFECT_JUDGEMENT_LINE_DGS_Y:
			nYLineNumber++;	// 真正的线条
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//x方向
	//////////////////////////////////////////////////////////////////////////

	if (nXLineNumber > 0)//如果有X Line,就必须工作
	{
		//比较
		int	X1DefectPattern = 0;
		int X1FistPattern = 0; // 最先出来的Pattern数~
		int X1InitialPattern = 20;

		//明线不良Pattern优先级Black>Red>Green>Blue
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//基于X方向明线最先出现的Pattern的Pattern
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT)
			{
				CPoint	X1St, X1End, X1Center;

				X1DefectPattern = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式
				X1FistPattern = min(X1InitialPattern, X1DefectPattern);
				X1InitialPattern = X1FistPattern;
			}
		}

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT)
			{
				CPoint	X1St, X1End, X1Center;
				int		X1DefectImageType;

				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//应该是比较后得到的Pattern。呵
				if (X1DefectImageType != X1FistPattern) continue;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_OPEN &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
					{
						j++; continue;
					}

					CPoint	X2St, X2End, X2Center;
					int		X2DefectImageType;

					X2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//不能与标准Pattern相同,其他Pattern的缺陷
	//if ( X1DefectImageType == X2DefectImageType ){ j++; continue; }

					//如果在周围检测到Line
	//if ( (X1St.y <= X2Center.y) && (X1End.y >= X2Center.y) )
					if (abs(X1Center.y - X2Center.y) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}

		//DGS
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_X)
			{
				CPoint	X1St, X1End, X1Center;
				int		X1DefectImageType;

				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_OPEN &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS_X)
					{
						j++; continue;
					}

					CPoint	X2St, X2End, X2Center;
					int		X2DefectImageType;

					X2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//if (X1DefectImageType == X2DefectImageType) j++; continue;

									//如果在周围检测到Line
					if (abs(X1Center.y - X2Center.y) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}

		//删除PCD Crack部分
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_PCD_CRACK)
			{
				CPoint	X1St, X1End, X1Center;
				int		X1DefectImageType;

				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
	//if (resultPanelData.m_ListDefectInfo[j].Defect_Type != 	E_DEFECT_JUDGEMENT_LINE_PCD_CRACK)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_OPEN &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS_X &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_PCD_CRACK)
					{
						j++; continue;
					}

					CPoint	X2St, X2End, X2Center;
					int		X2DefectImageType;

					X2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//不能与标准Pattern相同,其他Pattern的缺陷
					if (X1DefectImageType == X2DefectImageType) { j++; continue; }

					//如果在周围检测到Line
					if (abs(X1Center.y - X2Center.y) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//y方向
	//////////////////////////////////////////////////////////////////////////

	if (nYLineNumber > 0)//如果有Y Line,就必须工作
	{
		//比较
		int	Y1DefectPattern = 0;
		int Y1FistPattern = 0; // 最先出来的Pattern数~
		int Y1InitialPattern = 20;

		//明线不良Pattern优先级Black/Red/Green/Blue
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//基于最先出现Y方向明线的Pattern的Pattern
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT)
			{
				CPoint	Y1St, Y1End, Y1Center;

				Y1DefectPattern = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式
				Y1FistPattern = min(Y1InitialPattern, Y1DefectPattern);
				Y1InitialPattern = Y1FistPattern;
			}
		}

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT)
			{
				CPoint	Y1St, Y1End, Y1Center;
				int		Y1DefectImageType;

				Y1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1Center.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//应该是比较后得到的Pattern。呵
				if (Y1DefectImageType != Y1FistPattern) continue;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int		Y2DefectImageType;

					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//不能与标准Pattern相同,其他Pattern的缺陷
	//if ( Y1DefectImageType == Y2DefectImageType ){ j++; continue; }

					//如果在周围检测到Line
	//if ( (X1St.y <= X2Center.y) && (X1End.y >= X2Center.y) )
					if (abs(Y1Center.x - Y2Center.x) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}

		//DGS Y Line
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_Y)
			{
				CPoint	Y1St, Y1End, Y1Center;
				int		Y1DefectImageType;

				Y1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1Center.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS_Y)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int		Y2DefectImageType;

					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//如果在周围检测到Line
					if (abs(Y1Center.x - Y2Center.x) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////
//标题:消除因Camera Tap导致的约Line不良过检
//工作日:2018-07-31
//操作员:PNZ
//////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementCameraTapOverKill(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{
	//如果没有异常参数
	if (dAlignPara == NULL)
		return false;

	int		CTO_OnOff = (int)(dAlignPara[E_PARA_CAMERA_TAP_OVERKILL_ONOFF]);
	int		CTO_CameraType = (int)(dAlignPara[E_PARA_CAMERA_TAP_OVERKILL_CAMERATYPE]);
	int		CTO_Offset = (int)(dAlignPara[E_PARA_CAMERA_TAP_OVERKILL_OFFSET]);

	// On/Off
	if (CTO_OnOff == 0)
		return false;

	//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	//初始化
	int	nLINE_X_DEFECT_Number = 0;
	int	nLINE_Y_DEFECT_Number = 0;

	//LINE Y Defect不良Count
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		//X方向LINE DEFECT
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK)
		{
			//LINE坏Count
			nLINE_X_DEFECT_Number++;
		}

		//Y方向LINE DEFECT
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK)
		{
			//LINE坏Count
			nLINE_Y_DEFECT_Number++;
		}
	}

	//如果LINE Y Defect不存在,则退出
	if (nLINE_Y_DEFECT_Number <= 0 && nLINE_X_DEFECT_Number <= 0) return true;

	//////////////////////////////////////////////////////////////////////////
		//获取Camera Type Info
	vector<int> Direction;
	vector<int> Position;
	vector<double> Bright_Diff_GV;
	vector<double> Dark_Diff_GV;

	bool CameraTapLoad = CameraTapInfo(CTO_CameraType, Direction, Position, Bright_Diff_GV, Dark_Diff_GV);
	if (!CameraTapLoad) return false;

	if (CTO_CameraType == 0 && Direction.size() > 4) return false; // 29M
	if (CTO_CameraType == 1 && Direction.size() < 4) return false; // 47M

	//////////////////////////////////////////////////////////////////////////
		//检查LINE Y DEFECT
	if (nLINE_Y_DEFECT_Number > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//仅在LINE出现故障时
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DARK)
			{
				i++;
				continue;
			}

			//比较坐标
			int nDefectLocation = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;

			//参数初始化
			double DefectMeanGV = 0;
			double LeftMeanGV = 0;
			double RightMeanGV = 0;

			//Camera Type故障诊断
			bool CameraTypeOverKill = false;

			//比较坐标
			for (int m = 0; m < Position.size(); m++)
			{
				//Y方向或下一个
				if (Direction[m] != 2) continue;

				//获取坐标
				int CTO_Position_Y = Position[m];

				//检查是否在Tap边界附近
				if (abs(nDefectLocation - CTO_Position_Y) <= CTO_Offset)
				{
					//检查具体值模式
					int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
					int	nImgNum1 = theApp.GetImageClassify(nImgNum);

					//前往相应的Pattern进行确认
					cv::Mat DefectImage = MatOrgImage[theApp.GetImageNum(nImgNum1)][0];

					// Defect ROI
					cv::Rect DefectRect;

					DefectRect.x = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					DefectRect.width = resultPanelData.m_ListDefectInfo[i].Pixel_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.height = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;

					double MaxGV;
					DefectLDRCompair(DefectImage, DefectRect, LeftMeanGV, DefectMeanGV, RightMeanGV, MaxGV);

					//禁用Memory
					DefectImage.release();

					//判定
					if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT)
					{
						if (DefectMeanGV > RightMeanGV && (DefectMeanGV - RightMeanGV) < Bright_Diff_GV[m] && (DefectMeanGV - RightMeanGV) > 0)
							CameraTypeOverKill = true;

						else if (LeftMeanGV <= DefectMeanGV && DefectMeanGV <= RightMeanGV)
							CameraTypeOverKill = true;

						else
							CameraTypeOverKill = false;
					}

					else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
						resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK)
					{
						if (DefectMeanGV < LeftMeanGV && (DefectMeanGV - LeftMeanGV) > Dark_Diff_GV[m] && (DefectMeanGV - LeftMeanGV) <= 0)
							CameraTypeOverKill = true;

						else if (LeftMeanGV <= DefectMeanGV && DefectMeanGV <= RightMeanGV)
							CameraTypeOverKill = true;

						else
							CameraTypeOverKill = false;
					}

					else	CameraTypeOverKill = false;
				}
			}

			if (CameraTypeOverKill)
			{
				//如果是Dimple,则删除
				resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}
			else { i++; continue; }
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//验证LINE X DEFECT
	if (nLINE_X_DEFECT_Number > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); )
		{
			//仅在LINE出现故障时
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK &&
				resultPanelData.m_ListDefectInfo[i].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DARK)
			{
				i++;
				continue;
			}

			//比较坐标
			int nDefectLocation = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

			//参数初始化
			double DefectMeanGV = 0;
			double LeftMeanGV = 0;
			double RightMeanGV = 0;

			//Camera Type故障诊断
			bool CameraTypeOverKill = false;

			//比较坐标
			for (int m = 0; m < Position.size(); m++)
			{
				//Y方向或下一个
				if (Direction[m] != 1) continue;

				//获取坐标
				int CTO_Position_X = Position[m];

				//检查是否在Tap边界附近
				if (abs(nDefectLocation - CTO_Position_X) <= CTO_Offset)
				{
					//检查具体值模式
					int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
					int	nImgNum1 = theApp.GetImageClassify(nImgNum);

					//前往相应的Pattern进行确认
					cv::Mat DefectImage = MatOrgImage[theApp.GetImageNum(nImgNum1)][0];

					// Defect ROI
					cv::Rect DefectRect;

					DefectRect.x = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					DefectRect.width = resultPanelData.m_ListDefectInfo[i].Pixel_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.height = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;

					if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK && DefectRect.height <= 2)
					{
						CameraTypeOverKill = true;
						break;
					}

					double MaxGV;
					DefectLDRCompair(DefectImage, DefectRect, LeftMeanGV, DefectMeanGV, RightMeanGV, MaxGV);

					//禁用Memory
					DefectImage.release();

					//判定
					if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT)
					{
						if (DefectMeanGV > RightMeanGV && (DefectMeanGV - RightMeanGV) < Bright_Diff_GV[m] && (DefectMeanGV - RightMeanGV) > 0)
							CameraTypeOverKill = true;

						else if (LeftMeanGV <= DefectMeanGV && DefectMeanGV <= RightMeanGV)
							CameraTypeOverKill = true;

						else
							CameraTypeOverKill = false;
					}

					else if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
						resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK)
					{
						if (DefectMeanGV < LeftMeanGV && (DefectMeanGV - LeftMeanGV) > Dark_Diff_GV[m] && (DefectMeanGV - LeftMeanGV) <= 0)
							CameraTypeOverKill = true;

						else if (LeftMeanGV <= DefectMeanGV && DefectMeanGV <= RightMeanGV)
							CameraTypeOverKill = true;

						else
							CameraTypeOverKill = false;
					}

					else	CameraTypeOverKill = false;
				}
			}

			if (CameraTypeOverKill)
			{
				//如果是Dimple,则删除
				resultPanelData.m_ListDefectInfo.RemoveAt(i);
			}
			else { i++; continue; }
		}
	}

	return true;
}

bool AviInspection::DefectLDRCompair(cv::Mat matSrcImage, cv::Rect rectTemp, double& Left_MeanValue, double& Defect_MeanValue, double& Right_MeanValue, double& Defect_MaxGV)
{
	//初始化
	cv::Scalar Left_Mean, Left_StdDev, Right_Mean, Right_StdDev, Defect_Mean, Defect_StdDev;
	cv::Rect rectTemp_Left, rectTemp_Right;
	int XShiftUnit = 0;
	int YShiftUnit = 0;
	int CutUnit_X_1 = 600;
	int CutUnit_X_2 = 150;
	int CutUnit_Y_1 = 150;
	int CutUnit_Y_2 = 150;
	int nOffset = 50;

	//Y判定
	if (rectTemp.width / rectTemp.height < 1)
	{
		//异常处理
		if (rectTemp.height <= CutUnit_Y_1 + CutUnit_Y_2 + nOffset) return false;

		rectTemp.y = rectTemp.y + CutUnit_Y_1;
		rectTemp.height = rectTemp.height - CutUnit_Y_1 - CutUnit_Y_2;

		XShiftUnit = 40;
		YShiftUnit = 0;
		rectTemp_Left.width = 20;
		rectTemp_Left.height = rectTemp.height;
		rectTemp_Right.width = 20;
		rectTemp_Right.height = rectTemp.height;

		//增加不良宽度很小的不良幅度
		if (rectTemp.width <= 4)
		{
			rectTemp.width = rectTemp.width + 10;
			rectTemp.x = rectTemp.x - 5;
		}
	}

	// X Defect
	else if (rectTemp.width / rectTemp.height >= 1)
	{
		//异常处理
		if (rectTemp.width <= CutUnit_X_1 + CutUnit_X_2 + nOffset) return false;

		rectTemp.x = rectTemp.x + CutUnit_X_1;
		rectTemp.width = rectTemp.width - CutUnit_X_1 - CutUnit_X_2;

		XShiftUnit = 0;
		YShiftUnit = 40;
		rectTemp_Left.width = rectTemp.width;
		rectTemp_Left.height = 20;
		rectTemp_Right.width = rectTemp.width;
		rectTemp_Right.height = 20;
	}

	//////////////////////////////////////////////////////////////////////////

		// Left ROI
	rectTemp_Left.x = rectTemp.x - XShiftUnit;
	rectTemp_Left.y = rectTemp.y - YShiftUnit;

	// Right ROI
	rectTemp_Right.x = rectTemp.x + XShiftUnit;
	rectTemp_Right.y = rectTemp.y + YShiftUnit;

	// GaussianBlur
	cv::resize(matSrcImage, matSrcImage, matSrcImage.size() / 2, INTER_LINEAR);
	cv::GaussianBlur(matSrcImage, matSrcImage, cv::Size(5, 5), 3.0);
	cv::resize(matSrcImage, matSrcImage, matSrcImage.size() * 2, INTER_LINEAR);

	//区域图像
	cv::Mat LeftImage = matSrcImage(rectTemp_Left);
	cv::Mat RightImage = matSrcImage(rectTemp_Right);
	cv::Mat DefectImage = matSrcImage(rectTemp);

	cv::meanStdDev(LeftImage, Left_Mean, Left_StdDev);
	cv::meanStdDev(RightImage, Right_Mean, Right_StdDev);
	cv::meanStdDev(DefectImage, Defect_Mean, Defect_StdDev);

	double minvalue = 0;

	cv::minMaxIdx(DefectImage, &minvalue, &Defect_MaxGV, NULL, NULL);

	Left_MeanValue = Left_Mean[0];
	Right_MeanValue = Right_Mean[0];
	Defect_MeanValue = Defect_Mean[0];

	return true;
}

bool 	AviInspection::CameraTapInfo(int CameraType, vector<int>& Direction, vector<int>& Position, vector<double>& BrightDiffGV, vector<double>& DarkDiffGV)
{
	CString CameraTypeFile;
	//更改CameraTap Path hjf
	CameraTypeFile.Format(_T("%s\\CCD\\CameraTap.Index"), theApp.m_Config.GETCmdDRV());

	//需要确认文件是否存在
	CFileFind find;
	BOOL bFindFile = FALSE;

	bFindFile = find.FindFile(CameraTypeFile);
	find.Close();

	//如果没有文件
	if (!bFindFile) return false;

	char szFileName[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, CameraTypeFile, -1, szFileName, sizeof(szFileName), NULL, NULL);

	FILE* out = NULL;
	fopen_s(&out, szFileName, "r");

	//?
	if (!out)	return false;

	vector<int> vDirection;
	vector<int> vPosition;
	vector<double> vGV;

	int nDirectionBuf;
	int nPositionBuf;
	double dbBrightDiffGVBuf;
	double dbDarktDiffGVBuf;

	//////////////////////////////////////////////////////////////////////////
		//选择Camera Type
	int X_Index;
	int Y_Index;
	int CalBuff;

	if (CameraType == 0) { X_Index = 3288;		Y_Index = 4384; } // 29M Camera 8  Tap / PS Image : 13152 x 8768
	else if (CameraType == 1) { X_Index = 2214;		Y_Index = 5280; } // 47M Camera 16 Tap / PS Image : 17712 x 10560
	else
		return false;

	for (int m = 0; ; m++)
	{
		//读取
		fscanf_s(out, "%d,%d,%lf,%lf\n", &nDirectionBuf, &nPositionBuf, &dbBrightDiffGVBuf, &dbDarktDiffGVBuf);

		//计算位置
		if (nDirectionBuf == 1) { CalBuff = nPositionBuf * Y_Index; }//X方向
		if (nDirectionBuf == 2) { CalBuff = nPositionBuf * X_Index; }//Y方向

		//与之前的Data类似,Break
		if (m != 0 && nDirectionBuf == Direction[m - 1] && CalBuff == Position[m - 1] &&
			dbBrightDiffGVBuf == BrightDiffGV[m - 1] && dbDarktDiffGVBuf == DarkDiffGV[m - 1])
			break;

		//保存数据
		Direction.push_back(nDirectionBuf);
		Position.push_back(CalBuff);
		BrightDiffGV.push_back(dbBrightDiffGVBuf);
		DarkDiffGV.push_back(dbDarktDiffGVBuf);

		if (CameraType == 0 && Direction.size() == 4) break;
	}

	fclose(out);
	out = NULL;

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:PCD CRACK判定																																				//
//工作日:2017-12-01																																									//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementPCDCrack(ResultPanelData& resultPanelData, double* dAlignPara)
{
	//如果没有值
	if (dAlignPara == NULL)	return false;

	//如果禁用
	if (dAlignPara[E_PARA_PCD_CRACK_FLAG] <= 0)	return true;

	// Start Position Y
	cv::Point ptAlignCorner[4]; // Align后Cell Corner

	ptAlignCorner[0].x = m_stThrdAlignInfo.rcAlignCellROI.x;
	ptAlignCorner[0].y = m_stThrdAlignInfo.rcAlignCellROI.y;

	//在Black,Red,Blue Pattern中
	//固定位置X1=740~820,X2=2320~2400
	int nRange1Start = (740 * 2) + ptAlignCorner[0].y;
	int nRange1End = (820 * 2) + ptAlignCorner[0].y;
	int nRange2Start = (2320 * 2) + ptAlignCorner[0].y;
	int nRange2End = (2400 * 2) + ptAlignCorner[0].y;

	int nXLineNumber = 0;	// 真正的线条不良
	int nOffSet = 20;	// 中心点距离差

	//Merge后剩余的X Line Bright不良
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在X Line真的不好的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
			nXLineNumber++;	// 真正的线条
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//x方向只显示Line
	//////////////////////////////////////////////////////////////////////////

	if (nXLineNumber > 0)//如果有X Line,就必须工作
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT)
			{
				CPoint	X1Center;

				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				//AfxMessageBox(_T("Center Distance: '%d'"), X1Center.y);

							//基准坐标会像Ratio一样变化。
				int nSubRange1_Start = nRange1Start / resultPanelData.m_ListDefectInfo[i].nRatio;
				int nSubRange1_End = nRange1End / resultPanelData.m_ListDefectInfo[i].nRatio;
				int nSubRange2_Start = nRange2Start / resultPanelData.m_ListDefectInfo[i].nRatio;
				int nSubRange2_End = nRange2End / resultPanelData.m_ListDefectInfo[i].nRatio;

				int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
				int	nImgNum1 = theApp.GetImageClassify(nImgNum);

				switch (nImgNum1)
				{
				case E_IMAGE_CLASSIFY_AVI_BLACK:
				case E_IMAGE_CLASSIFY_AVI_R:
				case E_IMAGE_CLASSIFY_AVI_B:
				{
					if ((nSubRange1_Start - nOffSet < X1Center.y) && (X1Center.y < nSubRange1_End + nOffSet))
					{
						resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_PCD_CRACK;
					}

					else if ((nSubRange2_Start - nOffSet < X1Center.y) && (X1Center.y < nSubRange2_End + nOffSet))
					{
						resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_PCD_CRACK;
					}

					else continue;
				}

				break;
				}
			}
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:DGS判定新标准ver0.3																																//
//工作日:2017-11-03,2017-11-15修改																														//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementNewDGS(ResultPanelData& resultPanelData)
{
	int nXLineNumber = 0;	// X线条不良
	int nYLineNumber = 0;	// Y线条不良
	int nOffSet = 20;	// 中心点距离差
	int nCenterOffSet = 30;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在真的Line性不良的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_VTH_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_VTH_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_DARK:
			nXLineNumber++;	// 真正的线条
			break;

			//只有在真的Line性不良的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_VTH_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_VTH_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_DARK:
			nYLineNumber++;	// 真正的线条
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//如果在同一模式下交叉,则判定为DGS
	//////////////////////////////////////////////////////////////////////////

	int nCountDGS = 0;

	if ((nXLineNumber > 0) && (nYLineNumber > 0))
	{
		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_VTH_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_VTH_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_VTH_DGS_X)
			{
				CPoint	X1St, X1End;
				int X1DefectPattern;

				X1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectPattern = resultPanelData.m_ListDefectInfo[i].Pattern_Type;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); j++)
				{
					//避免类似的不良比较
					if (i == j) { continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_VTH_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_VTH_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_DARK &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS_Y)
					{
						continue;
					}

					CPoint	Y2St, Y2End;
					int Y2DefectPattern;

					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectPattern = resultPanelData.m_ListDefectInfo[j].Pattern_Type;

					if (X1DefectPattern != Y2DefectPattern) { continue; }//非同一模式不行,这是DGS...

					//如果在周围检测到Line
					if ((X1St.x <= Y2St.x) && (X1End.x >= Y2End.x))
					{
						if ((X1St.y >= Y2St.y) && (X1End.y <= Y2End.y))
						{

							resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_DGS_X;
							resultPanelData.m_ListDefectInfo[j].Defect_Type = E_DEFECT_JUDGEMENT_LINE_DGS_Y;

							ResultDefectInfo SubDefect = resultPanelData.m_ListDefectInfo[i];

							SubDefect.Pixel_Start_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
							SubDefect.Pixel_End_Y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;
							SubDefect.Pixel_Start_X = resultPanelData.m_ListDefectInfo[j].Pixel_Start_X;
							SubDefect.Pixel_End_X = resultPanelData.m_ListDefectInfo[j].Pixel_End_X;

							SubDefect.Defect_Type = E_DEFECT_JUDGEMENT_LINE_DGS;

							resultPanelData.m_ListDefectInfo.Add(SubDefect);

							nCountDGS++;

						}
						else continue;
					}
					else continue;
				}
			}
		}
	}

	//消除DGS_X,DGS_Y周围线路故障
	if (nCountDGS > 0)
	{
		//DGS重复数据删除
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS)
			{
				CPoint	X1St, X1End, X1Center;
				int		X1DefectPattern;
				X1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				//X1Center.x		= (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X	/ resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.x = (long)((X1St.x + X1End.y) / 2.0);
				//X1Center.y		= (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y	/ resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)((X1St.y + X1End.y) / 2.0);
				X1DefectPattern = resultPanelData.m_ListDefectInfo[i].Pattern_Type;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int		Y2DefectPattern;

					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;

					//Y2Center.x		= (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X	/ resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.x = (long)((Y2St.x + Y2End.x) / 2.0);
					//Y2Center.y		= (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y	/ resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.y = (long)((Y2St.y + Y2End.y) / 2.0);
					Y2DefectPattern = resultPanelData.m_ListDefectInfo[j].Pattern_Type;

					//如果在周围检测到Line
					if (abs(X1Center.y - Y2Center.y) <= nCenterOffSet)
					{
						if (abs(X1Center.x - Y2Center.x) <= nCenterOffSet)
						{
							//删除小列表时
							if (i > j)	 i--;

							//删除其他模式的行
							resultPanelData.m_ListDefectInfo.RemoveAt(j);
						}
						else  j++;
					}
					else  j++;
				}
			}

			//仅删除X方向DGS X和重复DGS X
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_X)
			{
				CPoint	X1St, X1End, X1Center;
				int X1DefectPattern;

				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectPattern = resultPanelData.m_ListDefectInfo[i].Pattern_Type;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS_X)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int Y2DefectPattern;

					Y2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectPattern = resultPanelData.m_ListDefectInfo[j].Pattern_Type;

					//如果在周围检测到Line
					if (abs(X1Center.y - Y2Center.y) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式的行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					//如果周围没有缺陷
					else  j++;
				}
			}

			//仅删除Y方向DGS Y和重复DGS Y
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_Y)
			{
				CPoint	Y1St, Y1End, Y1Center;
				int Y1DefectPattern;

				Y1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1Center.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1DefectPattern = resultPanelData.m_ListDefectInfo[i].Pattern_Type;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount();)
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///Y方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_DGS_Y)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int Y2DefectPattern;

					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectPattern = resultPanelData.m_ListDefectInfo[j].Pattern_Type;

					//如果在周围检测到Line
					if (abs(Y1Center.x - Y2Center.x) <= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式的行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
					//如果周围没有缺陷
					else  j++;
				}
			}
		}
	}
	return true;
}
// YDS 18.05.25

bool AviInspection::JudgementDGS_Vth(ResultPanelData& resultPanelData)
{
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_X ||
			resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_Y)
		{
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_VTH)
			{
				switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
				{
				case E_DEFECT_JUDGEMENT_LINE_DGS:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_VTH_DGS;
					break;
				case E_DEFECT_JUDGEMENT_LINE_DGS_X:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_VTH_DGS_X;
					break;
				case E_DEFECT_JUDGEMENT_LINE_DGS_Y:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_VTH_DGS_Y;
					break;
				case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_X_VTH_BRIGHT;
					break;
				case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_VTH_BRIGHT;
					break;
				case E_DEFECT_JUDGEMENT_LINE_X_DARK:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_X_VTH_DARK;
					break;
				case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_VTH_DARK;
					break;
				case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_BRIGHT;
					break;
				case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_BRIGHT;
					break;
				case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_DARK;
					break;
				case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_DARK;
					break;
				}
			}
		}

		/*if(	resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_VTH_BRIGHT			||
		resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_VTH_BRIGHT			||
		resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_VTH_DEFECT_BRIGHT	||
		resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_VTH_DEFECT_BRIGHT	)
		{
		resultPanelData.m_ListDefectInfo.RemoveAt(i);
		i--;
		}*/
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:Crack判定ver0.2																																//
//工作日:2017-11-10																																								//
//操作员:PNZ																																									//
//Crack Pattern:基于G32(Image Number 9)																								//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementCrack(ResultPanelData& resultPanelData)
{
	int nXLineNumber = 0;	// 线路不良

	int nCPLNumber_Left = 0;	// Crack Pattern Left Line线路故障
	int nCPLNumber_Right = 0;	// Crack Pattern Right Line线路故障
	int nOffSet = 10;	// 中心点距离差

	// Position of Crack Defect
	int Distance1 = 1467;
	int Distance2 = 1761;
	int Distance3 = 2054;

	cv::Point ptAlignCorner[4]; // Align后Cell Corner

	ptAlignCorner[0].x = m_stThrdAlignInfo.rcAlignCellROI.x;
	ptAlignCorner[0].y = m_stThrdAlignInfo.rcAlignCellROI.y;

	ptAlignCorner[2].x = m_stThrdAlignInfo.rcAlignCellROI.x + m_stThrdAlignInfo.rcAlignCellROI.width;
	ptAlignCorner[2].y = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height;

	int CenterLocation = ptAlignCorner[0].y + abs(ptAlignCorner[2].y - ptAlignCorner[0].y) / 2;
	int Distance = 0;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
		int	nImgNum1 = theApp.GetImageClassify(nImgNum);

		if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_32) continue; //choikwangil64边远服

		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//只有在Line性能真的不好的情况下才会Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
			nXLineNumber++;	// 真正的线条
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//在Black Pattern中出现固定位置明信片时,Crack出现故障
	//////////////////////////////////////////////////////////////////////////

	if (nXLineNumber > 0)
	{
		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_32) continue;

			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT)
			{
				CPoint	X1Center;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;
				//X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y	/ resultPanelData.m_ListDefectInfo[i].nRatio;

				if (X1Center.y <= CenterLocation) { nCPLNumber_Left++; }	//Left固定范围内的故障数量
				if (X1Center.y > CenterLocation) { nCPLNumber_Right++; }	//Right固定范围内的故障数量
			}
			else continue;
		}
	}

	if (nCPLNumber_Left >= 3)
	{
		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_32) continue;

			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT)
			{
				CPoint	X1Center;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				if (X1Center.y <= CenterLocation)
				{
					if (((abs(X1Center.y - ptAlignCorner[0].y) - Distance1) < nOffSet) ||
						((abs(X1Center.y - ptAlignCorner[0].y) - Distance2) < nOffSet) ||
						((abs(X1Center.y - ptAlignCorner[0].y) - Distance3) < nOffSet))
					{
						resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_CRACK_LEFT;
					}
				}
			}
			else continue;
		}
	}

	if (nCPLNumber_Right >= 3)
	{
		//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
			int	nImgNum1 = theApp.GetImageClassify(nImgNum);

			if (nImgNum1 != E_IMAGE_CLASSIFY_AVI_GRAY_32) continue;

			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT)
			{
				CPoint	X1Center;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				if (X1Center.y > CenterLocation)
				{
					if (((abs(X1Center.y - ptAlignCorner[2].y) - Distance1) < nOffSet) ||
						((abs(X1Center.y - ptAlignCorner[2].y) - Distance2) < nOffSet) ||
						((abs(X1Center.y - ptAlignCorner[2].y) - Distance3) < nOffSet))
					{
						resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_CRACK_RIGHT;
					}
				}
			}
			else continue;
		}
	}

	//如果Left和Right都超过3个,则将Crack Left和Right转换为Both
	if ((nCPLNumber_Left >= 3) && (nCPLNumber_Right >= 3))
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_CRACK_LEFT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_CRACK_RIGHT)
			{
				resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_CRACK_BOTH;
			}
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:Line缺陷分类ver0.2																															//
//工作日:2017-11-14																																										//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementClassifyLineDefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT])
{
	int nLineNumber = 0;	// 真正的线条不良
	int nLineDefect = 0;	// 弱线
	int nOffSet = 20;	// Line不良邻接范围

	cv::Mat ImgBK;
	if (!MatOrgImage[MAX_GRAB_STEP_COUNT - 4][0].empty())
	{
		ImgBK = MatOrgImage[MAX_GRAB_STEP_COUNT - 4][0].clone();
	}
	else return false;

	//不良数量
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//仅当Line性能不良时Count
		case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_X_OPEN:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
			nLineNumber++;
			break;
		}
	}

	//如果存在Line缺陷
	if (nLineNumber > 0)
	{
		//////////////////////////////////////////////////////////////////////////
				//DO,GO判定
		//////////////////////////////////////////////////////////////////////////

				//不良数量
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_OPEN)
			{
				CPoint	X1St, X1End;

				X1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
				X1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;

				int nMaxGV = resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
				int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
				int	nImgNum1 = theApp.GetImageClassify(nImgNum);

				//如果有Black或Crack Pattern河Line,5.5时G32为Crack Pattern
				if ((nImgNum1 == E_IMAGE_CLASSIFY_AVI_BLACK) || (nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_32))
				{
					if (nMaxGV >= 200)
					{
						nOffSet = 150; // 设置为大
					}
				}
				//在RGB Pattern Dark Line中增加非检查区域的offset
				else if ((nImgNum1 == E_IMAGE_CLASSIFY_AVI_R) || (nImgNum1 == E_IMAGE_CLASSIFY_AVI_G) || (nImgNum1 == E_IMAGE_CLASSIFY_AVI_B))
				{
					if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK && nOffSet < 30)
						nOffSet = 30;
				}

				//异常处理

				int BKImageLenght = ImgBK.cols;
				int StartPoint = X1St.x - nOffSet;
				int EndPoint = X1End.x + nOffSet;

				if (StartPoint < 0)	StartPoint = 0;
				if (EndPoint > BKImageLenght - 1)	EndPoint = BKImageLenght - 1;

				uchar* BKGV_St = ImgBK.ptr<uchar>(X1St.y, StartPoint);
				uchar* BKGV_End = ImgBK.ptr<uchar>(X1End.y, EndPoint);
				if (*BKGV_St > 0 && *BKGV_End > 0) continue;
				else if ((*BKGV_St > 0 && *BKGV_End == 0) || (*BKGV_St == 0 && *BKGV_End > 0))
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_X_OPEN;
				else continue;
			}

			//Y方向
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT)
			{
				CPoint	Y1St, Y1End;//, Y1Center;
				Y1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
				Y1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
				Y1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
				Y1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;

				int nMaxGV = resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;
				int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
				int	nImgNum1 = theApp.GetImageClassify(nImgNum);

				//如果有Black或Crack Pattern河Line,5.5时G32为Crack Pattern
				if ((nImgNum1 == E_IMAGE_CLASSIFY_AVI_BLACK) || (nImgNum1 == E_IMAGE_CLASSIFY_AVI_GRAY_32))
				{
					if (nMaxGV >= 200)
					{
						nOffSet = 150; // 设置为大
					}
				}
				//在RGB Pattern Dark Line中增加非检查区域的offset
				else if ((nImgNum1 == E_IMAGE_CLASSIFY_AVI_R) || (nImgNum1 == E_IMAGE_CLASSIFY_AVI_G) || (nImgNum1 == E_IMAGE_CLASSIFY_AVI_B))
				{
					if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DARK && nOffSet < 30)
						nOffSet = 30;
				}

				//异常处理

				int BKImageLenght = ImgBK.rows;
				int StartPoint = Y1St.y - nOffSet;
				int EndPoint = Y1End.y + nOffSet;

				if (StartPoint < 0)	StartPoint = 0;
				if (EndPoint > BKImageLenght - 1)	EndPoint = BKImageLenght - 1;

				uchar* BKGV_St = ImgBK.ptr<uchar>(StartPoint, Y1St.x);
				uchar* BKGV_End = ImgBK.ptr<uchar>(EndPoint, Y1End.x);
				if (*BKGV_St > 0 && *BKGV_End > 0) continue;
				else if ((*BKGV_St > 0 && *BKGV_End == 0))
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT;
				else if ((*BKGV_St == 0 && *BKGV_End > 0))
					resultPanelData.m_ListDefectInfo[i].Defect_Type = E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT;
			}
		}
	}
	ImgBK.release();
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	周四:删除6.39 QHD Notch Y Line																								//
//工作日:2018-07.04																																										//
//工作人员:YDS																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementNotchDefect(ResultPanelData& resultPanelData, cv::Mat MatOrgImage[][MAX_CAMERA_COUNT], double* dAlignPara)
{

	if (dAlignPara == NULL)
		return false;

	int		NORCH_OnOff = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_ONOFF]);
	int		NORCH_UNIT_X = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_UNIT_X]);
	int		NORCH_UNIT_Y = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_UNIT_Y]);
	double	NORCH_THGV_GREEN = (double)(dAlignPara[E_PARA_NORCH_OVERKILL_THGV_GREEN]);
	double	NORCH_THGV_RB = (double)(dAlignPara[E_PARA_NORCH_OVERKILL_THGV_RB]);
	int		NORCH_ABS_GREEN = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_ABS_GREEN]);
	int		NORCH_ABS_RB = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_ABS_RB]);
	int		NORCH_OFFSET = (int)(dAlignPara[E_PARA_NORCH_OVERKILL_OFFSET]);

	//打开/关闭确认(需要使用)Norch不存在的被关闭
	if (NORCH_OnOff == 0) return false;

	int		nYLineNumber = 0;	// 真正的线条不良
	int		nOffSet = 0;	// Line不良邻接范围
	int		nLengthOffSet = 0;	// Norch坏长度Offset
	bool	ImgBKEmpty = false;

	cv::Mat ImgBK;
	if (!MatOrgImage[MAX_GRAB_STEP_COUNT - 4][0].empty())
	{
		ImgBK = MatOrgImage[MAX_GRAB_STEP_COUNT - 4][0].clone();
	}
	else
		ImgBKEmpty = true;

	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//仅当Line性能不良时Count
		case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_DGS_Y:
			nYLineNumber++;	// 真正的线条
			break;
		}
	}

	if (nYLineNumber > 0 && ImgBKEmpty == false)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{

			int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number; // UI ( list )		
			int	nImgNum1 = theApp.GetImageClassify(nImgNum); // Alg ( E_IMAGE_CLASSIFY_AVI_R ... )

			if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_BLACK) continue;

			nOffSet = NORCH_OFFSET;
			nLengthOffSet = 150;

			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_Y ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_XLINE_SPOT)
			{
				//检查Norch部分是否有问题
				vector<int> NorchIndex;
				CPoint OrgIndex, DefectY_Center;
				int DefectY_XStart, NorchLocation, TopNorchLocation, BottomNorchLocation, DefectLength, DefectY_YStart, DefectY_YEnd, DefectY_MaxGV;

				// UI ( list )
				bool bNorchInfoGet = GetModelNorchInfo(theApp.m_pGrab_Step[nImgNum].tRoundSet, NorchIndex, OrgIndex);

				int NorchAvgWidth = 0;
				int NorchAvgHeight = 0;

				if (bNorchInfoGet == true)
				{
					NorchAvgWidth = (NorchIndex[0] + NorchIndex[2]) / 2;
					NorchAvgHeight = (NorchIndex[1] + NorchIndex[3]) / 2;

					NorchLocation = NorchAvgWidth + m_stThrdAlignInfo.rcAlignCellROI.x + nOffSet;
					TopNorchLocation = OrgIndex.x + m_stThrdAlignInfo.rcAlignCellROI.y;
					BottomNorchLocation = OrgIndex.y + m_stThrdAlignInfo.rcAlignCellROI.y;
				}

				//计算Norch局部坐标
				else if (bNorchInfoGet == false)
				{
					NorchAvgWidth = NORCH_UNIT_X;
					NorchAvgHeight = NORCH_UNIT_Y;

					NorchLocation = NORCH_UNIT_X + m_stThrdAlignInfo.rcAlignCellROI.x + nOffSet;
					TopNorchLocation = NORCH_UNIT_Y + m_stThrdAlignInfo.rcAlignCellROI.y;
					BottomNorchLocation = m_stThrdAlignInfo.rcAlignCellROI.y + m_stThrdAlignInfo.rcAlignCellROI.height - NORCH_UNIT_Y;
				}

				//不良信息
				DefectY_Center.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				DefectY_Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;
				DefectY_YStart = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
				DefectY_YEnd = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;
				DefectY_XStart = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
				DefectY_MaxGV = (long)resultPanelData.m_ListDefectInfo[i].Defect_MaxGV;

				DefectLength = abs(DefectY_YStart - DefectY_YEnd);

				//不处理除Case1 Norch部分以外的LINE
				if (DefectY_Center.x >= NorchLocation) continue;
				// 2022.07.04
				//if ( DefectY_MaxGV >= 250				) continue;

				int DefectY_GV = ImgBK.at<uchar>(DefectY_Center.y, (DefectY_XStart - nOffSet));

				//案例1:两个Norch之间的故障
				if ((TopNorchLocation <= DefectY_Center.y && DefectY_Center.y <= BottomNorchLocation) && DefectY_GV == 255)
				{
					//参数初始化
					double DefectMeanGV = 0;
					double LeftMeanGV = 0;
					double RightMeanGV = 0;

					//前往相应的Pattern进行确认
					cv::Mat DefectImage = MatOrgImage[theApp.GetImageNum(nImgNum1)][0](Rect(0, 0, NorchLocation + nLengthOffSet, MatOrgImage[theApp.GetImageNum(nImgNum1)][0].rows));

					// Defect ROI
					cv::Rect DefectRect;

					DefectRect.x = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					DefectRect.width = resultPanelData.m_ListDefectInfo[i].Pixel_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.height = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;

					double MaxGV;

					DefectLDRCompair(DefectImage, DefectRect, LeftMeanGV, DefectMeanGV, RightMeanGV, MaxGV);

					double dbDiffGV = 0;

					if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_R || nImgNum1 == E_IMAGE_CLASSIFY_AVI_B) dbDiffGV = NORCH_ABS_RB;
					if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_G)										dbDiffGV = NORCH_ABS_GREEN;

					if (abs(DefectMeanGV - RightMeanGV) <= dbDiffGV)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(i);
						i--;
					}
				}

				else if ((TopNorchLocation <= DefectY_Center.y && DefectY_Center.y <= BottomNorchLocation) && (DefectLength > NorchAvgHeight + nLengthOffSet))
				{
					int DefectY_Start_GV = ImgBK.at<uchar>(DefectY_YStart, (DefectY_XStart - nOffSet));
					int DefectY_End_GV = ImgBK.at<uchar>(DefectY_YEnd, (DefectY_XStart - nOffSet));

					if (DefectY_Start_GV + DefectY_End_GV > 0)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(i);
						i--;
					}
				}

				else if ((DefectY_Center.y < TopNorchLocation || BottomNorchLocation < DefectY_Center.y) && (DefectLength > NorchAvgHeight + nLengthOffSet))
				{
					int DefectY_Start_GV = ImgBK.at<uchar>(DefectY_YStart, (DefectY_XStart - nOffSet));
					int DefectY_End_GV = ImgBK.at<uchar>(DefectY_YEnd, (DefectY_XStart - nOffSet));

					if (DefectY_Start_GV + DefectY_End_GV > 0)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(i);
						i--;
					}
				}

				//额外的Norch故障检查
				else if ((DefectY_Center.y < TopNorchLocation || BottomNorchLocation < DefectY_Center.y) && (DefectLength <= NorchAvgHeight + nLengthOffSet))
				{
					//参数初始化
					double DefectMeanGV = 0;
					double LeftMeanGV = 0;
					double RightMeanGV = 0;

					//Camera Type故障诊断
					bool NorchOverKill = false;

					int	nImgNum = resultPanelData.m_ListDefectInfo[i].Img_Number;
					int	nImgNum1 = theApp.GetImageClassify(nImgNum);

					double NORCH_THGV = 0;

					//按Pattern选择不同的Diff GV
					if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_G) NORCH_THGV = NORCH_THGV_GREEN;
					if (nImgNum1 == E_IMAGE_CLASSIFY_AVI_R || nImgNum1 == E_IMAGE_CLASSIFY_AVI_B) NORCH_THGV = NORCH_THGV_RB;

					//如果说在Norch边界部分
					if ((NorchLocation - nOffSet) >= DefectY_XStart)								  NORCH_THGV = NORCH_THGV_RB;

					//前往相应的Pattern进行确认
					cv::Mat DefectImage = MatOrgImage[theApp.GetImageNum(nImgNum1)][0](Rect(0, 0, NorchLocation + nLengthOffSet, MatOrgImage[theApp.GetImageNum(nImgNum1)][0].rows));

					// Defect ROI
					cv::Rect DefectRect;

					DefectRect.x = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					DefectRect.width = resultPanelData.m_ListDefectInfo[i].Pixel_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					DefectRect.height = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;

					double MaxGV;

					bool CompairResult = DefectLDRCompair(DefectImage, DefectRect, LeftMeanGV, DefectMeanGV, RightMeanGV, MaxGV);

					if (CompairResult == false)															NorchOverKill = true;

					if (MaxGV >= 200)																	continue;

					if (LeftMeanGV <= DefectMeanGV && DefectMeanGV <= RightMeanGV)						NorchOverKill = true;

					else if (abs(DefectMeanGV - (LeftMeanGV + RightMeanGV) / 2) < NORCH_THGV)	NorchOverKill = true;

					else																				NorchOverKill = false;

					if (NorchOverKill == true)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(i);
						i--;
					}
				}
			}
		}
	}

	if (nYLineNumber > 0 && ImgBKEmpty == true)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_RIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_OPEN_LEFT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_DGS_Y)
			{
				int DefectY_Center = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;

				if (abs((m_stThrdAlignInfo.rcAlignCellROI.x + 400) - DefectY_Center) < 100)
				{
					resultPanelData.m_ListDefectInfo.RemoveAt(i);
					i--;
				}
			}
		}
	}

	ImgBK.release();
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//第	目:Special Pattern和Gray Pattern弱线检测结果比较ver.0.1					//
//工作日:2017-11-22																																									//
//操作员:PNZ																																									//
//////////////////////////////////////////////////////////////////////////////////////////////////

bool AviInspection::JudgementSpecialGrayLineDefect(ResultPanelData& resultPanelData)
{
	int nXLineNumber = 0;	// 真正的线条不良
	int nYLineNumber = 0;	// 真正的线条不良
	int nOffSet = 10;	// 中心点距离差

	//计算不良数量的部分
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK:
			nXLineNumber++;	// 真正的线条
			break;

		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:
		case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK:
			nYLineNumber++;	// 真正的线条
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//x方向
	//////////////////////////////////////////////////////////////////////////

	if (nXLineNumber > 1)//如果有X Line,就必须工作
	{
		//比较
		int PatternIndex;
		int	XLine_G64 = 0;
		int XLine_SP2 = 0;

		//计数在G64 Pattern中检测到的Line Defect
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//基于X方向明线最先出现的Pattern的Pattern
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
			{
				PatternIndex = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;
				if (PatternIndex == 8)
				{
					XLine_G64++;
				}
			}
		}

		//Count在Special-2 Pattern中检测到的Line Defect
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//基于X方向明线最先出现的Pattern的Pattern
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
			{
				PatternIndex = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;
				if (PatternIndex == 6)
				{
					XLine_SP2++;
				}
			}
		}

		//SP2没有线路故障时,G64出现故障时
		if ((XLine_G64 > 0) && (XLine_SP2 == 0))
		{
			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
			{
				if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
					resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
				{
					j++; continue;
				}

				int		X2DefectImageType;
				int		FirstLINEDefect = 0;

				X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

				//在G64 Pattern中
				if (X2DefectImageType == 9)
				{
					//第一个Line Defect
					FirstLINEDefect++;
					if (FirstLINEDefect == 1)
					{
						resultPanelData.m_ListDefectInfo[j].Defect_Type = E_DEFECT_JUDGEMENT_RETEST_LINE_BRIGHT;
					}

					if (FirstLINEDefect > 1)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}
				}
				else  j++;
			}
		}

		//以G64 Pattern为基准,与Special Pattern中检测到的Line Defect进行比较,如果位置相同,则清除Special Pattern的不良行为。呵
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
			{
				CPoint	X1St, X1End, X1Center;
				int		X1DefectImageType;

				X1St.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1End.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1Center.y = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[i].nRatio;
				X1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//应该是比较后得到的Pattern。呵
				if (X1DefectImageType != 6) continue;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
					{
						j++; continue;
					}

					CPoint	X2St, X2End, X2Center;
					int		X2DefectImageType;

					X2St.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2End.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2Center.y = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_Y / resultPanelData.m_ListDefectInfo[j].nRatio;
					X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//不能与标准Pattern相同,其他Pattern的缺陷
					if (X2DefectImageType == X1DefectImageType) { j++; continue; }

					//如果在周围检测到Line
					if (abs(X1Center.y - X2Center.y) >= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//y方向
	//////////////////////////////////////////////////////////////////////////

	if (nYLineNumber > 1)//如果有X Line,就必须工作
	{
		//比较
		int PatternIndex;
		int	 YLine_G64 = 0;
		int YLine_SP2 = 0;

		//计数在G64 Pattern中检测到的Line Defect
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//基于X方向明线最先出现的Pattern的Pattern
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
			{
				PatternIndex = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;
				if (PatternIndex == 8)
				{
					YLine_G64++;
				}
			}
		}

		//Count在Special-2 Pattern中检测到的Line Defect
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//基于X方向明线最先出现的Pattern的Pattern
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
			{
				PatternIndex = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;
				if (PatternIndex == 6)
				{
					YLine_SP2++;
				}
			}
		}

		//以G64 Pattern为基准,与Special Pattern中检测到的Line Defect进行比较,如果位置相同,则清除Special Pattern的不良行为。呵
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			//X方向不不良除外(因为可能会出现弱线,所以添加了)
			if (resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
				resultPanelData.m_ListDefectInfo[i].Defect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
			{
				CPoint	Y1St, Y1End, Y1Center;
				int		Y1DefectImageType;

				Y1St.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Start_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1End.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_End_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1Center.x = (long)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X / resultPanelData.m_ListDefectInfo[i].nRatio;
				Y1DefectImageType = (long)resultPanelData.m_ListDefectInfo[i].Pattern_Type;	// 模式

				//应该是比较后得到的Pattern。呵
				if (Y1DefectImageType != 6) continue;

				//无法比较
				for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
				{
					//避免类似的不良比较
					if (i == j) { j++; continue; }

					///X方向不不良除外(因为可能会出现弱线,所以添加了)
					if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
						resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
					{
						j++; continue;
					}

					CPoint	Y2St, Y2End, Y2Center;
					int		Y2DefectImageType;

					Y2St.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Start_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2End.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_End_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2Center.x = (long)resultPanelData.m_ListDefectInfo[j].Pixel_Center_X / resultPanelData.m_ListDefectInfo[j].nRatio;
					Y2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

					//不能与标准Pattern相同,其他Pattern的缺陷
					if (Y2DefectImageType == Y1DefectImageType) { j++; continue; }

					//如果在周围检测到Line
					if (abs(Y1Center.x - Y2Center.x) >= nOffSet)
					{
						//删除小列表时
						if (i > j)	 i--;

						//删除其他模式中的所有行
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
					}

					//如果周围没有缺陷
					else  j++;
				}
			}
		}
	}

	//删除Special Pattern中的Line Defect。Z Z
	for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); )
	{
		if (resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
			resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK &&
			resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
			resultPanelData.m_ListDefectInfo[j].Defect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
		{
			j++; continue;
		}

		int		X2DefectImageType;

		X2DefectImageType = (long)resultPanelData.m_ListDefectInfo[j].Pattern_Type;	// 模式

		if (X2DefectImageType == 6)
		{
			resultPanelData.m_ListDefectInfo.RemoveAt(j);
		}
		else  j++;
	}

	return true;
}


long AviInspection::LabelProcess(cv::Mat matSrcBuf, int nImageNum, int nCameraNum, tAlignInfo& stCamAlignInfo)
{
	// test
	CCPUTimer tact;
	tact.Start();


	double* dAlgPara = theApp.GetAlignParameter(nCameraNum);

	int nEqpType = theApp.m_Config.GetEqpType();

	CRect rectLabelrea[MAX_MEM_SIZE_LABEL_COUNT];

	long nErrorCode = Align_FindDefectLabel(matSrcBuf, dAlgPara, stCamAlignInfo.ptCorner, stCamAlignInfo.dAlignTheta, rectLabelrea, nEqpType);

	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].nLabelROICnt = 0;
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Label Inspect Failed!"));
		return nErrorCode;
	}
	else
	{
		//赋值
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].nLabelROICnt = 1;
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].LabelROI[0].bUseROI = true;
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].LabelROI[0].strROIName[0] = NULL;
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].LabelROI[0].rectROI.left = rectLabelrea[0].left;
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].LabelROI[0].rectROI.top = rectLabelrea[0].top;
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].LabelROI[0].rectROI.right = rectLabelrea[0].right;
		theApp.m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].LabelROI[0].rectROI.bottom = rectLabelrea[0].bottom;
	}


	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Inspect Label : %.2f"), tact.Stop(false) / 1000.);

	return nErrorCode;
}


long AviInspection::MarkProcess(cv::Mat matSrcBuf, int nImageNum, int  nCameraNum, tAlignInfo& stCamAlignInfo)
{
	// test
	CCPUTimer tact;
	tact.Start();

	int			nMarkROICnt = theApp.GetMarkROICnt(nImageNum, nCameraNum);

	CRect		rectMarkArea[MAX_MEM_SIZE_MARK_COUNT];

	if (nMarkROICnt <= 0)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("The number of MarkROICnt < 0"));
		return E_ERROR_CODE_TRUE;
	}
	else
	{

		for (int nROICnt = 0; nROICnt < nMarkROICnt; nROICnt++)
		{
			rectMarkArea[nROICnt] = theApp.GetMarkROI(nImageNum, nCameraNum, nROICnt);
		}

	}

	// 俺喊 舅绊府硫 八荤 颇扼固磐 啊廉坷扁
	double* dAlgPara = theApp.GetAlignParameter(nCameraNum);

	// 厘厚 鸥涝
	int nEqpType = theApp.m_Config.GetEqpType();

	long nErrorCode = Align_FindDefectMark(matSrcBuf, dAlgPara, stCamAlignInfo.ptCorner, stCamAlignInfo.dAlignTheta, stCamAlignInfo.rcMarkROI, rectMarkArea, nEqpType, nMarkROICnt);

	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		for (int nROICnt = 0; nROICnt < nMarkROICnt; nROICnt++)
		{
			stCamAlignInfo.rcMarkROI[nROICnt].x = 0;
			stCamAlignInfo.rcMarkROI[nROICnt].y = 0;
			stCamAlignInfo.rcMarkROI[nROICnt].width = 0;
			stCamAlignInfo.rcMarkROI[nROICnt].height = 0;
		}
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Mark Inspect Failed!"));
		return nErrorCode;
	}
	else
	{
		for (int nROICnt = 0; nROICnt < nMarkROICnt; nROICnt++)
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Mark%d's Position is: Rect(%d,%d,%d,%d)"), nROICnt,
				stCamAlignInfo.rcMarkROI[nROICnt].x, stCamAlignInfo.rcMarkROI[nROICnt].y, stCamAlignInfo.rcMarkROI[nROICnt].width, stCamAlignInfo.rcMarkROI[nROICnt].height);
		}
	}
	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Inspect Mark : %.2f"), tact.Stop(false) / 1000.);

	return nErrorCode;
}



//////////////////////////////////////////////////////////////////////////
//Align后确认ROI亮度异常
//////////////////////////////////////////////////////////////////////////
long AviInspection::CheckADGV(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nStageNo, int nImageNum, int nCameraNum, int nRatio, cv::Point* ptCorner, ResultBlob_Total* pResultBlobTotal, double* dMeanResult,
	bool& bChkDustEnd, bool& bNeedRetry, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool& bIsHeavyAlarm, ENUM_INSPECT_MODE eInspMode)
{
	const int DUST_RETRY_LIMIT = 1;
	// test
	CCPUTimer tact;
	tact.Start();
	long nErrorCode = E_ERROR_CODE_TRUE;

	//设备类型
	int nEqpType = theApp.m_Config.GetEqpType();

	//获取单个算法检查参数
	//要单独定义AD Parameter。ToDo.
	double* dAlgPara = theApp.GetFindDefectADParameter(nImageNum, nCameraNum);

	//算法画面号
	int nAlgImg = theApp.GetImageClassify(nImageNum);

	//用于Alg日志
	wchar_t strAlgLog[MAX_PATH] = { 0, };
	swprintf(strAlgLog, _T("ID:%s\tPat:%s"), (LPCWSTR)strPanelID, theApp.GetGrabStepName(nImageNum));

	//如果是Dust和DUSTDOWN模式 hjf
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST || nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)
	{
		if (!theApp.GetUseFindDefectAD(nImageNum, nCameraNum))
		{
			//如果不检查Dust GV,将发出警报
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Disabled Check DUST - InspSkip !!!. PanelID: %s, Stage: %02d, CAM: %02d"),
				strPanelID, nStageNo, nCameraNum);

			theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eInspMode, eALARMID_DIABLE_CHECK_DUST, eALARMTYPE_HEAVY, bIsHeavyAlarm);

			// Seq. 结束Dust并将结果发送给Task
			bNeedRetry = false;
			bIsNormalDust = true;
			bChkDustEnd = true;
		}
		else
		{
			//AD-检查亮度值
			nErrorCode = Align_FindDefectAD_GV(MatOrgImage, dAlgPara, dMeanResult, ptCorner, nEqpType, nAlgImg, strAlgLog);

			if (nErrorCode != E_ERROR_CODE_TRUE)
			{
				bIsNormalDust = false;

				if (bUseDustRetry && nDustRetryCnt < DUST_RETRY_LIMIT)
				{
					// Seq. 结束Dust并将结果发送给Task
					bNeedRetry = true;
					//不发出警报,不进行检查和报告
					bIsHeavyAlarm = true;
				}
				else
				{
					bNeedRetry = false;
					//Dust状态异常警报
					theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eInspMode, eALARMID_DUST_ABNORMAL, eALARMTYPE_HEAVY, bIsHeavyAlarm);
				}

				// Alg DLL Stop Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("DUST Abnormal - InspStop !!!. PanelID: %s, Stage: %02d, CAM: %02d, Img: %s..\n\t\t\t\t( Avg : %.2f GV )"),
					strPanelID, nStageNo, nCameraNum, theApp.GetGrabStepName(nImageNum), dMeanResult[0]);

				CString strOrgFileName = _T("");
				strOrgFileName.Format(_T("%s_CAM%02d_Abnormal_Dust"), theApp.GetGrabStepName(nImageNum), nCameraNum);
				strOrgFileName = strOrgFileName + _T(".bmp");

				CString strOriginDrive = theApp.m_Config.GetOriginDriveForAlg();
				ImageSave(MatOrgImage, _T("%s\\%s\\%02d_%s"),
					ORIGIN_PATH, strPanelID, nImageNum, strOrgFileName);
			}
			else
			{
				// Seq. 结束Dust并将结果发送给Task
				bNeedRetry = false;
				bIsNormalDust = true;
				// Alg DLL End Log
				theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<AD GV> 算法结束. 载台号: %d GV均值: %.2f"),
					strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, dMeanResult[0]);
			}
			bChkDustEnd = true;
		}
	}
	//如果不是Dust和DUSTDOWN模式
	else
	{
		//AVI等待Dust灯光状态检查结束
		while (!bChkDustEnd)
		{
			Sleep(10);
		}

		if (bIsNormalDust)
		{
			//点亮区域AD GV值验证代码
			if (theApp.GetUseFindDefectAD(nImageNum, nCameraNum))
			{
				//////////////////////////////////////////////////////////////////////////
				// 17.11.23
								//5.99"出现工艺问题-特定位置线存在不良
							//因线路不良判定AD(线路不良判定为亮AD)->未进行检查:E级判定
							//排除非检查区域后,修改为进行AD检查
				//////////////////////////////////////////////////////////////////////////

							//如果启用
				if (dAlgPara[E_PARA_CHECK_NON_DEL_FLAG] > 0)
				{
					cv::Mat matTempBuf = MatOrgImage.clone();

					//获取非检查区域ROI数量
					int nFilterROICnt = theApp.GetFilterROICnt(nImageNum, nCameraNum);

					//非检查区域ROI数量
					for (int nROICnt = 0; nROICnt < nFilterROICnt; nROICnt++)
					{
						//验证是否已启用
						if (theApp.GetUseFilterROI(nImageNum, nCameraNum, nROICnt))
						{
							//获取无检查区域ROI
							///在P/S模式下进行校正
							CRect rectFilterArea = theApp.GetFilterROI(nImageNum, nCameraNum, nROICnt, nRatio);

							//非检查区域-以Left-Top坐标为原点具有坐标值
							rectFilterArea.OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

							//设置大一点使用(需要参数化)
							int nOut = (int)dAlgPara[E_PARA_CHECK_NON_DEL_OFFSET];
							rectFilterArea.InflateRect(nOut, nOut, nOut, nOut);

							//异常处理
							if (rectFilterArea.left < 0)	rectFilterArea.left = 0;
							if (rectFilterArea.top < 0)	rectFilterArea.top = 0;
							if (rectFilterArea.right >= matTempBuf.cols)	rectFilterArea.right = matTempBuf.cols - 1;
							if (rectFilterArea.bottom >= matTempBuf.rows)	rectFilterArea.bottom = matTempBuf.rows - 1;

							// ROI
							cv::Mat matZeroBuf = matTempBuf(cv::Rect(rectFilterArea.left, rectFilterArea.top, rectFilterArea.Width(), rectFilterArea.Height()));

							//初始化为ROI 0
							matZeroBuf.setTo(0);

							//禁用
							matZeroBuf.release();
						}
					}

					//检查结果画面
	//cv::imwrite("E:\\IMTC\\asd.bmp", matTempBuf);

						//AD-检查亮度值
					nErrorCode = Align_FindDefectAD_GV(matTempBuf, dAlgPara, dMeanResult, ptCorner, nEqpType, nAlgImg, strAlgLog);

					matTempBuf.release();
				}
				//未启用
				else
				{
					//AD-检查亮度值
					nErrorCode = Align_FindDefectAD_GV(MatOrgImage, dAlgPara, dMeanResult, ptCorner, nEqpType, nAlgImg, strAlgLog);
				}

				//如果有错误,则输出错误代码和日志
				if (nErrorCode != E_ERROR_CODE_TRUE)
				{
					// Alg DLL Stop Log



					//亮度以上
					if (nErrorCode == E_ERROR_CODE_ALIGN_DISPLAY)
					{
						JudgeADDefect(nImageNum, nCameraNum, nStageNo, MatOrgImage.cols, MatOrgImage.rows, pResultBlobTotal, (int)dMeanResult[1], eInspMode);
					}
					//没有缓冲区/参数异常/等等...
					else
					{
						//需要单独报告为不良的规格-汇总报告为当前AD GV不良
						JudgeADDefect(nImageNum, nCameraNum, nStageNo, MatOrgImage.cols, MatOrgImage.rows, pResultBlobTotal, E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT, eInspMode);
					}

					// Alg DLL End Log
					theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<AD GV> 算法异常 跳过算法检测. 载台号: %d GV均值: %.2f GV "),
						strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, dMeanResult[0]);
				}
				else
				{
					// Alg DLL End Log
					theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<AD GV> 算法结束. 载台号: %d GV均值: %.2f GV "),
						strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nStageNo, dMeanResult[0]);
				}
			}
		}
		else
		{
			nErrorCode = E_ERROR_CODE_FALSE;
		}
	}
	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Check AD GV (Image : %s) : %.2f"), theApp.GetGrabStepName(nImageNum), tact.Stop(false) / 1000.);

	return nErrorCode;
}

double AviInspection::calcResolution(CWriteResultInfo WrtResultInfo)
{
	// Panel Size
	double dPanelSizeX = WrtResultInfo.GetPanelSizeX();
	double dPanelSizeY = WrtResultInfo.GetPanelSizeY();

	//渐变的宽度
	double dTempX, dTempY, dImageSizeX, dImageSizeY;
	dTempX = m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_TOP].x - m_stThrdAlignInfo.ptCorner[E_CORNER_RIGHT_TOP].x;
	dTempY = m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_TOP].y - m_stThrdAlignInfo.ptCorner[E_CORNER_RIGHT_TOP].y;
	dImageSizeX = sqrt(dTempX * dTempX + dTempY * dTempY);

	//渐变的垂直长度
	dTempX = m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_TOP].x - m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_BOTTOM].x;
	dTempY = m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_TOP].y - m_stThrdAlignInfo.ptCorner[E_CORNER_LEFT_BOTTOM].y;
	dImageSizeY = sqrt(dTempX * dTempX + dTempY * dTempY);

	//横向,纵向Resolution
	double dResolutionX = dPanelSizeX / dImageSizeX;
	double dResolutionY = dPanelSizeY / dImageSizeY;

	//平均分辨率(非P/S模式下的分辨率)
	double dResolution = (dResolutionX + dResolutionY) / 2.0 * m_stThrdAlignInfo.nRatio;

	//CString strTemp;
	//strTemp.Format(_T("%f, %f\n%f, %f\n%f, %f, %f"), dPanelSizeX, dPanelSizeY, dImageSizeX, dImageSizeY, dResolutionX, dResolutionY, dResolution);
	//AfxMessageBox(strTemp);

	return dResolution;
}

bool AviInspection::JudgementRepeatCount(CString strPanelID, ResultPanelData& resultPanelData)
{
	//避免线程重复访问
	//按当前结果的顺序检查重复次数-不考虑面板的进度顺序
	EnterCriticalSection(&theApp.m_csJudgeRepeatCount);

	ListCurDefect* pStCurDefList = new ListCurDefect();
	std::list<RepeatDefectInfo>* pList = theApp.GetRepeatDefectInfo();					// 现有不良列表(Pixel坐标)
	static bool bIsAlarm = false;														// 是否存在现有警报

	CCPUTimer ttRepeatCount;
	ttRepeatCount.Start();

	memcpy(pStCurDefList->bUseChkRptDefect, theApp.m_Config.GetUseCCDAlarm(), sizeof(pStCurDefList->bUseChkRptDefect));

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Judge Repeat (0) %.2f"), ttRepeatCount.Stop(false) / 1000.);

	//如果同时禁用Pixel/Work坐标检查,则清除现有的List并返回正常值
	if (!pStCurDefList->bUseChkRptDefect[ePIXEL] && !pStCurDefList->bUseChkRptDefect[eWORK])
	{
		//删除现有的进度历史
		for (int nKind = 0; nKind < eMACHINE; nKind++)			// 2018. 09.21由于添加MDJ APP Repeat Defect,更改为少于类型1个
		{
			if (pList[nKind].size() != 0)
				pList[nKind].clear();
		}

		m_fnSaveRepeatDefInfo(pList);
		bIsAlarm = false;
		LeaveCriticalSection(&theApp.m_csJudgeRepeatCount);
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Skip Repeat Defect !!! - Not Used"));
		return true;
	}

	const int DIGIT_PNLID = 17;		// 设置面板ID位数
	CString strOrgPanelID = strPanelID;
	static CString strOldPanelID = _T("");

	// [PANELID]_1, [PANELID]_2 ... 等,如果Panel ID大于实际位数,则只按Panel ID的大小进行剪切比较
	if (strOrgPanelID.GetLength() > DIGIT_PNLID)
		strOrgPanelID = strPanelID.Left(DIGIT_PNLID);

	//在没有出现警报的情况下,在同一Panel连续进行时不进行检查
	//如果已经发出警报,即使重新进行相同的Panel ID,在解决导致的问题之前,也会持续发出警报
	if (strOrgPanelID.Compare(strOldPanelID) == 0 && !bIsAlarm)
	{
		LeaveCriticalSection(&theApp.m_csJudgeRepeatCount);
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Skip Repeat Defect !!! - Overlap Panel ID (%s)"), strOrgPanelID);
		return true;
	}
	else
	{
		strOldPanelID = strOrgPanelID;
	}

	bool bRet = false;
	//当前重复错误检查最多限制为1000个
	//如果需要增加最大数量,请考虑提高速度
	const int MAX_REPEAT_DEFECT = 1000;
	//[0]:CCD/[1]:工程(工作坐标)不良
	int* nRptOffset = theApp.m_Config.GetCCDOffset();
	int* nRptLArmCnt = theApp.m_Config.GetCCDLightAlarmCount();
	int* nRptHArmCnt = theApp.m_Config.GetCCDHeavyAlarmCount();

	int nTotalCount = 0;
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				///在此添加需要检查的项目(Pixel/Work)
		switch (resultPanelData.m_ListDefectInfo[i].Defect_Type)
		{
			//仅在属于Point Dark/Bright的情况下创建比较list
			//如果是Point Dark
		case E_DEFECT_JUDGEMENT_POINT_DARK:
		case E_DEFECT_JUDGEMENT_POINT_GROUP_DARK:
			//添加Pixel,Work坐标-只添加需要比较的项目
			pStCurDefList->Add_Tail(ePIXEL, E_DEFECT_JUDGEMENT_POINT_DARK, &resultPanelData.m_ListDefectInfo[i]);
			pStCurDefList->Add_Tail(eWORK, E_DEFECT_JUDGEMENT_POINT_DARK, &resultPanelData.m_ListDefectInfo[i]);
			nTotalCount++;
			break;

			//如果是Point Bright
		case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
		case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT:
		case E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT:
			pStCurDefList->Add_Tail(ePIXEL, E_DEFECT_JUDGEMENT_POINT_BRIGHT, &resultPanelData.m_ListDefectInfo[i]);
			pStCurDefList->Add_Tail(eWORK, E_DEFECT_JUDGEMENT_POINT_BRIGHT, &resultPanelData.m_ListDefectInfo[i]);
			nTotalCount++;
			break;

		default:
			break;
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				//如果当前不良品太多,Skip-
		if (nTotalCount > MAX_REPEAT_DEFECT)
		{
			theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Skip Repeat Defect !!! - Too Many Defect (Repeat Defect Count : Over %d)"), MAX_REPEAT_DEFECT);
			LeaveCriticalSection(&theApp.m_csJudgeRepeatCount);
			return false;
		}
	}

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Judge Repeat (1) %.2f"), ttRepeatCount.Stop(false) / 1000.);

	//比较现有不良列表和当前不良列表
	//删除不重叠的错误
	//获得重叠的不良平均值并增加计数	
	int nMaxRepeatCount[eMACHINE] = { 0, };					// 2018. 09. 21由于添加MDJ APP Repeat Defect,更改为比类型少1个

#pragma omp parallel for schedule(dynamic)		//添加并行处理
	for (int nKind = 0; nKind < eMACHINE; nKind++)			// 2018. 09.21由于添加MDJ APP Repeat Defect,更改为少于类型1个
	{
		std::list<RepeatDefectInfo>::iterator iterDst, iterSrc;
		bool bRepeat = false;
		//现有不良列表
		for (iterDst = pList[nKind].begin(); iterDst != pList[nKind].end(); )
		{
			bRepeat = false;

			//创建现有不良位置区域
			//CCD不良-单位Pixel/工序不良-单位um
			CRect rcDst(iterDst->ptCenterPos.x - nRptOffset[nKind] - 1, iterDst->ptCenterPos.y - nRptOffset[nKind] - 1,			// LT
				iterDst->ptCenterPos.x + nRptOffset[nKind] + 1, iterDst->ptCenterPos.y + nRptOffset[nKind] + 1);					// RB

			//当前不良列表
			for (iterSrc = pStCurDefList->listCurDefInfo[nKind].begin(); iterSrc != pStCurDefList->listCurDefInfo[nKind].end(); )
			{
				if (iterDst->eDefType == iterSrc->eDefType)
				{
					//如果重复的位置有新的故障
					if (PtInRect(rcDst, iterSrc->ptCenterPos))
					{
						bRepeat = true;

						//获得中心坐标平均值
						iterDst->ptCenterPos += iterSrc->ptCenterPos;
						iterDst->ptCenterPos.x /= 2;
						iterDst->ptCenterPos.y /= 2;
						iterDst->nRepeatCount++;
						//计数增加后刷新最大重复次数
						if (iterDst->nRepeatCount >= nRptLArmCnt[nKind])
						{
							nMaxRepeatCount[nKind] = MAX(nMaxRepeatCount[nKind], iterDst->nRepeatCount);		// 最大重复次数

							//日志的字符串
							CString strErrMsg = _T("");
							CString strErrType = _T("");
							CString strDefType = _T("");

							if (iterDst->nRepeatCount >= nRptHArmCnt[nKind])		strErrMsg = _T("Error");
							else												strErrMsg = _T("Warning");

							if (nKind == ePIXEL)	strErrType = _T("Pixel");
							else					strErrType = _T("Work");

							switch (iterDst->eDefType)
							{
							case E_DEFECT_JUDGEMENT_POINT_DARK:
								strDefType = _T("POINT_DARK");
								break;
							case E_DEFECT_JUDGEMENT_POINT_BRIGHT:
								strDefType = _T("POINT_BRIGHT");
								break;
							default:
								strDefType.Format(_T("%d"), (int)iterDst->eDefType);
								break;
							}
							theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("%s !! Repeat Defect(%s) %d Times !!! (DefCode <%s> : %d, %d)"),
								strErrMsg,
								strErrType,
								iterDst->nRepeatCount,
								strDefType,
								iterDst->ptCenterPos.x, iterDst->ptCenterPos.y);
						}
						//获取现有列表平均值并增加计数后删除当前不良
						iterSrc = pStCurDefList->listCurDefInfo[nKind].erase(iterSrc);
						//删除重复不良后不用担心不良
						//(假设新的不良列表在Offset内没有相互重叠的不良列表)
						break;
					}
					else
					{
						iterSrc++;
					}
				}
				else
				{
					iterSrc++;
				}
			}

			//如果没有任何重叠,则删除现有列表中的错误
			if (!bRepeat)
				iterDst = pList[nKind].erase(iterDst);
			else
				iterDst++;
		}
	}

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Judge Repeat (2) %.2f"), ttRepeatCount.Stop(false) / 1000.);

	//相同位置重复不良中重复次数最多的不良超过设定的轻/重严重警报数时->禁用轻严重警报
	for (int nKind = 0; nKind < eMACHINE; nKind++)			// 2018. 09.21由于添加MDJ APP Repeat Defect,更改为少于类型1个
	{
		if (pStCurDefList->bUseChkRptDefect[nKind])
		{
			if (nMaxRepeatCount[nKind] >= nRptHArmCnt[nKind])
			{
				bIsAlarm = true;
				//nKind0:CCD中的严重警报(3000)/nKind1:工艺不良中的严重警报(3001)
				theApp.m_AlgorithmTask->VS_Send_Alarm_Occurred_To_MainPC(eAutoRun, eALARMID_CCD_DEFECT_ERROR + nKind, eALARMTYPE_HEAVY);
			}
			else
			{
				bIsAlarm = false;
			}

			//将新出现的错误添加到只剩下重叠项目的现有列表中
			//添加在merge之前排序的语法
			pList[nKind].sort();
			pStCurDefList->listCurDefInfo[nKind].sort();
			pList[nKind].merge(pStCurDefList->listCurDefInfo[nKind]);
		}
	}

	//另存为新的列表文件
	bRet = m_fnSaveRepeatDefInfo(pList);

	SAFE_DELETE(pStCurDefList);

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Judge Repeat (3) %.2f"), ttRepeatCount.Stop(false) / 1000.);

	LeaveCriticalSection(&theApp.m_csJudgeRepeatCount);

	return bRet;
}

bool AviInspection::m_fnSaveRepeatDefInfo(std::list<RepeatDefectInfo>* pListRepeatInfo)
{
	//如果没有List
	if (pListRepeatInfo == NULL)			return false;

	//保存TXT
	std::list<RepeatDefectInfo>::iterator iterList;
	CStdioFile	fileWriter;
	CString		strRepeatInfoHeader, strRepeatInfo;
	BOOL bRet = FALSE;

#if CCD
	//禁用功能时自动清除
	for (int nKind = 0; nKind < eMACHINE; nKind++)			// 2018. 09.21由于添加MDJ APP Repeat Defect,更改为少于类型1个
	{
		//打开文件
		switch (nKind)
		{
		case ePIXEL:
			bRet = fileWriter.Open(REPEAT_DEFECT_PIXEL_INFO_PATH, CFile::modeCreate | CFile::modeWrite);
			break;
		case eWORK:
			bRet = fileWriter.Open(REPEAT_DEFECT_WORK_INFO_PATH, CFile::modeCreate | CFile::modeWrite);
			break;
		default:
			bRet = FALSE;
			break;
		}
#else
	for (int nKind = 0; nKind < eMACHINE; nKind++)			// 2018. 09.21由于添加MDJ APP Repeat Defect,更改为少于类型1个
	{
		//打开文件
		switch (nKind)
		{
		case ePIXEL:
			//bRet = fileWriter.Open(REPEAT_DEFECT_PIXEL_INFO_PATH, CFile::modeCreate | CFile::modeWrite);
			break;
		case eWORK:
			//bRet = fileWriter.Open(REPEAT_DEFECT_WORK_INFO_PATH, CFile::modeCreate | CFile::modeWrite);
			break;
		default:
			bRet = FALSE;
			break;
		}
#endif

		if (bRet)
		{
			//头信息-for GUI
			strRepeatInfoHeader = _T("Type,DefectX,DefectY,RepeatCount\r\n");
			fileWriter.WriteString(strRepeatInfoHeader);
			//重复的坐标数
			for (iterList = pListRepeatInfo[nKind].begin(); iterList != pListRepeatInfo[nKind].end(); ++iterList)
			{
				//关于同位坐标
				strRepeatInfo.Format(_T("%d,%d,%d,%d\r\n"), (int)iterList->eDefType,
					iterList->ptCenterPos.x, iterList->ptCenterPos.y, iterList->nRepeatCount);

				fileWriter.SeekToEnd();
				fileWriter.WriteString(strRepeatInfo);
			}

			//仅在文件打开时关闭
			fileWriter.Close();
		}
	}

	return true;
}

bool AviInspection::NewMaxGVMethold(Mat matSrcImage, double OldMaxGV, double& NewMaxGV, int nTopCountGV)
{
	// Histogram Calculation

	cv::Mat matHisto;
	int nHistSize = 256;

	float fHistRange[] = { (float)0, (float)nHistSize - 1 };
	const float* ranges[] = { fHistRange };

	cv::calcHist(&matSrcImage, 1, 0, Mat(), matHisto, 1, &nHistSize, ranges, true, false);

	float* pVal = (float*)matHisto.data;

	// Diff x GV Calculation
	__int64 nPixelSum = 0;
	__int64 nPixelCount = 0;

	//int nTopCountGV		= 3;
	int nCountNumber = 0;
	int nCountGV = 0;

	pVal = (float*)matHisto.ptr(0) + (int)OldMaxGV;

	for (int m = (int)OldMaxGV; m >= 0; m--, pVal--)
	{
		nPixelSum = (__int64)(m * *pVal);

		if (nPixelSum != 0)
		{
			nCountGV += m;
			nCountNumber++;
			if (nCountNumber == nTopCountGV) break;
			else continue;
		}

		else continue;
	}

	NewMaxGV = (double)nCountGV / nCountNumber;

	return true;
}

bool 	AviInspection::GetModelNorchInfo(ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], vector<int>&NorchIndex, CPoint & OrgIndex)
{
	if (tRoundSet == NULL) return false;

	//检查ROI数量
	int ROICount = 0;

	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		if (tRoundSet[i].nContourCount != 0)
			ROICount++;
	}

	if (ROICount <= 4) return false;

	int TopDataLenght = tRoundSet[6].nContourCount;
	int BottomDataLenght = tRoundSet[5].nContourCount;

	if (TopDataLenght == 0 && BottomDataLenght == 0) return false;

	//原点Index
	int ValueBuff1 = m_stThrdAlignInfo.rcAlignCellROI.x;
	int ValueBuff2 = m_stThrdAlignInfo.rcAlignCellROI.y;

	int ValueBuff3 = tRoundSet[6].ptContours[TopDataLenght - 1].x;
	int ValueBuff4 = tRoundSet[6].ptContours[0].y;
	int ValueBuff5 = tRoundSet[5].ptContours[0].x;
	int ValueBuff6 = tRoundSet[5].ptContours[BottomDataLenght - 1].y;

	NorchIndex.push_back(ValueBuff3); // Top Length
	NorchIndex.push_back(ValueBuff4); // Top Height
	NorchIndex.push_back(ValueBuff5); // Bottom Length
	NorchIndex.push_back(abs(ValueBuff6)); // Bottom Height

	OrgIndex.x = ValueBuff4;
	OrgIndex.y = ValueBuff6 + m_stThrdAlignInfo.rcAlignCellROI.height;

	return true;
}

bool	AviInspection::JudgeCHoleJudgment(ResultPanelData & resultPanelData, tCHoleAlignInfo tCHoleAlignData, double* dAlignPara)
{
	//////////////////////////////////////////////////////////////////////////
	// Parameter setting
	//////////////////////////////////////////////////////////////////////////
		//>检查Chole Point时,修改Judgment删除并跳过位于Chole的所有不良行为
	bool bCholePointFlag = dAlignPara[E_PARA_CHOLE_POINT_USE] > 0 ? 1 : 0;

	if (bCholePointFlag)
		return true;

	//获取规格
	int nDelete_Defect_Area;			// 不良大小
	int nDelete_Defect_Offset;			// CHole Offset

	//////////////////////////////////////////////////////////////////////////

											//如果没有不良列表,请退出
	if (resultPanelData.m_ListDefectInfo.GetCount() <= 0)
		return true;

	int nCHoleADCnt = 0;

	//CHole AD有/无不良检查
	for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
	{
		int nDefect_Type = resultPanelData.m_ListDefectInfo[i].Defect_Type;
		if (nDefect_Type == E_DEFECT_JUDGEMENT_DISPLAY_CHOLE_ABNORMAL) nCHoleADCnt++;
	}

	//CHole AD判定
	if (nCHoleADCnt > 0)
	{
		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nDefect1_Type = resultPanelData.m_ListDefectInfo[i].Defect_Type;
			int nDefect1_ImgNo = resultPanelData.m_ListDefectInfo[i].Img_Number;

			if (nDefect1_Type != E_DEFECT_JUDGEMENT_DISPLAY_CHOLE_ABNORMAL) continue;

			//CHole区域
			CRect rectCHole_Rect;
			rectCHole_Rect.left = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
			rectCHole_Rect.top = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
			rectCHole_Rect.right = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
			rectCHole_Rect.bottom = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;

			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); j++)
			{
				//在同样的不良情况下跳过
				if (i == j) continue;

				int nDefect2_Type = resultPanelData.m_ListDefectInfo[j].Defect_Type;
				int nDefect2_ImgNo = resultPanelData.m_ListDefectInfo[j].Img_Number;

				//切换到CHole AD
				if (nDefect2_Type == E_DEFECT_JUDGEMENT_DISPLAY_CHOLE_ABNORMAL) continue;

				CRect rectDefect_Rect;
				rectDefect_Rect.left = resultPanelData.m_ListDefectInfo[j].Pixel_Start_X;
				rectDefect_Rect.top = resultPanelData.m_ListDefectInfo[j].Pixel_Start_Y;
				rectDefect_Rect.right = resultPanelData.m_ListDefectInfo[j].Pixel_End_X;
				rectDefect_Rect.bottom = resultPanelData.m_ListDefectInfo[j].Pixel_End_Y;

				CRect aatest;

				if (nDefect1_ImgNo == nDefect2_ImgNo)
				{
					if (aatest.IntersectRect(rectCHole_Rect, rectDefect_Rect))
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
						if (i > j) i--;
						j--;
					}
				}
			}
		}
	}

	//CHole二次判定
	else
	{
		int nDefect_Cnt = 0;		// Point不良
		int nDefect2_Cnt = 0;		// Line&Type 2Y不良

		//int nDefect3_Cnt = 0;			// 临时Type 2Y方向

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			int nDefect_Type = resultPanelData.m_ListDefectInfo[i].Defect_Type;

			switch (nDefect_Type)
			{
				//暗点
			case E_DEFECT_JUDGEMENT_POINT_DARK:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
			case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:
			case E_DEFECT_JUDGEMENT_POINT_GROUP_DARK:
				nDefect_Cnt++;
				break;

				// Weak Line & Type2
			case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
			case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT:

			case E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y:
				nDefect2_Cnt++;
				break;

				//nDefect3_Cnt++;
				//break;

			default:
				break;
			}
		}

		if (nDefect_Cnt > 0)//用于删除暗点
		{
			nDelete_Defect_Area = (int)dAlignPara[E_PARA_CHOLE_SMALL_DEFECT_SIZE];
			nDelete_Defect_Offset = (int)dAlignPara[E_PARA_CHOLE_SMALL_DEFECT_OFFSET];

			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{
				int nDefect_Type = resultPanelData.m_ListDefectInfo[i].Defect_Type;
				int nDefect_ImgNo = resultPanelData.m_ListDefectInfo[i].Img_Number;

				if (nDefect_Type != E_DEFECT_JUDGEMENT_POINT_DARK &&
					nDefect_Type != E_DEFECT_JUDGEMENT_POINT_DARK_SP_1 &&
					nDefect_Type != E_DEFECT_JUDGEMENT_POINT_DARK_SP_2 &&
					nDefect_Type != E_DEFECT_JUDGEMENT_POINT_DARK_SP_3 &&
					nDefect_Type != E_DEFECT_JUDGEMENT_POINT_GROUP_DARK) continue;

				//检查暗点规格
				//暗点大小
				int nDefect_Width = resultPanelData.m_ListDefectInfo[i].Pixel_End_X - resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
				int nDefect_Height = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y - resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;

				//只检查小故障
				if (nDefect_Width * nDefect_Height > nDelete_Defect_Area) continue;

				//暗点中心坐标
				int nDefect_CenX = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
				int nDefect_CenY = (int)resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

				// Offset
				cv::Rect rectDefect_Center;
				rectDefect_Center.x = nDefect_CenX - nDelete_Defect_Offset;
				rectDefect_Center.y = nDefect_CenY - nDelete_Defect_Offset;
				rectDefect_Center.width = 1 + nDelete_Defect_Offset * 2;
				rectDefect_Center.height = 1 + nDelete_Defect_Offset * 2;

				for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
				{
					if (tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j].empty()) continue;
					if (tCHoleAlignData.matCHoleROIBuf[theApp.GetImageClassify(nDefect_ImgNo)][j].empty()) continue;

					cv::Mat matCHoleCoordBuf = tCHoleAlignData.matCHoleROIBuf[theApp.GetImageClassify(nDefect_ImgNo)][j];

					rectDefect_Center.x -= tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j].x;
					rectDefect_Center.y -= tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j].y;

					//异常处理
					if (rectDefect_Center.x < 0) continue;
					if (rectDefect_Center.y < 0) continue;
					if (rectDefect_Center.x >= matCHoleCoordBuf.cols - rectDefect_Center.width)		continue;
					if (rectDefect_Center.y >= matCHoleCoordBuf.rows - rectDefect_Center.height)	continue;

					cv::Scalar scrMean = cv::mean(matCHoleCoordBuf(rectDefect_Center));

					if (scrMean[0] > 0)
					{
						resultPanelData.m_ListDefectInfo.RemoveAt(i--);
						break;
					}
				}
			}
		}

		if (nDefect2_Cnt > 0)//用于消除Type 2&Line过检
		{
			nDelete_Defect_Offset = (int)dAlignPara[E_PARA_CHOLE_BIG_DEFECT_OFFSET];
			bool bDelLineFlag = ((int)dAlignPara[E_PARA_CHOLE_LINE_DELETE_FLAG] > 0) ? 1 : 0;

			for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
			{
				int nDefect_Type = resultPanelData.m_ListDefectInfo[i].Defect_Type;
				int nDefect_ImgNo = resultPanelData.m_ListDefectInfo[i].Img_Number;

				if (nDefect_Type != E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y &&
					nDefect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT &&
					nDefect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT &&
					nDefect_Type != E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK &&
					nDefect_Type != E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK) continue;

				if ((nDefect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
					nDefect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
					nDefect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK ||
					nDefect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK) && bDelLineFlag)
				{

					int nDefect_Start_X = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					int nDefect_Start_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					int nDefect_End_X = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
					int nDefect_End_Y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;

					int nDefect_Center_X = resultPanelData.m_ListDefectInfo[i].Pixel_Center_X;
					int nDefect_Center_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Center_Y;

					for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
					{
						if (tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j].empty()) continue;

						cv::Rect rectCHoleBuf = tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j];

						int nCHole_Start_X = rectCHoleBuf.x;
						int nCHole_Start_Y = rectCHoleBuf.y;
						int nCHole_End_X = rectCHoleBuf.width + rectCHoleBuf.x;
						int nCHole_End_Y = rectCHoleBuf.height + rectCHoleBuf.y;

						if (nDefect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT ||
							nDefect_Type == E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK)
						{
							if ((nDefect_Center_Y > (nCHole_Start_Y - nDelete_Defect_Offset) && nDefect_Center_Y < (nCHole_Start_Y + nDelete_Defect_Offset)) ||
								(nDefect_Center_Y > (nCHole_End_Y - nDelete_Defect_Offset) && nDefect_Center_Y < (nCHole_End_Y + nDelete_Defect_Offset)))
							{
								resultPanelData.m_ListDefectInfo.RemoveAt(i--);
								break;
							}
						}

						else if (nDefect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT ||
							nDefect_Type == E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK)
						{
							if ((nDefect_Center_X > (nCHole_Start_X - nDelete_Defect_Offset) && nDefect_Center_X < (nCHole_Start_X + nDelete_Defect_Offset)) ||
								(nDefect_Center_X > (nCHole_End_X - nDelete_Defect_Offset) && nDefect_Center_X < (nCHole_End_X + nDelete_Defect_Offset)))
							{
								resultPanelData.m_ListDefectInfo.RemoveAt(i--);
								break;
							}
						}
					}
				}
				//Type 2位置和大小
				else if (nDefect_Type == E_DEFECT_JUDGEMENT_MURA_LINEMURA_Y)
				{
					int nDefect_Start_X = resultPanelData.m_ListDefectInfo[i].Pixel_Start_X;
					int nDefect_Start_Y = resultPanelData.m_ListDefectInfo[i].Pixel_Start_Y;
					int nDefect_End_X = resultPanelData.m_ListDefectInfo[i].Pixel_End_X;
					int nDefect_End_Y = resultPanelData.m_ListDefectInfo[i].Pixel_End_Y;
					int nDefect_Width = nDefect_End_X - nDefect_End_X;
					int nDefect_Height = nDefect_End_Y - nDefect_Start_Y;

					int nDefect_Area = nDefect_Width * nDefect_Height;

					for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
					{
						if (tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j].empty()) continue;

						cv::Rect rectCHoleBuf = tCHoleAlignData.rcCHoleROI[theApp.GetImageClassify(nDefect_ImgNo)][j];

						int nCHole_Start_X = rectCHoleBuf.x - nDelete_Defect_Offset;
						int nCHole_Start_Y = rectCHoleBuf.y - nDelete_Defect_Offset;
						int nCHole_End_X = rectCHoleBuf.width + rectCHoleBuf.x + nDelete_Defect_Offset;
						int nCHole_End_Y = rectCHoleBuf.height + rectCHoleBuf.y + nDelete_Defect_Offset;

						//仅在CHole周围
						if (nDefect_Start_X > nCHole_End_X)		continue;
						if (nDefect_End_X < nCHole_Start_X)		continue;
						if (nDefect_Start_Y > nCHole_End_Y)		continue;
						if (nDefect_End_Y < nCHole_Start_Y)		continue;

						//比较CHole Area
						int nOverLap_Start_X = (nDefect_Start_X > nCHole_Start_X) ? nDefect_Start_X : nCHole_Start_X;
						int nOverLap_Start_Y = (nDefect_Start_Y > nCHole_Start_Y) ? nDefect_Start_Y : nCHole_Start_Y;
						int nOverLap_End_X = (nDefect_End_X > nCHole_End_X) ? nCHole_End_X : nDefect_End_X;
						int nOverLap_End_Y = (nDefect_End_Y > nCHole_End_Y) ? nCHole_End_Y : nDefect_End_Y;
						int nOverLap_Width = nOverLap_End_X - nOverLap_Start_X;
						int nOverLap_Height = nOverLap_End_Y - nOverLap_Start_Y;
						int nOverLap_Area = nOverLap_Width * nOverLap_Height;

						if (nOverLap_Area >= (nDefect_Area - nOverLap_Area))
						{
							resultPanelData.m_ListDefectInfo.RemoveAt(i--);
							break;
						}
					}
				}
			}
		}
	}

	return true;
}

bool AviInspection::PrepareAITask(cv::Mat dics, double dicsRatio, int cropExpand, stDefectInfo * pResultBlob, AIReJudgeParam & aiParam,
	CString strPanelID, const int nImageNum, const int nAlgNum,
	std::vector<TaskInfoPtr>&taskList)
{
	TaskInfoPtr muraBrightTaskInfo = std::make_shared<stTaskInfo>();
	AIInfoPtr aiInfoBright = std::make_shared<STRU_AI_INFO>();
	muraBrightTaskInfo->inspParam = aiInfoBright;

	TaskInfoPtr muraDarkTaskInfo = std::make_shared<stTaskInfo>();
	AIInfoPtr aiInfoDark = std::make_shared<STRU_AI_INFO>();
	muraDarkTaskInfo->inspParam = aiInfoDark;

	for (int i = 0; i < pResultBlob->nDefectCount; i++) {
		if ((pResultBlob->nDefectJudge[i] >= E_DEFECT_JUDGEMENT_LINE_START &&
			pResultBlob->nDefectJudge[i] <= E_DEFECT_JUDGEMENT_MURANORMAL_TYPE3_F_GRADE) ||
			(pResultBlob->nDefectJudge[i] >= E_DEFECT_JUDGEMENT_MURA_NUGI &&
				pResultBlob->nDefectJudge[i] <= E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI_)) {
			break;
		}

		int lt_x = int((pResultBlob->ptLT[i].x - m_stThrdAlignInfo.ptCorner[0].x) * dicsRatio / 100) - cropExpand;
		int lt_y = int((pResultBlob->ptLT[i].y - m_stThrdAlignInfo.ptCorner[0].y) * dicsRatio / 100) - cropExpand;
		int rb_x = int((pResultBlob->ptRB[i].x - m_stThrdAlignInfo.ptCorner[0].x) * dicsRatio / 100) + cropExpand;
		int rb_y = int((pResultBlob->ptRB[i].y - m_stThrdAlignInfo.ptCorner[0].y) * dicsRatio / 100) + cropExpand;

		if (lt_x < 0) lt_x = 0;
		if (lt_y < 0) lt_y = 0;
		if (rb_x > dics.cols) rb_x = dics.cols - 1;
		if (rb_y > dics.rows) rb_y = dics.rows - 1;

		int cropped_width = rb_x - lt_x;
		int cropped_height = rb_y - lt_y;

		cv::Mat cropped_image(dics, cv::Rect(lt_x, lt_y, cropped_width, cropped_height));
		cv::resize(cropped_image, cropped_image, cv::Size(64, 64));
		cv::cvtColor(cropped_image, cropped_image, cv::COLOR_GRAY2RGB);

		if (pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT) {
			muraBrightTaskInfo->imageData.emplace_back(cropped_image);
			aiInfoBright->defectNoList.emplace_back(i);
		}
		else if (pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_AMORPH_DARK) {
			muraDarkTaskInfo->imageData.emplace_back(cropped_image);
			aiInfoDark->defectNoList.emplace_back(i);
		}
	}

	if (muraBrightTaskInfo->imageData.size() > 0) {
		aiInfoBright->algoNum = nAlgNum;
		aiInfoBright->imgNum = theApp.GetImageClassify(nImageNum);
		aiInfoBright->strPanelID = strPanelID;
		aiInfoBright->imgName = theApp.GetGrabStepName(nImageNum);
		muraBrightTaskInfo->promiseResult = new std::promise<ModelResultPtr>();
		muraBrightTaskInfo->modelId = aiParam.modelID[0];
		taskList.emplace_back(muraBrightTaskInfo);
	}
	if (muraDarkTaskInfo->imageData.size() > 0) {
		aiInfoDark->algoNum = nAlgNum;
		aiInfoDark->imgNum = theApp.GetImageClassify(nImageNum);
		aiInfoDark->strPanelID = strPanelID;
		aiInfoDark->imgName = theApp.GetGrabStepName(nImageNum);
		muraDarkTaskInfo->promiseResult = new std::promise<ModelResultPtr>();
		muraBrightTaskInfo->modelId = aiParam.modelID[1];
		taskList.emplace_back(muraDarkTaskInfo);
	}

	return true;
}

bool AviInspection::ApplyMergeRule(ResultPanelData & resultPanelData)
{
	if (resultPanelData.m_ListDefectInfo.GetCount() < 2) return true;

	float interRatio = 0.5;
	std::wstring_convert<std::codecvt_utf8_utf16<TCHAR>, TCHAR> converter;

	for (int li = 0; li < theApp.GetMergeLogicCount(); li++) {
		std::vector<std::vector<std::string>>* vLogic = theApp.GetMergeLogic(li);

		for (int i = 0; i < resultPanelData.m_ListDefectInfo.GetCount(); i++)
		{
			ResultDefectInfo* leftDef = &resultPanelData.m_ListDefectInfo[i];
			std::wstring wsDefCode1 = theApp.GetDefectTypeName(leftDef->Defect_Type);
			std::string sDefCode1 = converter.to_bytes(wsDefCode1);

			int contourCnt1 = leftDef->nContoursCount;
			if (contourCnt1 == 0) continue;

			std::vector<cv::Point> contour1;
			for (int ci = 0; ci < contourCnt1; ci++) {
				contour1.push_back({ leftDef->nContoursX[ci], leftDef->nContoursY[ci] });
			}
			std::vector<cv::Point> contour1_convex;
			cv::convexHull(contour1, contour1_convex);

			int iMove = 0;

			for (int j = 0; j < resultPanelData.m_ListDefectInfo.GetCount(); j++)
			{
				int iIndex = i + iMove;
				if (iIndex == j) continue;

				ResultDefectInfo* rightDef = &resultPanelData.m_ListDefectInfo[j];
				int contourCnt2 = rightDef->nContoursCount;
				if (contourCnt2 == 0) continue;

				std::wstring wsDefCode2 = theApp.GetDefectTypeName(rightDef->Defect_Type);
				std::string sDefCode2 = converter.to_bytes(wsDefCode2);

				std::vector<cv::Point> contour2;
				for (int ci = 0; ci < contourCnt2; ci++) {
					contour2.push_back({ rightDef->nContoursX[ci], rightDef->nContoursY[ci] });
				}

				std::vector<cv::Point> contour2_convex;
				cv::convexHull(contour2, contour2_convex);

				cv::Mat intersection;
				cv::intersectConvexConvex(contour1, contour2, intersection);
				//如果轮廓之间无交集 则intersection为空
				if (intersection.empty()) continue;

				double interArea = cv::contourArea(intersection);
				double area1 = cv::contourArea(contour1);
				double area2 = cv::contourArea(contour2);
				if (interArea / area1 < interRatio && interArea / area2 < interRatio) {
					continue;
				}

				// 应用merge规则
				int jIdx = -1;
				int iIdx = -1;
				for (int rj = 0; rj < vLogic->size(); rj++)
				{
					if ((*vLogic)[rj][0] == sDefCode1)
					{
						jIdx = rj;
						break;
					}
				}

				for (int ri = 0; ri < (*vLogic)[0].size(); ri++)
				{
					if ((*vLogic)[0][ri] == sDefCode2)
					{
						iIdx = ri;
						break;
					}
				}

				if (iIdx == -1 || jIdx == -1) {
					continue;
				}

				std::string strMerge = (*vLogic)[jIdx][iIdx];

				if (strMerge == "-") // "-", 都不要
				{
					resultPanelData.m_ListDefectInfo.RemoveAt(iIndex);
					if (iIndex > j) {
						resultPanelData.m_ListDefectInfo.RemoveAt(j);
						iMove--;
					}
					else {
						resultPanelData.m_ListDefectInfo.RemoveAt(j - 1);
					}
					iMove--;
					break;
				}
				else if (strMerge == "L") { // "L", 保留Left
					resultPanelData.m_ListDefectInfo.RemoveAt(j);
					if (iIndex > j) iMove--;
					j--;
				}
				else if (strMerge == "T") { // "T", 保留Top
					resultPanelData.m_ListDefectInfo.RemoveAt(iIndex);
					iMove--;
					break;
				}
				else if (strMerge == "A") { // "A", 都保留
					// Do Nothing
				}

				if (resultPanelData.m_ListDefectInfo.GetCount() < 2) {
					break;
				}
			} // End for j

			if (resultPanelData.m_ListDefectInfo.GetCount() < 2) {
				break;
			}
			i += iMove;
		} // End for i

		if (resultPanelData.m_ListDefectInfo.GetCount() < 2) {
			break;
		}
	} // End for li
	return true;
}
