#include "stdafx.h"
#include "AlgorithmInterface.h"
#include "VSAlgorithmTask.h"
#include "../../visualstation/CommonHeader/Class/CostTime.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"


//并行处理
#include <omp.h>
//#include <ppl.h>
//using namespace Concurrency;

//使用Align_RotateImage()
#include "DllInterface.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#define new DEBUG_NEW
#endif

using namespace Concurrency;

InspectAlgorithmInterface::InspectAlgorithmInterface()
{
}

InspectAlgorithmInterface::~InspectAlgorithmInterface(void)
{
}

void InspectAlgorithmInterface::DrawAdjustROI(cv::Mat& MatDrawBuffer, cv::Point* pPtCorner, LPCTSTR strROIName, int nCurCount, int nDrawMode)
{
	CString strNumber;

	if (!MatDrawBuffer.empty())
	{
		cv::Scalar color;
		int nThickness = 1;	// Rectangle厚度

		// Teaching Draw Rect
		if (nDrawMode == eBasicROI)
		{
			color.val[0] = 0;	color.val[1] = 255;	color.val[2] = 0;
		}	//设置ROI
		else if (nDrawMode == eRndROI)
		{
			color.val[0] = 255;	color.val[1] = 0;	color.val[2] = 0;
		}	// Pol ROI
		else
		{
			color.val[0] = 0;	color.val[1] = 255;	color.val[2] = 255;
		}	//无检查区域ROI

//矩形->更改矩形
//cv::rectangle(MatDrawBuffer, cv::Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top), color, nThickness);
		cv::line(MatDrawBuffer, pPtCorner[E_CORNER_LEFT_TOP], pPtCorner[E_CORNER_RIGHT_TOP], color, nThickness);
		cv::line(MatDrawBuffer, pPtCorner[E_CORNER_RIGHT_TOP], pPtCorner[E_CORNER_RIGHT_BOTTOM], color, nThickness);
		cv::line(MatDrawBuffer, pPtCorner[E_CORNER_RIGHT_BOTTOM], pPtCorner[E_CORNER_LEFT_BOTTOM], color, nThickness);
		cv::line(MatDrawBuffer, pPtCorner[E_CORNER_LEFT_BOTTOM], pPtCorner[E_CORNER_LEFT_TOP], color, nThickness);

		// Text
		strNumber.Format(_T("%d,%s,(%d,%d),(%d,%d),(%d,%d),(%d,%d)"), nCurCount + 1, strROIName,
			pPtCorner[E_CORNER_LEFT_TOP].x, pPtCorner[E_CORNER_LEFT_TOP].y,
			pPtCorner[E_CORNER_RIGHT_TOP].x, pPtCorner[E_CORNER_RIGHT_TOP].y,
			pPtCorner[E_CORNER_RIGHT_BOTTOM].x, pPtCorner[E_CORNER_RIGHT_BOTTOM].y,
			pPtCorner[E_CORNER_LEFT_BOTTOM].x, pPtCorner[E_CORNER_LEFT_BOTTOM].y);
		char* pTemp = CSTR2PCH(strNumber);
		cv::putText(MatDrawBuffer, pTemp, cv::Point(pPtCorner[E_CORNER_LEFT_TOP].x - 5, pPtCorner[E_CORNER_LEFT_TOP].y - 5), FONT_HERSHEY_SIMPLEX, 1., color, 1);
		SAFE_DELETE_ARR(pTemp);
	}
}

//----------------------------------------------------------------------------------------------------
//	 1. 函数名:	LogicStart
//	 2. 函数功能:	检查一个单元格
//	 3. 收购:	MatOriginImage		:原始画面
//					MatDrawBuffer		:结果Draw图像
//					MatResultImg		:算法结果图像
//					resultBlob_Total	:按模式聚合所有算法结果Buffer
//	 4. 返回值:	完成:true
//				失败:false
//	 5. 创建日期:2015/02/07
//	 6. 作者:	
//	 7. 修改历史记录:2017/03/02 by CWH
//	 8. 注意:	
//----------------------------------------------------------------------------------------------------
long InspectAlgorithmInterface::StartLogic(CString strPanelID, CString strDrive, tAlignInfo stThrdAlignInfo,
	cv::Mat MatOriginImage[][MAX_CAMERA_COUNT], cv::Mat& MatDrawBuffer, cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE],
	ResultBlob_Total* pResultBlob_Total, const int nImageNum, const int nCameraNum, const int nThreadIndex, bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtResultInfo, const double* _mtp, STRU_LabelMarkInfo& labelMarkInfo)
{
	//////////////////////////////////////////////////////////////////////////
		//设置参数
	//////////////////////////////////////////////////////////////////////////
	tLogicPara LogicPara;

	LogicPara.MatOrgImage = MatOriginImage[nImageNum][nCameraNum];
	LogicPara.nCameraNum = nCameraNum;
	LogicPara.nThreadLog = nThreadIndex;
	const int nEqpType = theApp.m_Config.GetEqpType();
	int TimeCount = 0;

	switch (nEqpType)
	{
		//////////////////////////////////////////////////////////////////////////
				//将RGB模式临时复制到最后3个图像缓冲区。
				//如果makePolygonCellROI,图像将被破坏,在此之前复制原始图像。
		//////////////////////////////////////////////////////////////////////////
	case EQP_AVI:

		//		//等待RGB拍摄完毕
		// 		TimeCount = 0;
		// 		while(	MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_R)][nCameraNum].empty()	||
		// 				MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_G)][nCameraNum].empty()	||
		// 				MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_B)][nCameraNum].empty()	)
		// 		{
		// 			TimeCount++;
		// 			Sleep(10);
		// 
		// 			if (TimeCount >= 1000)
		// 				return E_ERROR_CODE_FALSE;
		// 		}

		switch (theApp.GetImageClassify(nImageNum))
		{
		case E_IMAGE_CLASSIFY_AVI_R:
			MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_R)][nCameraNum].copyTo(MatOriginImage[MAX_GRAB_STEP_COUNT - 3][nCameraNum]);		//完整的内存分配画面
			break;
		case E_IMAGE_CLASSIFY_AVI_G:
			MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_G)][nCameraNum].copyTo(MatOriginImage[MAX_GRAB_STEP_COUNT - 2][nCameraNum]);		//完整的内存分配画面
			break;
		case E_IMAGE_CLASSIFY_AVI_B:
			MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_AVI_B)][nCameraNum].copyTo(MatOriginImage[MAX_GRAB_STEP_COUNT - 1][nCameraNum]);		//完整的内存分配画面
			break;
		}

		//		//复制RGB之前
		// 		TimeCount = 0;
		// 		while(	MatOriginImage[MAX_GRAB_STEP_COUNT - 3][nCameraNum].empty()	||
		// 				MatOriginImage[MAX_GRAB_STEP_COUNT - 2][nCameraNum].empty()	||
		// 				MatOriginImage[MAX_GRAB_STEP_COUNT - 1][nCameraNum].empty()	)
		// 		{
		// 			TimeCount++;
		// 			Sleep(10);
		// 
		// 			// 18.05.30
		// 			if (TimeCount >= 1000)
		// 				return E_ERROR_CODE_EMPTY_RGB;
		// 		}

		//		//准备把RGB模式交给算法。索引一起使用相同。=>如果以后出错,需要更改
		// 		LogicPara.MatOrgRGB[0]		= MatOriginImage[MAX_GRAB_STEP_COUNT - 3][nCameraNum];
		// 		LogicPara.MatOrgRGB[1]		= MatOriginImage[MAX_GRAB_STEP_COUNT - 2][nCameraNum];
		// 		LogicPara.MatOrgRGB[2]		= MatOriginImage[MAX_GRAB_STEP_COUNT - 1][nCameraNum];

		LogicPara.MatOrgRGBAdd[0] = &MatOriginImage[MAX_GRAB_STEP_COUNT - 3][nCameraNum];
		LogicPara.MatOrgRGBAdd[1] = &MatOriginImage[MAX_GRAB_STEP_COUNT - 2][nCameraNum];
		LogicPara.MatOrgRGBAdd[2] = &MatOriginImage[MAX_GRAB_STEP_COUNT - 1][nCameraNum];
		break;

	case EQP_SVI:
		break;

	case EQP_APP:
		// No Dust Image. 20230323.xb
		//TimeCount = 0;
		//while(	MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_APP_DUST)][nCameraNum].empty()	)
		//{
		//	TimeCount++;
		//	Sleep(10);

		//	if (TimeCount >= 1000)
		//		return E_ERROR_CODE_FALSE;
		//}
		//LogicPara.MatDust		= MatOriginImage[theApp.GetImageNum(E_IMAGE_CLASSIFY_APP_DUST)][nCameraNum];
		break;

	default:
		break;
	}

	//面板ID
	int nLength = 0;
	nLength = _stprintf_s(LogicPara.tszPanelID, MAX_PATH, _T("%s"), (LPCWSTR)strPanelID);
	LogicPara.tszPanelID[nLength] = _T('\0');
	LogicPara.strPanelID.Format(_T("%s"), strPanelID);

	//////////////////////////////////////////////////////////////////////////
		//影像和ROI校正
	//////////////////////////////////////////////////////////////////////////

		//在上一步中检查
		//	//如果没有缓冲区。/不可检查的错误代码&日志输出
	// 	if( MatOriginImage[nImageNum][nCameraNum].empty() )
	// 	{
	// 		// Error Log
	// 		return E_ERROR_CODE_EMPTY_BUFFER;
	// 	}	

		//获取教学中当前ROI设置的数量
	//int nROICnt = theApp.GetROICnt(nImageNum, nCameraNum);

		//每个映像固定一个ROI
	int nROICnt = 1;

	//如果没有设置ROI数量,则不检查。
	//如果有错误,则输出错误代码&日志
	if (nROICnt <= 0)
	{
		return E_ERROR_CODE_EMPTY_SET_ROI;
	}

	//////////////////////////////////////////////////////////////////////////
		//用于Cell Edge部分处理(不使用APP)
	//////////////////////////////////////////////////////////////////////////
	double* dAlgPara;
	dAlgPara = theApp.GetAlignParameter(nCameraNum);
	ENUM_KIND_OF_LOG eLogCamera;
	long nRoundErrorCode = E_ERROR_CODE_TRUE;
	DICS_B11 DICS;

	//AVI PAD区添加
	//2024.05.07 for develop
	//CString strDirectoryPath = DICS.CheckDirectory(INIT_FILE_PATH, strPanelID, strDrive);
	//DICS.Preprocessing_AVI_DUST_PAD(LogicPara.MatOrgImage, dAlgPara, theApp.GetGrabStepName(nImageNum), strDirectoryPath);
	switch (nEqpType)
	{
	case EQP_AVI:
	{
		//内存分配
		LogicPara.MatBKG = cv::Mat::zeros(LogicPara.MatOrgImage.size(), LogicPara.MatOrgImage.type()); //每个内存分配模式的完整画面

		//获取Align算法检查参数

		//17.10.24[Round]-获取注册坐标
		nRoundErrorCode = makePolygonCellROI(LogicPara, MatDrawBuffer, stThrdAlignInfo, labelMarkInfo, nImageNum, nCameraNum, dAlgPara, theApp.GetImageClassify(nImageNum), nRatio);

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			if (stThrdAlignInfo.tCHoleAlignData->bCHoleAD[theApp.GetImageClassify(nImageNum)][i])
			{
				if (stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].empty()) continue;
				stDefectInfo* pCHole = new stDefectInfo(2, nImageNum);

				if (pCHole != NULL)
				{
					pCHole->nArea[0] = 0;
					pCHole->nMaxGV[0] = 255;
					pCHole->nMinGV[0] = 0;
					pCHole->dMeanGV[0] = 0;

					pCHole->ptLT[0].x = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].x;
					pCHole->ptLT[0].y = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].y;

					pCHole->ptRT[0].x = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].x + stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].width;
					pCHole->ptRT[0].y = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].y;

					pCHole->ptRB[0].x = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].x + stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].width;
					pCHole->ptRB[0].y = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].y + stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].height;

					pCHole->ptLB[0].x = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].x;
					pCHole->ptLB[0].y = stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].y + stThrdAlignInfo.tCHoleAlignData->rcCHoleROI[theApp.GetImageClassify(nImageNum)][i].height;

					pCHole->dBackGroundGV[0] = 0;
					pCHole->dCompactness[0] = 0;
					pCHole->dSigma[0] = 0;
					pCHole->dBreadth[0] = 0;
					pCHole->dF_Min[0] = 0;
					pCHole->dF_Max[0] = 0;
					pCHole->dF_Elongation[0] = 0;
					pCHole->dCompactness[0] = 0;

					//亮度
					pCHole->nDefectColor[0] = E_DEFECT_COLOR_DARK;

					pCHole->nDefectJudge[0] = E_DEFECT_JUDGEMENT_DISPLAY_CHOLE_ABNORMAL;
					pCHole->nPatternClassify[0] = nImageNum;

					//计数增加
					pCHole->nDefectCount = 1;
				}
				pResultBlob_Total->AddTail_ResultBlob(pCHole);
			}
		}

		//保存Back Ground Image
		if (theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_WHITE)
		{
			LogicPara.MatBKG.copyTo(MatOriginImage[MAX_GRAB_STEP_COUNT - 4][nCameraNum]); //完整的内存分配画面
		}

		//保存Back Ground Image
		if (theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_BLACK || theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_PCD || theApp.GetImageClassify(nImageNum) == E_IMAGE_CLASSIFY_AVI_VINIT)
		{
			TimeCount = 0;
			while (MatOriginImage[MAX_GRAB_STEP_COUNT - 4][nCameraNum].empty())
			{
				TimeCount++;
				Sleep(10);

#ifdef _DEBUG	//Debug
#else			//Release
				if (TimeCount >= 1500)
					return E_ERROR_CODE_FALSE;
#endif
			}

			while (cv::mean(MatOriginImage[MAX_GRAB_STEP_COUNT - 4][nCameraNum])[0] == 0)
			{
				Sleep(10);
			}

			Sleep(10);
			MatOriginImage[MAX_GRAB_STEP_COUNT - 4][nCameraNum].copyTo(LogicPara.MatBKG);
		}
	}
	break;

	case EQP_SVI:
	{
		//SVI内存分配(8 bit)
		LogicPara.MatBKG = cv::Mat::zeros(LogicPara.MatOrgImage.size(), CV_8UC1);

		//获取Align算法检查参数
//dAlgPara = theApp.GetAlignParameter(nCameraNum);

			//外围处理
		nRoundErrorCode = makePolygonCellROI(LogicPara, MatDrawBuffer, stThrdAlignInfo, labelMarkInfo, nImageNum, nCameraNum, dAlgPara, theApp.GetImageClassify(nImageNum), nRatio);
	}
	break;

	case EQP_APP:
	{
		//获取Align算法检查参数
//dAlgPara = theApp.GetAlignParameter(nCameraNum);

		if (dAlgPara[E_PARA_APP_MAKE_ACTIVE_MASK])
		{
			nRoundErrorCode = E_ERROR_CODE_ALIGN_ROUND_SETTING;
		}
		else
		{
			nRoundErrorCode = E_ERROR_CODE_TRUE;
		}
	} break;
	}

	//添加B11DICS
	//不检查Round设置标志On

	cv::Mat matROI;
	if (nRoundErrorCode != E_ERROR_CODE_ALIGN_ROUND_SETTING)
	{
		//DICS.DICSStart(LogicPara.MatDics, LogicPara.MatOrgImage, stThrdAlignInfo.ptCorner, nEqpType, INIT_FILE_PATH, strPanelID, theApp.GetGrabStepName(nImageNum), nCameraNum, dAlgPara, strDrive);
		//DICS.Generate(LogicPara.MatOrgImage, LogicPara.MatDics, matROI, stThrdAlignInfo.ptCorner, nEqpType, INIT_FILE_PATH, strPanelID, theApp.GetGrabStepName(nImageNum), nCameraNum, dAlgPara, strDrive);
		//DICS.SaveStart(matROI, LogicPara.MatDics, INIT_FILE_PATH, strPanelID, theApp.GetGrabStepName(nImageNum), nEqpType, nCameraNum, strDrive);
	}

	// Alg DLL Success Log
	eLogCamera = (ENUM_KIND_OF_LOG)nCameraNum;

	//不检查Round设置标志On
	if (nRoundErrorCode == E_ERROR_CODE_ALIGN_ROUND_SETTING)
	{
		theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<创建ROI区域>成功! 算法跳过."), strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));

		//未执行检查逻辑
		return E_ERROR_CODE_TRUE;
	}

	//发生错误代码时Log
	else if (nRoundErrorCode != E_ERROR_CODE_TRUE)
	{
		theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<创建ROI区域>失败! 算法跳过. 错误码:%d."), strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum), nRoundErrorCode);

		//未执行检查逻辑
		return E_ERROR_CODE_TRUE;
	}

	//如果操作没有错误
	theApp.WriteLog(eLogCamera, eLOGLEVEL_DETAIL, TRUE, FALSE, _T("(%s)[%02d][%s]<ROI区域>成功!"), strPanelID, nCameraNum, theApp.GetGrabStepName(nImageNum));

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//算法函数行为
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//获取在Teaching中设置的Algorithm List数量
	int nAlgCnt = MAX_MEM_SIZE_E_ALGORITHM_NUMBER;

	//在Teaching中设置的ROI数(1个AVI......)
// #ifndef _DEBUG
// #pragma omp parallel for
// #endif
	for (int nROINum = 0; nROINum < nROICnt; nROINum++)
	{
		//ROI名称当前已禁用
		//绘制检查ROI
		DrawAdjustROI(MatDrawBuffer, stThrdAlignInfo.ptCorner, /*theApp.GetROIName(nImageNum, nCameraNum, nROINum)*/_T("ROI"), nROINum, eBasicROI);

		//在Teaching中设置的Algorithm List数量。

#ifndef _DEBUG
						//在ini中临时添加数据读取190218YWS
		int nNested_offset = GetPrivateProfileInt(_T("InspInfo"), _T("Nested offset"), 1, VS_ALGORITHM_TASK_INI_FILE);
		int nThread_Count = GetPrivateProfileInt(_T("InspInfo"), _T("Thread Count"), 1, VS_ALGORITHM_TASK_INI_FILE);

		omp_set_nested(nNested_offset); // 现有1
#pragma omp parallel for schedule(dynamic)num_threads(nThread_Count)//现有4

#endif
		for (int nAlgNum = 0; nAlgNum < nAlgCnt; nAlgNum++)
		{
			//在Teaching中确认选择Algorithm List。
			if (theApp.GetUseAlgorithm(nImageNum, LogicPara.nCameraNum, nROINum, nAlgNum))
			{
				CostTime algoCt(true);

				long nErrorCode = StartLogicAlgorithm(strDrive, LogicPara, MatResultImg, MatDrawBuffer, nImageNum, nROINum, nAlgNum, stThrdAlignInfo, pResultBlob_Total, bpInspectEnd, nRatio, eInspMode, WrtResultInfo, _mtp);

				LogSendToUI::getInstance()->SendAlgoLog(EModuleType::ALGO, ELogLevel::INFO_, EAlgoInfoType::ALGO_INSP_END,
					algoCt.get_cost_time(), 0, LogicPara.strPanelID, theApp.m_Config.GetPCNum(), nImageNum, nAlgNum,
					_T("[%s]<%s>算法检测结束，耗时:%d(ms) PID:%s"), theApp.GetGrabStepName(nImageNum), theApp.GetAlgorithmName(nAlgNum), algoCt.get_cost_time(), LogicPara.strPanelID);
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}

//----------------------------------------------------------------------------------------------------
//	 1. 函数名:	Blob Feature Save
//	 2. 函数功能:保存	Defect结果文件
//	 3. 参数:	resultBlob		:关于Blob
//				strPath			:存储路径
//					nThreadID		:线程ID
//	 4. 返回值:	
//	 5. 创建日期:2015/03/27
//	 6. 作者:	
//	 7. 修改历史记录:
//	 8. 注意:
//----------------------------------------------------------------------------------------------------
void InspectAlgorithmInterface::BlobFeatureSave(stDefectInfo* resultBlob, CString strPath, int* nImageDefectCount)
{
	if (resultBlob == NULL)	return;

	int	nDefectNum = 0;

	//保存结果日志
	FILE* out = NULL;

	if (nImageDefectCount == NULL)
		nImageDefectCount = &nDefectNum;

	if (*nImageDefectCount == 0)
		_wfopen_s(&out, strPath, _T("wt"));
	else
		_wfopen_s(&out, strPath, _T("at"));

	//异常处理。
	if (out == NULL)		return;

	//在填写多个ROI结果时,仅在第一次填写Header
	if (*nImageDefectCount == 0)
	{
		fprintf_s(out, "No					,\
 						Defect_Judge		,\
 						Defect_Color		,\
 						Area				,\
						ptLT_X				,\
						ptLT_Y				,\
						ptRT_X				,\
 						ptRT_Y				,\
						ptRB_X				,\
						ptRB_Y				,\
						ptLB_X				,\
						ptLB_Y				,\
 						Mean_GV				,\
 						Sigma				,\
 						Min_GV				,\
 						Max_GV				,\
 						BackGround_GV		,\
						Center_X			,\
						Center_Y			,\
 						Breadth				,\
 						Compactness			,\
						Roundness			,\
 						F_Elongation		,\
 						F_Min				,\
 						F_Max				,\
						MuraObj				,\
						MuraBk				,\
						MuraActive			,\
						MuraBright			,\
						MeanAreaRatio		,\
						In_Count			,\
						Judge_GV			,\
						Area_Per			,\
 						Lab_Avg_L			,\
 						Lab_Avg_a			,\
 						Lab_Avg_b			,\
 						Lab_diff_L			,\
						Lab_diff_a			,\
						Lab_diff_b			,\
						Use_Report			,,");
#if USE_ALG_HIST
		//17.06.24对象直方图
		for (int m = 0; m < IMAGE_MAX_GV; m++)
		{
			fprintf_s(out, "%d,", m);
		}
#endif
		fprintf_s(out, "\n");
	}

	//17.07.27程序下载修复
	char szPath[MAX_PATH] = { 0, };
	for (int nFori = 0; nFori < resultBlob->nDefectCount; nFori++)
	{
		//17.07.27程序下载修复
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, theApp.GetDefectTypeName(resultBlob->nDefectJudge[nFori]), -1, szPath, sizeof(szPath), NULL, NULL);

		//17.07.27不良数量增多会导致程序死机
//USES_CONVERSION;
//char *cstrName = W2A( theApp.GetDefectTypeName(resultBlob->nDefectJudge[nFori]) );

		fprintf_s(out, "%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%d,%d,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%s,%s,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%s,,",
			(*nImageDefectCount)++,
			szPath,
			resultBlob->nDefectColor[nFori],
			resultBlob->nArea[nFori],
			resultBlob->ptLT[nFori].x,
			resultBlob->ptLT[nFori].y,
			resultBlob->ptRT[nFori].x,
			resultBlob->ptRT[nFori].y,
			resultBlob->ptRB[nFori].x,
			resultBlob->ptRB[nFori].y,
			resultBlob->ptLB[nFori].x,
			resultBlob->ptLB[nFori].y,
			resultBlob->dMeanGV[nFori],
			resultBlob->dSigma[nFori],
			resultBlob->nMinGV[nFori],
			resultBlob->nMaxGV[nFori],
			resultBlob->dBackGroundGV[nFori],
			resultBlob->nCenterx[nFori],
			resultBlob->nCentery[nFori],
			resultBlob->dBreadth[nFori],
			resultBlob->dCompactness[nFori],
			resultBlob->dRoundness[nFori],
			resultBlob->dF_Elongation[nFori],
			resultBlob->dF_Min[nFori],
			resultBlob->dF_Max[nFori],
			resultBlob->dMuraObj[nFori],
			resultBlob->dMuraBk[nFori],
			resultBlob->bMuraActive[nFori] ? "Y" : "N",
			resultBlob->bMuraBright[nFori] ? "Y" : "N",
			resultBlob->dF_MeanAreaRatio[nFori],
			resultBlob->nIn_Count[nFori],
			resultBlob->nJudge_GV[nFori],
			resultBlob->dF_AreaPer[nFori],
			resultBlob->Lab_avg_L[nFori],
			resultBlob->Lab_avg_a[nFori],
			resultBlob->Lab_avg_b[nFori],
			resultBlob->Lab_diff_L[nFori],
			resultBlob->Lab_diff_a[nFori],
			resultBlob->Lab_diff_b[nFori],
			resultBlob->bUseResult[nFori] ? "Y" : "N");

#if USE_ALG_HIST
		//17.06.24对象直方图
		for (int m = 0; m < IMAGE_MAX_GV; m++)
		{
			fprintf_s(out, "%d,", resultBlob->nHist[nFori][m]);
		}
#endif

		fprintf_s(out, "\n");
	}

	fclose(out);
	out = NULL;
}

void InspectAlgorithmInterface::AlgoFeatureSave(stDefectInfo* resultBlob, CString strPath, CString strPanelID, int nImageNum, CString strAlgorithmName, int nStageNum, int* nDefectCount)
{
	if (resultBlob == NULL)	return;

	FILE* out = NULL;

	if (nImageNum == 0)
		_wfopen_s(&out, strPath, _T("wt"));
	else
		_wfopen_s(&out, strPath, _T("at"));

	if (out == NULL)		return;

	/**
	 * 	在填写多个ROI结果时，仅在第一张画面填写Header hjf
	 *
	 * \param resultBlob
	 * \param strPath
	 * \param strPanelID
	 * \param nImageNum
	 * \param strAlgorithmName
	 */
	if (nImageNum == 0)
	{
		fprintf_s(out, "ImageNum			,\
 						strAlgorithmName	,\
						stageNumber			,\
 						Defect_Judge		,\
 						Defect_Color		,\
 						Area				,\
						ptLT_X				,\
						ptLT_Y				,\
						ptRT_X				,\
 						ptRT_Y				,\
						ptRB_X				,\
						ptRB_Y				,\
						ptLB_X				,\
						ptLB_Y				,\
 						Mean_GV				,\
 						Sigma				,\
 						Min_GV				,\
 						Max_GV				,\
 						BackGround_GV		,\
						Center_X			,\
						Center_Y			,\
 						Breadth				,\
 						Compactness			,\
						Roundness			,\
 						F_Elongation		,\
 						F_Min				,\
 						F_Max				,\
						MuraObj				,\
						MuraBk				,\
						MuraActive			,\
						MuraBright			,\
						MeanAreaRatio		,\
						In_Count			,\
						Judge_GV			,\
						Area_Per			,\
 						Lab_Avg_L			,\
 						Lab_Avg_a			,\
 						Lab_Avg_b			,\
 						Lab_diff_L			,\
						Lab_diff_a			,\
						Lab_diff_b			,\
						Use_Report			,,");

		fprintf_s(out, "\n");
	}

	char szPath[MAX_PATH] = { 0, };

	char szImgName[MAX_PATH] = { 0, };
	char szAlgName[MAX_PATH] = { 0, };
	memset(szImgName, 0, sizeof(char) * MAX_PATH);
	memset(szAlgName, 0, sizeof(char) * MAX_PATH);
	WideCharToMultiByte(CP_ACP, 0, theApp.GetGrabStepName(nImageNum), -1, szImgName, sizeof(szImgName), NULL, NULL);
	WideCharToMultiByte(CP_ACP, 0, strAlgorithmName, -1, szAlgName, sizeof(szAlgName), NULL, NULL);

	for (int nFori = 0; nFori < resultBlob->nDefectCount; nFori++)
	{
		memset(szPath, 0, sizeof(char) * MAX_PATH);

		WideCharToMultiByte(CP_ACP, 0, theApp.GetDefectTypeName(resultBlob->nDefectJudge[nFori]), -1, szPath, sizeof(szPath), NULL, NULL);
		fprintf_s(out, "%s,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%d,%d,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%s,%s,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%s,,",
			szImgName,
			szAlgName,
			nStageNum,
			szPath,
			resultBlob->nDefectColor[nFori],
			resultBlob->nArea[nFori],
			resultBlob->ptLT[nFori].x,
			resultBlob->ptLT[nFori].y,
			resultBlob->ptRT[nFori].x,
			resultBlob->ptRT[nFori].y,
			resultBlob->ptRB[nFori].x,
			resultBlob->ptRB[nFori].y,
			resultBlob->ptLB[nFori].x,
			resultBlob->ptLB[nFori].y,
			resultBlob->dMeanGV[nFori],
			resultBlob->dSigma[nFori],
			resultBlob->nMinGV[nFori],
			resultBlob->nMaxGV[nFori],
			resultBlob->dBackGroundGV[nFori],
			resultBlob->nCenterx[nFori],
			resultBlob->nCentery[nFori],
			resultBlob->dBreadth[nFori],
			resultBlob->dCompactness[nFori],
			resultBlob->dRoundness[nFori],
			resultBlob->dF_Elongation[nFori],
			resultBlob->dF_Min[nFori],
			resultBlob->dF_Max[nFori],
			resultBlob->dMuraObj[nFori],
			resultBlob->dMuraBk[nFori],
			resultBlob->bMuraActive[nFori] ? "Y" : "N",
			resultBlob->bMuraBright[nFori] ? "Y" : "N",
			resultBlob->dF_MeanAreaRatio[nFori],
			resultBlob->nIn_Count[nFori],
			resultBlob->nJudge_GV[nFori],
			resultBlob->dF_AreaPer[nFori],
			resultBlob->Lab_avg_L[nFori],
			resultBlob->Lab_avg_a[nFori],
			resultBlob->Lab_avg_b[nFori],
			resultBlob->Lab_diff_L[nFori],
			resultBlob->Lab_diff_a[nFori],
			resultBlob->Lab_diff_b[nFori],
			resultBlob->bUseResult[nFori] ? "Y" : "N");

		fprintf_s(out, "\n");
	}

	fclose(out);
	out = NULL;
}

//////////////////////////////////////////////////////////////////////////
//对于AVI-不旋转。(仅接收顶点坐标ptCorner)
//SVI需要交换
//对于APP-旋转(Align_RotateImage)后,必须更改Pad/Active/Edge ROI。
//			设置距离最外框->检查ROI设置
//				如果以后Burr有很大的不良,Auto可能不行。(需要审查)
//////////////////////////////////////////////////////////////////////////	
long InspectAlgorithmInterface::AcquireAutoRoiAndAlignImage(CString strPanelID, CString strDrive, cv::Mat& MatOrgImage, int nRatio, int nImageNum, int nCameraNum, tAlignInfo& stCamAlignInfo, double* dResult, double dCamResolution, double dPannelSizeX, double dPannelSizeY)
{
	// test
	CCPUTimer tact;
	tact.Start();
	tAlignInfo* pStCamAlignInfo = new tAlignInfo();

	//获取Align算法检查参数
	//要单独定义Align Parameter。ToDo.
	double* dAlgPara = theApp.GetAlignParameter(nCameraNum);

	// Cell ID
	wchar_t wstrID[MAX_PATH] = { 0, };
	swprintf(wstrID, _T("%s"), (LPCWSTR)strPanelID);

	//因为只有0号图像是虚线的,所以改变了虚线的位置
	//现有APP正在将套路图像转交给扫描线程
	//17.07.10添加Cell中心坐标
	//17.09.07添加EQType分类参数
	//增加SCJ 18.08.03 Cam Resolution和Pannel实测尺寸
	int nEQType = theApp.m_Config.GetEqpType();
	// 得到的ptCorner是AA区最小外接矩形4个点
	long nErrorCode = Align_FindActive(MatOrgImage, dAlgPara, pStCamAlignInfo->dAlignTheta, pStCamAlignInfo->ptCorner, pStCamAlignInfo->ptContCorner, nCameraNum, nEQType, dCamResolution, dPannelSizeX, dPannelSizeY, nRatio, pStCamAlignInfo->ptCellCenter, wstrID);

	//Align是否有画面P/S模式
	pStCamAlignInfo->nRatio = nRatio;

	//图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)
	//根据PS画面,更改为单杆坐标
	pStCamAlignInfo->ptCellCenter.x /= nRatio;
	pStCamAlignInfo->ptCellCenter.y /= nRatio;

	//获取计算的角度
	dResult[0] = pStCamAlignInfo->dAlignTheta;

	//y轴Pixel差异
	dResult[1] = abs(pStCamAlignInfo->ptCorner[E_CORNER_LEFT_TOP].y - pStCamAlignInfo->ptCorner[E_CORNER_RIGHT_TOP].y);

	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Find Active : %.2f"), tact.Stop(false) / 1000.);

	//如果有错误,则输出错误代码&日志
	//if(nErrorCode==E_ERROR_CODE_TRUE)	//即使是Warning,也会进行检查
	{
		//旋转中心
		pStCamAlignInfo->ptAlignCenter.x = (int)(MatOrgImage.cols / 2);
		pStCamAlignInfo->ptAlignCenter.y = (int)(MatOrgImage.rows / 2);

		cv::Point ptAdjCorner[E_CORNER_END] = { (0, 0), };

		//ptCorner校正->ptAdjCorner
		AdjustAlignInfo(pStCamAlignInfo, ptAdjCorner);

		//yuxuefei Corner坐标旋转
		if (dAlgPara[E_PARA_AVI_Rotate_Image] > 0)
		{
			double	dTheta = -pStCamAlignInfo->dAlignTheta;

			if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
			if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;
			double PI = acos(-1.);
			dTheta *= PI;
			dTheta /= 180.0;
			double	dSin = sin(dTheta);
			double	dCos = cos(dTheta);
			double	dSin_ = sin(-dTheta);
			double	dCos_ = cos(-dTheta);
			int		nCx = MatOrgImage.cols / 2;
			int		nCy = MatOrgImage.rows / 2;

			for (int i = 0; i < E_CORNER_END; i++)
			{
				pStCamAlignInfo->ptCorner[i].x = (int)(dCos * (pStCamAlignInfo->ptCorner[i].x - nCx) - dSin * (pStCamAlignInfo->ptCorner[i].y - nCy) + nCx);
				pStCamAlignInfo->ptCorner[i].y = (int)(dSin * (pStCamAlignInfo->ptCorner[i].x - nCx) + dCos * (pStCamAlignInfo->ptCorner[i].y - nCy) + nCy);
			}
		}
		//yuxuefei

		pStCamAlignInfo->SetAdjustAlignInfo(ptAdjCorner, dAlgPara, MatOrgImage.cols - 1, MatOrgImage.rows - 1);

		theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Align Image : %.2f"), tact.Stop(false) / 1000.);

		stCamAlignInfo.SetAdjustAlignInfoRatio(pStCamAlignInfo, 1.0 / nRatio);		// 根据PixelShift与否,以"原始大小"写入基线信息(线程范围)
	}

	SAFE_DELETE(pStCamAlignInfo);

	return nErrorCode;
}

long InspectAlgorithmInterface::PanelCurlJudge(cv::Mat& matSrcBuf, double* dPara, tAlignInfo* stCamAlignInfo, ResultBlob_Total* pResultBlobTotal, int nImageNum, stMeasureInfo* stCurlMeasure, CString strPath)
{
	BOOL bCurl = FALSE;
	BOOL bSaveImage = theApp.GetCommonParameter()->bIFImageSaveFlag;
	CString strSavePath;
	strSavePath.Format(_T("%s%s"), strPath, pResultBlobTotal->GetPanelID());
	Panel_Curl_Judge(matSrcBuf, dPara, stCamAlignInfo->ptCorner, bCurl, stCurlMeasure, bSaveImage, strSavePath);

	if (bCurl)
	{
		stDefectInfo* pCurl = new stDefectInfo(4, nImageNum);

		pCurl->ptLT[0].x = 0;
		pCurl->ptLT[0].y = 0;
		pCurl->ptRT[0].x = matSrcBuf.cols;
		pCurl->ptRT[0].y = 0;
		pCurl->ptRB[0].x = matSrcBuf.cols;
		pCurl->ptRB[0].y = matSrcBuf.rows;
		pCurl->ptLB[0].x = 0;
		pCurl->ptLB[0].y = matSrcBuf.rows;
		pCurl->nDefectColor[0] = E_DEFECT_COLOR_DARK;
		pCurl->nPatternClassify[0] = nImageNum;
		pCurl->nDefectJudge[0] = E_DEFECT_JUDGEMENT_APP_CURL;

		pCurl->nDefectCount = 1;

		pResultBlobTotal->AddTail_ResultBlobAndAddOffset(pCurl, NULL);
	}

	return E_ERROR_CODE_TRUE;
}
