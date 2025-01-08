
/************************************************************************/
//Mura不良检测相关源
//修改日期:18.05.31
/************************************************************************/

#include "StdAfx.h"
#include "InspectMuraNormal.h"
#include "AlgoBase.h"



// Camera String
CString gg_strCamNormal[2] = {
	_T("Coaxial"),
	_T("Side")
};

CInspectMuraNormal::CInspectMuraNormal(void)
{
	cMem[0] = NULL;
	cMem[1] = NULL;
	cMem[2] = NULL;
	cMem[3] = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
}

CInspectMuraNormal::~CInspectMuraNormal(void)
{
}

//Main检查算法
long CInspectMuraNormal::DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDstBright, cv::Mat& matDstDark,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{
	//如果参数为NULL
	if (dPara == NULL)							return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 			return E_ERROR_CODE_EMPTY_PARA;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Mura Nomral Insp. Start"));

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())					return E_ERROR_CODE_EMPTY_BUFFER;

	long	nWidth = (long)matSrcBuffer.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuffer.rows;	// 影像垂直尺寸	

	//设置范围
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)				return E_ERROR_CODE_ROI_OVER;

	if (rectROI.left >= rectROI.right)			return E_ERROR_CODE_ROI_OVER;
	if (rectROI.top >= rectROI.bottom)			return E_ERROR_CODE_ROI_OVER;

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//缓冲区分配和初始化
	cv::Mat matDstImage[E_DEFECT_COLOR_COUNT];
	cv::Mat matDstImage_Type2[E_DEFECT_LINEMURA_TOTAL];
	cv::Mat matDstImage_Type3[E_DEFECT_COLOR_COUNT];

	//算法画面号
	int nAlgImgNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];

	//缓冲区分配
	matDstImage[E_DEFECT_COLOR_DARK] = cMem[3]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
	matDstImage[E_DEFECT_COLOR_BRIGHT] = cMem[3]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());

	matDstImage_Type2[E_DEFECT_LINEMURA_X] = cMem[1]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
	matDstImage_Type2[E_DEFECT_LINEMURA_Y] = cMem[1]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
	matDstImage_Type3[E_DEFECT_COLOR_DARK] = cMem[2]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
	matDstImage_Type3[E_DEFECT_COLOR_BRIGHT] = cMem[2]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Get Memory"));

	//如果结果缓冲区不是NULL,则出现错误
	if (matDstImage[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_DEFECT_COLOR_BRIGHT].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage_Type2[E_DEFECT_LINEMURA_X].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage_Type2[E_DEFECT_LINEMURA_Y].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage_Type3[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage_Type3[E_DEFECT_COLOR_BRIGHT].empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Inspection Start"));

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process optimization
	// ROI Check Size
	int nTempHeight = rectROI.bottom - rectROI.top;
	int nTempWidth = rectROI.right - rectROI.left;

	// Get Buff
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[3]);

	cv::Mat matPreProTemp = cMem[3]->GetMat(cv::Size(nTempWidth, nTempHeight), matSrcBuffer.type());

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("PreProcess Start."));

	PreProOptimization(matSrcBuffer, matPreProTemp, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("PreProcess End."));

	//每个画面的算法不同
	switch (nAlgImgNum)
	{
	case E_IMAGE_CLASSIFY_AVI_R:
	case E_IMAGE_CLASSIFY_AVI_G:
	case E_IMAGE_CLASSIFY_AVI_B:
	{

	}
	break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	{

#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(4)
#endif

		for (int i = 0; i < 4; i++)
		{
			switch (i)
			{
			case 0:
			{
				// Type 1
				if (dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_TEXT] > 0)		// 2022.06.14 Mura Nomal Type On/Off
					nErrorCode = LogicStart_MuraType1(matPreProTemp, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
			}
			break;

			case 1:
			{
				// Type 2
				if (dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_TEXT] > 0)		// 2022.06.14 Mura Nomal Type On/Off
					nErrorCode = LogicStart_MuraType2(matPreProTemp, matDstImage_Type2, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);
			}
			break;

			case 2:
			{
				// Type 3
				if (dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_TEXT] > 0)		// 2022.06.14 Mura Nomal Type On/Off
					nErrorCode = LogicStart_MuraType3(matPreProTemp, matDstImage_Type3, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);
			}
			break;

			case 3:
			{
				// Type 4
				if (dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_TEXT] > 0)		// 2022.06.14 Mura Nomal Type On/Off
					nErrorCode = LogicStart_MuraType4(matPreProTemp, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
			}
			break;
			}
		}
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_BLACK:
	case E_IMAGE_CLASSIFY_AVI_PCD:
	case E_IMAGE_CLASSIFY_AVI_VINIT:
	{
		//不检查
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:

	{

	}
	break;

	case E_IMAGE_CLASSIFY_AVI_WHITE:
	{
		//不检查
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_DUST:
	{
		//使用Point检查的结果
	}
	break;

	//如果画面号码输入错误。
	default:
		return E_ERROR_CODE_TRUE;
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Inspection End."));

	//如果不是空画面
	if (!matDstImage_Type2[E_DEFECT_LINEMURA_X].empty() && !matDstImage_Type2[E_DEFECT_LINEMURA_Y].empty() &&
		!matDstImage_Type3[E_DEFECT_COLOR_DARK].empty() && !matDstImage_Type3[E_DEFECT_COLOR_BRIGHT].empty())
	{
		// D = T2_X + T3_D / B = T2_Y + T3_B
		cv::bitwise_or(matDstImage_Type2[E_DEFECT_LINEMURA_X], matDstImage_Type3[E_DEFECT_COLOR_DARK], matDstImage[E_DEFECT_COLOR_DARK]);
		cv::bitwise_or(matDstImage_Type2[E_DEFECT_LINEMURA_Y], matDstImage_Type3[E_DEFECT_COLOR_BRIGHT], matDstImage[E_DEFECT_COLOR_BRIGHT]);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Result Add."));

		//移除点亮区域以外的检测(移除倒圆角区域)
		if (!matBKBuffer.empty())
		{
			cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDstDark);
			cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBKBuffer, matDstBright);

			writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Copy CV Sub Result."));
		}

		else
		{
			matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDstDark);
			matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matDstBright);

			writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Copy Clone Result."));
		}

		//取消分配
		matDstImage_Type2[E_DEFECT_LINEMURA_X].release();
		matDstImage_Type2[E_DEFECT_LINEMURA_Y].release();
		matDstImage_Type3[E_DEFECT_COLOR_DARK].release();
		matDstImage_Type3[E_DEFECT_COLOR_BRIGHT].release();

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Memory Release."));
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//删除Dust后,转交结果向量
long CInspectMuraNormal::GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL。
	if (dPara == NULL)	return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)	return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)	return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuffer[E_DEFECT_COLOR_DARK].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_DARK].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	//使用参数
	bool	bDustPatternOnOff = (dPara[E_PARA_AVI_MURA_NORMAL_DUST_TEXT] > 0) ? true : false;
	bool	bFlagW = (dPara[E_PARA_AVI_MURA_NORMAL_DUST_BRIGHT_FLAG] > 0) ? true : false;
	bool	bFlagD = (dPara[E_PARA_AVI_MURA_NORMAL_DUST_DARK_FLAG] > 0) ? true : false;
	int		nSize = (int)dPara[E_PARA_AVI_MURA_NORMAL_DUST_BIG_AREA];
	int		nRange = (int)dPara[E_PARA_AVI_MURA_NORMAL_DUST_ADJUST_RANGE];

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	//////////////////////////////////////////////////////////////////////////

	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;

	//////////////////////////////////////////////////////////////////////////

		//Dust模式没有问题
	if (nImageNum == E_IMAGE_CLASSIFY_AVI_DUST)		return E_ERROR_CODE_TRUE;

	//范围设置(已检查,已检查异常处理)
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	//////////////////////////////////////////////////////////////////////////
	// Check Input Image
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_BrightDefect.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_DARK]);
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_DarkDefect.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);
	}

	//////////////////////////////////////////////////////////////////////////
		//缓冲区分配和初始化

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[0]);

	cv::Mat matDustTemp = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type(), false);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Dust Buff."));

	//////////////////////////////////////////////////////////////////////////
	// Defect Extraction
	cv::Mat matDefect[4];

	matDefect[E_MURANORMAL_TYPE2_X] = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type());
	matDefect[E_MURANORMAL_TYPE2_Y] = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type());
	matDefect[E_MURANORMAL_TYPE3_D] = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type());
	matDefect[E_MURANORMAL_TYPE3_B] = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type());

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Defect Buff."));

	cv::inRange(matDstBuffer[E_DEFECT_COLOR_DARK], 90, 110, matDefect[E_MURANORMAL_TYPE2_X]);
	cv::inRange(matDstBuffer[E_DEFECT_COLOR_BRIGHT], 90, 110, matDefect[E_MURANORMAL_TYPE2_Y]);
	cv::inRange(matDstBuffer[E_DEFECT_COLOR_DARK], 140, 160, matDefect[E_MURANORMAL_TYPE3_D]);
	cv::inRange(matDstBuffer[E_DEFECT_COLOR_BRIGHT], 140, 160, matDefect[E_MURANORMAL_TYPE3_B]);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Result separate."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Type2_X_Defect.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDefect[E_MURANORMAL_TYPE2_X]);
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Type2_Y_Defect.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDefect[E_MURANORMAL_TYPE2_Y]);
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Type3_D_Defect.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDefect[E_MURANORMAL_TYPE3_D]);
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Type3_B_Defect.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDefect[E_MURANORMAL_TYPE3_B]);
	}

	//////////////////////////////////////////////////////////////////////////
	// Type3 Defect Dust Delete

		//如果有Dust画面,则直接运行除尘逻辑
	if (!matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty())
	{
		//修改17.06.27
		//Size错误
		if (matDustBuffer[E_DEFECT_COLOR_BRIGHT].rows != matDstBuffer[E_DEFECT_COLOR_DARK].rows ||
			matDustBuffer[E_DEFECT_COLOR_BRIGHT].cols != matDstBuffer[E_DEFECT_COLOR_DARK].cols)
		{
			cv::resize(matDustBuffer[E_DEFECT_COLOR_BRIGHT], matDustTemp, matDefect[E_MURANORMAL_TYPE3_D].size());
		}
		else
		{
			matDustBuffer[E_DEFECT_COLOR_BRIGHT].copyTo(matDustTemp);
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Resize & Copy."));

		if (bDustPatternOnOff)
		{
			//清除Dark不良Dust
			cv::subtract(matDefect[E_MURANORMAL_TYPE3_D], matDustTemp, matDefect[E_MURANORMAL_TYPE3_D]);
			cv::subtract(matDefect[E_MURANORMAL_TYPE3_B], matDustTemp, matDefect[E_MURANORMAL_TYPE3_B]);
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Type3_Dark_DeletDust.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDefect[E_MURANORMAL_TYPE3_D]);
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Type3_Bright_DeletDust.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDefect[E_MURANORMAL_TYPE3_B]);
		strTemp.Format(_T("%s%s_%02d_%02d_MN_Judge_%02d_Bright_Dust.jpg"), strAlgPath, gg_strPat[nImageNum], nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDustTemp);
	}

	//////////////////////////////////////////////////////////////////////////
		//错误判定&发送结果
	{
		// ROI Set
		cv::Rect rectBlobROI;
		Insp_RectSet(rectBlobROI, rectROI, matSrcBuffer.cols, matSrcBuffer.rows);

		//标签
		CFeatureExtraction cFeatureExtraction;
		cFeatureExtraction.SetMem(&cMatBufTemp);
		cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_MURA_NORMAL, m_tInitTime, m_tBeforeTime, m_strAlgLog);

		//////////////////////////////////////////////////////////////////////////
		// Type 2 X
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDefect[E_MURANORMAL_TYPE2_X](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("T2X_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDefect[E_MURANORMAL_TYPE2_X].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("DoDefectBlobJudgment-Type2-X."));

		// Type 2 Y
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDefect[E_MURANORMAL_TYPE2_Y](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("T2Y_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDefect[E_MURANORMAL_TYPE2_Y].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("DoDefectBlobJudgment-Type2-Y."));

		//////////////////////////////////////////////////////////////////////////
		// Type 3 Dark
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDefect[E_MURANORMAL_TYPE3_D](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("T3D_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDefect[E_MURANORMAL_TYPE3_D].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("DoDefectBlobJudgment-Type3-Bright."));

		//确认画面比例是否相同
		int nModePS = 1;
		if (matSrcBuffer.rows == matDustBuffer[E_DEFECT_COLOR_DARK].rows &&
			matSrcBuffer.cols == matDustBuffer[E_DEFECT_COLOR_DARK].cols)
			nModePS = 1;
		else
			nModePS = 2;

		//如果Dust面积较大,则删除
		if (bFlagD && bDustPatternOnOff)
		{
			nErrorCode = DeleteCompareDust(matDustTemp, nRange, pResultBlob, 0, nModePS);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;
			writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("DeleteCompareDust-Dark."));
		}

		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Draw Dark Mura Defect."));

		// Type 3 Bright
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDefect[E_MURANORMAL_TYPE3_B](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("T3B_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDefect[E_MURANORMAL_TYPE3_B].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("DoDefectBlobJudgment-Type3-Dark."));

		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Draw Bright Mura Defect."));
	}

	//禁用内存
	matSrcBuffer.release();
	matDefect[E_MURANORMAL_TYPE2_X].release();
	matDefect[E_MURANORMAL_TYPE2_Y].release();
	matDefect[E_MURANORMAL_TYPE3_D].release();
	matDefect[E_MURANORMAL_TYPE3_B].release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

long CInspectMuraNormal::LogicStart_MuraType2(cv::Mat& matSrcImage, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////
		//参数
	long	nBlurLoop = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_BLURLOOP];		// 防摩押
	long	nReSize = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_RESIZE];		// 消除噪音
	long	nGausSize = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_GAUSSIAN_SIZE];
	float	fSigma = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_GAUSSIAN_SIGMA];
	long	nSegX = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_SEG_X];		// 区域分割
	long	nSegY = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_SEG_Y];
	long	nBGblurSizeX = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_BG_BLUESIZE_X];		// BG Projection
	long	nBGblurSizeY = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_BG_BLUESIZE_Y];
	long	nProblurSize = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_PRO_BLURSIZE];		// 结果projection
	long	nEdgeAreaOffSet = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_EDGE_AREA_OFFSET];
	double  dbProCheckSTD = (double)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_PRO_CHECK_STD];
	long	nThresholdX = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_THRES_X];		// Threshold OffSet
	long	nThresholdY = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE2_THRES_Y];

	//////////////////////////////////////////////////////////////////////////
		//放大(影像的平均校正功能)
	float fPow = 110;

	//重置BlurSize
	nBGblurSizeX = (nBGblurSizeX * 2) + 1;
	nBGblurSizeY = (nBGblurSizeY * 2) + 1;
	nProblurSize = (nProblurSize * 2) + 1;

	//////////////////////////////////////////////////////////////////////////
		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[1]);

	cv::Mat matSrcROIBuf = matSrcImage;
	cv::Mat matDstROIBuf_X = matDstImage[E_DEFECT_LINEMURA_X](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	cv::Mat matDstROIBuf_Y = matDstImage[E_DEFECT_LINEMURA_Y](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_Input.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	//Norch X坐标
	long nNorchX;

	//背景区域
	cv::Mat matBkROIBuf;
	if (!matBKBuffer.empty())
	{
		matBkROIBuf = matBKBuffer(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
		uchar* ptr = matBkROIBuf.ptr<uchar>(matBkROIBuf.rows / 2);
		for (int nX = 0; nX < matBkROIBuf.cols; nX++, ptr++)
		{
			if (*ptr > 0) continue;
			nNorchX = nX;
			break;
		}

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_BG.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matBkROIBuf);
		}
	}

	long	nWidth = (long)matSrcROIBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcROIBuf.rows;	// 图像垂直尺寸
	int		nOffSetUnit = 300;

	int		nCutUnit_ADD = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_ADD_UNIT];

	//CUT检查区域
	CRect rectTemp;
	rectTemp.left = nNorchX + nEdgeAreaOffSet + nCutUnit_ADD;
	rectTemp.top = nEdgeAreaOffSet;
	rectTemp.right = nWidth - nEdgeAreaOffSet - nOffSetUnit;
	rectTemp.bottom = nHeight - nEdgeAreaOffSet;

	// Get Buff
	cv::Mat matSrcActiveBuf = matSrcROIBuf(cv::Rect(rectTemp.left, rectTemp.top, rectTemp.Width(), rectTemp.Height()));
	cv::Mat matDstActiveBuf_X = matDstROIBuf_X(cv::Rect(rectTemp.left, rectTemp.top, rectTemp.Width(), rectTemp.Height()));
	cv::Mat matDstActiveBuf_Y = matDstROIBuf_Y(cv::Rect(rectTemp.left, rectTemp.top, rectTemp.Width(), rectTemp.Height()));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_CUT.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcActiveBuf);
	}

	//////////////////////////////////////////////////////////////////////////
		//16bit&放大(自动亮度校正)
	cv::Mat matPowBuf16 = cMatBufTemp.GetMat(matSrcActiveBuf.size(), CV_16UC1);

	float fMean = cv::mean(matSrcActiveBuf)[0];
	float fAutoPow = log(fPow * 16.0) / log(fMean);

	nErrorCode = AlgoBase::Pow(matSrcActiveBuf, matPowBuf16, fAutoPow, 4095, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	matSrcActiveBuf.release();

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matPowBuf16);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Pow."));

	//////////////////////////////////////////////////////////////////////////
		//消除噪音
	//////////////////////////////////////////////////////////////////////////
	cv::Mat matBlurBuf16 = cMatBufTemp.GetMat(matPowBuf16.size(), matPowBuf16.type());

	// ReSize Gaussian
	resizeGaussian(matPowBuf16, matBlurBuf16, nReSize, nGausSize, fSigma, &cMatBufTemp);
	matPowBuf16.release();

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_Gaus.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBlurBuf16);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("resize Gaussian Blur"));

	//////////////////////////////////////////////////////////////////////////
	// Range ProJection
	//////////////////////////////////////////////////////////////////////////

		//完整的Projection画面
	cv::Mat matproBufX = cMatBufTemp.GetMat(nSegX, matBlurBuf16.rows, matBlurBuf16.type());
	cv::Mat matproBufY = cMatBufTemp.GetMat(nSegY, matBlurBuf16.cols, matBlurBuf16.type());

	//一个区域的pixel数量
	int nRangeX = (int)(matBlurBuf16.cols / nSegX);
	int nRangeY = (int)(matBlurBuf16.rows / nSegY);
	{
		// Range Rect
		cv::Rect* rectRangeX = new cv::Rect[nSegX];
		memset(rectRangeX, 0, sizeof(cv::Rect) * nSegX);
		cv::Rect* rectRangeY = new cv::Rect[nSegY];
		memset(rectRangeY, 0, sizeof(cv::Rect) * nSegY);

		//区域计算X方向
		for (int nX = 0; nX < nSegX; nX++)
		{
			// Range Rect
			int nEx, nSx = nRangeX * nX;

			if (nX == nSegX - 1)	nEx = matBlurBuf16.cols - 1;
			else					nEx = nSx + nRangeX;

			rectRangeX[nX].x = nSx;
			rectRangeX[nX].y = 0;
			rectRangeX[nX].height = matBlurBuf16.rows;
			rectRangeX[nX].width = nEx - nSx;
		}

		//区域计算Y方向
		for (int nY = 0; nY < nSegY; nY++)
		{
			// Range Rect
			int nEy, nSy = nRangeY * nY;

			if (nY == nSegY - 1)	nEy = matBlurBuf16.rows - 1;
			else					nEy = nSy + nRangeY;

			rectRangeY[nY].x = 0;
			rectRangeY[nY].y = nSy;
			rectRangeY[nY].height = nEy - nSy;
			rectRangeY[nY].width = matBlurBuf16.cols;
		}

		//区域X Projection
		for (int nX = 0; nX < nSegX; nX++)
		{
			cv::Mat matROIBuf1 = matBlurBuf16(rectRangeX[nX]);

			long lWidth = matROIBuf1.cols;
			long lHeight = matROIBuf1.rows;

			cv::Mat MatProBuf = cMatBufTemp.GetMat(cv::Size(lHeight, 1), CV_16U);
			cv::Mat MatProBuf2 = matproBufX(cv::Rect(0, nX, MatProBuf.cols, 1));

			AlgoBase::MakeProfile(matROIBuf1, MatProBuf, lWidth, lHeight, 3);

			MatProBuf.copyTo(MatProBuf2);
			MatProBuf.release();
		}

		//按区域Y Projection
		for (int nY = 0; nY < nSegY; nY++)
		{
			cv::Mat matROIBuf1 = matBlurBuf16(rectRangeY[nY]);

			long lWidth = matROIBuf1.cols;
			long lHeight = matROIBuf1.rows;

			cv::Mat MatProBuf = cMatBufTemp.GetMat(cv::Size(lWidth, 1), CV_16U);
			cv::Mat MatProBuf2 = matproBufY(cv::Rect(0, nY, MatProBuf.cols, 1));

			AlgoBase::MakeProfile(matROIBuf1, MatProBuf, lWidth, lHeight, 2);

			MatProBuf.copyTo(MatProBuf2);
			MatProBuf.release();
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_ProX_Ori.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matproBufX.cols; i++)
			{
				fprintf_s(out, "%d,", i);
				for (int nX = 0; nX < nSegX; nX++)
				{
					fprintf_s(out, "%d,", matproBufX.at<ushort>(nX, i));
				}
				fprintf_s(out, "\n");
			}
			fclose(out);
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_ProY_Ori.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matproBufY.cols; i++)
			{
				fprintf_s(out, "%d,", i);
				for (int nY = 0; nY < nSegY; nY++)
				{
					fprintf_s(out, "%d,", matproBufY.at<ushort>(nY, i));
				}
				fprintf_s(out, "\n");
			}
			fclose(out);
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Range Projection"));

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//BG Projection
	//////////////////////////////////////////////////////////////////////////
	cv::Mat MatProBufX2 = cMatBufTemp.GetMat(matproBufX.size(), CV_16U);
	cv::Mat MatProBufY2 = cMatBufTemp.GetMat(matproBufY.size(), CV_16U);

	cv::blur(matproBufX, MatProBufX2, cv::Size(nBGblurSizeX, 1));
	cv::blur(matproBufY, MatProBufY2, cv::Size(nBGblurSizeY, 1));

	//Sub Data清理PNZ
	cv::Mat matSubData = cMatBufTemp.GetMat(matproBufY.size(), CV_16U);
	cv::Mat matSubDataAvg = cMatBufTemp.GetMat(cv::Size(matproBufY.cols, 1), CV_32FC1);
	matSubData = matproBufY - (MatProBufY2 - 500);

	AlgoBase::MakeProfile(matSubData, matSubDataAvg, matSubData.cols, matSubData.rows, 4);

	cv::Mat tempMean, tempStd;
	cv::meanStdDev(matSubDataAvg, tempMean, tempStd);

	double Meanvalue = tempMean.at<double>(0, 0);	// 平均
	double Stdvalue = tempStd.at<double>(0, 0);	// 标准偏差

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_MURANormal_Type2_BG_StdDev.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "Std, %.3f\n", Stdvalue);

			for (int i = 0; i < matSubDataAvg.cols; i++)
			{
				fprintf_s(out, "%d,%.3f\n", i, matSubDataAvg.at<float>(0, i));
			}

			fclose(out);
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_ProX_BG.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matproBufX.cols; i++)
			{
				fprintf_s(out, "%d,", i);
				for (int nX = 0; nX < nSegX; nX++)
				{
					fprintf_s(out, "%d,%d,", matproBufX.at<ushort>(nX, i), MatProBufX2.at<ushort>(nX, i));
				}
				fprintf_s(out, ",");
				for (int nX = 0; nX < nSegX; nX++)
				{
					fprintf_s(out, "%d,", matproBufX.at<ushort>(nX, i) - (MatProBufX2.at<ushort>(nX, i) - 500));
				}
				fprintf_s(out, "\n");
			}
			fclose(out);
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_ProY_BG.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matproBufY.cols; i++)
			{
				fprintf_s(out, "%d,", i);
				for (int nY = 0; nY < nSegY; nY++)
				{
					fprintf_s(out, "%d,%d,", matproBufY.at<ushort>(nY, i), MatProBufY2.at<ushort>(nY, i));
				}
				fprintf_s(out, ",");
				for (int nY = 0; nY < nSegY; nY++)
				{
					fprintf_s(out, "%d,", matproBufY.at<ushort>(nY, i) - (MatProBufY2.at<ushort>(nY, i) - 500));
				}
				fprintf_s(out, "\n");
			}
			fclose(out);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Image Profile StdDev Check

	if (Stdvalue < dbProCheckSTD) return E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////
	//SubTract Projection
	//////////////////////////////////////////////////////////////////////////
	cv::Mat MatProBufX3 = cMatBufTemp.GetMat(matproBufX.size(), CV_16U);
	cv::Mat MatProBufY3 = cMatBufTemp.GetMat(matproBufY.size(), CV_16U);

	cv::Mat MatProBufX4 = cMatBufTemp.GetMat(matproBufX.size(), CV_16U);
	cv::Mat MatProBufY4 = cMatBufTemp.GetMat(matproBufY.size(), CV_16U);

	//源-背景(Bright)
	cv::subtract(matproBufX, MatProBufX2, MatProBufX3);
	cv::subtract(matproBufY, MatProBufY2, MatProBufY3);

	//背景-原始(Dark)
	cv::subtract(MatProBufX2, matproBufX, MatProBufX4);
	cv::subtract(MatProBufY2, matproBufY, MatProBufY4);

	// (Bright + Dark)
	cv::add(MatProBufX3, MatProBufX4, MatProBufX4);
	cv::add(MatProBufY3, MatProBufY4, MatProBufY4);

	//调整Line Mura的宽度->nProblurSize
	cv::blur(MatProBufX4, MatProBufX3, cv::Size(nProblurSize, 1));
	cv::blur(MatProBufY4, MatProBufY3, cv::Size(nProblurSize, 1));

	vector <ushort> vProXMax;
	vector <ushort> vProYMax;
	vector <ushort>().swap(vProXMax);
	vector <ushort>().swap(vProYMax);

	int nDDUnit = (int)nBGblurSizeX / 2;

	for (int nX = 0; nX < nSegX; nX++)
	{
		double dbCalcMinX = 0;
		double dbCalcMaxX = 0;

		cv::minMaxIdx(MatProBufX3(cv::Rect(nDDUnit, nX, MatProBufX3.cols - nDDUnit, 1)), &dbCalcMinX, &dbCalcMaxX, NULL, NULL);

		vProXMax.push_back(dbCalcMaxX);
	}

	for (int nY = 0; nY < nSegY; nY++)
	{
		double dbCalcMinY = 0;
		double dbCalcMaxY = 0;

		cv::minMaxIdx(MatProBufY3(cv::Rect(nDDUnit, nY, MatProBufY3.cols - nDDUnit, 1)), &dbCalcMinY, &dbCalcMaxY, NULL, NULL);

		vProYMax.push_back(dbCalcMaxY);
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_ProX_Sub.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "Blur1Max, %d\n", vProXMax[0]);
			fprintf_s(out, "Blur2Max, %d\n", vProXMax[1]);
			fprintf_s(out, "Blur3Max, %d\n", vProXMax[2]);
			fprintf_s(out, "Blur4Max, %d\n", vProXMax[3]);
			fprintf_s(out, "Blur5Max, %d\n", vProXMax[4]);
			fprintf_s(out, "Blur6Max, %d\n", vProXMax[5]);
			fprintf_s(out, "Blur7Max, %d\n", vProXMax[6]);
			fprintf_s(out, "\n");

			fprintf_s(out, "Index, Org1, Org2, Org3, Org4, Org5, Org6, Org7, ,Blur1, Blur2, Blur3, Blur4, Blur5, Blur6, Blur7\n");

			//按区域|源-背景|投影,Threshold输入画面
			for (int i = 0; i < matproBufX.cols; i++)
			{
				fprintf_s(out, "%d,", i);
				for (int nX = 0; nX < nSegX; nX++)
				{
					fprintf_s(out, "%d,", MatProBufX4.at<ushort>(nX, i));
				}
				fprintf_s(out, ",");
				for (int nX = 0; nX < nSegX; nX++)
				{
					fprintf_s(out, "%d,", MatProBufX3.at<ushort>(nX, i));
				}
				fprintf_s(out, "\n");
			}
			fclose(out);
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_ProY_Sub.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "Blur1Max, %d\n", vProYMax[0]);
			fprintf_s(out, "Blur2Max, %d\n", vProYMax[1]);
			fprintf_s(out, "Blur3Max, %d\n", vProYMax[2]);
			fprintf_s(out, "Blur4Max, %d\n", vProYMax[3]);
			fprintf_s(out, "\n");

			fprintf_s(out, "Index, Org1, Org2, Org3, Org4, ,Blur1, Blur2, Blur3, Blur4\n");

			//按区域|源-背景|投影,Threshold输入画面
			for (int i = 0; i < matproBufY.cols; i++)
			{
				fprintf_s(out, "%d,", i);
				for (int nY = 0; nY < nSegY; nY++)
				{
					fprintf_s(out, "%d,", MatProBufY4.at<ushort>(nY, i));
				}
				fprintf_s(out, ",");
				for (int nY = 0; nY < nSegY; nY++)
				{
					fprintf_s(out, "%d,", MatProBufY3.at<ushort>(nY, i));
				}
				fprintf_s(out, "\n");
			}
			fclose(out);
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Sub Projection"));

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matXResurt = cMatBufTemp.GetMat(matBlurBuf16.size(), CV_8UC1);
	cv::Mat matYResurt = cMatBufTemp.GetMat(matBlurBuf16.size(), CV_8UC1);
	//////////////////////////////////////////////////////////////////////////
	// ThresHold
	//////////////////////////////////////////////////////////////////////////
		//Edge区域Size
	long nOutSide = nBGblurSizeX / 2;

	// X LineMura Threshold
	for (int nX = 0; nX < nSegX; nX++)
	{
		//检测Edge区域X
		ushort* ptr = (ushort*)MatProBufX3.ptr(nX) + nOutSide;
		for (int i = nOutSide; i < matproBufX.cols - nOutSide; i++, ptr++)
		{
			//二进制Projection->Image
			if (*ptr > nThresholdX)
			{
				cv::line(matXResurt, cv::Point(nX * nRangeX, i), cv::Point((nX + 1) * nRangeX, i), cv::Scalar(100), 1);
			}
		}
	}

	cv::resize(matXResurt, matDstActiveBuf_X, matDstActiveBuf_X.size());

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_X_LineMura.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstROIBuf_X);
	}

	//Edge区域Size
	nOutSide = nBGblurSizeY / 2;
	// Y LineMura Threshold
	for (int nY = 0; nY < nSegY; nY++)
	{
		//检测Edge区域X
		ushort* ptr = (ushort*)MatProBufY3.ptr(nY) + nOutSide;
		for (int i = nOutSide; i < matproBufY.cols - nOutSide; i++, ptr++)
		{
			//二进制Projection->Image
			if (*ptr > nThresholdY)
			{
				cv::line(matYResurt, cv::Point(i, nY * nRangeY), cv::Point(i, (nY + 1) * nRangeY), cv::Scalar(100), 1);
			}
		}
	}

	cv::resize(matYResurt, matDstActiveBuf_Y, matDstActiveBuf_Y.size());

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type2_Y_LineMura.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstROIBuf_Y);
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//保存8bit和12bit画面
long CInspectMuraNormal::ImageSave(CString strPath, cv::Mat matSrcBuf)
{
	if (matSrcBuf.type() == CV_8U)
	{
		cv::imwrite((cv::String)(CStringA)strPath, matSrcBuf);
	}
	else
	{
		cv::Mat matTemp = cv::Mat::zeros(matSrcBuf.size(), CV_8U);
		matSrcBuf.convertTo(matTemp, CV_8U, 1. / 16.);

		cv::imwrite((cv::String)(CStringA)strPath, matTemp);

		matTemp.release();
	}

	return E_ERROR_CODE_TRUE;
}

//删除小面积
long CInspectMuraNormal::DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrcImage.size(), CV_32SC1);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matSrcImage, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

	//如果没有个数,请退出
	if (nTotalLabel <= 0)	return E_ERROR_CODE_TRUE;

	//Blob计数
	for (int idx = 1; idx <= nTotalLabel; idx++)
	{
		//对象面积
		long nArea = matStats.at<int>(idx, CC_STAT_AREA);

		//Blob区域Rect
		cv::Rect rectTemp;
		rectTemp.x = matStats.at<int>(idx, CC_STAT_LEFT);
		rectTemp.y = matStats.at<int>(idx, CC_STAT_TOP);
		rectTemp.width = matStats.at<int>(idx, CC_STAT_WIDTH);
		rectTemp.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

		//面积限制
		if (nArea <= nCount)
		{
			//初始化为0GV后,跳过
			cv::Mat matTempROI = matSrcImage(rectTemp);
			cv::Mat matLabelROI = matLabel(rectTemp);

			for (int y = 0; y < rectTemp.height; y++)
			{
				int* ptrLabel = (int*)matLabelROI.ptr(y);
				uchar* ptrGray = (uchar*)matTempROI.ptr(y);

				for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++)
				{
					//对象
					if (*ptrLabel == idx)	*ptrGray = 0;
				}
			}

			continue;
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}
	//18.04.11-改变方式
/************************************************************************
	//如果没有缓冲区。
if( matSrcImage.empty() ) return E_ERROR_CODE_EMPTY_BUFFER;

cv::Mat DstBuffer;

CMatBuf cMatBufTemp;

	//Temp内存分配
cMatBufTemp.SetMem(cMemSub);
DstBuffer = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	//周围8个方向
bool	bConnect[8];
int		nConnectX[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
int		nConnectY[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
int		nConnectCnt;

for(int y=1 ; y<matSrcImage.rows-1 ; y++)
{
	for(int x=1 ; x<matSrcImage.cols-1 ; x++)
	{
		if( matSrcImage.at<uchar>(y, x) == 0)		continue;

		memset(bConnect, 0, sizeof(bool) * 8);
		nConnectCnt = 1;

					//确定周围的数量
		for (int z=0 ; z<8 ; z++)
		{
			if( matSrcImage.at<uchar>(y + nConnectY[z], x + nConnectX[z]) != 0)
			{
				bConnect[z] = true;
				nConnectCnt++;
			}
		}

					//不包括周边数量设置
		if( nConnectCnt < nCount )	continue;

					//绘制周围
		for (int z=0 ; z<8 ; z++)
		{
			if( !bConnect[z] )	continue;

			DstBuffer.at<uchar>(y + nConnectY[z], x + nConnectX[z]) = (BYTE)255;
		}

					//绘制中心
		DstBuffer.at<uchar>(y, x) = (BYTE)255;
	}
}

matSrcImage = DstBuffer.clone();

DstBuffer.release();
************************************************************************/

	return E_ERROR_CODE_TRUE;
}

//如果Dust面积较大,则删除
long CInspectMuraNormal::DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL
	if (matSrcBuffer.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (pResultBlob == NULL)		return E_ERROR_CODE_EMPTY_PARA;

	//检测到的不良数量
	int nDefectCount = pResultBlob->nDefectCount;

	//根据不良数量...
	for (int i = nStartIndex; i < nDefectCount; i++)
	{
		//除非Mura有问题
		if (pResultBlob->nDefectJudge[i] <= E_DEFECT_JUDGEMENT_MURA_START &&
			pResultBlob->nDefectJudge[i] > E_DEFECT_JUDGEMENT_MURA_END)
		{
			//i++;
			continue;
		}

		//18.06.13-努基,无定形村,等等...排除...
		if (pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_LINE_X ||
			pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_NUGI ||
			pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_BOX_SCRATCH)
		{
			//i++;
			continue;
		}

		//18.06.13-面积大的除外...？
		if (pResultBlob->nArea[i] > 10000)
		{
			//i++;
			continue;
		}

		//PNZ 19.05.28不包括Type 1不良
		//添加PNZ 19.07.23修改NotchDefect,或者添加Type 4
		if (pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_X ||
			pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_Y ||
			pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURANORMAL_NOTCH_DEFECT ||
			pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_TYPE4 ||
			pResultBlob->nDefectJudge[i] == E_DEFECT_JUDGEMENT_MURA_TYPE3_RING)
		{
			//i++;
			continue;
		}

		//设置周边区域
		cv::Rect rect(
			pResultBlob->ptLT[i].x,
			pResultBlob->ptLT[i].y,
			pResultBlob->ptRB[i].x - pResultBlob->ptLT[i].x,
			pResultBlob->ptRB[i].y - pResultBlob->ptLT[i].y);

		//根据画面大小进行比较(nModePS)
		//Dust无条件处于P/S模式
		//如果画面大小不正确,则检查的模式画面较小
		rect.x *= nModePS;
		rect.y *= nModePS;
		rect.width *= nModePS;
		rect.height *= nModePS;

		//nOffset扩展
		rect.x -= nOffset;
		rect.y -= nOffset;
		rect.width += (nOffset + nOffset);
		rect.height += (nOffset + nOffset);

		//异常处理
		if (rect.x < 0)	rect.x = 0;
		if (rect.y < 0)	rect.y = 0;
		if (rect.x + rect.width >= matSrcBuffer.cols)	rect.width = matSrcBuffer.cols - rect.x - 1;
		if (rect.y + rect.height >= matSrcBuffer.rows)	rect.height = matSrcBuffer.rows - rect.y - 1;

		//获取相应的ROI
		cv::Mat matTempBuf = matSrcBuffer(rect);

		//////////////////////////////////////////////////////////////////////////
				//检查Dust是否存在于不良环境中
		//////////////////////////////////////////////////////////////////////////

				//不良周边->寻找Dust画面Max GV
		double valMax;
		cv::minMaxLoc(matTempBuf, NULL, &valMax);

		//检查Dust时,没有太大的不良(存在太大的不良,存在255/4096值)
		if (valMax == 0)
		{
			//i++;
			continue;
		}

		//不报告不良情况
		pResultBlob->bUseResult[i] = false;

	}

	return nErrorCode;
}

// PNZ ShiftCopy Parameter Check ( 18.10.18 )
long	CInspectMuraNormal::ShiftCopyParaCheck(int ShiftValue, int& nCpyX, int& nCpyY, int& nLoopX, int& nLoopY)
{
	if (ShiftValue == 0) return false;

	nCpyX = (int)(ShiftValue / 1000 % 10);	// X方向单元
	nCpyY = (int)(ShiftValue / 100 % 10);	// Y方向Unit
	nLoopX = (int)(ShiftValue / 10 % 10);	// X方向Loop
	nLoopY = (int)(ShiftValue / 1 % 10);	// Y方向Loop

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::resizeGaussian(Mat& InPutImage, Mat& OutPutImage, int reSizeValue, int MaskSize, float fSigma, CMatBuf* cMemSub)
{

	if (reSizeValue > 1)
	{
		CMatBuf cMatBufTemp;
		cMatBufTemp.SetMem(cMemSub);

		//异常处理
		if (MaskSize % 2 == 0) MaskSize += 1;

		int reSizeWidth = InPutImage.cols / reSizeValue;
		int reSizeHeight = InPutImage.rows / reSizeValue;

		cv::Mat matreSizeBuf1 = cMatBufTemp.GetMat(cv::Size(reSizeWidth, reSizeHeight), InPutImage.type(), false);
		cv::Mat matreSizeBuf2 = cMatBufTemp.GetMat(cv::Size(reSizeWidth, reSizeHeight), InPutImage.type(), false);

		cv::resize(InPutImage, matreSizeBuf1, cv::Size(reSizeWidth, reSizeHeight), 0, 0, INTER_LINEAR);
		cv::GaussianBlur(matreSizeBuf1, matreSizeBuf2, cv::Size(MaskSize / reSizeValue, MaskSize / reSizeValue), fSigma / reSizeValue);
		cv::resize(matreSizeBuf2, OutPutImage, InPutImage.size(), 0, 0, INTER_LINEAR);

		matreSizeBuf1.release();
		matreSizeBuf2.release();

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}
	}
	else
		cv::GaussianBlur(InPutImage, OutPutImage, cv::Size(MaskSize, MaskSize), fSigma);

	return E_ERROR_CODE_TRUE;
}

//////////////////////////////////////////////////////////////////////////
// Type 4 Mura Judgement
// Profile Check Method
// 2018-11-26 ver 1.0 : PCM
//2019-06-06 ver 2.0:PCM修改
//2019-07-20 ver 2.1:Notch Defect添加
// PNZ
//////////////////////////////////////////////////////////////////////////

long CInspectMuraNormal::LogicStart_MuraType4(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	// Parameter
	int		nShiftCopy = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SHIFT_UNIT];
	int		nCutUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_CUT_UNIT];
	int		nCutUnit_ADD = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_ADD_UNIT];
	int		nImageBlur = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_MEANBLUR];
	int		nResize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_RESIZE];
	int		nNotchLength = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_UNIT];

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process
	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 4 Start."));

	// Parameter Setting
	cv::Rect InputROI = cv::Rect(rectROI.left + nCutUnit + nCutUnit_ADD, rectROI.top + nCutUnit, rectROI.Width() - nCutUnit * 2 - nCutUnit_ADD, rectROI.Height() - nCutUnit * 2);
	cv::Rect ProcessROI = cv::Rect(nCutUnit + nCutUnit_ADD, nCutUnit, rectROI.Width() - nCutUnit * 2 - nCutUnit_ADD, rectROI.Height() - nCutUnit * 2);

	int nRectWidth = ProcessROI.width;
	int nRectHeight = ProcessROI.height;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[3]);

	// Image Buff
	cv::Mat matSrcROIBuf = cMatBufTemp.GetMat(nRectHeight, nRectWidth, CV_8UC1);
	cv::Mat matSrcReBuf = cMatBufTemp.GetMat(nRectHeight / nResize, nRectWidth / nResize, CV_8UC1);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 4 Get Image Buff."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_InputImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	matSrcROIBuf = matSrcImage(ProcessROI);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 4 Select ROI Image."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	// Resize
	cv::resize(matSrcROIBuf, matSrcReBuf, cv::Size(nRectWidth / nResize, nRectHeight / nResize), 0, 0, INTER_CUBIC);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 4 Resize."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_Resize_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcReBuf);
	}

	//////////////////////////////////////////////////////////////////////////
	// Type 4 Check Function
	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 4 Check Start."));

	ProfileJudgment(matSrcReBuf, nCommonPara, strAlgPath, dPara, pResultBlob, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 4 Check End."));

	//////////////////////////////////////////////////////////////////////////
	// Notch Region Condition Check
	//////////////////////////////////////////////////////////////////////////

	if (nNotchLength <= 0) return false;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Defect Check Start."));

	cv::Rect ProcessROINotch = cv::Rect(0, nCutUnit, nCutUnit * 3, rectROI.Height() - nCutUnit * 2);
	cv::Rect ProROINotchBG = cv::Rect(rectROI.left, rectROI.top, nNotchLength, rectROI.Height());

	cv::Mat matSrcNotchBuf = cMatBufTemp.GetMat(nRectHeight, nCutUnit * 3, CV_8UC1);
	cv::Mat matSrcNotchReBuf = cMatBufTemp.GetMat(nRectHeight / nResize, nCutUnit * 3 / nResize, CV_8UC1);
	cv::Mat matSrcNotchBGBuf = cMatBufTemp.GetMat(rectROI.Height(), nNotchLength, CV_8UC1);
	cv::Mat matSrcNotchReBGBuf = cMatBufTemp.GetMat(rectROI.Height() / nResize, nNotchLength / nResize, CV_8UC1);

	matSrcNotchBuf = matSrcImage(ProcessROINotch);
	matSrcNotchBGBuf = matBKBuffer(ProROINotchBG);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Buff Set."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_NC_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcNotchBuf);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_NC_BG_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcNotchBGBuf);
	}

	cv::resize(matSrcNotchBuf, matSrcNotchReBuf, cv::Size(nCutUnit * 3 / nResize, nRectHeight / nResize), 0, 0, INTER_CUBIC);
	cv::resize(matSrcNotchBGBuf, matSrcNotchReBGBuf, cv::Size(nNotchLength / nResize, rectROI.Height() / nResize), 0, 0, INTER_CUBIC);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Buff Resize."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_NC_Resize_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcNotchReBuf);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_NC_Resize_BG_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcNotchReBGBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check Start."));

	NotchROICheck(matSrcNotchReBuf, matSrcNotchReBGBuf, ProROINotchBG, nCommonPara, strAlgPath, dPara, pResultBlob, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check End."));

	// 	//////////////////////////////////////////////////////////////////////////
	// 	// SuperRing Defect Inspection
	// 	//////////////////////////////////////////////////////////////////////////
	// 
	// 	cv::Rect rectTopCheckRegion, rectBottomCheckRegion;
	// 
	// 	int nRegionHeight	= 100;
	// 	int nRegionOffset	= 50;
	// 	int nRegionTop		= 135;
	// 	int nRegionBottom	= 455;
	// 
	// 	rectTopCheckRegion.x		 = (int) nRegionOffset;
	// 	rectTopCheckRegion.y		 = (int) (nRegionTop	- nRegionHeight / 2);
	// 	rectTopCheckRegion.width	 = (int) rectROI.Width()- nRegionOffset * 2;
	// 	rectTopCheckRegion.height	 = (int) nRegionHeight;
	// 
	// 	rectBottomCheckRegion.x		 = (int) nRegionOffset;
	// 	rectBottomCheckRegion.y		 = (int) (nRegionBottom	- nRegionHeight / 2);
	// 	rectBottomCheckRegion.width  = (int) rectROI.Width()- nRegionOffset * 2;
	// 	rectBottomCheckRegion.height = (int) nRegionHeight;
	// 
	// 
	// 
	// 
	// 
	// 
	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode = E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::LoopBlur(cv::Mat& matSrcImage, cv::Mat& MatBlur, int nLoop, CMatBuf* cMemSub)
{
	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matBlurBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	matSrcImage.copyTo(matBlurBuf);

	int nBlur = 5;

	if (nLoop > 0)
	{
		// Filtering
		for (int i = 0; i < nLoop; i++)
		{
			cv::blur(matBlurBuf, matBlurBuf1, cv::Size(nBlur, nBlur));
			matBlurBuf1.copyTo(matBlurBuf);
		}

		matBlurBuf.copyTo(MatBlur);
	}

	else
	{
		matBlurBuf.copyTo(MatBlur);
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ProfileJudgment(cv::Mat& matSrcImage, int* nCommonPara, CString strAlgPath, double* dPara, stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile X Check : Start "));

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int	nSampleSize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_SIZE];
	int	nAverageUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_AVG_UNIT]; // 10
	int	nUnitMinMax = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_MINMAX_UNIT]; // 15
	int	nJudge_Area = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_DIFF_AREA];

	int	XSampleSize = nSampleSize;

	// Make Profile
	int width = matSrcImage.cols;
	int height = matSrcImage.rows;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat MatproYBuf1 = cMatBufTemp.GetMat(cv::Size(width, XSampleSize), CV_8UC1);

	// Make Sampling Profile Calculation
	SampleProfileCalc(matSrcImage, MatproYBuf1, dPara, 0);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile X Check : SampleProfileCalc "));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_SampleX.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproYBuf1.cols; i++)
			{
				for (int j = 0; j < MatproYBuf1.rows; j++)
				{
					fprintf_s(out, "%d,", MatproYBuf1.at<uchar>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	// Profile Blur
	cv::Mat matTemp = cMatBufTemp.GetMat(cv::Size(width, 1), CV_8UC1);
	cv::Mat matTempIn = cMatBufTemp.GetMat(cv::Size(width, 1), CV_32FC1);
	cv::Mat matTempOut = cMatBufTemp.GetMat(cv::Size(width, 1), CV_32FC1);
	cv::Mat matBlur = cMatBufTemp.GetMat(cv::Size(width, XSampleSize), CV_32FC1);

	for (int i = 0; i < XSampleSize; i++)
	{
		matTemp = MatproYBuf1(cv::Rect(0, i, width, 1));
		matTemp.convertTo(matTempIn, CV_32FC1);
		cv::blur(matTempIn, matTempOut, cv::Size(nAverageUnit, 1));
		matTempOut.copyTo(matBlur(cv::Rect(0, i, width, 1)));
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile X Check : Profile Blur."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_SampleX_Blur.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matBlur.cols; i++)
			{
				for (int j = 0; j < matBlur.rows; j++)
				{
					fprintf_s(out, "%.3f,", matBlur.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	// Calculate Min-Max
	cv::Mat matTempOut_Min = cMatBufTemp.GetMat(cv::Size(width, 1), CV_32FC1);
	cv::Mat matTempOut_Max = cMatBufTemp.GetMat(cv::Size(width, 1), CV_32FC1);
	cv::Mat matTempOut_Diff = cMatBufTemp.GetMat(cv::Size(width, 1), CV_32FC1);

	cv::Mat matProMin = cMatBufTemp.GetMat(cv::Size(width, XSampleSize), CV_32FC1);
	cv::Mat matProMax = cMatBufTemp.GetMat(cv::Size(width, XSampleSize), CV_32FC1);
	cv::Mat matProDiff = cMatBufTemp.GetMat(cv::Size(width, XSampleSize), CV_32FC1);

	for (int i = 0; i < XSampleSize; i++)
	{
		matTempIn = matBlur(cv::Rect(0, i, width, 1));
		ProfileUnitMinMax(matTempIn, nUnitMinMax, matTempOut_Min, matTempOut_Max);
		matTempOut_Min.copyTo(matProMin(cv::Rect(0, i, width, 1)));
		matTempOut_Max.copyTo(matProMax(cv::Rect(0, i, width, 1)));

		matTempOut_Diff = matTempOut_Max - matTempOut_Min;
		matTempOut_Diff.copyTo(matProDiff(cv::Rect(0, i, width, 1)));
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile X Check : Profile Min-Max Diff Calc."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_Sample_MINFilter.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matProMin.cols; i++)
			{
				for (int j = 0; j < matProMin.rows; j++)
				{
					fprintf_s(out, "%.3f,", matProMin.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type4_Sample_MAXFilter.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < matProMax.cols; i++)
			{
				for (int j = 0; j < matProMax.rows; j++)
				{
					fprintf_s(out, "%.3f,", matProMax.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	// Result SUM
	float fmatDataDiffSUM = 0;

	for (int i = 0; i < matProDiff.rows; i++)
	{
		fmatDataDiffSUM += (float)(cv::sum(matProDiff.row(i))[0]);
	}

	int nDiffArea = (int)fmatDataDiffSUM;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_Type4_Sample_matDiff.csv"), strAlgPath, nImageNum, nCamNum, nROINumber);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "TotalDiffSUM, %d\n", nDiffArea);

			for (int i = 0; i < matProDiff.cols; i++)
			{
				for (int j = 0; j < matProDiff.rows; j++)
				{
					fprintf_s(out, "%.3f,", matProDiff.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile X Check : Result SUM."));

	//////////////////////////////////////////////////////////////////////////
	// Judge Result

	int  nCountNumber = pResultBlob->nDefectCount;
	int  nDefectIndex = 0;
	bool bOnlyType4Defect = true;

	if (nCountNumber == 0)	bOnlyType4Defect = true;
	else					bOnlyType4Defect = false;

	// Defect Index Calc.
	if (bOnlyType4Defect == true)	nDefectIndex = 0;
	else if (bOnlyType4Defect == false) nDefectIndex = nCountNumber;

	if (nDiffArea == 0) return false;

	// Data Check
	if (nDiffArea >= nJudge_Area)
	{
		if (pResultBlob != NULL)
		{
			pResultBlob->nArea[nDefectIndex] = 0;
			pResultBlob->nMaxGV[nDefectIndex] = 255;
			pResultBlob->nMinGV[nDefectIndex] = 0;
			pResultBlob->dMeanGV[nDefectIndex] = 0;

			pResultBlob->ptLT[nDefectIndex].x = 0;
			pResultBlob->ptLT[nDefectIndex].y = 0;
			pResultBlob->ptRT[nDefectIndex].x = 0;
			pResultBlob->ptRT[nDefectIndex].y = 0;
			pResultBlob->ptRB[nDefectIndex].x = 0;
			pResultBlob->ptRB[nDefectIndex].y = 0;
			pResultBlob->ptLB[nDefectIndex].x = 0;
			pResultBlob->ptLB[nDefectIndex].y = 0;

			pResultBlob->dBackGroundGV[nDefectIndex] = 0;
			pResultBlob->dCompactness[nDefectIndex] = 0;
			pResultBlob->dSigma[nDefectIndex] = 0;
			pResultBlob->dBreadth[nDefectIndex] = 0;
			pResultBlob->dF_Min[nDefectIndex] = 0;
			pResultBlob->dF_Max[nDefectIndex] = 0;
			pResultBlob->dF_Elongation[nDefectIndex] = 0;
			pResultBlob->dCompactness[nDefectIndex] = 0;

			//亮度
			pResultBlob->nDefectColor[nDefectIndex] = E_DEFECT_COLOR_DARK;
			pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURA_TYPE4;
			pResultBlob->nPatternClassify[nDefectIndex] = nImageUI;

			//计数增加
			pResultBlob->nDefectCount = nDefectIndex + 1;
		}
	}

	int nLimitArea = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_PROFILE_DIFF_AREA_LIMITE];

	//数据检查限制Uniformity
	if (nDiffArea >= nLimitArea && nDiffArea < nJudge_Area)
	{
		if (pResultBlob != NULL)
		{
			pResultBlob->nArea[nDefectIndex] = 0;
			pResultBlob->nMaxGV[nDefectIndex] = 255;
			pResultBlob->nMinGV[nDefectIndex] = 0;
			pResultBlob->dMeanGV[nDefectIndex] = 0;

			pResultBlob->ptLT[nDefectIndex].x = 0;
			pResultBlob->ptLT[nDefectIndex].y = 0;
			pResultBlob->ptRT[nDefectIndex].x = 0;
			pResultBlob->ptRT[nDefectIndex].y = 0;
			pResultBlob->ptRB[nDefectIndex].x = 0;
			pResultBlob->ptRB[nDefectIndex].y = 0;
			pResultBlob->ptLB[nDefectIndex].x = 0;
			pResultBlob->ptLB[nDefectIndex].y = 0;

			pResultBlob->dBackGroundGV[nDefectIndex] = 0;
			pResultBlob->dCompactness[nDefectIndex] = 0;
			pResultBlob->dSigma[nDefectIndex] = 0;
			pResultBlob->dBreadth[nDefectIndex] = 0;
			pResultBlob->dF_Min[nDefectIndex] = 0;
			pResultBlob->dF_Max[nDefectIndex] = 0;
			pResultBlob->dF_Elongation[nDefectIndex] = 0;
			pResultBlob->dCompactness[nDefectIndex] = 0;

			//亮度
			pResultBlob->nDefectColor[nDefectIndex] = E_DEFECT_COLOR_DARK;
			pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURA_TYPE3_RING;
			pResultBlob->nPatternClassify[nDefectIndex] = nImageUI;

			//计数增加
			pResultBlob->nDefectCount = nDefectIndex + 1;
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile X Check : Judgement End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::SampleProfileCalc(cv::Mat& matSrcImage, cv::Mat& MatproYBuf, double* dPara, int nDirection)
{
	//异常处理
	if (dPara == NULL) return false;

	// Parameter
	int	nSampleSize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_SIZE];
	int	SampleROIX = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_ROI_X];
	int	SampleROIY = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SAMPLE_ROI_Y];

	int nSampleSizeX = nSampleSize;
	int nSampleSizeY = nSampleSize * 2 + 1;

	// Image Info.
	int width = matSrcImage.cols;
	int height = matSrcImage.rows;

	//X方向Sampling
	if (nDirection == 0)
	{
		// Mat Buff
		cv::Mat matSampleROI;
		cv::Mat matProBuf = cv::Mat::zeros(cv::Size(width, 1), CV_8U);

		//异常处理
		if (nSampleSizeX * SampleROIX > height) return false;

		// Sample Para
		int nSampleUnit = (int)height / nSampleSizeX;

		//如果可用数据小于设置值,则返回
		if (nSampleUnit < SampleROIX) return false;

		// Sample Profile
		for (int i = 0; i < nSampleSizeX; i++)
		{
			cv::Rect SampleRect;
			SampleRect.x = 0;
			SampleRect.y = nSampleUnit * i;
			SampleRect.width = width;
			SampleRect.height = SampleROIX;

			cv::Mat matSampleROI = matSrcImage(SampleRect);

			// Make Profile
			AlgoBase::MakeProfile(matSampleROI, matProBuf, width, SampleROIX, 0);
			matProBuf.copyTo(MatproYBuf(cv::Rect(0, i, width, 1)));
		}
	}

	//Y方向Sampling
	else if (nDirection == 1)
	{
		// Mat Buff
		cv::Mat matSampleROI;
		cv::Mat matProBuf = cv::Mat::zeros(cv::Size(height, 1), CV_8U);

		//异常处理
		if (nSampleSizeY * SampleROIY > width) return false;

		// Sample Para
		int nSampleUnit = (int)width / nSampleSizeY;

		//如果可用数据小于设置值,则返回
		if (nSampleUnit < SampleROIY) return false;

		// Sample Profile
		for (int i = 0; i < nSampleSizeY; i++)
		{
			cv::Rect SampleRect;
			SampleRect.x = nSampleUnit * i;
			SampleRect.y = 0;
			SampleRect.width = SampleROIY;
			SampleRect.height = height;

			cv::Mat matSampleROI = matSrcImage(SampleRect);

			// Make Profile
			AlgoBase::MakeProfile(matSampleROI, matProBuf, SampleROIY, height, 1);
			matProBuf.copyTo(MatproYBuf(cv::Rect(0, i, height, 1)));
		}
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ProfileUnitMinMax(cv::Mat& matSrcImage, int nUnit, cv::Mat& matDstImageMin, cv::Mat& matDstImageMax)
{
	//异常处理
	if (matSrcImage.empty() || nUnit == 0) return false;

	// Parameter
	int nSampleLength = matSrcImage.cols;
	int nSampleNumber = matSrcImage.rows;

	int nSubrange = (int)nSampleLength / nUnit + 1;

	vector<float>	TestVector_Unit;

	int nSubStart = 0;
	int nSubEnd = 0;

	for (int i = 0; i < nSubrange; i++)
	{
		// Data Start - End
		nSubStart = i * nUnit;
		if (i == nSubrange - 1)	nSubEnd = nSampleLength;
		else					nSubEnd = nSubStart + nUnit;

		//初始化
		float* ptrData = (float*)matSrcImage.ptr(0) + nSubStart;
		vector <float>().swap(TestVector_Unit);

		for (int j = nSubStart; j < nSubEnd; j++, ptrData++)
		{
			float nDataValue = *ptrData;
			TestVector_Unit.push_back(nDataValue);
		}

		// Calc Min Max
		double minvalue = 0;
		double maxvlaue = 0;

		cv::minMaxIdx(TestVector_Unit, &minvalue, &maxvlaue, NULL, NULL);

		float* ptrDataMin = (float*)matDstImageMin.ptr(0) + nSubStart;
		float* ptrDataMax = (float*)matDstImageMax.ptr(0) + nSubStart;

		for (int j = nSubStart; j < nSubEnd; j++, ptrDataMin++, ptrDataMax++)
		{
			*ptrDataMin = (float)minvalue;
			*ptrDataMax = (float)maxvlaue;
		}
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ProfileSlopeCheck(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nSlopCalceUnite)
{
	//异常处理
	if (matSrcImage.empty() || nSlopCalceUnite == 0) return false;

	//初始化
	float* ptrB = (float*)matSrcImage.ptr(0);							// Data左侧Data
	float* ptrData = (float*)matSrcImage.ptr(0) + nSlopCalceUnite;		// Data
	float* ptrF = (float*)matSrcImage.ptr(0) + nSlopCalceUnite * 2;	// Data右侧Data
	float* ptrResult = (float*)matDstImage.ptr(0);							// Result Data

	for (int i = 0; i < matSrcImage.cols; i++, ptrResult++)
	{
		// Profile Edge Offset
		if (i >= nSlopCalceUnite && i < matSrcImage.cols - nSlopCalceUnite)
		{
			// Slope Calculation
			float nForeSlope = (float)(*ptrF - *ptrData) / nSlopCalceUnite;
			float nBackSlope = (float)(*ptrData - *ptrB) / nSlopCalceUnite;

			float fCalcuValue = 0.0;

			// Case 1 : Dark Slope Calculation
			if (nForeSlope > 0 && nBackSlope < 0)	fCalcuValue = (float)(abs(nForeSlope) + abs(nBackSlope));
			else									fCalcuValue = (float)0.0;

			*ptrResult = fCalcuValue;

			//下一个Data
			ptrF++; ptrB++;	ptrData++;
		}

		else
			*ptrResult = 0.0;
	}

	return  E_ERROR_CODE_TRUE;
}

//////////////////////////////////////////////////////////////////////////
//Type 3 Mura检测算法
// 2018-12-11 ver 0.0.1
// 2019-04-26 ver 1.0.0
// 2019-12-05 ver 1.0.1
// PNZ
//////////////////////////////////////////////////////////////////////////

long CInspectMuraNormal::LogicStart_MuraType3(cv::Mat& matSrcImage, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	// Parameter
	int		nShiftCopy = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SHIFT_UNIT];
	int		nCutUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_CUT_UNIT];
	int		nImageBlur = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_MEANBLUR];
	int		nResize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_RESIZE];

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process
	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 3 Start."));

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[2]);

	// Image Buff
	cv::Mat matSrcReBuf = cMatBufTemp.GetMat(matSrcImage.rows / nResize, matSrcImage.cols / nResize, CV_8UC1);
	cv::Mat matBGReBuf = cMatBufTemp.GetMat(matSrcImage.rows / nResize, matSrcImage.cols / nResize, CV_8UC1);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 3 Get Image Buff."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_InputImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BK_buffer.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKBuffer);
	}

	//////////////////////////////////////////////////////////////////////////
	// BG ROI

	// Resize
	cv::resize(matSrcImage, matSrcReBuf, matSrcReBuf.size(), 0, 0, INTER_CUBIC);
	cv::resize(matBKBuffer(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height())), matBGReBuf, matSrcReBuf.size(), 0, 0, INTER_CUBIC);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 3 Resize."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcReBuf);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Resize_BG.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBGReBuf);
	}

	//////////////////////////////////////////////////////////////////////////
	// Main Processing
	cv::Mat matDstBuf[E_DEFECT_COLOR_COUNT];

	matDstBuf[E_DEFECT_COLOR_BRIGHT] = cMatBufTemp.GetMat(matSrcReBuf.size(), CV_8UC1);
	matDstBuf[E_DEFECT_COLOR_DARK] = cMatBufTemp.GetMat(matSrcReBuf.size(), CV_8UC1);

	MainProcessingT3(matSrcReBuf, matDstBuf, matBGReBuf, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 3 Main Process End."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuf[E_DEFECT_COLOR_BRIGHT]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuf[E_DEFECT_COLOR_DARK]);
	}

	// Image Reset
	ResultImageReset(matDstBuf, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 3 Result Image Reset End."));

	//matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matDstImage[E_DEFECT_COLOR_DARK]);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BrightFinal.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_DarkFinal.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode = E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::MainProcessingT3(cv::Mat& matSrcImage, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Main Process Start."));

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process Kernel Blur
	int nKernelSize_Small = 3;
	int nKernelRoundWidth_Small = 1;

	cv::Mat matKernel_Mid_Small = cv::Mat::zeros(nKernelSize_Small * 2 + 1, nKernelSize_Small * 2 + 1, CV_8UC1);
	cv::Mat matKernel_Cen_Small = cv::Mat::zeros(nKernelSize_Small * 2 + 1, nKernelSize_Small * 2 + 1, CV_8UC1);

	cv::Mat matKBBuff = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);

	SetRoundKernel(matKernel_Mid_Small, matKernel_Cen_Small, nKernelSize_Small, nKernelRoundWidth_Small);	// Set Kernel
	setKernelConvolution(matSrcImage, matKernel_Cen_Small, matKBBuff, &cMatBufTemp);						// Convolution

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Pre_Input.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matKBBuff);
	}

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		nMophSize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MOPHOSIZE];
	double	dbDistTh = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_DISTTH];
	int		nOutLine = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_DELET_OUTLINE];
	int		nEdgeUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX] / 2;	// Seg X Unit

	int		nOutDelet = 3;
	long	lSegX1 = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BGSEG_XY];		// BG	Seg X 100
	long	lSegY1 = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BGSEG_XY];		// BG	Seg Y 100
	long	lSegX2 = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_SEG_XY];		// Data Seg X 25
	long	lSegY2 = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_SEG_XY];		// Data Seg Y 25

	double	dbBRatio1 = (double)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BG_BRIGHT_RATIO];		// BG Bright Ratio		5%	1.05
	double	dbDRatio1 = (double)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_BG_DARK_RATIO];		// BG Dark Ratio		5%	0.95
	double	dbBRatio2 = (double)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_BRIGHT_RATIO];		// Data Bright Ratio	15% 1.15
	double	dbDRatio2 = (double)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_DATA_DARK_RATIO];		// Data Dark Ratio		15% 0.86

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("SBS Process Start."));

	//////////////////////////////////////////////////////////////////////////
	// Select Region
	int nSubDelUnit = nEdgeUnit + nOutDelet;

	cv::Rect GlobleROI = cv::Rect(nOutDelet, nOutDelet, matSrcImage.cols - nOutDelet * 2, matSrcImage.rows - nOutDelet * 2);
	cv::Rect SubROI = cv::Rect(nSubDelUnit, nSubDelUnit, matSrcImage.cols - nSubDelUnit * 2, matSrcImage.rows - nSubDelUnit * 2);

	cv::Mat matBBSBG = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), false);
	cv::Mat matBBSData = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), false);
	cv::Mat matBBSResult = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), true);
	cv::Mat matSBSAll = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), false);
	cv::Mat matSBSSub = cMatBufTemp.GetMat(SubROI.size(), matSrcImage.type(), false);
	cv::Mat matSBSResult = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), true);

	RangComProcess(matKBBuff(GlobleROI), matBBSBG, lSegX1, lSegY1, dbBRatio1, dbDRatio1, dPara, nCommonPara, strAlgPath, &cMatBufTemp);
	RangComProcess(matKBBuff(GlobleROI), matBBSData, lSegX2, lSegY2, dbBRatio2, dbDRatio2, dPara, nCommonPara, strAlgPath, &cMatBufTemp);
	RangComProcess(matKBBuff(GlobleROI), matSBSAll, lSegX1, lSegY1, dbBRatio2, dbDRatio2, dPara, nCommonPara, strAlgPath, &cMatBufTemp);
	RangComProcess(matKBBuff(SubROI), matSBSSub, lSegX1, lSegY1, dbBRatio2, dbDRatio2, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BGTemp.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBBSBG);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_DataTemp.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBBSData);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_AllTemp.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSBSAll);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SubTemp.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSBSSub);
	}

	//////////////////////////////////////////////////////////////////////////
	// Small Black Spot ( SBS ) Inspection Algorithm
	cv::Mat matSBSAllResult = cMatBufTemp.GetMat(GlobleROI.size(), CV_8UC1);
	cv::Mat matSBSSubResult = cMatBufTemp.GetMat(SubROI.size(), CV_8UC1);

	SBSInspection(matKBBuff(GlobleROI), matSBSAll, matSBSAllResult, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);
	//SBSInspection(matBBSData, matSBSAll, matSBSAllResult, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);
	SBSInspection(matKBBuff(SubROI), matSBSSub, matSBSSubResult, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	//////////////////////////////////////////////////////////////////////////
	// Out Line Delete
	int nImageWidth = matSBSSubResult.cols;
	int nImageHeight = matSBSSubResult.rows;
	int nSubRegionOutLine = 10;

	cv::Point cpLT, cpRT, cpLB, cpRB;

	cpLT.x = 0;				cpLT.y = 0;
	cpRT.x = nImageWidth;	cpRT.y = 0;
	cpLB.x = 0;				cpLB.y = nImageHeight;
	cpRB.x = nImageWidth;	cpRB.y = nImageHeight;

	cv::line(matSBSSubResult, cpLT, cpRT, cv::Scalar(0, 0, 0), nSubRegionOutLine);
	cv::line(matSBSSubResult, cpRT, cpRB, cv::Scalar(0, 0, 0), nSubRegionOutLine);
	cv::line(matSBSSubResult, cpRB, cpLB, cv::Scalar(0, 0, 0), nSubRegionOutLine);
	cv::line(matSBSSubResult, cpLB, cpLT, cv::Scalar(0, 0, 0), nSubRegionOutLine);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("SBS Delete Out Line."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_SubRegionOutLine.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSBSSubResult);
	}

	matSBSAllResult.copyTo(matSBSResult);

	ResultCombination(matSBSAllResult, matSBSSubResult, matSBSResult, nEdgeUnit, nCommonPara, strAlgPath, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_ResultSUM.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSBSResult);
	}

	///////////////////////////////////////////////////////////////////////////
	// Calc. Image Info.

	cv::Mat matSubDataData = cMatBufTemp.GetMat(GlobleROI.size(), CV_8UC1);

	cv::subtract(matSBSAll, matKBBuff(GlobleROI), matSubDataData);

	long	nSegX = 50;
	long	nSegY = 50;

	long	nWidth = (long)matSubDataData.cols;	// 图像宽度大小
	long	nHeight = (long)matSubDataData.rows;	// 图像垂直尺寸

	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	cv::Mat matDrawResult = cMatBufTemp.GetMat(matSubDataData.size(), matSubDataData.type(), false);
	cv::Mat matAvgValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matOrgValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matMaxValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matStdValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matPCRValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matHISTCount = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);

	vector<vector<float>> vHistData;

	RangeDataCalc(matSubDataData, matDrawResult, matKBBuff(GlobleROI), matAvgValue, matOrgValue, matMaxValue, matStdValue, matPCRValue, matHISTCount, vHistData, dPara, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Small_SubResultDraw.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDrawResult);
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Small_SubResultInfo.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{

			fprintf_s(out, "ImageSegAvg+StdDev\n");

			for (int j = 0; j < matAvgValue.rows; j++)
			{
				for (int i = 0; i < matAvgValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matAvgValue.at<float>(j, i) + matStdValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegAvgData\n");

			for (int j = 0; j < matAvgValue.rows; j++)
			{
				for (int i = 0; i < matAvgValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matAvgValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegStdData\n");

			for (int j = 0; j < matStdValue.rows; j++)
			{
				for (int i = 0; i < matStdValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matStdValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegOrgAvg\n");

			for (int j = 0; j < matOrgValue.rows; j++)
			{
				for (int i = 0; i < matOrgValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matOrgValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegPCRData\n");

			for (int j = 0; j < matPCRValue.rows; j++)
			{
				for (int i = 0; i < matPCRValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matPCRValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegHistCount\n");

			for (int j = 0; j < matHISTCount.rows; j++)
			{
				for (int i = 0; i < matHISTCount.cols; i++)
				{
					fprintf_s(out, "%.3f,", matHISTCount.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageMax\n");

			for (int j = 0; j < matMaxValue.rows; j++)
			{
				for (int i = 0; i < matMaxValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matMaxValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// SBS OutLine Delete
	int nEdgeDeleteUnit = 5;

	cv::Mat matSBSBK = cMatBufTemp.GetMat(matBKBuffer.size(), matBKBuffer.type(), false);

	Morphology(matBKBuffer, matSBSBK, nEdgeDeleteUnit, nEdgeDeleteUnit, EP_MORP_MURANORMAL_DILATE, &cMatBufTemp);

	cv::subtract(matSBSResult, matSBSBK(GlobleROI), matSBSResult);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_FinalResult.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSBSResult);
	}

	//////////////////////////////////////////////////////////////////////////
	// Big Black Spot ( BBS ) Inspection Algorithm

	cv::Mat matBBSTemp = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), false);

	BBSInspection(matBBSData, matBBSBG, matBBSResult, matBBSTemp, matBKBuffer, GlobleROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_FinalResult.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBBSResult);
	}

	//////////////////////////////////////////////////////////////////////////
	// Edge Data ReCheck

	cv::Mat matSBSERecheck = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), false);
	int nEdgeReCheckUnit = 50;

	SBSandBBSEdgeReCheck(matSBSResult, matBBSTemp, matSBSERecheck, matAvgValue, matStdValue, matMaxValue, nEdgeReCheckUnit, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_EdgeRE.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSBSERecheck);
	}

	//////////////////////////////////////////////////////////////////////////
	// Result SUM
	cv::Mat matFinalResult = cMatBufTemp.GetMat(GlobleROI.size(), matSrcImage.type(), false);

	//matFinalResult = matSBSResult + matBBSResult;

	matFinalResult = matSBSERecheck + matBBSResult;

	cv::threshold(matFinalResult, matFinalResult, 100.0, 150.0, THRESH_BINARY);

	matFinalResult.copyTo(matDstImage[E_DEFECT_COLOR_DARK](GlobleROI));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::SBSInspection(cv::Mat& matSrcImage, cv::Mat& matSrcImageBG, cv::Mat& matDstImage, CRect rectROI, double* dPara, int* nCommonPara,
	CString strAlgPath, CMatBuf* cMemSub)
{
	if (dPara == NULL) return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		nMophSize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MOPHOSIZE];
	double	dbDistTh = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_DISTTH];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("SBS Process Start."));

	//////////////////////////////////////////////////////////////////////////
		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matRFTResult[E_DEFECT_COLOR_COUNT];

	matRFTResult[E_DEFECT_COLOR_DARK] = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
	matRFTResult[E_DEFECT_COLOR_BRIGHT] = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);

	cv::Mat matResult = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matDistBuff = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matMorphBuff = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	cv::subtract(matSrcImageBG, matSrcImage, matResult); //choi 05.14

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matResult);
	}

	RangeFitThreshold(matResult, matRFTResult, dPara, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_RFT.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matRFTResult[E_DEFECT_COLOR_BRIGHT]);
	}

	Morphology(matRFTResult[E_DEFECT_COLOR_BRIGHT], matMorphBuff, nMophSize, nMophSize, EP_MORP_MURANORMAL_CLOSE, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Morph_C.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matMorphBuff);
	}

	DistanceTransformThreshold(matMorphBuff, matDistBuff, dbDistTh, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Dist.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDistBuff);
	}

	Morphology(matDistBuff, matMorphBuff, dbDistTh * 2, dbDistTh * 2, EP_MORP_MURANORMAL_DILATE, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Morph_D.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matMorphBuff);
	}

	matMorphBuff.copyTo(matDstImage);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;

}

long CInspectMuraNormal::Morphology(Mat& matSrcImage, Mat& matDstImage, long nSizeX, long nSizeY, int nOperation, CMatBuf* cMemSub, int nIter)
{
	if (nSizeX < 1)			return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSizeY < 1)			return E_ERROR_CODE_POINT_WARNING_PARA;
	if (matSrcImage.empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	int nRep = nIter;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat tempBuffer = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	cv::Point anchor(-1, -1);

	cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nSizeX, nSizeY));

	switch (nOperation)
	{
	case EP_MORP_MURANORMAL_ERODE:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_ERODE, StructElem, anchor, nRep);
		break;

	case EP_MORP_MURANORMAL_DILATE:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_DILATE, StructElem, anchor, nRep);
		break;

	case EP_MORP_MURANORMAL_OPEN:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_OPEN, StructElem, anchor, nRep);
		break;

	case EP_MORP_MURANORMAL_CLOSE:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_CLOSE, StructElem, anchor, nRep);
		break;

	case EP_MORP_MURANORMAL_GRADIENT:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_GRADIENT, StructElem, anchor, nRep);
		break;

	case EP_MORP_MURANORMAL_TOPHAT:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_TOPHAT, StructElem, anchor, nRep);
		break;

	case EP_MORP_MURANORMAL_BLACKHAT:
		cv::morphologyEx(matSrcImage, tempBuffer, MORPH_BLACKHAT, StructElem, anchor, nRep);
		break;

	default:
		StructElem.release();
		return E_ERROR_CODE_POINT_WARNING_PARA;
		break;
	}

	tempBuffer.copyTo(matDstImage);

	StructElem.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//分区进化
long CInspectMuraNormal::RangeAvgThreshold(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub)
{
	//异常处理
	if (dPara == NULL) return false;

	long	nSegX = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX];
	long	nSegY = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGY];
	long	nEdgeArea = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGEAREA];
	float	fActiveDarkRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_DR];
	float	fActiveBrightRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_BR];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_DR];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_BR];
	int		nMinThGV = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MINGV];
	int		nMedian = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MEDIANSIZE];

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start."));

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, matSrcImage.type());

	matSrcImage.copyTo(matBlurBuf);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Get Memory & Copy Image."));

	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//平均计算
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		BYTE* ptr = (BYTE*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols) rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows) rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);

			double dblAverage;
			double dblStdev;
			AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

			//设置平均范围
			int nMinGV = (int)(dblAverage - dblStdev);
			int nMaxGV = (int)(dblAverage + dblStdev);
			if (nMinGV < 0)	nMinGV = 0;
			if (nMaxGV > 255)	nMaxGV = 255;

			//初始化
			__int64 nPixelSum = 0;
			__int64 nPixelCount = 0;

			//仅按设置的平均范围重新平均
			float* pVal = (float*)matHisto.ptr(0) + nMinGV;

			for (int m = nMinGV; m <= nMaxGV; m++, pVal++)
			{
				nPixelSum += (m * *pVal);
				nPixelCount += *pVal;
			}

			//至少要有一个数量...
			if (nPixelCount > 0)
				ptr[x] = (BYTE)(nPixelSum / (float)nPixelCount);
		}
	}

	//设置为周边平均值->中间值
	AlgoBase::MedianFilter(matAvgBuf, matAvgBuf, nMedian, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("MedianFilter."));

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		BYTE* ptr = (BYTE*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols) rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows) rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//平均*Ratio
			long nDarkTemp, nBrightTemp;

			//单独设置Edge
			if (x < nEdgeArea ||
				y < nEdgeArea ||
				x >= nRangeX - nEdgeArea ||
				y >= nRangeY - nEdgeArea)
			{
				nDarkTemp = (long)(ptr[x] * fEdgeDarkRatio);
				nBrightTemp = (long)(ptr[x] * fEdgeBrightRatio);
			}
			else
			{
				nDarkTemp = (long)(ptr[x] * fActiveDarkRatio);
				nBrightTemp = (long)(ptr[x] * fActiveBrightRatio);
			}

			//Gray有太暗的情况。
			//(平均GV2~3*fBrightRatio->二进制:噪声全部上升)
			if (nBrightTemp < nMinThGV)	nBrightTemp = nMinThGV;

			//异常处理
			if (nDarkTemp < 0)		nDarkTemp = 0;
			if (nBrightTemp > 255)	nBrightTemp = 255;

			//参数0时异常处理
			if (fActiveDarkRatio == 0)		nDarkTemp = -1;
			if (fActiveBrightRatio == 0)	nBrightTemp = 256;

			// E_DEFECT_COLOR_DARK Threshold
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, nDarkTemp, 150.0, THRESH_BINARY_INV);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, nBrightTemp, 150.0, THRESH_BINARY);
		}
	}

	//禁用
	matAvgBuf.release();
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ResultImageReset(cv::Mat* matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//异常处理
	if (dPara == NULL)								return false;
	if (matSrcImage[E_DEFECT_COLOR_BRIGHT].empty()) return false;
	if (matSrcImage[E_DEFECT_COLOR_DARK].empty())	return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Image Reset Start."));

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		nResize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_RESIZE];

	int		nImageWidth = matSrcImage[E_DEFECT_COLOR_BRIGHT].cols;
	int		nImageHeight = matSrcImage[E_DEFECT_COLOR_BRIGHT].rows;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matResultBuff[E_DEFECT_COLOR_COUNT];

	matResultBuff[E_DEFECT_COLOR_BRIGHT] = cMatBufTemp.GetMat(nImageHeight * nResize, nImageWidth * nResize, CV_8UC1);
	matResultBuff[E_DEFECT_COLOR_DARK] = cMatBufTemp.GetMat(nImageHeight * nResize, nImageWidth * nResize, CV_8UC1);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Reset_B_Input.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage[E_DEFECT_COLOR_BRIGHT]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Reset_D_Input.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage[E_DEFECT_COLOR_DARK]);
	}

	// UpSize Buff
	cv::resize(matSrcImage[E_DEFECT_COLOR_BRIGHT], matResultBuff[E_DEFECT_COLOR_BRIGHT], cv::Size(nImageWidth * nResize, nImageHeight * nResize), 0, 0, 0);
	cv::resize(matSrcImage[E_DEFECT_COLOR_DARK], matResultBuff[E_DEFECT_COLOR_DARK], cv::Size(nImageWidth * nResize, nImageHeight * nResize), 0, 0, 0);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Reset_B_Sizeup.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matResultBuff[E_DEFECT_COLOR_BRIGHT]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Reset_D_Sizeup.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matResultBuff[E_DEFECT_COLOR_DARK]);
	}

	cv::Rect ResetRect;

	ResetRect.x = rectROI.left;
	ResetRect.y = rectROI.top;
	ResetRect.width = matResultBuff[E_DEFECT_COLOR_DARK].cols;
	ResetRect.height = matResultBuff[E_DEFECT_COLOR_DARK].rows;

	// Image Reset
	matResultBuff[E_DEFECT_COLOR_BRIGHT].copyTo(matDstImage[E_DEFECT_COLOR_BRIGHT](cv::Rect(ResetRect.x, ResetRect.y, ResetRect.width, ResetRect.height)));
	matResultBuff[E_DEFECT_COLOR_DARK].copyTo(matDstImage[E_DEFECT_COLOR_DARK](cv::Rect(ResetRect.x, ResetRect.y, ResetRect.width, ResetRect.height)));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::BBSInspection(cv::Mat& matSrcImageData, cv::Mat& matSrcImageBG, cv::Mat& matDstImage, cv::Mat& matDiffTemp, cv::Mat& matBKBuffer, cv::Rect rectROI,
	double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// Paramter

		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matReFitBuff = cMatBufTemp.GetMat(matSrcImageData.size(), matSrcImageData.type(), false);
	cv::Mat matReFitBuffSub = cMatBufTemp.GetMat(matSrcImageData.size(), matSrcImageData.type(), false);

	RangeReFit(matSrcImageBG, matSrcImageData, matReFitBuff, &cMatBufTemp);

	cv::subtract(matReFitBuff, matSrcImageData, matReFitBuffSub);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_BGReFit.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matReFitBuff);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_BGsubData.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matReFitBuffSub);
	}

	int nEdgeDeleteUnit = 5;

	Morphology(matBKBuffer, matBKBuffer, nEdgeDeleteUnit, nEdgeDeleteUnit, EP_MORP_MURANORMAL_DILATE, &cMatBufTemp);

	cv::subtract(matReFitBuffSub, matBKBuffer(rectROI), matReFitBuffSub);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_SubResultBG.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matReFitBuffSub);
	}

	matReFitBuffSub.copyTo(matDiffTemp);

	//////////////////////////////////////////////////////////////////////////
	// Calc. Image Info.

	long	nSegX = 50;
	long	nSegY = 50;

	long	nWidth = (long)matSrcImageData.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImageData.rows;	// 图像垂直尺寸

	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	cv::Mat matDrawResult = cMatBufTemp.GetMat(matSrcImageData.size(), matSrcImageData.type(), true);
	cv::Mat matAvgValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matOrgValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matMaxValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matStdValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matPCRValue = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matHISTCount = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);

	vector<vector<float>> vHistData;

	RangeDataCalc(matReFitBuffSub, matDrawResult, matSrcImageData, matAvgValue, matOrgValue, matMaxValue, matStdValue, matPCRValue, matHISTCount, vHistData, dPara, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Test_SubResultDraw.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDrawResult);
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Test_SubResultInfo.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{

			fprintf_s(out, "ImageSegAvg+StdDev\n");

			for (int j = 0; j < matAvgValue.rows; j++)
			{
				for (int i = 0; i < matAvgValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matAvgValue.at<float>(j, i) + matStdValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegAvgData\n");

			for (int j = 0; j < matAvgValue.rows; j++)
			{
				for (int i = 0; i < matAvgValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matAvgValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegStdData\n");

			for (int j = 0; j < matStdValue.rows; j++)
			{
				for (int i = 0; i < matStdValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matStdValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegOrgAvg\n");

			for (int j = 0; j < matOrgValue.rows; j++)
			{
				for (int i = 0; i < matOrgValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matOrgValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegPCRData\n");

			for (int j = 0; j < matPCRValue.rows; j++)
			{
				for (int i = 0; i < matPCRValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matPCRValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageSegHistCount\n");

			for (int j = 0; j < matHISTCount.rows; j++)
			{
				for (int i = 0; i < matHISTCount.cols; i++)
				{
					fprintf_s(out, "%.3f,", matHISTCount.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fprintf_s(out, "\n");

			fprintf_s(out, "ImageMax\n");

			for (int j = 0; j < matMaxValue.rows; j++)
			{
				for (int i = 0; i < matMaxValue.cols; i++)
				{
					fprintf_s(out, "%.3f,", matMaxValue.at<float>(j, i));
				}
				fprintf_s(out, "\n");
			}

			fclose(out);
		}
	}

	RangeCheckThreshold(matReFitBuffSub, matDrawResult, matSrcImageData, matAvgValue, matOrgValue, matMaxValue, matStdValue, matPCRValue, matHISTCount, vHistData, dPara, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Test_SubResultCheck.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDrawResult);
	}

	// 	//////////////////////////////////////////////////////////////////////////
	// 	// Diff Data 
	// 	cv::Mat matDiff = cMatBufTemp.GetMat(matReFitBuffSub.size(), matReFitBuffSub.type(), false);
	// 	matDiff = matReFitBuffSub - matDrawResult;
	// 
	// 	if (bImageSave)
	// 	{
	// 		CString strTemp;
	// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_Test_SubResult_Diff.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
	// 		ImageSave(strTemp, matDiff);
	// 	}
	// 
	// 	matDiff.copyTo(matDiffTemp);

		//////////////////////////////////////////////////////////////////////////
	cv::Scalar ResultMean, ResultStd;

	cv::meanStdDev(matDrawResult, ResultMean, ResultStd);

	double dbAvg = ResultMean[0];
	double dbStd = ResultStd[0];

	double dbTHValue = dbAvg + dbStd;

	/*	if (dbTHValue <= 1.0) dbTHValue = 1.0;*/

	cv::threshold(matDrawResult, matDrawResult, dbTHValue, 150.0, THRESH_BINARY);

	int nOpenUnit = 5;

	Morphology(matDrawResult, matDrawResult, nOpenUnit, nOpenUnit, EP_MORP_MURANORMAL_OPEN, &cMatBufTemp);
	Morphology(matDrawResult, matDrawResult, nOpenUnit, nOpenUnit, EP_MORP_MURANORMAL_DILATE, &cMatBufTemp);

	matDrawResult.copyTo(matDstImage);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return  E_ERROR_CODE_TRUE;
}

// long CInspectMuraNormal::RoundConverlution(cv::Mat& matSrcImage, cv::Mat& matDstImage, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
// {
//	//异常处理
// 	if (dPara == NULL) return false;
// 
// 	//////////////////////////////////////////////////////////////////////////
//	//公共参数
// 	int		nMaxDefectCount				= nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT						];
// 	bool	bImageSave					= (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG						] > 0) ? true : false;
// 	int&	nSaveImageCount				= nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT						];
// 	int		nImageNum					= nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER						];
// 	int		nCamNum						= nCommonPara[E_PARA_COMMON_CAMERA_NUMBER							];
// 	int		nROINumber					= nCommonPara[E_PARA_COMMON_ROI_NUMBER								];
// 	int		nAlgorithmNumber			= nCommonPara[E_PARA_COMMON_ALG_NUMBER								];
// 	int		nThrdIndex					= nCommonPara[E_PARA_COMMON_THREAD_ID								];
// 	int		nPS							= nCommonPara[E_PARA_COMMON_PS_MODE									];
// 	int		nImageUI					= nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER							];
// 
// 	//////////////////////////////////////////////////////////////////////////
// 	int nKernelSize_Big					= (int) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_KERNEL_SIZE_BIG		];	// 30
// 	int nKernelRoundWidth_Big			= (int) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_ROUND_WIDTH_BIG		];	// 10
// 	int nKernelSize_Mid					= (int) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_KERNEL_SIZE_MID		];	// 30
// 	int nKernelRoundWidth_Mid			= (int) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_ROUND_WIDTH_MID		];	// 10
// 	int nKernelSize_Small				= (int) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_KERNEL_SIZE_SMALL	];	// 30
// 	int nKernelRoundWidth_Small			= (int) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_ROUND_WIDTH_SMALL	];	// 10
// 
// 	float nTH_GV_BigRing				= (float) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_BIGRING	];	// 40
// 	float nTH_GV_MidRing				= (float) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_MIDRING	];	// 40
// 	float nTH_GV_SmallRing				= (float) dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_SMALLRING	];	// 40
// 	int	  nMophSize						= (int)	  dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_MOPHO_SIZE			];	// 40
// 
// 	CMatBuf cMatBufTemp;
// 	cMatBufTemp.SetMem(cMemSub);
// 
// 	cv::Mat matKernel_Mid_Big			= cv::Mat::zeros(nKernelSize_Big   * 2 + 1,	nKernelSize_Big   * 2 + 1,	CV_8UC1);
// 	cv::Mat matKernel_Cen_Big			= cv::Mat::zeros(nKernelSize_Big   * 2 + 1,	nKernelSize_Big   * 2 + 1,	CV_8UC1);
// 	cv::Mat matKernel_Mid_Mid			= cv::Mat::zeros(nKernelSize_Mid   * 2 + 1,	nKernelSize_Mid   * 2 + 1,	CV_8UC1);
// 	cv::Mat matKernel_Cen_Mid			= cv::Mat::zeros(nKernelSize_Mid   * 2 + 1,	nKernelSize_Mid	  * 2 + 1,	CV_8UC1);
// 	cv::Mat matKernel_Mid_Small			= cv::Mat::zeros(nKernelSize_Small * 2 + 1, nKernelSize_Small * 2 + 1,	CV_8UC1);
// 	cv::Mat matKernel_Cen_Small			= cv::Mat::zeros(nKernelSize_Small * 2 + 1, nKernelSize_Small * 2 + 1,	CV_8UC1);
// 
// 	cv::Mat MatConResult_Mid_Big		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_Cen_Big		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_Mid_Mid		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_Cen_Mid		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_Mid_Small		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_Cen_Small		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 
// 	cv::Mat MatConResult_TH_BigRing		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_TH_MidRing		= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 	cv::Mat MatConResult_TH_SmallRing	= cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 
// 	cv::Mat MatConResult_TH_Buff = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);
// 
// 	// Set Kernel
// 	SetRoundKernel(matKernel_Mid_Big,	matKernel_Cen_Big,		nKernelSize_Big,	nKernelRoundWidth_Big	);
// 	SetRoundKernel(matKernel_Mid_Mid,	matKernel_Cen_Mid,		nKernelSize_Mid,	nKernelRoundWidth_Mid	);
// 	SetRoundKernel(matKernel_Mid_Small, matKernel_Cen_Small,	nKernelSize_Small,  nKernelRoundWidth_Small	);
// 
// 	if (bImageSave)
// 	{
// 		CString strTemp;
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_RoundCon_Kernel_MB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, matKernel_Mid_Big);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_RoundCon_Kernel_CB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, matKernel_Cen_Big);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_RoundCon_Kernel_MM.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, matKernel_Mid_Mid);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_RoundCon_Kernel_CM.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, matKernel_Cen_Mid);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_RoundCon_Kernel_MS.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, matKernel_Mid_Small);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_RoundCon_Kernel_CS.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, matKernel_Cen_Small);
// 	}
// 
// 	// Convolution
// 	setKernelConvolution(matSrcImage, matKernel_Mid_Big,	MatConResult_Mid_Big,	&cMatBufTemp);
// 	setKernelConvolution(matSrcImage, matKernel_Cen_Big,	MatConResult_Cen_Big,	&cMatBufTemp);
// 	setKernelConvolution(matSrcImage, matKernel_Mid_Mid,	MatConResult_Mid_Mid,	&cMatBufTemp);
// 	setKernelConvolution(matSrcImage, matKernel_Cen_Mid,	MatConResult_Cen_Mid,	&cMatBufTemp);
// 	setKernelConvolution(matSrcImage, matKernel_Mid_Small,	MatConResult_Mid_Small, &cMatBufTemp);
// 	setKernelConvolution(matSrcImage, matKernel_Cen_Small,	MatConResult_Cen_Small, &cMatBufTemp);
// 
// 	if (bImageSave)
// 	{
// 		CString strTemp;
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Result_MB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_Mid_Big);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Result_CB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_Cen_Big);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Result_MM.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_Mid_Mid);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Result_CM.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_Cen_Mid);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Result_MS.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_Mid_Small);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Result_CS.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_Cen_Small);
// 	}
// 
// 	//////////////////////////////////////////////////////////////////////////
// 	// Thresholding
// 	setSubThresholding(MatConResult_Cen_Big,	MatConResult_Mid_Big,	MatConResult_TH_BigRing,	nTH_GV_BigRing,		nCommonPara, strAlgPath, &cMatBufTemp);
// 	setSubThresholding(MatConResult_Cen_Mid,	MatConResult_Mid_Mid,	MatConResult_TH_MidRing,	nTH_GV_MidRing,		nCommonPara, strAlgPath, &cMatBufTemp);
// 	setSubThresholding(MatConResult_Cen_Small,	MatConResult_Mid_Small, MatConResult_TH_SmallRing,	nTH_GV_SmallRing,	nCommonPara, strAlgPath, &cMatBufTemp);
// 
// 	if (bImageSave)
// 	{
// 		CString strTemp;
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_TH_BigRing.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_TH_BigRing);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_TH_MidRing.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_TH_MidRing);
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_TH_SmallRing.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_TH_SmallRing);
// 	}
// 
// 	cv::bitwise_or(MatConResult_TH_BigRing, MatConResult_TH_MidRing,	MatConResult_TH_Buff);
// 	cv::bitwise_or(MatConResult_TH_Buff,	MatConResult_TH_SmallRing,	MatConResult_TH_BigRing);
// 
// 	// Mophology
// 	Morphology(MatConResult_TH_BigRing, MatConResult_TH_MidRing, nMophSize, nMophSize, EP_MORP_MURANORMAL_CLOSE, &cMatBufTemp);
// 
// 	if (bImageSave)
// 	{
// 		CString strTemp;
// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_MophResult.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, MatConResult_TH_MidRing);
// 	}
// 
// 	MatConResult_TH_MidRing.copyTo(matDstImage);
// 	
// 	return  E_ERROR_CODE_TRUE;
// }

long	CInspectMuraNormal::SetRoundKernel(cv::Mat& matKernel_Mid, cv::Mat& matKernel_Center, int nKernelSize, int nRoundWidth)
{
	if (nKernelSize == 0 || nRoundWidth == 0) return false;

	CPoint cpKernelCenter;
	cpKernelCenter.x = (int)(nKernelSize)+1;
	cpKernelCenter.y = (int)(nKernelSize)+1;

	for (int i = 0; i < nKernelSize * 2; i++)
	{
		for (int j = 0; j < nKernelSize * 2; j++)
		{
			int ndy = j - cpKernelCenter.x;
			int ndx = i - cpKernelCenter.y;

			int fTempLength = (int)sqrt(ndy * ndy + ndx * ndx);

			if (fTempLength <= nKernelSize - nRoundWidth && fTempLength > nKernelSize - nRoundWidth * 2)	matKernel_Mid.at<uchar>(j, i) = (uchar)1;
			else if (fTempLength <= nKernelSize - nRoundWidth * 2)	matKernel_Center.at<uchar>(j, i) = (uchar)1;
			else																										continue;
		}
	}

	return  E_ERROR_CODE_TRUE;
}

long	CInspectMuraNormal::setKernelConvolution(cv::Mat& matSrcImage, cv::Mat& matKernel, cv::Mat& matDstImage, CMatBuf* cMemSub)
{
	if (matSrcImage.empty())	return false;
	if (matKernel.empty())		return false;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat MatTempg = cMatBufTemp.GetMat(matSrcImage.size(), CV_32FC1, false);

	cv::filter2D(matSrcImage, MatTempg, CV_32FC1, matKernel);

	double minValue, maxValue;
	minMaxLoc(MatTempg, &minValue, &maxValue);

	MatTempg -= minValue;
	MatTempg.convertTo(matDstImage, CV_8UC1, 255.0 / (maxValue - minValue));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return  E_ERROR_CODE_TRUE;
}

long	CInspectMuraNormal::setSubThresholding(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matDstImage, float nThresholdGV, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//异常处理
	if (nThresholdGV == 0)		return false;
	if (matSrcImage1.empty())	return false;
	if (matSrcImage2.empty())	return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat MatTempBuff1 = cMatBufTemp.GetMat(matDstImage.size(), CV_8UC1);
	cv::Mat MatTempBuff2 = cMatBufTemp.GetMat(matDstImage.size(), CV_8UC1);
	cv::Mat MatTempBuff3 = cMatBufTemp.GetMat(matDstImage.size(), CV_8UC1);

	cv::subtract(matSrcImage1, matSrcImage2, MatTempBuff1);
	cv::subtract(matSrcImage2, matSrcImage1, MatTempBuff2);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_CM_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, MatTempBuff1);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_MC_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, MatTempBuff2);
	}

	MatTempBuff3 = MatTempBuff1 + MatTempBuff2;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_BBS_Sum_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, MatTempBuff3);
	}

	double dblMean;
	cv::Scalar m = cv::mean(MatTempBuff3);
	dblMean = m[0];

	cv::threshold(MatTempBuff3, MatTempBuff1, (double)dblMean * nThresholdGV, 150.0, THRESH_BINARY);

	MatTempBuff1.copyTo(matDstImage);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return  E_ERROR_CODE_TRUE;
}

// Feature Extraction ROI Process
void CInspectMuraNormal::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset)
{
	rectInspROI.x = rectROI.left - nOffset;
	rectInspROI.y = rectROI.top - nOffset;
	rectInspROI.width = rectROI.Width() + nOffset * 2;
	rectInspROI.height = rectROI.Height() + nOffset * 2;

	if (rectInspROI.x < 0) rectInspROI.x = 0;
	if (rectInspROI.y < 0) rectInspROI.y = 0;
	if (rectInspROI.width > nWidth - rectInspROI.x) rectInspROI.width = nWidth - rectInspROI.x;
	if (rectInspROI.height > nHeight - rectInspROI.y) rectInspROI.height = nHeight - rectInspROI.y;
}

//////////////////////////////////////////////////////////////////////////
// 2019-04-22 PNZ
//目的:Type 1不良检查算法(重新开发)
//版本:ver 1.0.0
//备注
//	-修改预处理->检查方式
//////////////////////////////////////////////////////////////////////////

long CInspectMuraNormal::LogicStart_MuraType1(cv::Mat& matSrcImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
	stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	// Parameter
	int		nShiftCopy = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SHIFT_UNIT];
	int		nCutUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_CUT_UNIT];
	int		nCutUnit_ADD = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_ADD_UNIT];
	int		nImageBlur = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_MEANBLUR];
	int		nResize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_RESIZE];
	int		nResultMorph = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_RESULT_MORPH]; // 5 Result Morph

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process
	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 Start."));

	// Parameter Setting
	cv::Rect InputROI = cv::Rect(rectROI.left + nCutUnit + nCutUnit_ADD, rectROI.top + nCutUnit, rectROI.Width() - nCutUnit * 2 - nCutUnit_ADD, rectROI.Height() - nCutUnit * 2);
	cv::Rect ProcessROI = cv::Rect(nCutUnit + nCutUnit_ADD, nCutUnit, rectROI.Width() - nCutUnit * 2 - nCutUnit_ADD, rectROI.Height() - nCutUnit * 2);

	int nRectWidth = ProcessROI.width;
	int nRectHeight = ProcessROI.height;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[0]);

	// Image Buff
	cv::Mat matSrcROIBuf = cMatBufTemp.GetMat(nRectWidth, nRectHeight, CV_8UC1);
	cv::Mat matSrcReBuf = cMatBufTemp.GetMat(nRectWidth / nResize, nRectHeight / nResize, CV_8UC1);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 Get Image Buff."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_InputImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	matSrcROIBuf = matSrcImage(ProcessROI);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 Select ROI Image."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	// Resize
	cv::resize(matSrcROIBuf, matSrcReBuf, cv::Size(nRectWidth / nResize, nRectHeight / nResize), 0, 0, INTER_CUBIC);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 Resize."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_Resize_ROIImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcReBuf);
	}

	//////////////////////////////////////////////////////////////////////////
	// Type 1 Main Processing
	cv::Mat matDstBuf[E_DEFECT_LINEMURA_TOTAL];
	cv::Mat matLocationBuf[E_DEFECT_LINEMURA_TOTAL];

	matDstBuf[E_DEFECT_LINEMURA_X] = cMatBufTemp.GetMat(matSrcReBuf.size(), CV_8UC1);
	matDstBuf[E_DEFECT_LINEMURA_Y] = cMatBufTemp.GetMat(matSrcReBuf.size(), CV_8UC1);

	MainProcessingT1(matSrcReBuf, matDstBuf, rectROI, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 Main Process."));

	//////////////////////////////////////////////////////////////////////////
	// Draw Result

	// Morphology
	Morphology(matDstBuf[E_DEFECT_LINEMURA_X], matDstBuf[E_DEFECT_LINEMURA_X], nResultMorph, nResultMorph, EP_MORP_MURANORMAL_CLOSE);
	Morphology(matDstBuf[E_DEFECT_LINEMURA_Y], matDstBuf[E_DEFECT_LINEMURA_Y], nResultMorph, nResultMorph, EP_MORP_MURANORMAL_CLOSE);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 Morphology."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_X_Defect_Morph.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuf[E_MURANORMAL_TYPE1_DEFECT_X]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_Y_Defect_Morph.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuf[E_MURANORMAL_TYPE1_DEFECT_Y]);
	}

	// Find & Draw Result
	FindDrawResult(matDstBuf, pResultBlob, InputROI, dPara, nResize, nImageUI, nPS);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Type 1 DefectResultSet & End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode = E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::MainProcessingT1(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Main Process Start."));

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	float	fPow = 110;																// 放大(影像的平均校正功能)固定Para
	int		nBlurSize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_PROFILE_BLUR];	// 15 Profile Blur

	float	fDarkXThValue = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_ACTIVE_TH_X_VALUE];	// 5.0 Defect Check X Direction TH
	float	fDarkYThValue = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_ACTIVE_TH_Y_VALUE];	// 3.0 Defect Check Y Direction TH
	int		nEdgeUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_EDGE_UNIT];	// 100 Edge Unit
	float	fEdge_ThValue_X = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_EDGE_TH_X_VALUE];	// -3.0 Edge Para X
	float	fEdge_ThValue_Y = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_EDGE_TH_Y_VALUE];	// -3.0 Edge Para X

	float	fSUMTempTHX = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTH_X];	// 0 SUM TH X
	float	fSUMTempTHY = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTH_Y];	// 0 SUM TH Y
	int		nSUMTypeX = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTYPE_X];	// 1 SUM Type X
	int		nSUMTypeY = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_SUMTYPE_Y];	// 1 SUM Type Y

	int		nWidth = matSrcImage.cols;
	int		nHeight = matSrcImage.rows;

	//////////////////////////////////////////////////////////////////////////
		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//////////////////////////////////////////////////////////////////////////
		//创建Image Buff
	cv::Mat matType1Defect_Active[E_DEFECT_LINEMURA_TOTAL], matType1Defect_Edge[E_DEFECT_LINEMURA_TOTAL];

	matType1Defect_Active[E_DEFECT_LINEMURA_X] = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);		// Mat 8bit Final Result Buff
	matType1Defect_Active[E_DEFECT_LINEMURA_Y] = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);		// Mat 8bit Final Result Buff
	matType1Defect_Edge[E_DEFECT_LINEMURA_X] = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);		// Mat 8bit Final Result Buff
	matType1Defect_Edge[E_DEFECT_LINEMURA_Y] = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1);		// Mat 8bit Final Result Buff

	cv::Mat matPowBuf16 = cMatBufTemp.GetMat(matSrcImage.size(), CV_16UC1);		// Mat 16bit Buff
	cv::Mat matProfileBufX = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_16UC1);		// Mat 16bit Profile Buff
	cv::Mat matProfileBufY = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_16UC1);		// Mat 16bit Profile Buff
	cv::Mat matProBlurX = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_16UC1);		// Mat 16bit Profile Buff
	cv::Mat matProBlurY = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_16UC1);		// Mat 16bit Profile Buff

	cv::Mat matProSlopeXDark7 = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeYDark7 = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeXDark15 = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeYDark15 = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeXDark25 = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeYDark25 = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeXDark50 = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeYDark50 = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeXDark75 = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeYDark75 = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeXDarkSUM = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff
	cv::Mat matProSlopeYDarkSUM = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Dark Profile Buff

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Image Buff."));

	//////////////////////////////////////////////////////////////////////////
		//16bit&放大(自动亮度校正)
	float fMean = cv::mean(matSrcImage)[0];
	float fAutoPow = log(fPow * 16.0) / log(fMean);

	AlgoBase::Pow(matSrcImage, matPowBuf16, fAutoPow, 4095, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Pow Process."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matPowBuf16);
	}

	//////////////////////////////////////////////////////////////////////////
	// Image to Profile
	AlgoBase::MakeProfile(matPowBuf16, matProfileBufX, nWidth, nHeight, 2);
	AlgoBase::MakeProfile(matPowBuf16, matProfileBufY, nWidth, nHeight, 3);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Make Profile."));

	// Noise Reduction
	cv::blur(matProfileBufX, matProBlurX, cv::Size(nBlurSize, 1));
	cv::blur(matProfileBufY, matProBlurY, cv::Size(nBlurSize, 1));

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile Noise Reduction."));

	//////////////////////////////////////////////////////////////////////////
	// Slope Calculation

	// Dark Slope Calculation Result
	ProfileSlopeCalc(matProBlurX, matProSlopeXDark7, 7);
	ProfileSlopeCalc(matProBlurY, matProSlopeYDark7, 7);
	ProfileSlopeCalc(matProBlurX, matProSlopeXDark15, 15);
	ProfileSlopeCalc(matProBlurY, matProSlopeYDark15, 15);
	ProfileSlopeCalc(matProBlurX, matProSlopeXDark25, 25);
	ProfileSlopeCalc(matProBlurY, matProSlopeYDark25, 25);
	ProfileSlopeCalc(matProBlurX, matProSlopeXDark50, 50);
	ProfileSlopeCalc(matProBlurY, matProSlopeYDark50, 50);
	ProfileSlopeCalc(matProBlurX, matProSlopeXDark75, 75);
	ProfileSlopeCalc(matProBlurY, matProSlopeYDark75, 75);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile Dark Slope Calc."));

	ProfileSlopeSUM(matProSlopeXDark7, matProSlopeXDark15, matProSlopeXDark25, matProSlopeXDark50, matProSlopeXDark75, fSUMTempTHX, nSUMTypeX, matProSlopeXDarkSUM);
	ProfileSlopeSUM(matProSlopeYDark7, matProSlopeYDark15, matProSlopeYDark25, matProSlopeYDark50, matProSlopeYDark75, fSUMTempTHY, nSUMTypeY, matProSlopeYDarkSUM);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile Dark Slope SUM."));

	double dbSlopeSUMX_MIN = 0;
	double dbSlopeSUMX_MAX = 0;
	double dbSlopeSUMY_MIN = 0;
	double dbSlopeSUMY_MAX = 0;

	cv::minMaxIdx(matProSlopeXDarkSUM, &dbSlopeSUMX_MIN, &dbSlopeSUMX_MAX, NULL, NULL);
	cv::minMaxIdx(matProSlopeYDarkSUM, &dbSlopeSUMY_MIN, &dbSlopeSUMY_MAX, NULL, NULL);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_ProfileX_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "SlopeSUMMax, %.3f\n", dbSlopeSUMX_MAX);

			fprintf_s(out, "Index, Data, Blur, DSlope7, DSlope15, DSlope25, DSlope50, DSlope75, DarkSUM\n");

			for (int i = 0; i < matProBlurX.cols; i++)
			{
				fprintf_s(out, "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
					i, matProfileBufX.at<ushort>(0, i), matProBlurX.at<ushort>(0, i),
					matProSlopeXDark7.at<float>(0, i), matProSlopeXDark15.at<float>(0, i), matProSlopeXDark25.at<float>(0, i),
					matProSlopeXDark50.at<float>(0, i), matProSlopeXDark75.at<float>(0, i), matProSlopeXDarkSUM.at<float>(0, i));
			}

			fclose(out);
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_ProfileY_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "SlopeSUMMax, %.3f\n", dbSlopeSUMY_MAX);

			fprintf_s(out, "Index, Data, Blur, DSlope7, DSlope15, DSlope25, DSlope50, DSlope75, DarkSUM\n");

			for (int i = 0; i < matProBlurY.cols; i++)
			{
				fprintf_s(out, "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
					i, matProfileBufY.at<ushort>(0, i), matProBlurY.at<ushort>(0, i),
					matProSlopeYDark7.at<float>(0, i), matProSlopeYDark15.at<float>(0, i), matProSlopeYDark25.at<float>(0, i),
					matProSlopeYDark50.at<float>(0, i), matProSlopeYDark75.at<float>(0, i), matProSlopeYDarkSUM.at<float>(0, i));
			}

			fclose(out);
		}
	}

	// Find Type 1 Defect ( Active )
	ProfileFindDefect(matProSlopeXDarkSUM, fDarkXThValue, matType1Defect_Active[E_DEFECT_LINEMURA_X], E_DEFECT_LINEMURA_X);
	ProfileFindDefect(matProSlopeYDarkSUM, fDarkYThValue, matType1Defect_Active[E_DEFECT_LINEMURA_Y], E_DEFECT_LINEMURA_Y);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Profile Find Defect_Active."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_TH_X_Active.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matType1Defect_Active[E_DEFECT_LINEMURA_X]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_TH_Y_Active.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matType1Defect_Active[E_DEFECT_LINEMURA_Y]);
	}

	// Find Type 1 Defect ( Edge )
	ProfileFindDefectEdge(matProBlurX, fEdge_ThValue_X, nEdgeUnit, matType1Defect_Edge[E_DEFECT_LINEMURA_X], E_DEFECT_LINEMURA_X, nCommonPara, strAlgPath);
	ProfileFindDefectEdge(matProBlurY, fEdge_ThValue_Y, nEdgeUnit, matType1Defect_Edge[E_DEFECT_LINEMURA_Y], E_DEFECT_LINEMURA_Y, nCommonPara, strAlgPath);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_TH_X_Edge.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matType1Defect_Edge[E_DEFECT_LINEMURA_X]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_TH_Y_Edge.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matType1Defect_Edge[E_DEFECT_LINEMURA_Y]);
	}

	cv::bitwise_or(matType1Defect_Active[E_DEFECT_LINEMURA_X], matType1Defect_Edge[E_DEFECT_LINEMURA_X], matDstImage[E_DEFECT_LINEMURA_X]);
	cv::bitwise_or(matType1Defect_Active[E_DEFECT_LINEMURA_Y], matType1Defect_Edge[E_DEFECT_LINEMURA_Y], matDstImage[E_DEFECT_LINEMURA_Y]);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Add Result."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_TH_X_Result.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_LINEMURA_X]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_TH_Y_Result.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_LINEMURA_Y]);
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::FindDrawResult(cv::Mat* matDstBuf, stDefectInfo* pResultBlob, cv::Rect InputROI, double* dPara, int nResize, int nImageUI, int nPS)
{
	//异常处理
	if (matDstBuf[E_MURANORMAL_TYPE1_DEFECT_X].empty() || matDstBuf[E_MURANORMAL_TYPE1_DEFECT_Y].empty()) return false;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("FindDrawResult Start."));

	//数据初始化
	vector<vector<cv::Point>>	ptDefectContours_X;
	vector<vector<cv::Point>>	ptDefectContours_Y;
	vector<vector<cv::Point>>	ptDefectContours_TOTAL;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Vector Reset."));

	// Find Contour & Location Set
	FindnCalContours(matDstBuf[E_MURANORMAL_TYPE1_DEFECT_X], InputROI, nResize, ptDefectContours_X);
	FindnCalContours(matDstBuf[E_MURANORMAL_TYPE1_DEFECT_Y], InputROI, nResize, ptDefectContours_Y);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("FindnCalContours."));

	// Result Add
	AddDefectContours(ptDefectContours_X, ptDefectContours_Y, ptDefectContours_TOTAL);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("AddDefectContours."));

	// Draw Result
	DrawDefectContours(ptDefectContours_TOTAL, pResultBlob, dPara, nImageUI, nPS);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("DrawDefectContours."));

	// Release Data
	vector<vector<cv::Point>>().swap(ptDefectContours_X);
	vector<vector<cv::Point>>().swap(ptDefectContours_Y);
	vector<vector<cv::Point>>().swap(ptDefectContours_TOTAL);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("FindDrawResult End."));

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::FindnCalContours(cv::Mat& matSrcBuf, cv::Rect InputROI, int nResize, vector<vector<cv::Point>>& ptDefectContours)
{
	//异常处理
	if (matSrcBuf.empty())																	return false;
	if (InputROI.x == 0 && InputROI.y == 0 && InputROI.width == 0 && InputROI.height == 0)	return false;

	//拯救Contours
	vector<vector<cv::Point>>		 ptCalContours;
	vector<vector<cv::Point>>().swap(ptCalContours);

	cv::findContours(matSrcBuf, ptCalContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	if (ptCalContours.size() <= 0) return false;

	//Contours计数
	for (int m = 0; m < ptCalContours.size(); m++)
	{
		// Buff Reset
		vector<cv::Point>		 ptBuffVector;

		//更改为Resize原始坐标
		for (int k = 0; k < ptCalContours.at(m).size(); k++)
		{
			ptBuffVector.push_back(cv::Point(ptCalContours.at(m)[k].x * nResize + InputROI.x, ptCalContours.at(m)[k].y * nResize + InputROI.y));
		}

		ptDefectContours.push_back(ptBuffVector);
		vector<cv::Point>().swap(ptBuffVector);
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::AddDefectContours(vector<vector<cv::Point>>& ptDefect_X, vector<vector<cv::Point>>& ptDefect_Y, vector<vector<cv::Point>>& ptDefectALL)
{

	if (ptDefect_X.size() == 0 && ptDefect_Y.size() == 0) return false;

	// Parameter
	int nDefectNum_X = ptDefect_X.size();
	int nDefectNum_Y = ptDefect_Y.size();

	//检查Input Data状态并发送数据

	//只有Case1:X出现故障时
	if (nDefectNum_X > 0 && nDefectNum_Y == 0)
	{
		for (int i = 0; i < nDefectNum_X; i++)
		{
			ptDefectALL.push_back(ptDefect_X.at(i));
		}
	}

	//Case2:Y只有坏的时候
	if (nDefectNum_X == 0 && nDefectNum_Y > 0)
	{
		for (int i = 0; i < nDefectNum_Y; i++)
		{
			ptDefectALL.push_back(ptDefect_Y.at(i));
		}
	}

	//Case3全部存在时
	if (nDefectNum_X > 0 && nDefectNum_Y > 0)
	{
		int nTotal = nDefectNum_X + nDefectNum_Y;

		for (int m = 0; m < nDefectNum_X; m++)
		{
			ptDefectALL.push_back(ptDefect_X.at(m));
		}

		for (int m = 0; m < nDefectNum_Y; m++)
		{
			ptDefectALL.push_back(ptDefect_Y.at(m));
		}

		if (ptDefectALL.size() != nTotal) return false;
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::DrawDefectContours(vector<vector<cv::Point>>& ptDefectContours_TOTAL, stDefectInfo* pResultBlob, double* dPara, int nImageUI, int nPS)
{
	//异常处理
	if (ptDefectContours_TOTAL.size() <= 0) return false;
	if (pResultBlob == NULL) return false;

	// Parameters
	int  nMinDefectWidth = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE1_JUDGE_MIN_DEFECT_WIDTH];	// 50 Min Defect Unit
	int  nDefectNumber = (int)ptDefectContours_TOTAL.size();

	// Draw Data
	for (int i = 0; i < nDefectNumber; i++)
	{
		//确认事件
		int  nCountNumber = pResultBlob->nDefectCount;
		int  nDefectIndex = 0;
		bool bOnlyType1Defect = true;

		if (nCountNumber == 0)	bOnlyType1Defect = true;
		else					bOnlyType1Defect = false;

		// Defect Index Calc.
		if (bOnlyType1Defect == true) nDefectIndex = 0;
		else if (bOnlyType1Defect == false) nDefectIndex = nCountNumber;

		//////////////////////////////////////////////////////////////////////////
		// Parameters
		double Xmax = 0;
		double Xmin = 0;
		double Ymax = 0;
		double Ymin = 0;

		//数据计算
		CalcDataMinMax(ptDefectContours_TOTAL.at(i), Xmax, Xmin, Ymax, Ymin);

		int nXLength = Xmax - Xmin;
		int nYLength = Ymax - Ymin;

		//////////////////////////////////////////////////////////////////////////
		// Defect Name Change
		// X Defect
		if (nXLength > nYLength)
		{
			if (nYLength <= nMinDefectWidth)
				continue;
			else
			{
				pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_X;
			}
		}

		// Y Defect
		if (nXLength < nYLength)
		{
			if (nXLength <= nMinDefectWidth)
				continue;
			else
			{
				pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURA_TYPE1MURA_Y;
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// Basic Feature Info Write
		pResultBlob->nArea[nDefectIndex] = 0;
		pResultBlob->nMaxGV[nDefectIndex] = 0;
		pResultBlob->nMinGV[nDefectIndex] = 0;
		pResultBlob->dMeanGV[nDefectIndex] = 0;
		pResultBlob->dBackGroundGV[nDefectIndex] = 0;
		pResultBlob->dCompactness[nDefectIndex] = 0;
		pResultBlob->dSigma[nDefectIndex] = 0;
		pResultBlob->dBreadth[nDefectIndex] = 0;
		pResultBlob->dF_Min[nDefectIndex] = 0;
		pResultBlob->dF_Max[nDefectIndex] = 0;
		pResultBlob->dF_Elongation[nDefectIndex] = 0;
		pResultBlob->dCompactness[nDefectIndex] = 0;
		pResultBlob->nDefectColor[nDefectIndex] = E_DEFECT_COLOR_DARK;
		pResultBlob->nPatternClassify[nDefectIndex] = nImageUI;

		//Defect坐标
		pResultBlob->ptLT[nDefectIndex].x = Xmin;
		pResultBlob->ptLT[nDefectIndex].y = Ymin;
		pResultBlob->ptLB[nDefectIndex].x = Xmin;
		pResultBlob->ptLB[nDefectIndex].y = Ymax;
		pResultBlob->ptRT[nDefectIndex].x = Xmax;
		pResultBlob->ptRT[nDefectIndex].y = Ymin;
		pResultBlob->ptRB[nDefectIndex].x = Xmax;
		pResultBlob->ptRB[nDefectIndex].y = Ymax;

		// Contour Sampling
		int		nDefectLength = ptDefectContours_TOTAL.at(i).size();
		float	fRatio = 1.0;

		if (nDefectLength >= MAX_CONTOURS)
		{
			fRatio = nDefectLength / (float)MAX_CONTOURS;
			nDefectLength = MAX_CONTOURS;
		}

		//保存轮廓
		for (int j = 0; j < nDefectLength; j++)
		{
			int index = (int)(j * fRatio);

			pResultBlob->nContoursX[nDefectIndex][j] = (int)(ptDefectContours_TOTAL.at(i)[index].x / nPS);
			pResultBlob->nContoursY[nDefectIndex][j] = (int)(ptDefectContours_TOTAL.at(i)[index].y / nPS);
		}

		//添加不良
		pResultBlob->nDefectCount = nDefectIndex + 1;
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::CalcDataMinMax(vector<cv::Point>& ptDefectContour, double& nXmax, double& nXmin, double& nYmax, double& nYmin)
{
	if (ptDefectContour.size() <= 0) return false;

	cv::Mat matXBuff = cv::Mat::zeros(cv::Size(ptDefectContour.size(), 1), CV_16UC1);
	cv::Mat matYBuff = cv::Mat::zeros(cv::Size(ptDefectContour.size(), 1), CV_16UC1);

	for (int i = 0; i < ptDefectContour.size(); i++)
	{
		matXBuff.at<ushort>(0, i) = ptDefectContour[i].x;
		matYBuff.at<ushort>(0, i) = ptDefectContour[i].y;
	}

	cv::minMaxIdx(matXBuff, &nXmin, &nXmax);
	cv::minMaxIdx(matYBuff, &nYmin, &nYmax);

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ProfileSlopeCalc(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nSlopCalceUnite)
{
	//异常处理
	if (matSrcImage.empty() || nSlopCalceUnite == 0) return false;

	//初始化
	ushort* ptrB = (ushort*)matSrcImage.ptr(0);							// Data左侧Data
	ushort* ptrData = (ushort*)matSrcImage.ptr(0) + nSlopCalceUnite;		// Data
	ushort* ptrF = (ushort*)matSrcImage.ptr(0) + nSlopCalceUnite * 2;	// Data右侧Data
	float* ptrResult = (float*)matDstImage.ptr(0);							// Result Data

	for (int i = 0; i < matSrcImage.cols; i++, ptrResult++)
	{
		// Profile Edge Offset
		if (i >= nSlopCalceUnite && i < matSrcImage.cols - nSlopCalceUnite)
		{
			// Slope Calculation
			float nForeSlope = (float)(*ptrF - *ptrData) / nSlopCalceUnite;
			float nBackSlope = (float)(*ptrData - *ptrB) / nSlopCalceUnite;

			float fCalcuValue = 0;

			// Case 1 : Dark Slope Calculation
			if (nForeSlope > 0 && nBackSlope < 0)	fCalcuValue = (float)(abs(nForeSlope) + abs(nBackSlope));
			else									fCalcuValue = (float)0;

			*ptrResult = fCalcuValue;

			//下一个Data
			ptrF++; ptrB++;	ptrData++;
		}

		else
			*ptrResult = 0.0;
	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ProfileSlopeSUM(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matSrcImage3, cv::Mat& matSrcImage4, cv::Mat& matSrcImage5, float fSUMTempTH, int nSUMType, cv::Mat& matDstImage)
{
	//异常处理
	if (matSrcImage1.empty() || matSrcImage2.empty() || matSrcImage3.empty() || matSrcImage4.empty() || matSrcImage5.empty()) return false;

	//数据初始化
	float* prtSlope07 = (float*)matSrcImage1.ptr(0);	// Slope 7
	float* prtSlope15 = (float*)matSrcImage2.ptr(0);	// Slope 15
	float* prtSlope25 = (float*)matSrcImage3.ptr(0);	// Slope 25
	float* prtSlope50 = (float*)matSrcImage4.ptr(0);	// Slope 50
	float* prtSlope75 = (float*)matSrcImage5.ptr(0);	// Slope 75
	float* prtResult = (float*)matDstImage.ptr(0);		// Result

	// Parameter
	float fTempValue07, fTempValue15, fTempValue25, fTempValue50, fTempValue75;

	// Mat Sum
	for (int i = 0; i < matDstImage.cols; i++, prtSlope07++, prtSlope15++, prtSlope25++, prtSlope50++, prtSlope75++, prtResult++)
	{
		// Case 1 : SUM
		if (nSUMType == 1)
			*prtResult = *prtSlope07 + *prtSlope15 + *prtSlope25 + *prtSlope50 + *prtSlope75;

		// Case 2 : TH + SUM
		else if (nSUMType == 2)
		{
			// Thresholding
			if (*prtSlope07 < fSUMTempTH) fTempValue07 = 0; else fTempValue07 = *prtSlope07;
			if (*prtSlope15 < fSUMTempTH) fTempValue15 = 0; else fTempValue15 = *prtSlope15;
			if (*prtSlope25 < fSUMTempTH) fTempValue25 = 0; else fTempValue25 = *prtSlope25;
			if (*prtSlope50 < fSUMTempTH) fTempValue50 = 0; else fTempValue50 = *prtSlope50;
			if (*prtSlope75 < fSUMTempTH) fTempValue75 = 0; else fTempValue75 = *prtSlope75;

			//SUM
			*prtResult = fTempValue07 + fTempValue15 + fTempValue25 + fTempValue50 + fTempValue75;
		}

		// Case 3 : TH + Filter + SUM
		else if (nSUMType == 3)
		{
			// Thresholding
			if (*prtSlope07 < fSUMTempTH) fTempValue07 = 0; else fTempValue07 = *prtSlope07;
			if (*prtSlope15 < fSUMTempTH) fTempValue15 = 0; else fTempValue15 = *prtSlope15;
			if (*prtSlope25 < fSUMTempTH) fTempValue25 = 0; else fTempValue25 = *prtSlope25;
			if (*prtSlope50 < fSUMTempTH) fTempValue50 = 0; else fTempValue50 = *prtSlope50;
			if (*prtSlope75 < fSUMTempTH) fTempValue75 = 0; else fTempValue75 = *prtSlope75;

			// Filtering
			if (fTempValue15 >= 0 && fTempValue25 >= 0 && fTempValue50 == 0)	*prtResult = 0;

			// SUM
			else
				*prtResult = fTempValue07 + fTempValue15 + fTempValue25 + fTempValue50 + fTempValue75;
		}

		else
			*prtResult = 0;

	}

	return  E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::ProfileDefectCheckLogic(cv::Mat& matSrcImage, float nThresholdValue)
{
	//异常处理
	if (matSrcImage.empty() || nThresholdValue == 0) return false;

	//////////////////////////////////////////////////////////////////////////
	// Fast Check
	double minValue, maxValue;
	minMaxLoc(matSrcImage, &minValue, &maxValue);

	if (maxValue < nThresholdValue)	return	false;	// 没有不良
	else							return  true;	// 有不良
}

long  CInspectMuraNormal::ProfileFindDefect(cv::Mat& matSrcImageDark, float fThresholdValue, cv::Mat& matResult, int nDefectDirection)
{
	//异常处理
	if (matSrcImageDark.empty() || fThresholdValue == 0) return false;

	//////////////////////////////////////////////////////////////////////////
	// Find Defect

	// Parameter
	int		nImageWidth = matResult.cols;
	int		nImageHeight = matResult.rows;

	// Data Reset
	float* ptrDark = (float*)matSrcImageDark.ptr(0);

	cv::Point cpLT, cpRT, cpLB, cpRB;

	//////////////////////////////////////////////////////////////////////////
	// Defect Find Logic

	// Active Process
	for (int i = 0; i < matSrcImageDark.cols; i++, ptrDark++)
	{
		// Reset
		bool bDefectDraw = false;

		// Defect Judge
		if ((float)*ptrDark >= fThresholdValue) bDefectDraw = true;

		// Defect Draw
		if (bDefectDraw)
		{
			if (nDefectDirection == E_DEFECT_LINEMURA_X)
			{
				cpLT.x = i;				cpLT.y = 0;
				cpLB.x = i;				cpLB.y = nImageHeight;

				cv::line(matResult, cpLT, cpLB, cv::Scalar(50), 1);
			}

			else if (nDefectDirection == E_DEFECT_LINEMURA_Y)
			{
				cpLT.x = 0;				cpLT.y = i;
				cpRT.x = nImageWidth;	cpRT.y = i;

				cv::line(matResult, cpLT, cpRT, cv::Scalar(50), 1);
			}
		}

		else
			continue;
	}

	return E_ERROR_CODE_TRUE;
}

long  CInspectMuraNormal::ProfileFindDefectEdge(cv::Mat& matSrcImageDark, float fThresholdValue, int nEdgeUnit, cv::Mat& matResult, int nDefectDirection, int* nCommonPara, CString strAlgPath)
{
	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//异常处理
	if (matSrcImageDark.empty() || fThresholdValue == 0 || nEdgeUnit <= 0) return false;

	// Parameter
	int		nImageWidth = matResult.cols;
	int		nImageHeight = matResult.rows;
	int		nDataLength = matSrcImageDark.cols - nEdgeUnit;
	int		nTopDefectStart = 0;
	int		nBottomDefectStart = 0;
	float	fNegativeTH = fThresholdValue;
	float	fPositiveTH = fThresholdValue;

	cv::Point cpLT, cpRT, cpLB, cpRB;

	// Select Region
	cv::Mat matTopRegion = matSrcImageDark(cv::Rect(0, 0, nEdgeUnit, 1));
	cv::Mat matBottomRegion = matSrcImageDark(cv::Rect(matSrcImageDark.cols - nEdgeUnit, 0, nEdgeUnit, 1));
	cv::Mat matTopRegionSlop = cv::Mat::zeros(matTopRegion.size(), CV_32FC1);
	cv::Mat matBottomRegionSlop = cv::Mat::zeros(matBottomRegion.size(), CV_32FC1);

	// Find Defect Position
	NegativeSlopeDataCheck(matTopRegion, fNegativeTH, nEdgeUnit, nTopDefectStart, matTopRegionSlop);
	PositiveSlopeDataCheck(matBottomRegion, fPositiveTH, nDataLength, nBottomDefectStart, matBottomRegionSlop);

	double dbSlopeTop_MIN = 0;
	double dbSlopeTop_MAX = 0;
	double dbSlopeBottom_MIN = 0;
	double dbSlopeBottom_MAX = 0;

	cv::minMaxIdx(matTopRegionSlop, &dbSlopeTop_MIN, &dbSlopeTop_MAX, NULL, NULL);
	cv::minMaxIdx(matBottomRegionSlop, &dbSlopeBottom_MIN, &dbSlopeBottom_MAX, NULL, NULL);

	// Save Data
	if (bImageSave)
	{
		CString strTemp;

		if (nDefectDirection == E_DEFECT_LINEMURA_X) { strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_EdgeDataX.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++); }
		if (nDefectDirection == E_DEFECT_LINEMURA_Y) { strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type1_EdgeDataY.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++); }

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "TopMIN, %.3f, BottomMIN, %.3f\n", dbSlopeTop_MIN, dbSlopeBottom_MIN);

			fprintf_s(out, "Index, DataTop, DataTopSlope, DataBottom, DataBottomSlope\n");

			for (int i = 0; i < nEdgeUnit; i++)
			{
				fprintf_s(out, "%d,%d,%.3f,%d,%.3f\n",
					i, matTopRegion.at<ushort>(0, i), matTopRegionSlop.at<float>(0, i), matBottomRegion.at<ushort>(0, i), matBottomRegionSlop.at<float>(0, i));
			}

			fclose(out);
		}
	}

	//如果没有Defect,请退出
	if (nTopDefectStart == 0 && nBottomDefectStart == 0) return E_ERROR_CODE_TRUE;

	// Dark Defect X Direction Defect
	if (nDefectDirection == E_DEFECT_LINEMURA_X)
	{
		//Tope位置不良
		if (nTopDefectStart != 0)
		{
			for (int i = 0; i < nTopDefectStart; i++)
			{
				cpLT.x = i;				cpLT.y = 0;
				cpLB.x = i;				cpLB.y = nImageHeight;

				cv::line(matResult, cpLT, cpLB, cv::Scalar(50), 1);
			}
		}

		//Bottom位置不良
		if (nBottomDefectStart != 0)
		{
			for (int i = nBottomDefectStart; i < matSrcImageDark.cols; i++)
			{
				cpLT.x = i;				cpLT.y = 0;
				cpLB.x = i;				cpLB.y = nImageHeight;

				cv::line(matResult, cpLT, cpLB, cv::Scalar(50), 1);
			}
		}
	}

	// Dark Defect Y Direction Defect
	if (nDefectDirection == E_DEFECT_LINEMURA_Y)
	{
		//Tope位置不良
		if (nTopDefectStart != 0)
		{
			for (int i = 0; i < nTopDefectStart; i++)
			{
				cpLT.x = 0;				cpLT.y = i;
				cpRT.x = nImageWidth;	cpRT.y = i;

				cv::line(matResult, cpLT, cpRT, cv::Scalar(50), 1);
			}
		}

		//Tope位置不良
		if (nBottomDefectStart != 0)
		{
			for (int i = nBottomDefectStart; i < matSrcImageDark.cols; i++)
			{
				cpLT.x = 0;				cpLT.y = i;
				cpRT.x = nImageWidth;	cpRT.y = i;

				cv::line(matResult, cpLT, cpRT, cv::Scalar(50), 1);
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}

long  CInspectMuraNormal::NegativeSlopeDataCheck(cv::Mat& matSrcImage, float fThresholdValue, int nCalcLength, int& nDefectStart, cv::Mat& matSlope)
{
	//异常处理
	if (matSrcImage.empty() || fThresholdValue == 0 || nCalcLength == 0) return false;

	int nSlopCalceUnite = 10;
	int nDefectRegion = 0;

	//float	*ptrSlop = (float*)matSlope.ptr(0);															// Data左侧Data

	for (int i = 0; i < matSrcImage.cols - nSlopCalceUnite; i++)
	{
		//初始化
		ushort* ptrData = (ushort*)matSrcImage.ptr(0) + matSrcImage.cols - i;						// Data左侧Data
		ushort* ptrB = (ushort*)matSrcImage.ptr(0) + matSrcImage.cols - nSlopCalceUnite - i;	// Data左侧Data
		float* ptrSlop = (float*)matSlope.ptr(0) + matSrcImage.cols - i;

		// Slope Calculation
		float	nBackSlope = (float)(*ptrB - *ptrData) / nSlopCalceUnite;
		*ptrSlop = nBackSlope;

		if (nBackSlope <= fThresholdValue) nDefectRegion++;
	}

	if (nDefectRegion == 0)	nDefectStart = 0;
	else					nDefectStart = nCalcLength - ((int)nDefectRegion / 2 + 1);

	return E_ERROR_CODE_TRUE;
}

long  CInspectMuraNormal::PositiveSlopeDataCheck(cv::Mat& matSrcImage, float fThresholdValue, int nCalcLength, int& nDefectStart, cv::Mat& matSlope)
{
	//异常处理
	if (matSrcImage.empty() || fThresholdValue == 0) return false;

	int nSlopCalceUnite = 10;
	int nDefectRegion = 0;

	//初始化
	ushort* ptrData = (ushort*)matSrcImage.ptr(0);							// Data左侧Data
	float* ptrSlop = (float*)matSlope.ptr(0);							// Data左侧Data
	ushort* ptrF = (ushort*)matSrcImage.ptr(0) + nSlopCalceUnite;		// Data左侧Data

	for (int i = 0; i < matSrcImage.cols - nSlopCalceUnite; i++, ptrData++, ptrF++, ptrSlop++)
	{
		// Slope Calculation
		float nForeSlope = (float)(*ptrF - *ptrData) / nSlopCalceUnite;
		*ptrSlop = nForeSlope;

		if (nForeSlope <= fThresholdValue) nDefectRegion++;
	}

	if (nDefectRegion == 0)	nDefectStart = 0;
	else					nDefectStart = nCalcLength + ((int)nDefectRegion / 2 + 1);

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::PreProOptimization(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//异常处理
	if (matSrcImage.empty())	return false;
	if (dPara == NULL)			return false;

	//////////////////////////////////////////////////////////////////////////
	//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		nShiftCopy = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_SHIFT_UNIT];
	int		nImageBlur = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_MEANBLUR];
	int		nCpyX = 0;
	int		nCpyY = 0;
	int		nLoopX = 0;
	int		nLoopY = 0;

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process Start
	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start Process."));

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[3]);

	//固定Edge Cut
	int nEdgeCutUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_EDGECUT_UNIT];

	// ROI Select
	cv::Mat matROITemp = matSrcImage(cv::Rect(rectROI.left + nEdgeCutUnit, rectROI.top + nEdgeCutUnit, rectROI.Width() - nEdgeCutUnit * 2, rectROI.Height() - nEdgeCutUnit * 2));

	cv::Mat matTemp1 = cMatBufTemp.GetMat(matROITemp.size(), matROITemp.type());
	cv::Mat matTemp2 = cMatBufTemp.GetMat(matROITemp.size(), matROITemp.type());

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Get Buff."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Input.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Input_ROI.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matROITemp);
	}

	// ShiftCopy
	ShiftCopyParaCheck(nShiftCopy, nCpyX, nCpyY, nLoopX, nLoopY);
	AlgoBase::ShiftCopy(matROITemp, matTemp1, nCpyX, nCpyY, nLoopX, nLoopY, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Shift Copy."));

	// LoopBlur
	LoopBlur(matTemp1, matTemp2, nImageBlur, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Loop Blur."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_ShiftCopy.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTemp1);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_LoopBlur.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTemp2);
	}

	// Copy Result
	matTemp2.copyTo(matDstImage);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Copy Result & End Process."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//////////////////////////////////////////////////////////////////////////
//内容		:Notch Screen不良检查功能
//操作员	:PNZ
//日期		:19.07.16
//////////////////////////////////////////////////////////////////////////

long CInspectMuraNormal::NotchROICheck(cv::Mat& matSrcImage, cv::Mat& matSrcNotchReBGBuf, cv::Rect ProROINotchBG, int* nCommonPara, CString strAlgPath, double* dPara, stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Start "));

	// Parameter
	int	nResize = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE4_RESIZE];

	int nWidth = matSrcImage.cols;
	int nHeight = matSrcImage.rows;

	int	nRectOffSet = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_RECT_OFFSET]; // 3
	int	nRectWidth = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_RECT_WIDTH]; // 70
	int	nRectHeight = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_RECT_HEIGHT]; // 200
	int	nAverageUnit = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_AVG_UNIT]; // 5
	int	nCheckUite = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_CHECK_UNIT]; // 20
	int	nSlopCalceUnite = (int)dPara[E_PARA_AVI_MURA_NORMAL_NOTCH_SLOPE_CALC_UNIT]; // 5

	// ROI Set
	int nTopStartX = nRectOffSet;
	int nTopStartY = nRectOffSet;
	int nBottomStartX = nRectOffSet;
	int nBottomStartY = nHeight - nRectHeight - nRectOffSet;

	cv::Rect RectROITop = cv::Rect(nTopStartX, nTopStartY, nRectWidth, nRectHeight);
	cv::Rect RectROIBottom = cv::Rect(nBottomStartX, nBottomStartY, nRectWidth, nRectHeight);

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matROITop = cMatBufTemp.GetMat(cv::Size(nRectWidth, nRectHeight), CV_8UC1);
	cv::Mat matROIBottom = cMatBufTemp.GetMat(cv::Size(nRectWidth, nRectHeight), CV_8UC1);
	cv::Mat matProfileTop = cMatBufTemp.GetMat(cv::Size(nRectWidth, 1), CV_32FC1);
	cv::Mat matProfileBottom = cMatBufTemp.GetMat(cv::Size(nRectWidth, 1), CV_32FC1);
	cv::Mat matProfBlurTop = cMatBufTemp.GetMat(cv::Size(nRectWidth, 1), CV_32FC1);
	cv::Mat matProfBlurBottom = cMatBufTemp.GetMat(cv::Size(nRectWidth, 1), CV_32FC1);
	cv::Mat matPSTop = cMatBufTemp.GetMat(cv::Size(nRectWidth - nCheckUite * 2, 1), CV_32FC1);
	cv::Mat matPSBottom = cMatBufTemp.GetMat(cv::Size(nRectWidth - nCheckUite * 2, 1), CV_32FC1);
	cv::Mat matPSBlurTop = cMatBufTemp.GetMat(cv::Size(nRectWidth - nCheckUite * 2, 1), CV_32FC1);
	cv::Mat matPSBlurBottom = cMatBufTemp.GetMat(cv::Size(nRectWidth - nCheckUite * 2, 1), CV_32FC1);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Set Buff. "));

	matROITop = matSrcImage(RectROITop);
	matROIBottom = matSrcImage(RectROIBottom);

	AlgoBase::MakeProfile(matROITop, matProfileTop, nRectWidth, nRectHeight, 4);
	AlgoBase::MakeProfile(matROIBottom, matProfileBottom, nRectWidth, nRectHeight, 4);

	cv::blur(matProfileTop, matProfBlurTop, cv::Size(nAverageUnit, 1));
	cv::blur(matProfileBottom, matProfBlurBottom, cv::Size(nAverageUnit, 1));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_NC_ProfileX_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "Index, TopData, TopBlurData, BottomData, BottomBlurData\n");

			for (int i = 0; i < nRectWidth; i++)
			{
				fprintf_s(out, "%d,%.3f,%.3f,%.3f,%.3f\n",
					i, matProfileTop.at<float>(0, i), matProfileBottom.at<float>(0, i), matProfBlurTop.at<float>(0, i), matProfBlurBottom.at<float>(0, i));
			}

			fclose(out);
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Pre-Process. "));

	//////////////////////////////////////////////////////////////////////////
	// Profile Check
	double ProfTop_LeftAvg = 0;
	double ProfTop_RightAvg = 0;
	double ProfBottom_LeftAvg = 0;
	double ProfBottom_RightAvg = 0;
	double ProfBlurTop_LeftAvg = 0;
	double ProfBlurTop_RightAvg = 0;
	double ProfBlurBottom_LeftAvg = 0;
	double ProfBlurBottom_RightAvg = 0;

	// Avg Check
	NorchProfileCheck(matProfileTop, nCheckUite, ProfTop_LeftAvg, ProfTop_RightAvg, &cMatBufTemp);
	NorchProfileCheck(matProfileBottom, nCheckUite, ProfBottom_LeftAvg, ProfBottom_RightAvg, &cMatBufTemp);
	NorchProfileCheck(matProfBlurTop, nCheckUite, ProfBlurTop_LeftAvg, ProfBlurTop_RightAvg, &cMatBufTemp);
	NorchProfileCheck(matProfBlurBottom, nCheckUite, ProfBlurBottom_LeftAvg, ProfBlurBottom_RightAvg, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Notch Profile GV Check. "));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_MURANormal_NC_AvgData.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "AvgRegion,AvgLeft,AvgRight\n");
			fprintf_s(out, "ProfileTop,%.3f,%.3f\n", ProfTop_LeftAvg, ProfTop_RightAvg);
			fprintf_s(out, "ProfileBottom,%.3f,%.3f\n", ProfBottom_LeftAvg, ProfBottom_RightAvg);
			fprintf_s(out, "ProfileBlurTop,%.3f,%.3f\n", ProfBlurTop_LeftAvg, ProfBlurTop_RightAvg);
			fprintf_s(out, "ProfileBlurBottom,%.3f,%.3f\n", ProfBlurBottom_LeftAvg, ProfBlurBottom_RightAvg);

			fclose(out);
		}
	}

	// Slope Check
	NorchSlopeCalc(matProfileTop, nCheckUite, nSlopCalceUnite, matPSTop, &cMatBufTemp);
	NorchSlopeCalc(matProfileBottom, nCheckUite, nSlopCalceUnite, matPSBottom, &cMatBufTemp);
	NorchSlopeCalc(matProfBlurTop, nCheckUite, nSlopCalceUnite, matPSBlurTop, &cMatBufTemp);
	NorchSlopeCalc(matProfBlurBottom, nCheckUite, nSlopCalceUnite, matPSBlurBottom, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Notch Profile Slope Check. "));

	double dbMin_PSTop = 0;
	double dbMax_PSTop = 0;
	double dbMin_PSBottom = 0;
	double dbMax_PSBottom = 0;
	double dbMin_PSTopBlur = 0;
	double dbMax_PSTopBlur = 0;
	double dbMin_PSBottomBLur = 0;
	double dbMax_PSBottomBlur = 0;

	cv::Rect minMaxRange = cv::Rect(10, 0, 10, 1);

	cv::minMaxIdx(matPSTop(minMaxRange), &dbMin_PSTop, &dbMax_PSTop, NULL, NULL);
	cv::minMaxIdx(matPSBottom(minMaxRange), &dbMin_PSBottom, &dbMax_PSBottom, NULL, NULL);
	cv::minMaxIdx(matPSBlurTop(minMaxRange), &dbMin_PSTopBlur, &dbMax_PSTopBlur, NULL, NULL);
	cv::minMaxIdx(matPSBlurBottom(minMaxRange), &dbMin_PSBottomBLur, &dbMax_PSBottomBlur, NULL, NULL);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_MURANormal_NC_SlopeData.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "MaxSlope,%.3f,%.3f,%.3f,%.3f\n", dbMax_PSTop, dbMax_PSBottom, dbMax_PSTopBlur, dbMax_PSBottomBlur);

			fprintf_s(out, "Index, TopSlope, BottomSlope, TopBlurSlope, BottomBlurSlope\n");

			for (int i = 0; i < matPSTop.cols; i++)
			{
				fprintf_s(out, "%d,%.3f,%.3f,%.3f,%.3f\n", i, matPSTop.at<float>(0, i), matPSBottom.at<float>(0, i), matPSBlurTop.at<float>(0, i), matPSBlurBottom.at<float>(0, i));
			}

			fclose(out);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Defect Judge

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Defect Judge Start. "));

	// Input Data
	double dbBlurTopDiff = abs(ProfBlurTop_LeftAvg - ProfBlurTop_RightAvg);
	double dbBlurBottomDiff = abs(ProfBlurBottom_LeftAvg - ProfBlurBottom_RightAvg);

	int nDefectType = 0;

	DefectJudgement(dbBlurTopDiff, dbBlurBottomDiff, dbMax_PSTopBlur, dbMax_PSBottomBlur, dPara, nDefectType);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Defect Judge. "));

	//////////////////////////////////////////////////////////////////////////
	// Select Defect Region

	cv::Mat matBGTemp = cMatBufTemp.GetMat(matSrcNotchReBGBuf.size(), matSrcNotchReBGBuf.type());
	cv::threshold(matSrcNotchReBGBuf, matBGTemp, 100, 255, THRESH_BINARY_INV);

	if (nDefectType == E_MURANORMAL_NOTCH_DEFECT_NONE) return false;

	if (nDefectType == E_MURANORMAL_NOTCH_DEFECT_TOP)
	{
		matBGTemp(cv::Rect(0, matSrcNotchReBGBuf.rows / 2, matSrcNotchReBGBuf.cols, matSrcNotchReBGBuf.rows / 2)).setTo(0);
	}

	if (nDefectType == E_MURANORMAL_NOTCH_DEFECT_BOTTOM)
	{
		matBGTemp(cv::Rect(0, 0, matSrcNotchReBGBuf.cols, matSrcNotchReBGBuf.rows / 2)).setTo(0);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Region Select. "));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_matBGTemp.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBGTemp);
	}

	//////////////////////////////////////////////////////////////////////////
	// Draw Result
	DrawNotchDefect(matBGTemp, pResultBlob, ProROINotchBG, nResize, nImageUI, nPS);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Notch Check : Draw Notch Defect. "));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long	CInspectMuraNormal::NorchProfileCheck(cv::Mat& matSrcImage, int nCheckUite, double& ProfTop_LeftAvg, double& ProfTop_RightAvg, CMatBuf* cMemSub)
{
	//异常处理
	if (matSrcImage.empty() || nCheckUite == 0) return false;

	// ROI Set
	int nLeftStart = 0;
	int nRightStart = matSrcImage.cols - nCheckUite;

	cv::Rect RectROILeft = cv::Rect(nLeftStart, 0, nCheckUite, 1);
	cv::Rect RectROIRight = cv::Rect(nRightStart, 0, nCheckUite, 1);

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matROILeft = cMatBufTemp.GetMat(cv::Size(nCheckUite, 1), CV_32FC1);
	cv::Mat matROIRight = cMatBufTemp.GetMat(cv::Size(nCheckUite, 1), CV_32FC1);

	matROILeft = matSrcImage(RectROILeft);
	matROIRight = matSrcImage(RectROIRight);

	cv::Mat LeftMean, LeftStd, RightMean, RightStd;

	cv::meanStdDev(matROILeft, LeftMean, LeftStd);
	cv::meanStdDev(matROIRight, RightMean, RightStd);

	ProfTop_LeftAvg = LeftMean.at<double>(0, 0);
	ProfTop_RightAvg = RightMean.at<double>(0, 0);

	return E_ERROR_CODE_TRUE;
}

long	CInspectMuraNormal::NorchSlopeCalc(cv::Mat& matSrcImage, int nCheckUite, int nSlopCalceUnite, cv::Mat& matDstImage, CMatBuf* cMemSub)
{
	//异常处理
	if (matSrcImage.empty() || nCheckUite == 0) return false;

	// ROI Set
	int nDataStart = nCheckUite;
	int nDataLength = matSrcImage.cols - nCheckUite * 2;

	cv::Rect RectROI = cv::Rect(nDataStart, 0, nDataLength, 1);

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matROIBuff = cMatBufTemp.GetMat(cv::Size(nDataLength, 1), CV_32FC1);

	matROIBuff = matSrcImage(RectROI);

	// Slope Calculation

		//初始化
	float* ptr1 = (float*)matROIBuff.ptr(0);						// Data左侧Data
	float* ptr2 = (float*)matROIBuff.ptr(0) + nSlopCalceUnite;	// Data
	float* ptrResult = (float*)matDstImage.ptr(0);						// Result Data

	for (int i = 0; i < matROIBuff.cols; i++, ptr1++, ptr2++, ptrResult++)
	{
		if (i < matROIBuff.cols - nSlopCalceUnite)	*ptrResult = (float)abs((*ptr2 - *ptr1) / nSlopCalceUnite);
		else										*ptrResult = (float)0.0;
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long	CInspectMuraNormal::DefectJudgement(double dbBlurTopDiff, double dbBlurBottomDiff, double dbMax_PSTopBlur, double dbMax_PSBottomBlur, double* dPara, int& nDefectType)
{
	if (dPara == NULL)			return false;

	// 	//Parameter
	// 	double dbTopGTH_GV			= 2;
	// 	double dbBottomTH_GV		= 2;
	// 	double dbLimitCheck_GV		= 0;
	// 
	// 	double dbTopGTH_Slope		= 0.18;
	// 	double dbBottomTH_Slope		= 0.18;
	// 	double dbLimitCheck_Slope	= 0.13;

		//Parameter
	double dbTopGTH_GV = 1.5;
	double dbBottomTH_GV = 1.5;
	double dbLimitCheck_GV = 0;

	double dbTopGTH_Slope = 0.35;
	double dbBottomTH_Slope = 0.35;
	double dbLimitCheck_Slope = 0.20;

	bool nTopResult = false;
	bool nBottomResult = false;

	//////////////////////////////////////////////////////////////////////////
	// Top Judgement
	if (dbBlurTopDiff >= dbTopGTH_GV && dbMax_PSTopBlur >= dbLimitCheck_Slope)
		nTopResult = true;

	else if (dbLimitCheck_GV <= dbBlurTopDiff && dbBlurTopDiff < dbTopGTH_GV && dbMax_PSTopBlur >= dbTopGTH_Slope)
		nTopResult = true;

	else
		nTopResult = false;

	//////////////////////////////////////////////////////////////////////////
	// Bottom Judgement
	if (dbBlurBottomDiff >= dbBottomTH_GV && dbMax_PSBottomBlur >= dbLimitCheck_Slope)
		nBottomResult = true;

	else if (dbLimitCheck_GV <= dbBlurBottomDiff && dbBlurBottomDiff < dbBottomTH_GV && dbMax_PSBottomBlur >= dbBottomTH_Slope)
		nBottomResult = true;

	else
		nBottomResult = false;

	//////////////////////////////////////////////////////////////////////////
	// Find Judgement
	if (nTopResult == false && nBottomResult == false)		nDefectType = E_MURANORMAL_NOTCH_DEFECT_NONE;
	else if (nTopResult == true && nBottomResult == false)		nDefectType = E_MURANORMAL_NOTCH_DEFECT_TOP;
	else if (nTopResult == false && nBottomResult == true)		nDefectType = E_MURANORMAL_NOTCH_DEFECT_BOTTOM;
	else if (nTopResult == true && nBottomResult == true)		nDefectType = E_MURANORMAL_NOTCH_DEFECT_BOTH;
	else														nDefectType = E_MURANORMAL_NOTCH_DEFECT_NONE;

	return E_ERROR_CODE_TRUE;
}

long	CInspectMuraNormal::DrawNotchDefect(cv::Mat& matBGTemp, stDefectInfo* pResultBlob, cv::Rect InputROI, int nResize, int nImageUI, int nPS)
{
	if (matBGTemp.empty()) return false;

	vector<vector<cv::Point>>	ptDefectContours;

	// Find Contours
	FindnCalContours(matBGTemp, InputROI, nResize, ptDefectContours);

	if (ptDefectContours.size() == 0) return false;

	//////////////////////////////////////////////////////////////////////////
		//事件检查
	int  nCountNumber = pResultBlob->nDefectCount;
	int  nDefectIndex = 0;

	if (nCountNumber == 0)	nDefectIndex = 0;
	else					nDefectIndex = nCountNumber;

	//参数计算
	double Xmax = 0;
	double Xmin = 0;
	double Ymax = 0;
	double Ymin = 0;

	//数据计算
	CalcDataMinMax(ptDefectContours.at(0), Xmax, Xmin, Ymax, Ymin);

	//////////////////////////////////////////////////////////////////////////
	// Basic Feature Info Write
	pResultBlob->nArea[nDefectIndex] = 0;
	pResultBlob->nMaxGV[nDefectIndex] = 0;
	pResultBlob->nMinGV[nDefectIndex] = 0;
	pResultBlob->dMeanGV[nDefectIndex] = 0;
	pResultBlob->dBackGroundGV[nDefectIndex] = 0;
	pResultBlob->dCompactness[nDefectIndex] = 0;
	pResultBlob->dSigma[nDefectIndex] = 0;
	pResultBlob->dBreadth[nDefectIndex] = 0;
	pResultBlob->dF_Min[nDefectIndex] = 0;
	pResultBlob->dF_Max[nDefectIndex] = 0;
	pResultBlob->dF_Elongation[nDefectIndex] = 0;
	pResultBlob->dCompactness[nDefectIndex] = 0;
	pResultBlob->nDefectColor[nDefectIndex] = E_DEFECT_COLOR_DARK;
	pResultBlob->nPatternClassify[nDefectIndex] = nImageUI;
	pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURANORMAL_NOTCH_DEFECT;

	//Defect坐标
	pResultBlob->ptLT[nDefectIndex].x = Xmin;
	pResultBlob->ptLT[nDefectIndex].y = Ymin;
	pResultBlob->ptLB[nDefectIndex].x = Xmin;
	pResultBlob->ptLB[nDefectIndex].y = Ymax;
	pResultBlob->ptRT[nDefectIndex].x = Xmax;
	pResultBlob->ptRT[nDefectIndex].y = Ymin;
	pResultBlob->ptRB[nDefectIndex].x = Xmax;
	pResultBlob->ptRB[nDefectIndex].y = Ymax;

	// Contour Sampling
	int		nDefectLength = ptDefectContours.at(0).size();
	float	fRatio = 1.0;

	if (nDefectLength >= MAX_CONTOURS)
	{
		fRatio = nDefectLength / (float)MAX_CONTOURS;
		nDefectLength = MAX_CONTOURS;
	}

	//保存轮廓
	for (int j = 0; j < nDefectLength; j++)
	{
		int index = (int)(j * fRatio);

		pResultBlob->nContoursX[nDefectIndex][j] = (int)(ptDefectContours.at(0)[index].x / nPS);
		pResultBlob->nContoursY[nDefectIndex][j] = (int)(ptDefectContours.at(0)[index].y / nPS);
	}

	//添加不良
	pResultBlob->nDefectCount = nDefectIndex + 1;

	vector<vector<cv::Point>>().swap(ptDefectContours);

	return E_ERROR_CODE_TRUE;
}

//x方向管接头
long CInspectMuraNormal::RangeFit_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double dEstiBright, double dEstiDark, double* dPara)
{
	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	int		nDimensionX = 2;
	int		nEstiStepX = 50;
	// 	double	dEstiBright = 1.15;
	// 	double	dEstiDark = 0.85;
	// 	double	dEstiBright = 1.10;
	// 	double	dEstiDark = 0.90;
		// 	double	dEstiBright = 1.05;
		// 	double	dEstiDark = 0.95;

		//int nStepX = matSrcBuf.cols / nEstiStepX;
	int nStepX = matSrcBuf.cols / matSrcBuf.cols;

	if (nStepX <= 0) nStepX = 1;

	int nStepCols = matSrcBuf.cols / nStepX;
	int nHalfCols = matSrcBuf.cols / 2;

	cv::Mat matM = cv::Mat_<double>(nStepCols, nDimensionX + 1);
	cv::Mat matL = cv::Mat_<double>(nStepCols, 1);
	cv::Mat matQ;

	double x, quad, dTemp;
	int i, j, k, m;

	//////////////////////////////////////////////////////////////////////////
		//直方图
	cv::Mat matHisto;
	AlgoBase::GetHistogram(matSrcBuf, matHisto, false);

	double dblAverage;
	double dblStdev;
	AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

	//设置平均范围
	int nMinGVHist = (int)(dblAverage - dblStdev);
	int nMaxGVHist = (int)(dblAverage + dblStdev);
	if (nMinGVHist <= 0)	nMinGVHist = 1;
	if (nMaxGVHist > 255)	nMaxGVHist = 255;

	//初始化
	__int64 nPixelSum = 0;
	__int64 nPixelCount = 0;

	//仅重新平均设置的平均范围
	float* pVal = (float*)matHisto.ptr(0) + nMinGVHist;

	for (int m = nMinGVHist; m <= nMaxGVHist; m++, pVal++)
	{
		nPixelSum += (m * *pVal);
		nPixelCount += *pVal;
	}

	float fHistAvg = (float)(nPixelSum / (float)nPixelCount);

	//亮度值限制
	int nMinGV = (int)(fHistAvg * dEstiDark);
	int nMaxGV = (int)(fHistAvg * dEstiBright);

	// 	// //////////////////////////////////////////////////////////////////////////
	//	//亮度值限制
	// 	cv::Scalar mean = cv::mean(matSrcBuf);
	// 	int nMinGV = (int)(mean[0] * dEstiDark);
	// 	int nMaxGV = (int)(mean[0] * dEstiBright);

			//如果是原始8U
	if (matSrcBuf.type() == CV_8U)
	{
		for (i = 0; i < matSrcBuf.rows; i++)
		{
			for (j = 0; j < nStepCols; j++)
			{
				x = (j * nStepX - nHalfCols) / double(matSrcBuf.cols);

				//x = j+1;

				matM.at<double>(j, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					matM.at<double>(j, k) = dTemp;
				}

				m = matSrcBuf.at<uchar>(i, j * nStepX);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(j, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (j = 0; j < matDstBuf.cols; j++)
			{
				x = (j - nHalfCols) / double(matSrcBuf.cols);

				//x = j + 1;

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<uchar>(i, j) = saturate_cast<uchar>(quad);
			}
		}
	}
	//如果是源16U
	else
	{
		for (i = 0; i < matSrcBuf.rows; i++)
		{
			for (j = 0; j < nStepCols; j++)
			{
				x = (j * nStepX - nHalfCols) / double(matSrcBuf.cols);

				matM.at<double>(j, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					matM.at<double>(j, k) = dTemp;
				}

				//I.at<double>(j, 0) = matSrcBuf.at<ushort>(i, j*nStepX);
				m = matSrcBuf.at<ushort>(i, j * nStepX);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(j, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (j = 0; j < matDstBuf.cols; j++)
			{
				x = (j - nHalfCols) / double(matSrcBuf.cols);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<ushort>(i, j) = saturate_cast<ushort>(quad);
			}
		}
	}

	matM.release();
	matL.release();
	matQ.release();

	return E_ERROR_CODE_TRUE;
}

//y方向管接头
long CInspectMuraNormal::RangeFit_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double dEstiBright, double dEstiDark, double* dPara)
{
	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	int		nDimensionY = 2;
	int		nEstiStepY = 50;
	// 	double	dEstiBright = 1.15;
	// 	double	dEstiDark = 0.85;
	// 	double	dEstiBright = 1.10;
	// 	double	dEstiDark = 0.90;
	//   	double	dEstiBright = 1.05;
	//   	double	dEstiDark	= 0.95;

	int nStepY = matSrcBuf.rows / matSrcBuf.rows;

	if (nStepY <= 0) nStepY = 1;

	int nStepRows = matSrcBuf.rows / nStepY;
	int nHalfRows = matSrcBuf.rows / 2;

	cv::Mat matM = cv::Mat_<double>(nStepRows, nDimensionY + 1);
	cv::Mat matL = cv::Mat_<double>(nStepRows, 1);
	cv::Mat matQ;

	double y, quad, dTemp;
	int i, j, k, m;

	//////////////////////////////////////////////////////////////////////////
		//直方图
	cv::Mat matHisto;
	AlgoBase::GetHistogram(matSrcBuf, matHisto, false);

	double dblAverage;
	double dblStdev;
	AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

	//设置平均范围
	int nMinGVHist = (int)(dblAverage - dblStdev);
	int nMaxGVHist = (int)(dblAverage + dblStdev);
	if (nMinGVHist <= 0)	nMinGVHist = 1;
	if (nMaxGVHist > 255)	nMaxGVHist = 255;

	//初始化
	__int64 nPixelSum = 0;
	__int64 nPixelCount = 0;

	//仅重新平均设置的平均范围
	float* pVal = (float*)matHisto.ptr(0) + nMinGVHist;

	for (int m = nMinGVHist; m <= nMaxGVHist; m++, pVal++)
	{
		nPixelSum += (m * *pVal);
		nPixelCount += *pVal;
	}

	float fHistAvg = (float)(nPixelSum / (float)nPixelCount);

	//亮度值限制
	int nMinGV = (int)(fHistAvg * dEstiDark);
	int nMaxGV = (int)(fHistAvg * dEstiBright);

	//////////////////////////////////////////////////////////////////////////////
 //	//亮度值限制
 //	cv::Scalar mean = cv::mean(matSrcBuf);
 //	int nMinGV = (int)(mean[0] * dEstiDark);
 //	int nMaxGV = (int)(mean[0] * dEstiBright);

		//如果是原始8U
	if (matSrcBuf.type() == CV_8U)
	{
		for (j = 0; j < matSrcBuf.cols; j++)
		{
			for (i = 0; i < nStepRows; i++)
			{
				y = (i * nStepY - nHalfRows) / double(matSrcBuf.rows);

				matM.at<double>(i, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					matM.at<double>(i, k) = dTemp;
				}

				//I.at<double>(i, 0) = matSrcBuf.at<uchar>(i*nStepY, j);
				m = matSrcBuf.at<uchar>(i * nStepY, j);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(i, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (i = 0; i < matSrcBuf.rows; i++)
			{
				y = (i - nHalfRows) / double(matSrcBuf.rows);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<uchar>(i, j) = saturate_cast<uchar>(quad);
			}
		}
	}
	//如果是源16U
	else
	{
		for (j = 0; j < matSrcBuf.cols; j++)
		{
			for (i = 0; i < nStepRows; i++)
			{
				y = (i * nStepY - nHalfRows) / double(matSrcBuf.rows);

				matM.at<double>(i, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					matM.at<double>(i, k) = dTemp;
				}

				//I.at<double>(i, 0) = matSrcBuf.at<ushort>(i*nStepY, j);
				m = matSrcBuf.at<ushort>(i * nStepY, j);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(i, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (i = 0; i < matSrcBuf.rows; i++)
			{
				y = (i - nHalfRows) / double(matSrcBuf.rows);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<ushort>(i, j) = saturate_cast<ushort>(quad);
			}
		}
	}

	matM.release();
	matL.release();
	matQ.release();

	return E_ERROR_CODE_TRUE;
}

//x方向管接头
long CInspectMuraNormal::Estimation_X(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara)
{
	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	int		nDimensionX = 2;
	int		nEstiStepX = 300;
	double	dEstiBright = 1.05;
	double	dEstiDark = 0.95;

	int nStepX = matSrcBuf.cols / nEstiStepX;

	int nStepCols = matSrcBuf.cols / nStepX;
	int nHalfCols = matSrcBuf.cols / 2;

	cv::Mat matM = cv::Mat_<double>(nStepCols, nDimensionX + 1);
	cv::Mat matL = cv::Mat_<double>(nStepCols, 1);
	cv::Mat matQ;

	double x, quad, dTemp;
	int i, j, k, m;

	//亮度值限制
	cv::Scalar mean = cv::mean(matSrcBuf);
	int nMinGV = (int)(mean[0] * dEstiDark);
	int nMaxGV = (int)(mean[0] * dEstiBright);

	//如果是原始8U
	if (matSrcBuf.type() == CV_8U)
	{
		for (i = 0; i < matSrcBuf.rows; i++)
		{
			for (j = 0; j < nStepCols; j++)
			{
				x = (j * nStepX - nHalfCols) / double(matSrcBuf.cols);

				matM.at<double>(j, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					matM.at<double>(j, k) = dTemp;
				}

				//I.at<double>(j, 0) = matSrcBuf.at<uchar>(i, j*nStepX);
				m = matSrcBuf.at<uchar>(i, j * nStepX);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(j, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (j = 0; j < matDstBuf.cols; j++)
			{
				x = (j - nHalfCols) / double(matSrcBuf.cols);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<uchar>(i, j) = saturate_cast<uchar>(quad);
			}
		}
	}
	//如果是源16U
	else
	{
		for (i = 0; i < matSrcBuf.rows; i++)
		{
			for (j = 0; j < nStepCols; j++)
			{
				x = (j * nStepX - nHalfCols) / double(matSrcBuf.cols);

				matM.at<double>(j, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					matM.at<double>(j, k) = dTemp;
				}

				//I.at<double>(j, 0) = matSrcBuf.at<ushort>(i, j*nStepX);
				m = matSrcBuf.at<ushort>(i, j * nStepX);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(j, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (j = 0; j < matDstBuf.cols; j++)
			{
				x = (j - nHalfCols) / double(matSrcBuf.cols);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionX; k++)
				{
					dTemp *= x;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<ushort>(i, j) = saturate_cast<ushort>(quad);
			}
		}
	}

	matM.release();
	matL.release();
	matQ.release();

	return E_ERROR_CODE_TRUE;
}

//y方向管接头
long CInspectMuraNormal::Estimation_Y(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara)
{
	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	int		nDimensionY = 2;
	int		nEstiStepY = 300;
	double	dEstiBright = 1.05;
	double	dEstiDark = 0.95;

	int nStepY = matSrcBuf.rows / nEstiStepY;

	int nStepRows = matSrcBuf.rows / nStepY;
	int nHalfRows = matSrcBuf.rows / 2;

	cv::Mat matM = cv::Mat_<double>(nStepRows, nDimensionY + 1);
	cv::Mat matL = cv::Mat_<double>(nStepRows, 1);
	cv::Mat matQ;

	double y, quad, dTemp;
	int i, j, k, m;

	//亮度值限制
	cv::Scalar mean = cv::mean(matSrcBuf);
	int nMinGV = (int)(mean[0] * dEstiDark);
	int nMaxGV = (int)(mean[0] * dEstiBright);

	//如果是原始8U
	if (matSrcBuf.type() == CV_8U)
	{
		for (j = 0; j < matSrcBuf.cols; j++)
		{
			for (i = 0; i < nStepRows; i++)
			{
				y = (i * nStepY - nHalfRows) / double(matSrcBuf.rows);

				matM.at<double>(i, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					matM.at<double>(i, k) = dTemp;
				}

				//I.at<double>(i, 0) = matSrcBuf.at<uchar>(i*nStepY, j);
				m = matSrcBuf.at<uchar>(i * nStepY, j);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(i, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (i = 0; i < matSrcBuf.rows; i++)
			{
				y = (i - nHalfRows) / double(matSrcBuf.rows);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<uchar>(i, j) = saturate_cast<uchar>(quad);
			}
		}
	}
	//如果是源16U
	else
	{
		for (j = 0; j < matSrcBuf.cols; j++)
		{
			for (i = 0; i < nStepRows; i++)
			{
				y = (i * nStepY - nHalfRows) / double(matSrcBuf.rows);

				matM.at<double>(i, 0) = 1.0;
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					matM.at<double>(i, k) = dTemp;
				}

				//I.at<double>(i, 0) = matSrcBuf.at<ushort>(i*nStepY, j);
				m = matSrcBuf.at<ushort>(i * nStepY, j);

				if (m < nMinGV)	m = nMinGV;
				if (m > nMaxGV)	m = nMaxGV;

				matL.at<double>(i, 0) = m;
			}

			cv::SVD svd(matM);
			svd.backSubst(matL, matQ);

			for (i = 0; i < matSrcBuf.rows; i++)
			{
				y = (i - nHalfRows) / double(matSrcBuf.rows);

				quad = matQ.at<double>(0, 0);
				dTemp = 1.0;
				for (k = 1; k <= nDimensionY; k++)
				{
					dTemp *= y;
					quad += (matQ.at<double>(k, 0) * dTemp);
				}

				matDstBuf.at<ushort>(i, j) = saturate_cast<ushort>(quad);
			}
		}
	}

	matM.release();
	matL.release();
	matQ.release();

	return E_ERROR_CODE_TRUE;
}

//xy方向管接头
long CInspectMuraNormal::Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////

		//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;

	int		nEstiDimX = 2;
	int		nEstiDimY = 2;
	int		nEstiStepX = 50;
	int		nEstiStepY = 50;

	//异常处理
	if (nEstiDimX <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiDimY <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiStepX <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiStepY <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matDstBufX = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	cv::Mat matDstBufY = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif

	for (int i = 0; i < 2; i++)
	{
		switch (i)
		{
		case 0:
			nErrorCode |= Estimation_X(matSrcBuf, matDstBufX, dPara);
			break;
		case 1:
			nErrorCode |= Estimation_Y(matSrcBuf, matDstBufY, dPara);
			break;
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Test_BackGround_X.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBufX);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_Test_BackGround_Y.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBufY);
	}

	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	nErrorCode = AlgoBase::TwoImg_Average(matDstBufX, matDstBufY, matDstBuf);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	matDstBufX.release();
	matDstBufY.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMuraNormal::RangeFitProcess(cv::Mat matSrcBuf, cv::Mat* matDstBuf, long lSegXUnit, long lSegYUnit, double dEstiBright, double dEstiDark, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	if (matSrcBuf.empty())	return false;
	if (dPara == NULL)		return false;

	// 	long	nSegX		= (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX		];
	// 	long	nSegY		= (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGY		];

	long	nSegX = lSegXUnit;
	long	nSegY = lSegYUnit;
	long	nEdgeArea = (int)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGEAREA];

	//图像大小
	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (long x = 0; x < nRangeX; x++)
		{

			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matSrcBuf.cols)	rectPtr[x].width = matSrcBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matSrcBuf.rows)	rectPtr[x].height = matSrcBuf.rows - rectPtr[x].y;

			// Get Buff
			cv::Mat matRFInput = cMatBufTemp.GetMat(rectPtr[x].height, rectPtr[x].width, matSrcBuf.type());
			cv::Mat matRFOutPut = cMatBufTemp.GetMat(rectPtr[x].height, rectPtr[x].width, matSrcBuf.type());

			matRFInput = matSrcBuf(rectPtr[x]);

			// X Direction Fit
			matRFOutPut = matDstBuf[E_DEFECT_COLOR_DARK](rectPtr[x]);
			RangeFit_X(matRFInput, matRFOutPut, dEstiBright, dEstiDark, dPara);

			// Y Direction Fit
			matRFOutPut = matDstBuf[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			RangeFit_Y(matRFInput, matRFOutPut, dEstiBright, dEstiDark, dPara);

		}
	}

	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//分区进化
long CInspectMuraNormal::RangeDataCalc(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matOrgImage, cv::Mat& matAvgValue, cv::Mat& matOrgValue, cv::Mat& matMaxValue,
	cv::Mat& matStdValue, cv::Mat& matPCRValue, cv::Mat& matHISTCount, vector<vector<float>>& matHistData, double* dPara, CMatBuf* cMemSub)
{
	//异常处理
	if (dPara == NULL) return false;

	//  long	nSegX				= (long )dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX		];
	//  long	nSegY				= (long )dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGY		];
	long	nEdgeArea = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGEAREA];
	float	fActiveDarkRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_DR];
	float	fActiveBrightRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_BR];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_DR];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_BR];
	int		nMinThGV = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MINGV];
	int		nMedian = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MEDIANSIZE];

	long	nSegX = 50;
	long	nSegY = 50;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start."));

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), true);
	cv::Mat matOrgBuff = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), true);

	// Range Avg
	cv::Mat matAvgSub = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matOrgSub = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matMaxSub = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matStdSub = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	cv::Mat matPCRate = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1); // Pixel Count Rate : None"0" / Total
	cv::Mat matHASCount = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1); // Pixel Count : Pixel > Avg + Std

	matSrcImage.copyTo(matBlurBuf);
	matOrgImage.copyTo(matOrgBuff);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Get Memory & Copy Image."));

	//////////////////////////////////////////////////////////////////////////
	//
	cv::Scalar TempMean, TempStd;

	// Min Max Calc.
	double TempMin = 0;
	double TempMax = 0;

	cv::meanStdDev(matBlurBuf, TempMean, TempStd);

	double dbTempAvg = TempMean[0];
	double dbTempStd = TempStd[0];

	//直方图
	cv::Mat matHisto;
	AlgoBase::GetHistogram(matBlurBuf, matHisto, false);

	double dblAverage;
	double dblStdev;
	AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

	//设置平均范围
	int nMinGV = (int)(dblAverage - dblStdev);
	int nMaxGV = (int)(dblAverage + dblStdev);

	if (nMinGV <= 0)	nMinGV = 1;
	if (nMaxGV > 255)	nMaxGV = 255;

	//初始化
	__int64 nPixelSum = 0;
	__int64 nPixelCount = 0;

	//仅重新平均设置的平均范围
	float* pVal = (float*)matHisto.ptr(0) + nMinGV;

	for (int m = nMinGV; m <= nMaxGV; m++, pVal++)
	{
		nPixelSum += (m * *pVal);
		nPixelCount += *pVal;
	}

	float fTotalAvg = (float)(nPixelSum / (float)nPixelCount);

	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	int nHistIndex = 0;
	//vector<vector<float>>		 matHistData;
	vector<vector<float>>().swap(matHistData);

	//平均计算
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		float* ptrAvg = (float*)matAvgSub.ptr(y);
		float* ptrOrg = (float*)matOrgSub.ptr(y);
		float* ptrMax = (float*)matMaxSub.ptr(y);
		float* ptrStd = (float*)matStdSub.ptr(y);
		float* ptrPCR = (float*)matPCRate.ptr(y);
		float* ptrHASC = (float*)matHASCount.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols) rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows) rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			cv::Scalar TempMean, TempStd;

			// Min Max Calc.
			double TempMin = 0;
			double TempMax = 0;

			cv::meanStdDev(matTempBuf, TempMean, TempStd);
			cv::minMaxIdx(matTempBuf, &TempMin, &TempMax, NULL, NULL);

			double dbTempAvg = TempMean[0];
			double dbTempStd = TempStd[0];

			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);

			// Org Avg
			cv::Mat matOrgBuf = matOrgBuff(rectPtr[x]);

			cv::Scalar orgMean, orgStd;

			cv::meanStdDev(matOrgBuf, TempMean, TempStd);

			double dbOrgAvg = TempMean[0];

			float* pZeroVal = (float*)matHisto.ptr(0);
			int nRectArea = matTempBuf.rows * matTempBuf.cols;

			ptrAvg[x] = (float)(dbTempAvg);
			ptrOrg[x] = (float)(dbOrgAvg);
			ptrMax[x] = (float)(TempMax);
			ptrStd[x] = (float)(dbTempStd);
			ptrPCR[x] = (float)((nRectArea - *pZeroVal) / nRectArea);

			// Buff Reset
			vector<float>		 SubHistData;

			float* pSubHistValue = (float*)matHisto.ptr(0);

			//更改为Reseze原始坐标
			for (int j = 0; j < matHisto.rows; j++, pSubHistValue++)
			{
				SubHistData.push_back(*pSubHistValue);
			}

			matHistData.push_back(SubHistData);
			vector<float>().swap(SubHistData);

			//仅按设置的平均范围重新平均
			float* pValHist = (float*)matHisto.ptr(0);

			__int64 nPixelCountHist = 0;

			for (int m = 0; m <= 255; m++, pValHist++)
			{
				if (m > fTotalAvg) nPixelCountHist += *pValHist;
				else continue;
			}
			ptrHASC[x] = nPixelCountHist;

		}
	}

	matAvgSub.copyTo(matAvgValue);
	matOrgSub.copyTo(matOrgValue);
	matMaxSub.copyTo(matMaxValue);
	matStdSub.copyTo(matStdValue);
	matPCRate.copyTo(matPCRValue);
	matHASCount.copyTo(matHISTCount);

	//////////////////////////////////////////////////////////////////////////
	// Draw

	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols) rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows) rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			cv::rectangle(matBlurBuf(rectPtr[x]), cv::Rect(0, 0, rectPtr[x].width, rectPtr[x].height), cv::Scalar(255, 255, 255));

		}
	}

	matBlurBuf.copyTo(matDstImage);

	//禁用
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//分区进化
long CInspectMuraNormal::RangeReFit(cv::Mat& matSrcImageBG, cv::Mat& matSrcImageData, cv::Mat& matDstImage, CMatBuf* cMemSub)
{

	long	nSegX = 25;
	long	nSegY = 25;
	int	nEdgeArea = 1;
	float	fRecheckRatio = 10;

	//图像大小
	long	nWidth = (long)matSrcImageBG.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImageBG.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start."));

	//Temp内存分配
	cv::Mat matBGBuff = cMatBufTemp.GetMat(matSrcImageBG.size(), matSrcImageBG.type());
	cv::Mat matDataBuff = cMatBufTemp.GetMat(matSrcImageBG.size(), matSrcImageBG.type());
	cv::Mat matResultBuff = cMatBufTemp.GetMat(matSrcImageBG.size(), matSrcImageBG.type());

	matSrcImageBG.copyTo(matBGBuff);
	matSrcImageData.copyTo(matDataBuff);

	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//平均计算
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBGBuff.cols) rectPtr[x].width = matBGBuff.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBGBuff.rows) rectPtr[x].height = matBGBuff.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBufBG = matBGBuff(rectPtr[x]);
			cv::Mat matTempBufData = matDataBuff(rectPtr[x]);
			cv::Mat matSubResult = cMatBufTemp.GetMat(matTempBufData.size(), matTempBufData.type());

			cv::Scalar BGMean, BGStd, DataMean, DataStd;

			cv::meanStdDev(matTempBufBG, BGMean, BGStd);
			cv::meanStdDev(matTempBufData, DataMean, DataStd);

			double dbBGAvg = BGMean[0];
			double dbDataAvg = DataMean[0];

			// 			 if (abs(dbDataAvg - dbBGAvg) >= fRecheckRatio)	matTempBufBG.copyTo(matResultBuff(rectPtr[x]));
			// 			 else											matTempBufData.copyTo(matResultBuff(rectPtr[x]));

			for (int i = 0; i < matTempBufBG.rows; i++)
			{
				uchar* ptrBG = (uchar*)matTempBufBG.ptr(i);
				uchar* ptrData = (uchar*)matTempBufData.ptr(i);
				uchar* ptrResult = (uchar*)matSubResult.ptr(i);

				for (int j = 0; j < matTempBufBG.cols; j++, ptrBG++, ptrData++, ptrResult++)
				{
					if (abs(*ptrBG - *ptrData) >= fRecheckRatio) 	*ptrResult = dbBGAvg;
					else 											*ptrResult = *ptrData;
				}
			}

			matSubResult.copyTo(matResultBuff(rectPtr[x]));
		}
	}

	matResultBuff.copyTo(matDstImage);

	//禁用
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

// Distance Transform & Threshold
long CInspectMuraNormal::DistanceTransformThreshold(cv::Mat& matSrcImage, cv::Mat& matDstImage, double dThreshold, CMatBuf* cMemSub)
{
	if (dThreshold <= 0.0)		return E_ERROR_CODE_TRUE;
	if (dThreshold >= 255.0)	return E_ERROR_CODE_TRUE;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), CV_32FC1, false);
	cv::Mat matTempBuf2 = cMatBufTemp.GetMat(matSrcImage.size(), CV_32FC1, false);

	//源8bit/结果32F
	if (matSrcImage.type() == CV_8U)
	{
		cv::distanceTransform(matSrcImage, matTempBuf1, CV_DIST_L2, 3);

		cv::threshold(matTempBuf1, matTempBuf2, dThreshold, 150.0, THRESH_BINARY);
		//cv::threshold(matTempBuf1, matTempBuf2, dThreshold, 255.0, THRESH_BINARY);
	}
	else
	{
		cv::Mat matSrc8bit = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1, false);
		matSrcImage.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

		cv::distanceTransform(matSrc8bit, matTempBuf1, CV_DIST_L2, 3);

		matSrc8bit.release();

		cv::threshold(matTempBuf1, matTempBuf2, dThreshold, 4095.0, THRESH_BINARY);
		//cv::threshold(matTempBuf1, matTempBuf2, dThreshold, 4095.0, THRESH_BINARY);
	}

	//bit转换
	matTempBuf2.convertTo(matDstImage, matSrcImage.type());

	matTempBuf1.release();
	matTempBuf2.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

// Rang Combination Process
long CInspectMuraNormal::RangComProcess(cv::Mat& matSrcImage, cv::Mat& matDstImage, long lSegXUnit, long lSegYUnit, double dEstiBright, double dEstiDark, double* dPara,
	int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{

	//异常处理
	if (dPara == NULL) return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("T3 RCP Start."));

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matRFBGBuf[E_DEFECT_COLOR_COUNT];

	matRFBGBuf[E_DEFECT_COLOR_DARK] = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	matRFBGBuf[E_DEFECT_COLOR_BRIGHT] = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	cv::Mat matAvgDefect = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matSubData = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matMorphBuff = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("T3 RCP Get Buff."));

	RangeFitProcess(matSrcImage, matRFBGBuf, lSegXUnit, lSegYUnit, dEstiBright, dEstiDark, dPara, nCommonPara, strAlgPath, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("T3 RCP Rang Fit Process-RF."));

	AlgoBase::TwoImg_Average(matRFBGBuf[E_DEFECT_COLOR_DARK], matRFBGBuf[E_DEFECT_COLOR_BRIGHT], matAvgDefect);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("T3 RCP Rang Fit Process-Avg."));

	matAvgDefect.copyTo(matDstImage);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

// Result Combination Process
long CInspectMuraNormal::ResultCombination(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matDstImage, int nEdgePara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//异常处理
	if (matSrcImage1.empty() || matSrcImage2.empty() || nEdgePara == 0) return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Result Combination Start."));

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matBuff1 = cMatBufTemp.GetMat(matSrcImage2.size(), matSrcImage2.type(), false);
	cv::Mat matBuff2 = cMatBufTemp.GetMat(matSrcImage2.size(), matSrcImage2.type(), false);

	cv::Rect SubmatROI = cv::Rect(nEdgePara, nEdgePara, matSrcImage1.cols - nEdgePara * 2, matSrcImage1.rows - nEdgePara * 2);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Result Combination set Buff & ROI."));

	matBuff1 = matSrcImage1(SubmatROI) + matSrcImage2;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURANormal_%02d_Type3_SBS_ResultCom_ResultSUM.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBuff1);
	}

	cv::threshold(matBuff1, matBuff2, 100.0, 150.0, THRESH_BINARY);

	matBuff2.copyTo(matDstImage(SubmatROI));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

// Rang Background Enhancement Process
long CInspectMuraNormal::RangBGEnhance(cv::Mat& matSrcImage1, cv::Mat& matSrcImage2, cv::Mat& matAvgDefect, cv::Mat& matBGEnhanceResult, cv::Mat& matAvgResult, long lSegXUnit, long lSegYUnit, double* dPara, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
{
	//异常处理
	if (matSrcImage1.empty() || matSrcImage2.empty() || matAvgDefect.empty()) return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//////////////////////////////////////////////////////////////////////////
	// X-Y Avg Data BackGround Estimation
// 	long	nSegX				= (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX			];
// 	long	nSegY				= (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGY			];

	long	nSegX = lSegXUnit;
	long	nSegY = lSegYUnit;
	long	nEdgeArea = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGEAREA];
	int		nMedian = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MEDIANSIZE];

	//图像大小
	long	nWidth = (long)matAvgDefect.cols;	// 图像宽度大小
	long	nHeight = (long)matAvgDefect.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);

	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//平均计算
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		//BYTE *ptr = (BYTE *)matAvgBuf.ptr(y);

		float* ptr = (float*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matAvgDefect.cols) rectPtr[x].width = matAvgDefect.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matAvgDefect.rows) rectPtr[x].height = matAvgDefect.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBuf = matAvgDefect(rectPtr[x]);

			//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);

			double dblAverage;
			double dblStdev;
			AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

			//设置平均范围
			int nMinGV = (int)(dblAverage - dblStdev);
			int nMaxGV = (int)(dblAverage + dblStdev);
			if (nMinGV < 0)	nMinGV = 0;
			if (nMaxGV > 255)	nMaxGV = 255;

			//初始化
			__int64 nPixelSum = 0;
			__int64 nPixelCount = 0;

			//仅按设置的平均范围重新平均
			float* pVal = (float*)matHisto.ptr(0) + nMinGV;

			for (int m = nMinGV; m <= nMaxGV; m++, pVal++)
			{
				nPixelSum += (m * *pVal);
				nPixelCount += *pVal;
			}

			//至少要有一个数量...
			if (nPixelCount > 0)	ptr[x] = (float)(nPixelSum / (float)nPixelCount);
			if (ptr[x] < 1)	ptr[x] = 1;
		}
	}

	matAvgBuf.copyTo(matAvgResult);

	//////////////////////////////////////////////////////////////////////////
	// X-Y Avg Fit Result

	cv::Mat matResultBuff = cMatBufTemp.GetMat(matBGEnhanceResult.size(), matBGEnhanceResult.type(), false);

	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matAvgDefect.cols) rectPtr[x].width = matAvgDefect.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matAvgDefect.rows) rectPtr[x].height = matAvgDefect.rows - rectPtr[x].y;

			// EstimateX Data
			cv::Mat matEstimatX = matSrcImage1(rectPtr[x]);
			cv::Mat matEstimatY = matSrcImage2(rectPtr[x]);

			cv::Mat matEXProfile = cMatBufTemp.GetMat(cv::Size(rectPtr[x].width, 1), CV_8UC1);
			cv::Mat matEYProfile = cMatBufTemp.GetMat(cv::Size(1, rectPtr[x].height), CV_8UC1);
			cv::Mat matXYBuff = cMatBufTemp.GetMat(cv::Size(rectPtr[x].width, 1), CV_8UC1);
			cv::Mat matXYEstimate = cMatBufTemp.GetMat(cv::Size(rectPtr[x].width, rectPtr[x].height), CV_8UC1);

			cv::Mat matLineFillX = cMatBufTemp.GetMat(cv::Size(rectPtr[x].width, rectPtr[x].height), CV_8UC1);
			cv::Mat matLineFillY = cMatBufTemp.GetMat(cv::Size(rectPtr[x].width, rectPtr[x].height), CV_8UC1);
			cv::Mat matLineFillAvg = cMatBufTemp.GetMat(cv::Size(rectPtr[x].width, rectPtr[x].height), CV_8UC1);

			AlgoBase::MakeProfile(matEstimatX, matEXProfile, rectPtr[x].width, rectPtr[x].height, 0);
			AlgoBase::MakeProfile(matEstimatY, matEYProfile, rectPtr[x].width, rectPtr[x].height, 1);

			//////////////////////////////////////////////////////////////////////////
			// Estimate Data Value ( Test ver 1.0 )
// 			BYTE *ptrEstY = (BYTE *)	matEYProfile.ptr(0);
// 			
// 			for (int i = 0; i < matXYEstimate.rows; i++, ptrEstY++)
// 			{
// 				BYTE *ptrEstX		= (BYTE *)	matEXProfile.ptr(0);
// 				BYTE *ptrEstXYBuff	= (BYTE *)	matXYBuff.ptr(0);
// 
// 				for (int j = 0; j < matXYEstimate.cols;j++, ptrEstX++, ptrEstXYBuff++)
// 				{
// 					*ptrEstXYBuff = (BYTE) ((*ptrEstY + *ptrEstX) / 2);
// 				}
// 
// 				matXYBuff.copyTo(matXYEstimate(cv::Rect(0, i, matXYEstimate.cols, 1)));
// 			}
// 
// 			// Copy Result
// 			matXYEstimate.copyTo(matResultBuff(rectPtr[x]));

			//////////////////////////////////////////////////////////////////////////
			// Estimate Data Value ( Test ver 2.0 )

			for (int i = 0; i < matXYEstimate.rows; i++)
			{
				matEXProfile.copyTo(matLineFillX(cv::Rect(0, i, matLineFillX.cols, 1)));
			}

			for (int i = 0; i < matXYEstimate.cols; i++)
			{
				matEYProfile.copyTo(matLineFillY(cv::Rect(i, 0, 1, matLineFillY.rows)));
			}

			AlgoBase::TwoImg_Average(matLineFillX, matLineFillY, matLineFillAvg);

			matLineFillAvg.copyTo(matResultBuff(rectPtr[x]));

		}
	}

	matResultBuff.copyTo(matBGEnhanceResult);

	//禁用
	matAvgBuf.release();
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//分区进化
long CInspectMuraNormal::RangeCheckThreshold(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matHistAvgValue, cv::Mat& matAvgValue, cv::Mat& matOrgValue, cv::Mat& matMaxValue,
	cv::Mat& matStdValue, cv::Mat& matPCRValue, cv::Mat& matHISTCount, vector<vector<float>>& matHistData, double* dPara, CMatBuf* cMemSub)
{
	//异常处理
	if (dPara == NULL) return false;

	long	nEdgeArea = 1;
	long	nSegX = 50;
	long	nSegY = 50;

	float	fTH_STR = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_STR];	//40;
	float	fTH_Mid = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_MID1];	//35;
	float	fTH_Mid2 = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_MID2];	//15;
	float	fTH_Weak1 = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_WEAK1];	//10;
	float	fTH_Weak2 = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_BBS_TH_VALUE_WEAK2];	//8;

	//float	fTH_Weak2 = 3;

	float	fCheck_Edge1 = 20;
	float	fCheck_Edge2 = 10;
	float	fCheck_Active = 5;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start."));

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	matSrcImage.copyTo(matBlurBuf);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Get Memory & Copy Image."));

	//////////////////////////////////////////////////////////////////////////
	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Projection Data Check
	cv::Mat matProDataX = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Profile Buff
	cv::Mat matProDataY = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Profile Buff
	cv::Mat matRegionData = cMatBufTemp.GetMat(cv::Size(nRangeY, 1), CV_32FC1);

	AlgoBase::MakeProfile(matBlurBuf, matProDataX, nWidth, nHeight, 6);
	AlgoBase::MakeProfile(matBlurBuf, matProDataY, nWidth, nHeight, 7);

	//////////////////////////////////////////////////////////////////////////
	// Region Data Check

	float* ptrRegionData = (float*)matRegionData.ptr(0);

	for (long y = 0; y < nRangeY; y++, ptrRegionData++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		int nStartX = rectPtr[0].y;
		int nLength = rectPtr[0].height;

		if (nStartX + nLength > matProDataY.cols) nLength = matProDataY.cols - nStartX;

		// Min Max Calc.
		double TempMin = 0;
		double TempMax = 0;

		cv::minMaxIdx(matProDataY(cv::Rect(nStartX, 0, nLength, 1)), &TempMin, &TempMax, NULL, NULL);

		*ptrRegionData = TempMax;
	}

	//////////////////////////////////////////////////////////////////////////
	// Region Max Data

	// Avg-StdDev
	cv::Scalar MaxAvg, MaxStd;
	cv::meanStdDev(matMaxValue(cv::Rect(1, 1, matMaxValue.cols - 2, matMaxValue.rows - 2)), MaxAvg, MaxStd);

	double dbMaxAvg = MaxAvg[0];
	double dbMaxStd = MaxStd[0];

	// Min-Max
	double dbMaxMaxValue = 0;
	double dbMaxMinValue = 0;

	cv::minMaxIdx(matMaxValue(cv::Rect(1, 1, matMaxValue.cols - 2, matMaxValue.rows - 2)), &dbMaxMinValue, &dbMaxMaxValue, NULL, NULL);

	int nMaxStdTimePara = 2;

	float fTHMaxValue = (float)(dbMaxAvg + dbMaxStd * nMaxStdTimePara);

	//////////////////////////////////////////////////////////////////////////
	// Region Check

	float* fRegionData = (float*)matRegionData.ptr(0);

	vector<int> vfYDefect;
	vector<int>().swap(vfYDefect);

	for (long y = 0; y < nRangeY; y++, fRegionData++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		int nRegionCount_Edge_H = 0;
		int nRegionCount_Edge_V = 0;
		int nRegionCount_Edge_Weak = 0;
		int nRegionCount_Active_Strong = 0;
		int nRegionCount_Active_Weak = 0;

		bool bActiveDefect = false;
		bool bEdgeDefect = false;

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols)		rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows)		rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			// Region Check Data
			float fDataValue = matAvgValue.at<float>(y, x) + matStdValue.at<float>(y, x);
			float fDataAvg = matAvgValue.at<float>(y, x);
			float fDataRatio = matPCRValue.at<float>(y, x);
			float fTempMax = matMaxValue.at<float>(y, x);
			float fDataHistC = matHISTCount.at<float>(y, x);

			if (fDataAvg <= 1.5 && fDataRatio <= 0.15 && fDataValue < 4 && (fTempMax < fTHMaxValue))
			{
				matBlurBuf(rectPtr[x]).setTo(0);
				continue;
			}

			bool bDeleteRegion = false;
			bool bRegionMaxCheck = false;
			bool bEdge2Region = false;
			bool bNeighborDefect = false;

			// Conner Data Check
			if ((y == 0 && x == 0) || (y == 0 && x == nRangeX - 1) || (y == nRangeY - 1 && x == 0) || (y == nRangeY - 1 && x == nRangeX - 1))
			{
				if (fDataValue >= fTH_STR)	bDeleteRegion = false;
				else						bDeleteRegion = true;
			}

			// Edge Data Check
			else if ((y == 0 && (x > 0 && x < nRangeX - 1)) || (y == nRangeY - 1 && (x > 0 && x < nRangeX - 1)) ||
				((x == 0 && (y > 0 && y < nRangeY - 1)) || (x == nRangeX - 1 && (y > 0 && y < nRangeY - 1))))
			{
				//案例1:20或更高版本
				if (*fRegionData >= fCheck_Edge1)
				{
					if (fDataValue >= fTH_STR)	bDeleteRegion = false;
					else							bDeleteRegion = true;
				}

				// Case 2 : 10 ~ 20
				else if (*fRegionData >= fCheck_Edge2 && *fRegionData < fCheck_Edge1)
				{
					if (fDataValue >= 10)	nRegionCount_Edge_H++;
					if (fDataValue >= fTH_STR)	bEdgeDefect = true;
					if (fDataValue >= fTH_Mid) 	bDeleteRegion = false;
					else							bDeleteRegion = true;
				}

				// Case 3 : 0 ~ 10
				else if (*fRegionData >= 0 && *fRegionData < fCheck_Edge2)
				{
					if (fDataValue >= 8)	nRegionCount_Edge_Weak++;
					if (fDataValue >= fTH_STR)	bEdgeDefect = true;
					if (fDataValue >= fTH_Mid) 	bDeleteRegion = false;
					else							bDeleteRegion = true;
				}
			}

			// Active Data Check
			else if ((y > 0 && y < nRangeY - 1) && (x > 0 && x < nRangeX - 1))
			{

				if (fTempMax == dbMaxMaxValue)		bRegionMaxCheck = true;

				// X Direction Profile Check
				if (*fRegionData >= fCheck_Active)
				{
					if (fDataValue >= 4)	nRegionCount_Active_Strong++;
					if (fDataValue >= fTH_STR)	bActiveDefect = true;
					if (fDataValue >= fTH_Mid2) 	bDeleteRegion = false;
					else							bDeleteRegion = true;
				}

				else
				{
					if (fDataValue >= 4)	nRegionCount_Active_Weak++;
					if (fDataValue >= fTH_Mid2)	bActiveDefect = true;
					if (fDataValue >= fTH_Weak2) 	bDeleteRegion = false;
					else							bDeleteRegion = true;
				}

				//疑似不良
				if ((bDeleteRegion == false) || ((bDeleteRegion == true) && (bRegionMaxCheck == true)))
				{
					bool bSameCol = false;

					// Check vfYDefect
					if (vfYDefect.size() > 0)
					{
						for (int k = 0; k < vfYDefect.size(); k++)
						{
							if (x == vfYDefect[k]) { bSameCol = true; break; }
						}
					}

					if (bSameCol == true) continue;

					int nStartX = rectPtr[x].x;
					int nLength = rectPtr[x].width;

					if (nStartX + nLength > matProDataX.cols) nLength = matProDataX.cols - nStartX;

					// Min Max Calc.
					double TempMin = 0;
					double TempMax = 0;

					cv::minMaxIdx(matProDataX(cv::Rect(nStartX, 0, nLength, 1)), &TempMin, &TempMax, NULL, NULL);

					int nYDefectCount = 0;

					//Y好像有问题
					if (TempMax >= 2)
					{
						for (int j = 1; j < nRangeY - 1; j++)
						{
							fDataValue = matAvgValue.at<float>(j, x) + matStdValue.at<float>(j, x);
							if (fDataValue >= 4) nYDefectCount++;
						}
					}

					if (nYDefectCount >= 3) vfYDefect.push_back(x);
				}

				// Max Value Check ???
				if (bDeleteRegion == false)
				{
					if ((fTempMax >= fTHMaxValue) && (fTempMax == dbMaxMaxValue))	bDeleteRegion = false;
					else if (fDataHistC > 900)											bDeleteRegion = false;
					else																bDeleteRegion = true;
				}

				// Neighbor Region Check
				if (x == 1 || x == nRangeX - 2) bEdge2Region = true;
				else
				{
					if ((fTempMax == dbMaxMaxValue) && (fDataValue < fTH_Weak2) && (fDataValue >= 5))
					{
						int nNeighborCount = 0;

						for (int i = y - 1; i <= y + 1; i++)
						{
							for (int j = x - 1; j <= x + 1; j++)
							{
								float fNeighborMax = matMaxValue.at<float>(i, j);
								if (fNeighborMax == dbMaxMaxValue) nNeighborCount++;
							}
						}

						if (nNeighborCount >= 4) bNeighborDefect = true;
					}
				}
			}

			// Delete Or Not
			if (bDeleteRegion == true && bRegionMaxCheck == false && bNeighborDefect == false)	matBlurBuf(rectPtr[x]).setTo(0);
			else if (bDeleteRegion == true && bRegionMaxCheck == true && bNeighborDefect == false && bEdge2Region == false)	matBlurBuf(rectPtr[x]).setTo(0);
			else if (bDeleteRegion == true && bRegionMaxCheck == true && bEdge2Region == true)	continue;
			else																												continue;
		}

		//////////////////////////////////////////////////////////////////////////
				//检查是否有特殊分布
		if (nRegionCount_Edge_H >= 10 && bEdgeDefect == false) matBlurBuf(cv::Rect(rectPtr[0].x, rectPtr[0].y, nWidth, rectPtr[0].height)).setTo(0);
		if (nRegionCount_Edge_Weak >= 10 && bEdgeDefect == false) matBlurBuf(cv::Rect(rectPtr[0].x, rectPtr[0].y, nWidth, rectPtr[0].height)).setTo(0);
		if ((nRegionCount_Edge_H >= 10 && bEdgeDefect == true) || (nRegionCount_Edge_Weak >= 10 && bEdgeDefect == false))
		{
			for (long x = 0; x < nRangeX; x++)
			{
				//异常处理
				if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols)		rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
				if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows)		rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

				// Region Check Data
				float fDataValue = matAvgValue.at<float>(y, x) + matStdValue.at<float>(y, x);

				if (fDataValue >= fTH_STR) continue;
				else                       matBlurBuf(rectPtr[x]).setTo(0);
			}
		}

		if (nRegionCount_Active_Strong >= 6 && bActiveDefect == false) matBlurBuf(cv::Rect(rectPtr[0].x, rectPtr[0].y, nWidth, rectPtr[0].height)).setTo(0);
		if (nRegionCount_Active_Weak >= 6 && bActiveDefect == false) matBlurBuf(cv::Rect(rectPtr[0].x, rectPtr[0].y, nWidth, rectPtr[0].height)).setTo(0);
		if ((nRegionCount_Active_Strong >= 10 && bActiveDefect == true) || (nRegionCount_Active_Weak >= 6 && bActiveDefect == true))
		{
			for (long x = 0; x < nRangeX; x++)
			{
				//异常处理
				if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols)		rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
				if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows)		rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

				// Region Check Data
				float fDataValue = matAvgValue.at<float>(y, x) + matStdValue.at<float>(y, x);

				if (fDataValue >= fTH_STR) continue;
				else                       matBlurBuf(rectPtr[x]).setTo(0);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
		//Y Projection Data验证

	if (vfYDefect.size() > 0)
	{
		for (int k = 0; k < vfYDefect.size(); k++)
		{
			cv::Rect* rectPtr = &rectRange[vfYDefect[k]];

			for (int i = 0; i < nRangeY; i++)
			{
				float fDataValue = matAvgValue.at<float>(i, vfYDefect[k]) + matStdValue.at<float>(i, vfYDefect[k]);

				if (fDataValue >= fTH_STR) continue;
				else	  				  matBlurBuf(rectPtr[0 + i * nRangeX]).setTo(0);
			}
		}
	}

	matBlurBuf.copyTo(matDstImage);

	//禁用
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//分区进化
long CInspectMuraNormal::RangeFitThreshold(cv::Mat& matSrcImage, cv::Mat* matDstImage, double* dPara, CMatBuf* cMemSub)
{
	//异常处理
	if (dPara == NULL) return false;

	long	nSegX = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGX];
	long	nSegY = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_SEGY];
	long	nEdgeArea = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGEAREA];
	float	fActiveDarkRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_DR];
	float	fActiveBrightRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_ACTIVE_BR];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_DR];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_EDGE_BR];
	int		nMinThGV = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MINGV];
	int		nMedian = (long)dPara[E_PARA_AVI_MURA_NORMAL_TYPE3_SBS_MEDIANSIZE];

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start."));

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, CV_32FC1);
	//cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, matSrcImage.type());

	matSrcImage.copyTo(matBlurBuf);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Get Memory & Copy Image."));

	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//平均计算
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		//BYTE *ptr = (BYTE *)matAvgBuf.ptr(y);

		float* ptr = (float*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols) rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows) rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);

			double dblAverage;
			double dblStdev;
			AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

			//设置平均范围
			int nMinGV = (int)(dblAverage - dblStdev);
			int nMaxGV = (int)(dblAverage + dblStdev);
			if (nMinGV < 0)	nMinGV = 0;
			if (nMaxGV > 255)	nMaxGV = 255;

			//初始化
			__int64 nPixelSum = 0;
			__int64 nPixelCount = 0;

			//仅按设置的平均范围重新平均
			float* pVal = (float*)matHisto.ptr(0) + nMinGV;

			for (int m = nMinGV; m <= nMaxGV; m++, pVal++)
			{
				nPixelSum += (m * *pVal);
				nPixelCount += *pVal;
			}

			//至少要有一个数量...
//if (nPixelCount > 0)	ptr[x] = (BYTE)(nPixelSum / (float)nPixelCount);
			if (nPixelCount > 0)	ptr[x] = (float)(nPixelSum / (float)nPixelCount);
			if (ptr[x] < 1)	ptr[x] = 1;
		}
	}

	//设置为周边平均值->中间值
//MedianFilter(matAvgBuf, matAvgBuf, nMedian, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("MedianFilter."));

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		//BYTE *ptr = (BYTE *)matAvgBuf.ptr(y);
		float* ptr = (float*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matBlurBuf.cols) rectPtr[x].width = matBlurBuf.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matBlurBuf.rows) rectPtr[x].height = matBlurBuf.rows - rectPtr[x].y;

			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//平均*Ratio
			long nDarkTemp = 0;
			long nBrightTemp = 0;

			//单独设置Edge
			if (x < nEdgeArea ||
				y < nEdgeArea ||
				x >= nRangeX - nEdgeArea ||
				y >= nRangeY - nEdgeArea)
			{
				// Conner
				if ((y == 0 && x == 0) || (y == 0 && x == nRangeX - 1) || (y == nRangeY - 1 && x == 0) || (y == nRangeY - 1 && x == nRangeX - 1))
				{
					nDarkTemp = (long)(ptr[x] * fEdgeDarkRatio);
					nBrightTemp = (long)(ptr[x] * fEdgeBrightRatio);
				}

				// Edge
				else
				{
					// Min Max Calc.
					double TempMin = 0;
					double TempMax = 0;

					cv::minMaxIdx(matTempBuf, &TempMin, &TempMax, NULL, NULL);

					if (TempMax >= 90) nBrightTemp = (long)(ptr[x] * 13);
					else              nBrightTemp = (long)(ptr[x] * fEdgeBrightRatio);
				}

			}

			else
			{
				nDarkTemp = (long)(ptr[x] * fActiveDarkRatio);
				nBrightTemp = (long)(ptr[x] * fActiveBrightRatio);
			}

			//////////////////////////////////////////////////////////////////////////
			// Edge Region Recheck

						//Gray有太暗的情况。
						//(平均GV2~3*fBrightRatio->二进制:噪声全部上升)
			if (nBrightTemp < nMinThGV)	nBrightTemp = nMinThGV;

			//异常处理
//if (nDarkTemp < 0)		nDarkTemp = 0;
			if (nBrightTemp > 255)	nBrightTemp = 255;

			//参数0时异常处理
	// 			if (fActiveDarkRatio == 0)		nDarkTemp = -1;
	// 			if (fActiveBrightRatio == 0)	nBrightTemp = 256;

				// E_DEFECT_COLOR_DARK Threshold
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, 0, 0, THRESH_BINARY_INV);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, nBrightTemp, 150.0, THRESH_BINARY);
		}
	}

	//禁用
	matAvgBuf.release();
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//分区进化
long CInspectMuraNormal::SBSandBBSEdgeReCheck(cv::Mat& matSBSTemp, cv::Mat& matBBSTemp, cv::Mat& matDstImage, cv::Mat& matAvgTemp, cv::Mat& matStdTemp, cv::Mat& matMaxTemp, int nEdgeUnite, CMatBuf* cMemSub)
{
	// Check
	if (matSBSTemp.empty() && matBBSTemp.empty()) return false;
	if (nEdgeUnite <= 0)return false;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_NORMAL, __FUNCTION__, _T("Start."));

	// Rect
	long	nSegX = 50;
	long	nSegY = 50;
	long	nEdgeArea = 1;
	//图像大小
	long	nWidth = (long)matSBSTemp.cols;	// 图像宽度大小
	long	nHeight = (long)matSBSTemp.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = nWidth / nSegX + 1;
	int nRangeY = nHeight / nSegY + 1;

	//////////////////////////////////////////////////////////////////////////
	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY];
	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

	//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		if (y < nRangeY - nEdgeArea)
		{
			nStart_Y = y * nSegY;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = nHeight - nEdgeArea * nSegY;
			else									nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = nHeight - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = nWidth - nEdgeArea * nSegX;
				else									nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = nWidth - (nRangeX - x) * nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	float fJudgeValue = 20;
	int	  nJudgeMax = 50;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//Temp内存分配
	cv::Mat matSBSEdgeBuff = cMatBufTemp.GetMat(matSBSTemp.size(), matSBSTemp.type());
	cv::Mat matBBSEdgeBuff = cMatBufTemp.GetMat(matSBSTemp.size(), matSBSTemp.type());
	cv::Mat matSubEdgeBuff = cMatBufTemp.GetMat(matSBSTemp.size(), matSBSTemp.type());
	//cv::Mat matSubEdgeCalc  = cMatBufTemp.GetMat(matSBSTemp.size(), matSBSTemp.type());

	matSBSTemp.copyTo(matSBSEdgeBuff);
	matSBSTemp.copyTo(matSubEdgeBuff);
	//////////////////////////////////////////////////////////////////////////

	cv::Scalar ResultMean, ResultStd;

	cv::meanStdDev(matBBSTemp, ResultMean, ResultStd);

	double dbAvg = ResultMean[0];
	double dbStd = ResultStd[0];

	double dbTHValue = dbAvg + dbStd;

	cv::threshold(matBBSTemp, matBBSEdgeBuff, dbTHValue, 255.0, THRESH_BINARY);

	//////////////////////////////////////////////////////////////////////////
	// Projection Data Check
	int nProjectionTH = 20;
	int nDeleteUnit = 5;

	cv::Mat matProDataX = cMatBufTemp.GetMat(cv::Size(nWidth, 1), CV_32FC1);		// Mat 16bit Profile Buff
	cv::Mat matProDataY = cMatBufTemp.GetMat(cv::Size(nHeight, 1), CV_32FC1);		// Mat 16bit Profile Buff

	AlgoBase::MakeProfile(matBBSTemp, matProDataX, nWidth, nHeight, 6);
	AlgoBase::MakeProfile(matBBSTemp, matProDataY, nWidth, nHeight, 7);

	double dbTopEdgeMin = 0;
	double dbTopEdgeMax = 0;
	double dbBottomEdgeMin = 0;
	double dbBottomEdgeMax = 0;
	double dbLeftEdgeMin = 0;
	double dbLeftEdgeMax = 0;
	double dbRightEdgeMin = 0;
	double dbRightEdgeMax = 0;

	cv::minMaxIdx(matProDataY(cv::Rect(0, 0, nSegX, 1)), &dbTopEdgeMin, &dbTopEdgeMax, NULL, NULL);
	cv::minMaxIdx(matProDataY(cv::Rect(nHeight - nSegX, 0, nSegX, 1)), &dbBottomEdgeMin, &dbBottomEdgeMax, NULL, NULL);
	cv::minMaxIdx(matProDataX(cv::Rect(0, 0, nSegX, 1)), &dbLeftEdgeMin, &dbLeftEdgeMax, NULL, NULL);
	cv::minMaxIdx(matProDataX(cv::Rect(nWidth - nSegX, 0, nSegX, 1)), &dbRightEdgeMin, &dbRightEdgeMax, NULL, NULL);

	// 	int nOpenUnit = 7;
	// 
	// 	Morphology(matBBSEdgeBuff, matBBSEdgeBuff, nOpenUnit, nOpenUnit, EP_MORP_MURANORMAL_OPEN, &cMatBufTemp);

		//////////////////////////////////////////////////////////////////////////
		// Edge Max Calc.
	// 	double dbEdgeMin = 0;
	// 	double dbEdgeMax = 0;
	// 
	// 	matBBSTemp.copyTo(matSubEdgeCalc);
	// 	matSubEdgeCalc(cv::Rect(nSegX, nSegX, matBBSTemp.cols - nSegX * 2, matBBSTemp.rows - nSegX * 2)).setTo(0);
	// 
	// 	cv::minMaxIdx(matSubEdgeCalc, &dbEdgeMin, &dbEdgeMax, NULL, NULL);

		//////////////////////////////////////////////////////////////////////////
		// 

	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(1)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//异常处理
			if (rectPtr[x].x + rectPtr[x].width > matSBSTemp.cols) rectPtr[x].width = matSBSTemp.cols - rectPtr[x].x;
			if (rectPtr[x].y + rectPtr[x].height > matSBSTemp.rows) rectPtr[x].height = matSBSTemp.rows - rectPtr[x].y;

			// Region Check Data
			float fASValue = matAvgTemp.at<float>(y, x) + matStdTemp.at<float>(y, x);
			float fMaxValue = matMaxTemp.at<float>(y, x);

			// Active Region
			if ((y > 0 && y < nRangeY - 1) && (x > 0 && x < nRangeX - 1)) continue;

			//else if (fMaxValue == dbEdgeMax) continue;

			// Edge Region ( Edge + Conner )
			else
			{
				if ((fASValue >= fJudgeValue) && (fMaxValue >= nJudgeMax)) continue;
				else
				{
					matSubEdgeBuff(rectPtr[x]) = matSBSEdgeBuff(rectPtr[x]) - matBBSEdgeBuff(rectPtr[x]);
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////

	if (dbTopEdgeMax >= nProjectionTH || dbBottomEdgeMax >= nProjectionTH || dbLeftEdgeMax >= nProjectionTH || dbRightEdgeMax >= nProjectionTH)
	{
		if (dbTopEdgeMax >= nProjectionTH) matSubEdgeBuff(cv::Rect(0, 0, nWidth, nDeleteUnit)).setTo(0);
		if (dbBottomEdgeMax >= nProjectionTH) matSubEdgeBuff(cv::Rect(0, nHeight - nDeleteUnit, nWidth, nDeleteUnit)).setTo(0);
		if (dbLeftEdgeMax >= nProjectionTH) matSubEdgeBuff(cv::Rect(0, 0, nDeleteUnit, nHeight)).setTo(0);
		if (dbRightEdgeMax >= nProjectionTH) matSubEdgeBuff(cv::Rect(nWidth - nDeleteUnit, 0, nDeleteUnit, nHeight)).setTo(0);
	}

	matSubEdgeBuff.copyTo(matDstImage);

	//禁用
	delete[] rectRange;
	rectRange = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());

	}

	return E_ERROR_CODE_TRUE;
}
