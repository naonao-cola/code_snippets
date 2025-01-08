
/************************************************************************/
//Mura不良检测相关源
//修改日期:18.05.31
/************************************************************************/

#include "StdAfx.h"
#include "InspectMura.h"
#include "AlgoBase.h"
#include <numeric>





CInspectMura::CInspectMura(void)
{
	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
	sz = 11;
}

CInspectMura::~CInspectMura(void)
{
}

//Main检查算法
long CInspectMura::DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{
	//如果参数为NULL
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

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
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ROI_OVER;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ROI_OVER;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ROI_OVER;

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//缓冲区分配和初始化
	cv::Mat matDstImage[E_DEFECT_COLOR_COUNT];

	//算法画面号
	int nAlgImgNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];

	//缓冲区分配
	matDstImage[E_DEFECT_COLOR_DARK] = cMem->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
	matDstImage[E_DEFECT_COLOR_BRIGHT] = cMem->GetMat(matSrcBuffer.size(), matSrcBuffer.type());

	//如果结果缓冲区不是NULL,则出现错误
	if (matDstImage[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	//每个画面的算法不同
	switch (nAlgImgNum)
	{
	case E_IMAGE_CLASSIFY_AVI_R:
	case E_IMAGE_CLASSIFY_AVI_G:
	case E_IMAGE_CLASSIFY_AVI_B:
	{
		//检测RGB Line Mura
		//为避免在2022.12.21 R,G,B模式下进行Mura检查而修补潜艇
		//2023.02.16设置为可检查客户的请求
		nErrorCode = LogicStart_RGB_LINE_MURA(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

		//18.09.21-RGB检测出不良,也需要进行漏气检测
		//检测到R,G,B Line故障
//if ( nErrorCode == E_ERROR_CODE_MURA_RGB_LINE_DEFECT )
//{
			//	//取消分配
//	matDstImage[E_DEFECT_COLOR_BRIGHT].release();
//	matDstImage[E_DEFECT_COLOR_DARK].release();
//}
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_BLACK:
	case E_IMAGE_CLASSIFY_AVI_PCD:
	case E_IMAGE_CLASSIFY_AVI_VINIT:
	{
		//不检查
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_DUST:
	{
		//使用Point检查的结果
	}
	break;
	//case E_IMAGE_CLASSIFY_AVI_GRAY_128:
// 	{
// 		bool bFlag = true;
// 		bool bOnOff = ((int)dPara[E_PARA_AVI_MURA_G3_CM2_TEXT] > 0 ? true : false);
// 
// 		//nErrorCode = LogicStart_MuraG3CM(matSrcBuffer, matBKBuffer, rectROI, dPara, nCommonPara,strAlgPath, EngineerBlockDefectJudge, pResultBlob, &bFlag);
// 		// 2022.02.28
// 		if(bFlag && nErrorCode == E_ERROR_CODE_TRUE && bOnOff)
// 			//nErrorCode = LogicStart_MuraG3CM2(matSrcBuffer, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
// 
// 		nErrorCode = LogicStart_MuraG3CM3(matSrcBuffer, matBKBuffer, matDstImage[E_DEFECT_COLOR_DARK],rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
// 	}
	break;
	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	case E_IMAGE_CLASSIFY_AVI_WHITE:
	{
		//检测出100分
		nErrorCode = LogicStart_SPOT(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);

		//if ((int)dPara[E_PARA_AVI_MURA_RING_TEXT] > 0 ? true : false)
		//{
		//	long nErrorCode2 = E_ERROR_CODE_TRUE;
		//	cv::Mat matDst = cv::Mat::zeros(matDstImage[E_DEFECT_COLOR_BRIGHT].size(),CV_8UC1);
		//	nErrorCode2 = LogicStart_RingMura(matSrcBuffer, matSrcBufferRGB, matDst, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);

		//	cv::add(matDstImage[E_DEFECT_COLOR_BRIGHT], matDst, matDstImage[E_DEFECT_COLOR_BRIGHT]);

		//	if (nErrorCode != E_ERROR_CODE_TRUE || nErrorCode2 != E_ERROR_CODE_TRUE)
		//	{
		//		if (nErrorCode2 != E_ERROR_CODE_TRUE)
		//			nErrorCode = nErrorCode2;
		//	}
		//}

	}
	break;
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
				//根据2022.04.13客户要求,G3(G128)模式也可以进行与G64相同的Mura检查
			//检测出100分
				if ((int)dPara[E_PARA_AVI_MURA_G3_MURA_INSP] > 0)
					nErrorCode = LogicStart_SPOT(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);
			}
			break;

			case 1:
			{
				bool bFlag = true;
				bool bOnOff = ((int)dPara[E_PARA_AVI_MURA_G3_CM2_TEXT] > 0 ? true : false);

				nErrorCode = LogicStart_MuraG3CM(matSrcBuffer, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, &bFlag);

				// 2022.02.28
				if (bFlag && nErrorCode == E_ERROR_CODE_TRUE && bOnOff)
					nErrorCode = LogicStart_MuraG3CM2(matSrcBuffer, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
			}
			break;

			case 2:
			{
				if ((int)dPara[E_PARA_AVI_MURA_G3_CM3_TEXT] > 0)
				{
					cv::Mat matDarkdst = cv::Mat::zeros(matDstImage[E_DEFECT_COLOR_DARK].size(), CV_8UC1);
					cv::Mat matBrightdst = cv::Mat::zeros(matDstImage[E_DEFECT_COLOR_BRIGHT].size(), CV_8UC1);
					nErrorCode = LogicStart_MuraG3CM3(matSrcBuffer, matBKBuffer, matDarkdst, matBrightdst, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

					cv::add(matDstImage[E_DEFECT_COLOR_DARK], matDarkdst, matDstImage[E_DEFECT_COLOR_DARK]);
					cv::add(matDstImage[E_DEFECT_COLOR_BRIGHT], matBrightdst, matDstImage[E_DEFECT_COLOR_BRIGHT]);
				}
			}
			break;

			case 3:
			{
				if ((int)dPara[E_PARA_AVI_MURA_G3_CM4_TEXT] > 0)
				{
					//类型3重写
					cv::Mat matDarkdst = cv::Mat::zeros(matDstImage[E_DEFECT_COLOR_DARK].size(), CV_8UC1);
					nErrorCode = LogicStart_MuraG3CM4(matSrcBuffer, matBKBuffer, matDarkdst, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
					cv::add(matDstImage[E_DEFECT_COLOR_DARK], matDarkdst, matDstImage[E_DEFECT_COLOR_DARK]);
				}
			}
			break;
			}
		}

	}
	break;
	case E_IMAGE_CLASSIFY_AVI_DUSTDOWN:
	{
		//使用Point检查的结果
	}
	break;
	//如果画面号码输入错误。
	default:
		return E_ERROR_CODE_TRUE;
	}

	//如果不是空画面
	if (!matDstImage[E_DEFECT_COLOR_BRIGHT].empty() &&
		!matDstImage[E_DEFECT_COLOR_DARK].empty()	/*&&
		nAlgImgNum != E_IMAGE_CLASSIFY_AVI_GRAY_128*/)
	{
		//移除点亮区域以外的检测(移除倒圆角区域)
		if (!matBKBuffer.empty())
		{
			cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBKBuffer, matBrightBuffer);		//内存分配
			cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDarkBuffer);		//内存分配

			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Copy CV Sub Result."));
		}
		//转交结果
		else
		{
			// 			matBrightBuffer	= matDstImage[E_DEFECT_COLOR_BRIGHT].clone();		//内存分配
			// 			matDarkBuffer	= matDstImage[E_DEFECT_COLOR_DARK].clone();			//内存分配

			matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDarkBuffer);
			matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matBrightBuffer);

			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Copy Clone Result."));
		}

		//取消分配
		matDstImage[E_DEFECT_COLOR_BRIGHT].release();
		matDstImage[E_DEFECT_COLOR_DARK].release();
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return nErrorCode;
}

//删除Dust后,转交结果向量
long CInspectMura::GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL。
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())							return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuffer[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_DARK].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	//使用参数
	bool	bFlagW = (dPara[E_PARA_AVI_MURA_DUST_BRIGHT_FLAG] > 0) ? true : false;
	bool	bFlagD = (dPara[E_PARA_AVI_MURA_DUST_DARK_FLAG] > 0) ? true : false;
	int		nSize = (int)dPara[E_PARA_AVI_MURA_DUST_BIG_AREA];
	int		nRange = (int)dPara[E_PARA_AVI_MURA_DUST_ADJUST_RANGE];
	int     nWhiteMura_Judge_Flag = (int)dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_FLAG];
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
	if (nImageNum == E_IMAGE_CLASSIFY_AVI_DUST)
		return E_ERROR_CODE_TRUE;

	//范围设置(已检查,已检查异常处理)
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);
	cv::Mat matDustTemp = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type(), false);

	//设置临时缓冲区
	cv::Mat matResBuf = cMatBufTemp.GetMat(matDstBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDstBuffer[E_DEFECT_COLOR_BRIGHT].type(), false);

	int nModePS = 1;

	//如果有Dust画面,则直接运行除尘逻辑
	if (!matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty() &&
		!matDustBuffer[E_DEFECT_COLOR_DARK].empty())
	{
		//确认画面比例是否相同		
		if (matSrcBuffer.rows == matDustBuffer[E_DEFECT_COLOR_DARK].rows &&
			matSrcBuffer.cols == matDustBuffer[E_DEFECT_COLOR_DARK].cols)
			nModePS = 1;
		else
			nModePS = 2;

		//只有明亮的灰尘画面
//matDustBuffer[E_DEFECT_COLOR_BRIGHT].copyTo(matDustTemp);

		//亮&暗灰尘画面
		cv::add(matDustBuffer[E_DEFECT_COLOR_BRIGHT], matDustBuffer[E_DEFECT_COLOR_DARK], matDustTemp);

		//只留下大灰尘...
		nErrorCode = DeleteArea1(matDustTemp, nSize, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteArea."));
	}
	//没有Dust画面缓冲区
	//没有清除逻辑的不良提取
	else
	{
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Empty Dust Image."));
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_ResThreshold.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_ResThreshold.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Dust.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDustTemp);
	}

	//错误判定&发送结果
	{
		cv::Rect rectBlobROI;
		rectBlobROI.x = rectROI.left;
		rectBlobROI.y = rectROI.top;
		rectBlobROI.width = rectROI.Width();
		rectBlobROI.height = rectROI.Height();
		//标签
		CFeatureExtraction cFeatureExtraction;
		cFeatureExtraction.SetMem(&cMatBufTemp);
		cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_MURA, m_tInitTime, m_tBeforeTime, m_strAlgLog);

		//E_DEFECT_COLOR_DARK结果
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstBuffer[E_DEFECT_COLOR_DARK](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("DM_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DoDefectBlobJudgment-Dark."));

		//如果Dust面积较大,则删除
		if (bFlagD)
		{
			nErrorCode = DeleteCompareDust(matDustTemp, nRange, pResultBlob, 0, nModePS);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;
			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteCompareDust-Dark."));
		}

		//Nugi重新分类
		JudgeNugi(matSrcBuffer, matDstBuffer[E_DEFECT_COLOR_DARK], rectROI, dPara, nCommonPara, strAlgPath, pResultBlob, &cMatBufTemp);
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("JudgeNugi."));

		//Dark错误计数
		int nStartIndex = pResultBlob->nDefectCount;

		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

		//////////////////////////////////////////////////////////////////////////
// 
// 		BOOL bUse[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];
// 
//		//如果Spot检查
// 		if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse || EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse) //04.16 choi
// 		{
//			//为了一百分而二进制(一百分:255/百村200)
// 			cv::threshold(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matResBuf, 220, 255.0, CV_THRESH_BINARY);
// 
//			//复制区分参数
// 			for (int p = 0; p < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; p++)
// 			{
// 				bUse[p] = EngineerDefectJudgment[p].bDefectItemUse;
// 
//			//禁用所有参数检查
// 				EngineerDefectJudgment[p].bDefectItemUse = false;
// 			}
// 
//			//只设置Spot检查
// 			EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse = true;
// 			EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse = true; //choi 04.16
// 
//			//E_DEFECT_COLOR_BRIGHT结果
// 			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matResBuf(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
// 				nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BM_"), EngineerBlockDefectJudge, pResultBlob);
// 			if (nErrorCode != E_ERROR_CODE_TRUE)
// 			{
//			//禁用内存
// 				matSrcBuffer.release();
// 				matDstBuffer[E_DEFECT_COLOR_DARK].release();
// 				matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();
// 
// 				return nErrorCode;
// 			}
// 			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DoDefectBlobJudgment-Bright Spot."));
// 
		//重新分类White Spot
// 			JudgeWhiteSpot(matSrcBuffer, matDstBuffer[E_DEFECT_COLOR_BRIGHT], rectROI, dPara, nCommonPara, strAlgPath, pResultBlob, &cMatBufTemp);
// 			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("JudgeWhiteSpot."));
// 
//			//如果您使用的是外围信息,Judgement()会保存文件(重复数据删除时,相应的坏外围方案图)
//			//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			// 				cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);
			// 
			//			//绘制结果轮廓
			// 			cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
			// 
			//			//元福
			// 			for (int p = 0; p < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; p++)
			// 			{
			// 				EngineerDefectJudgment[p].bDefectItemUse = bUse[p];
			// 			}
			// 		}

					//////////////////////////////////////////////////////////////////////////

							//为了白村,李振华(白点:255/白村200)
			// 		cv::threshold(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matResBuf, 190, 255.0, CV_THRESH_BINARY);
			// 		
			//		//只禁用Spot检查
			// 		bUse[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT] = EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse;
			// 		bUse[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT] = EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse; //04.16 choi
			// 
			// 		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse = false;
			// 		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse = false;									//04.16 choi

							//E_DEFECT_COLOR_BRIGHT结果
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstBuffer[E_DEFECT_COLOR_BRIGHT](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BM_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DoDefectBlobJudgment-Bright."));

		//元福
// 		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse = bUse[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT];
// 		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse = bUse[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT]; //04.16 choi

				//如果Dust面积较大,则删除
		if (bFlagW)
		{
			//nStartIndex:Dark错误计数后开始
			nErrorCode = DeleteCompareDust(matDustTemp, nRange, pResultBlob, nStartIndex, nModePS);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;
			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteCompareDust-Bright."));
		}

		//White Mura重新分类
		if (nWhiteMura_Judge_Flag > 0) {
			JudgeWhiteMura(matSrcBuffer, matDstBuffer[E_DEFECT_COLOR_BRIGHT], rectROI, dPara, nCommonPara, strAlgPath, pResultBlob, &cMatBufTemp);
			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("JudgeWhiteMura."));
		}
		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
	}

	//禁用内存
	matSrcBuffer.release();
	matDstBuffer[E_DEFECT_COLOR_DARK].release();
	matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}


long CInspectMura::LogicStart_SPOT(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//使用参数
	int		nGauSize = (int)dPara[E_PARA_AVI_MURA_COMMON_GAUSSIAN_SIZE];
	double	dGauSig = dPara[E_PARA_AVI_MURA_COMMON_GAUSSIAN_SIGMA];

	//使用Inspect Flag
	int     nBright_Inpect_Flag = (int)dPara[E_PARA_AVI_MURA_GRAY_INSPECT_BRIGHT_FLAG];
	int     nDark_Inpect_Flag = (int)dPara[E_PARA_AVI_MURA_GRAY_INSPECT_DARK_FLAG];
	int     nBright_inspect_new_Flag = (int)dPara[E_PARA_AVI_MURA_GRAY_INSPECT_MID_BRIGHT_FLAG];

	int		nEdge_Area = (int)dPara[E_PARA_AVI_MURA_GRAY_BRIGHT_THRESHOLD_WHITE_MURA_EDGE_AREA];
	int		nBrightTh1 = (int)dPara[E_PARA_AVI_MURA_GRAY_BRIGHT_THRESHOLD_WHITE_MURA_ACTIVE];
	int		nBrightTh2 = (int)dPara[E_PARA_AVI_MURA_GRAY_BRIGHT_THRESHOLD_WHITE_MURA_EDGE];
	int		nBrightMorp = (int)dPara[E_PARA_AVI_MURA_GRAY_BRIGHT_MORP];
	int		nBrightDelArea = (int)dPara[E_PARA_AVI_MURA_GRAY_BRIGHT_DEL_AREA];

	int		nDarkTh = (int)dPara[E_PARA_AVI_MURA_GRAY_DARK_THRESHOLD];
	
	int		nDarkMorp = (int)dPara[E_PARA_AVI_MURA_GRAY_DARK_MORP];
	int		nDarkDelArea = (int)dPara[E_PARA_AVI_MURA_GRAY_DARK_DEL_AREA];

	//异常处理
	if (nBrightMorp % 2 == 0)		nBrightMorp++;
	if (nDarkMorp % 2 == 0)		nDarkMorp++;
	//if (nBrightTh1 < nBrightTh2)	nBrightTh2 = nBrightTh1;

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

	//////////////////////////////////////////////////////////////////////////
		//用于Mid Biright
	int nAdjust1_Min_GV = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_ADJUST1_MIN_GV]; //20
	int nResize_Loop_Cnt = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_RESIZE_LOOP_CNT]; //5

	double dContrast_Value = (double)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_CONTRAST_VALUE]; //0.01
	int	   nAdjust2_Muti_Value = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_ADJUST2_MUTI_VALUE]; //3

	int    nCanny_th_min = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_CANNY_MIN]; //120
	int    nCanny_th_max = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_CANNY_MAX]; //255

	int    nEdge_Del_Bk_loop_cnt = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_EDGE_DEL_LOOP]; //10
	int    nEdge_Del_Bk_morp_size = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_EDGE_DEL_MORP_SIZE]; //21

	int    nMid_Bright_Del_Area = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_DEL_AREA]; //100

	int    nDefect_Morp_resize = (int)dPara[E_PARA_AVI_MURA_GRAY_MID_BRIGHT_DEFECT_MORP_RESIZE]; //80
	//////////////////////////////////////////////////////////////////////////

		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	//检查区域
	CRect rectTemp(rectROI);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat matSrcROIBuf_tmp = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), true);		 //用于Mid_Bright
	cv::Mat matSrcROIBuf_result = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);	 //用于Mid_Bright

	cv::Mat matSrcROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matSrcImage.type(), false);
	cv::Mat matBrROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matDstImage[E_DEFECT_COLOR_BRIGHT].type(), false);
	cv::Mat matDaROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matDstImage[E_DEFECT_COLOR_DARK].type(), false);

	matSrcROIBuf = matSrcImage(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	matBrROIBuf = matDstImage[E_DEFECT_COLOR_BRIGHT](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	matDaROIBuf = matDstImage[E_DEFECT_COLOR_DARK](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	//////////////////////////////////////////////////////////////////////////

		//背景区域
	cv::Mat matBkROIBuf;
	if (!matBKBuffer.empty())
		matBkROIBuf = matBKBuffer(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Src.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	///////////////////////////////////////////////////////////////////////////*/
	// Blur
	cv::Mat matGauBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::GaussianBlur(matSrcROIBuf, matGauBuf, cv::Size(nGauSize, nGauSize), dGauSig, dGauSig);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Gau.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("GaussianBlur."));

	//背景画面
	cv::Mat matBGBuf = cMatBufTemp.GetMat(matGauBuf.size(), matGauBuf.type(), false);
	nErrorCode = Estimation_XY(matGauBuf, matBGBuf, dPara, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Esti.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBGBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Estimation_XY."));

	//缓冲区分配
	cv::Mat matSubBrBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::Mat matSubDaBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);

	cv::Mat matThBrBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::Mat matThBrBuf_Act = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::Mat matThBrBuf_Edge = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);

	cv::Mat matThDaBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::Mat matTempBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);

#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int i = 0; i < 3; i++)
	{
		switch (i)
		{
		case 0:
		{
			//////////////////////////////////////////////////////////////////////////
						//查找亮不良
			//////////////////////////////////////////////////////////////////////////
						//减号
			if (nBright_Inpect_Flag > 0) {
				cv::subtract(matGauBuf, matBGBuf, matSubBrBuf);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Sub.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matSubBrBuf);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("subtract - Bright."));

				//////////////////////////////////////////////////////////////////////////choi 06.04

				cv::Rect R_Mask(0 + nEdge_Area, 0 + nEdge_Area, matSubBrBuf.cols - (nEdge_Area * 2), matSubBrBuf.rows - (nEdge_Area * 2));

				cv::Mat matActive_Mask = cMatBufTemp.GetMat(matSubBrBuf.size(), matSubBrBuf.type(), false);
				cv::Mat matEdge_Mask = cMatBufTemp.GetMat(matSubBrBuf.size(), matSubBrBuf.type(), false);

				//添加2022.02.15 SetTo(0)
				//在原本可以不用的动作或设备中,Active Mask除了Active区域外还会出现很多杂音,因此添加到测试用途
				matActive_Mask.setTo(0);
				matActive_Mask(R_Mask).setTo(255);
				cv::bitwise_not(matActive_Mask, matEdge_Mask);

				//添加2022.02.15 SetTo(0)
				//在原本可以不用的动作或设备中,Active Mask除了Active区域外还会出现很多杂音,因此添加到测试用途
				//添加图像存储以查找上述原因
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Active_Mask.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matActive_Mask);

					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_matEdge_Mask.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matEdge_Mask);
				}

				cv::Mat matActive_Br_Temp = cMatBufTemp.GetMat(matSubBrBuf.size(), matSubBrBuf.type(), false);
				cv::Mat matEdge_Br_Temp = cMatBufTemp.GetMat(matSubBrBuf.size(), matSubBrBuf.type(), false);

				cv::bitwise_and(matSubBrBuf, matActive_Mask, matActive_Br_Temp);
				cv::bitwise_and(matSubBrBuf, matEdge_Mask, matEdge_Br_Temp);

				cv::threshold(matActive_Br_Temp, matThBrBuf_Act, nBrightTh1, 255.0, CV_THRESH_BINARY);
				cv::threshold(matEdge_Br_Temp, matThBrBuf_Edge, nBrightTh2, 255.0, CV_THRESH_BINARY);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Th_Active.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matThBrBuf_Act);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("threshold - Bright."));

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Th_Edge.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matThBrBuf_Edge);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("threshold - Bright."));

				cv::bitwise_or(matThBrBuf_Act, matThBrBuf_Edge, matThBrBuf);

				matActive_Mask.release();
				matEdge_Mask.release();
				matActive_Br_Temp.release();
				matEdge_Br_Temp.release();
				matThBrBuf_Act.release();
				matThBrBuf_Edge.release();
				//////////////////////////////////////////////////////////////////////////

							//二进制
	// 			cv::threshold(matSubBrBuf, matThBrBuf, nBrightTh1, 255.0, CV_THRESH_BINARY);
	// 			if (bImageSave)
	// 			{
	// 				CString strTemp;
	// 				strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Th.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
	// 				ImageSave(strTemp, matThBrBuf);
	// 			}
	// 
	// 			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("threshold - Bright."));

							//粘贴不良内容
				if (nBrightMorp > 1)
					cv::morphologyEx(matThBrBuf, matBrROIBuf, MORPH_CLOSE, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(nBrightMorp, nBrightMorp)));
				else
					matThBrBuf.copyTo(matBrROIBuf);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Morp.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matBrROIBuf);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("morphologyEx - Bright."));

				//除了背景只看里面的...
				//如果Edge页存在不良内容,则保留行
				if (!matBkROIBuf.empty())
					cv::subtract(matBrROIBuf, matBkROIBuf, matBrROIBuf);
			}
			else {
				matBrROIBuf.setTo(0);
				break;
			}
			//白村用二进制(白点:255/白村200)
//cv::threshold(matSubBrBuf, matTempBuf, nBrightTh2, 200.0, CV_THRESH_BINARY);

		}
		break;
		case 1:
		{
			//////////////////////////////////////////////////////////////////////////
						//寻找暗不良
			//////////////////////////////////////////////////////////////////////////
						//减号
			if (nDark_Inpect_Flag > 0) {
				cv::subtract(matBGBuf, matGauBuf, matSubDaBuf);
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_Sub.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matSubDaBuf);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("subtract - Dark."));

				//二进制
				cv::threshold(matSubDaBuf, matThDaBuf, nDarkTh, 255.0, CV_THRESH_BINARY);
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_Th.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matThDaBuf);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("threshold - Dark."));

				//湿气不良...
				if (nDarkMorp > 1)
					cv::morphologyEx(matThDaBuf, matDaROIBuf, MORPH_OPEN, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(nDarkMorp, nDarkMorp)));
				else
					matThDaBuf.copyTo(matDaROIBuf);
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_Morp.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matDaROIBuf);
				}

				writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("morphologyEx - Dark."));
			}
			else {
				matDaROIBuf.setTo(0);
				break;
			}

		}
		break;
		case 2:
		{
			//////////////////////////////////////////////////////////////////////////
						//寻找中等大小的亮不良
			//////////////////////////////////////////////////////////////////////////
						//减号
			if (nBright_inspect_new_Flag > 0) {
				/////////////////////////////////////////////////////////////////////////// 07.02 Test

				cv::Mat matResize = cMatBufTemp.GetMat(matSrcROIBuf_tmp.size(), matSrcROIBuf_tmp.type());
				matSrcImage.copyTo(matResize);

				cv::Mat matTmp = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matSrcImage.type(), false);
				matResize(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height())).copyTo(matTmp);

				//////////////////////////////////////////////////////////////////////////
							//提高到暗部平均值
				//////////////////////////////////////////////////////////////////////////
				cv::Scalar m, s;
				cv::meanStdDev(matTmp, m, s);

				for (int i = 0; i < matTmp.cols * matTmp.rows; i++) {
					if (matTmp.data[i] < nAdjust1_Min_GV) {
						matTmp.data[i] = m[0];
					}
				}

				matTmp.copyTo(matResize(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height())));

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Adjust1.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matResize);
				}

				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//LISA
				//////////////////////////////////////////////////////////////////////////
				for (int i = 0; i < nResize_Loop_Cnt; i++) {
					cv::resize(matResize, matResize, matResize.size() / 2, 3, 3, INTER_AREA);
				}

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_resize.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matResize);
				}

				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//对比
				//////////////////////////////////////////////////////////////////////////
				int var_brightness1 = 0;
				double var_contrast1 = dContrast_Value;
				double c, d;
				if (var_contrast1 > 0)
				{
					double delta1 = 127.0 * var_contrast1;
					c = 255.0 / (255.0 - delta1 * 2);
					d = c * (var_brightness1 - delta1);
				}
				else
				{
					double delta1 = -128.0 * var_contrast1;
					c = (256.0 - delta1 * 2) / 255.0;
					d = c * var_brightness1 + delta1;
				}

				cv::Mat temp1 = cMatBufTemp.GetMat(matResize.size(), matResize.type());
				cv::Mat dst1 = cMatBufTemp.GetMat(matResize.size(), matResize.type());
				matResize.copyTo(temp1);

				temp1.convertTo(dst1, CV_8U, c, d);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_contrast.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, dst1);
				}
				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//亮度校正
				//////////////////////////////////////////////////////////////////////////
				for (int i = 0; i < dst1.cols * dst1.rows; i++)
				{

					if (dst1.data[i] * nAdjust2_Muti_Value >= 255)
					{
						dst1.data[i] = 255;
					}
					else
						dst1.data[i] = dst1.data[i] * nAdjust2_Muti_Value;
				}

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Adjust2.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, dst1);
				}
				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//墨西哥过滤器
				//////////////////////////////////////////////////////////////////////////
				AlgoBase::C_Mexican_filter(dst1, sz);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Mexican.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, dst1);
				}

				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//卡尼Edge
				//////////////////////////////////////////////////////////////////////////
				cv::Canny(dst1, dst1, nCanny_th_min, nCanny_th_max, 3, false);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Canny.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, dst1);
				}

				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//尺码原装
				//////////////////////////////////////////////////////////////////////////

				cv::resize(dst1, dst1, matSrcROIBuf_result.size(), 3, 3, INTER_AREA);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Canny_inv_resize.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, dst1);
				}
				//////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////
							//清除侧面和剑
				//////////////////////////////////////////////////////////////////////////
				cv::Mat matBkTmp = cMatBufTemp.GetMat(matBKBuffer.size(), matBKBuffer.type(), false);
				matBKBuffer.copyTo(matBkTmp);

				for (int i = 0; i < nEdge_Del_Bk_loop_cnt; i++) {
					cv::morphologyEx(matBkTmp, matBkTmp, MORPH_DILATE, cv::getStructuringElement(MORPH_RECT, cv::Size(nEdge_Del_Bk_morp_size, nEdge_Del_Bk_morp_size)));
				}

				cv::subtract(dst1, matBkTmp, dst1);

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Result.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
					ImageSave(strTemp, dst1);
				}
				//////////////////////////////////////////////////////////////////////////

				dst1.copyTo(matSrcROIBuf_result);
			}
			else {
				matSrcROIBuf_tmp.setTo(0);
				break;
			}
		}

		break;
		}
	}
	//////////////////////////////////////////////////////////////////////////
		//查找亮不良
	//////////////////////////////////////////////////////////////////////////
	if (nBright_Inpect_Flag > 0)
	{
		//删除小面积&接线
		nErrorCode = DeleteArea3(matBrROIBuf, nBrightDelArea, 10, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteArea - Bright."));

		//		//删除小面积
		// 		nErrorCode = DeleteArea1(matTempBuf, nBrightDelArea, &cMatBufTemp);
		// 		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;
		// 
		//		//整合不良
		// 		cv::add(matBrROIBuf, matTempBuf, matBrROIBuf);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Del.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matBrROIBuf);
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Th Add - Bright."));
	}

	//////////////////////////////////////////////////////////////////////////
		//寻找暗不良
	//////////////////////////////////////////////////////////////////////////
	if (nDark_Inpect_Flag > 0)
	{
		//删除小面积
		nErrorCode = DeleteArea1(matDaROIBuf, nDarkDelArea, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_Del.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDaROIBuf);
		}

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteArea - Dark."));
	}

	//////////////////////////////////////////////////////////////////////////
		//寻找中间亮的不良
	//////////////////////////////////////////////////////////////////////////
	if (nBright_inspect_new_Flag > 0) {

		nErrorCode = DeleteArea3(matSrcROIBuf_result, nMid_Bright_Del_Area, 10, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteArea - Bright."));

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Result_Delete.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matSrcROIBuf_result);
		}

		vector<vector<cv::Point>>	contours;
		vector<vector<cv::Point>>().swap(contours);
		vector<cv::Point > hierarchy;
		vector<cv::Point >().swap(hierarchy);

		cv::findContours(matSrcROIBuf_result, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

		// 	for (int m = 0; m < contours.size(); m++)
		// 	{
		// 		for (int k = 0; k < contours.at(m).size(); k++)
		// 			ptContours.push_back(contours[m][k]);
		// 	}

		// 	// convexHull
		// 	for (int i = 0; i < contours.size(); i++) {
		// 		cv::convexHull(contours[i], ptConvexHull[i]);
		// 		cv::fillConvexPoly(matDstImage[E_DEFECT_COLOR_BRIGHT], ptConvexHull[i], cv::Scalar(255, 255, 255));
		// 	}
		// 	vector<vector<cv::Point>>().swap(contours);
		// 	vector<vector<cv::Point>>().swap(ptConvexHull);

		for (int k = 0; k < contours.size(); k++)
		{
			drawContours(matSrcROIBuf_tmp, contours, k, Scalar(255), CV_FILLED, 8, hierarchy);
		}

		vector<vector<cv::Point>>().swap(contours);
		vector<cv::Point >().swap(hierarchy);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Result_Contours.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matSrcROIBuf_tmp);
		}

		cv::resize(matSrcROIBuf_tmp, matSrcROIBuf_tmp, matSrcImage.size() / nDefect_Morp_resize, 3, 3, INTER_AREA);
		cv::resize(matSrcROIBuf_tmp, matSrcROIBuf_tmp, matSrcImage.size(), 3, 3, INTER_AREA);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Result_Morp.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matSrcROIBuf_tmp);
		}

		cv::bitwise_or(matDstImage[E_DEFECT_COLOR_BRIGHT], matSrcROIBuf_tmp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}
	return nErrorCode;
}

long CInspectMura::LogicStart_RGB_LINE_MURA(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//使用参数
	int		nResize = (int)dPara[E_PARA_AVI_MURA_RGB_RESIZE];
	int		nGauSize = (int)dPara[E_PARA_AVI_MURA_RGB_GAUSSIAN_SIZE];
	double	dGauSig = dPara[E_PARA_AVI_MURA_RGB_GAUSSIAN_SIGMA];
	int		nEMD_Edge_Area = (int)dPara[E_PARA_AVI_MURA_RGB_DARK_EDGE_AREA];
	int		nEdge_DarkTh = (int)dPara[E_PARA_AVI_MURA_RGB_DARK_EDGE_THRESHOLD];
	int		nAct_DarkTh = (int)dPara[E_PARA_AVI_MURA_RGB_DARK_ACTIVE_THRESHOLD];
	int		nDarkMorp = (int)dPara[E_PARA_AVI_MURA_RGB_DARK_MORP];
	float	fPow = (float)dPara[E_PARA_AVI_MURA_RGB_POW];
	int		nBlurX = (int)dPara[E_PARA_AVI_MURA_RGB_BLUR_X];
	int		nBlurY = (int)dPara[E_PARA_AVI_MURA_RGB_BLUR_Y1];
	int		nBlurY2 = (int)dPara[E_PARA_AVI_MURA_RGB_BLUR_Y2];
	int		nEdgeArea = (int)dPara[E_PARA_AVI_MURA_RGB_EDGE_AREA];
	int		nActiveTh = (int)dPara[E_PARA_AVI_MURA_RGB_THRESHOLD];
	int		nEdgeTh = (int)dPara[E_PARA_AVI_MURA_RGB_EDGE_THRESHOLD];
	int		nCut = (int)dPara[E_PARA_AVI_MURA_RGB_INSIDE];

	if (nResize <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;

	int nNugi_Inspect_Flag = (int)dPara[E_PARA_AVI_MURA_RGB_INSPECT_DARK_FLAG];
	int nMuraLine_Inspect_Flag = (int)dPara[E_PARA_AVI_MURA_RGB_INSPECT_LINE_MURA_FLAG];

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

	//////////////////////////////////////////////////////////////////////////
	int nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	//原始
	cv::Mat matSrcROIBuf = matSrcImage(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	//cv::Mat matBrROIBuf		= matDstImage[E_DEFECT_COLOR_BRIGHT](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	cv::Mat matDaROIBuf = matDstImage[E_DEFECT_COLOR_DARK](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Src.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Src."));

	/************************************************************************
		基本预处理
	************************************************************************/

	//背景区域
	cv::Mat matBkROIBuf;
	if (!matBKBuffer.empty())
		matBkROIBuf = matBKBuffer(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	//////////////////////////////////////////////////////////////////////////
		//噪声消除1
	cv::Mat matBlurBuf8 = cMatBufTemp.GetMat(matSrcROIBuf.size(), CV_8UC1, false);
	cv::blur(matSrcROIBuf, matBlurBuf8, cv::Size(5, 5));
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("blur 1."));

	//////////////////////////////////////////////////////////////////////////
	// Resize
	cv::Mat matReBuf8 = cMatBufTemp.GetMat(matBlurBuf8.size() / nResize, matBlurBuf8.type(), false);
	cv::resize(matBlurBuf8, matReBuf8, matReBuf8.size(), INTER_AREA);
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Re.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matReBuf8);
	}
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Resize."));

	//////////////////////////////////////////////////////////////////////////
		//16bit&放大
	cv::Mat matGauBuf16 = cMatBufTemp.GetMat(matReBuf8.size(), CV_16UC1, false);
	cv::Mat matPowBuf16 = cMatBufTemp.GetMat(matReBuf8.size(), CV_16UC1, false);
	matReBuf8.convertTo(matGauBuf16, CV_16UC1);

	if (fPow > 10.0)
	{
		//自动调整亮度
		float fMean = cv::mean(matGauBuf16)[0];

		float fAutoPow = log(fPow * 16.0) / log(fMean);

		nErrorCode = AlgoBase::Pow(matGauBuf16, matPowBuf16, fAutoPow, 4095, &cMatBufTemp);
	}
	else
	{
		//异常处理
		if (fPow <= 0)	fPow = 1.0f;

		//作为设置值应用
		nErrorCode = AlgoBase::Pow(matGauBuf16, matPowBuf16, fPow, 4095, &cMatBufTemp);
	}
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_P.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matPowBuf16);
	}
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("P16."));

	//////////////////////////////////////////////////////////////////////////
	// Gaussian or Blur
	if (false)
	{
		cv::GaussianBlur(matPowBuf16, matGauBuf16, cv::Size(nGauSize, nGauSize), dGauSig);
	}
	else
	{
		//缓冲区分配和初始化
		CMatBuf cMatBufTemp2;
		cMatBufTemp2.SetMem(&cMatBufTemp);

		// Blur
		cv::Mat matTempBuf16 = cMatBufTemp2.GetMat(matPowBuf16.size(), matPowBuf16.type(), false);

		// Avg
		cv::blur(matPowBuf16, matGauBuf16, cv::Size(nGauSize, nGauSize));
		for (int i = 1; i < dGauSig; i++)
		{
			cv::blur(matGauBuf16, matTempBuf16, cv::Size(nGauSize, nGauSize));

			matTempBuf16.copyTo(matGauBuf16);
		}

		matTempBuf16.release();

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp2.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp2.Get_AutoMemory());
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Gau.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauBuf16);
	}
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("GaussianBlur."));

	////////////////////////////////////////////////////////////////////////// common
	cv::Mat matThBuf = cMatBufTemp.GetMat(matGauBuf16.size(), CV_8UC1, true);
	cv::Mat matMorpBuf = cMatBufTemp.GetMat(matThBuf.size(), matThBuf.type(), true);

	/************************************************************************
		暗不良检测ex)漏气
	************************************************************************/
	if (nNugi_Inspect_Flag > 0) {
		//////////////////////////////////////////////////////////////////////////
				//16bit二进制
		//cv::Mat matThBuf = cMatBufTemp.GetMat(matGauBuf16.size(), CV_8UC1, false);
	//////////////////////////////////////////////////////////////////////////
		if (nEMD_Edge_Area >= (matGauBuf16.rows / 2) || nEMD_Edge_Area >= (matGauBuf16.cols / 2)) nEMD_Edge_Area = 0;

		if (nEMD_Edge_Area > 0) {

			//////////////////////////////////////////////////////////////////////////
			cv::Mat matGauBuf16_Edge = cMatBufTemp.GetMat(matReBuf8.size(), CV_16UC1, false);
			//////////////////////////////////////////////////////////////////////////

			cv::Rect rtActiveArea;
			rtActiveArea.x = 0 + nEMD_Edge_Area;
			rtActiveArea.y = 0 + nEMD_Edge_Area;
			rtActiveArea.width = matGauBuf16.cols - (nEMD_Edge_Area * 2);
			rtActiveArea.height = matGauBuf16.rows - (nEMD_Edge_Area * 2);

			matGauBuf16.copyTo(matGauBuf16_Edge);
			matGauBuf16_Edge(rtActiveArea).setTo(4096);

			nErrorCode = AlgoBase::Threshold16_INV(matGauBuf16_Edge, matThBuf, nEdge_DarkTh, 255);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

			nErrorCode = AlgoBase::Threshold16_INV(matGauBuf16(rtActiveArea), matThBuf(rtActiveArea), nAct_DarkTh, 255);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		}
		else {

			nErrorCode = AlgoBase::Threshold16_INV(matGauBuf16, matThBuf, nAct_DarkTh, 255);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		}
		//////////////////////////////////////////////////////////////////////////

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_Th.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matThBuf);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Dark Th."));

		//////////////////////////////////////////////////////////////////////////
			//将Dark Defect放大(目的是将其消除到漏气的周围)

		cv::morphologyEx(matThBuf, matMorpBuf, MORPH_DILATE, cv::getStructuringElement(MORPH_RECT, cv::Size(nDarkMorp, nDarkMorp)));
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_DarkM.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matMorpBuf);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DarkM."));

		//////////////////////////////////////////////////////////////////////////
			//用于Nugi检测结果(恢复到原始画面大小)
		cv::resize(matMorpBuf, matDaROIBuf, matDaROIBuf.size(), INTER_AREA);
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Nugi.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDaROIBuf);
		}

		//////////////////////////////////////////////////////////////////////////
			//清除暗线不良(只留下漏水)
		nErrorCode = DeleteDarkLine(matDaROIBuf, 1.15f, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Nugi_Del.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDaROIBuf);
		}
	}
	else {
		matDaROIBuf.setTo(0);
	}
	/************************************************************************
		RGB线路故障检测
	************************************************************************/
	if (nMuraLine_Inspect_Flag > 0) {
		//检查E_DEFECT_JUDGEMENT_MURA_LINE_X时,RGB行检测算法的行为
		if (EngineerBlockDefectJudge == NULL)
			return E_ERROR_CODE_TRUE;
		//以第一个分区判定项决定 hjf
		if (!EngineerBlockDefectJudge[0].stDefectItem[E_DEFECT_JUDGEMENT_MURA_LINE_X].bDefectItemUse)
			return E_ERROR_CODE_TRUE;



		//////////////////////////////////////////////////////////////////////////
			//横向Max GV限制:防止在明点等明亮的不良环境中检测到时发生
		nErrorCode = LimitMaxGV16X(matGauBuf16, 1.15f);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_MaxGV.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matGauBuf16);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("P16 MaxGV."));

		//////////////////////////////////////////////////////////////////////////
			//仅横向Blur(RGB不良时,仅横向存在不良)
		cv::Mat matBlurBuf16 = cMatBufTemp.GetMat(matGauBuf16.size(), matGauBuf16.type(), false);
		cv::blur(matGauBuf16, matBlurBuf16, cv::Size(nBlurX, nBlurY));

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Blur.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matBlurBuf16);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Blur."));

		//////////////////////////////////////////////////////////////////////////
			//创建背景(通过垂直Blur创建背景)
		cv::Mat matBKBuf16 = cMatBufTemp.GetMat(matBlurBuf16.size(), matBlurBuf16.type(), false);
		cv::blur(matBlurBuf16, matBKBuf16, cv::Size(1, nBlurY2));

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bk.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matBKBuf16);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("BK."));

		//////////////////////////////////////////////////////////////////////////
			//(背景噪声消除)=>二进制16bit
			//Active&Edge Spec.单独

		// Active
		cv::Rect rectActive(nEdgeArea, nEdgeArea, matBKBuf16.cols - nEdgeArea - nEdgeArea, matBKBuf16.rows - nEdgeArea - nEdgeArea);

		//全部(Edge)
		nErrorCode = AlgoBase::SubThreshold16(matBKBuf16, matBlurBuf16, matThBuf, nEdgeTh, 255);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		//初始化活动区域
		matThBuf(rectActive).setTo(0);

		//活动区域进化
		nErrorCode = AlgoBase::SubThreshold16(matBKBuf16(rectActive), matBlurBuf16(rectActive), matThBuf(rectActive), nActiveTh, 255);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_SubTh.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matThBuf);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Sub Th."));

		//////////////////////////////////////////////////////////////////////////
			//清除阴暗部分,如漏气不良
		cv::Mat matResBuf = cMatBufTemp.GetMat(matThBuf.size(), matThBuf.type(), false);
		cv::subtract(matThBuf, matMorpBuf, matResBuf);
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_SubD.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matResBuf);
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Sub."));

		//////////////////////////////////////////////////////////////////////////
			//如果有背景缓冲区,则清除背景区域(清除点亮区域以外的部分)
		if (!matBkROIBuf.empty())
		{
			//调整画面大小
			cv::resize(matBkROIBuf, matMorpBuf, matSrcROIBuf.size() / nResize, INTER_AREA);

			//背景区域设置大一点
			int nM = nCut;
			cv::morphologyEx(matMorpBuf, matThBuf, MORPH_DILATE, cv::getStructuringElement(MORPH_RECT, cv::Size(nM, nM)));

			//复制数据
			matResBuf.copyTo(matMorpBuf);

			//删除背景区域
			cv::subtract(matMorpBuf, matThBuf, matResBuf);

			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_SubD2.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matResBuf);
			}
			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Sub2."));
		}

		//////////////////////////////////////////////////////////////////////////
			//限制RGB Line Mura面积
			//限制面积,消除不良(RGB不良的情况下,没有像线一样连接,而是断开的)
		nErrorCode = LimitArea(matResBuf, dPara, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Limit Area."));

		//////////////////////////////////////////////////////////////////////////
			//查找RGB Line Mura
		if (bImageSave)
		{
			//保存Excel
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Res.csv"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);

			nErrorCode = JudgeRGBLineMuraSave(matResBuf, matBKBuf16, dPara, nCommonPara, rectROI, pResultBlob, strTemp, &cMatBufTemp);

			//保存画面
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Res.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matResBuf);
		}
		else
		{
			nErrorCode = JudgeRGBLineMura(matResBuf, matBKBuf16, dPara, nCommonPara, rectROI, pResultBlob, &cMatBufTemp);
		}

		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Limit Area."));
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//保存8bit和12bit画面
long CInspectMura::ImageSave(CString strPath, cv::Mat matSrcBuf)
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
long CInspectMura::DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrcImage.size(), CV_32SC1, false);

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

//删除小面积
long CInspectMura::DeleteArea2(cv::Mat& matSrcImage, int nCount, int nLength, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrcImage.size(), CV_32SC1, false);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matSrcImage, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

	//如果没有个数,请退出
	if (nTotalLabel <= 0)	return E_ERROR_CODE_TRUE;

	//长度限制
	int nX = (int)(matSrcImage.cols * 0.7);
	int nY = (int)(matSrcImage.rows * 0.7);

	//按数量分配内存
	cv::RotatedRect* rectRo = NULL;
	rectRo = new cv::RotatedRect[nTotalLabel + 1];
	memset(rectRo, 0, sizeof(cv::RotatedRect) * (nTotalLabel + 1));

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

		if (nArea <= nCount |	//面积限制
			rectTemp.width >= nX |	//横向长度限制(线路不良...)
			rectTemp.height >= nY)	//垂直长度限制(线路不良...)
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

		//如果面积满足
		{
			//裁剪ROI
			cv::Mat matLabelROI = matLabel(rectTemp);

			//初始化
			vector<cv::Point>	ptIndexs;
			vector<cv::Point>().swap(ptIndexs);

			for (int y = 0; y < rectTemp.height; y++)
			{
				int* ptrLabel = (int*)matLabelROI.ptr(y);

				for (int x = 0; x < rectTemp.width; x++, ptrLabel++)
				{
					//对象
					if (*ptrLabel == idx)
					{
						//插入原始坐标
						ptIndexs.push_back(cv::Point(x + rectTemp.x, y + rectTemp.y));
					}
				}
			}

			//查找最小框
			rectRo[idx] = cv::minAreaRect(ptIndexs);

			//扩展
			rectRo[idx].size.width += (nLength + nLength);
			rectRo[idx].size.height += (nLength + nLength);

			//确认
//if( 0 )
//{
			//	//查找4个顶点
//	cv::Point2f vertices[E_CORNER_END], ptC;
//	rectRo[idx].points(vertices);

			//	//引用
//	cv::line(matSrcImage, vertices[0], vertices[1], cv::Scalar(100, 100, 100));
//	cv::line(matSrcImage, vertices[1], vertices[2], cv::Scalar(100, 100, 100));
//	cv::line(matSrcImage, vertices[2], vertices[3], cv::Scalar(100, 100, 100));
//	cv::line(matSrcImage, vertices[3], vertices[0], cv::Scalar(100, 100, 100));
//}
		}
	}

	//Blob计数
	for (int k = 1; k <= nTotalLabel; k++)
	{
		//如果没有值
		if (rectRo[k].size.width <= 0)	continue;

		for (int m = 1; m <= nTotalLabel; m++)
		{
			//排除相同的故障
			if (k == m)	continue;

			//如果没有值
			if (rectRo[m].size.width <= 0)	continue;

			//范围为南的情况下划线合并
			if (OrientedBoundingBox(rectRo[k], rectRo[m]))
			{
				//获取Center Point
				cv::Point ptLine[2];
				ptLine[0].x = (int)matCentroid.at<double>(k, 0);
				ptLine[0].y = (int)matCentroid.at<double>(k, 1);
				ptLine[1].x = (int)matCentroid.at<double>(m, 0);
				ptLine[1].y = (int)matCentroid.at<double>(m, 1);

				//绘制中心点线
				cv::line(matSrcImage, ptLine[0], ptLine[1], cv::Scalar(255, 255, 255));
			}
		}
	}

	//禁用内存
	delete rectRo;
	rectRo = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMura::DeleteArea3(cv::Mat& matSrcImage, int nCount, int nLength, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrcImage.size(), CV_32SC1, false);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matSrcImage, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

	//如果没有个数,请退出
	if (nTotalLabel <= 0)	return E_ERROR_CODE_TRUE;

	//长度限制
	int nX = (int)(matSrcImage.cols * 0.7);
	int nY = (int)(matSrcImage.rows * 0.7);

	//按数量分配内存
	cv::RotatedRect* rectRo = NULL;
	rectRo = new cv::RotatedRect[nTotalLabel + 1];
	memset(rectRo, 0, sizeof(cv::RotatedRect) * (nTotalLabel + 1));

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

		//如果面积满足
		{
			//裁剪ROI
			cv::Mat matLabelROI = matLabel(rectTemp);

			//初始化
			vector<cv::Point>	ptIndexs;
			vector<cv::Point>().swap(ptIndexs);

			for (int y = 0; y < rectTemp.height; y++)
			{
				int* ptrLabel = (int*)matLabelROI.ptr(y);

				for (int x = 0; x < rectTemp.width; x++, ptrLabel++)
				{
					//对象
					if (*ptrLabel == idx)
					{
						//插入原始坐标
						ptIndexs.push_back(cv::Point(x + rectTemp.x, y + rectTemp.y));
					}
				}
			}

			//查找最小框
			rectRo[idx] = cv::minAreaRect(ptIndexs);

			//扩展
			rectRo[idx].size.width += (nLength + nLength);
			rectRo[idx].size.height += (nLength + nLength);

			//确认
//if( 0 )
//{
			//	//查找4个顶点
//	cv::Point2f vertices[E_CORNER_END], ptC;
//	rectRo[idx].points(vertices);

			//	//引用
//	cv::line(matSrcImage, vertices[0], vertices[1], cv::Scalar(100, 100, 100));
//	cv::line(matSrcImage, vertices[1], vertices[2], cv::Scalar(100, 100, 100));
//	cv::line(matSrcImage, vertices[2], vertices[3], cv::Scalar(100, 100, 100));
//	cv::line(matSrcImage, vertices[3], vertices[0], cv::Scalar(100, 100, 100));
//}
		}
	}

	//Blob计数
	for (int k = 1; k <= nTotalLabel; k++)
	{
		//如果没有值
		if (rectRo[k].size.width <= 0)	continue;

		for (int m = 1; m <= nTotalLabel; m++)
		{
			//排除相同的故障
			if (k == m)	continue;

			//如果没有值
			if (rectRo[m].size.width <= 0)	continue;

			//范围为南的情况下划线合并
			if (OrientedBoundingBox(rectRo[k], rectRo[m]))
			{
				//获取Center Point
				cv::Point ptLine[2];
				ptLine[0].x = (int)matCentroid.at<double>(k, 0);
				ptLine[0].y = (int)matCentroid.at<double>(k, 1);
				ptLine[1].x = (int)matCentroid.at<double>(m, 0);
				ptLine[1].y = (int)matCentroid.at<double>(m, 1);

				//绘制中心点线
				cv::line(matSrcImage, ptLine[0], ptLine[1], cv::Scalar(255, 255, 255));
			}
		}
	}

	//禁用内存
	delete rectRo;
	rectRo = NULL;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//x方向管接头
long CInspectMura::Estimation_X(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, /*double* dPara*/int nDimensionX, int nEstiStepX, double dEstiBright, double dEstiDark)
{
	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	/*int		nDimensionX		= (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_X	];
	int		nEstiStepX		= (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_X	];
	double	dEstiBright		= dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_BRIGHT		];
	double	dEstiDark		= dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DARK			];*/

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
long CInspectMura::Estimation_Y(cv::Mat& matSrcBuf, cv::Mat& matDstBuf,/* double* dPara*/int nDimensionY, int nEstiStepY, double dEstiBright, double dEstiDark)
{
	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	/*int		nDimensionY		= (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_Y	];
	int		nEstiStepY		= (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_Y	];
	double	dEstiBright		= dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_BRIGHT		];
	double	dEstiDark		= dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DARK			];*/

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

//y方向管接头&平均值
long CInspectMura::Estimation_Y_N_Average(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf, cv::Mat& matDstBuf, double* dPara)
{
	if (matSrc1Buf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrc2Buf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())				return E_ERROR_CODE_EMPTY_BUFFER;

	if (matSrc1Buf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matSrc2Buf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;

	if (matSrc1Buf.rows != matSrc2Buf.rows ||
		matSrc1Buf.cols != matSrc2Buf.cols)			//应为垂直尺寸
		return E_ERROR_CODE_IMAGE_SIZE;

	int		nDimensionY = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_Y];
	int		nEstiStepY = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_Y];
	double	dEstiBright = dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_BRIGHT];
	double	dEstiDark = dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DARK];

	int nStepY = matSrc1Buf.rows / nEstiStepY;

	int nStepRows = matSrc1Buf.rows / nStepY;
	int nHalfRows = matSrc1Buf.rows / 2;

	cv::Mat matM = cv::Mat_<double>(nStepRows, nDimensionY + 1);
	cv::Mat matL = cv::Mat_<double>(nStepRows, 1);
	cv::Mat matQ;

	double y, quad, dTemp;
	int i, j, k, m;

	//亮度值限制
	cv::Scalar mean = cv::mean(matSrc1Buf);
	int nMinGV = (int)(mean[0] * dEstiDark);
	int nMaxGV = (int)(mean[0] * dEstiBright);

	for (j = 0; j < matSrc1Buf.cols; j++)
	{
		for (i = 0; i < nStepRows; i++)
		{
			y = (i * nStepY - nHalfRows) / double(matSrc1Buf.rows);

			matM.at<double>(i, 0) = 1.0;
			dTemp = 1.0;
			for (k = 1; k <= nDimensionY; k++)
			{
				dTemp *= y;
				matM.at<double>(i, k) = dTemp;
			}

			//I.at<double>(i, 0) = matSrc1Buf.at<uchar>(i*nStepY, j);
			m = matSrc1Buf.at<uchar>(i * nStepY, j);

			if (m < nMinGV)	m = nMinGV;
			if (m > nMaxGV)	m = nMaxGV;

			matL.at<double>(i, 0) = m;
		}

		cv::SVD svd(matM);
		svd.backSubst(matL, matQ);

		for (i = 0; i < matSrc1Buf.rows; i++)
		{
			y = (i - nHalfRows) / double(matSrc1Buf.rows);

			quad = matQ.at<double>(0, 0);
			dTemp = 1.0;
			for (k = 1; k <= nDimensionY; k++)
			{
				dTemp *= y;
				quad += (matQ.at<double>(k, 0) * dTemp);
			}

			//平均值
			int nVal = (quad + matSrc2Buf.at<uchar>(i, j)) / 2;

			//matDstBuf.at<uchar>(i, j) = saturate_cast<uchar>(quad);
			matDstBuf.at<uchar>(i, j) = saturate_cast<uchar>(nVal);
		}
	}

	matM.release();
	matL.release();
	matQ.release();

	return E_ERROR_CODE_TRUE;
}

//xy方向管接头
long CInspectMura::Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;

	int		nEstiDimX = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_X];
	int		nEstiDimY = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_Y];
	int		nEstiStepX = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_X];
	int		nEstiStepY = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_Y];

	double	dEstiBright = dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_BRIGHT];
	double	dEstiDark = dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DARK];

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
			nErrorCode |= Estimation_X(matSrcBuf, matDstBufX,/* dPara*/nEstiDimX, nEstiStepX, dEstiBright, dEstiDark);
			break;
		case 1:
			nErrorCode |= Estimation_Y(matSrcBuf, matDstBufY, /*dPara*/nEstiDimY, nEstiStepY, dEstiBright, dEstiDark);
			break;
		}
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

//xy方向管接头
long CInspectMura::Estimation_XY2(cv::Mat& matSrcBuf, cv::Mat& matDstBuf,/* double* dPara,*/ int nEstiDimX, int nEstiDimY, int nEstiStepX, int nEstiStepY, double dEstiBright, double dEstiDark, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;

	/*int		nEstiDimX = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_X];
	int		nEstiDimY = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DIM_Y];
	int		nEstiStepX = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_X];
	int		nEstiStepY = (int)dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_STEP_Y];

	double	dEstiBright = dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_BRIGHT];
	double	dEstiDark = dPara[E_PARA_AVI_MURA_COMMON_ESTIMATION_DARK];*/

	//异常处理
	if (nEstiDimX <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiDimY <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiStepX <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiStepY <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	// 2023.0120
	cv::Mat matDstBufX = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());
	cv::Mat matDstBufY = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());
	//cv::Mat matDstBufX =  cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	//cv::Mat matDstBufY = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif

	for (int i = 0; i < 2; i++)
	{
		switch (i)
		{
		case 0:
			nErrorCode |= Estimation_X(matSrcBuf, matDstBufX,/* dPara*/nEstiDimX, nEstiStepX, dEstiBright, dEstiDark);
			break;
		case 1:
			nErrorCode |= Estimation_Y(matSrcBuf, matDstBufY, /*dPara*/nEstiDimY, nEstiStepY, dEstiBright, dEstiDark);
			break;
		}
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

//如果Dust面积较大,则删除
long CInspectMura::DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS)
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
		if (pResultBlob->nArea[i] > 1000)
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

		/************************************************************************
				//清除不良
				//最后一个index错误-->放入当前index
		{
			pResultBlob->nDefectJudge		[i] = pResultBlob->nDefectJudge		[nDefectCount - 1];
			pResultBlob->nDefectColor		[i] = pResultBlob->nDefectColor		[nDefectCount - 1];
			pResultBlob->nPatternClassify	[i] = pResultBlob->nPatternClassify	[nDefectCount - 1];
			pResultBlob->nArea				[i] = pResultBlob->nArea			[nDefectCount - 1];
			pResultBlob->ptLT				[i] = pResultBlob->ptLT				[nDefectCount - 1];
			pResultBlob->ptRT				[i] = pResultBlob->ptRT				[nDefectCount - 1];
			pResultBlob->ptRB				[i] = pResultBlob->ptRB				[nDefectCount - 1];
			pResultBlob->ptLB				[i] = pResultBlob->ptLB				[nDefectCount - 1];
			pResultBlob->dMeanGV			[i] = pResultBlob->dMeanGV			[nDefectCount - 1];
			pResultBlob->dSigma				[i] = pResultBlob->dSigma			[nDefectCount - 1];
			pResultBlob->nMinGV				[i] = pResultBlob->nMinGV			[nDefectCount - 1];
			pResultBlob->nMaxGV				[i] = pResultBlob->nMaxGV			[nDefectCount - 1];
			pResultBlob->dBackGroundGV		[i] = pResultBlob->dBackGroundGV	[nDefectCount - 1];
			pResultBlob->nCenterx			[i] = pResultBlob->nCenterx			[nDefectCount - 1];
			pResultBlob->nCentery			[i] = pResultBlob->nCentery			[nDefectCount - 1];
			pResultBlob->dBreadth			[i] = pResultBlob->dBreadth			[nDefectCount - 1];
			pResultBlob->dCompactness		[i] = pResultBlob->dCompactness		[nDefectCount - 1];
			pResultBlob->dF_Elongation		[i] = pResultBlob->dF_Elongation	[nDefectCount - 1];
			pResultBlob->dF_Min				[i] = pResultBlob->dF_Min			[nDefectCount - 1];
			pResultBlob->dF_Max				[i] = pResultBlob->dF_Max			[nDefectCount - 1];
			pResultBlob->Lab_avg_L			[i] = pResultBlob->Lab_avg_L		[nDefectCount - 1];
			pResultBlob->Lab_avg_a			[i] = pResultBlob->Lab_avg_a		[nDefectCount - 1];
			pResultBlob->Lab_avg_b			[i] = pResultBlob->Lab_avg_b		[nDefectCount - 1];
			pResultBlob->Lab_diff_L			[i] = pResultBlob->Lab_diff_L		[nDefectCount - 1];
			pResultBlob->Lab_diff_a			[i] = pResultBlob->Lab_diff_a		[nDefectCount - 1];
			pResultBlob->Lab_diff_b			[i] = pResultBlob->Lab_diff_b		[nDefectCount - 1];

#if USE_ALG_HIST
			memcpy(pResultBlob->nHist[i], pResultBlob->nHist[nDefectCount - 1], sizeof(__int64) * IMAGE_MAX_GV);
#endif

			pResultBlob->bUseResult			[i] = pResultBlob->bUseResult		[nDefectCount - 1];

						//清除一个不良总数
			nDefectCount--;
		}
		************************************************************************/
	}

	//重置最终不良计数
//pResultBlob->nDefectCount = nDefectCount;

	return nErrorCode;
}

//清除暗线不良(只留下漏水)
long CInspectMura::DeleteDarkLine(cv::Mat& matSrcBuffer, float fMajorAxisRatio, CMatBuf* cMemSub)
{
	//异常处理
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrcBuffer.size(), CV_32SC1, false);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matSrcBuffer, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

	cv::Rect rectBox;
	int nHW = matSrcBuffer.cols / 2;
	int nHH = matSrcBuffer.rows / 2;

	//根据不良数量
	for (int idx = 1; idx <= nTotalLabel; idx++)
	{
		//标签矩形区域
		rectBox.x = matStats.at<int>(idx, CC_STAT_LEFT);
		rectBox.y = matStats.at<int>(idx, CC_STAT_TOP);
		rectBox.width = matStats.at<int>(idx, CC_STAT_WIDTH);
		rectBox.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

		//清除坏长度大于已点区域的半长度
		if (rectBox.width > nHW)
		{
			__int64 nSum = 0;
			int nEndX = rectBox.x + rectBox.width;

			//求平均线厚度
			for (int x = rectBox.x; x < nEndX; x++)
				nSum += (unsigned int)(cv::sum(matSrcBuffer.col(x))[0]);

			//设置为略大于厚度
			nSum *= fMajorAxisRatio;
			nSum /= rectBox.width;

			bool bLeft = false;

			for (int x = rectBox.x; x < nEndX; x++)
			{
				//删除线厚度小于平均值的情况
				//如果DGS漏电,则保留漏电区域
				if (cv::sum(matSrcBuffer.col(x))[0] < nSum)
					matSrcBuffer(cv::Rect(x, rectBox.y, 1, rectBox.height)).setTo(0);
				//留下
				else
				{
					if (bLeft)
					{
						if (rectBox.x < x)
						{
							bLeft = true;
							rectBox.x = x;
						}
					}
					else
					{
						rectBox.width = x - rectBox.x;
					}
				}
			}
		}

		//清除不良竖向长度大于已点区域的半竖向长度
		if (rectBox.height > nHH)
		{
			__int64 nSum = 0;
			int nEndY = rectBox.y + rectBox.height;

			//求线宽平均厚度
			for (int y = rectBox.y; y < nEndY; y++)
				nSum += (unsigned int)(cv::sum(matSrcBuffer.row(y))[0]);

			//设置为略大于厚度
			nSum *= fMajorAxisRatio;
			nSum /= rectBox.height;

			for (int y = rectBox.y; y < nEndY; y++)
			{
				//如果线条宽度小于平均厚度,则删除
				//如果DGS漏电,则保留漏电区域
				if (cv::sum(matSrcBuffer.row(y))[0] < nSum)
					matSrcBuffer(cv::Rect(rectBox.x, y, rectBox.width, 1)).setTo(0);
			}
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//横向Max GV限制:防止在明点等明亮的不良环境中检测出时发生
long CInspectMura::LimitMaxGV16X(cv::Mat& matSrcBuffer, float fOffset)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	//仅横向...
	for (int y = 0; y < matSrcBuffer.rows; y++)
	{
		//比横向平均值亮一点
		ushort nAvgGV = (ushort)(cv::sum(matSrcBuffer.row(y))[0] / matSrcBuffer.cols * fOffset);

		ushort* ptr = (ushort*)matSrcBuffer.ptr(y);

		//限制亮度,改为平均亮度
		for (int x = 0; x < matSrcBuffer.cols; x++, ptr++)
		{
			if (*ptr > nAvgGV)	*ptr = nAvgGV;
		}
	}

	return nErrorCode;
}

//限制RGB Line Mura面积
//限制面积,消除不良(RGB不良的情况下,没有像线一样连接,而是断开的)
long CInspectMura::LimitArea(cv::Mat& matSrcBuffer, double* dPara, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//参数
	int		nCount = (int)dPara[E_PARA_AVI_MURA_RGB_AREA];
	int		nCut = (int)dPara[E_PARA_AVI_MURA_RGB_INSIDE];

	//删除外围
	cv::Mat matROIBuf = matSrcBuffer(cv::Rect(nCut, nCut, matSrcBuffer.cols - nCut - nCut, matSrcBuffer.rows - nCut - nCut));

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matROIBuf.size(), CV_32SC1, false);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matROIBuf, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

	//长轴/缩短率(用于删除线条)
	float	fLineRatio = 30;

	//删除行时,将其扩展到周围
	int		nLineOffset = 7;

	for (int idx = 1; idx <= nTotalLabel; idx++)
	{
		//对象面积
		long nArea = matStats.at<int>(idx, CC_STAT_AREA);

		//不包括面积太小的
		//不包括太大的(RGB不良的情况下,没有像线一样连接,而是断开的)
		if (30 < nArea && nArea <= nCount)	continue;

		//Blob区域Rect
		cv::Rect rectTemp;
		rectTemp.x = matStats.at<int>(idx, CC_STAT_LEFT);
		rectTemp.y = matStats.at<int>(idx, CC_STAT_TOP);
		rectTemp.width = matStats.at<int>(idx, CC_STAT_WIDTH);
		rectTemp.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

		//如果像横线一样长,则删除该行
		if (rectTemp.width / rectTemp.height > fLineRatio)
		{
			//横向
			cv::rectangle(matROIBuf, cv::Rect(0, rectTemp.y - nLineOffset, matROIBuf.cols, rectTemp.height + nLineOffset + nLineOffset), cv::Scalar(0), -1);
		}
		//如果像竖线一样长,则删除该行
		else if (rectTemp.height / rectTemp.width > fLineRatio)
		{
			//垂直方向
			cv::rectangle(matROIBuf, cv::Rect(rectTemp.x - nLineOffset, 0, rectTemp.width + nLineOffset + nLineOffset, matROIBuf.rows), cv::Scalar(0), -1);
		}
		//如果只是面积小或大的话...
		else
		{
			//初始化为0GV
			cv::Mat matTempROI = matROIBuf(rectTemp);
			cv::Mat matLabelROI = matLabel(rectTemp);

			for (int y = 0; y < rectTemp.height; y++)
			{
				int* ptrLabel = (int*)matLabelROI.ptr(y);
				uchar* ptrGray = (uchar*)matTempROI.ptr(y);

				for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++)
				{
					//将对象删除为0GV
					if (*ptrLabel == idx)	*ptrGray = 0;
				}
			}
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//查找RGB Line Mura
long CInspectMura::JudgeRGBLineMura(cv::Mat& matSrcBuffer, cv::Mat& matBKBuf16, double* dPara, int* nCommonPara, CRect rectROI, stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//参数
	int		nCut = (int)dPara[E_PARA_AVI_MURA_RGB_INSIDE];
	float	fSegX = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_SEG_X];
	float	fSegY = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_SEG_Y];
	float	fArea1Ratio = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_1_RATIO];
	float	fArea2Cnt = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_2_COUNT];
	float	fArea2Ratio = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_2_RATIO];
	float	fAreaMinGV = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_MIN_GV];

	//异常处理
	if (fSegX <= 0)	fSegX = 1;
	if (fSegY <= 0)	fSegY = 1;

	//删除外围
	cv::Mat matROIBuf = matSrcBuffer(cv::Rect(nCut, nCut, matSrcBuffer.cols - nCut - nCut, matSrcBuffer.rows - nCut - nCut));
	cv::Mat matROIBkBuf = matBKBuf16(cv::Rect(nCut, nCut, matSrcBuffer.cols - nCut - nCut, matSrcBuffer.rows - nCut - nCut));

	//总平均值
	float dMean = cv::mean(matROIBkBuf)[0];
	float dMinGV = dMean;

	//一个区域的pixel数量
	float nRangeX = matROIBuf.cols / fSegX;
	float nRangeY = matROIBuf.rows / fSegY;

	cv::Rect rectRange;
	float fTotalMax = 0;
	int nCnt = 0;

	__int64 nSumCount = 0;

	//数据初始化(用于Defect Map轮廓)
	cv::Mat matContoursBuf = cMatBufTemp.GetMat(matSrcBuffer.size(), CV_8UC1);

	int nSegX = fSegX;
	int nSegY = fSegY;

	for (int nY = 0; nY < fSegY; nY++)
	{
		rectRange.y = (int)(nRangeY * nY);
		rectRange.height = (int)(nRangeY * (nY + 1)) - rectRange.y;

		for (int nX = 0; nX < fSegX; nX++)
		{
			rectRange.x = (int)(nRangeX * nX);
			rectRange.width = (int)(nRangeX * (nX + 1)) - rectRange.x;

			//平均
			float dTemp = cv::mean(matROIBkBuf(rectRange))[0];

			//区域的不良面积
			nSumCount = (int)(cv::sum(matROIBuf(rectRange))[0] / 255);

			//点亮面积
			float fTotal = nSumCount / (float)(rectRange.width * rectRange.height) * 100.f;

			//检查最大值
			if (fTotalMax < fTotal)
				fTotalMax = fTotal;

			//如果很弱,传播范围很广...
			//确定区域数量
			if (fTotal > fArea2Ratio)
			{
				//绘制不良区域(用于Defect Map轮廓)
				cv::rectangle(matContoursBuf, cv::Rect(rectRange.x + nCut, rectRange.y + nCut, rectRange.width, rectRange.height), cv::Scalar(2558), -1);

				nCnt++;

				//////////////////////////////////////////////////////////////////////////

							//边除外
				if ((nX == 0 && nY == 0) ||
					(nX == 0 && nY == nSegY - 1) ||
					(nX == nSegX - 1 && nY == 0) ||
					(nX == nSegX - 1 && nY == nSegY - 1))
				{
					//如果是边...
				}
				else
				{
					//查找最小GV
					if (dMinGV > dTemp)
						dMinGV = dTemp;
				}
			}
		}
	}

	//添加RGB Line Mura列表
	if (fTotalMax > fArea1Ratio |	//强烈RGB不良
		nCnt > fArea2Cnt)	//如果存在多个
	{
		//如果存在一个暗区域
		if (dMinGV < fAreaMinGV)
			AddRGBLineMuraDefect(matContoursBuf, dPara, nCommonPara, rectROI, pResultBlob);
	}

	// Log
	wchar_t wcLogTemp[MAX_PATH] = { 0 };
	swprintf_s(wcLogTemp, _T("RGB Value : %.5f, Count :%d, MinGV : %.3f,"), fTotalMax, nCnt, dMinGV);
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, wcLogTemp);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//查找RGB Line Mura(用于保存画面)
long CInspectMura::JudgeRGBLineMuraSave(cv::Mat& matSrcBuffer, cv::Mat& matBKBuf16, double* dPara, int* nCommonPara, CRect rectROI, stDefectInfo* pResultBlob, CString strAlgPath, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//参数
	int		nCut = (int)dPara[E_PARA_AVI_MURA_RGB_INSIDE];
	float	fSegX = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_SEG_X];
	float	fSegY = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_SEG_Y];
	float	fArea1Ratio = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_1_RATIO];
	float	fArea2Cnt = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_2_COUNT];
	float	fArea2Ratio = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_2_RATIO];
	float	fAreaMinGV = (float)dPara[E_PARA_AVI_MURA_RGB_AREA_MIN_GV];

	//异常处理
	if (fSegX <= 0)	fSegX = 1;
	if (fSegY <= 0)	fSegY = 1;

	//删除外围
	cv::Mat matROIBuf = matSrcBuffer(cv::Rect(nCut, nCut, matSrcBuffer.cols - nCut - nCut, matSrcBuffer.rows - nCut - nCut));
	cv::Mat matROIBkBuf = matBKBuf16(cv::Rect(nCut, nCut, matSrcBuffer.cols - nCut - nCut, matSrcBuffer.rows - nCut - nCut));

	//总平均值
	float dMean = cv::mean(matROIBkBuf)[0];
	float dMinGV = dMean;

	//一个区域的pixel数量
	float nRangeX = matROIBuf.cols / fSegX;
	float nRangeY = matROIBuf.rows / fSegY;

	cv::Rect rectRange;
	float fTotalMax = 0;
	int nCnt = 0;

	__int64 nSumCount = 0;

	//数据初始化(用于Defect Map轮廓)
	cv::Mat matContoursBuf = cMatBufTemp.GetMat(matSrcBuffer.size(), CV_8UC1);

	//文本坐标
	cv::Point ptTxT;

	int nSegX = fSegX;
	int nSegY = fSegY;

	//写入文件
	char szFileName[MAX_PATH] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strAlgPath, -1, szFileName, sizeof(szFileName), NULL, NULL);

	FILE* out = NULL;
	fopen_s(&out, szFileName, "wt");

	if (out)
	{
		fprintf_s(out, "nX,nY,Area Ratio,Min GV\n");
	}

	for (int nY = 0; nY < nSegY; nY++)
	{
		rectRange.y = (int)(nRangeY * nY);
		rectRange.height = (int)(nRangeY * (nY + 1)) - rectRange.y;

		//文本坐标
		ptTxT.y = rectRange.y + 25 + nCut;

		for (int nX = 0; nX < nSegX; nX++)
		{
			rectRange.x = (int)(nRangeX * nX);
			rectRange.width = (int)(nRangeX * (nX + 1)) - rectRange.x;

			//平均
			float dTemp = cv::mean(matROIBkBuf(rectRange))[0];

			//文本坐标
			ptTxT.x = rectRange.x + 5 + nCut;

			//区域的不良面积
			nSumCount = (int)(cv::sum(matROIBuf(rectRange))[0] / 255);

			//点亮面积
			float fTotal = nSumCount / (float)(rectRange.width * rectRange.height) * 100.f;

			//显示区域
			cv::rectangle(matROIBuf(rectRange), cv::Rect(0, 0, rectRange.width, rectRange.height), cv::Scalar(128, 128, 128));

			//如果很弱,传播范围很广...
			//确定区域数量
			if (fTotal > fArea2Ratio)
			{
				cv::Mat matGray1 = cv::Mat(rectRange.size(), matROIBuf.type(), 128);
				cv::Mat matGray2 = cv::Mat(rectRange.size(), matROIBuf.type(), 128);
				matROIBuf(rectRange).copyTo(matGray1);

				cv::addWeighted(matGray1, 0.5, matGray2, 0.5, 1.0, matROIBuf(rectRange));

				//绘制不良区域
				cv::rectangle(matContoursBuf, cv::Rect(rectRange.x + nCut, rectRange.y + nCut, rectRange.width, rectRange.height), cv::Scalar(2558), -1);

				nCnt++;

				//////////////////////////////////////////////////////////////////////////

							//边除外
				if ((nX == 0 && nY == 0) ||
					(nX == 0 && nY == nSegY - 1) ||
					(nX == nSegX - 1 && nY == 0) ||
					(nX == nSegX - 1 && nY == nSegY - 1))
				{
					//如果是边...
				}
				else
				{
					//查找最小GV
					if (dMinGV > dTemp)
						dMinGV = dTemp;

					//低于设置GV的区域
					if (fAreaMinGV > dTemp)
					{
						int nTemp = 10;

						//显示区域
						cv::rectangle(matROIBuf(rectRange), cv::Rect(nTemp, nTemp, rectRange.width - nTemp - nTemp, rectRange.height - nTemp - nTemp), cv::Scalar(255));
					}
				}
			}

			//显示文本
			CString strFont;
			strFont.Format(_T("(%02d, %02d)"), nX + 1, nY + 1);
			cv::putText(matSrcBuffer, (cv::String)(CStringA)strFont, ptTxT, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, LINE_AA);

			ptTxT.y += 40;
			strFont.Format(_T("Ratio : %.3f %%"), fTotal);
			cv::putText(matSrcBuffer, (cv::String)(CStringA)strFont, ptTxT, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, LINE_AA);
			ptTxT.y -= 40;

			ptTxT.y += 80;
			strFont.Format(_T("Average : %.3f"), dTemp);
			cv::putText(matSrcBuffer, (cv::String)(CStringA)strFont, ptTxT, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, LINE_AA);
			ptTxT.y -= 80;

			//写入文件
			if (out)
			{
				fprintf_s(out, "%d,%d,%.3f,%.3f\n", nX + 1, nY + 1, fTotal, dTemp);
			}

			//检查最大值
			if (fTotalMax < fTotal)
				fTotalMax = fTotal;
		}
	}

	//写入文件
	if (out)
	{
		fclose(out);
	}

	//总平均值
	CString strFont2;
	strFont2.Format(_T("Total Max Ratio : %.3f %%"), fTotalMax);
	cv::putText(matSrcBuffer, (cv::String)(CStringA)strFont2, cv::Point(nCut + 5, nCut + 145), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, LINE_AA);
	strFont2.Format(_T("Total MinGV : %.3f"), dMinGV);
	cv::putText(matSrcBuffer, (cv::String)(CStringA)strFont2, cv::Point(nCut + 5, nCut + 185), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, LINE_AA);

	//添加RGB Line Mura列表
	if (fTotalMax > fArea1Ratio |	//强烈RGB不良
		nCnt > fArea2Cnt)	//如果存在多个
	{
		//如果存在一个暗区域
		if (dMinGV < fAreaMinGV)
			AddRGBLineMuraDefect(matContoursBuf, dPara, nCommonPara, rectROI, pResultBlob);
	}

	// Log
	wchar_t wcLogTemp[MAX_PATH] = { 0 };
	swprintf_s(wcLogTemp, _T("RGB Value : %.5f, Count :%d, MinGV : %.3f,"), fTotalMax, nCnt, dMinGV);
	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, wcLogTemp);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//添加RGB Line Mura列表
long CInspectMura::AddRGBLineMuraDefect(cv::Mat& matContoursBuf, double* dPara, int* nCommonPara, CRect rectROI, stDefectInfo* pResultBlob)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//异常处理
	if (matContoursBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	//使用参数
	int		nResize = (int)dPara[E_PARA_AVI_MURA_RGB_RESIZE];
	if (nResize <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];

	//18.09.25-RGB Line Mura需要添加
	int		nPatUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	if (pResultBlob != NULL)
	{
		pResultBlob->nArea[0] = 0;
		pResultBlob->nMaxGV[0] = 255;
		pResultBlob->nMinGV[0] = 0;
		pResultBlob->dMeanGV[0] = 0;

		pResultBlob->ptLT[0].x = 0;
		pResultBlob->ptLT[0].y = 0;
		pResultBlob->ptRT[0].x = 0;
		pResultBlob->ptRT[0].y = 0;
		pResultBlob->ptRB[0].x = 0;
		pResultBlob->ptRB[0].y = 0;
		pResultBlob->ptLB[0].x = 0;
		pResultBlob->ptLB[0].y = 0;

		pResultBlob->dBackGroundGV[0] = 0;
		pResultBlob->dCompactness[0] = 0;
		pResultBlob->dSigma[0] = 0;
		pResultBlob->dBreadth[0] = 0;
		pResultBlob->dF_Min[0] = 0;
		pResultBlob->dF_Max[0] = 0;
		pResultBlob->dF_Elongation[0] = 0;
		pResultBlob->dCompactness[0] = 0;

		//亮度
		pResultBlob->nDefectColor[0] = E_DEFECT_COLOR_DARK;

		pResultBlob->nDefectJudge[0] = E_DEFECT_JUDGEMENT_MURA_LINE_X;
		pResultBlob->nPatternClassify[0] = nPatUI;

		//计数增加
		pResultBlob->nDefectCount = 1;

		//拯救Contours
		vector <cv::Point>	ptContoursArr;
		vector <cv::Point>().swap(ptContoursArr);

		vector<vector<cv::Point>>	ptContours;
		vector<vector<cv::Point>>().swap(ptContours);
		cv::findContours(matContoursBuf, ptContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		//Contours计数
		for (int m = 0; m < ptContours.size(); m++)
		{
			//更改为Reseze原始坐标
			for (int k = 0; k < ptContours.at(m).size(); k++)
				ptContoursArr.push_back(cv::Point(ptContours.at(m)[k].x * nResize + rectROI.left, ptContours.at(m)[k].y * nResize + rectROI.top));
		}

		//如果超过设置数量
		int nContoursCount = ptContoursArr.size();
		float fRatio = 1.0;
		if (nContoursCount >= MAX_CONTOURS)
		{
			fRatio = nContoursCount / (float)MAX_CONTOURS;
			nContoursCount = MAX_CONTOURS;
		}

		//保存轮廓
		for (int j = 0; j < nContoursCount; j++)
		{
			//实际使用的轮廓Index
			int i = (int)(j * fRatio);

			//根据P/S模式修改坐标(更改为单杆坐标)
			pResultBlob->nContoursX[0][j] = (int)(ptContoursArr[i].x / nPS);
			pResultBlob->nContoursY[0][j] = (int)(ptContoursArr[i].y / nPS);
		}

		//初始化
		vector <cv::Point>().swap(ptContoursArr);
		vector<vector<cv::Point>>().swap(ptContours);
	}

	return nErrorCode;
}

//重新分类White Spot
long CInspectMura::JudgeWhiteSpot(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
	stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//如果没有出现故障,请退出
	int nCount = pResultBlob->nDefectCount;
	if (nCount <= 0)		return E_ERROR_CODE_TRUE;

	//使用参数
	int		nMorpObj = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_MORP_OBJ];
	int		nMorpBKG = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_MORP_BKG];
	int		nThreshold = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_THRESHOLD];

	// Active Spec
	double	dSpecActiveBrightRatio = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_BRIGHT_RATIO];

	double	dSpecActiveDarkRatio1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_1];
	double	dSpecActiveDarkArea1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA_1];
	double	dSpecActiveDarkDiff1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_1];

	double	dSpecActiveDarkRatio2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_2];
	double	dSpecActiveDarkArea2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA_2];
	double	dSpecActiveDarkDiff2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_2];

	double	dSpecActiveDarkRatio3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_3];
	double	dSpecActiveDarkArea3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA_3];
	double	dSpecActiveDarkDiff3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_3];

	//设置Edge区域
	double	dSpecEdgeAreaL = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_LEFT];
	double	dSpecEdgeAreaT = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_TOP];
	double	dSpecEdgeAreaR = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_RIGHT];
	double	dSpecEdgeAreaB = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_AREA_BOTTOM];

	//异常处理
	if (dSpecEdgeAreaL < 0)				dSpecEdgeAreaL = 0;
	if (dSpecEdgeAreaT < 0)				dSpecEdgeAreaT = 0;
	if (dSpecEdgeAreaR < 0)				dSpecEdgeAreaR = 0;
	if (dSpecEdgeAreaB < 0)				dSpecEdgeAreaB = 0;

	// Edge Spec
	double	dSpecEdgeBrightRatio = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_BRIGHT_RATIO];

	double	dSpecEdgeDarkRatio1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_1];
	double	dSpecEdgeDarkArea1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA_1];
	double	dSpecEdgeDarkDiff1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_1];

	double	dSpecEdgeDarkRatio2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_2];
	double	dSpecEdgeDarkArea2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA_2];
	double	dSpecEdgeDarkDiff2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_2];

	double	dSpecEdgeDarkRatio3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_3];
	double	dSpecEdgeDarkArea3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA_3];
	double	dSpecEdgeDarkDiff3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_3];

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

		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	long	nWidth = (long)matSrcBuffer.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuffer.rows;	// 图像垂直尺寸

	int nOffset = 200;
	cv::Mat matDefectMorp1Buf = cMatBufTemp.GetMat(cv::Size(nOffset, nOffset), matSrcBuffer.type(), false);
	cv::Mat matDefectMorp2Buf = cMatBufTemp.GetMat(cv::Size(nOffset, nOffset), matSrcBuffer.type(), false);
	cv::Mat matDefectBKBuf = cMatBufTemp.GetMat(cv::Size(nOffset, nOffset), matSrcBuffer.type(), false);
	cv::Mat matDefectThBuf = cMatBufTemp.GetMat(cv::Size(nOffset, nOffset), matSrcBuffer.type(), false);

	//根据不良数量
	for (int i = 0; i < nCount; i++)
	{
		//如果Spot不坏,请跳过
		if (pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT || pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT) //04.16 choi
			continue;

		//不良周边区域
		cv::Rect rectTempROI(pResultBlob->nCenterx[i] - nOffset / 2, pResultBlob->nCentery[i] - nOffset / 2, nOffset, nOffset);
		if (rectTempROI.x < 0)						rectTempROI.x = 0;
		if (rectTempROI.y < 0)						rectTempROI.y = 0;
		if (rectTempROI.x + nOffset >= nWidth)		rectTempROI.x = nWidth - nOffset - 1;
		if (rectTempROI.y + nOffset >= nHeight)	rectTempROI.y = nHeight - nOffset - 1;

		//坏区域
		cv::Mat matDefectSrcBuf = matSrcBuffer(rectTempROI);
		cv::Mat matTempBuf = matDstBuffer(rectTempROI);

		//为了一百分,李振华(一百分:255/百村200)
		cv::Mat matDefectResBuf = cMatBufTemp.GetMat(matTempBuf.size(), matTempBuf.type(), false);
		cv::threshold(matTempBuf, matDefectResBuf, 220, 255.0, CV_THRESH_BINARY);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Src_%02d.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i);
			ImageSave(strTemp, matDefectSrcBuf);

			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Res_%02d.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i);
			ImageSave(strTemp, matDefectResBuf);
		}

		//将不良设置为稍大
		int nValue = nMorpObj * 2 + 1;
		if (nMorpObj >= 1)
			cv::morphologyEx(matDefectResBuf, matDefectMorp1Buf, MORPH_DILATE, cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(nValue, nValue)));
		else
			matDefectResBuf.copyTo(matDefectMorp1Buf);

		//背景部分
		cv::bitwise_not(matDefectMorp1Buf, matDefectBKBuf);

		//寻找黑暗的部分
		cv::threshold(matDefectSrcBuf, matDefectThBuf, 20, 255.0, THRESH_BINARY_INV);

		//清除黑暗
		cv::subtract(matDefectMorp1Buf, matDefectThBuf, matDefectMorp1Buf);	// 检测
		cv::subtract(matDefectBKBuf, matDefectThBuf, matDefectBKBuf);	// 背景

		//获取背景mean&stdDev
		cv::Scalar meanBK, stdBK;
		cv::meanStdDev(matDefectSrcBuf, meanBK, stdBK, matDefectBKBuf);

		//查找亮的部分
		cv::threshold(matDefectSrcBuf, matDefectThBuf, nThreshold, 255.0, THRESH_BINARY);

		//将亮部分设置得大一点
		nValue = nMorpBKG * 2 + 1;
		if (nMorpBKG >= 1)
			cv::morphologyEx(matDefectThBuf, matDefectMorp2Buf, MORPH_DILATE, cv::getStructuringElement(MORPH_RECT, cv::Size(nValue, nValue)));
		else
			matDefectThBuf.copyTo(matDefectMorp2Buf);

		//检查检测出的不良品中是否有亮的部分
		double valMax = 0;
		cv::bitwise_and(matDefectMorp1Buf, matDefectMorp2Buf, matDefectThBuf);
		cv::minMaxLoc(matDefectThBuf, NULL, &valMax);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Bri_%02d.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i);
			ImageSave(strTemp, matDefectThBuf);
		}

		//检查是否有亮区域
		bool bPoint = (valMax > 100) ? true : false;

		//仅设置活动区域
		CRect rectTemp;
		rectTemp.left = rectROI.left + dSpecEdgeAreaL;
		rectTemp.top = rectROI.top + dSpecEdgeAreaT;
		rectTemp.right = rectROI.right - dSpecEdgeAreaR;
		rectTemp.bottom = rectROI.bottom - dSpecEdgeAreaB;

		//不良中心坐标
		CPoint ptSrc(pResultBlob->nCenterx[i], pResultBlob->nCentery[i]);

		//活动范围内Spot存在不良
		if (rectTemp.PtInRect(ptSrc))
		{
			//Mura Active故障
			pResultBlob->bMuraActive[i] = true;

			//如果存在亮点,则Spec
			if (bPoint)
			{
				//Mura存在明亮的GV
				pResultBlob->bMuraBright[i] = true;

				//检测出的不良内容中去除亮部分的画面
				cv::subtract(matDefectMorp1Buf, matDefectThBuf, matDefectMorp2Buf);

				//获取检测部分mean&stdDev
				cv::Scalar meanObj, stdObj;
				cv::meanStdDev(matDefectSrcBuf, meanObj, stdObj, matDefectMorp2Buf);

				//输入Mura亮度信息
				pResultBlob->dMuraObj[i] = meanObj[0];
				pResultBlob->dMuraBk[i] = meanBK[0];

				if (meanObj[0] - meanBK[0] > dSpecActiveBrightRatio) { pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT; }
				else													pResultBlob->bUseResult[i] = false;

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_Src_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanObj[0]);
					ImageSave(strTemp, matDefectMorp2Buf);

					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_BK_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanBK[0]);
					ImageSave(strTemp, matDefectBKBuf);
				}
			}
			//明暗点不存在时规格
			else
			{
				//Mura不存在亮GV
				pResultBlob->bMuraBright[i] = false;

				//检测出的不良内容中去除亮部分的画面
				cv::subtract(matDefectMorp1Buf, matDefectThBuf, matDefectMorp2Buf);

				//获取检测部分mean&stdDev
				cv::Scalar meanObj, stdObj;
				cv::meanStdDev(matDefectSrcBuf, meanObj, stdObj, matDefectMorp2Buf);

				//输入Mura亮度信息
				pResultBlob->dMuraObj[i] = meanObj[0];
				pResultBlob->dMuraBk[i] = meanBK[0];

				//村信息
				double	dArea = pResultBlob->nArea[i];
				double	dSub = meanObj[0] - meanBK[0];
				double	dDiff = pResultBlob->dMeanGV[i] / pResultBlob->dBackGroundGV[i];

				// Spec1
				if (dArea >= dSpecActiveDarkArea1 &&
					dSpecActiveDarkArea1 > 0 &&
					dSub > dSpecActiveDarkRatio1 &&
					dDiff > dSpecActiveDarkDiff1)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec2
				else if (dArea >= dSpecActiveDarkArea2 &&
					dSpecActiveDarkArea2 > 0 &&
					dSub > dSpecActiveDarkRatio2 &&
					dSub <= dSpecActiveDarkRatio1 &&
					dDiff > dSpecActiveDarkDiff2)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec3
				else if (dArea >= dSpecActiveDarkArea3 &&
					dSpecActiveDarkArea3 > 0 &&
					dSub > dSpecActiveDarkRatio3 &&
					dSub <= dSpecActiveDarkRatio2 &&
					dDiff > dSpecActiveDarkDiff3)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				else
				{
					pResultBlob->bUseResult[i] = false;
				}

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_Src_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanObj[0]);
					ImageSave(strTemp, matDefectMorp2Buf);

					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_BK_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanBK[0]);
					ImageSave(strTemp, matDefectBKBuf);
				}
			}
		}
		//Edge范围内存在Spot不良
		else
		{
			//Mura Active没有问题
			pResultBlob->bMuraActive[i] = false;

			//如果存在亮点,则Spec
			if (bPoint)
			{
				//Mura存在明亮的GV
				pResultBlob->bMuraBright[i] = true;

				//检测出的不良内容中去除亮部分的画面
				cv::subtract(matDefectMorp1Buf, matDefectThBuf, matDefectMorp2Buf);

				//获取检测部分mean&stdDev
				cv::Scalar meanObj, stdObj;
				cv::meanStdDev(matDefectSrcBuf, meanObj, stdObj, matDefectMorp2Buf);

				//输入Mura亮度信息
				pResultBlob->dMuraObj[i] = meanObj[0];
				pResultBlob->dMuraBk[i] = meanBK[0];

				if (meanObj[0] - meanBK[0] > dSpecEdgeBrightRatio)	pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				else												pResultBlob->bUseResult[i] = false;

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_Src_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanObj[0]);
					ImageSave(strTemp, matDefectMorp2Buf);

					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_BK_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanBK[0]);
					ImageSave(strTemp, matDefectBKBuf);
				}
			}
			//明暗点不存在时规格
			else
			{
				//Mura不存在亮GV
				pResultBlob->bMuraBright[i] = false;

				//检测出的不良内容中去除亮部分的画面
				cv::subtract(matDefectMorp1Buf, matDefectThBuf, matDefectMorp2Buf);

				//获取检测部分mean&stdDev
				cv::Scalar meanObj, stdObj;
				cv::meanStdDev(matDefectSrcBuf, meanObj, stdObj, matDefectMorp2Buf);

				//输入Mura亮度信息
				pResultBlob->dMuraObj[i] = meanObj[0];
				pResultBlob->dMuraBk[i] = meanBK[0];

				//关于Mura
				double	dArea = pResultBlob->nArea[i];
				double	dSub = meanObj[0] - meanBK[0];
				double	dDiff = pResultBlob->dMeanGV[i] / pResultBlob->dBackGroundGV[i];

				// Spec1
				if (dArea >= dSpecEdgeDarkArea1 &&
					dSpecEdgeDarkArea1 > 0 &&
					dSub > dSpecEdgeDarkRatio1 &&
					dDiff > dSpecEdgeDarkDiff1)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec2
				else if (dArea >= dSpecEdgeDarkArea2 &&
					dSpecEdgeDarkArea2 > 0 &&
					dSub > dSpecEdgeDarkRatio2 &&
					dSub <= dSpecEdgeDarkRatio1 &&
					dDiff > dSpecEdgeDarkDiff2)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec3
				else if (dArea >= dSpecEdgeDarkArea3 &&
					dSpecEdgeDarkArea3 > 0 &&
					dSub > dSpecEdgeDarkRatio3 &&
					dSub <= dSpecEdgeDarkRatio2 &&
					dDiff > dSpecEdgeDarkDiff3)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				else
				{
					pResultBlob->bUseResult[i] = false;
				}

				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_Src_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanObj[0]);
					ImageSave(strTemp, matDefectMorp2Buf);

					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_BK_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanBK[0]);
					ImageSave(strTemp, matDefectBKBuf);
				}
			}
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//White Mura重新分类
long CInspectMura::JudgeWhiteMura(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
	stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//如果没有出现故障,请退出
	int nCount = pResultBlob->nDefectCount;
	if (nCount <= 0)		return E_ERROR_CODE_TRUE;

	//使用参数
//////////////////////////////////////////////////////////////////////////spec1
	int		nSpec1_Act_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC1_FLAG];
	double	dSpecActiveDarkRatio1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_1];
	double	dSpecActiveDarkArea1_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA1_MIN];
	double	dSpecActiveDarkArea1_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA1_MAX];
	double	dSpecActiveDarkDiff1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_1];

	//////////////////////////////////////////////////////////////////////////spec2
	int		nSpec2_Act_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC2_FLAG];
	double	dSpecActiveDarkRatio2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_2];
	double	dSpecActiveDarkArea2_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA2_MIN];
	double	dSpecActiveDarkArea2_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA2_MAX];
	double	dSpecActiveDarkDiff2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_2];

	//////////////////////////////////////////////////////////////////////////spec3
	int		nSpec3_Act_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC3_FLAG];
	double	dSpecActiveDarkRatio3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_3];
	double	dSpecActiveDarkArea3_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA3_MIN];
	double	dSpecActiveDarkArea3_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA3_MAX];
	double	dSpecActiveDarkDiff3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_3];

	//////////////////////////////////////////////////////////////////////////spec4
	int		nSpec4_Act_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_SPEC4_FLAG];
	double	dSpecActiveDarkRatio4 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_RATIO_4];
	double	dSpecActiveDarkArea4_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA4_MIN];
	double	dSpecActiveDarkArea4_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_AREA4_MAX];
	double	dSpecActiveDarkDiff4 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_ACTIVE_SPEC_DARK_DIFF_4];

	//设置Edge区域
	double	dSpecEdgeAreaL = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_LEFT];
	double	dSpecEdgeAreaT = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_TOP];
	double	dSpecEdgeAreaR = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_RIGHT];
	double	dSpecEdgeAreaB = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_AREA_BOTTOM];

	//异常处理
	if (dSpecEdgeAreaL < 0)				dSpecEdgeAreaL = 0;
	if (dSpecEdgeAreaT < 0)				dSpecEdgeAreaT = 0;
	if (dSpecEdgeAreaR < 0)				dSpecEdgeAreaR = 0;
	if (dSpecEdgeAreaB < 0)				dSpecEdgeAreaB = 0;

	// Edge Spec

	//////////////////////////////////////////////////////////////////////////SPEC1
	int		nSpec1_Edge_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC1_FLAG];
	double	dSpecEdgeDarkRatio1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_1];
	double	dSpecEdgeDarkArea1_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA1_MIN];
	double	dSpecEdgeDarkArea1_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA1_MAX];
	double	dSpecEdgeDarkDiff1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_1];

	//////////////////////////////////////////////////////////////////////////SPEC2
	int		nSpec2_Edge_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC2_FLAG];
	double	dSpecEdgeDarkRatio2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_2];
	double	dSpecEdgeDarkArea2_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA2_MIN];
	double	dSpecEdgeDarkArea2_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA2_MAX];
	double	dSpecEdgeDarkDiff2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_2];

	//////////////////////////////////////////////////////////////////////////SPEC3
	int		nSpec3_Edge_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC3_FLAG];
	double	dSpecEdgeDarkRatio3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_3];
	double	dSpecEdgeDarkArea3_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA3_MIN];
	double	dSpecEdgeDarkArea3_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA3_MAX];
	double	dSpecEdgeDarkDiff3 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_3];

	//////////////////////////////////////////////////////////////////////////SPEC4
	int		nSpec4_Edge_Flag = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_SPEC4_FLAG];
	double	dSpecEdgeDarkRatio4 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_RATIO_4];
	double	dSpecEdgeDarkArea4_MIN = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA4_MIN];
	double	dSpecEdgeDarkArea4_MAX = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_AREA4_MAX];
	double	dSpecEdgeDarkDiff4 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_EDGE_SPEC_DARK_DIFF_4];

	//////////////////////////////////////////////////////////////////////////
		//根据不良数量
	for (int i = 0; i < nCount; i++)
	{
		//如果White Mura不是不良行为,请跳过
		if (pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT)
			continue;

		//仅设置活动区域
		CRect rectTemp;
		rectTemp.left = rectROI.left + dSpecEdgeAreaL;
		rectTemp.top = rectROI.top + dSpecEdgeAreaT;
		rectTemp.right = rectROI.right - dSpecEdgeAreaR;
		rectTemp.bottom = rectROI.bottom - dSpecEdgeAreaB;

		//不良中心坐标
		CPoint ptSrc(pResultBlob->nCenterx[i], pResultBlob->nCentery[i]);

		//活动范围内存在WM不良
		if (rectTemp.PtInRect(ptSrc))
		{

			//村信息
			double	dArea = pResultBlob->nArea[i];
			double	dSub = pResultBlob->dMeanGV[i] - pResultBlob->dBackGroundGV[i];
			double	dDiff = pResultBlob->dMeanGV[i] / pResultBlob->dBackGroundGV[i];

			// Spec1
			if (dArea >= dSpecActiveDarkArea1_MIN &&
				dArea < dSpecActiveDarkArea1_MAX &&
				nSpec1_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio1 &&
				dDiff > dSpecActiveDarkDiff1)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			// Spec2
			else if (dArea >= dSpecActiveDarkArea2_MIN &&
				dArea < dSpecActiveDarkArea2_MAX &&
				nSpec2_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio2 &&
				dDiff > dSpecActiveDarkDiff2)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			// Spec3
			else if (dArea >= dSpecActiveDarkArea3_MIN &&
				dArea < dSpecActiveDarkArea3_MAX &&
				nSpec3_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio3 &&
				dDiff > dSpecActiveDarkDiff3)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			// Spec4
			else if (dArea >= dSpecActiveDarkArea4_MIN &&
				dArea < dSpecActiveDarkArea4_MAX &&
				nSpec4_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio4 &&
				dDiff > dSpecActiveDarkDiff4)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			else
			{
				pResultBlob->bUseResult[i] = false;
			}
		}
		//Edge范围内存在WM不良
		else {
			//村信息
			double	dArea = pResultBlob->nArea[i];
			double	dSub = pResultBlob->dMeanGV[i] - pResultBlob->dBackGroundGV[i];
			double	dDiff = pResultBlob->dMeanGV[i] / pResultBlob->dBackGroundGV[i];

			// Spec1
			if (dArea >= dSpecActiveDarkArea1_MIN &&
				dArea < dSpecActiveDarkArea1_MAX &&
				nSpec1_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio1 &&
				dDiff > dSpecActiveDarkDiff1)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			// Spec2
			else if (dArea >= dSpecActiveDarkArea2_MIN &&
				dArea < dSpecActiveDarkArea2_MAX &&
				nSpec2_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio2 &&
				dDiff > dSpecActiveDarkDiff2)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			// Spec3
			else if (dArea >= dSpecActiveDarkArea3_MIN &&
				dArea < dSpecActiveDarkArea3_MAX &&
				nSpec3_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio3 &&
				dDiff > dSpecActiveDarkDiff3)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			// Spec4
			else if (dArea >= dSpecActiveDarkArea4_MIN &&
				dArea < dSpecActiveDarkArea4_MAX &&
				nSpec4_Act_Flag > 0 &&
				dSub > dSpecActiveDarkRatio4 &&
				dDiff > dSpecActiveDarkDiff4)
			{
				pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
			}
			else
			{
				pResultBlob->bUseResult[i] = false;
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}

//White Mura重新分类
/*long CInspectMura::JudgeWhiteMura(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
	stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
		//如果没有出现故障,请退出
	int nCount = pResultBlob->nDefectCount;
	if (nCount <= 0)		return E_ERROR_CODE_TRUE;

		//使用参数
	double	dSpecMuraRatio1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_DIFF_1];
	double	dSpecMuraArea1 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_AREA_1];

	double	dSpecMuraRatio2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_DIFF_2];
	double	dSpecMuraArea2 = dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_AREA_2];

	//////////////////////////////////////////////////////////////////////////

		//根据不良数量
	for (int i = 0; i < nCount; i++)
	{
				//如果White Mura不是不良行为,请跳过
		if (pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT)
			continue;

				//村信息
		double	dArea = pResultBlob->nArea[i];
		double	dDiff = pResultBlob->dMeanGV[i] / pResultBlob->dBackGroundGV[i];

		// Spec1
		if (dArea >= dSpecMuraArea1		&&
			dSpecMuraArea1 > 0 &&
			dDiff > dSpecMuraRatio1)
		{
			pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
		}
		// Spec2
		else if (dArea >= dSpecMuraArea2	&&
			dSpecMuraArea2 > 0 &&
			dDiff > dSpecMuraRatio2)
		{
			pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT;
		}
		else
		{
			pResultBlob->bUseResult[i] = false;
		}
	}

	return E_ERROR_CODE_TRUE;
}*/

//Nugi重新分类
//SCJ 20.02.18-B11曹文请求EDGE区域//ACTIVE区域NUGI错误分类请求
long CInspectMura::JudgeNugi(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
	stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//如果没有出现故障,请退出
	int nCount = pResultBlob->nDefectCount;
	if (nCount <= 0)		return E_ERROR_CODE_TRUE;

	//要使用吗？
	double	dUse = dPara[E_PARA_AVI_MURA_JUDGE_EDGE_NUGI_USE];
	if (dUse <= 0)			return E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////

		//根据不良数量
	for (int i = 0; i < nCount; i++)
	{
		//如果Nugi不坏,跳过
		if (pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_NUGI)
			continue;

		//耳朵部分区域大
//if (pResultBlob->nArea[i] < 450000)
//	continue;
//
		////耳朵部分区域很长
//if (pResultBlob->dF_Elongation[i] < 3.0)
//	continue;

		//只设置贵区域
//CRect rectTemp;
//rectTemp.left = rectROI.left;
//rectTemp.top = rectROI.top;
//rectTemp.right = rectROI.left + 400;
//rectTemp.bottom = rectROI.bottom;

		//设置整个ACTIVE区域的EDGE标准
		int nRectInX = 150, nRectInY = 150;
		CRect rectTemp;
		rectTemp.left = rectROI.left + nRectInX;
		rectTemp.top = rectROI.top + nRectInY;
		rectTemp.right = rectROI.right - nRectInX;
		rectTemp.bottom = rectROI.bottom - nRectInY;

		//不良中心坐标
		CPoint ptSrc(pResultBlob->nCenterx[i], pResultBlob->nCentery[i]);

		//耳朵部分范围内存在不良
//if (!rectTemp.PtInRect(ptSrc))
//	continue;

		if (rectTemp.PtInRect(ptSrc))
			continue;

		pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI;
	}

	return E_ERROR_CODE_TRUE;
}

bool CInspectMura::OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2)
{
	// Distance Vector
	cv::Point2d dist, unit, vec[4];
	dist.x = rect1.center.x - rect2.center.x;
	dist.y = rect1.center.y - rect2.center.y;

	// rect1 Height Vector
	vec[0].x = rect1.size.height * cos((rect1.angle - 90.0) / 180.0 * PI) / 2.0;
	vec[0].y = rect1.size.height * sin((rect1.angle - 90.0) / 180.0 * PI) / 2.0;

	// rect2 Height Vector
	vec[1].x = rect2.size.height * cos((rect2.angle - 90.0) / 180.0 * PI) / 2.0;
	vec[1].y = rect2.size.height * sin((rect2.angle - 90.0) / 180.0 * PI) / 2.0;

	// rect1 Width Vector
	vec[2].x = rect1.size.width * cos(rect1.angle / 180.0 * PI) / 2.0;
	vec[2].y = rect1.size.width * sin(rect1.angle / 180.0 * PI) / 2.0;

	// rect2 Width Vector
	vec[3].x = rect2.size.width * cos(rect2.angle / 180.0 * PI) / 2.0;
	vec[3].y = rect2.size.width * sin(rect2.angle / 180.0 * PI) / 2.0;

	//四个向量中...
	bool bRes = true;
	for (int i = 0; i < 4; i++)
	{
		double dSum = 0;

		double dSize = sqrt(vec[i].x * vec[i].x + vec[i].y * vec[i].y);
		unit.x = (double)(vec[i].x / dSize);
		unit.y = (double)(vec[i].y / dSize);

		//内积4个...
		for (int j = 0; j < 4; j++)
			dSum += abs(unit.x * vec[j].x + unit.y * vec[j].y);

		//如果存在一个可以分离的情况
		if (abs(unit.x * dist.x + unit.y * dist.y) > dSum)
		{
			bRes = false;
			break;
		}
	}

	return bRes;
}

//////////////////////////////////////////////////////////////////////////choi 21.02.05
//////////////////////////////////////////////////////////////////////////
//G3 Pattern G3/L3混色不良检测算法
// 20.07.13
// PNZ
//////////////////////////////////////////////////////////////////////////

long CInspectMura::LogicStart_MuraG3CM(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool* bFlag)
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
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3-CM Start."));

	// Parameters
	int		nShiftCopy = (int)dPara[E_PARA_AVI_MURA_G3_PREPROCESS_SHIFTCOPY];		// 5010;
	int		nResizeUnitPara = (int)dPara[E_PARA_AVI_MURA_G3_PREPROCESS_RESIZEUNIT];		// 100;
	int		nLimitLength = (int)dPara[E_PARA_AVI_MURA_G3_PREPROCESS_LIMITLENGTH];		// 10000;	
	int		nBlurLevel01 = (int)dPara[E_PARA_AVI_MURA_G3_PREPROCESS_BLUR_L01];		// 15;
	int		nBlurLevel02 = (int)dPara[E_PARA_AVI_MURA_G3_PREPROCESS_BLUR_L02];		// 9;
	int		nBlurLevel03 = (int)dPara[E_PARA_AVI_MURA_G3_PREPROCESS_BLUR_L03];		// 3;
	int		nJudge_DiffGV = (int)dPara[E_PARA_AVI_MURA_G3_JUDGE_DIFF_GV];		// 13;

	// Memory Reset
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	cv::Rect rectTemp;

	int nRatio = 0;
	int nResizeUnit = 0;
	int nCpyX = 0;
	int nCpyY = 0;
	int nLoopX = 0;
	int nLoopY = 0;

	if (matSrcImage.cols > nLimitLength) { nRatio = 2; nResizeUnit = nResizeUnitPara; }
	else { nRatio = 1; nResizeUnit = nResizeUnitPara / 2; }

	rectTemp.x = rectROI.left * nRatio;
	rectTemp.y = rectROI.top * nRatio;
	rectTemp.width = rectROI.right * nRatio - rectROI.left * nRatio;
	rectTemp.height = rectROI.bottom * nRatio - rectROI.top * nRatio;

	cv::Mat matSrcTemp = cMatBufTemp.GetMat(rectTemp.height, rectTemp.width, CV_8UC1, false);
	cv::Mat matSubTemp_SC = cMatBufTemp.GetMat(rectTemp.height, rectTemp.width, CV_8UC1, false);
	cv::Mat matSubTemp_Blur1 = cMatBufTemp.GetMat(rectTemp.height / nResizeUnit, rectTemp.width / nResizeUnit, CV_8UC1, false);
	cv::Mat matSubTemp_Blur2 = cMatBufTemp.GetMat(rectTemp.height / nResizeUnit, rectTemp.width / nResizeUnit, CV_8UC1, false);
	cv::Mat matSubTemp_RE = cMatBufTemp.GetMat(rectTemp.height / nResizeUnit, rectTemp.width / nResizeUnit, CV_8UC1, false);

	matSrcImage(rectTemp).copyTo(matSrcTemp);

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3-CM Get Memory & Set Para."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_Input.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_Input_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcTemp);
	}

	//////////////////////////////////////////////////////////////////////////
	// Pre-Process

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3-CM Pre-Process Start."));

	ShiftCopyParaCheck(nShiftCopy, nCpyX, nCpyY, nLoopX, nLoopY);
	AlgoBase::ShiftCopy(matSrcTemp, matSubTemp_SC, nCpyX, nCpyY, nLoopX, nLoopY, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_ShiftCopy.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSubTemp_SC);
	}

	cv::blur(matSubTemp_SC, matSubTemp_Blur1, cv::Size(nBlurLevel01, nBlurLevel01));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_Blur1.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSubTemp_Blur1);
	}

	AveragingReducer(matSubTemp_Blur1, matSubTemp_Blur2, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_AR.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSubTemp_Blur2);
	}

	cv::blur(matSubTemp_Blur2, matSubTemp_Blur2, cv::Size(nBlurLevel02, nBlurLevel02));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_Blur2.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSubTemp_Blur2);
	}

	cv::resize(matSubTemp_Blur2, matSubTemp_RE, matSubTemp_SC.size() / nResizeUnit, INTER_CUBIC);

	cv::blur(matSubTemp_RE, matSubTemp_RE, cv::Size(nBlurLevel03, nBlurLevel03));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_Resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSubTemp_RE);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3-CM Pre-Process End."));

	// Main Process

	cv::Mat matHist_Total, matHist_Resize;

	AlgoBase::GetHistogram(matSubTemp_Blur1, matHist_Total, false);
	AlgoBase::GetHistogram(matSubTemp_RE, matHist_Resize, false);

	int		nLowerIndex = 0;
	int		nUpperIndex = 0;
	int		nLowUpDiff = 0;
	int		nLowVArea = 0;
	int		nTopVArea = 0;

	HistAreaCalc(matHist_Resize, nLowerIndex, nUpperIndex, nLowUpDiff, nLowVArea, nTopVArea, dPara);

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("Main Process End."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_CM_DataHIST.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{

			fprintf_s(out, "LowIndex, %d\n", nLowerIndex);
			fprintf_s(out, "LowArea	, %d\n", nLowVArea);
			fprintf_s(out, "TopIndex, %d\n", nUpperIndex);
			fprintf_s(out, "TopArea	, %d\n", nTopVArea);
			fprintf_s(out, "LTDiff	, %d\n", nLowUpDiff);

			fprintf_s(out, "\n");

			fprintf_s(out, "Hist, Total, Resize\n");

			for (int j = 0; j < 256; j++)
			{

				fprintf_s(out, "%d,%f,%f\n", j, matHist_Total.at<float>(j, 0), matHist_Resize.at<float>(j, 0));
			}

			fclose(out);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// G3 Judgement

	int  nCountNumber = pResultBlob->nDefectCount;
	int  nDefectIndex = 0;
	bool bOnlyG3CM = true;

	if (nCountNumber == 0)	bOnlyG3CM = true;
	else					bOnlyG3CM = false;

	// Defect Index Calc.
	if (bOnlyG3CM == true)	 nDefectIndex = 0;
	else if (bOnlyG3CM == false) nDefectIndex = nCountNumber;

	if (nLowUpDiff == 0) return false;

	// Data Check
	if (nLowUpDiff >= nJudge_DiffGV)
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
			pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURA_CLOUD;
			pResultBlob->nPatternClassify[nDefectIndex] = nImageUI;

			//计数增加
			pResultBlob->nDefectCount = nDefectIndex + 1;
		}
		*bFlag = false;
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3 Judgement End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode = E_ERROR_CODE_TRUE;
}

// PNZ ShiftCopy Parameter Check ( 18.10.18 )
long	CInspectMura::ShiftCopyParaCheck(int ShiftValue, int& nCpyX, int& nCpyY, int& nLoopX, int& nLoopY)
{
	if (ShiftValue == 0) return false;

	nCpyX = (int)(ShiftValue / 1000 % 10);	// X方向单元
	nCpyY = (int)(ShiftValue / 100 % 10);	// Y方向Unit
	nLoopX = (int)(ShiftValue / 10 % 10);	// X方向Loop
	nLoopY = (int)(ShiftValue / 1 % 10);	// Y方向Loop

	return E_ERROR_CODE_TRUE;
}

long	CInspectMura::AveragingReducer(cv::Mat& matSrcImage, cv::Mat& matDstImage, CMatBuf* cMemSub)
{
	if (matSrcImage.empty()) return false;

	int nMatWidth = matSrcImage.cols;
	int nMatHeight = matSrcImage.rows;

	int nMatWidth_New = (int)nMatWidth / 2;
	int nMatHeight_New = (int)nMatHeight / 2;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matTemp = cMatBufTemp.GetMat(matSrcImage.size() / 2, CV_8UC1, false);

	matTemp.setTo(0);

	for (int j = 0; j < nMatHeight_New; j++)
	{
		for (int i = 0; i < nMatWidth_New; i++)
		{
			int nData0 = matSrcImage.at<uchar>(j * 2, i * 2);
			int nData1 = matSrcImage.at<uchar>(j * 2 + 1, i * 2);
			int nData2 = matSrcImage.at<uchar>(j * 2, i * 2 + 1);
			int nData3 = matSrcImage.at<uchar>(j * 2 + 1, i * 2 + 1);

			matTemp.at<uchar>(j, i) = (int)((nData0 + nData1 + nData2 + nData3) / (4 + 0.5));

		}
	}

	matTemp.copyTo(matDstImage);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMura::HistAreaCalc(cv::Mat& matSrcImage, int& nLowerIndex, int& nUpperIndex, int& nLowUpDiff, int& nLowValueArea, int& nTopValueArea, double* dPara)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	// Parameter
	double	dbLowTH = (int)dPara[E_PARA_AVI_MURA_G3_MAIN_LOWAREA_TH]; //77;
	double	dbTopTH = (int)dPara[E_PARA_AVI_MURA_G3_MAIN_TOPAREA_TH]; //41;

	double	dbminValue = 0;
	double	dbmaxValue = 0;

	int		nMaxIndex = 0;
	int		nLowIndex = 0;
	int		nTopIndex = 0;
	float	fLowValue = 0;
	float	fTopValue = 0;

	int		nZeroCount = 0;
	int		nLT_Diff = 0;

	cv::minMaxLoc(matSrcImage, &dbminValue, &dbmaxValue);

	float* ptrIndexValue = (float*)matSrcImage.ptr(0);

	for (int i = 0; i < 256; i++, ptrIndexValue++)
	{
		if (*ptrIndexValue == dbmaxValue) { nMaxIndex = i; break; }
	}

	// Low Calc
	float* ptrLowValue = (float*)matSrcImage.ptr(0) + nMaxIndex;

	for (int i = nMaxIndex; i > 0; i--, ptrLowValue--)
	{
		if (*ptrLowValue <= dbLowTH) { nLowIndex = i + 1; fLowValue = matSrcImage.at<float>(nLowIndex, 0); break; }
	}

	// Top Calc
	float* ptrTopValue = (float*)matSrcImage.ptr(0) + nMaxIndex;

	for (int i = nMaxIndex; i < 256; i++, ptrTopValue++)
	{
		if (*ptrTopValue <= dbTopTH) { nTopIndex = i - 1; fTopValue = matSrcImage.at<float>(nTopIndex, 0); break; }
	}

	nLowerIndex = nLowIndex;
	nUpperIndex = nTopIndex;
	nLowUpDiff = nUpperIndex - nLowerIndex;
	nLowValueArea = (int)fLowValue;
	nTopValueArea = (int)fTopValue;

	return E_ERROR_CODE_TRUE;
}

long CInspectMura::LogicStart_MuraG3CM2(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
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
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3_Type2 Start."));

	// Parameters
	int		nResize = (int)dPara[E_PARA_AVI_MURA_G3_CM2_ZOOM];		// 7;

	double	SteDev_B = (double)dPara[E_PARA_AVI_MURA_G3_CM2_STDDEV_BRIGHT];		// 5.0;
	double	SteDev_D = (double)dPara[E_PARA_AVI_MURA_G3_CM2_STDDEV_DARK];		// 7.0;
	//////////////////////////////////////////////////////////////////////////

																																										//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
		// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MN_%02d_G3CM2_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		// 		ImageSave(strTemp, matBKBuffer);
	}

	cv::Mat matGauss;
	cv::GaussianBlur(matSrcImage, matGauss, cv::Size(31, 31), 2.0);

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Gaussian_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauss);
	}

	/////////////////////////////////////预处理过程/////////////////////
	//提取检查区域
	cv::Mat matSrcROI = matGauss(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)).clone();
	//cv::Mat matBKROI = matBKBuffer(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)).clone();
	matGauss.release();

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Active_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
		// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MN_%02d_G3CM2_Active_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		// 		ImageSave(strTemp, matBKROI);
	}

	//1)不良特性是即使图像变小也可以接受的村落形式。
	//为了稍微提高扫描速度,将图像大小减小到与SVI相同的大小	
	cv::resize(matSrcROI, matSrcROI, cv::Size(matSrcROI.cols / nResize, matSrcROI.rows / nResize));
	//cv::resize(matBKROI, matBKROI, cv::Size(matBKROI.cols / nResize, matBKROI.rows / nResize));

		//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Resize_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
		// 		strTemp.Format(_T("%s\\%02d_%02d_%02d_MN_%02d_G3CM2_Resize_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		// 		ImageSave(strTemp, matBKROI);
	}

	//2)将平均GV值调整为128
	//根据图像的中心部分求出平均值,修改图像的GV,使其平均值为128 GV
	//G10图像的亮度太暗了,所以提高了亮度。

	double dAvg = cv::mean(matSrcROI(Rect(matSrcROI.cols / 2 - 50, matSrcROI.rows / 2 - 50, 100, 100)))[0]; // 大小可任意指定

	double dMulti = 128 / dAvg;

	matSrcROI *= dMulti;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Multiply_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	//3)使用亮度均匀度来创建背景
	cv::Mat matBKImage = matSrcROI.clone();

	CSize csSize(matBKImage.cols, matBKImage.rows);
	Flattening(3, (BYTE*)matBKImage.data, csSize);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Flattening_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKImage);
	}

	//4)该不良特性是亮的,因此检测出亮的不良
	//>也有暗不良
	cv::Mat matDff[2];

	cv::subtract(matSrcROI, matBKImage, matDff[0]);	//	明亮的不良
	cv::subtract(matBKImage, matSrcROI, matDff[1]);	//	黑暗不良

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_subtract_Brignt.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDff[0]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_subtract_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDff[1]);
	}

	matBKImage.release();

	// 5) Horizontal Profile
	cv::Mat matHori_Brignt = cv::Mat::zeros(cv::Size(matDff[0].cols, 1), CV_8UC1);
	cv::Mat matHori_Dark = cv::Mat::zeros(cv::Size(matDff[1].cols, 1), CV_8UC1);

	uchar* ptr_B = (uchar*)matHori_Brignt.ptr(0);
	uchar* ptr_D = (uchar*)matHori_Dark.ptr(0);

	for (int i = 0; i < matSrcROI.cols; i++, ptr_B++, ptr_D++)
	{
		*ptr_B = cv::mean(matDff[0](Rect(i, 0, 1, matDff[0].rows)))[0];
		*ptr_D = cv::mean(matDff[1](Rect(i, 0, 1, matDff[1].rows)))[0];
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Profile_Brignt.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matHori_Brignt);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_Profile_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matHori_Dark);
	}
	matDff->release();

	cv::Scalar Mean_B, StdDev_B, Mean_D, StdDev_D;
	cv::meanStdDev(matHori_Brignt, Mean_B, StdDev_B);
	cv::meanStdDev(matHori_Dark, Mean_D, StdDev_D);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type2_DataProfile.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "Bright,,,Dark\n");
			fprintf_s(out, "Mean Gv,%f,,Mean Gv,%f\n", Mean_B[0], Mean_D[0]);
			fprintf_s(out, "StdDev,%f,,StdDev,%f\n", StdDev_B[0], StdDev_D[0]);

			fprintf_s(out, "\n Profile Index\n");

			for (int i = 0; i < matHori_Brignt.cols; i++)
			{
				fprintf_s(out, "%d,%d,,%d,%d\n", i, matHori_Brignt.at<uchar>(0, i), i, matHori_Dark.at<uchar>(0, i));
			}
			fclose(out);
		}
	}
	matHori_Brignt.release();
	matHori_Dark.release();

	//////////////// Judgemunt  /////////////////

	int  nCountNumber = pResultBlob->nDefectCount;
	int  nDefectIndex = 0;
	bool bOnlyG3CM = true;

	if (nCountNumber == 0)	bOnlyG3CM = true;
	else					bOnlyG3CM = false;

	// Defect Index Calc.
	if (bOnlyG3CM == true)	 nDefectIndex = 0;
	else if (bOnlyG3CM == false) nDefectIndex = nCountNumber;

	if (StdDev_B[0] < 0 || StdDev_D[0] < 0) return false;

	// Data Check
	if (StdDev_B[0] >= SteDev_B || StdDev_D[0] >= SteDev_D)
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
			pResultBlob->nDefectJudge[nDefectIndex] = E_DEFECT_JUDGEMENT_MURA_CLOUD;
			pResultBlob->nPatternClassify[nDefectIndex] = nImageUI;

			//计数增加
			pResultBlob->nDefectCount = nDefectIndex + 1;
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3 Type2 Judgement End."));

	return nErrorCode = E_ERROR_CODE_TRUE;
}

long CInspectMura::LogicStart_MuraG3CM3(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, cv::Mat& matDst_Dark, cv::Mat& matDst_Bright, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
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
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3_Type3 Start."));

	// Parameters

	int		nGaussianSize = (int)dPara[E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIZE]; //31;
	double	dGaussianSigma = (double)dPara[E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIGMA]; //2.0;
	int		nGaussianSize2 = (int)dPara[E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIZE2]; // 31;
	double	dGaussianSigma2 = (double)dPara[E_PARA_AVI_MURA_G3_CM3_GAUSSIAN_SIGMA2]; //2.0;

	int		nResize = (int)dPara[E_PARA_AVI_MURA_G3_CM3_ZOOM];		// 7;

	int		nThreshold_D = (int)dPara[E_PARA_AVI_MURA_G3_CM3_THRESHOLD_DARK]; //21;
	int		nThreshold_B = (int)dPara[E_PARA_AVI_MURA_G3_CM3_THRESHOLD_BRIGHT]; //21;

	//根据客户要求添加2022.10.13
	int		nEstiDimX = (int)dPara[E_PARA_AVI_MURA_G3_CM3_ESTIMATION_DIM_X];
	int		nEstiDimY = (int)dPara[E_PARA_AVI_MURA_G3_CM3_ESTIMATION_DIM_Y];
	int		nEstiStepX = (int)dPara[E_PARA_AVI_MURA_G3_CM3_ESTIMATION_STEP_X];
	int		nEstiStepY = (int)dPara[E_PARA_AVI_MURA_G3_CM3_ESTIMATION_STEP_Y];

	double	dEstiBright = dPara[E_PARA_AVI_MURA_G3_CM3_ESTIMATION_BRIGHT];
	double	dEstiDark = dPara[E_PARA_AVI_MURA_G3_CM3_ESTIMATION_DARK];

	//////////////////////////////////////////////////////////////////////////

		//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKBuffer);
	}

	cv::Mat matGauss;
	cv::GaussianBlur(matSrcImage, matGauss, cv::Size(nGaussianSize, nGaussianSize), dGaussianSigma);

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Gaussian_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauss);
	}

	/////////////////////////////////////预处理过程/////////////////////
	//提取检查区域
	cv::Mat matSrcROI = matGauss(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)).clone();
	cv::Mat matBKROI = matBKBuffer(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)).clone();
	matGauss.release();

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Active_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Active_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKROI);
	}

	//1)不良特性是即使图像变小也可以接受的村落形式。
	//为了稍微提高扫描速度,将图像大小减小到与SVI相同的大小	
	int nWidth = matSrcROI.cols;
	int nHight = matSrcROI.rows;

	cv::resize(matSrcROI, matSrcROI, cv::Size(matSrcROI.cols / nResize, matSrcROI.rows / nResize));
	cv::resize(matBKROI, matBKROI, cv::Size(matBKROI.cols / nResize, matBKROI.rows / nResize));

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Resize_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Resize_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKROI);
	}

	//2)将平均GV值调整为128
	//根据图像的中心部分求出平均值,修改图像的GV,使其平均值为128 GV
	//G10图像的亮度太暗了,所以提高了亮度。

	double dAvg = cv::mean(matSrcROI(Rect(matSrcROI.cols / 2 - 50, matSrcROI.rows / 2 - 50, 100, 100)))[0]; // 大小可任意指定

	double dMulti = 128 / dAvg;

	matSrcROI *= dMulti;

	//更改填充轮廓的方式之前
/*cv::Mat matMask = cv::Mat::zeros(matSrcROI.size(), CV_8UC1);
cv::threshold(matSrcROI, matMask, 10, 128, CV_THRESH_BINARY_INV);*/

/*cv::add(matSrcROI, matMask, matSrcROI);*/
//cv::threshold(matMask, matMask, 100, 255, CV_THRESH_BINARY);
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Multiply_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	cv::GaussianBlur(matSrcROI, matSrcROI, cv::Size(nGaussianSize2, nGaussianSize2), dGaussianSigma2);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Gaussian.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);
	//背景画面
	cv::Mat matBKImage = matSrcROI.clone();
	nErrorCode = Estimation_XY2(matSrcROI, matBKImage, nEstiDimX, nEstiDimY, nEstiStepX, nEstiStepY, dEstiBright, dEstiDark, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Estimation.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKImage);
	}

	//减号
	cv::Mat matBuff[E_DEFECT_COLOR_COUNT];
	cv::subtract(matBKImage, matSrcROI, matBuff[E_DEFECT_COLOR_DARK]);// Dark
	cv::subtract(matSrcROI, matBKImage, matBuff[E_DEFECT_COLOR_BRIGHT]);// Bright
	cv::GaussianBlur(matBuff[E_DEFECT_COLOR_DARK], matBuff[E_DEFECT_COLOR_DARK], cv::Size(31, 31), 2.0);
	cv::GaussianBlur(matBuff[E_DEFECT_COLOR_BRIGHT], matBuff[E_DEFECT_COLOR_BRIGHT], cv::Size(31, 31), 2.0);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Subtract_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBuff[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Subtract_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBuff[E_DEFECT_COLOR_BRIGHT]);
	}

	//不良进化
	cv::threshold(matBuff[E_DEFECT_COLOR_DARK], matBuff[E_DEFECT_COLOR_DARK], nThreshold_D, 255, CV_THRESH_BINARY);
	cv::subtract(matBuff[E_DEFECT_COLOR_DARK], matBKROI, matBuff[E_DEFECT_COLOR_DARK]);

	cv::threshold(matBuff[E_DEFECT_COLOR_BRIGHT], matBuff[E_DEFECT_COLOR_BRIGHT], nThreshold_B, 255, CV_THRESH_BINARY);
	cv::subtract(matBuff[E_DEFECT_COLOR_BRIGHT], matBKROI, matBuff[E_DEFECT_COLOR_BRIGHT]);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Threshold_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBuff[E_DEFECT_COLOR_DARK]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type3_Threshold_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBuff[E_DEFECT_COLOR_BRIGHT]);
	}

	cv::resize(matBuff[E_DEFECT_COLOR_DARK], matBuff[E_DEFECT_COLOR_DARK], Size(nWidth, nHight));
	cv::resize(matBuff[E_DEFECT_COLOR_BRIGHT], matBuff[E_DEFECT_COLOR_BRIGHT], Size(nWidth, nHight));

	matBuff[E_DEFECT_COLOR_DARK].copyTo(matDst_Dark(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)));
	matBuff[E_DEFECT_COLOR_BRIGHT].copyTo(matDst_Bright(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)));

	matBuff->release();
	matSrcROI.release();
	/*matMask.release();*/
	matBKImage.release();

	//标签
// 	CFeatureExtraction cFeatureExtraction;
// 	//cFeatureExtraction.SetMem(cMem[0]);
// 	cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_SVI_MURA, m_tInitTime, m_tBeforeTime, m_strAlgLog);
// 
// 	cv::Mat matDrawBuffer = cv::Mat::zeros(matSrcImage.size(), CV_8UC1);
// 
//	//E_DEFECT_COLOR_DARK结果
// 	nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcImage, matDst, matDrawBuffer, rectROI, nCommonPara, E_DEFECT_COLOR_DARK, _T("DP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_RETEST_MURA, FALSE);

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3 Type3 Judgement End."));

	return nErrorCode = E_ERROR_CODE_TRUE;
}

long CInspectMura::LogicStart_MuraG3CM4(cv::Mat& matSrcImage, cv::Mat& matBKBuffer, cv::Mat& matDstImage, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
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
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];
	int		nImageUI = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3_Type4 Start."));

	// Parameters

	// Area
		//2022.10.13客户要求Edge区域全部不同
	/*int		nTBArea = (int)dPara[E_PARA_AVI_MURA_G3_CM4_EDGE_TB_AREA];
	int		nLRArea = (int)dPara[E_PARA_AVI_MURA_G3_CM4_EDGE_LR_AREA];*/
	int		nTArea = (int)dPara[E_PARA_AVI_MURA_G3_CM4_EDGE_T_AREA];
	int		nBArea = (int)dPara[E_PARA_AVI_MURA_G3_CM4_EDGE_B_AREA];
	int		nLArea = (int)dPara[E_PARA_AVI_MURA_G3_CM4_EDGE_L_AREA];
	int		nRArea = (int)dPara[E_PARA_AVI_MURA_G3_CM4_EDGE_R_AREA];

	int		nContrastOffset = (int)dPara[E_PARA_AVI_MURA_G3_CM4_CONTRAST_OFFSET];
	int		nMinimum_Offset = (int)dPara[E_PARA_AVI_MURA_G3_CM4_CONTRAST_MINIMUM];
	int		nMaximum_Offset = (int)dPara[E_PARA_AVI_MURA_G3_CM4_CONTRAST_MAXIMUM];

	int		nGaussianSize = (int)dPara[E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIZE]; //31;
	double	dGaussianSigma = (double)dPara[E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIGMA]; //2.0;
	int		nGaussianSize2 = (int)dPara[E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIZE2]; // 31;
	double	dGaussianSigma2 = (double)dPara[E_PARA_AVI_MURA_G3_CM4_GAUSSIAN_SIGMA2]; //2.0;

	int		nResize = (int)dPara[E_PARA_AVI_MURA_G3_CM4_ZOOM];		// 7;

	int		nThreshold_D = (int)dPara[E_PARA_AVI_MURA_G3_CM4_THRESHOLD_DARK]; //21;

	//根据客户要求添加2022.10.13
	int		nEstiDimX = (int)dPara[E_PARA_AVI_MURA_G3_CM4_ESTIMATION_DIM_X];
	int		nEstiDimY = (int)dPara[E_PARA_AVI_MURA_G3_CM4_ESTIMATION_DIM_Y];
	int		nEstiStepX = (int)dPara[E_PARA_AVI_MURA_G3_CM4_ESTIMATION_STEP_X];
	int		nEstiStepY = (int)dPara[E_PARA_AVI_MURA_G3_CM4_ESTIMATION_STEP_Y];

	double	dEstiBright = dPara[E_PARA_AVI_MURA_G3_CM4_ESTIMATION_BRIGHT];
	double	dEstiDark = dPara[E_PARA_AVI_MURA_G3_CM4_ESTIMATION_DARK];
	//////////////////////////////////////////////////////////////////////////

		//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKBuffer);
	}

	cv::Mat matGauss;
	cv::GaussianBlur(matSrcImage, matGauss, cv::Size(nGaussianSize, nGaussianSize), dGaussianSigma);

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Gaussian_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauss);
	}

	/////////////////////////////////////预处理过程/////////////////////
	//提取检查区域
	cv::Mat matSrcROI = matGauss(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)).clone();
	cv::Mat matBKROI = matBKBuffer(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)).clone();
	matGauss.release();

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Active_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Active_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKROI);
	}

	//1)不良特性是即使图像变小也可以接受的村落形式。
	//为了稍微提高扫描速度,将图像大小减小到与SVI相同的大小	
	int nWidth = matSrcROI.cols;
	int nHight = matSrcROI.rows;

	cv::resize(matSrcROI, matSrcROI, cv::Size(matSrcROI.cols / nResize, matSrcROI.rows / nResize));
	cv::resize(matBKROI, matBKROI, cv::Size(matBKROI.cols / nResize, matBKROI.rows / nResize));

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Resize_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Resize_BKImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKROI);
	}

	//2)将平均GV值调整为128
	//根据图像的中心部分求出平均值,修改图像的GV,使其平均值为128 GV
	//G10图像的亮度太暗了,所以提高了亮度。

	double dAvg = cv::mean(matSrcROI(Rect(matSrcROI.cols / 2 - 50, matSrcROI.rows / 2 - 50, 100, 100)))[0]; // 大小可任意指定

	double dMulti = 128 / dAvg;

	matSrcROI *= dMulti;

	/*cv::Mat matMask = cv::Mat::zeros(matSrcROI.size(), CV_8UC1);
	cv::threshold(matSrcROI, matMask, 10, 128, CV_THRESH_BINARY_INV);
*/
//cv::add(matSrcROI, matMask, matSrcROI);
//cv::threshold(matMask, matMask, 100, 255, CV_THRESH_BINARY_INV);

//cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(5, 5), cv::Point(5 / 2, 5 / 2));
//
	//Morphology减少掩码大小
//cv::morphologyEx(matMask, matMask, MORPH_ERODE, StructElem);
//StructElem.release();

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Multiply_SrcImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);

		/*strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Avtive Mask.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matMask);*/
	}

	cv::GaussianBlur(matSrcROI, matSrcROI, cv::Size(nGaussianSize2, nGaussianSize2), dGaussianSigma2);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Gaussian.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	//Contrast割据
	cv::Mat matContrastImage;
	AlgoBase::Contrast(matSrcROI, matContrastImage, nMinimum_Offset, nMaximum_Offset, nContrastOffset);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Contrast.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matContrastImage);
	}

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);
	//背景画面
	cv::Mat matBKImage = matSrcROI.clone();
	nErrorCode = Estimation_XY2(matContrastImage, matBKImage, nEstiDimX, nEstiDimY, nEstiStepX, nEstiStepY, dEstiBright, dEstiDark, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Estimation.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBKImage);
	}

	//减号
	cv::subtract(matBKImage, matContrastImage, matSrcROI);
	cv::GaussianBlur(matSrcROI, matSrcROI, cv::Size(31, 31), 2.0);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Subtract.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	//只保留Edge区域
//matSrcROI(Rect(nLRArea / nResize, nTBArea / nResize, matSrcROI.cols - nLRArea / nResize *2, matSrcROI.rows - nTBArea / nResize *2)).setTo(0);
	//2022.10.13客户要求Edge区域全部不同
	matSrcROI(Rect(nLArea / nResize, nTArea / nResize, matSrcROI.cols - (nLArea + nRArea) / nResize, matSrcROI.rows - (nTArea + nBArea) / nResize)).setTo(0);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Edge Area.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	//不良进化
	cv::threshold(matSrcROI, matSrcROI, nThreshold_D, 255, CV_THRESH_BINARY);
	cv::subtract(matSrcROI, matBKROI, matSrcROI);
	//cv::bitwise_and(matSrcROI, matMask, matSrcROI);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_%02d_G3_Type4_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	cv::Mat matDst;
	cv::resize(matSrcROI, matDst, Size(nWidth, nHight));

	matDst.copyTo(matDstImage(Rect(rectROI.left, rectROI.top, rectROI.right - rectROI.left, rectROI.bottom - rectROI.top)));

	matDst.release();
	matSrcROI.release();
	/*matMask.release();*/
	matContrastImage.release();
	matBKImage.release();

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("G3 Type4 Judgement End."));

	return nErrorCode = E_ERROR_CODE_TRUE;
}

void CInspectMura::Flattening(int nFlatteningType, BYTE* pImage, CSize szImage, int nMeanGV)
{
	if (!pImage || szImage.cx <= 0 || szImage.cy <= 0) return;

	BYTE* LowGVImage = new BYTE[szImage.cx * szImage.cy];

	FlattenFillLowGV((BYTE*)pImage, szImage, (BYTE*)LowGVImage);

	switch (nFlatteningType)
	{
	case 1:
		FlattenMeanHorizontal(pImage, szImage, nMeanGV);
		break;
	case 2:
		FlattenMeanVertical(pImage, szImage, nMeanGV);
		break;
	case 3:
		FlattenMeanHorizontal(pImage, szImage, nMeanGV);
		FlattenMeanVertical(pImage, szImage, nMeanGV);
		break;
	case 4:
		FlattenMeanVertical(pImage, szImage, nMeanGV);
		FlattenMeanHorizontal(pImage, szImage, nMeanGV);
		break;
	}

	//
	register int nX = 0, nY = 0, nXY = 0;
	for (nY = 0; nY < szImage.cy; nY++)
	{

		nXY = szImage.cx * nY;
		for (nX = 0; nX < szImage.cx; nX++)
			if (LowGVImage[nXY + nX] < 255)
			{
				pImage[nXY + nX] = LowGVImage[nXY + nX];
			}

	}
}

void CInspectMura::FlattenFillLowGV(BYTE* pImage, CSize szImage, BYTE* LowGVImage)
{
	if (!pImage || szImage.cx <= 0 || szImage.cy <= 0) return;

	register int nX = 0, nY = 0, nXY = 0;
	float fSum = 0, fMean = 0;
	float fWidth = 0.0f;

	//求该图像的整体平均值

	fWidth = (float)szImage.cx * (float)szImage.cy;

	fSum = 0;
	for (nY = 0; nY < szImage.cy; nY++)
	{

		nXY = szImage.cx * nY;
		for (nX = 0; nX < szImage.cx; nX++)
			fSum += pImage[nXY + nX];
	}

	fMean = fSum / fWidth;

	for (nY = 0; nY < szImage.cy; nY++)
	{

		nXY = szImage.cx * nY;
		for (nX = 0; nX < szImage.cx; nX++)
			if (pImage[nXY + nX] < 10)		//潮湿的话会很暗
			{
				LowGVImage[nXY + nX] = pImage[nXY + nX];
				pImage[nXY + nX] = (BYTE)fMean;
			}
			else
			{
				LowGVImage[nXY + nX] = (BYTE)255;
			}
	}
}

void CInspectMura::FlattenMeanHorizontal(BYTE* pImage, CSize szImage, int nMeanGV)
{
	if (!pImage || szImage.cx <= 0 || szImage.cy <= 0) return;

	register int nX = 0, nY = 0, nXY = 0;
	float fSum = 0, fMean = 0, fFlatten;
	float fWidth = 0.0f;

	fWidth = (float)szImage.cx;
	for (nY = 0; nY < szImage.cy; nY++)
	{
		fSum = 0;
		nXY = szImage.cx * nY;
		for (nX = 0; nX < szImage.cx; nX++)
			fSum += pImage[nXY + nX];

		fMean = fSum / fWidth;
		for (nX = 0; nX < szImage.cx; nX++)
		{
			fFlatten = pImage[nXY + nX] + (nMeanGV - fMean);

			if (fFlatten < 0) pImage[nXY + nX] = 0;
			else if (fFlatten > 255) pImage[nXY + nX] = 255;
			else pImage[nXY + nX] = (BYTE)fFlatten;
		}
	}
}

void CInspectMura::FlattenMeanVertical(BYTE* pImage, CSize szImage, int nMeanGV)
{
	if (!pImage || szImage.cx <= 0 || szImage.cy <= 0) return;

	register int nX = 0, nY = 0, nXY = 0;
	float fSubX = 0, fFlatten = 0;
	float fHeight = 0.0f;
	float* pfSum = new float[szImage.cx];

	memset(pfSum, 0, sizeof(float) * szImage.cx);

	for (nY = 0; nY < szImage.cy; nY++)
	{
		nXY = szImage.cx * nY;
		for (nX = 0; nX < szImage.cx; nX++)
			pfSum[nX] += pImage[nXY + nX];
	}

	fHeight = (float)szImage.cy;
	for (nX = 0; nX < szImage.cx; nX++)
		pfSum[nX] /= fHeight;

	for (nY = 0; nY < szImage.cy; nY++)
	{
		nXY = szImage.cx * nY;
		for (nX = 0; nX < szImage.cx; nX++)
		{
			fFlatten = pImage[nXY + nX] - pfSum[nX] + nMeanGV;

			if (fFlatten < 0) pImage[nXY + nX] = 0;
			else if (fFlatten > 255) pImage[nXY + nX] = 255;
			else pImage[nXY + nX] = (BYTE)fFlatten;
		}
	}

	delete[] pfSum;
}

long CInspectMura::LogicStart_RingMura(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat& matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//使用参数
	int		nGauSize = (int)dPara[E_PARA_AVI_MURA_RING_GAUSSIAN_SIZE];
	double	dGauSig = dPara[E_PARA_AVI_MURA_RING_GAUSSIAN_SIGMA];

	int nResize = (int)dPara[E_PARA_AVI_MURA_RING_IMAGE_RESIZE]; //7

	int nContrastOffset = (int)dPara[E_PARA_AVI_MURA_RING_CONTRAST_OFFSET];

	int nThreshold_B = (int)dPara[E_PARA_AVI_MURA_RING_THRESHOLD_BRIGHT];//137;
	int nThreshold_D = (int)dPara[E_PARA_AVI_MURA_RING_THRESHOLD_DARK];//125;

	int nMorp = (int)dPara[E_PARA_AVI_MURA_RING_MORPHOLOGY_SIZE]; //9;

	int nDarkArea = (int)dPara[E_PARA_AVI_MURA_RING_DELAREA_DARK]; //10923;
	int nBrightArea = (int)dPara[E_PARA_AVI_MURA_RING_DELAREA_BRIGHT]; //16800;

	int nSmallOffset = (int)dPara[E_PARA_AVI_MURA_RING_DELAREA_SMALLOFFSET]; //500;

	nDarkArea /= (nResize * 2);
	nBrightArea /= (nResize * 2);

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

	//////////////////////////////////////////////////////////////////////////

		//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Input Image.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	//检查区域
	CRect rectTemp(rectROI);
	cv::Mat matSrcROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matSrcImage.type(), false);
	matSrcROIBuf = matSrcImage(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_ActiveArea.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	// Blur
	cv::Mat matGauBuf = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::GaussianBlur(matSrcROIBuf, matGauBuf, cv::Size(nGauSize, nGauSize), dGauSig, dGauSig);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Gaussian.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("GaussianBlur."));

	//减小大小
	cv::resize(matGauBuf, matGauBuf, cv::Size(matGauBuf.cols / nResize, matGauBuf.rows / nResize));
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Resize.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matGauBuf);
	}
	//Contrast割据
	cv::Mat matContrastImage;
	AlgoBase::Contrast(matGauBuf, matContrastImage, 0, 0, nContrastOffset);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Contrast.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matContrastImage);
	}

	//使用Fillter检测不良
	long	nWidth = (long)matContrastImage.cols;	// 图像宽度大小
	long	nHeight = (long)matContrastImage.rows;	// 图像垂直尺寸

	cv::Mat matBuff = cv::Mat::zeros(matContrastImage.size(), CV_8UC1);
	FunFilter(matContrastImage, matBuff, nWidth, nHeight);

	cv::Mat matBright = cv::Mat::zeros(matBuff.size(), CV_8UC1), matDark = cv::Mat::zeros(matBuff.size(), CV_8UC1);;
	FunWhiteMura(matBuff, matBright, nWidth, nHeight, nThreshold_B);

	FunBlackMura(matBuff, matDark, nWidth, nHeight, nThreshold_D);

	//保存中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_FunFilter.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBuff);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_FunWhiteMura.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBright);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_FunBlackMura.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDark);
	}

	//通过mopology运算继承大部分Dark不良部分
	//该林村的特性是亮的部分内有暗的部分,即使进行适当的波罗地运算也没有问题
	//进行mopolo延迟运算的原因是为了通过去除下面的大面积,最大限度地留下不良的东西。

	cv::Mat	StructElem = cv::getStructuringElement(MORPH_ELLIPSE, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

	cv::morphologyEx(matDark, matDark, MORPH_CLOSE, StructElem);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Morphology_Dark.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDark);
	}

	//清除大面积不良
	DeleteArea1_Re(matBright, nBrightArea, &cMatBufTemp);
	DeleteArea1_Re(matDark, nDarkArea, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_DelereArea_Big_Bright.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBright);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_DelereArea_Big_Dark.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDark);
	}

	//消除小面积不良
	DeleteArea1(matBright, nDarkArea - nSmallOffset, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_DelereArea_Small_Bright.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBright);
	}

	//必须查找Dark和Bright重叠的部分
	//只在Bright不良中留下Dark不良
	cv::Mat matDstBuff = cv::Mat::zeros(matBuff.size(), CV_8UC1);
	DarkInBright(matBright, matDark, matDstBuff);
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_DarkInBrightr.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuff);
	}

	cv::morphologyEx(matDstBuff, matDstBuff, MORPH_CLOSE, StructElem);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Morphology.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuff);
	}

	matSrcImage(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	//恢复大小
	cv::resize(matDstBuff, matDstBuff, cv::Size(rectROI.Width(), rectROI.Height()));
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_Ring_Mura_Resize.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuff);
	}

	matDstBuff.copyTo(matDstImage(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height())));

	return nErrorCode;
}

void CInspectMura::FunFilter(cv::Mat& Intput, cv::Mat& Output, int width, int height)
{
	int i, j, k, m;
	long nmax, nmin, nsum;
	double dvalue;
	int widthnew, heightnew;
	int coef[16] = { 2,6,10,15,20,13,14,18,8,2,-2,-3,-5,-5,-3,-2 };
	m = 16;

	widthnew = width + 2 * m;
	heightnew = height + 2 * m;

	//转换为Image Ptr,以后将进行全面更改
	uchar* data_ouput = Output.data;

	//Remove area
	// Mat Buffer = Mat::zeros(widthnew, heightnew, CV_8UC1);
	//uchar *data_Buffer = Buffer.data;

	long* lResizeBuff = new long[widthnew * heightnew];
	long* ldata_buff = new long[widthnew * heightnew];

	//图像大小增加到左,右,上,下16 Pixel(用于Padding的Mask大小)
	FunImageResize(Intput, lResizeBuff, widthnew, heightnew, width, height, m);

	for (j = 1; j < widthnew; j++) lResizeBuff[j] += lResizeBuff[j - 1];

	for (i = 1; i < heightnew; i++) lResizeBuff[i * widthnew] += lResizeBuff[(i - 1) * widthnew];

	for (i = 1; i < heightnew; i++)
		for (j = 1; j < widthnew; j++)
			lResizeBuff[i * widthnew + j] += lResizeBuff[(i - 1) * widthnew + j] + lResizeBuff[i * widthnew + (j - 1)] - lResizeBuff[(i - 1) * widthnew + (j - 1)];

	for (i = m; i < heightnew - m; i++)
		for (j = m; j < widthnew - m; j++)
		{
			nsum = 0;

			for (k = 0; k < m; k++)
				nsum += (long)(coef[k] * (lResizeBuff[(i + k) * widthnew + (j + k)] - lResizeBuff[(i - (k + 1)) * widthnew + (j + k)] - lResizeBuff[(i + k) * widthnew + (j - (k + 1))] + lResizeBuff[(i - (k + 1)) * widthnew + (j - (k + 1))]));

			ldata_buff[i * widthnew + j] = nsum;
			//remove area
			// data_Buffer[i*widthnew + j] = nsum;
		}
	//ImageSave(_T("d:\\Filter_middle.bmp"), Buffer);
	nmax = 250000;
	nmin = -250000;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dvalue = ((double)((ldata_buff[(i + m) * widthnew + (j + m)] - nmin) * 255.0f) / (double)(nmax - nmin));

			if (dvalue < 0.0f)        data_ouput[i * width + j] = (BYTE)0;
			else if (dvalue > 255.0f) data_ouput[i * width + j] = (BYTE)255;
			else	                data_ouput[i * width + j] = (BYTE)dvalue;

		}
	}

	if (lResizeBuff)
		delete[] lResizeBuff;

	if (ldata_buff)
		delete[] ldata_buff;

}

void CInspectMura::FunWhiteMura(cv::Mat& Intput, cv::Mat& Output, int width, int height, int nThres)
{
	int i, j;

	//转换为Image Ptr,以后将进行全面更改
	Mat matTemp = Mat::zeros(Intput.size(), CV_8UC1);
	matTemp = Intput.clone();

	uchar* data_FilImg = matTemp.data;
	uchar* data_Output = Output.data;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			data_Output[i * width + j] = data_FilImg[i * width + j] > nThres ? (BYTE)255 : (BYTE)0;
		}

	//禁用Temp Image内存
	if (!matTemp.empty())				matTemp.release();
}

void CInspectMura::FunBlackMura(cv::Mat& Intput, cv::Mat& Output, int width, int height, int nThres)
{
	int i, j;

	//转换为Image Ptr,以后将进行全面更改
	Mat matTemp = Mat::zeros(Intput.size(), CV_8UC1);
	matTemp = Intput.clone();

	uchar* data_FilImg = matTemp.data;
	uchar* data_Output = Output.data;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			data_Output[i * width + j] = data_FilImg[i * width + j] < nThres ? (BYTE)255 : (BYTE)0;
		}

	//禁用Temp Image内存
	if (!matTemp.empty())				matTemp.release();
}

void CInspectMura::FunImageResize(cv::Mat& Intput, long* lResizeBuff, int widthnew, int heightnew, int width, int height, int m)
{
	int i, j;

	uchar* data_input = Intput.data;

	// [1] copy
	for (i = m; i < heightnew - m; i++)
		for (j = m; j < widthnew - m; j++)
			lResizeBuff[i * widthnew + j] = data_input[(i - m) * width + (j - m)];

	// [2] top,bottom
	for (j = m; j < widthnew - m; j++)
		for (i = 0; i < m; i++)
		{
			lResizeBuff[i * widthnew + j] = data_input[(m - 1 - i) * width + (j - m)];
			lResizeBuff[(m + height + i) * widthnew + j] = data_input[(height - 1 - i) * width + (j - m)];
		}

	// [3] left, right
	for (i = 0; i < heightnew; i++)
		for (j = 0; j < m; j++)
		{
			lResizeBuff[i * widthnew + j] = lResizeBuff[i * widthnew + (2 * m - 1 - j)];
			lResizeBuff[i * widthnew + (m + width + j)] = lResizeBuff[i * widthnew + (m + width - 1 - j)];
		}
}

//删除大面积
long CInspectMura::DeleteArea1_Re(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrcImage.size(), CV_32SC1, false);

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
		if (nArea >= nCount)
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

	return E_ERROR_CODE_TRUE;
}

void CInspectMura::DarkInBright(cv::Mat& Input_Bright, cv::Mat& Input_Dark, cv::Mat& Output)
{
	// Bright
	cv::Mat matLabel_B, matStats_B, matCentroid_B;
	// Dark
	cv::Mat matLabel_D, matStats_D, matCentroid_D;

	//Blob计数
	__int64 nTotalLabel_B = cv::connectedComponentsWithStats(Input_Bright, matLabel_B, matStats_B, matCentroid_B, 8, CV_32S, CCL_GRANA) - 1;
	//Blob计数
	__int64 nTotalLabel_D = cv::connectedComponentsWithStats(Input_Dark, matLabel_D, matStats_D, matCentroid_D, 8, CV_32S, CCL_GRANA) - 1;

	//如果没有个数,请退出
	if (nTotalLabel_B > 0 && nTotalLabel_D > 0)
	{
		//Blob计数
		for (int idx = 1; idx <= nTotalLabel_B; idx++)
		{
			//对象面积
			long nArea = matStats_B.at<int>(idx, CC_STAT_AREA);

			//Blob区域Rect
			cv::Rect rectTemp;
			rectTemp.x = matStats_B.at<int>(idx, CC_STAT_LEFT);
			rectTemp.y = matStats_B.at<int>(idx, CC_STAT_TOP);
			rectTemp.width = matStats_B.at<int>(idx, CC_STAT_WIDTH);
			rectTemp.height = matStats_B.at<int>(idx, CC_STAT_HEIGHT);

			//Blob计数
			for (int idx2 = 1; idx2 <= nTotalLabel_D; idx2++)
			{
				//Dark不良比Bright不良小。
				if (nArea < matStats_D.at<int>(idx2, CC_STAT_AREA))
					continue;

				cv::Rect rectTemp2;
				rectTemp2.x = matStats_D.at<int>(idx2, CC_STAT_LEFT);
				rectTemp2.y = matStats_D.at<int>(idx2, CC_STAT_TOP);
				rectTemp2.width = matStats_D.at<int>(idx2, CC_STAT_WIDTH);
				rectTemp2.height = matStats_D.at<int>(idx2, CC_STAT_HEIGHT);

				//如果Bright ROI中有Dark ROI
				if (rectTemp.x < rectTemp2.x && rectTemp.x + rectTemp.width > rectTemp2.x + rectTemp2.width && rectTemp.y <rectTemp2.y && rectTemp.y + rectTemp.height > rectTemp2.y + rectTemp2.height)
				{
					cv::add(Output(rectTemp), Input_Bright(rectTemp), Output(rectTemp));
					cv::add(Output(rectTemp2), Input_Dark(rectTemp2), Output(rectTemp2));
					break;
				}
			}

		}
	}

}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////