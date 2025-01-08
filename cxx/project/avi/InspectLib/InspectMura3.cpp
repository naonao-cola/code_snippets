
/************************************************************************/
//Mura不良检测相关源
//修改日期:18.05.31
/************************************************************************/

#include "StdAfx.h"
#include "InspectMura3.h"
#include "AlgoBase.h"

CInspectMura3::CInspectMura3(void)
{
	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;

	//////////////////////////////////////////////////////////////////////////
	//sz = 11;
	//////////////////////////////////////////////////////////////////////////
}

CInspectMura3::~CInspectMura3(void)
{
}

//Main检查算法
long CInspectMura3::DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{
	//如果参数为NULL
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 	return E_ERROR_CODE_EMPTY_PARA;

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

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	case E_IMAGE_CLASSIFY_AVI_WHITE:
	{
		//检测出100分
		nErrorCode = LogicStart_SPOT(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);
	}
	break;

	//如果画面号码输入错误。
	default:
		return E_ERROR_CODE_TRUE;
	}

	//如果不是空画面
	if (!matDstImage[E_DEFECT_COLOR_BRIGHT].empty() &&
		!matDstImage[E_DEFECT_COLOR_DARK].empty())
	{
		//移除点亮区域以外的检测(移除倒圆角区域)
		if (!matBKBuffer.empty())
		{
			cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBKBuffer, matBrightBuffer);		//内存分配
			cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDarkBuffer);		//内存分配

			writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Copy CV Sub Result."));
		}
		//转交结果
		else
		{
			// 			matBrightBuffer	= matDstImage[E_DEFECT_COLOR_BRIGHT].clone();		//内存分配
			// 			matDarkBuffer	= matDstImage[E_DEFECT_COLOR_DARK].clone();			//内存分配

			matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDarkBuffer);
			matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matBrightBuffer);

			writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Copy Clone Result."));
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

long CInspectMura3::DoFindMuraDefect2(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Mat& matDrawBuffer, wchar_t* strContourTxt)
{
	//如果参数为NULL
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 	return E_ERROR_CODE_EMPTY_PARA;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Inspect start."));
	////////////////////////////////////////////////////////////////////////// choi 05.13
	int JudgeSpot_Flag = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_FLAG];
	//////////////////////////////////////////////////////////////////////////
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

	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Dst Buf Set."));

	//每个画面的算法不同
	switch (nAlgImgNum)
	{
	case E_IMAGE_CLASSIFY_AVI_R:
	case E_IMAGE_CLASSIFY_AVI_G:
	case E_IMAGE_CLASSIFY_AVI_B:
	{

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

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	case E_IMAGE_CLASSIFY_AVI_WHITE:
	{
		//检测出100分
		nErrorCode = LogicStart_SPOT(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);
	}
	break;

	//如果画面号码输入错误。
	default:
		return E_ERROR_CODE_TRUE;
	}

	//如果不是空画面
	if (!matDstImage[E_DEFECT_COLOR_BRIGHT].empty() &&
		!matDstImage[E_DEFECT_COLOR_DARK].empty())
	{
		//移除点亮区域以外的检测(移除倒圆角区域)
		if (!matBKBuffer.empty())
		{
			cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBKBuffer, matDstImage[E_DEFECT_COLOR_BRIGHT]);		//内存分配
			cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDstImage[E_DEFECT_COLOR_DARK]);		//内存分配

			writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Copy CV Sub Result."));
		}
		//转交结果
// 		else
// 		{
// 			// 			matBrightBuffer	= matDstImage[E_DEFECT_COLOR_BRIGHT].clone();		//内存分配
// 			// 			matDarkBuffer	= matDstImage[E_DEFECT_COLOR_DARK].clone();			//内存分配
// 
// 			matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDarkBuffer);
// 			matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matBrightBuffer);
// 
// 			writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Copy Clone Result."));
// 		}

		///////////////////////////// woojin 19.08.28
		CMatBuf cMatBufTemp;
		cMatBufTemp.SetMem(cMem);

		cv::Rect rectBlobROI;
		rectBlobROI.x = rectROI.left;
		rectBlobROI.y = rectROI.top;
		rectBlobROI.width = rectROI.Width();
		rectBlobROI.height = rectROI.Height();
		//标签
		CFeatureExtraction cFeatureExtraction;
		cFeatureExtraction.SetMem(&cMatBufTemp);
		cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_MURA_3, m_tInitTime, m_tBeforeTime, m_strAlgLog);

		//E_DEFECT_COLOR_bright结果
// choikwangil 12.03 black hole
		switch (nAlgImgNum)
		{
		case E_IMAGE_CLASSIFY_AVI_WHITE:
		{
			//////////////////////////////////////////////////////////////////////////
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstImage[E_DEFECT_COLOR_BRIGHT](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("M3_B"), EngineerBlockDefectJudge, pResultBlob);

			if (JudgeSpot_Flag == 1) {
				//重新分类White Spot
				JudgeWhiteSpot(matSrcBuffer, matDstImage[E_DEFECT_COLOR_BRIGHT], rectROI, dPara, nCommonPara, strAlgPath, pResultBlob, &cMatBufTemp);
				writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("JudgeWhiteSpot."));
			}

			if (!USE_ALG_CONTOURS)	//保存结果轮廓
				cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

			//绘制结果轮廓
			cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
			//////////////////////////////////////////////////////////////////////////

			//////////////////////////////////////////////////////////////////////////
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstImage[E_DEFECT_COLOR_DARK](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_DARK, _T("M3_D"), EngineerBlockDefectJudge, pResultBlob);

			if (!USE_ALG_CONTOURS)	//保存结果轮廓
				cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

			//绘制结果轮廓
			cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
			//////////////////////////////////////////////////////////////////////////
		}
		break;
		case E_IMAGE_CLASSIFY_AVI_GRAY_64:
		{
			//////////////////////////////////////////////////////////////////////////
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstImage[E_DEFECT_COLOR_BRIGHT](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("M3_B"), EngineerBlockDefectJudge, pResultBlob);

			if (JudgeSpot_Flag == 1) {
				//重新分类White Spot
				JudgeWhiteSpot(matSrcBuffer, matDstImage[E_DEFECT_COLOR_BRIGHT], rectROI, dPara, nCommonPara, strAlgPath, pResultBlob, &cMatBufTemp);
				writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("JudgeWhiteSpot."));
			}

			if (!USE_ALG_CONTOURS)	//保存结果轮廓
				cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

			//绘制结果轮廓
			cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
			//////////////////////////////////////////////////////////////////////////

			//////////////////////////////////////////////////////////////////////////
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstImage[E_DEFECT_COLOR_DARK](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_DARK, _T("M3_D"), EngineerBlockDefectJudge, pResultBlob);

			if (!USE_ALG_CONTOURS)	//保存结果轮廓
				cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

			//绘制结果轮廓
			cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
			//////////////////////////////////////////////////////////////////////////
		}
		break;
		}

		//如果使用外围信息,请在Judgement()中保存文件(重复数据删除时,相应的坏外围方案图)//choikwangil04.07添加draw错误修复
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)

		//取消分配
		matDstImage[E_DEFECT_COLOR_BRIGHT].release();
		matDstImage[E_DEFECT_COLOR_DARK].release();

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 End."));

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}

	}

	return nErrorCode;
}

void CInspectMura3::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset)
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

//删除Dust后,转交结果向量
long CInspectMura3::GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL。
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())							return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuffer[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_DARK].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;

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

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("DeleteArea."));
	}
	//没有Dust画面缓冲区
	//没有清除逻辑的不良提取
	else
	{
		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Empty Dust Image."));
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
		cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_MURA_3, m_tInitTime, m_tBeforeTime, m_strAlgLog);

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
		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("DoDefectBlobJudgment-Dark."));

		//Dark错误计数
		int nStartIndex = pResultBlob->nDefectCount;

		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

		//////////////////////////////////////////////////////////////////////////

		BOOL bUse[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];
		//for ()
		//临时选取块1默认为点检测[hjf]
		STRU_DEFECT_ITEM* EngineerDefectJudgment = EngineerBlockDefectJudge[0].stDefectItem;
		//EngineerDefectJudgment = EngineerBlockDefectJudge;

				//如果Spot检查
		if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse || EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse) // 04.16 choi
		{
			//为了一百分,李振华(一百分:255/百村200)
			cv::threshold(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matResBuf, 220, 255.0, CV_THRESH_BINARY);

			//复制区分参数
			for (int p = 0; p < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; p++)
			{
				bUse[p] = EngineerDefectJudgment[p].bDefectItemUse;

				//禁用所有参数检查
				EngineerDefectJudgment[p].bDefectItemUse = false;
			}

			//只设置Spot检查
			EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse = true;
			EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse = true; // 04.16 choi

			//E_DEFECT_COLOR_BRIGHT结果
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matResBuf(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BM_"), EngineerBlockDefectJudge, pResultBlob);
			if (nErrorCode != E_ERROR_CODE_TRUE)
			{
				//禁用内存
				matSrcBuffer.release();
				matDstBuffer[E_DEFECT_COLOR_DARK].release();
				matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

				return nErrorCode;
			}
			writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("DoDefectBlobJudgment-Bright Spot."));

			//重新分类White Spot
//		JudgeWhiteSpot(matSrcBuffer, matDstBuffer[E_DEFECT_COLOR_BRIGHT], rectROI, dPara, nCommonPara, strAlgPath, pResultBlob, &cMatBufTemp);
			writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("JudgeWhiteSpot."));

			//如果您使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围方案图)
			//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
			if (!USE_ALG_CONTOURS)	//保存结果轮廓
				cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

			//绘制结果轮廓
			cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

			//元福
			for (int p = 0; p < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; p++)
			{
				EngineerDefectJudgment[p].bDefectItemUse = bUse[p];
			}
		}

		//////////////////////////////////////////////////////////////////////////

				//为了白村,李振华(白点:255/白村200)
		cv::threshold(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matResBuf, 190, 255.0, CV_THRESH_BINARY);

		//仅禁用Spot检查
		bUse[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT] = EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse;
		bUse[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT] = EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse; //04.16 choi

		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse = false;
		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse = false; //04.16 choi

		//E_DEFECT_COLOR_BRIGHT结果
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matResBuf(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BM_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}
		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("DoDefectBlobJudgment-Bright."));

		//元福
		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT].bDefectItemUse = bUse[E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT];
		EngineerDefectJudgment[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT].bDefectItemUse = bUse[E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT]; // 04.16 choi

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

long CInspectMura3::LogicStart_SPOT(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{
	int nAlgImgNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int nresize = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_RESIZE];
	int nblur_size = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIZE];
	int nblur_sigma = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIGMA];

	int	  nBright_Flag = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_FLAG];
	float fBrightRatio_RGB = (float)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_RATIO];
	float fBrightRatio_RGB_Edge = (float)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_EDGE_RATIO];

	int   nDark_Flag = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_FLAG];
	float fDarkRatio_RGB = (float)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_RATIO];
	float fDarkRatio_RGB_Edge = (float)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_EDGE_RATIO];
	int   nDark_Minimum_Size = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_DARK_MINIMUM_SIZE];

	int nSegX = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SEG_X];
	int nSegY = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SEG_Y];

	int nBright_Mexican_size = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_FILTER_SIZE];
	int nBright_Mxblur_size = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIZE];
	int nBright_Mxblur_sigma = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIGMA];

	int nDark_Mexican_size = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_FILTER_SIZE];
	int nDark_Mxblur_size = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_BLUR_SIZE];
	int nDark_Mxblur_sigma = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MEXICAN_DARK_BLUR_SIGMA];

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
	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot Start."));
	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	//检查区域
	CRect rectTemp(rectROI);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot Buf Set."));

	int nResizeWidth = matSrcROIBuf.cols / nresize;
	int nResizeHeight = matSrcROIBuf.rows / nresize;
	cv::Mat matResize = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type());

	cv::resize(matSrcROIBuf, matResize, cv::Size(nResizeWidth, nResizeHeight), 3, 3, INTER_AREA);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot Resize."));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResize);		//更改为matResult
	}

	cv::GaussianBlur(matResize, matResize, cv::Size(nblur_size, nblur_size), nblur_sigma, nblur_sigma);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot Blur."));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_Gaussian.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResize);		//更改为matResult
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 	C_Mexican_filter(matResize, nMexican_size,nMxblur_size,nMxblur_sigma);
// 
// 	writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot CM Filter."));
	//////////////////////////////////////////////////////////////////////////

// #pragma omp parallel
// 	{
// #pragma omp sections
// 		{
// #pragma omp section
// 			{
	if (nBright_Flag > 0) {

		cv::Mat mat_Br_result = cMatBufTemp.GetMat(matResize.size(), matResize.type());
		matResize.copyTo(mat_Br_result);

		nErrorCode = AlgoBase::C_Mexican_filter(mat_Br_result, nBright_Mexican_size, nBright_Mxblur_size, nBright_Mxblur_sigma);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot CM Filter."));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_Br_Mexican_filter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, mat_Br_result);		//更改为matResult
		}

		CRect resize_Rect(0, 0, mat_Br_result.cols - 1, mat_Br_result.rows - 1);
		RangeAvgThreshold_Gray(mat_Br_result, mat_Br_result, resize_Rect, 1, nSegX, nSegY, fDarkRatio_RGB, fBrightRatio_RGB, fDarkRatio_RGB_Edge, fBrightRatio_RGB_Edge, 1, &cMatBufTemp); //choi 05.01

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot (Bright) RangeTH"));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_WHITE_SPOT_Range_Threshold_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, mat_Br_result);
		}

		cv::resize(mat_Br_result, matBrROIBuf, cv::Size(matSrcROIBuf.cols, matSrcROIBuf.rows), 3, 3, INTER_AREA);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot (Bright) Resize"));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_WHITE_SPOT_Bright_resize_INV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, matBrROIBuf);
		}

	}
	//			}
	// #pragma omp section
	// 			{
	if (nDark_Flag > 0) {

		cv::Mat mat_Dark_result = cMatBufTemp.GetMat(matResize.size(), matResize.type());
		matResize.copyTo(mat_Dark_result);

		nErrorCode = AlgoBase::C_Mexican_filter(mat_Dark_result, nDark_Mexican_size, nDark_Mxblur_size, nDark_Mxblur_sigma);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot CM Filter."));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_Da_Mexican_filter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, mat_Dark_result);		//更改为matResult
		}

		AlgoBase::MinimumFilter(mat_Dark_result, mat_Dark_result, nDark_Minimum_Size);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot (Dark) MinimumFilter"));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_Da_Minimum_filter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, mat_Dark_result);		//更改为matResult
		}

		CRect resize_Rect(0, 0, mat_Dark_result.cols - 1, mat_Dark_result.rows - 1);
		RangeAvgThreshold_Gray(mat_Dark_result, mat_Dark_result, resize_Rect, 1, nSegX, nSegY, fDarkRatio_RGB, fBrightRatio_RGB, fDarkRatio_RGB_Edge, fBrightRatio_RGB_Edge, 0, &cMatBufTemp); //choi 05.01

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot (Dark) RangeTH"));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_BLACK_SPOT_Range_Threshold_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, mat_Dark_result);
		}

		cv::resize(mat_Dark_result, matDaROIBuf, cv::Size(matSrcROIBuf.cols, matSrcROIBuf.rows), 3, 3, INTER_AREA);

		writeInspectLog(E_ALG_TYPE_AVI_MURA_3, __FUNCTION__, _T("Mura3 Logic_Spot (Dark) Resize"));

		if (bImageSave)
		{

			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA3_%02d_BLACK_SPOT_Dark_resize_INV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, matDaROIBuf);
		}

	}
	// 			}
	// 		}
	// 	}
	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//保存8bit和12bit画面
long CInspectMura3::ImageSave(CString strPath, cv::Mat matSrcBuf)
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

bool CInspectMura3::OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2)
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

void CInspectMura3::Filter8(BYTE* InImg, BYTE* OutImg, int nMin, int nMax, int width, int height)
{
	int i, j, k, m;
	long nmax, nmin, nsum;
	double dvalue;

	int coef[8] = { 13,14,20,14,7,-4,-7,-1 };
	//	int coef[5]={48, 22,	20,	-2,	-8};
	//	int coef[3]={12, 1, -3};
	//	int coef[7]={6,	7,	9,	5,	2,	-3,	-2};
	//	int coef[4]={10, 18, -1, -3};
	//int coef[4] = { 42, 14, 7, -7 };
	//	int coef[16]={2, 6, 10, 15, 20, 13, 14, 18, 8, 2, -2, -3, -5,-5, -3, -2};
	//	int coef[26]={ 12,	22,	32,	41,	51,	60,	50,	45,	40,	42,	40,	35,	34,	20,	11,	11,	7,	-8,	-14,	-15,	-11,	-12,	-7,	-10,	-4,	-8};
	nmax = nMax;
	nmin = nMin;
	long* NewImg = new long[(width + 1) * (height + 1)];
	long* NewLoG = new long[(width + 1) * (height + 1)];

	//	memcpy(NewImg,InImg, width*height);
	memset(NewLoG, (long)0, width * height);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			NewImg[(i + 1) * (width + 1) + j + 1] = (long)InImg[i * width + j];
		}
		NewImg[(i + 1) * (width + 1)] = 0; // 这个好像错了。NewImg[i * width ] = 0;  这个好像是对的？
	}
	for (j = 0; j <= width; j++)
		NewImg[j] = 0;

	m = 8;

	width++; // width = width +1;
	height++; // height = height +1;
	for (j = 1; j < width; j++) NewImg[j] += NewImg[j - 1]; // 在第一条横线上加价。要进去的价格都是0,为什么要进去呢？就在上面已经初始化为0了？又在做同样的行为。毫无意义的行为

	for (i = 1; i < height; i++) NewImg[i * width] += NewImg[(i - 1) * width]; // 这也是顶部的NewImg[i*width]=0;这样更改的话,代码就没有意义了。

	for (i = 1; i < height; i++)
		for (j = 1; j < width; j++)
			NewImg[i * width + j] += NewImg[(i - 1) * width + j] + NewImg[i * width + (j - 1)] - NewImg[(i - 1) * width + (j - 1)];

	// 	nmax = -10000000; //不需要的内容
	// 	nmin = 10000000;
			//	 	nmax=-12000; //不需要的内容
		//	 	nmin= 12000;

	for (i = m + 1; i < height - m; i++)
		for (j = m + 1; j < width - m; j++)
		{
			nsum = 0;

			//13,14,20,14,7,-4,-7,-1

			for (k = 0; k < m; k++)
				nsum += (long)(coef[k] * (NewImg[(i + k) * width + (j + k)] - NewImg[(i - (k + 1)) * width + (j + k)] - NewImg[(i + k) * width + (j - (k + 1))] + NewImg[(i - (k + 1)) * width + (j - (k + 1))]));
			//将正方形形状中的第1,3象限和减去第2,4象限和后的值乘以系数替换为第3象限值。

			NewLoG[i * width + j] = nsum;

			if (NewLoG[i * width + j] > nmax) nmax = NewLoG[i * width + j]; //NewLoG[i*width+j] = nmax; 
			if (NewLoG[i * width + j] < nmin) nmin = NewLoG[i * width + j];

		}
	// 
	// 	nmax = 8000; // Min,Max值上升时平均亮度下降(Gray scale)
	// 	nmin = -8000;
		// 		nmax= 10000;
		// 		nmin=-10000;
			//min值和max值可能不同。
	for (i = 1; i < height; i++)
	{
		for (j = 1; j < width; j++)
		{
			if (i >= m && i < height - m && j >= m && j < width - m) {
				dvalue = ((double)((NewLoG[i * width + j] - nmin) * 255) / (double)(nmax - nmin));
				if (dvalue < 0.0f)			OutImg[(i - 1) * (width - 1) + (j - 1)] = (BYTE)0;
				else if (dvalue > 255.0f)	OutImg[(i - 1) * (width - 1) + (j - 1)] = (BYTE)255;
				else					OutImg[(i - 1) * (width - 1) + (j - 1)] = (BYTE)dvalue;
			}
			else						OutImg[(i - 1) * (width - 1) + (j - 1)] = (BYTE)127;

		}
	}

	delete[] NewImg;
	delete[] NewLoG;

}

float* CInspectMura3::diff2Gauss1D(int r) {
	int sz = 2 * r + 1;
	double sigma2 = ((double)r / 3.0 + 1 / 6) * ((double)r / 3.0 + 1 / 6.0);
	//int *num;

	float* kernel;
	kernel = (float*)malloc(sz * sizeof(float));
	//((w^2-r^2)*%e^(-r^2/(2*w^2)))/(2^(3/2)*sqrt(%pi)*w^4*abs(w))
	float sum = 0;
	double PIs = 1 / sqrt(2 * PI * sigma2);
	for (int u = -r; u <= r; u++) {
		double x2 = u * u;
		int idx = u + r;
		kernel[idx] = (float)((x2 - sigma2) * exp(-0.5 * x2 / sigma2) * PIs);

	}
	sum = abs(sum);
	if (sum < 1e-5) sum = 1;
	if (sum != 1) {
		for (int i = 0; i < sz; i++) {
			kernel[i] /= sum;
			//System.out.print(kernel[i] +" ");
		}
	}
	return kernel;
}
int CInspectMura3::GetBitFromImageDepth(int nDepth)
{
	int nRet = -1;

	switch (nDepth)
	{
	case CV_8U:
	case CV_8S:
		nRet = 1;
		break;

	case CV_16U:
	case CV_16S:
		nRet = 2;
		break;

	case CV_32S:
	case CV_32F:
		nRet = 4;
		break;

	case CV_64F:
		nRet = 8;
		break;

	default:
		nRet = -1;
		break;
	}

	return nRet;
}

long CInspectMura3::RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI,
	long nLoop, long nSegX, long nSegY, float fDarkRatio, float fBrightRatio, float fDarkRatio_Edge, float fBrightRatio_Edge, int Defect_Color_mode, CMatBuf* cMemSub)
{
	//如果设置值小于10。
	if (nSegX <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSegY <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long x, y;
	long nStart_X, nStart_Y, nEnd_X, nEnd_Y;

	long nPixelSum, nPixelCount, nPixelAvg;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//Temp内存分配	
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matBlurBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	int nBlur = 5;

	//减少计算量的目的
//cv::Rect rtInspROI;
//rtInspROI.x = rectROI.left - nBlur;
//rtInspROI.y = rectROI.top - nBlur;
//rtInspROI.width = rectROI.Width() + nBlur * 2;
//rtInspROI.height = rectROI.Height() + nBlur * 2;

//Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nBlur);

	if (nLoop > 0)
	{
		cv::blur(matSrcImage, matBlurBuf, cv::Size(nBlur, nBlur));

		if (nLoop > 1)
		{
			// Avg
			for (int i = 1; i < nLoop; i++)
			{
				cv::blur(matBlurBuf, matBlurBuf1, cv::Size(nBlur, nBlur));

				matBlurBuf1.copyTo(matBlurBuf);
			}
		}
	}
	else
	{
		matSrcImage.copyTo(matBlurBuf);
	}
	matBlurBuf1.release();

	//////////////////////////////////////////////////////////////////////////

	for (y = 0; y < nRangeY; y++)
	{
		//计算y范围
		nStart_Y = y * nSegY + rectROI.top;
		if (y == nRangeY - 1)		nEnd_Y = rectROI.bottom;
		else					nEnd_Y = nStart_Y + nSegY;

		for (x = 0; x < nRangeX; x++)
		{
			double dbDarkRatio = fDarkRatio;
			double dbBrightRatio = fBrightRatio;

			//计算x范围
			nStart_X = x * nSegX + rectROI.left;
			if (x == nRangeX - 1)		nEnd_X = rectROI.right;
			else					nEnd_X = nStart_X + nSegX;

			//Edge部分
			if (nStart_X == rectROI.left || y == rectROI.top || nEnd_X == rectROI.right || nEnd_Y == rectROI.bottom)
			{
				dbDarkRatio = fDarkRatio_Edge;
				dbBrightRatio = fBrightRatio_Edge;
			}

			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = nStart_X;
			rectTemp.y = nStart_Y;
			rectTemp.width = nEnd_X - nStart_X + 1;
			rectTemp.height = nEnd_Y - nStart_Y + 1;

			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectTemp);

			//直方图
			cv::Mat matHisto;

			//临时  取消加密库  hjf
			AlgoBase::GetHistogram(matTempBuf, matHisto);

			double dblAverage;
			double dblStdev;

			if (Defect_Color_mode == 1) {
				AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);
			}
			if (Defect_Color_mode == 0) {
				AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 256, dblAverage, dblStdev);
			}

			//设置平均范围
			int nMinGV = (int)(dblAverage - dblStdev);
			int nMaxGV = (int)(dblAverage + dblStdev);
			if (nMinGV < 0)	nMinGV = 0;
			if (nMaxGV > 255)	nMaxGV = 255;

			//初始化
			nPixelSum = 0;
			nPixelCount = 0;
			nPixelAvg = 0;

			//仅按设置的平均范围重新平均
			float* pVal = (float*)matHisto.ptr(0) + nMinGV;

			for (int m = nMinGV; m <= nMaxGV; m++, pVal++)
			{
				nPixelSum += (m * *pVal);
				nPixelCount += *pVal;
			}

			if (nPixelCount == 0)	continue;

			//范围内的平均值
			nPixelAvg = (long)(nPixelSum / (double)nPixelCount);

			//平均*Ratio
			long nDarkTemp = (long)(nPixelAvg * dbDarkRatio);
			long nBrightTemp = (long)(nPixelAvg * dbBrightRatio);

			//Gray有太暗的情况。
			//(平均GV2~3*fBrightRatio->二进制:噪声全部上升)
			if (nBrightTemp < 15)	nBrightTemp = 15;

			//异常处理
			if (nDarkTemp < 0)		nDarkTemp = 0;
			if (nBrightTemp > 255)	nBrightTemp = 255;

			//参数0时异常处理
			if (dbDarkRatio == 0)	nDarkTemp = -1;
			if (dbBrightRatio == 0)	nBrightTemp = 256;

			// E_DEFECT_COLOR_DARK Threshold
			if (Defect_Color_mode == 1) {
				cv::Mat matTempBufT = matDstImage(rectTemp);
				cv::threshold(matTempBuf, matTempBufT, nBrightTemp, 255.0, THRESH_BINARY);
			}

			if (Defect_Color_mode == 0) {
				cv::Mat matTempBufT = matDstImage(rectTemp);
				cv::threshold(matTempBuf, matTempBufT, nDarkTemp, 255.0, THRESH_BINARY_INV);
			}
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

// void CInspectMura3::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset)
// {
// 	rectInspROI.x = rectROI.left - nOffset;
// 	rectInspROI.y = rectROI.top - nOffset;
// 	rectInspROI.width = rectROI.Width() + nOffset * 2;
// 	rectInspROI.height = rectROI.Height() + nOffset * 2;
// 
// 	if (rectInspROI.x < 0) rectInspROI.x = 0;
// 	if (rectInspROI.y < 0) rectInspROI.y = 0;
// 	if (rectInspROI.width > nWidth - rectInspROI.x) rectInspROI.width = nWidth - rectInspROI.x;
// 	if (rectInspROI.height > nHeight - rectInspROI.y) rectInspROI.height = nHeight - rectInspROI.y;
// }

//////////////////////////////////////////////////////////////////////////choi 04.26
bool CInspectMura3::cMeanFilte(cv::Mat matActiveImage, cv::Mat& matDstImage)
{
	try
	{
		cv::Scalar mean, std;
		cv::meanStdDev(matActiveImage, mean, std);
		double sub;
		double com;
#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (int i = 0; i < matActiveImage.rows * matActiveImage.cols; i++) {

			sub = mean[0] - matActiveImage.data[i];
			sub /= 2.0;
			com = matActiveImage.data[i] + sub;
			if (com < 0) { matActiveImage.data[i] = 0; }
			else if (com > 254) { matActiveImage.data[i] = 255; }
			else { matActiveImage.data[i] = com; }
		}
		matActiveImage.copyTo(matDstImage);

		//cv::medianBlur(matActiveImage, matActiveImage, 5)
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		cout << "Exception Caught: " << err_msg << endl;

		return false;
	}

	return true;
}

long CInspectMura3::JudgeWhiteSpot(cv::Mat& matSrcBuffer, cv::Mat& matDstBuffer, CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath,
	stDefectInfo* pResultBlob, CMatBuf* cMemSub)
{
	//如果没有出现故障,请退出
	int nCount = pResultBlob->nDefectCount;
	if (nCount <= 0)		return E_ERROR_CODE_TRUE;

	//使用参数
	int		nMorpObj = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MORP_OBJ];
	int		nMorpBKG = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_MORP_BKG];
	double  dThreshold_Ratio = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_THRESHOLD];

	// Active Spec
	double	dSpecActiveBrightRatio = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_BRIGHT_RATIO];

	//////////////////////////////////////////////////////////////////////////spec1
	int		nSpec1_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC1_FLAG];
	double	dSpecActiveDarkRatio1 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_1];
	double	dSpecActiveDarkArea1_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA1_MIN];
	double	dSpecActiveDarkArea1_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA1_MAX];
	double	dSpecActiveDarkDiff1 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_1];

	//////////////////////////////////////////////////////////////////////////spec2
	int		nSpec2_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC2_FLAG];
	double	dSpecActiveDarkRatio2 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_2];
	double	dSpecActiveDarkArea2_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA2_MIN];
	double	dSpecActiveDarkArea2_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA2_MAX];
	double	dSpecActiveDarkDiff2 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_2];

	//////////////////////////////////////////////////////////////////////////spec3
	int		nSpec3_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC3_FLAG];
	double	dSpecActiveDarkRatio3 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_3];
	double	dSpecActiveDarkArea3_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA3_MIN];
	double	dSpecActiveDarkArea3_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA3_MAX];
	double	dSpecActiveDarkDiff3 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_3];

	//////////////////////////////////////////////////////////////////////////spec4
	int		nSpec4_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC4_FLAG];
	double	dSpecActiveDarkRatio4 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_4];
	double	dSpecActiveDarkArea4_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA4_MIN];
	double	dSpecActiveDarkArea4_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA4_MAX];
	double	dSpecActiveDarkDiff4 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_4];

	//////////////////////////////////////////////////////////////////////////spec5
	int		nSpec5_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC5_FLAG];
	double	dSpecActiveDarkRatio5 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_5];
	double	dSpecActiveDarkArea5_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA5_MIN];
	double	dSpecActiveDarkArea5_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA5_MAX];
	double	dSpecActiveDarkDiff5 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_5];

	//////////////////////////////////////////////////////////////////////////spec6
	int		nSpec6_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC6_FLAG];
	double	dSpecActiveDarkRatio6 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_6];
	double	dSpecActiveDarkArea6_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA6_MIN];
	double	dSpecActiveDarkArea6_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA6_MAX];
	double	dSpecActiveDarkDiff6 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_6];

	//////////////////////////////////////////////////////////////////////////spec7
	int		nSpec7_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC7_FLAG];
	double	dSpecActiveDarkRatio7 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_7];
	double	dSpecActiveDarkArea7_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA7_MIN];
	double	dSpecActiveDarkArea7_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA7_MAX];
	double	dSpecActiveDarkDiff7 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_7];

	//////////////////////////////////////////////////////////////////////////spec8
	int		nSpec8_Act_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_SPEC8_FLAG];
	double	dSpecActiveDarkRatio8 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_RATIO_8];
	double	dSpecActiveDarkArea8_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA8_MIN];
	double	dSpecActiveDarkArea8_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_AREA8_MAX];
	double	dSpecActiveDarkDiff8 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_ACTIVE_SPEC_DARK_DIFF_8];

	//设置Edge区域
	double	dSpecEdgeAreaL = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_LEFT];
	double	dSpecEdgeAreaT = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_TOP];
	double	dSpecEdgeAreaR = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_RIGHT];
	double	dSpecEdgeAreaB = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_AREA_BOTTOM];

	//异常处理
	if (dSpecEdgeAreaL < 0)				dSpecEdgeAreaL = 0;
	if (dSpecEdgeAreaT < 0)				dSpecEdgeAreaT = 0;
	if (dSpecEdgeAreaR < 0)				dSpecEdgeAreaR = 0;
	if (dSpecEdgeAreaB < 0)				dSpecEdgeAreaB = 0;

	// Edge Spec
	double	dSpecEdgeBrightRatio = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_BRIGHT_RATIO];

	//////////////////////////////////////////////////////////////////////////SPEC1
	int		nSpec1_Edge_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC1_FLAG];
	double	dSpecEdgeDarkRatio1 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_1];
	double	dSpecEdgeDarkArea1_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA1_MIN];
	double	dSpecEdgeDarkArea1_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA1_MAX];
	double	dSpecEdgeDarkDiff1 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_1];

	//////////////////////////////////////////////////////////////////////////SPEC2
	int		nSpec2_Edge_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC2_FLAG];
	double	dSpecEdgeDarkRatio2 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_2];
	double	dSpecEdgeDarkArea2_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA2_MIN];
	double	dSpecEdgeDarkArea2_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA2_MAX];
	double	dSpecEdgeDarkDiff2 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_2];

	//////////////////////////////////////////////////////////////////////////SPEC3
	int		nSpec3_Edge_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC3_FLAG];
	double	dSpecEdgeDarkRatio3 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_3];
	double	dSpecEdgeDarkArea3_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA3_MIN];
	double	dSpecEdgeDarkArea3_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA3_MAX];
	double	dSpecEdgeDarkDiff3 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_3];

	//////////////////////////////////////////////////////////////////////////SPEC4
	int		nSpec4_Edge_Flag = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_SPEC4_FLAG];
	double	dSpecEdgeDarkRatio4 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_RATIO_4];
	double	dSpecEdgeDarkArea4_MIN = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA4_MIN];
	double	dSpecEdgeDarkArea4_MAX = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_AREA4_MAX];
	double	dSpecEdgeDarkDiff4 = dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_EDGE_SPEC_DARK_DIFF_4];
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
		if (pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT && pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT) //04.16 choi
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
			// 			CString strTemp;
			// 			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Src_%02d.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i);
			// 			ImageSave(strTemp, matDefectSrcBuf);
			// 
			// 			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Res_%02d.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i);
			// 			ImageSave(strTemp, matDefectResBuf);
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

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cv::Scalar meanORI, stdORI;
		cv::meanStdDev(matDefectSrcBuf, meanORI, stdORI, matDefectMorp1Buf);

		int nTH = meanORI[0] * dThreshold_Ratio;
		//////////////////////////////////////////////////////////////////////////

				//查找亮的部分
		cv::threshold(matDefectSrcBuf, matDefectThBuf, nTH, 255.0, THRESH_BINARY);

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
			// 			CString strTemp;
			// 			strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Bri_%02d.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i);
			// 			ImageSave(strTemp, matDefectThBuf);
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
					// 					CString strTemp;
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_Src_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanObj[0]);
					// 					ImageSave(strTemp, matDefectMorp2Buf);
					// 
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_BK_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanBK[0]);
					// 					ImageSave(strTemp, matDefectBKBuf);
				}
			}
			//明暗点不存在时规格
			else
			{
				//Mura不存在亮GV
				pResultBlob->bMuraBright[i] = false;

				//检测出的不良内容中去除亮部分的画面
				cv::subtract(matDefectMorp1Buf, matDefectThBuf, matDefectMorp2Buf);

				// 				//////////////////////////////////////////////////////////////////////////
				// 				cv:Mat test;
				// 				cv::bitwise_and(matDefectSrcBuf, matDefectMorp2Buf, test);
				// 				cv::threshold(test, matDefectMorp2Buf, pResultBlob->dMeanGV[i]*2.0, 255, THRESH_BINARY);
								//////////////////////////////////////////////////////////////////////////
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
				if (dArea >= dSpecActiveDarkArea1_MIN &&
					dArea < dSpecActiveDarkArea1_MAX &&
					nSpec1_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio1 &&
					dDiff > dSpecActiveDarkDiff1)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec2
				else if (dArea >= dSpecActiveDarkArea2_MIN &&
					dArea < dSpecActiveDarkArea2_MAX &&
					nSpec2_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio2 &&
					dDiff > dSpecActiveDarkDiff2)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec3
				else if (dArea >= dSpecActiveDarkArea3_MIN &&
					dArea < dSpecActiveDarkArea3_MAX &&
					nSpec3_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio3 &&
					dDiff > dSpecActiveDarkDiff3)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec4
				else if (dArea >= dSpecActiveDarkArea4_MIN &&
					dArea < dSpecActiveDarkArea4_MAX &&
					nSpec4_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio4 &&
					dDiff > dSpecActiveDarkDiff4)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec5
				else if (dArea >= dSpecActiveDarkArea5_MIN &&
					dArea < dSpecActiveDarkArea5_MAX &&
					nSpec5_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio5 &&
					dDiff > dSpecActiveDarkDiff5)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec6
				else if (dArea >= dSpecActiveDarkArea6_MIN &&
					dArea < dSpecActiveDarkArea6_MAX &&
					nSpec6_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio6 &&
					dDiff > dSpecActiveDarkDiff6)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec7
				else if (dArea >= dSpecActiveDarkArea7_MIN &&
					dArea < dSpecActiveDarkArea7_MAX &&
					nSpec7_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio7 &&
					dDiff > dSpecActiveDarkDiff7)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec7
				else if (dArea >= dSpecActiveDarkArea8_MIN &&
					dArea < dSpecActiveDarkArea8_MAX &&
					nSpec8_Act_Flag > 0 &&
					dSub > dSpecActiveDarkRatio8 &&
					dDiff > dSpecActiveDarkDiff8)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				else
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT;
				}

				if (bImageSave)
				{
					// 					CString strTemp;
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_Src_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanObj[0]);
					// 					ImageSave(strTemp, matDefectMorp2Buf);
					// 
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Active_BK_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanBK[0]);
					// 					ImageSave(strTemp, matDefectBKBuf);
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
					// 					CString strTemp;
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_Src_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanObj[0]);
					// 					ImageSave(strTemp, matDefectMorp2Buf);
					// 
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_BK_O_%02d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, meanBK[0]);
					// 					ImageSave(strTemp, matDefectBKBuf);
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
				if (dArea >= dSpecEdgeDarkArea1_MIN &&
					dArea < dSpecEdgeDarkArea1_MAX &&
					nSpec1_Edge_Flag > 0 &&
					dSub > dSpecEdgeDarkRatio1 &&
					dDiff > dSpecEdgeDarkDiff1)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec2
				else if (dArea >= dSpecEdgeDarkArea2_MIN &&
					dArea < dSpecEdgeDarkArea2_MAX &&
					nSpec2_Edge_Flag > 0 &&
					dSub > dSpecEdgeDarkRatio2 &&
					dDiff > dSpecEdgeDarkDiff2)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec3
				else if (dArea >= dSpecEdgeDarkArea3_MIN &&
					dArea < dSpecEdgeDarkArea3_MAX &&
					nSpec3_Edge_Flag > 0 &&
					dSub > dSpecEdgeDarkRatio3 &&
					dDiff > dSpecEdgeDarkDiff3)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				// Spec4
				else if (dArea >= dSpecEdgeDarkArea4_MIN &&
					dArea < dSpecEdgeDarkArea4_MAX &&
					nSpec4_Edge_Flag > 0 &&
					dSub > dSpecEdgeDarkRatio4 &&
					dDiff > dSpecEdgeDarkDiff4)
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT;
				}
				else
				{
					pResultBlob->nDefectJudge[i] = E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT;
				}

				if (bImageSave)
				{
					// 					CString strTemp;
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_Src_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanObj[0]);
					// 					ImageSave(strTemp, matDefectMorp2Buf);
					// 
					// 					strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Spot_Edge_BK_X_%02d_Area%d_%.3f.bmp"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++, i, pResultBlob->nArea[i], meanBK[0]);
					// 					ImageSave(strTemp, matDefectBKBuf);
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

/*8bit和12bit画面存储
long CInspectMura3::ImageSave(CString strPath, cv::Mat matSrcBuf)
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
}*/