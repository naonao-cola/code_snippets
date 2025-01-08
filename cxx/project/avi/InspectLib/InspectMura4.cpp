
/************************************************************************/
//Mura不良检测相关源
//修改日期:18.05.31
/************************************************************************/

#include "StdAfx.h"
#include "InspectMura4.h"
#include "AlgoBase.h"

CInspectMura4::CInspectMura4(void)
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
CString g1_strPat[E_IMAGE_CLASSIFY_AVI_COUNT] = {
	_T("R"),			// E_IMAGE_CLASSIFY_SVI_R			,	// 00 R
	_T("G"),			// E_IMAGE_CLASSIFY_SVI_G			,	// 01 G
	_T("B"),			// E_IMAGE_CLASSIFY_SVI_B			,	// 02 B
	_T("BLACK"),		// E_IMAGE_CLASSIFY_SVI_BLACK		,	// 03 BLACK
	_T("WHITE"),		// E_IMAGE_CLASSIFY_SVI_WHITE		,	// 04 WHITE
	_T("GRAY32"),		// E_IMAGE_CLASSIFY_SVI_GRAY_32		,	// 06 GRAY_32
	_T("GRAY64"),		// E_IMAGE_CLASSIFY_SVI_GRAY_64		,	// 07 GRAY_64
	_T("GRAY87"),		// E_IMAGE_CLASSIFY_SVI_GRAY_87		,	// 08 GRAY_87
	_T("GRAY128")		// E_IMAGE_CLASSIFY_SVI_GRAY_128	,	// 09 GRAY_128
};

CString g1_strCam[2] = {
	_T("Coaxial"),
	_T("Side")
};

CInspectMura4::~CInspectMura4(void)
{
}

//Main检查算法
long CInspectMura4::DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
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

			writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Copy CV Sub Result."));
		}
		//转交结果
		else
		{
			// 			matBrightBuffer	= matDstImage[E_DEFECT_COLOR_BRIGHT].clone();		//内存分配
			// 			matDarkBuffer	= matDstImage[E_DEFECT_COLOR_DARK].clone();			//内存分配

			matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDarkBuffer);
			matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matBrightBuffer);

			writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Copy Clone Result."));
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

long CInspectMura4::DoFindMuraDefect2(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Mat& matDrawBuffer, wchar_t* strContourTxt)
{
	//如果参数为NULL
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 	return E_ERROR_CODE_EMPTY_PARA;

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Inspect start."));
	////////////////////////////////////////////////////////////////////////// choi 05.13
//	int JudgeSpot_Flag = (int)dPara[E_PARA_AVI_MURA3_GRAY_JUDGE_SPOT_FLAG];
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

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Dst Buf Set."));

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
		break;
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

			writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Copy CV Sub Result."));
		}
		//转交结果
		else
		{
			// 			matBrightBuffer	= matDstImage[E_DEFECT_COLOR_BRIGHT].clone();		//内存分配
			// 			matDarkBuffer	= matDstImage[E_DEFECT_COLOR_DARK].clone();			//内存分配

			matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDarkBuffer);
			matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matBrightBuffer);

			writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Copy Clone Result."));
		}

		//-客户请求
		//白色模式Amorp Dark检测Dust模式中有异物时删除
		//确认结果是用Amorph Dark检测出了异物
		//要在清除灰尘中进行判定,在这里进行注释处理

///////////////////////////// woojin 19.08.28
		CMatBuf cMatBufTemp;
		cMatBufTemp.SetMem(cMem);

		//		//标签

		//		//E_DEFECT_COLOR_bright结果

		// 		case E_IMAGE_CLASSIFY_AVI_GRAY_64:

		// // 			if (JudgeSpot_Flag == 1) {
		///			//重新分类White Spot

		//		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		//		if(!USE_ALG_CONTOURS)	//保存结果轮廓

		//		//绘制结果轮廓

						//取消分配
		matDstImage[E_DEFECT_COLOR_BRIGHT].release();
		matDstImage[E_DEFECT_COLOR_DARK].release();

		writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 End."));

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}
	}

	return nErrorCode;
}

void CInspectMura4::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset)
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
//
//使用OnOff标志设置是否删除Dust
//根据Dust的大小设置是否删除
//复制Mura(G64模式)Dust删除算法
long CInspectMura4::GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
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

	//使用参数
	bool	bFlagW = (dPara[E_PARA_AVI_MURA4_DUST_BRIGHT_FLAG] > 0) ? true : false;
	bool	bFlagD = (dPara[E_PARA_AVI_MURA4_DUST_DARK_FLAG] > 0) ? true : false;
	int		nSize = (int)dPara[E_PARA_AVI_MURA4_DUST_BIG_AREA];
	int		nRange = (int)dPara[E_PARA_AVI_MURA4_DUST_ADJUST_RANGE];
	//int     nWhiteMura_Judge_Flag = (int)dPara[E_PARA_AVI_MURA_GRAY_JUDGE_WHITE_MURA_FLAG];
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

		//只留下大的小灰尘(与去除Mura DUst相反)
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
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA4_Dark_ResThreshold.jpg"), strAlgPath, g1_strPat[nImageNum], g1_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA4_Bright_ResThreshold.jpg"), strAlgPath, g1_strPat[nImageNum], g1_strCam[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA4_Bright_Dust.jpg"), strAlgPath, g1_strPat[nImageNum], g1_strCam[nCamNum], nROINumber, nSaveImageCount++);
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
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstBuffer[E_DEFECT_COLOR_BRIGHT](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
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

		//Dark错误计数
		int nStartIndex = pResultBlob->nDefectCount;

		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

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

		//如果Dust面积较大,则删除
		if (bFlagW)
		{
			//nStartIndex:Dark错误计数后开始
			nErrorCode = DeleteCompareDust(matDustTemp, nRange, pResultBlob, nStartIndex, nModePS);
			if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;
			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteCompareDust-Bright."));
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

long CInspectMura4::LogicStart_SPOT(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{
	int nAlgImgNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int nresize = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_RESIZE];
	int nblur_size = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIZE];
	int nblur_sigma = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BLUR_SIGMA];

	float fBrightRatio_RGB = (float)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_RATIO];
	float fBrightRatio_RGB_Edge = (float)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_BRIGHT_EDGE_RATIO];

	int nSegX = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_SEG_X];
	int nSegY = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_SEG_Y];

	float fDarkRatio_RGB = (float)dPara[E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_RATIO];
	float fDarkRatio_RGB_Edge = (float)dPara[E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_EDGE_RATIO];
	int   nDark_Minimum_Size = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_ACTIVE_DARK_MINIMUM_SIZE];

	float fContrast = (float)dPara[E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_CONTRAST];
	int nInspect_Area = (int)dPara[E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_DARK_INSPECTAREA];
	int nRignt_edge_offset = (float)dPara[E_PARA_AVI_MURA4_WHITE_JUDGE_SPOT_ACTIVE_DARK_RIGHT_EDGE_OFFSET];

	int nMexican_size = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_FILTER_SIZE];
	int nMxblur_size = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIZE];
	int nMxblur_sigma = (int)dPara[E_PARA_AVI_MURA4_GRAY_JUDGE_SPOT_MEXICAN_BLUR_SIGMA];
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
	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot Start."));
	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	//检查区域
	CRect rectTemp(rectROI);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROIBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot Buf Set."));

	int nResizeWidth = matSrcROIBuf.cols / nresize;
	int nResizeHeight = matSrcROIBuf.rows / nresize;
	cv::Mat matResize = cMatBufTemp.GetMat(matSrcROIBuf.size(), matSrcROIBuf.type());

	cv::resize(matSrcROIBuf, matResize, cv::Size(nResizeWidth, nResizeHeight), 3, 3, INTER_AREA);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot Resize."));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResize);		//更改为matResult
	}

	int var_brightness1 = 0;
	double var_contrast1 = fContrast; // 必须转换为变量
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

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Contrast."));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_MURA4_contrast.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, dst1);
	}

	//cv::Mat tt = dst1.clone();
	//matResize = tt.clone();

	cv::GaussianBlur(dst1, matResize, cv::Size(nblur_size, nblur_size), nblur_sigma, nblur_sigma);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Blur."));

	nErrorCode = AlgoBase::C_Mexican_filter(matResize, nMexican_size, nMxblur_size, nMxblur_sigma);
	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot CM Filter."));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_Mexican_filter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResize);
	}

	///////////////////////////
	cv::Mat OutTestmat = cMatBufTemp.GetMat(matResize.size(), matResize.type());
	matResize.copyTo(OutTestmat);

	AlgoBase::MinimumFilter(OutTestmat, OutTestmat, nDark_Minimum_Size);
	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot MinimumFilter"));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_Minimum_filter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, OutTestmat);
	}

	//float	fDarkRatio_RGB = 2.0;
	//float	fBrightRatio_RGB = 1.1;
	//float	fDarkRatio_RGB_Edge = 2.0;
	//float	fBrightRatio_RGB_Edge = 1.2;

	CRect resize_Rect(0, 0, OutTestmat.cols - 1, OutTestmat.rows - 1);
	RangeAvgThreshold_Gray(OutTestmat, OutTestmat, resize_Rect, 1, nSegX, nSegY, fDarkRatio_RGB, fBrightRatio_RGB, fDarkRatio_RGB_Edge, fBrightRatio_RGB_Edge, 0, &cMatBufTemp); //choi 05.01
	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot RangeTH"));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_RANGE_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, OutTestmat);
	}

	cv::morphologyEx(OutTestmat, matResize, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(3, 3)));
	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot Morph"));
	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_Morp.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResize);		//更改为matResult
	}

	for (int i = 0; i < matResize.rows; i++)
	{
		for (int j = 0; j < matResize.cols; j++)
		{
			//清除右侧最外围(在检测黑色不良的内容中,这是否有意义？记不清...)
			if (j > matResize.cols - nRignt_edge_offset)
			{
				matResize.data[i * matResize.cols + j] = 0;

			}
			//设置检查区域...首先暂时将除检查区域设置为0...
			if (j < matResize.cols - (matResize.cols / nInspect_Area))
			{
				matResize.data[i * matResize.cols + j] = 0;

			}
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot Edge Del"));

	cv::resize(matResize, matBrROIBuf, cv::Size(matSrcROIBuf.cols, matSrcROIBuf.rows), 3, 3, INTER_AREA);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot Resize"));

	if (bImageSave)
	{

		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_resize_INV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matBrROIBuf);
	}

	if (!matBkROIBuf.empty())
	{
		cv::subtract(matBrROIBuf, matBkROIBuf, matBrROIBuf);
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA4_%02d_BG_SUB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matBrROIBuf);
	}

	writeInspectLog(E_ALG_TYPE_AVI_MURA_4, __FUNCTION__, _T("Mura4 Logic_Spot BG Sub"));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//保存8bit和12bit画面
long CInspectMura4::ImageSave(CString strPath, cv::Mat matSrcBuf)
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

bool CInspectMura4::OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2)
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

void CInspectMura4::Filter8(BYTE* InImg, BYTE* OutImg, int nMin, int nMax, int width, int height)
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

float* CInspectMura4::diff2Gauss1D(int r) {
	int sz = 2 * r + 1;
	double sigma2 = ((double)r / 3.0 + 1 / 6) * ((double)r / 3.0 + 1 / 6.0);

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

int CInspectMura4::GetBitFromImageDepth(int nDepth)
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

long CInspectMura4::RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat& matDstImage, CRect rectROI,
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

//////////////////////////////////////////////////////////////////////////choi 04.26
bool CInspectMura4::cMeanFilte(cv::Mat matActiveImage, cv::Mat& matDstImage)
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

//删除小面积
long CInspectMura4::DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
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

//如果Dust面积较大,则删除
long CInspectMura4::DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS)
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

	}

	return nErrorCode;
}

