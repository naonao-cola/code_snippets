
/************************************************************************/
//Mura不良检测相关源
//修改日期:18.05.31
/************************************************************************/

#include "StdAfx.h"
#include "InspectMuraChole.h"
#include"ExportLibrary.h"//使用TIANMA

#define round(fp) (int)((fp) >= 0 ? (fp) + 0.5 : (fp) - 0.5)



// Camera String
CString gg_strCamChole[2] = {
	_T("Coaxial"),
	_T("Side")
};

CInspectMuraChole::CInspectMuraChole(void)
{
	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
}

CInspectMuraChole::~CInspectMuraChole(void)
{
}

//Main检查算法
long CInspectMuraChole::DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Mat& matDrawBuffer)
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
	case E_IMAGE_CLASSIFY_AVI_BLACK:
	case E_IMAGE_CLASSIFY_AVI_VINIT:
	case E_IMAGE_CLASSIFY_AVI_DUST:
	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
		if (nAlgImgNum == E_IMAGE_CLASSIFY_AVI_GRAY_64)
		{
			nErrorCode = LogicStart_CholeMura(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, matCholeBuffer, rectROI, rcCHoleROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

			break;
		}
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	case E_IMAGE_CLASSIFY_AVI_WHITE:
	default:
		return E_ERROR_CODE_TRUE;
	}

	//删除BK图像
	cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBKBuffer, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDstImage[E_DEFECT_COLOR_DARK]);

	//制作成天马算法库,不知道为什么要做这个
// 	matBrightBuffer = matDstImage[E_DEFECT_COLOR_BRIGHT].clone();		//内存分配
// 	matDarkBuffer = matDstImage[E_DEFECT_COLOR_DARK].clone();			//内存分配

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("Copy Clone Result."));

	//取消分配
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
	cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_MURA_CHOLE, m_tInitTime, m_tBeforeTime, m_strAlgLog);

	nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstImage[E_DEFECT_COLOR_BRIGHT](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
		nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("DM_"), EngineerBlockDefectJudge, pResultBlob);

	nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstImage[E_DEFECT_COLOR_DARK](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
		nCommonPara, E_DEFECT_COLOR_DARK, _T("DM_"), EngineerBlockDefectJudge, pResultBlob);

	//绘制结果轮廓
	cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

	matDstImage[E_DEFECT_COLOR_BRIGHT].release();
	matDstImage[E_DEFECT_COLOR_DARK].release();

	return nErrorCode;
}

//删除Dust后,转交结果向量
long CInspectMuraChole::GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
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
	if (matDustBuffer[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	//使用参数
	bool	bFlagW = (dPara[E_PARA_AVI_MURA_CHOLE_DUST_BRIGHT_FLAG] > 0) ? true : false;
	bool	bFlagD = (dPara[E_PARA_AVI_MURA_CHOLE_DUST_DARK_FLAG] > 0) ? true : false;
	int		nSize = (int)dPara[E_PARA_AVI_MURA_CHOLE_DUST_BIG_AREA];
	int		nRange = (int)dPara[E_PARA_AVI_MURA_CHOLE_DUST_ADJUST_RANGE];

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
		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Dark_ResThreshold.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCamChole[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_ResThreshold.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCamChole[nCamNum], nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);

		strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Dust.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCamChole[nCamNum], nROINumber, nSaveImageCount++);
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
			nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("DM_"), EngineerBlockDefectJudge, pResultBlob);

		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}

		//如果使用的是外围信息,Judgement()会保存文件(重复数据删除时,不正确的外围视图)
		//如果禁用,则在Alg端保存文件(即使重复数据删除,其坏轮廓图)
		if (!USE_ALG_CONTOURS)	//保存结果轮廓
			cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt, true);

		//绘制结果轮廓
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);
	}
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

	return nErrorCode;
}

//保存8bit和12bit画面
long CInspectMuraChole::ImageSave(CString strPath, cv::Mat matSrcBuf)
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

// Sub - Threshold ( 16bit )
long CInspectMuraChole::SubThreshold16(cv::Mat& matSrc1Buf, cv::Mat& matSrc2Buf, cv::Mat& matDstBuf, int nThreshold, int nMaxGV)
{
	//如果没有缓冲区。
	if (matSrc1Buf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrc2Buf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (matSrc1Buf.type() != CV_16UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (matSrc2Buf.type() != CV_16UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (matDstBuf.type() != CV_8UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;

	uchar LUT[4096] = { 0, };

	for (int i = 0; i < 4096; i++)
	{
		LUT[i] = (i > nThreshold) ? nMaxGV : 0;
	}

	MatIterator_<ushort> itSrc1, itSrc2, endSrc1;
	itSrc1 = matSrc1Buf.begin<ushort>();
	itSrc2 = matSrc2Buf.begin<ushort>();
	endSrc1 = matSrc1Buf.end<ushort>();
	MatIterator_<uchar> itDst = matDstBuf.begin<uchar>();

	int nSub = 0;
	for (; itSrc1 != endSrc1; itSrc1++, itSrc2++, itDst++)
	{
		nSub = *itSrc1 - *itSrc2;

		//如果值为负值...
		if (nSub < 0)	nSub = 0;

		*itDst = LUT[nSub];
	}

	return E_ERROR_CODE_TRUE;
}

// Threshold ( 16bit )
long CInspectMuraChole::Threshold16(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, int nThreshold, int nMaxGV)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (matSrcBuf.type() != CV_16UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (matDstBuf.type() != CV_8UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;

	uchar LUT[4096] = { 0, };

	for (int i = 0; i < 4096; i++)
	{
		LUT[i] = (i > nThreshold) ? nMaxGV : 0;
	}

	MatIterator_<ushort> itSrc1, endSrc1;
	itSrc1 = matSrcBuf.begin<ushort>();
	endSrc1 = matSrcBuf.end<ushort>();
	MatIterator_<uchar> itDst = matDstBuf.begin<uchar>();

	for (; itSrc1 != endSrc1; itSrc1++, itDst++)
	{
		*itDst = LUT[*itSrc1];
	}

	return E_ERROR_CODE_TRUE;
}

// Threshold ( 16bit )
long CInspectMuraChole::Threshold16_INV(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, int nThreshold, int nMaxGV)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (matSrcBuf.type() != CV_16UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (matDstBuf.type() != CV_8UC1)	return E_ERROR_CODE_MURA_WRONG_PARA;

	uchar LUT[4096] = { 0, };

	for (int i = 0; i < 4096; i++)
	{
		LUT[i] = (i < nThreshold) ? nMaxGV : 0;
	}

	MatIterator_<ushort> itSrc1, endSrc1;
	itSrc1 = matSrcBuf.begin<ushort>();
	endSrc1 = matSrcBuf.end<ushort>();
	MatIterator_<uchar> itDst = matDstBuf.begin<uchar>();

	for (; itSrc1 != endSrc1; itSrc1++, itDst++)
	{
		*itDst = LUT[*itSrc1];
	}

	return E_ERROR_CODE_TRUE;
}

// Pow ( 8bit & 12bit )
long CInspectMuraChole::Pow(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, double dPow, int nMaxGV, CMatBuf* cMemSub)
{
	if (dPow < 1.0)			return E_ERROR_CODE_MURA_WRONG_PARA;
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果没有缓冲区。
	if (matDstBuf.empty())
		matDstBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

	//如果是原始8U
	if (matSrcBuf.type() == CV_8U)
	{
		//如果结果为8U
		if (matDstBuf.type() == CV_8U)
		{
			uchar LUT[256] = { 0, };

			for (int i = 0; i < 256; i++)
			{
				double dVal = pow(i, dPow);
				if (dVal < 0)		dVal = 0;
				if (dVal > 255)	dVal = 255;

				LUT[i] = (uchar)dVal;
			}

			MatIterator_<uchar> itSrc, endSrc, itDst;
			itSrc = matSrcBuf.begin<uchar>();
			endSrc = matSrcBuf.end<uchar>();
			itDst = matDstBuf.begin<uchar>();

			for (; itSrc != endSrc; itSrc++, itDst++)
				*itDst = LUT[(*itSrc)];
		}
		//如果结果为16U
		else
		{
			ushort LUT[4096] = { 0, };

			if (nMaxGV > 4095)	nMaxGV = 4095;

			for (int i = 0; i < 4096; i++)
			{
				double dVal = pow(i, dPow);
				if (dVal < 0)		dVal = 0;
				if (dVal > nMaxGV)	dVal = nMaxGV;

				LUT[i] = (ushort)dVal;
			}

			MatIterator_<uchar> itSrc, endSrc;
			itSrc = matSrcBuf.begin<uchar>();
			endSrc = matSrcBuf.end<uchar>();
			MatIterator_<ushort> itDst = matDstBuf.begin<ushort>();

			for (; itSrc != endSrc; itSrc++, itDst++)
				*itDst = LUT[(*itSrc)];
		}
	}
	//如果是源16U
	else
	{
		ushort LUT[4096] = { 0, };

		for (int i = 0; i < 4096; i++)
		{
			double dVal = pow(i, dPow);
			if (dVal < 0)		dVal = 0;
			if (dVal > 4095)	dVal = 4095;

			LUT[i] = (ushort)dVal;
		}

		MatIterator_<ushort> itSrc, endSrc, itDst;
		itSrc = matSrcBuf.begin<ushort>();
		endSrc = matSrcBuf.end<ushort>();
		itDst = matDstBuf.begin<ushort>();

		for (; itSrc != endSrc; itSrc++, itDst++)
			*itDst = LUT[(*itSrc)];
	}

	return E_ERROR_CODE_TRUE;
}

//删除小面积
long CInspectMuraChole::DeleteArea1(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
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

	//排除在周边数量设置以下
if( nConnectCnt < nCount )	continue;

	//绘制周长
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
long CInspectMuraChole::DeleteArea2(cv::Mat& matSrcImage, int nCount, int nLength, CMatBuf* cMemSub)
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

	return E_ERROR_CODE_TRUE;
}

//两个画面平均
long CInspectMuraChole::TwoImg_Average(cv::Mat matSrc1Buf, cv::Mat matSrc2Buf, cv::Mat& matDstBuf)
{
	if (matSrc1Buf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrc2Buf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())				return E_ERROR_CODE_EMPTY_BUFFER;

	if (matSrc1Buf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;
	if (matSrc2Buf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;

	if (matSrc1Buf.rows != matSrc2Buf.rows ||
		matSrc1Buf.cols != matSrc2Buf.cols)		//应为垂直大小
		return E_ERROR_CODE_IMAGE_SIZE;

	//如果是原始8U
	if (matSrc1Buf.type() == CV_8U)
	{
		cv::MatIterator_<uchar> itSrc1, itSrc2, itDst, endDst;
		itSrc1 = matSrc1Buf.begin<uchar>();
		itSrc2 = matSrc2Buf.begin<uchar>();
		itDst = matDstBuf.begin<uchar>();
		endDst = matDstBuf.end<uchar>();

		for (; itDst != endDst; itSrc1++, itSrc2++, itDst++)
			*itDst = (uchar)((*itSrc1 + *itSrc2) / 2);
	}
	//如果是源16U
	else
	{
		cv::MatIterator_<ushort> itSrc1, itSrc2, itDst, endDst;
		itSrc1 = matSrc1Buf.begin<ushort>();
		itSrc2 = matSrc2Buf.begin<ushort>();
		itDst = matDstBuf.begin<ushort>();
		endDst = matDstBuf.end<ushort>();

		for (; itDst != endDst; itSrc1++, itSrc2++, itDst++)
			*itDst = (ushort)((*itSrc1 + *itSrc2) / 2);
	}

	/************************************************************************
	for (int y=0 ; y<matSrc1Buf.rows ; y++)
	{
	BYTE *ptr1 = (BYTE *)matSrc1Buf.ptr(y);
	BYTE *ptr2 = (BYTE *)matSrc2Buf.ptr(y);
	BYTE *ptr3 = (BYTE *)matDstBuf.ptr(y);

	for (int x=0 ; x<matSrc1Buf.cols ; x++, ptr1++, ptr2++, ptr3++)
	{
	*ptr3 = (BYTE)abs( (*ptr1 + *ptr2) / 2.0 );
	}
	}
	************************************************************************/

	return E_ERROR_CODE_TRUE;
}

//如果Dust面积较大,则删除
long CInspectMuraChole::DeleteCompareDust(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nStartIndex, int nModePS)
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
long CInspectMuraChole::DeleteDarkLine(cv::Mat& matSrcBuffer, float fMajorAxisRatio, CMatBuf* cMemSub)
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

	return nErrorCode;
}

//横向Max GV限制:防止在明点等明亮的不良环境中检测出时发生
long CInspectMuraChole::LimitMaxGV16X(cv::Mat& matSrcBuffer, float fOffset)
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

bool CInspectMuraChole::OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2)
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

//////////////////////////TIANMA CHOLE MURA AREA /////////////////////////////
void CInspectMuraChole::VarianceFilter(cv::Mat src, cv::Mat& dst, int nMaskSize)
{
	Mat prc;
	src.copyTo(prc);

	if (nMaskSize % 2 == 0)
		nMaskSize++;

	int nCols = src.cols;
	int nRows = src.rows;
	int nStep = src.step;

	//	 int nMaskX_ST, nMaskY_ST, nMask_ST;
	int nMaskSizeX, nMaskSizeY;

	//对于Mask Size,将按配方减去
	nMaskSizeX = nMaskSize;
	nMaskSizeY = nMaskSize;

	//掩码
	cv::Mat matFilterMask = cv::Mat::zeros(nMaskSizeX, nMaskSizeY, CV_8UC1);
	nMaskSize = nMaskSize / 2 * -1;

	//画面像素指针访问
	//输入Image
	uchar* ucSrcdata;
	ucSrcdata = prc.data;

	//结果图像
	uchar* ucDstdata;
	ucDstdata = dst.data;

	//Mask Image
	uchar* ucFilterMask;
	ucFilterMask = matFilterMask.data;

	//全影像垂直尺寸
	for (int nY = 0; nY < nRows; nY++)
	{
		//消除最外角区域的错误(画面处理区域默认位于画面中央,无需处理异常区域
		if (nY + nMaskSize < 0 || nY + abs(nMaskSize) > nRows - 1)
			continue;

		//完整画面宽度
		for (int nX = 0; nX < nCols; nX++)
		{

			//消除最外角区域的错误(画面处理区域默认位于画面中心,无需处理异常区域
			if (nX + nMaskSize < 0 || nX + nMaskSize > nCols - 1)
				continue;

			//Mask y Size
			for (int nFy = 0; nFy < nMaskSizeY; nFy++)
			{
				//Mask x Size
				for (int nFx = 0; nFx < nMaskSizeX; nFx++)
				{
					//矩形类型
					ucFilterMask[nFy * nMaskSizeX + nFx] = ucSrcdata[((nY - (nMaskSizeY / 2) + nFy) * nCols) + ((nX - (nMaskSizeX / 2)) + nFx)];
					//条形(横一竖一加)
				}//Mask y Size
			}//Mask x Size

						//获取整个图像的SDV过滤器掩码
			Scalar m, s;
			double dVariance = 0;
			cv::meanStdDev(matFilterMask, m, s);
			dVariance = (double)s[0] * (double)s[0];

			if (dVariance < 0.0f)   ucDstdata[nY * nCols + nX] = (BYTE)0;
			else if (dVariance > 255.0f) ucDstdata[nY * nCols + nX] = (BYTE)255;
			else						ucDstdata[nY * nCols + nX] = (BYTE)dVariance;

		}//整个画面宽度大小结束
	}//结束整个画面垂直大小
}
void CInspectMuraChole::FunFilter(cv::Mat& Intput, cv::Mat& Output, int width, int height)
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
	//ImageSave(_T("E:\\IMTC\\Filter_middle.bmp"), Buffer);
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

void CInspectMuraChole::FunWhiteMura(cv::Mat& Intput, cv::Mat& Output, int width, int height, int nThres)
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

void CInspectMuraChole::FunBlackMura(cv::Mat& Intput, cv::Mat& Output, int width, int height, int nThres)
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

void CInspectMuraChole::FunLineMuraBlack(cv::Mat& Intput, cv::Mat& OutputHorizontal, cv::Mat& OutputVertical, int width, int height, int nThresV, int nThresH, int nThresVEdge, int nThresHEdge)
{
	int i, j;
	long nSum;
	int N;
	int nmin, nmax;
	//	int ny;

	N = 90;

	CRect m_nROI = CRect(3, 3, width - 3, height - 3);

	//转换为Image Ptr,以后将进行全面更改
	Mat matTempHorizontal = Mat::zeros(Intput.size(), CV_8UC1);
	Mat matTempVertical = Mat::zeros(Intput.size(), CV_8UC1);

	Mat matTempHorizontalResize = Mat::zeros(Intput.rows - 5, Intput.cols - 6, CV_8UC1);
	Mat matTempVerticalResize = Mat::zeros(Intput.rows - 6, Intput.cols - 5, CV_8UC1);

	Mat matOutputHorizontalTemp = Mat::zeros(Intput.size(), CV_8UC1);
	Mat matOutputVerticalTemp = Mat::zeros(Intput.size(), CV_8UC1);

	uchar* data_FilImg = Intput.data;
	uchar* data_OutputHorizontal = OutputHorizontal.data;
	uchar* data_OutputVertical = OutputVertical.data;
	uchar* data_matTempHorizontal = matTempHorizontal.data;
	uchar* data_matTempVertical = matTempVertical.data;
	uchar* data_matTempHorizontalResize = matTempHorizontalResize.data;
	uchar* data_matTempVerticalResize = matTempVerticalResize.data;
	uchar* data_OutputHorizontalTemp = matOutputHorizontalTemp.data;
	uchar* data_OutputVerticalTemp = matOutputVerticalTemp.data;

	long* lSumImg = new long[width * height];

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			lSumImg[i * width + j] = data_FilImg[i * width + j];
		}

	for (j = 1; j < width; j++) lSumImg[j] += lSumImg[j - 1];

	for (i = 1; i < height; i++) lSumImg[i * width] += lSumImg[(i - 1) * width];

	for (i = 1; i < height; i++)
		for (j = 1; j < width; j++)
			lSumImg[i * width + j] += lSumImg[(i - 1) * width + j] + lSumImg[i * width + (j - 1)] - lSumImg[(i - 1) * width + (j - 1)];

	//	nmin=22000; //N=60;
	//	nmax=24000;
	//nmin = 32000; //N=90;
	//nmax = 36000;
	//	 nmin = 28000; //N=90;
	//	 nmax = 32000;
	nmin = 22000; //N=90;
	nmax = 48000;
	//	nmin=10000000;
	//	nmax=0;

	//	ny=(m_nROI.top+m_nROI.bottom)/2;

	//  [1] vertical binary image

	for (j = m_nROI.left; j <= m_nROI.right; j++)
		//for(i=m_nROI.top;i<=ny+N;i++)
		for (i = m_nROI.top + 1; i <= m_nROI.bottom - N - 1; i++)
		{
			nSum = lSumImg[(i + N - 1) * width + (j + 1)] - lSumImg[(i - 1) * width + (j + 1)] - lSumImg[(i + N - 1) * width + (j - 2)] + lSumImg[(i - 1) * width + (j - 2)];

			//			if(nSum<nmin) nmin=nSum;
			//			if(nSum>nmax) nmax=nSum;

			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
			data_matTempVertical[i * width + j] = (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f);
		}

	for (j = m_nROI.left; j <= m_nROI.right; j++)
		//for(i=m_nROI.bottom;i>=ny-N;i--)
		for (i = m_nROI.bottom; i >= m_nROI.top + N + 1; i--)
		{
			nSum = lSumImg[i * width + (j + 1)] - lSumImg[(i - N) * width + (j + 1)] - lSumImg[i * width + (j - 2)] + lSumImg[(i - N) * width + (j - 2)];

			//TmpImg[i*width+j]=nMAX(TmpImg[i*width+j],(BYTE)((double)nSum/(double)(3*N)));
			data_matTempVertical[i * width + j] = nMAX(data_matTempVertical[i * width + j], (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f));
			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
		}
	ImageSave(_T("E:\\IMTC\\VW.bmp"), matTempVertical);

	//调整外角区域大小(CRect m_nROI=CRect(3,3,width-3,height-3);语法"3"更改数字时,相应的校正数字已更改无法触摸
	for (i = 0; i <= height - 7; i++)
		for (j = 0; j <= width - 6; j++)
			data_matTempVerticalResize[i * (width - 5) + j] = data_matTempVertical[(i + 4) * width + (j + 3)];

	ImageSave(_T("E:\\IMTC\\VWE.bmp"), matTempVerticalResize);
	cv::resize(matTempVerticalResize, matOutputVerticalTemp, matOutputVerticalTemp.size(), INTER_LINEAR);

	for (i = 0; i <= height - 1; i++)
		for (j = 0; j <= width - 1; j++)
		{
			if (j < 10 || j > width - 10)
			{
				data_OutputVertical[i * width + j] = data_OutputVerticalTemp[i * width + j] > nThresVEdge ? (BYTE)0 : (BYTE)255; //m_nThresV输入的值

			}
			else
			{
				data_OutputVertical[i * width + j] = data_OutputVerticalTemp[i * width + j] > nThresV ? (BYTE)0 : (BYTE)255; //m_nThresV输入的值
			}
		}

	ImageSave(_T("E:\\IMTC\\VWEND.bmp"), OutputVertical);

	//	[2] hori_Binary_Imagez
	for (i = m_nROI.top; i <= m_nROI.bottom; i++)
		for (j = m_nROI.left + 1; j <= m_nROI.right - N - 1; j++)
		{
			nSum = lSumImg[(i + 1) * width + (j + N - 1)] - lSumImg[(i + 1) * width + (j - 1)] - lSumImg[(i - 2) * width + (j + N - 1)] + lSumImg[(i - 2) * width + (j - 1)];

			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
			data_matTempHorizontal[i * width + j] = (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f);
		}

	for (i = m_nROI.top; i <= m_nROI.bottom; i++)
		for (j = m_nROI.right; j >= m_nROI.left + N + 1; j--)
		{
			nSum = lSumImg[(i + 1) * width + j] - lSumImg[(i + 1) * width + (j - N)] - lSumImg[(i - 2) * width + j] + lSumImg[(i - 2) * width + (j - N)];

			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
			//TmpImg[i*width+j]=nMAX(TmpImg[i*width+j],(BYTE)((double)nSum/(double)(3*N)));
			data_matTempHorizontal[i * width + j] = nMAX(data_matTempHorizontal[i * width + j], (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f));
		}
	ImageSave(_T("E:\\IMTC\\XW.bmp"), matTempHorizontal);

	//调整外角区域大小(CRect m_nROI=CRect(3,3,width-3,height-3);语法"3"更改数字时,相应的校正数字已更改无法触摸
	for (i = 0; i <= height - 6; i++)
		for (j = 0; j <= width - 7; j++)
			data_matTempHorizontalResize[i * (width - 6) + j] = data_matTempHorizontal[(i + 3) * width + (j + 4)];

	ImageSave(_T("E:\\IMTC\\XWE.bmp"), matTempHorizontalResize);
	cv::resize(matTempHorizontalResize, matOutputHorizontalTemp, matOutputHorizontalTemp.size(), INTER_LINEAR);

	ImageSave(_T("E:\\IMTC\\XWENDResize.bmp"), matOutputHorizontalTemp);

	for (i = 0; i <= height - 1; i++)
		for (j = 0; j <= width - 1; j++)
		{
			if (i < 10 || i > height - 10)
			{
				data_OutputHorizontal[i * width + j] = data_OutputHorizontalTemp[i * width + j] > nThresHEdge ? (BYTE)0 : (BYTE)255; //  m_nThresH:输入变量
			}
			else
			{
				data_OutputHorizontal[i * width + j] = data_OutputHorizontalTemp[i * width + j] > nThresH ? (BYTE)0 : (BYTE)255; //  m_nThresH:输入变量
			}

		}

	ImageSave(_T("E:\\IMTC\\XWEND.bmp"), OutputHorizontal);

	//禁用Temp Image内存
	if (!matTempHorizontal.empty())					matTempHorizontal.release();
	if (!matTempVertical.empty())						matTempVertical.release();
	if (!matTempHorizontalResize.empty())				matTempHorizontalResize.release();
	if (!matTempVerticalResize.empty())				matTempVerticalResize.release();
	if (!matOutputHorizontalTemp.empty())				matOutputHorizontalTemp.release();
	if (!matOutputVerticalTemp.empty())				matOutputVerticalTemp.release();

	if (lSumImg)
		delete[] lSumImg;
}
void CInspectMuraChole::FunLineMuraWhite(cv::Mat& Intput, cv::Mat& OutputHorizontal, cv::Mat& OutputVertical, int width, int height, int nThresV, int nThresH, int nThresVEdge, int nThresHEdge)
{
	int i, j;
	long nSum;
	constexpr int N = 90;
	int nmin, nmax;
	//	int ny;

	CRect m_nROI = CRect(3, 3, width - 3, height - 3);

	//转换为Image Ptr,以后将进行全面更改
	Mat matTempHorizontal = Mat::zeros(Intput.size(), CV_8UC1);
	Mat matTempVertical = Mat::zeros(Intput.size(), CV_8UC1);

	Mat matTempHorizontalResize = Mat::zeros(Intput.rows - 5, Intput.cols - 6, CV_8UC1);
	Mat matTempVerticalResize = Mat::zeros(Intput.rows - 6, Intput.cols - 5, CV_8UC1);

	Mat matOutputHorizontalTemp = Mat::zeros(Intput.size(), CV_8UC1);
	Mat matOutputVerticalTemp = Mat::zeros(Intput.size(), CV_8UC1);

	uchar* data_FilImg = Intput.data;
	uchar* data_OutputHorizontal = OutputHorizontal.data;
	uchar* data_OutputVertical = OutputVertical.data;
	uchar* data_matTempHorizontal = matTempHorizontal.data;
	uchar* data_matTempVertical = matTempVertical.data;
	uchar* data_matTempHorizontalResize = matTempHorizontalResize.data;
	uchar* data_matTempVerticalResize = matTempVerticalResize.data;
	uchar* data_OutputHorizontalTemp = matOutputHorizontalTemp.data;
	uchar* data_OutputVerticalTemp = matOutputVerticalTemp.data;

	long* lSumImg = new long[width * height];

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			lSumImg[i * width + j] = data_FilImg[i * width + j];
		}

	for (j = 1; j < width; j++) lSumImg[j] += lSumImg[j - 1];

	for (i = 1; i < height; i++) lSumImg[i * width] += lSumImg[(i - 1) * width];

	for (i = 1; i < height; i++)
		for (j = 1; j < width; j++)
			lSumImg[i * width + j] += lSumImg[(i - 1) * width + j] + lSumImg[i * width + (j - 1)] - lSumImg[(i - 1) * width + (j - 1)];

	//	nmin=22000; //N=60;
	//	nmax=24000;
	//nmin = 32000; //N=90;
	//nmax = 36000;
	//	 nmin = 28000; //N=90;
	//	 nmax = 32000;
	nmin = 28000; //N=90;
	nmax = 32000;
	//	nmin=10000000;
	//	nmax=0;

	//	ny=(m_nROI.top+m_nROI.bottom)/2;

	//  [1] vertical binary image

	for (j = m_nROI.left; j <= m_nROI.right; j++)
		//for(i=m_nROI.top;i<=ny+N;i++)
		for (i = m_nROI.top + 1; i <= m_nROI.bottom - N - 1; i++)
		{
			nSum = lSumImg[(i + N - 1) * width + (j + 1)] - lSumImg[(i - 1) * width + (j + 1)] - lSumImg[(i + N - 1) * width + (j - 2)] + lSumImg[(i - 1) * width + (j - 2)];

			//			if(nSum<nmin) nmin=nSum;
			//			if(nSum>nmax) nmax=nSum;

			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
			data_matTempVertical[i * width + j] = (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f);
		}

	for (j = m_nROI.left; j <= m_nROI.right; j++)
		//for(i=m_nROI.bottom;i>=ny-N;i--)
		for (i = m_nROI.bottom; i >= m_nROI.top + N + 1; i--)
		{
			nSum = lSumImg[i * width + (j + 1)] - lSumImg[(i - N) * width + (j + 1)] - lSumImg[i * width + (j - 2)] + lSumImg[(i - N) * width + (j - 2)];

			//TmpImg[i*width+j]=nMAX(TmpImg[i*width+j],(BYTE)((double)nSum/(double)(3*N)));
			data_matTempVertical[i * width + j] = nMAX(data_matTempVertical[i * width + j], (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f));
			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
		}
	ImageSave(_T("E:\\IMTC\\VW.bmp"), matTempVertical);

	//调整外角区域大小(CRect m_nROI=CRect(3,3,width-3,height-3);语法"3"更改数字时,相应的校正数字已更改无法触摸
	for (i = 0; i <= height - 7; i++)
		for (j = 0; j <= width - 6; j++)
			data_matTempVerticalResize[i * (width - 5) + j] = data_matTempVertical[(i + 4) * width + (j + 3)];

	ImageSave(_T("E:\\IMTC\\VWE.bmp"), matTempVerticalResize);
	cv::resize(matTempVerticalResize, matOutputVerticalTemp, matOutputVerticalTemp.size(), INTER_LINEAR);

	for (i = 0; i <= height - 1; i++)
		for (j = 0; j <= width - 1; j++)
		{
			if (j < 10 || j > width - 10)
			{
				data_OutputVertical[i * width + j] = data_OutputVerticalTemp[i * width + j] > nThresVEdge ? (BYTE)0 : (BYTE)255; //m_nThresV输入的值

			}
			else
			{
				data_OutputVertical[i * width + j] = data_OutputVerticalTemp[i * width + j] > nThresV ? (BYTE)0 : (BYTE)255; //m_nThresV输入的值
			}
		}

	ImageSave(_T("E:\\IMTC\\VWEND.bmp"), OutputVertical);

	//	[2] hori_Binary_Imagez
	for (i = m_nROI.top; i <= m_nROI.bottom; i++)
		for (j = m_nROI.left + 1; j <= m_nROI.right - N - 1; j++)
		{
			nSum = lSumImg[(i + 1) * width + (j + N - 1)] - lSumImg[(i + 1) * width + (j - 1)] - lSumImg[(i - 2) * width + (j + N - 1)] + lSumImg[(i - 2) * width + (j - 1)];

			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
			data_matTempHorizontal[i * width + j] = (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f);
		}

	for (i = m_nROI.top; i <= m_nROI.bottom; i++)
		for (j = m_nROI.right; j >= m_nROI.left + N + 1; j--)
		{
			nSum = lSumImg[(i + 1) * width + j] - lSumImg[(i + 1) * width + (j - N)] - lSumImg[(i - 2) * width + j] + lSumImg[(i - 2) * width + (j - N)];

			//TmpImg[i*width+j]=(BYTE)((double)nSum/(double)(3*N));
			//TmpImg[i*width+j]=nMAX(TmpImg[i*width+j],(BYTE)((double)nSum/(double)(3*N)));
			data_matTempHorizontal[i * width + j] = nMAX(data_matTempHorizontal[i * width + j], (BYTE)((double)(nSum - nmin) / (double)(nmax - nmin) * 255.0f));
		}
	ImageSave(_T("E:\\IMTC\\XW.bmp"), matTempHorizontal);

	//调整外角区域大小(CRect m_nROI=CRect(3,3,width-3,height-3);语法"3"更改数字时,相应的校正数字已更改无法触摸
	for (i = 0; i <= height - 6; i++)
		for (j = 0; j <= width - 7; j++)
			data_matTempHorizontalResize[i * (width - 6) + j] = data_matTempHorizontal[(i + 3) * width + (j + 4)];

	ImageSave(_T("E:\\IMTC\\XWE.bmp"), matTempHorizontalResize);
	cv::resize(matTempHorizontalResize, matOutputHorizontalTemp, matOutputHorizontalTemp.size(), INTER_LINEAR);

	ImageSave(_T("E:\\IMTC\\XWENDResize.bmp"), matOutputHorizontalTemp);

	for (i = 0; i <= height - 1; i++)
		for (j = 0; j <= width - 1; j++)
		{
			if (i < 10 || i > height - 10)
			{
				data_OutputHorizontal[i * width + j] = data_OutputHorizontalTemp[i * width + j] > nThresHEdge ? (BYTE)0 : (BYTE)255; //  m_nThresH:输入变量
			}
			else
			{
				data_OutputHorizontal[i * width + j] = data_OutputHorizontalTemp[i * width + j] > nThresH ? (BYTE)0 : (BYTE)255; //  m_nThresH:输入变量
			}

		}

	ImageSave(_T("E:\\IMTC\\XWEND.bmp"), OutputHorizontal);

	//禁用Temp Image内存
	if (!matTempHorizontal.empty())					matTempHorizontal.release();
	if (!matTempVertical.empty())						matTempVertical.release();
	if (!matTempHorizontalResize.empty())				matTempHorizontalResize.release();
	if (!matTempVerticalResize.empty())				matTempVerticalResize.release();
	if (!matOutputHorizontalTemp.empty())				matOutputHorizontalTemp.release();
	if (!matOutputVerticalTemp.empty())				matOutputVerticalTemp.release();

	if (lSumImg)
		delete[] lSumImg;
}
void CInspectMuraChole::FunImageResize(cv::Mat& Intput, long* lResizeBuff, int widthnew, int heightnew, int width, int height, int m)
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

long CInspectMuraChole::LogicStart_CholeMura(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, cv::Mat* matCholeBuffer, CRect rectROI, Rect* rcCHoleROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
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
	//////////////////////////////////////////////////////////////////////////
		//检查参数
	int nCholeROI_X = (int)dPara[E_PARA_AVI_MURA_CHOLE_ROI_X];//1000;
	int nCholeROI_Y = (int)dPara[E_PARA_AVI_MURA_CHOLE_ROI_Y];//1000;

	int nResizeZoom = (int)dPara[E_PARA_AVI_MURA_CHOLE_RESIZEZOOM];//2;

	int nGauSize = (int)dPara[E_PARA_AVI_MURA_CHOLE_GAUSS_SIZE]; //15;
	double dGauSig = (double)dPara[E_PARA_AVI_MURA_CHOLE_GAUSS_SIGMA]; // 2.0;

	double dContrast_Max = (double)dPara[E_PARA_AVI_MURA_CHOLE_CONTRAST_MAX]; //1.02;
	double dContrast_Min = (double)dPara[E_PARA_AVI_MURA_CHOLE_CONTRAST_MIN]; //0.98;
	int nBrightness_Max = (int)dPara[E_PARA_AVI_MURA_CHOLE_BRIGHTNESS_MAX]; //0;
	int nBrightness_Min = (int)dPara[E_PARA_AVI_MURA_CHOLE_BRIGHTNESS_MIN]; //0;

	int nGauSize2 = (int)dPara[E_PARA_AVI_MURA_CHOLE_GAUSS_SIZE2]; //31;
	double dGauSig2 = (double)dPara[E_PARA_AVI_MURA_CHOLE_GAUSS_SIGMA2]; // 5.0;

	double dThRatio_Bright = (double)dPara[E_PARA_AVI_MURA_CHOLE_THRESHOLD_RATIO_WHITE]; // 1.02;
	double dThRatio_Dark = (double)dPara[E_PARA_AVI_MURA_CHOLE_THRESHOLD_RATIO_BLACK]; // 0.95;

	int nMorSize = (int)dPara[E_PARA_AVI_MURA_CHOLE_MORPHOLOGY];
	//
	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole Logic_Spot Start."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	/////////////////////////// PreProcessing Start ////////////////////////////////

		//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	cv::Mat matSrcROI = cMatBufTemp.GetMat(nCholeROI_X, nCholeROI_Y, matSrcImage.type());
	cv::Mat matCholeMask = cMatBufTemp.GetMat(nCholeROI_X, nCholeROI_Y, matSrcImage.type());
	cv::Mat matBGMask = cMatBufTemp.GetMat(nCholeROI_X, nCholeROI_Y, matSrcImage.type());
	//设置检查区域&Chole掩码
	//由于该缺陷在特定位置以特定形状出现,因此可以进行定位检查
	//根据Chole Rect指定区域

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole Logic_Spot Buf Set End."));

	Rect rcInspROI;

	InspectionROI(matSrcImage, matSrcROI, matCholeMask, matCholeBuffer, rcInspROI, rectROI, rcCHoleROI, nCholeROI_X, nCholeROI_Y, matBKBuffer, matBGMask);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole InspRoi Set End."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_SrcROI.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Chole Mask.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matCholeMask);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_BK_Mask.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBGMask);
	}

	//Image Resize(Resize以提高扫描速度)
	//现在只需将图像大小减少一半,然后根据需要获取参数来修改图像大小

	cv::resize(matSrcROI, matSrcROI, cv::Size(nCholeROI_X / nResizeZoom, nCholeROI_Y / nResizeZoom), 0, 0, CV_INTER_NN);

	cv::resize(matCholeMask, matCholeMask, cv::Size(nCholeROI_X / nResizeZoom, nCholeROI_Y / nResizeZoom), 0, 0, CV_INTER_NN);

	cv::resize(matBGMask, matBGMask, cv::Size(nCholeROI_X / nResizeZoom, nCholeROI_Y / nResizeZoom), 0, 0, CV_INTER_NN);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole Resize1 End."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_SrcROI_Resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Chole Mask_Resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matCholeMask);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_BK_Mask_Resize.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matBGMask);
	}

	//消除噪音
	cv::GaussianBlur(matSrcROI, matSrcROI, cv::Size(nGauSize, nGauSize), dGauSig, dGauSig);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole GaussianBlur1 End."));
	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_GaussianBlur.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	//亮度比增加
	Contrast(matSrcROI, dContrast_Max, dContrast_Min, nBrightness_Max, nBrightness_Min);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole Contrast End."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Contrast Image.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	////////////////////////// Main Processing Start ///////////////////////////

	// Image ProFile & Threshold

	cv::Mat matDstBright = cMatBufTemp.GetMat(matSrcROI.size(), CV_8UC1);
	cv::Mat matDstDark = cMatBufTemp.GetMat(matSrcROI.size(), CV_8UC1);

	cv::GaussianBlur(matSrcROI, matSrcROI, cv::Size(nGauSize2, nGauSize2), dGauSig2);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole GaussianBlur2 End."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_GaussianBlur2.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcROI);
	}

	ProfilingThreshold(matSrcROI, matDstBright, matCholeMask, matBGMask, dThRatio_Bright, THRESH_BINARY);
	ProfilingThreshold(matSrcROI, matDstDark, matCholeMask, matBGMask, dThRatio_Dark, THRESH_BINARY_INV);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole ProfileTH End."));
	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Defect_Bright_zoom.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBright);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Defect_Dark_zoom.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstDark);
	}

	cv::morphologyEx(matDstBright, matDstBright, MORPH_CLOSE, Mat(nMorSize, nMorSize, CV_8UC1, Scalar(1)));
	cv::morphologyEx(matDstDark, matDstDark, MORPH_CLOSE, Mat(nMorSize, nMorSize, CV_8UC1, Scalar(1)));

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole MorphologyEx End."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Morphology_Bright_zoom.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBright);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Morphology_Dark_zoom.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstDark);
	}

	// Image ReSize

	cv::resize(matDstBright, matDstBright, cv::Size(nCholeROI_X, nCholeROI_Y), 0, 0, CV_INTER_NN);
	cv::resize(matDstDark, matDstDark, cv::Size(nCholeROI_X, nCholeROI_Y), 0, 0, CV_INTER_NN);

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole Resize2 End."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Defect_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBright);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Defect_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstDark);
	}

	matDstBright.copyTo(matDstImage[E_DEFECT_COLOR_BRIGHT](rcInspROI));
	matDstDark.copyTo(matDstImage[E_DEFECT_COLOR_DARK](rcInspROI));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Bright_Final.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_%02d_Dark_Final.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
	}

	matSrcROI.release();
	matCholeMask.release();
	matDstBright.release();
	matDstDark.release();

	writeInspectLog(E_ALG_TYPE_AVI_MURA_CHOLE, __FUNCTION__, _T("MuraChole Logic_Spot End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

// Pow ( 8bit & 12bit )
long CInspectMuraChole::Pow(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, double dPow)
{
	//	if (dPow < 1.0)			return E_ERROR_CODE_MURA_WRONG_PARA;
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果没有缓冲区。
	if (matDstBuf.empty())
		matDstBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

	//如果是原始8U
	if (matSrcBuf.type() == CV_8U)
	{
		//如果结果为8U
		if (matDstBuf.type() == CV_8U)
		{
			uchar LUT[256] = { 0, };

			for (int i = 0; i < 256; i++)
			{
				double dVal = pow(i, dPow);
				if (dVal < 0)		dVal = 0;
				if (dVal > 255)	dVal = 255;

				LUT[i] = (uchar)dVal;
			}

			MatIterator_<uchar> itSrc, endSrc, itDst;
			itSrc = matSrcBuf.begin<uchar>();
			endSrc = matSrcBuf.end<uchar>();
			itDst = matDstBuf.begin<uchar>();

			for (; itSrc != endSrc; itSrc++, itDst++)
				*itDst = LUT[(*itSrc)];
		}
		//如果结果为16U
		else
		{
			ushort LUT[4096] = { 0, };

			for (int i = 0; i < 4096; i++)
			{
				double dVal = pow(i, dPow);
				if (dVal < 0)		dVal = 0;
				if (dVal > 4095)	dVal = 4095;

				LUT[i] = (ushort)dVal;
			}

			MatIterator_<uchar> itSrc, endSrc;
			itSrc = matSrcBuf.begin<uchar>();
			endSrc = matSrcBuf.end<uchar>();
			MatIterator_<ushort> itDst = matDstBuf.begin<ushort>();

			for (; itSrc != endSrc; itSrc++, itDst++)
				*itDst = LUT[(*itSrc)];
		}
	}
	//如果是源16U
	else
	{
		ushort LUT[4096] = { 0, };

		for (int i = 0; i < 4096; i++)
		{
			double dVal = pow(i, dPow);
			if (dVal < 0)		dVal = 0;
			if (dVal > 4095)	dVal = 4095;

			LUT[i] = (ushort)dVal;
		}

		MatIterator_<ushort> itSrc, endSrc, itDst;
		itSrc = matSrcBuf.begin<ushort>();
		endSrc = matSrcBuf.end<ushort>();
		itDst = matDstBuf.begin<ushort>();

		for (; itSrc != endSrc; itSrc++, itDst++)
			*itDst = LUT[(*itSrc)];
	}

	return E_ERROR_CODE_TRUE;
}

void CInspectMuraChole::MaximumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize)
{
	Mat prc;
	src.copyTo(prc);

	if (nMaskSize % 2 == 0)
		nMaskSize++;

	int nCols = src.cols;
	int nRows = src.rows;
	int nStep = src.step;

	int nMaskX_ST, nMaskY_ST, nMask_ST;

	nMask_ST = nMaskSize / 2 * -1;
	int nMax;

	uchar* ucSrcdata;
	uchar* ucDstdata;

	for (int nY = 0; nY < nRows; nY++)
	{
		if (nY + nMask_ST < 0 || nY + abs(nMask_ST) > nRows - 1)
			continue;
		ucDstdata = prc.ptr(nY);
		for (int nX = 0; nX < nCols; nX++)
		{
			if (nX + nMask_ST < 0 || nX + nMask_ST > nCols - 1)
				continue;

			nMax = 0;

			nMaskY_ST = nMask_ST;
			for (int nMaskY = 0; nMaskY < nMaskSize; nMaskY++)
			{
				int nY_ = nY + nMaskY_ST;
				ucSrcdata = src.ptr(nY_);

				nMaskX_ST = nMask_ST;
				for (int nMaskX = 0; nMaskX < nMaskSize; nMaskX++)
				{
					int nX_ = nX + nMaskX_ST;
					if ((int)ucSrcdata[nX_] > nMax)
						nMax = (int)ucSrcdata[nX_];
					nMaskX_ST++;
				}
				nMaskY_ST++;
			}
			ucDstdata[nX] = (uchar)nMax;
		}
	}

	prc.copyTo(dst);
}

//指定检查区域和创建Chole掩码的函数
void CInspectMuraChole::InspectionROI(cv::Mat& matSrcImage, cv::Mat& matSrcROI, cv::Mat& matCholeMask, cv::Mat* matCholeBuffer, Rect& rcInspROI, CRect rectROI, Rect* rcCHoleROI, int nCholeROI_X, int nCholeROI_Y, cv::Mat& matBKBuffer, cv::Mat& matBGMask)
{
	//如果认为Chole无条件只有一个而创建的算法成为后续问题,则需要修改

	//查找Chole中心
	cv::Point ptCHoleCenter;
	int nStartx = 0, nStarty = 0;
	int nCHoleraius = 0;

	cv::Mat matChole_Temp;
	Rect rectChol;

	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		int nArea = 0;
		if (nArea < rcCHoleROI[i].area())
		{
			ptCHoleCenter.x = rcCHoleROI[i].x + (rcCHoleROI[i].width / 2);
			ptCHoleCenter.y = rcCHoleROI[i].y + (rcCHoleROI[i].height / 2);

			rectChol.x = rcCHoleROI[i].x;
			rectChol.y = rcCHoleROI[i].y;
			rectChol.width = rcCHoleROI[i].width;
			rectChol.height = rcCHoleROI[i].height;

			// 			if (rcCHoleROI[i].width < rcCHoleROI[i].height)
			// 				nCHoleraius = rcCHoleROI[i].width / 3;
			// 			else
			// 				nCHoleraius = rcCHoleROI[i].height / 3;

			nArea = rcCHoleROI[i].area();
		}

		//2021.02.26 Chole大小与Chole In相同
		if (!matCholeBuffer[i].empty())
		{
			matCholeBuffer[i].copyTo(matChole_Temp);
		}
	}

	//计算检查区域的起始坐标
	nStartx = ptCHoleCenter.x - (nCholeROI_X / 2);
	if (nStartx < rectROI.left)
		nStartx = rectROI.left;

	nStarty = ptCHoleCenter.y - (nCholeROI_Y / 2);
	if (nStarty < rectROI.top)
		nStarty = rectROI.top;

	rcInspROI.x = nStartx;
	rcInspROI.y = nStarty;
	rcInspROI.width = nCholeROI_X;
	rcInspROI.height = nCholeROI_Y;

	matSrcImage(rcInspROI).copyTo(matSrcROI);

	cv::Mat matGholeBuff = cv::Mat::zeros(matSrcImage.size(), matSrcImage.type());

	//创建Chole Mask

	cv::Mat matSrcBuf = cv::Mat::zeros(matSrcImage.size(), matSrcImage.type());

	//cv::circle(matSrcBuf, ptCHoleCenter, nCHoleraius, cv::Scalar(255), -1);

	matChole_Temp.copyTo(matSrcBuf(rectChol));

	matSrcBuf(rcInspROI).copyTo(matCholeMask);

	//创建Back Ground Mask

	matBKBuffer(rcInspROI).copyTo(matBGMask);

	matChole_Temp.release();
	matSrcBuf.release();
}

//亮度对比度增加
void CInspectMuraChole::Contrast(cv::Mat& matSrcImage, double dContrast_Max, double dContrast_Min, int nBrightness_Max, int nBrightness_Min)
{

	double dAvg = cv::mean(matSrcImage, matSrcImage > 0)[0];

	int nMax = (int)(dAvg * (dContrast_Max));
	int nMin = (int)(dAvg * (dContrast_Min));
	if (nMax > 255)	nMax = 240;
	if (nMin < 0)	nMin = 0;

	float LUT[256] = { 0, };
	for (int i = 0; i < 256; i++)
	{
		if (i < nMin)		LUT[i] = nMin;
		else if (i > nMax)	LUT[i] = nMax;
		else				LUT[i] = i;
	}

	//宣布为Vec3f型会孤独吗？...提问吧...
	cv::MatIterator_<uchar> itSrc, endSrc;
	itSrc = matSrcImage.begin<uchar>();
	endSrc = matSrcImage.end<uchar>();

	for (; itSrc != endSrc; itSrc++)
	{
		(*itSrc) = LUT[((uchar)*itSrc)];
	}

	int nSize = 5;
	cv::blur(matSrcImage, matSrcImage, cv::Size(nSize, nSize));

	///////////////////////////////////////////////////////
	// Min, Max

	nMax += nBrightness_Max;
	nMin -= nBrightness_Min;

	double dVal = 255.0 / (nMax - nMin);
	cv::subtract(matSrcImage, nMin, matSrcImage);
	cv::multiply(matSrcImage, dVal, matSrcImage);

}

// ProFiling & Threshold
void CInspectMuraChole::ProfilingThreshold(cv::Mat& matSrcImage, cv::Mat& matDstBuf, cv::Mat& matCholeMask, cv::Mat& matBGMask, double dThresholdRatio, int nType)
{
	//使用Chole Mask查找Chole坐标

	cv::Mat matLabels, stats, centroids;
	int numOfLables;

	numOfLables = connectedComponentsWithStats(matCholeMask, matLabels, stats, centroids, 8, CV_32S, CCL_GRANA);

	Rect rectHoleROI(0, 0, 0, 0);
	int nArea = 0;
	for (int j = 1; j < numOfLables; j++)
	{
		int iArea = stats.at<int>(j, CC_STAT_AREA);

		if (nArea < iArea)
		{
			rectHoleROI.x = stats.at<int>(j, CC_STAT_LEFT);
			rectHoleROI.y = stats.at<int>(j, CC_STAT_TOP);
			rectHoleROI.width = stats.at<int>(j, CC_STAT_WIDTH);
			rectHoleROI.height = stats.at<int>(j, CC_STAT_HEIGHT);
		}
	}

	//设置ProFile区域

	Rect rcProX(0, 0, rectHoleROI.x, matSrcImage.rows);
	Rect rcProY(0, rectHoleROI.y + rectHoleROI.height, matSrcImage.cols, matSrcImage.rows - (rectHoleROI.y + rectHoleROI.height));

	// BG + Chole Mask
	cv::Mat matMask;
	cv::add(matBGMask, matCholeMask, matMask);
	matMask = ~matMask;

	//matMask(Rect(0, rectHoleROI.y + rectHoleROI.height, rectHoleROI.x, matMask.rows -(rectHoleROI.y + rectHoleROI.height))).setTo(0);

	// Profile
	cv::Mat matProX = cv::Mat::zeros(Size(1, matSrcImage.rows), CV_8UC1);
	cv::Mat matProY = cv::Mat::zeros(Size(matSrcImage.cols, 1), CV_8UC1);

	// 	uchar *ptr = (uchar *)matProX.ptr(0);
	// 	for (int i = 0; i < matSrcImage.rows; i++, ptr++)
	// 	{
	// 		*ptr = (uchar)(cv::mean(matSrcImage(rcProX).row(i), (matBGMask)(rcProX))[0]);
	// 	
	// 	}
	// 	
	// 	ptr = (uchar *)matProY.ptr(0);
	// 	for (int i = 0; i < matSrcImage.cols; i++, ptr++)
	// 	{
	// 		*ptr = (uchar)(cv::mean(matSrcImage(rcProY).col(i))[0]);
	// 	
	// 	}

		//////////////////////////////////////
			//X方向
	cv::Mat matTest = matMask(rcProX);
	cv::Mat matTestImage = matSrcImage(rcProX);
	uchar* ptr = (uchar*)matProX.ptr(0);

	for (int y = 0; y < matTestImage.rows; y++, ptr++)
	{
		int nSum = 0;
		int nCount = 0;

		for (int x = 0; x < matTestImage.cols; x++)
		{
			if (matTest.data[y * matTest.step + x] > 0)
			{
				nCount++;
				nSum += matTestImage.data[y * matTestImage.step + x];
			}
		}

		if (nCount > 0)
			*ptr = nSum / nCount;
		else
			*ptr = 0;
	}

	//Y方向
	matTest = matMask(rcProY);
	matTestImage = matSrcImage(rcProY);
	ptr = (uchar*)matProY.ptr(0);

	for (int x = 0; x < matTestImage.cols; x++, ptr++)
	{
		int nSum = 0;
		int nCount = 0;

		for (int y = 0; y < matTestImage.rows; y++)
		{
			if (matTest.data[y * matTest.step + x] > 0)
			{
				nCount++;
				nSum += matTestImage.data[y * matTestImage.step + x];
			}
		}

		if (nCount > 0)
			*ptr = nSum / nCount;
		else
			*ptr = 0;
	}
	/////////////////////////////////////////

		//二进制
	cv::threshold(matProX, matProX, (cv::mean(matProX, matProX > 0)[0] * dThresholdRatio), 255, nType);
	cv::threshold(matProY, matProY, (cv::mean(matProY, matProY > 0)[0] * dThresholdRatio), 255, nType);

	rcProX = Rect(0, 0, rectHoleROI.x + (rectHoleROI.width / 2), matSrcImage.rows);
	rcProY = Rect(0, rectHoleROI.y + (rectHoleROI.height / 2), matSrcImage.cols, matSrcImage.rows - (rectHoleROI.y + (rectHoleROI.height / 2)));

	cv::resize(matProX, matProX, Size(rectHoleROI.x + (rectHoleROI.width / 2), matSrcImage.rows));
	cv::resize(matProY, matProY, Size(matSrcImage.cols, matSrcImage.rows - (rectHoleROI.y + (rectHoleROI.height / 2))));

	matProX(Rect(0, rectHoleROI.y + rectHoleROI.height, matProX.cols, matProX.rows - (rectHoleROI.y + rectHoleROI.height))).setTo(0);
	matProY(Rect(0, 0, rectHoleROI.x, matProY.rows)).setTo(0);

	//cv::add(matDstBuf, matCholeMask, matDstBuf);
	cv::add(matProX, matDstBuf(rcProX), matDstBuf(rcProX));
	cv::add(matProY, matDstBuf(rcProY), matDstBuf(rcProY));

	cv::subtract(matDstBuf, ~matMask, matDstBuf);

	//创建不良掩码,只检测特定的不良形状
	cv::Mat matDefectMask = matCholeMask.clone();

	matDefectMask(Rect(0, rectHoleROI.y, rectHoleROI.x + rectHoleROI.width, rectHoleROI.height)).setTo(255);
	matDefectMask(Rect(rectHoleROI.x, rectHoleROI.y, rectHoleROI.width, matDefectMask.rows - rectHoleROI.y)).setTo(255);

	cv::bitwise_and(matDstBuf, matDefectMask, matDstBuf);

	matProX.release();
	matProY.release();
}