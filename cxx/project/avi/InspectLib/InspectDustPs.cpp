#pragma once

#include "StdAfx.h"
#include "InspectDustPs.h"
#include "AlgoBase.h"
#include <numeric>





CInspectDust::CInspectDust(void)
{
	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
	sz = 11;
}

CInspectDust::~CInspectDust(void)
{
}

//保存8bit和12bit画面
long CInspectDust::ImageSave(CString strPath, cv::Mat matSrcBuf)
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

//xy方向管接头
long CInspectDust::Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, CMatBuf* cMemSub)
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
			nErrorCode |= AlgoBase::Estimation_X(matSrcBuf, matDstBufX,/* dPara*/nEstiDimX, nEstiStepX, dEstiBright, dEstiDark);
			break;
		case 1:
			nErrorCode |= AlgoBase::Estimation_Y(matSrcBuf, matDstBufY, /*dPara*/nEstiDimY, nEstiStepY, dEstiBright, dEstiDark);
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

long CInspectDust::DeleteArea3(cv::Mat& matSrcImage, int nCount, int nLength, CMatBuf* cMemSub)
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

bool CInspectDust::OrientedBoundingBox(cv::RotatedRect& rect1, cv::RotatedRect& rect2)
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

//Main检查算法
long CInspectDust::DoFindMuraDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
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
		nErrorCode = LogicStart_Mura_PS(matSrcBuffer, matSrcBufferRGB, matDstImage, matBKBuffer, rectROI, dPara, nCommonPara, strAlgPath);
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_DUST:
	{
		
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	case E_IMAGE_CLASSIFY_AVI_WHITE:
	{


	}
	break;
	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	{


	}
	break;
	case E_IMAGE_CLASSIFY_AVI_DUSTDOWN:
	{
		
	}
	break;
	//如果画面号码输入错误。
	default:
		return E_ERROR_CODE_TRUE;
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return nErrorCode;
}

long CInspectDust::LogicStart_Mura_PS(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGB, cv::Mat* matDstImage, cv::Mat& matBKBuffer, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath)
{
	long	nErrorCode = E_ERROR_CODE_TRUE;


	int		nGauSize = (int)dPara[E_PARA_AVI_MURA_COMMON_GAUSSIAN_SIZE];
	double	dGauSig = dPara[E_PARA_AVI_MURA_COMMON_GAUSSIAN_SIGMA];


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


	if (nBrightMorp % 2 == 0)		nBrightMorp++;
	if (nDarkMorp % 2 == 0)		nDarkMorp++;
	//if (nBrightTh1 < nBrightTh2)	nBrightTh2 = nBrightTh1;

	//////////////////////////////////////////////////////////////////////////
	
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

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	CRect rectTemp(rectROI);

	long	nWidth = (long)matSrcImage.cols;	
	long	nHeight = (long)matSrcImage.rows;	

	////////////////////////////////////////////////////////////////////////
	cv::Mat matSrcROIBuf_tmp = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), true);		 
	cv::Mat matSrcROIBuf_result = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);	

	cv::Mat matSrcROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matSrcImage.type(), false);
	cv::Mat matBrROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matDstImage[E_DEFECT_COLOR_BRIGHT].type(), false);
	cv::Mat matDaROIBuf = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matDstImage[E_DEFECT_COLOR_DARK].type(), false);

	matSrcROIBuf = matSrcImage(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	matBrROIBuf = matDstImage[E_DEFECT_COLOR_BRIGHT](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	matDaROIBuf = matDstImage[E_DEFECT_COLOR_DARK](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	//////////////////////////////////////////////////////////////////////////

	
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

	// 배경 영상
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

	// 버퍼 할당
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
	for (int i = 0; i < 2; i++)
	{
		switch (i)
		{
		case 0:
		{
			//////////////////////////////////////////////////////////////////////////
			// 밝은 불량 찾기
			//////////////////////////////////////////////////////////////////////////
			// 빼기
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

				// 2022.02.15 HGM SetTo(0)추가
				// 원래는 사용안해도 되는 동작이나 설비에서 Active Mask가 Active 영역 이외에 잡음이 엄청 올라오는 경우가 있어 Test용도로 추가
				matActive_Mask.setTo(0);
				matActive_Mask(R_Mask).setTo(255);
				cv::bitwise_not(matActive_Mask, matEdge_Mask);

				// 2022.02.15 HGM SetTo(0)추가
				// 원래는 사용안해도 되는 동작이나 설비에서 Active Mask가 Active 영역 이외에 잡음이 엄청 올라오는 경우가 있어 Test용도로 추가
				// 위에 해당하는 원인 찾기 위해 이미지 저장 추가
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

				//cv::threshold(matActive_Br_Temp, matThBrBuf_Act, 8, 255.0, CV_THRESH_BINARY);
				//cv::threshold(matEdge_Br_Temp, matThBrBuf_Edge, 8, 255.0, CV_THRESH_BINARY);

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

				// 이진화
	// 			cv::threshold(matSubBrBuf, matThBrBuf, nBrightTh1, 255.0, CV_THRESH_BINARY);
	// 			if (bImageSave)
	// 			{
	// 				CString strTemp;
	// 				strTemp.Format(_T("%s%s_%s_%02d_%02d_AVI_MURA_Bright_Th.jpg"), strAlgPath, gg_strPat[nImageNum], gg_strCam[nCamNum], nROINumber, nSaveImageCount++);
	// 				ImageSave(strTemp, matThBrBuf);
	// 			}
	// 
	// 			writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("threshold - Bright."));

				// 불량 붙이기
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

				// 배경 빼고 안쪽것만...
				// Edge 쪽에 불량이 존재하는 경우 라인 이어버림
				if (!matBkROIBuf.empty())
					cv::subtract(matBrROIBuf, matBkROIBuf, matBrROIBuf);
			}
			else {
				matBrROIBuf.setTo(0);
				break;
			}
			// 백무라용 이진화  ( 백점 : 255 / 백무라 200 )
			//cv::threshold(matSubBrBuf, matTempBuf, nBrightTh2, 200.0, CV_THRESH_BINARY);

		}
		break;
		case 1:
		{
			//////////////////////////////////////////////////////////////////////////
			// 중간 크기 밝은 불량 찾기
			//////////////////////////////////////////////////////////////////////////
			// 빼기
			if (nBright_inspect_new_Flag > 0) {
				/////////////////////////////////////////////////////////////////////////// 07.02 Test

				cv::Mat matResize = cMatBufTemp.GetMat(matSrcROIBuf_tmp.size(), matSrcROIBuf_tmp.type());
				matSrcImage.copyTo(matResize);

				cv::Mat matTmp = cMatBufTemp.GetMat(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()).size(), matSrcImage.type(), false);
				matResize(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height())).copyTo(matTmp);

				//////////////////////////////////////////////////////////////////////////
				//어두운 부분 평균 값으로 올리기
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
				//리사이즈
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
				//콘트라스트
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
				//밝기 보정
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
				//멕시칸 필터
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
				//캐니 엣지
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
				//사이즈 원복
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
				//사이드 과검 제거 
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
	// 밝은 불량 찾기
	//////////////////////////////////////////////////////////////////////////
	if (nBright_Inpect_Flag > 0)
	{
		// 작은 면적 제거 & 라인 잇기
		nErrorCode = DeleteArea3(matBrROIBuf, nBrightDelArea, 10, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_MURA, __FUNCTION__, _T("DeleteArea - Bright."));

		// 		// 작은 면적 제거
		// 		nErrorCode = DeleteArea1(matTempBuf, nBrightDelArea, &cMatBufTemp);
		// 		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;
		// 
		// 		// 불량 통합
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
	//////////////////////////////////////////////////////////////////////////
	// 중간 밝은 불량 찾기
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
