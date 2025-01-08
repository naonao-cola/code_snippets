
/************************************************************************
Line算法相关源
************************************************************************/

#include "StdAfx.h"
#include "InspectLine.h"
#include "AlgoBase.h"

#define round(fp) (int)((fp) >= 0 ? (fp) + 0.5 : (fp) - 0.5)

CInspectLine::CInspectLine(void)
{
	m_bProcess = false;

	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
}

CInspectLine::~CInspectLine(void)
{

}

long CInspectLine::FindLineDefect(cv::Mat matSrcBuffer, cv::Mat& matDrawBuffer, cv::Mat& matBKBuffer, vector<int> NorchIndex, CPoint OrgIndex, cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath,
	stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, wchar_t* strContourTxt)
{
	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	//	int		nImageNum			=  nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER		];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	int		nUIImageNumber = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];
	//如果参数为NULL。
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	long	nWidth = (long)matSrcBuffer.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuffer.rows;	// 图像垂直尺寸

	//画面号码
	long	nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start."));

	//初始化
	cv::Mat matDstImage[E_BINARY_TOTAL_COUNT];

	matDstImage[E_BINARY_BRIGHT_X] = cMem->GetMat(matSrcBuffer.size(), matSrcBuffer.type(), false);
	matDstImage[E_BINARY_BRIGHT_Y] = cMem->GetMat(matSrcBuffer.size(), matSrcBuffer.type(), false);
	matDstImage[E_BINARY_DARK_X] = cMem->GetMat(matSrcBuffer.size(), matSrcBuffer.type(), false);
	matDstImage[E_BINARY_DARK_Y] = cMem->GetMat(matSrcBuffer.size(), matSrcBuffer.type(), false);

	//大外围
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

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	m_bProcess = false;
	CString strPath;
	strPath.Format(_T("%s"), strAlgPath);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Init."));

	//原始备份
	cv::Mat matSrcBuf8bit = cMem->GetMat(matSrcBuffer.size(), CV_8U, false);
	if (matSrcBuffer.type() == CV_8U)
		matSrcBuffer.copyTo(matSrcBuf8bit);
	else
		matSrcBuffer.convertTo(matSrcBuf8bit, CV_8U, 1. / 16.);

	switch (nImageNum)	//每个画面的算法都不同。
	{
	case E_IMAGE_CLASSIFY_AVI_R:
	case E_IMAGE_CLASSIFY_AVI_G:
	case E_IMAGE_CLASSIFY_AVI_B:
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	case E_IMAGE_CLASSIFY_AVI_WHITE://增加White pattern Line检查21.04.15pwj
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	{
		nErrorCode = LogicStart_RGB5(matSrcBuf8bit, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, ptCorner, rcCHoleROI, matCholeBuffer);
		nErrorCode = LogicStart_Weak(matSrcBuf8bit, matDstImage, NorchIndex, OrgIndex, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, ptCorner);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_XB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_BRIGHT_X]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_YB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_BRIGHT_Y]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_XD.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_DARK_X]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_YD.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_DARK_Y]);
		}
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_BLACK:
	case E_IMAGE_CLASSIFY_AVI_PCD:
	case E_IMAGE_CLASSIFY_AVI_VINIT:
		nErrorCode = LogicStart_BlackWhite3(matSrcBuf8bit, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, ptCorner);
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
		nErrorCode = LogicStart_Crack(matSrcBuf8bit, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, ptCorner);
		break;

		//case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	case E_IMAGE_CLASSIFY_AVI_VTH:
		//	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
	{
		nErrorCode = LogicStart_Weak(matSrcBuf8bit, matDstImage, NorchIndex, OrgIndex, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, ptCorner);
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_XB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_BRIGHT_X]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_YB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_BRIGHT_Y]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_XD.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_DARK_X]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalResult_YD.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_DARK_Y]);
		}
	}
	break;

	//17.08.09-PG List修改
// 	case E_IMAGE_CLASSIFY_AVI_GRAY_87:

	default:
		return E_ERROR_CODE_TRUE;		// 画面号码输入错误的情况。
	}
	//////////////////////////////////////////////////////////////////////////
	// Back Ground Sub
	//////////////////////////////////////////////////////////////////////////

	if (!matBKBuffer.empty())
	{
		cv::subtract(matDstImage[E_BINARY_BRIGHT_X], matBKBuffer, matDstImage[E_BINARY_BRIGHT_X]);
		cv::subtract(matDstImage[E_BINARY_BRIGHT_Y], matBKBuffer, matDstImage[E_BINARY_BRIGHT_Y]);
		if (nImageNum != E_IMAGE_CLASSIFY_AVI_BLACK && nImageNum != E_IMAGE_CLASSIFY_AVI_PCD && nImageNum != E_IMAGE_CLASSIFY_AVI_VINIT)
		{
			cv::subtract(matDstImage[E_BINARY_DARK_X], matBKBuffer, matDstImage[E_BINARY_DARK_X]);
			cv::subtract(matDstImage[E_BINARY_DARK_Y], matBKBuffer, matDstImage[E_BINARY_DARK_Y]);
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalSubResult_XB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_BINARY_BRIGHT_X]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalSubResult_YB.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_BINARY_BRIGHT_Y]);
		if (nImageNum != E_IMAGE_CLASSIFY_AVI_BLACK && nImageNum != E_IMAGE_CLASSIFY_AVI_PCD && nImageNum != E_IMAGE_CLASSIFY_AVI_VINIT)
		{
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalSubResult_XD.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_DARK_X]);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_FinalSubResult_YD.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_BINARY_DARK_Y]);
		}
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////

			//如果参数为NULL。
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuf8bit.empty())						return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_BINARY_BRIGHT_X].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_BINARY_BRIGHT_Y].empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_BINARY_DARK_X].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_BINARY_DARK_Y].empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start."));

	cv::Rect rectInspROI;
	Insp_RectSet(rectInspROI, rectROI, matSrcBuf8bit.cols, matSrcBuf8bit.rows);

	//错误判定&发送结果
	{
		//标签
		CFeatureExtraction cFeatureExtraction;
		cFeatureExtraction.SetMem(cMem);
		cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_LINE, m_tInitTime, m_tBeforeTime, m_strAlgLog);

		// Bright X
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuf8bit(rectInspROI), matDstImage[E_BINARY_BRIGHT_X](rectInspROI), matDrawBuffer(rectInspROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("LBX"), EngineerBlockDefectJudge, pResultBlob);

		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuf8bit.release();
			matDstImage[E_BINARY_BRIGHT_X].release();
			matDstImage[E_BINARY_BRIGHT_Y].release();
			matDstImage[E_BINARY_DARK_X].release();
			matDstImage[E_BINARY_DARK_Y].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("BlobJudgment (BrightX)."));

		// Bright Y
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuf8bit(rectInspROI), matDstImage[E_BINARY_BRIGHT_Y](rectInspROI), matDrawBuffer(rectInspROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("LBY"), EngineerBlockDefectJudge, pResultBlob);

		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuf8bit.release();
			matDstImage[E_BINARY_BRIGHT_X].release();
			matDstImage[E_BINARY_BRIGHT_Y].release();
			matDstImage[E_BINARY_DARK_X].release();
			matDstImage[E_BINARY_DARK_Y].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("BlobJudgment (BrightY)."));

		if (nImageNum != E_IMAGE_CLASSIFY_AVI_BLACK && nImageNum != E_IMAGE_CLASSIFY_AVI_GRAY_32 && nImageNum != E_IMAGE_CLASSIFY_AVI_PCD && nImageNum != E_IMAGE_CLASSIFY_AVI_VINIT)
		{
			// Dark X
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuf8bit(rectInspROI), matDstImage[E_BINARY_DARK_X](rectInspROI), matDrawBuffer(rectInspROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_DARK, _T("LDX"), EngineerBlockDefectJudge, pResultBlob);

			if (nErrorCode != E_ERROR_CODE_TRUE)
			{
				//禁用内存
				matSrcBuf8bit.release();
				matDstImage[E_BINARY_BRIGHT_X].release();
				matDstImage[E_BINARY_BRIGHT_Y].release();
				matDstImage[E_BINARY_DARK_X].release();
				matDstImage[E_BINARY_DARK_Y].release();

				return nErrorCode;
			}

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("BlobJudgment (DarkX)."));

			// Dark Y
			nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuf8bit(rectInspROI), matDstImage[E_BINARY_DARK_Y](rectInspROI), matDrawBuffer(rectInspROI), rectROI,
				nCommonPara, E_DEFECT_COLOR_DARK, _T("LDY"), EngineerBlockDefectJudge, pResultBlob);

			if (nErrorCode != E_ERROR_CODE_TRUE)
			{
				//禁用内存
				matSrcBuf8bit.release();
				matDstImage[E_BINARY_BRIGHT_X].release();
				matDstImage[E_BINARY_BRIGHT_Y].release();
				matDstImage[E_BINARY_DARK_X].release();
				matDstImage[E_BINARY_DARK_Y].release();

				return nErrorCode;
			}

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("BlobJudgment (DarkY)."));
		}
	}

	//禁用内存
	matSrcBuf8bit.release();
	matDstImage[E_BINARY_BRIGHT_X].release();
	matDstImage[E_BINARY_BRIGHT_Y].release();
	matDstImage[E_BINARY_DARK_X].release();
	matDstImage[E_BINARY_DARK_Y].release();

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Memory Release."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return nErrorCode;
}

long CInspectLine::LogicStart_BlackWhite3(Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner)
{
	long	nErrorCode = E_ERROR_CODE_TRUE;		// 错误代码

	int		nWindowSize = dPara[E_PARA_LINE_BLACK_WINDOW_SIZE];		// 5
	float	fSigma = dPara[E_PARA_LINE_BLACK_SIGMA];		// 3
	int		nResizesize = dPara[E_PARA_LINE_BLACK_RESIZE];		// 2
	float	fThresholdRatio = dPara[E_PARA_LINE_BLACK_THRESHOLD_RATIO];		// 1	
	int		nOutLine = dPara[E_PARA_LINE_BLACK_OUTLINE];
	int		nRotationOnOff = dPara[E_PARA_LINE_BLACK_ROTATION_FLAG];		// 0

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
	int		nUIImageNumber = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	if (rectROI.left <= 0)	rectROI.left = 0;
	if (rectROI.right >= matSrcImage.cols)	rectROI.right = matSrcImage.cols - 1;
	if (rectROI.top <= 0)	rectROI.top = 0;
	if (rectROI.bottom >= matSrcImage.rows)	rectROI.bottom = matSrcImage.rows - 1;

	if (rectTemp.left <= 0)	rectTemp.left = 0;
	if (rectTemp.right >= matSrcImage.cols)	rectTemp.right = matSrcImage.cols - 1;
	if (rectTemp.top <= 0)	rectTemp.top = 0;
	if (rectTemp.bottom >= matSrcImage.rows)	rectTemp.bottom = matSrcImage.rows - 1;

	//Resize影像大小
	int nResizeW = matSrcImage.cols / nResizesize;
	int nResizeH = matSrcImage.rows / nResizesize;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	// Temp Buf
	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	cv::Mat matTempBuf2 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf1 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf2 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf3 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);
	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start."));

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nWindowSize;
	//rtInspROI.y = rectROI.top - nWindowSize;
	//rtInspROI.width = rectROI.Width() + nWindowSize * 2;
	//rtInspROI.height = rectROI.Height() + nWindowSize * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nWindowSize);

	//高斯安布尔
	cv::GaussianBlur(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), cv::Size(nWindowSize, nWindowSize), fSigma);

	//重新设置
	cv::resize(matTempBuf1, matResizeTempBuf1, cv::Size(nResizeW, nResizeH), 0, 0, INTER_LINEAR);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Pre-processing."));

	//标准偏差二进制化？
	Sdtthreshold(matResizeTempBuf1, matResizeTempBuf2, fThresholdRatio);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Thresholding."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_LINE_%02d_BlackThreshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf2);
	}

	cv::Mat matProjection[4];

	matProjection[0] = cMatBufTemp.GetMat(matResizeTempBuf2.rows, 1, matResizeTempBuf2.type(), false);
	matProjection[1] = cMatBufTemp.GetMat(1, matResizeTempBuf2.cols, matResizeTempBuf2.type(), false);
	matProjection[2] = cMatBufTemp.GetMat(1, matResizeTempBuf2.cols, matResizeTempBuf2.type(), false);
	matProjection[3] = cMatBufTemp.GetMat(matResizeTempBuf2.rows, 1, matResizeTempBuf2.type(), false);

#pragma omp parallel for num_threads(2)


	for (int i = 0; i < 2; i++)
	{
		switch (i)
		{
		case 0:
		{
			//////////////////////////////////////////////////////////////////////////
						//X方向
			//////////////////////////////////////////////////////////////////////////
			matResizeTempBuf1.setTo(0);
			LineMeasurement(matResizeTempBuf2, matResizeTempBuf1, matProjection, dPara, rectROI, 1, nOutLine, &cMatBufTemp);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("LineMeasure X."));

			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_MeasureX.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf1);
			}

			cv::resize(matResizeTempBuf1, matTempBuf1, matSrcImage.size(), 0, 0, INTER_LINEAR);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Resize_Up X."));

			matTempBuf1.copyTo(matDstImage[E_BINARY_BRIGHT_X]);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("X Copy."));
		}
		break;
		case 1:
		{
			//////////////////////////////////////////////////////////////////////////
						//Y方向
			//////////////////////////////////////////////////////////////////////////
			matResizeTempBuf3.setTo(0);
			LineMeasurement(matResizeTempBuf2, matResizeTempBuf3, matProjection, dPara, rectROI, 2, nOutLine, &cMatBufTemp);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("LineMeasure Y."));

			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_MeasureY.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf3);
			}

			cv::resize(matResizeTempBuf3, matTempBuf2, matSrcImage.size(), 0, 0, INTER_LINEAR);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Resize_Up Y."));

			matTempBuf2.copyTo(matDstImage[E_BINARY_BRIGHT_Y]);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Copy."));
		}
		break;
		}

	}

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::LogicStart_Crack(Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner)
{
	long	nErrorCode = E_ERROR_CODE_TRUE;		// 错误代码

	int		nWindowSize = dPara[E_PARA_LINE_BLACK_WINDOW_SIZE];		// 5
	float		fSigma = dPara[E_PARA_LINE_BLACK_SIGMA];		// 3
	int		nResizesize = dPara[E_PARA_LINE_BLACK_RESIZE];		// 2
	float		fThresholdRatio = dPara[E_PARA_LINE_BLACK_THRESHOLD_RATIO];		// 1	
	int		nOutLine = dPara[E_PARA_LINE_BLACK_OUTLINE];
	int		nRotationOnOff = dPara[E_PARA_LINE_BLACK_ROTATION_FLAG];		// 0

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	if (rectROI.left <= 0)	rectROI.left = 0;
	if (rectROI.right >= matSrcImage.cols)	rectROI.right = matSrcImage.cols - 1;
	if (rectROI.top <= 0)	rectROI.top = 0;
	if (rectROI.bottom >= matSrcImage.rows)	rectROI.bottom = matSrcImage.rows - 1;

	if (rectTemp.left <= 0)	rectTemp.left = 0;
	if (rectTemp.right >= matSrcImage.cols)	rectTemp.right = matSrcImage.cols - 1;
	if (rectTemp.top <= 0)	rectTemp.top = 0;
	if (rectTemp.bottom >= matSrcImage.rows)	rectTemp.bottom = matSrcImage.rows - 1;

	//Resize影像大小
	int nResizeW = matSrcImage.cols / nResizesize;
	int nResizeH = matSrcImage.rows / nResizesize;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	// Temp Buf
	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matTempBuf2 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf1 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf2 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf3 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nWindowSize;
	//rtInspROI.y = rectROI.top - nWindowSize;
	//rtInspROI.width = rectROI.Width() + nWindowSize * 2;
	//rtInspROI.height = rectROI.Height() + nWindowSize * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nWindowSize);

	//高斯安布尔
	cv::GaussianBlur(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), cv::Size(nWindowSize, nWindowSize), fSigma);

	//重新设置
	cv::resize(matTempBuf1, matResizeTempBuf1, cv::Size(nResizeW, nResizeH), 0, 0, INTER_LINEAR);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Pre-processing."));

	cv::Mat tempMean, tempStd;
	double minvalue = 0;
	double maxvlaue = 0;
	cv::meanStdDev(matResizeTempBuf1, tempMean, tempStd);
	cv::minMaxIdx(matResizeTempBuf1, &minvalue, &maxvlaue, NULL, NULL);

	double Meanvalue = tempMean.at<double>(0, 0);
	double Stdvalue = tempStd.at<double>(0, 0);
	double dbthresh = 0;

	if ((Meanvalue <= 0.1) && (Stdvalue <= 5))
	{
		dbthresh = 1;
	}

	if (((Meanvalue > 0.1) && (Meanvalue <= 1)) && ((Stdvalue > 5.0) && (Stdvalue <= 10.0)))
	{
		dbthresh = Stdvalue * fThresholdRatio;
	}

	if ((Meanvalue > 1.0) && (Stdvalue > 10))
	{
		dbthresh = Meanvalue * Stdvalue;
	}

	if ((Meanvalue > 0.1) && (Stdvalue < 5))
	{
		dbthresh = maxvlaue / 2;
	}

	//二进制
	cv::threshold(matResizeTempBuf1, matResizeTempBuf2, dbthresh, 255.0, THRESH_BINARY);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Thresholding."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_BlackThreshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf2);
	}

	cv::Mat matProjection[4];

	matProjection[0] = cMatBufTemp.GetMat(matResizeTempBuf2.rows, 1, matResizeTempBuf2.type(), false);
	matProjection[1] = cMatBufTemp.GetMat(matResizeTempBuf2.cols, 1, matResizeTempBuf2.type(), false);
	matProjection[2] = cMatBufTemp.GetMat(matResizeTempBuf2.cols, 1, matResizeTempBuf2.type(), false);
	matProjection[3] = cMatBufTemp.GetMat(matResizeTempBuf2.rows, 1, matResizeTempBuf2.type(), false);

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
					   //X方向
			//////////////////////////////////////////////////////////////////////////
			LineMeasurement(matResizeTempBuf2, matResizeTempBuf1, matProjection, dPara, rectROI, 1, nOutLine);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("LineMeasure X."));

			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_MeasureX.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf1);
			}

			cv::resize(matResizeTempBuf1, matTempBuf1, matSrcImage.size(), 0, 0, INTER_LINEAR);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Resize_Up X."));

			matTempBuf1.copyTo(matDstImage[E_BINARY_BRIGHT_X]);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("X Copy."));
		}
		break;
		case 1:
		{
			//////////////////////////////////////////////////////////////////////////
					   //Y方向
			//////////////////////////////////////////////////////////////////////////

			LineMeasurement(matResizeTempBuf2, matResizeTempBuf3, matProjection, dPara, rectROI, 2, nOutLine);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("LineMeasure Y."));

			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_MeasureY.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf3);
			}

			cv::resize(matResizeTempBuf3, matTempBuf2, matSrcImage.size(), 0, 0, INTER_LINEAR);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Resize_Up Y."));

			matTempBuf2.copyTo(matDstImage[E_BINARY_BRIGHT_Y]);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Copy."));
		}
		break;
		}
	}
	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::LogicStart_RGB5(Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	double		dblTargetGV = (double)dPara[E_PARA_LINE_RGB_TARGET_GV];
	int			nBlurSize = (int)dPara[E_PARA_LINE_RGB_BLUR_SIZE];
	int			nMeanFilterSize = (int)dPara[E_PARA_LINE_RGB_MEAN_FILTER_SIZE];
	int			nBGFilterSize = (int)dPara[E_PARA_LINE_RGB_BG_FILTER_SIZE];
	int			nImageFilterSize = (int)dPara[E_PARA_LINE_RGB_IMAGE_FILTER_SIZE];
	int			nDeletArea = (int)dPara[E_PARA_LINE_RGB_DELETE_AREA];

	// Weak Parameter
	int			nWeakResize = (int)dPara[E_PARA_LINE_RGB_WEAK_RESIZE];

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

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	//Resize影像大小
	int nResizeW = matSrcImage.cols / nWeakResize;
	int nResizeH = matSrcImage.rows / nWeakResize;

	// Input Image to Image Buffer
	cv::Mat matResizeTempBuf1 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);
	cv::Mat matResizeTempBuf2 = cMatBufTemp.GetMat(cv::Size(nResizeW, nResizeH), matSrcImage.type(), false);

	//Resize检查区域
	CRect rectResize;
	rectResize.left = rectROI.left / nWeakResize;
	rectResize.top = rectROI.top / nWeakResize;
	rectResize.right = rectROI.right / nWeakResize;
	rectResize.bottom = rectROI.bottom / nWeakResize;

	//////////////////////////////////////////////////////////////////////////
	// Pre-processing
	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	//减少画面处理时间
	cv::resize(matSrcImage, matResizeTempBuf1, cv::Size(nResizeW, nResizeH), 0, 0, INTER_LINEAR);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("resize."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ResiezeRGBImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf1);
	}

	cv::Rect rtInspROI;
	//rtInspROI.x = rectResize.left - nBlurSize;
	//rtInspROI.y = rectResize.top - nBlurSize;
	//rtInspROI.width = rectResize.Width() + nBlurSize * 2;
	//rtInspROI.height = rectResize.Height() + nBlurSize * 2;

	Insp_RectSet(rtInspROI, rectResize, matResizeTempBuf1.cols, matResizeTempBuf1.rows, nBlurSize);

	//删除Noise
	cv::blur(matResizeTempBuf1(rtInspROI), matResizeTempBuf2(rtInspROI), cv::Size(nBlurSize, nBlurSize));
	cv::blur(matResizeTempBuf2(rtInspROI), matResizeTempBuf1(rtInspROI), cv::Size(nBlurSize, nBlurSize));

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("blur."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_BlurImage.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf1);
	}

	//平均亮度校正
	AlgoBase::ApplyMeanGV(matResizeTempBuf1, dblTargetGV, rectResize);
	////////////////////////////////////////////////////////////////////
	//choikwangil
	//BYTE* pmatActive = (BYTE*)matResizeTempBuf1.data;

	//filter8(pmatActive, pmatActive, -10000, 10000, nResizeW, nResizeH);
	////////////////////////////////////////////////////////////////////

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ApplyGV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTempBuf1);
	}

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("ApplyMeanGV"));

	//缓冲区分配和初始化
	cv::Mat matDstBuff[2];
	matDstBuff[E_DEFECT_COLOR_DARK] = cMatBufTemp.GetMat(matResizeTempBuf1.size(), matResizeTempBuf1.type());
	matDstBuff[E_DEFECT_COLOR_BRIGHT] = cMatBufTemp.GetMat(matResizeTempBuf1.size(), matResizeTempBuf1.type());

	//////////////////////////////////////////////////////////////////////////
	// Main
	//////////////////////////////////////////////////////////////////////////

	nErrorCode = calcRGBMain(matSrcImage, matResizeTempBuf1, matDstBuff, matDstImage, dPara, nCommonPara,
		rectResize, ptCorner, strAlgPath, EngineerBlockDefectJudge, rcCHoleROI, matCholeBuffer, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("calcRGBMain."));

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////Functions///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
long CInspectLine::Morphology(Mat& matSrcImage, Mat& matDstImage, long nSizeX, long nSizeY, int nOperation, CMatBuf* cMemSub, int nIter)
{
	if (nSizeX < 1)			return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSizeY < 1)			return E_ERROR_CODE_POINT_WARNING_PARA;
	if (matSrcImage.empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	int nRep = nIter;

	cv::Point anchor(-1, -1);

	cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nSizeX, nSizeY));

	switch (nOperation)
	{
	case EP_MORP_ERODE:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_ERODE, StructElem, anchor, nRep);
		break;

	case EP_MORP_DILATE:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_DILATE, StructElem, anchor, nRep);
		break;

	case EP_MORP_OPEN:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_OPEN, StructElem, anchor, nRep);
		break;

	case EP_MORP_CLOSE:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_CLOSE, StructElem, anchor, nRep);
		break;

	case EP_MORP_GRADIENT:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_GRADIENT, StructElem, anchor, nRep);
		break;

	case EP_MORP_TOPHAT:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_TOPHAT, StructElem, anchor, nRep);
		break;

	case EP_MORP_BLACKHAT:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_BLACKHAT, StructElem, anchor, nRep);
		break;

	default:
		StructElem.release();
		return E_ERROR_CODE_POINT_WARNING_PARA;
		break;
	}

	StructElem.release();

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::RangeAvgThreshold_Gray(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI,
	long nLoop, long nSegX, long nSegY, float fDarkRatio, float fBrightRatio, float fDarkRatio_Edge, float fBrightRatio_Edge, CMatBuf* cMemSub)
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
	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nBlur;
	//rtInspROI.y = rectROI.top - nBlur;
	//rtInspROI.width = rectROI.Width() + nBlur * 2;
	//rtInspROI.height = rectROI.Height() + nBlur * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nBlur);

	if (nLoop > 0)
	{
		cv::blur(matSrcImage(rtInspROI), matBlurBuf(rtInspROI), cv::Size(nBlur, nBlur));

		if (nLoop > 1)
		{
			// Avg
			for (int i = 1; i < nLoop; i++)
			{
				cv::blur(matBlurBuf(rtInspROI), matBlurBuf1(rtInspROI), cv::Size(nBlur, nBlur));

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

			//Edge部分
			if (x == 0 || y == 0 || x == nRangeX || y == nRangeY)
			{
				double dbDarkRatio = fDarkRatio_Edge;
				double dbBrightRatio = fBrightRatio_Edge;
			}
			//计算x范围
			nStart_X = x * nSegX + rectROI.left;
			if (x == nRangeX - 1)		nEnd_X = rectROI.right;
			else					nEnd_X = nStart_X + nSegX;

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
			AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

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
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectTemp);
			cv::threshold(matTempBuf, matTempBufT, nDarkTemp, 255.0, THRESH_BINARY_INV);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectTemp);
			cv::threshold(matTempBuf, matTempBufT, nBrightTemp, 255.0, THRESH_BINARY);
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

int CInspectLine::GetBitFromImageDepth(int nDepth)
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

void CInspectLine::LineMeasurement(cv::Mat matSrcImage, cv::Mat& matDstImage, Mat* matProjection, double* dPara, CRect rectROI, int nDirection, int nOutLine, CMatBuf* cMemSub)
{
	int		nOffset = (int)dPara[E_PARA_LINE_BLACK_OFFSET];			// 10
	int		nXYThreshold = (int)dPara[E_PARA_LINE_BLACK_THRESHOLD_XY];		// 5
	int		nXXThreshold = (int)dPara[E_PARA_LINE_BLACK_THRESHOLD_XX];		// 20
	int		nYXThreshold = (int)dPara[E_PARA_LINE_BLACK_THRESHOLD_YX];		// 5
	int		nYYThreshold = (int)dPara[E_PARA_LINE_BLACK_THRESHOLD_YY];		// 20
	int		nThickness = (int)dPara[E_PARA_LINE_BLACK_THICKNESS];			// 1
	int		nPixelDistance = (int)dPara[E_PARA_LINE_BLACK_PIXEL_DISTANCE];	// 10

	uchar PtrValue, Subvalue;

	cv::Mat tmpImage;

	//矢量初始化
	vector <int>	vecX, vecY;
	vector <int>().swap(vecX);
	vector <int>().swap(vecY);

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	if (nDirection == 1)
	{
		cv::Mat Profile = matProjection[0];
		AlgoBase::MakeProfile(E_PROFILE_ROW, matSrcImage, Profile, &cMatBufTemp);

		int nRows = Profile.rows;

		uchar* BuffValue = (uchar*)Profile.data;

		for (int y = 0; y < nRows; y++)
		{
			PtrValue = BuffValue[y * Profile.cols + 0];

			if (PtrValue >= nXYThreshold)
				vecY.push_back(y);
		}

		for (int y = 0; y < vecY.size(); y++)
		{
			vector <int>().swap(vecX);
			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = 0;
			rectTemp.y = vecY[y] - nOffset;
			rectTemp.width = matSrcImage.cols - 1;
			rectTemp.height = nOffset * 2;

			//异常处理
			if (rectTemp.y < 0) rectTemp.y = 0;
			if (rectTemp.y + rectTemp.height >= matSrcImage.rows) rectTemp.height = matSrcImage.rows - rectTemp.y - 1;

			// ROI
			tmpImage = matSrcImage(rectTemp);

			cv::Mat subProfile = matProjection[1];
			AlgoBase::MakeProfile(E_PROFILE_COL, tmpImage, subProfile, &cMatBufTemp);

			int nCols = subProfile.cols;

			uchar* BuffV = (uchar*)subProfile.data;
			for (int x = 0; x < nCols; x++)
			{
				Subvalue = BuffV[x * subProfile.rows + 0];

				if (Subvalue > nXXThreshold)
					vecX.push_back(x);
			}

			if (vecX.size() == 0)
			{
				vecX.push_back(0);
				vecX.push_back(0);
			}
			//17.09.30-错误修复
			for (int x = 0; x < vecX.size() - 1; x++)
			{
				int nCom1 = vecX[x];
				int nCom2 = vecX[x + 1];

				//距离差不大的话,连接
				if (abs(nCom1 - nCom2) <= nPixelDistance)
					cv::line(matDstImage, cv::Point(nCom1, vecY[y]), cv::Point(nCom2, vecY[y]), cv::Scalar(255, 255, 255), 1);
			}
		}
	}
	else if (nDirection == 2)
	{
		cv::Mat Profile = matProjection[2];
		AlgoBase::MakeProfile(E_PROFILE_COL, matSrcImage, Profile, &cMatBufTemp);

		int nCols = Profile.cols;

		uchar* BuffValue = (uchar*)Profile.data;

		for (int y = 0; y < nCols; y++)
		{
			PtrValue = BuffValue[y * Profile.rows + 0];

			if (PtrValue >= nYXThreshold)
				vecY.push_back(y);
		}

		for (int y = 0; y < vecY.size(); y++)
		{
			vector <int>().swap(vecX);
			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = vecY[y] - nOffset;
			rectTemp.y = 0;
			rectTemp.width = nOffset * 2;
			rectTemp.height = matSrcImage.rows - 1;

			//异常处理
			if (rectTemp.x < 0) rectTemp.x = 0;
			if (rectTemp.x + rectTemp.width >= matSrcImage.cols) rectTemp.width = matSrcImage.cols - rectTemp.x - 1;

			tmpImage = matSrcImage(rectTemp);

			cv::Mat subProfile = matProjection[3];
			AlgoBase::MakeProfile(E_PROFILE_ROW, tmpImage, subProfile, &cMatBufTemp);

			int nRow = subProfile.rows;
			uchar* BuffV = (uchar*)subProfile.data;

			for (int x = 0; x < nRow; x++)
			{
				Subvalue = BuffV[x * subProfile.cols + 0];

				if (Subvalue > nYYThreshold)
					vecX.push_back(x);
			}
			if (vecX.size() == 0)
			{
				vecX.push_back(0);
				vecX.push_back(0);
			}

			//17.09.30-错误修复
			for (int x = 0; x < vecX.size() - 1; x++)
			{
				int nCom1 = vecX[x];
				int nCom2 = vecX[x + 1];

				//距离差不大的话,连接
				if (abs(nCom1 - nCom2) <= nPixelDistance)
					cv::line(matDstImage, cv::Point(vecY[y], nCom1), cv::Point(vecY[y], nCom2), cv::Scalar(255, 255, 255), 1);
			}
		}
	}

	vector <int>().swap(vecX);
	vector <int>().swap(vecY);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

}

void CInspectLine::Sdtthreshold(cv::Mat matSrcImage, cv::Mat& matDstImage, float fThresholdRatio)
{
	cv::Mat tempMean, tempStd;
	double minvalue = 0;
	double maxvlaue = 0;
	cv::meanStdDev(matSrcImage, tempMean, tempStd);
	cv::minMaxIdx(matSrcImage, &minvalue, &maxvlaue, NULL, NULL);

	double Meanvalue = tempMean.at<double>(0, 0);	// 平均
	double Stdvalue = tempStd.at<double>(0, 0);	// 标准偏差
	double dbthresh = 0;

	if ((Meanvalue <= 0.1) && (Stdvalue <= 5))
	{
		dbthresh = 1;
	}

	else if (((Meanvalue > 0.1) && (Meanvalue <= 1)) && (Stdvalue <= 5))
	{
		dbthresh = 1;
	}

	else if (((Meanvalue > 0.1) && (Meanvalue <= 1)) && ((Stdvalue > 5.0) && (Stdvalue <= 10.0)))

	{
		dbthresh = Stdvalue * fThresholdRatio;
	}

	else if ((Meanvalue > 1.0) && (Stdvalue > 10))
	{
		dbthresh = Meanvalue * Stdvalue;
	}

	else
	{
		dbthresh = maxvlaue / 2;
	}

	//二进制
	cv::threshold(matSrcImage, matDstImage, dbthresh, 255.0, THRESH_BINARY);
}

long CInspectLine::calcLine_BrightX(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath)
{
	int	nMorpOpen = (int)dPara[E_PARA_LINE_RGB_MORP_OPEN];
	int	nMorpClose = (int)dPara[E_PARA_LINE_RGB_MORP_CLOSE];

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int			nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool			bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int			nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int			nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int			nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int			nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int			nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	cv::Rect rtInspROI;
	//rtInspROI.x = rectResize.left - (nMorpOpen * 4 + 1);
	//rtInspROI.y = rectResize.top - (nMorpOpen * 4 + 1);
	//rtInspROI.width = rectResize.Width() + (nMorpOpen * 4 + 1) * 2;
	//rtInspROI.height = rectResize.Height() + (nMorpOpen * 4 + 1) * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, (nMorpOpen * 4 + 1));

	//只保留X线和Y线的Morphology
	Morphology(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), nMorpOpen * 4 + 1, 1, EP_MORP_OPEN);

	//rtInspROI.x = rectResize.left - nMorpClose;
	//rtInspROI.y = rectResize.top - nMorpClose;
	//rtInspROI.width = rectResize.Width() + nMorpClose * 2;
	//rtInspROI.height = rectResize.Height() + nMorpClose * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, nMorpClose);

	//粘贴线条稍微分开
	Morphology(matTempBuf1(rtInspROI), matTempBuf2(rtInspROI), nMorpClose, 1, EP_MORP_CLOSE);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Bright_Defect_Morphology_X.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf2);
	}

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Morphology."));

	//升级二进制画面
	cv::resize(matTempBuf2, matDstImage, matDstImage.size(), 0, 0, 0);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("resize."));

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcLine_BrightY(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath)
{
	int	nMorpOpen = (int)dPara[E_PARA_LINE_RGB_MORP_OPEN];
	int	nMorpClose = (int)dPara[E_PARA_LINE_RGB_MORP_CLOSE];

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int			nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool			bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int			nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int			nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int			nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int			nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int			nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	cv::Rect rtInspROI;
	//rtInspROI.x = rectResize.left - (nMorpOpen * 4 + 1);
	//rtInspROI.y = rectResize.top - (nMorpOpen * 4 + 1);
	//rtInspROI.width = rectResize.Width() + (nMorpOpen * 4 + 1) * 2;
	//rtInspROI.height = rectResize.Height() + (nMorpOpen * 4 + 1) * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, (nMorpOpen * 4 + 1));

	//只保留X线和Y线的Morphology
	Morphology(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), 1, nMorpOpen * 4 + 1, EP_MORP_OPEN);

	//rtInspROI.x = rectResize.left - nMorpClose;
	//rtInspROI.y = rectResize.top - nMorpClose;
	//rtInspROI.width = rectResize.Width() + nMorpClose * 2;
	//rtInspROI.height = rectResize.Height() + nMorpClose * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, nMorpClose);

	//粘贴线条稍微分开
	Morphology(matTempBuf1(rtInspROI), matTempBuf2(rtInspROI), 1, nMorpClose, EP_MORP_CLOSE);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Morphology."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Bright_Defect_Morphology_Y.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf2);
	}

	//升级二进制画面
	cv::resize(matTempBuf2, matDstImage, matDstImage.size(), 0, 0, 0);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("resize."));

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcLine_DarkX(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath)
{
	int	nMorpOpen = (int)dPara[E_PARA_LINE_RGB_MORP_OPEN];
	int	nMorpClose = (int)dPara[E_PARA_LINE_RGB_MORP_CLOSE];

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int			nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool			bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int			nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int			nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int			nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int			nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int			nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	cv::Rect rtInspROI;
	//rtInspROI.x = rectResize.left - (nMorpOpen * 4 + 1);
	//rtInspROI.y = rectResize.top - (nMorpOpen * 4 + 1);
	//rtInspROI.width = rectResize.Width() + (nMorpOpen * 4 + 1) * 2;
	//rtInspROI.height = rectResize.Height() + (nMorpOpen * 4 + 1) * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, (nMorpOpen * 4 + 1));

	//只保留X线和Y线的Morphology
	Morphology(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), nMorpOpen * 4 + 1, 1, EP_MORP_OPEN);

	//rtInspROI.x = rectResize.left - nMorpClose;
	//rtInspROI.y = rectResize.top - nMorpClose;
	//rtInspROI.width = rectResize.Width() + nMorpClose * 2;
	//rtInspROI.height = rectResize.Height() + nMorpClose * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, nMorpClose);

	//粘贴线条稍微分开
	Morphology(matTempBuf1(rtInspROI), matTempBuf2(rtInspROI), nMorpClose, 1, EP_MORP_CLOSE);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Morphology."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_matBinaryMorphDrakX.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf2);
	}

	//升级二进制画面
	cv::resize(matTempBuf2, matDstImage, matDstImage.size(), 0, 0, 0);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("resize."));

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcLine_DarkY(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matTempBuf1, cv::Mat& matTempBuf2, CRect rectResize, double* dPara, int* nCommonPara, CString strAlgPath)
{
	int	nMorpOpen = (int)dPara[E_PARA_LINE_RGB_MORP_OPEN];
	int	nMorpClose = (int)dPara[E_PARA_LINE_RGB_MORP_CLOSE];

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	cv::Rect rtInspROI;
	//rtInspROI.x = rectResize.left - (nMorpOpen * 4 + 1);
	//rtInspROI.y = rectResize.top - (nMorpOpen * 4 + 1);
	//rtInspROI.width = rectResize.Width() + (nMorpOpen * 4 + 1) * 2;
	//rtInspROI.height = rectResize.Height() + (nMorpOpen * 4 + 1) * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, (nMorpOpen * 4 + 1));

	//只保留X线和Y线的Morphology
	Morphology(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), 1, nMorpOpen * 4 + 1, EP_MORP_OPEN);

	//rtInspROI.x = rectResize.left - nMorpClose;
	//rtInspROI.y = rectResize.top - nMorpClose;
	//rtInspROI.width = rectResize.Width() + nMorpClose * 2;
	//rtInspROI.height = rectResize.Height() + nMorpClose * 2;

	Insp_RectSet(rtInspROI, rectResize, matSrcImage.cols, matSrcImage.rows, nMorpClose);

	//粘贴线条稍微分开
	Morphology(matTempBuf1(rtInspROI), matTempBuf2(rtInspROI), 1, nMorpClose, EP_MORP_CLOSE);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Morphology."));

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Dark_Defect_Morphology_Y.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf2);
	}

	//升级二进制画面
	cv::resize(matTempBuf2, matDstImage, matDstImage.size(), 0, 0, 0);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("resize."));

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcRGBMain(cv::Mat& matSrcImage, cv::Mat& matThImage, cv::Mat* matDstImage, cv::Mat* matBinaryMorp, double* dPara, int* nCommonPara,
	CRect rectResize, cv::Point* ptCorner, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, CMatBuf* cMemSub)
{
	long		nSegX = (long)dPara[E_PARA_LINE_RGB_SEG_X];
	long		nSegY = (long)dPara[E_PARA_LINE_RGB_SEG_Y];
	float		fDarkRatio_RGB = (float)dPara[E_PARA_LINE_RGB_DARK_RATIO_X];
	float		fBrightRatio_RGB = (float)dPara[E_PARA_LINE_RGB_BRIGHT_RATIO_X];
	float		fDarkRatio_RGB_Edge = (float)dPara[E_PARA_LINE_RGB_DARK_RATIO_Y];
	float		fBrightRatio_RGB_Edge = (float)dPara[E_PARA_LINE_RGB_BRIGHT_RATIO_Y];
	long		nOutLine = (long)dPara[E_PARA_LINE_RGB_OUTLINE];
	int		nWeakResize = (int)dPara[E_PARA_LINE_RGB_WEAK_RESIZE];

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Mat Create."));

	//为RangeThresholding Point使用Alg
	RangeAvgThreshold_Gray(matThImage, matDstImage, rectResize, 1, nSegX, nSegY, fDarkRatio_RGB, fBrightRatio_RGB, fDarkRatio_RGB_Edge, fBrightRatio_RGB_Edge, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("RangeAvgThreshold_Gray."));

	//删除最外围方案
	cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_LEFT_TOP] / nWeakResize, ptCorner[E_CORNER_RIGHT_TOP] / nWeakResize, cv::Scalar(0, 0, 0), nOutLine);
	cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_RIGHT_TOP] / nWeakResize, ptCorner[E_CORNER_RIGHT_BOTTOM] / nWeakResize, cv::Scalar(0, 0, 0), nOutLine);
	cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_RIGHT_BOTTOM] / nWeakResize, ptCorner[E_CORNER_LEFT_BOTTOM] / nWeakResize, cv::Scalar(0, 0, 0), nOutLine);
	cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_LEFT_BOTTOM] / nWeakResize, ptCorner[E_CORNER_LEFT_TOP] / nWeakResize, cv::Scalar(0, 0, 0), nOutLine);
	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Out Line Delete."));

	//删除Chole区域-仅亮线有问题
	for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
	{
		if (!matCholeBuffer[j].empty() && !rcCHoleROI[j].empty())
		{
			Rect rectResize;
			rectResize.x = rcCHoleROI[j].x / nWeakResize;
			rectResize.y = rcCHoleROI[j].y / nWeakResize;
			rectResize.width = rcCHoleROI[j].width / nWeakResize;
			rectResize.height = rcCHoleROI[j].height / nWeakResize;

			//缩小画面
			cv::Mat matResizeChole;
			cv::resize(matCholeBuffer[j], matResizeChole, cv::Size(rectResize.width, rectResize.height), 0, 0, INTER_LINEAR);

			cv::Mat matDelChole = matDstImage[E_DEFECT_COLOR_BRIGHT](rectResize);
			cv::subtract(matDelChole, matResizeChole, matDelChole);
			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Chole Area Del"));

		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//为并行处理分配Temp内存
	cv::Mat matTempBuf1[4];
	cv::Mat matTempBuf2[4];
	for (int i = 0; i < 4; i++)
	{
		matTempBuf1[i] = cMatBufTemp.GetMat(matThImage.size(), matSrcImage.type(), false);
		matTempBuf2[i] = cMatBufTemp.GetMat(matThImage.size(), matSrcImage.type());
	}

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
			// Bright X
			nErrorCode |= calcLine_BrightX(matDstImage[E_DEFECT_COLOR_BRIGHT], matBinaryMorp[E_BINARY_BRIGHT_X], matTempBuf1[0], matTempBuf2[0], rectResize, dPara, nCommonPara, strAlgPath);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("calcLine_BrightX."));

		}
		break;
		case 1:
		{
			// Bright Y
			nErrorCode |= calcLine_BrightY(matDstImage[E_DEFECT_COLOR_BRIGHT], matBinaryMorp[E_BINARY_BRIGHT_Y], matTempBuf1[1], matTempBuf2[1], rectResize, dPara, nCommonPara, strAlgPath);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("calcLine_BrightY."));

		}
		break;
		case 2:
		{
			// Dark X
			nErrorCode |= calcLine_DarkX(matDstImage[E_DEFECT_COLOR_DARK], matBinaryMorp[E_BINARY_DARK_X], matTempBuf1[2], matTempBuf2[2], rectResize, dPara, nCommonPara, strAlgPath);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("calcLine_DarkX."));

		}
		break;
		case 3:
		{
			// Dark Y
			nErrorCode |= calcLine_DarkY(matDstImage[E_DEFECT_COLOR_DARK], matBinaryMorp[E_BINARY_DARK_Y], matTempBuf1[3], matTempBuf2[3], rectResize, dPara, nCommonPara, strAlgPath);

			writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("calcLine_DarkY."));

		}
		break;
		}
	}
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::LogicStart_Weak(cv::Mat& matSrcImage, cv::Mat* matDstImage, vector<int> NorchIndex, CPoint OrgIndex, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Point* ptCorner)
{

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	const int	nMaxGV = 4096;

	int		nOnOff = dPara[E_PARA_LINE_RGB_WEAK_FLAG];	//on/off
	//消除噪音
	double		dPow = 2;	//= dPara[2];	// 2; // 1.2
	int		nWindowSize = dPara[E_PARA_LINE_RGB_WEAK_GAUSSIAN];	//= dPara[3];	// 31;
	double		dSigma = dPara[E_PARA_LINE_RGB_WEAK_PROJ_SIGAM];	//= dPara[4];	// 4;
	int		nReSizeValue = dPara[E_PARA_LINE_RGB_WEAK_RESIZE];

	//删除最外框
	int		nOutLineDark = dPara[E_PARA_LINE_RGB_WEAK_OUTLINEBRIGHT];	// 61;
	int		nOutLineBright = dPara[E_PARA_LINE_RGB_WEAK_OUTLINEDARK];	// 61;

	//检测出错误	
	int		dbTargateGV = dPara[E_PARA_LINE_RGB_WEAK_TARGET_GV];	// 50;

	double		dbThresholdBX = dPara[E_PARA_LINE_RGB_WEAK_BRIGHT_RATIO_X];	// 50;
	double		dbThresholdBY = dPara[E_PARA_LINE_RGB_WEAK_BRIGHT_RATIO_Y];	// 50;
	double		dbThresholdDX = dPara[E_PARA_LINE_RGB_WEAK_DARK_RATIO_X];	// 50;
	double		dbThresholdDY = dPara[E_PARA_LINE_RGB_WEAK_DARK_RATIO_Y];	// 50;

	//创建背景
	int		nBlur = dPara[E_PARA_LINE_RGB_WEAK_PROJ_BLUR1];	// 101;
	int		nBlur2 = dPara[E_PARA_LINE_RGB_WEAK_PROJ_BLUR2];	// 71;
	int		nRange = dPara[E_PARA_LINE_RGB_WEAK_PROJ_RANGE];	// 20;
	int		nMorp = dPara[E_PARA_LINE_RGB_WEAK_PROJ_MORPHOLOGY];	// 51;

	//////////////////////////////////////////////////////////////////////////

	   //错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;
	if (nOnOff == 0) return nErrorCode;

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Start"));

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	// Input Image to Image Buffer
	cv::Mat matTempBuf0 = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1, false);
	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), CV_16U);
	cv::Mat matTempBuf2 = cMatBufTemp.GetMat(matSrcImage.size(), CV_16U);

	matSrcImage.copyTo(matTempBuf0);

	//平均亮度校正
	if (dbTargateGV > 0)
	{
		AlgoBase::ApplyMeanGV(matTempBuf0, dbTargateGV, rectROI);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ApplyGV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf0);
		}

	}

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Image Buffer"));

	int nOffSet = 100;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matTempBuf0.cols, matTempBuf0.rows, nOffSet);

	// Image 8bit Pow
	AlgoBase::Image_Pow(matSrcImage.type(), dPow, matTempBuf0(rtInspROI), matTempBuf2(rtInspROI));
	matTempBuf0.release();
	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Pow"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf2);
	}

	//消除噪音
	resizeGaussian(matTempBuf2(rtInspROI), matTempBuf1(rtInspROI), nReSizeValue, nWindowSize, dSigma, &cMatBufTemp);
	matTempBuf2.release();
	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("resize Gaussian Blur"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_Gaus.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf1);
	}

	// 	double var_contrast1 = 0.3; // 必须转换为变量

	// 	if (var_contrast1 > 0)

	// 	double delta1 = 127.0*var_contrast1;
	// 	c = 255.0 / (255.0 - delta1);

	// 	double delta1 = -128.0*var_contrast1;
	// 	c = (256.0 - delta1 * 2) / 255.0;

	// 	Mat_test.copyTo(temp1);

	cv::Mat matROIBuf = matTempBuf1(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	// Result Image
	cv::Mat matDstImage_Weak[E_BINARY_TOTAL_COUNT];

	matDstImage_Weak[E_BINARY_BRIGHT_X] = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	matDstImage_Weak[E_BINARY_BRIGHT_Y] = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	matDstImage_Weak[E_BINARY_DARK_X] = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	matDstImage_Weak[E_BINARY_DARK_Y] = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	cv::Mat matProjectionX[5];
	cv::Mat matProjectionY[5];

	for (int i = 0; i < 5; i++)
	{
		matProjectionX[i] = cMatBufTemp.GetMat(1, matROIBuf.rows, matROIBuf.type(), false);
		matProjectionY[i] = cMatBufTemp.GetMat(1, matROIBuf.cols, matROIBuf.type(), false);
	}
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
			/////////////////////////// Y Line ////////////////////////////

			calcWeakLine_Y(matROIBuf, matDstImage[E_BINARY_BRIGHT_Y], matDstImage[E_BINARY_DARK_Y], matProjectionY, NorchIndex, OrgIndex, rectROI, dbThresholdBY, dbThresholdDY,
				nBlur, nBlur2, nRange, nMorp, nOutLineBright, nOutLineDark, nCommonPara, strAlgPath, dPara);

			cv::bitwise_or(matDstImage_Weak[E_BINARY_BRIGHT_Y], matDstImage[E_BINARY_BRIGHT_Y], matDstImage[E_BINARY_BRIGHT_Y]);
			cv::bitwise_or(matDstImage_Weak[E_BINARY_DARK_Y], matDstImage[E_BINARY_DARK_Y], matDstImage[E_BINARY_DARK_Y]);
			///////////////////////////////////////////////////////////////
		}
		break;
		case 1:
		{
			/////////////////////////// X Line ////////////////////////////

			calcWeakLine_X(matROIBuf, matDstImage[E_BINARY_BRIGHT_X], matDstImage[E_BINARY_DARK_X], matProjectionX, rectROI, dbThresholdBX, dbThresholdDX,
				nBlur, nBlur2, nRange, nMorp, nOutLineBright, nOutLineDark, nCommonPara, strAlgPath, dPara);

			cv::bitwise_or(matDstImage_Weak[E_BINARY_BRIGHT_X], matDstImage[E_BINARY_BRIGHT_X], matDstImage[E_BINARY_BRIGHT_X]);
			cv::bitwise_or(matDstImage_Weak[E_BINARY_DARK_X], matDstImage[E_BINARY_DARK_X], matDstImage[E_BINARY_DARK_X]);
			///////////////////////////////////////////////////////////////
		}
		break;
		}
	}

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("End"));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}
	///////////////////////////////////////////////////////////////
	return E_ERROR_CODE_TRUE;
}

long CInspectLine::resizeGaussian(Mat& InPutImage, Mat& OutPutImage, int reSizeValue, int MaskSize, double dSigma, CMatBuf* cMemSub)
{

	if (reSizeValue > 1)
	{
		CMatBuf cMatBufTemp;
		cMatBufTemp.SetMem(cMemSub);

		int reSizeWidth = InPutImage.cols / reSizeValue;
		int reSizeHeight = InPutImage.rows / reSizeValue;

		cv::Mat matreSizeBuf1 = cMatBufTemp.GetMat(cv::Size(reSizeWidth, reSizeHeight), CV_16U, false);
		cv::Mat matreSizeBuf2 = cMatBufTemp.GetMat(cv::Size(reSizeWidth, reSizeHeight), CV_16U, false);

		cv::resize(InPutImage, matreSizeBuf1, cv::Size(reSizeWidth, reSizeHeight), 0, 0, INTER_LINEAR);
		cv::GaussianBlur(matreSizeBuf1, matreSizeBuf2, cv::Size(MaskSize / reSizeValue, MaskSize / reSizeValue), dSigma / reSizeValue);
		cv::resize(matreSizeBuf2, OutPutImage, InPutImage.size(), 0, 0, INTER_LINEAR);

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}
	}
	else
		cv::GaussianBlur(InPutImage, OutPutImage, cv::Size(MaskSize, MaskSize), dSigma);

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcWeakLine_Y(Mat& InPutImage, Mat& OutPutImage1, Mat& OutPutImage2, Mat* matProjectionY, vector<int> NorchIndex, CPoint OrgIndex, CRect rectROI, double dbThresholdBY, double dbThresholdDY,
	int nBlur, int nBlur2, int nRange, int nMorp, int nOutLineBright, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//是否使用Norch
	int		nNorchOnOff = dPara[E_PARA_LINE_RGB_NORCH_ONOFF];	// 51;
	int		nNorchUnit = 0;
	if (nNorchOnOff > 0)
		nNorchUnit = (NorchIndex[0] + NorchIndex[2]) / 2;

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	int width = InPutImage.cols;
	int height = InPutImage.rows;

	cv::Mat MatproYBuf = matProjectionY[0];
	AlgoBase::MakeProjection(InPutImage, MatproYBuf, width, height, 1, nNorchOnOff, nNorchUnit, NorchIndex, OrgIndex);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProY.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproYBuf.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproYBuf.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	cv::Mat MatproYBuf1 = matProjectionY[1];
	calcProjection(MatproYBuf, MatproYBuf1, matProjectionY, width, nBlur, nBlur2, nRange, nMorp, 1, nCommonPara, strAlgPath);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProY_Blur2.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproYBuf1.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproYBuf1.at<ushort>(0, i));
			}

			fclose(out);
		}
	}
	calcWeakLine_BrigtY(MatproYBuf, MatproYBuf1, matProjectionY, OutPutImage1, rectROI, nNorchUnit, width, dbThresholdBY, nOutLineBright, nCommonPara, strAlgPath, dPara);
	calcWeakLine_DarkY(MatproYBuf1, MatproYBuf, matProjectionY, OutPutImage2, rectROI, nNorchUnit, width, dbThresholdDY, nOutLineDark, nCommonPara, strAlgPath, dPara);

	MatproYBuf.release();
	MatproYBuf1.release();
	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcWeakLine_X(Mat& InPutImage, Mat& OutPutImage1, Mat& OutPutImage2, Mat* matProjectionX, CRect rectROI, double dbThresholdBX, double dbThresholdDX,
	int nBlur, int nBlur2, int nRange, int nMorp, int nOutLineBright, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	int width = InPutImage.cols;
	int height = InPutImage.rows;

	cv::Mat MatproXBuf = matProjectionX[0];

	vector<int> ValueBuff;

	AlgoBase::MakeProjection(InPutImage, MatproXBuf, width, height, 0, 0, 0, ValueBuff, 0);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProX.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproXBuf.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproXBuf.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	cv::Mat MatproXBuf1 = matProjectionX[1];
	calcProjection(MatproXBuf, MatproXBuf1, matProjectionX, height, nBlur, nBlur2, nRange, nMorp, 0, nCommonPara, strAlgPath);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProX_Blur2.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproXBuf1.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproXBuf1.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	calcWeakLine_BrigtX(MatproXBuf, MatproXBuf1, matProjectionX, OutPutImage1, rectROI, height, dbThresholdBX, nOutLineBright, nCommonPara, strAlgPath, dPara);
	calcWeakLine_DarkX(MatproXBuf1, MatproXBuf, matProjectionX, OutPutImage2, rectROI, height, dbThresholdDX, nOutLineDark, nCommonPara, strAlgPath, dPara);

	MatproXBuf.release();
	MatproXBuf1.release();
	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcProjection(Mat& MatproSrc, Mat& MatproDst, Mat* matProjection, int size, int nBlur, int nBlur2,
	int nRange, int nMorp, int Type, int* nCommonPara, CString strAlgPath)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	cv::blur(MatproSrc, MatproDst, cv::Size(nBlur, 1));

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("pro blur"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		if (Type == 1)		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProY_Blur.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		else if (Type == 0)	strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProX_Blur.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproDst.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproDst.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	cv::Mat MatproMax = matProjection[2];
	cv::Mat MatproMin = matProjection[3];

	cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, 1), Point((nMorp / 2) + 1, 0));
	// MORPH_ERODE
	cv::morphologyEx(MatproSrc, MatproMin, MORPH_ERODE, StructElem);
	// MORPH_DILATE
	cv::morphologyEx(MatproSrc, MatproMax, MORPH_DILATE, StructElem);
	StructElem.release();

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Min/Max Filterilng"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		if (Type == 1)		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProY_Morp.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		else if (Type == 0)	strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProX_Morp.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			fprintf_s(out, "No,Src,Blur,Min,Max\n");

			for (int i = 0; i < MatproSrc.cols; i++)
			{
				fprintf_s(out, "%d,%d,%d,%d,%d\n", i, MatproSrc.at<ushort>(0, i), MatproDst.at<ushort>(0, i), MatproMin.at<ushort>(0, i), MatproMax.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	ushort* ptrBlur = (ushort*)MatproDst.ptr(0);
	ushort* ptrErode = (ushort*)MatproMin.ptr(0);
	ushort* ptrDilate = (ushort*)MatproMax.ptr(0);

	for (int i = 0; i < MatproDst.cols; i++, ptrBlur++, ptrErode++, ptrDilate++)
	{
		if (*ptrDilate - *ptrErode < nRange)
		{
			*ptrErode = *ptrBlur;
			continue;
		}

		//加入偏差小的GV
		if (*ptrBlur - *ptrErode > *ptrDilate - *ptrBlur)
			*ptrErode = *ptrDilate;
	}

	cv::blur(MatproMin, MatproDst, cv::Size(nBlur2, 1));

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("MainProcessilng"));

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcWeakLine_BrigtY(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionY, Mat& OutPutImage, CRect rectROI, int nNorchUnit, int size,
	double dbThresholdBY, int nOutLineBright, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//////////////////////////////////////////////////////////////////////////
	   //检查Parameter
	double		dInspRatio = dPara[E_PARA_LINE_RGB_NORCH_INSP_RATIO_BRIGHT];

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	cv::Mat MatproSub = matProjectionY[2];
	cv::subtract(MatproSrc, MatproDst, MatproSub);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Subtract"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProY_B_Sub.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproSub.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproSub.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	// RANSAC Proceed Data
	cv::Mat MatSubRANSAC_NR = matProjectionY[3];
	cv::Mat MatSubRANSAC_Max = matProjectionY[4];

	MatSubRANSAC_NR.setTo(0);
	MatSubRANSAC_Max.setTo(0);

	ProfileMaxFilter(MatproSub, MatSubRANSAC_Max, size, nCommonPara, strAlgPath, dPara, nOutLineBright);

	// Norch Value Enhancement
	NorchValueProcess(MatproSub, MatSubRANSAC_Max, size, nCommonPara, strAlgPath, dPara, nOutLineBright, nNorchUnit);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Bright Norch Process."));

	RangRANSACProcess(MatSubRANSAC_Max, MatSubRANSAC_NR, size, nCommonPara, strAlgPath, dPara);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Bright RANSAC Process."));

	// Rang Threshold
	RRM_Thresholding(MatproSub, MatSubRANSAC_NR, OutPutImage, rectROI, nNorchUnit, size, dbThresholdBY, nOutLineBright, 1, nCommonPara, strAlgPath, dPara, dInspRatio);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Bright RANSAC Threshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_BY_Th.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, OutPutImage);
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcWeakLine_DarkY(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionY, Mat& OutPutImage, CRect rectROI, int nNorchUnit, int size,
	double dbThresholdDY, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//////////////////////////////////////////////////////////////////////////
	   //检查Parameter
	double		dInspRatio = dPara[E_PARA_LINE_RGB_NORCH_INSP_RATIO_DARK];

	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	cv::Mat MatproSub = matProjectionY[2];
	cv::subtract(MatproSrc, MatproDst, MatproSub);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Subtract"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProY_D_Sub.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproSub.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproSub.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	// RANSAC Proceed Data
	cv::Mat MatSubRANSAC_NR = matProjectionY[3];
	cv::Mat MatSubRANSAC_Max = matProjectionY[4];

	MatSubRANSAC_NR.setTo(0);
	MatSubRANSAC_Max.setTo(0);

	ProfileMaxFilter(MatproSub, MatSubRANSAC_Max, size, nCommonPara, strAlgPath, dPara, nOutLineDark);

	// Norch Value Enhancement
	NorchValueProcess(MatproSub, MatSubRANSAC_Max, size, nCommonPara, strAlgPath, dPara, nOutLineDark, nNorchUnit);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Dark RANSAC Norch Process."));

	RangRANSACProcess(MatSubRANSAC_Max, MatSubRANSAC_NR, size, nCommonPara, strAlgPath, dPara);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Dark RANSAC Process."));

	// Rang Threshold
	RRM_Thresholding(MatproSub, MatSubRANSAC_NR, OutPutImage, rectROI, nNorchUnit, size, dbThresholdDY, nOutLineDark, 1, nCommonPara, strAlgPath, dPara, dInspRatio);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Y Dark RANSAC Threshold"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_DY_Th.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, OutPutImage);
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcWeakLine_BrigtX(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionX, Mat& OutPutImage, CRect rectROI, int size,
	double dbThresholdBX, int nOutLineBright, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	cv::Mat MatproSub = matProjectionX[2];
	cv::subtract(MatproSrc, MatproDst, MatproSub);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Subtract"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProX_B_Sub.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproSub.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproSub.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	// RANSAC Proceed Data
	cv::Mat MatSubRANSAC_NR = matProjectionX[3];
	cv::Mat MatSubRANSAC_Max = matProjectionX[4];

	MatSubRANSAC_NR.setTo(0);
	MatSubRANSAC_Max.setTo(0);

	ProfileMaxFilter(MatproSub, MatSubRANSAC_Max, size, nCommonPara, strAlgPath, dPara, nOutLineBright);

	RangRANSACProcess(MatSubRANSAC_Max, MatSubRANSAC_NR, size, nCommonPara, strAlgPath, dPara);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("X Bright RANSAC Process."));

	// Rang Threshold
	RRM_Thresholding(MatproSub, MatSubRANSAC_NR, OutPutImage, rectROI, 0, size, dbThresholdBX, nOutLineBright, 0, nCommonPara, strAlgPath, dPara, 0);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("X Bright RANSAC Threshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_BX_Th.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, OutPutImage);
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::calcWeakLine_DarkX(Mat& MatproSrc, Mat& MatproDst, Mat* matProjectionX, Mat& OutPutImage, CRect rectROI, int size,
	double dbThresholdDX, int nOutLineDark, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];
	//////////////////////////////////////////////////////////////////////////

	cv::Mat MatproSub = matProjectionX[2];
	cv::subtract(MatproSrc, MatproDst, MatproSub);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("Subtract"));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_ProX_D_Sub.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproSub.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproSub.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	// RANSAC Proceed Data
	cv::Mat MatSubRANSAC_NR = matProjectionX[3];
	cv::Mat MatSubRANSAC_Max = matProjectionX[4];

	MatSubRANSAC_NR.setTo(0);
	MatSubRANSAC_Max.setTo(0);

	ProfileMaxFilter(MatproSub, MatSubRANSAC_Max, size, nCommonPara, strAlgPath, dPara, nOutLineDark);

	RangRANSACProcess(MatSubRANSAC_Max, MatSubRANSAC_NR, size, nCommonPara, strAlgPath, dPara);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("X Dark RANSAC Process."));

	// Rang Threshold
	RRM_Thresholding(MatproSub, MatSubRANSAC_NR, OutPutImage, rectROI, 0, size, dbThresholdDX, nOutLineDark, 0, nCommonPara, strAlgPath, dPara, 0);

	writeInspectLog(E_ALG_TYPE_AVI_LINE, __FUNCTION__, _T("X Dark RANSAC Threshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_DX_Th.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, OutPutImage);
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::RangeThreshold_Weak(Mat& MatproSub, Mat& OutPutImage, CRect rectROI, int size,
	double dbThreshold, int nOutLine, int Type, int* nCommonPara, CString strAlgPath)
{
	int SubUnite = 300;
	int nPixelCount = 0;
	int nPixelSum = 0;
	int SubStart = 0;
	int SubEnd = 0;
	double SubAverage = 0;

	int Subrange = size / SubUnite;

	ushort* ptr;

	for (int i = 0; i < Subrange; i++)
	{
		nPixelSum = 0;
		nPixelCount = 0;

		SubStart = i * SubUnite;
		if (i == Subrange - 1)	SubEnd = size;
		else				SubEnd = SubStart + SubUnite;

		ptr = (ushort*)MatproSub.ptr(0) + SubStart;

		for (int j = SubStart; j < SubEnd; j++, ptr++)
		{
			nPixelSum += *ptr;
			nPixelCount++;
		}

		SubAverage = (double)nPixelSum / nPixelCount;

		if (SubStart < nOutLine)		SubStart = nOutLine;
		if (SubEnd > size - nOutLine)		SubEnd = size - nOutLine;

		ptr = (ushort*)MatproSub.ptr(0) + SubStart;

		if (Type == 1)
		{
			for (int j = SubStart; j < SubEnd; j++, ptr++)
			{
				//如果值小于日程安排,则跳过
				if (*ptr < 5 || *ptr < SubAverage * dbThreshold)	continue;

				//如果值存在,则存在不良(画线)
				cv::line(OutPutImage, cv::Point(rectROI.left + j, rectROI.top), cv::Point(rectROI.left + j, rectROI.bottom), cv::Scalar(255), 1);
			}
		}

		else if (Type == 0)
		{
			for (int j = SubStart; j < SubEnd; j++, ptr++)
			{
				//如果值小于日程安排,则跳过
				if (*ptr < 5 || *ptr < SubAverage * dbThreshold)	continue;

				//如果值存在,则存在不良(画线)
				cv::line(OutPutImage, cv::Point(rectROI.left, rectROI.top + j), cv::Point(rectROI.right, rectROI.top + j), cv::Scalar(255), 1);
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectLine::RangRANSACProcess(Mat& MatproSub, Mat& MatSubRANSAC, int size, int* nCommonPara, CString strAlgPath, double* dPara)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		SubUnite = dPara[E_PARA_LINE_RGB_RANSAC_TH_UNIT];

	//初始化数据
	int SubStart = 0;
	int SubEnd = 0;
	int Subrange = size / SubUnite;

	long double TestValueA;
	long double TestValueB;

	vector<Point2i> TestVector;
	vector<Point2i> TestVector_Unit;

	ushort* ptr;

	// Range Process
	for (int i = 0; i < Subrange; i++)
	{
		SubStart = i * SubUnite;
		if (i == Subrange - 1)	SubEnd = size;
		else				SubEnd = SubStart + SubUnite;

		ptr = (ushort*)MatproSub.ptr(0) + SubStart;

		//参数初始化
		int nDataPosition = 0;
		int nDataIndex = 0;

		vector <Point2i>().swap(TestVector_Unit);

		for (int j = SubStart; j < SubEnd; j++, ptr++)
		{
			int nDataValue = *ptr;
			TestVector_Unit.push_back(Point2f(nDataPosition, nDataValue));
			nDataPosition++;
		}

		//求Fitting值
		AlgoBase::calcRANSAC(TestVector_Unit, TestValueA, TestValueB);

		//减去Org Data处理的Data
		ptr = (ushort*)MatproSub.ptr(0) + SubStart;

		//输出值
		for (int j = SubStart; j < SubEnd; j++, ptr++)
		{
			double FitValue = (TestValueA * nDataIndex + TestValueB);
			if (FitValue < 0) FitValue = 0;
			FitValue += 1;
			TestVector.push_back(Point2f(j, FitValue));
			nDataIndex++;
			MatSubRANSAC.at<ushort>(0, j) = FitValue;
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_RANSAC_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < TestVector.size(); i++)
			{
				fprintf_s(out, "%d,%d\n", i, TestVector[i].y);
			}

			fclose(out);
		}
	}

	return E_ERROR_CODE_TRUE;
}

// MaxFilter Process
long CInspectLine::ProfileMaxFilter(Mat& MatproSub, Mat& MatSubRANSAC, int size, int* nCommonPara, CString strAlgPath, double* dPara, int nOutLine)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		SubUnite = dPara[E_PARA_LINE_RGB_RANSAC_MAXFILTER_SIZE];
	double		dbMeanRatio = dPara[E_PARA_LINE_RGB_RANSAC_AVG_TH_RATIO];

	//验证是否使用Norch检查
	if (SubUnite == 0) return false;

	//初始化数据
	int SubStart = 0;
	int SubEnd = 0;
	int Subrange = size / SubUnite + 1;

	vector<int>	TestVector;
	vector<int>	TestVector_Unit;
	vector<int>	TestVector_MC; // Max-Filter Cut data

	ushort* ptr;

	// Range Process
	for (int i = 0; i < Subrange; i++)
	{
		SubStart = i * SubUnite;
		if (i == Subrange - 1)	SubEnd = size;
		else				SubEnd = SubStart + SubUnite;

		ptr = (ushort*)MatproSub.ptr(0) + SubStart;

		vector <int>().swap(TestVector_Unit);

		for (int j = SubStart; j < SubEnd; j++, ptr++)
		{
			int nDataValue = *ptr;
			TestVector_Unit.push_back(nDataValue);
		}

		double minvalue = 0;
		double maxvlaue = 0;

		cv::minMaxIdx(TestVector_Unit, &minvalue, &maxvlaue, NULL, NULL);

		for (int j = SubStart; j < SubEnd; j++)
		{
			int nDataValue = maxvlaue;
			TestVector.push_back(nDataValue);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// MaxFilter Data PostProcess

	 // Max Data Info
	double dblMean;
	cv::Scalar m = cv::mean(TestVector);
	dblMean = m[0];
	int nMaxFiterData;

	int nGVSum = 0;
	int nGVCount = 0;

	for (int i = 0; i < TestVector.size(); i++)
	{
		if (i < nOutLine || i > TestVector.size() - nOutLine)
		{
			nMaxFiterData = dblMean;
		}

		else
		{
			nMaxFiterData = TestVector[i];
			if (nMaxFiterData > dblMean * dbMeanRatio) nMaxFiterData = dblMean;
		}

		TestVector_MC.push_back(nMaxFiterData);
		MatSubRANSAC.at<ushort>(0, i) = nMaxFiterData;
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_MaxFilter_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < TestVector.size(); i++)
			{
				fprintf_s(out, "%d,%d\n", i, TestVector[i]);
			}

			fclose(out);
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_MaxFilter_CutData.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < TestVector_MC.size(); i++)
			{
				fprintf_s(out, "%d,%d\n", i, TestVector_MC[i]);
			}

			fclose(out);
		}
	}

	return E_ERROR_CODE_TRUE;
}

// MaxFilter Process
long CInspectLine::NorchValueProcess(Mat& MatproSub, Mat& MatNorchMaxADD, int size, int* nCommonPara, CString strAlgPath, double* dPara, int nOutLine, int nNorchLength)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	//////////////////////////////////////////////////////////////////////////
	// Parameter
	int		nNorchOnOff = dPara[E_PARA_LINE_RGB_NORCH_ONOFF];
	int		nNorchMaxRange = dPara[E_PARA_LINE_RGB_NORCH_MAXFILTER_SIZE];
	int		nNorchAvgRange = dPara[E_PARA_LINE_RGB_NORCH_AVGFILTER_SIZE];
	int		nNorchMinGV = dPara[E_PARA_LINE_RGB_NORCH_MIN_GV];
	double		dbNorchMeanRatio = dPara[E_PARA_LINE_RGB_NORCH_AVG_TH_RATIO];

	//验证是否使用Norch检查是否为X方向
	if (nNorchOnOff == 0 || nNorchLength == 0) return false;

	//初始化Parameter
	int		nNorchUnit_Max = nNorchLength / nNorchMaxRange;
	int		nNorchUnit_Avg = nNorchLength / nNorchAvgRange;
	int		nStart, nEnd, nMaxCut_Value, nDataValue, nDataCount;
	ushort* ptr, * ptr2;

	vector<int>  SubData, NorchData, vMaxFilterData, vAvgFilterData;

	// Norch Data Average
	ptr = (ushort*)MatproSub.ptr(0);

	nDataValue = 0;
	nDataCount = 0;

	for (int i = 0; i < nNorchLength; i++, ptr++)
	{
		int nValue = *ptr;
		nDataValue += nValue;
		nDataCount++;
	}

	double dbDataMean = (double)nDataValue / nDataCount;

	// Norch Max 25
	for (int i = 0; i < nNorchUnit_Max; i++)
	{
		nStart = i * nNorchMaxRange;
		if (i == nNorchUnit_Max - 1)		nEnd = nNorchLength;
		else							nEnd = nStart + nNorchMaxRange;

		vector <int>().swap(SubData);
		ptr = (ushort*)MatproSub.ptr(0) + nStart;

		for (int j = nStart; j < nEnd; j++, ptr++)
		{
			SubData.push_back(*ptr);
		}

		double minvalue = 0;
		double maxvlaue = 0;

		cv::minMaxIdx(SubData, &minvalue, &maxvlaue, NULL, NULL);

		for (int j = nStart; j < nEnd; j++)
		{
			vMaxFilterData.push_back(maxvlaue);
		}
	}

	// Norch Avg 75
	for (int i = 0; i < nNorchUnit_Avg; i++)
	{
		nStart = i * nNorchAvgRange;
		if (i == nNorchUnit_Avg - 1)		nEnd = nNorchLength;
		else							nEnd = nStart + nNorchAvgRange;

		vector <int>().swap(SubData);
		ptr = (ushort*)MatproSub.ptr(0) + nStart;

		for (int j = nStart; j < nEnd; j++, ptr++)
		{
			SubData.push_back(*ptr);
		}

		cv::Scalar sSubMean = cv::mean(SubData);
		double dbSubMean = sSubMean[0];

		for (int j = nStart; j < nEnd; j++)
		{
			vAvgFilterData.push_back(dbSubMean);
		}
	}

	// Norch Max Cut & Enhancement
	ptr = (ushort*)MatproSub.ptr(0) + nOutLine;
	ptr2 = (ushort*)MatNorchMaxADD.ptr(0) + nOutLine;

	for (int i = nOutLine; i < nNorchLength; i++, ptr++, ptr2++)
	{
		int nMF_Value = vMaxFilterData[i];
		int nAVG_Value = vAvgFilterData[i];

		if (nMF_Value > nAVG_Value * dbNorchMeanRatio)	 nMaxCut_Value = dbDataMean;
		else												 nMaxCut_Value = nMF_Value;

		*ptr2 = nMaxCut_Value;

		if (nMaxCut_Value <= *ptr)
		{
			int nOrg_Value = *ptr;
			if (nOrg_Value <= nNorchMinGV) continue;
			else *ptr = nOrg_Value * nOrg_Value;
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_NorchEnhancement_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatproSub.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatproSub.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_NorchMaxCut_Data.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < MatNorchMaxADD.cols; i++)
			{
				fprintf_s(out, "%d,%d\n", i, MatNorchMaxADD.at<ushort>(0, i));
			}

			fclose(out);
		}
	}

	return E_ERROR_CODE_TRUE;
}

// Range RANSAC MaxFilter Thresholding
long	CInspectLine::RRM_Thresholding(Mat& MatproSub, Mat& MatSubRANSAC, Mat& OutPutImage, CRect rectROI, int nNorchUnit,
	int size, double dbThreshold, int nOutLine, int Type, int* nCommonPara, CString strAlgPath, double* dPara, double dInspRatio)
{
	//////////////////////////////////////////////////////////////////////////
	   //公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];
	bool		bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];

	// Parameter
	int		nUseNorch = dPara[E_PARA_LINE_RGB_NORCH_ONOFF];  // Norch检查开/关

	//初始化数据
	ushort* ptr_Org, * prt_RANSAC;

	vector<int> SubData;

	int nDataStart = nOutLine;
	int nDataEnd = size - nOutLine;

	// Profile Data
	ptr_Org = (ushort*)MatproSub.ptr(0);
	prt_RANSAC = (ushort*)MatSubRANSAC.ptr(0);

	for (int i = 0; i < size; i++, ptr_Org++, prt_RANSAC++)
	{
		int nOrgData = *ptr_Org;
		int nRANSACData = *prt_RANSAC;

		int nSubValue = nOrgData - nRANSACData;

		if (nSubValue < 0) nSubValue = 0;

		SubData.push_back(nSubValue);
	}

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_LINE_%02d_RRMData.csv"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);

		char szPath[MAX_PATH] = { 0, };
		memset(szPath, 0, sizeof(char) * MAX_PATH);
		WideCharToMultiByte(CP_ACP, 0, strTemp, -1, szPath, sizeof(szPath), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szPath, "wt");

		if (out)
		{
			for (int i = 0; i < SubData.size(); i++)
			{
				fprintf_s(out, "%d,%d\n", i, SubData[i]);
			}

			fclose(out);
		}
	}

	// Thresholding
	for (int i = nDataStart; i < nDataEnd; i++)
	{
		// SubData[i] = SubData[i] + SubData[i];

		if (Type == 1 && nUseNorch == 0)
		{
			if (SubData[i] < dbThreshold)	continue;

			cv::line(OutPutImage, cv::Point(rectROI.left + i, rectROI.top), cv::Point(rectROI.left + i, rectROI.bottom), cv::Scalar(255), 1);
		}

		else if (Type == 1 && nUseNorch == 1)
		{
			if (i <= nNorchUnit)
			{
				if (SubData[i] < dInspRatio)	continue;

				cv::line(OutPutImage, cv::Point(rectROI.left + i, rectROI.top), cv::Point(rectROI.left + i, rectROI.bottom), cv::Scalar(255), 1);
			}

			else if (i > nNorchUnit)
			{
				if (SubData[i] < dbThreshold)	continue;

				cv::line(OutPutImage, cv::Point(rectROI.left + i, rectROI.top), cv::Point(rectROI.left + i, rectROI.bottom), cv::Scalar(255), 1);
			}
		}

		else if (Type == 0)
		{
			if (SubData[i] < dbThreshold)	continue;

			cv::line(OutPutImage, cv::Point(rectROI.left, rectROI.top + i), cv::Point(rectROI.right, rectROI.top + i), cv::Scalar(255), 1);
		}
	}

	return E_ERROR_CODE_TRUE;
}

//保存8bit和12bit画面
long CInspectLine::ImageSave(CString strPath, cv::Mat matSrcBuf)
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

void CInspectLine::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset)
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

//choikwangil
void CInspectLine::filter8(BYTE* pbyInImg, BYTE* pbyOutImg, int nMin, int nMax, int nWidth, int nHeight)
{
	long nSum;
	double dvalue;

	int coef[8] = { 13,14,20,14,7,-4,-7,-1 };

	int nImagesize = nWidth * nHeight;

	long* plImg = new long[nImagesize];
	long* plLoG = new long[nImagesize];

	//	memcpy(NewImg,InImg, width*height);
	memset(plLoG, (long)0, nImagesize);

	for (int i = 0; i < nImagesize; i++)
	{
		plImg[i] = (long)pbyInImg[i];

	}

	int nMaskSize = 8;

	for (int x = 1; x < nWidth; x++)
	{
		plImg[x] += plImg[x - 1];
	}

	for (int y = 1; y < nHeight; y++)
	{
		plImg[y * nWidth] += plImg[(y - 1) * nWidth];
	}
	for (int y = 2; y < nHeight; y++)
	{
		for (int x = 2; x < nWidth; x++)
		{
			plImg[y * nWidth + x] += plImg[(y - 1) * nWidth + x] + plImg[y * nWidth + (x - 1)] - plImg[(y - 1) * nWidth + (x - 1)];
		}
	}

	//nMax =-10000000;
	//nMin = 10000000;
#ifndef _DEBUG
#pragma omp parallel for
#endif
	for (int y = nMaskSize; y < nHeight - nMaskSize; y++)
	{
		for (int x = nMaskSize; x < nWidth - nMaskSize; x++)
		{
			nSum = 0;

			for (int m = 0; m < nMaskSize; m++)
			{
				nSum += (long)(coef[m] * (plImg[(y + m) * nWidth + (x + m)]
					- plImg[(y - (m + 1)) * nWidth + (x + m)]
					- plImg[(y + m) * nWidth + (x - (m + 1))]
					+ plImg[(y - (m + 1)) * nWidth + (x - (m + 1))]));
			}
			plLoG[y * nWidth + x] = nSum;

			/*if (plLoG[y * nWidth + x] > nMax)
			{
			nMax=plImg[y * nWidth + x];
			}

			if (plLoG[y * nWidth + x] < nMin)
			{
			nMin=plImg[y * nWidth + x];
			}*/
		}
	}
	//nMax = 8000;
	//nMin =-8000;
#ifndef _DEBUG
#pragma omp parallel for
#endif
	for (int y = 0; y < nHeight; y++)
	{
		for (int x = 0; x < nWidth; x++)
		{
			dvalue = ((double)((plLoG[y * nWidth + x] - nMin) * 255) / (double)(nMax - nMin));
			if (dvalue < 0.0f)
			{
				pbyOutImg[y * nWidth + x] = (BYTE)0;
			}
			else if (dvalue > 255.0f)
			{
				pbyOutImg[y * nWidth + x] = (BYTE)255;
			}
			else
			{
				pbyOutImg[y * nWidth + x] = (BYTE)dvalue;
			}
		}
	}

	delete[] plImg;
	delete[] plLoG;
}
