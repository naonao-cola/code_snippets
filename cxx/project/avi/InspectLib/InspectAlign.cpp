
/************************************************************************/
//Align集成相关源
//修改日期:18.02.07
/************************************************************************/

#include "StdAfx.h"
#include "InspectAlign.h"

#include "ExportLibrary.h"
#include "AlgoBase.h"
#include "InspectLabelPol.h"
#include <stdlib.h>

CInspectAlign::CInspectAlign(void)
{
	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
}

CInspectAlign::~CInspectAlign(void)
{
}

//查找Top Line角度
long CInspectAlign::DoFindTheta(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point& ptCenter, wchar_t* strID)
{
	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Start."));

	cv::Mat matGrayBuf, matTempBuf;

	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	matGrayBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);
	matTempBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);

	//颜色(SVI)
	if (matSrcBuf.channels() != 1)
		cv::cvtColor(matSrcBuf, matGrayBuf, COLOR_RGB2GRAY);
	//黑白(AVI,APP)
	else
	{
		if (matSrcBuf.type() == CV_8UC1)
			matSrcBuf.copyTo(matGrayBuf);
		else
			matSrcBuf.convertTo(matGrayBuf, CV_8UC1, 1. / 16.);

		//matSrcBuf.copyTo(matGrayBuf);
		//matGrayBuf = matSrcBuf.clone();
	}

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nThreshold = (int)dPara[E_PARA_ALIGN_THRESHOLD];
	int		nMorp = (int)dPara[E_PARA_ALIGN_MORP];
	double	dAngleError = dPara[E_PARA_ALIGN_ANGLE_ERR];
	double	dAngleWarning = dPara[E_PARA_ALIGN_ANGLE_WAR];

	//外观:Cell面积(Pixel数量)
	//点灯:点灯面积(Pixel数量)
	int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y]);

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

		//Morphology Close(在Cell之间填充空格)
		cv::morphologyEx(matGrayBuf, matTempBuf, MORPH_CLOSE, StructElem);

		StructElem.release();
	}
	else
		matGrayBuf.copyTo(matTempBuf);
	//matTempBuf = matGrayBuf.clone();

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Morphology."));

	// Threshold
	cv::threshold(matTempBuf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("threshold."));

	//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matTempBuf, nMinArea, rectCell);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("FindCellArea."));

	//Cell中心点
	ptCenter.x = rectCell.x + rectCell.width / 2;
	ptCenter.y = rectCell.y + rectCell.height / 2;

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matGrayBuf.release();
		matTempBuf.release();
		return nErrorCode;
	}

	long double	dValueA = 0, dValueB = 0;

	//查找Top直线
	nErrorCode = RobustFitLine(matTempBuf, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_TOP);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("FitLine."));

	//禁用内存
	matGrayBuf.release();
	matTempBuf.release();

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//使用Top Line(长边)求角度
	dTheta = atan(dValueA) * 180. / PI;

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//查找AVI扫描区域
long CInspectAlign::DoFindActive(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, cv::Point* ptContCorner, int nRatio, cv::Point& ptCenter, wchar_t* strID)
{
	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Start."));

	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	cv::Mat matGrayBuf, matTempBuf;

	//缓冲区分配和初始化
	matGrayBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);
	matTempBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);

	//颜色(SVI)
	if (matSrcBuf.channels() != 1)
		cv::cvtColor(matSrcBuf, matGrayBuf, COLOR_RGB2GRAY);
	//黑白(AVI,APP)
	else
	{
		if (matSrcBuf.type() == CV_8UC1)
			matSrcBuf.copyTo(matGrayBuf);
		else
			matSrcBuf.convertTo(matGrayBuf, CV_8UC1, 1. / 16.);

		//matSrc_8bit.copyTo(matGrayBuf);
		//matGrayBuf = matSrc_8bit.clone();
	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("cvtColor."));

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nThreshold = dPara[E_PARA_ALIGN_THRESHOLD];
	int		nMorp = dPara[E_PARA_ALIGN_MORP];
	double	dAngleError = dPara[E_PARA_ALIGN_ANGLE_ERR];
	double	dAngleWarning = dPara[E_PARA_ALIGN_ANGLE_WAR];
	int     nUseRotateFlg = dPara[E_PARA_AVI_Rotate_Image];

	//外观:Cell面积(Pixel数量)
	//点灯:点灯面积(Pixel数量)
	int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);	// 3800 * 1900;	// APP

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////
		//仅用于AVI,用于在Cell数组之间填充空格
	//////////////////////////////////////////////////////////////////////////

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

		//Morphology Close(在Cell之间填充空格)
		cv::morphologyEx(matGrayBuf, matTempBuf, MORPH_CLOSE, StructElem);

		StructElem.release();
	}
	else
		matGrayBuf.copyTo(matTempBuf);
	//matTempBuf = matGrayBuf.clone();

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Morphology."));

	//cv::imwrite("E:\\IMTC\\1.Src.bmp", matGrayBuf);
	//cv::imwrite("E:\\IMTC\\2.Morp.bmp", matTempBuf);

	// Threshold
	cv::threshold(matTempBuf, matGrayBuf, nThreshold, 255.0, THRESH_BINARY);

	//cv::imwrite("E:\\IMTC\\3.Th.bmp", matGrayBuf);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("threshold."));

	//////////////////////////////////////////////////////////////////////////
		//查找Cell区域
	//////////////////////////////////////////////////////////////////////////

		//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matGrayBuf, nMinArea, rectCell);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("FindCellArea."));

	// 启用旋转的话，保存轮廓的4个角点，用于透视变换矫正
	if (nUseRotateFlg > 0) {
		vector<vector<cv::Point>>	contours;

		cv::findContours(matGrayBuf, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

		vector<cv::Point> ptContours, ptConvexHull;
		for (int m = 0; m < contours.size(); m++)
		{
			for (int k = 0; k < contours.at(m).size(); k++)
				ptContours.push_back(contours[m][k]);
		}

		cv::convexHull(ptContours, ptConvexHull);
		cv::fillConvexPoly(matGrayBuf, ptConvexHull, cv::Scalar(255, 255, 255));


		vector<vector<cv::Point>>().swap(contours);
		vector<cv::Point>().swap(ptContours);
		vector<cv::Point>().swap(ptConvexHull);

		long double dValueAA[E_ALIGN_TYPE_END], dValueBB[E_ALIGN_TYPE_END];

		for (int nType = E_ALIGN_TYPE_LEFT; nType <= E_ALIGN_TYPE_BOTTOM; nType++)
		{
			nErrorCode = RobustFitLine(matGrayBuf, rectCell, dValueAA[nType], dValueBB[nType], nMinSamples, distThreshold, nType, 10);

			if (nErrorCode != E_ERROR_CODE_TRUE)
				return nErrorCode;
		}

		for (int i = E_CORNER_LEFT_TOP; i <= E_CORNER_LEFT_BOTTOM; i++)
		{
			double dA1 = dValueAA[i];
			double dB1 = dValueBB[i];
			double dA2 = dValueAA[(i + 1) % E_CORNER_END];
			double dB2 = dValueBB[(i + 1) % E_CORNER_END];

			int x = (int)((dB2 - dB1) / (dA1 - dA2));
			int y = (int)((dA2 * dB1 - dA1 * dB2) / (dA2 - dA1));
			ptContCorner[i].x = x;
			ptContCorner[i].y = y;
		}
	}

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matGrayBuf.release();
		matTempBuf.release();
		return nErrorCode;
	}

	//Cell中心点
	ptCenter.x = rectCell.x + rectCell.width / 2;
	ptCenter.y = rectCell.y + rectCell.height / 2;

	//////////////////////////////////////////////////////////////////////////
		//查找点亮区域
	//////////////////////////////////////////////////////////////////////////

		//拯救Contours
	vector<vector<cv::Point>>	contours;
	vector<vector<cv::Point>>().swap(contours);

	int nContourSize = 0;
	int nContourIdx = 0;

	cv::findContours(matGrayBuf, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++)
	{
		if (nContourSize < contours[i].size())
		{
			nContourSize = contours[i].size();
			nContourIdx = i;
		}
	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Contours."));

	cv::RotatedRect BoundingBox = cv::minAreaRect(contours[nContourIdx]);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("RotatedRect."));

	cv::Point2f vertices[E_CORNER_END];
	BoundingBox.points(vertices);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("BoundingBox."));

	//检查转角结果
//cv::Mat matTempBufff = matSrcBuf.clone();
//cv::line(matTempBufff, vertices[0], vertices[1], CvScalar(255, 255, 255));
//cv::line(matTempBufff, vertices[1], vertices[2], CvScalar(255, 255, 255));
//cv::line(matTempBufff, vertices[2], vertices[3], CvScalar(255, 255, 255));
//cv::line(matTempBufff, vertices[3], vertices[0], CvScalar(255, 255, 255));

	vector<vector<cv::Point>>().swap(contours);

	//禁用内存
	matGrayBuf.release();
	matTempBuf.release();

	//////////////////////////////////////////////////////////////////////////
		//查找4个拐角转角
	//////////////////////////////////////////////////////////////////////////

	nErrorCode = FindCornerPoint(vertices, ptResCorner, nWidth, nHeight);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Find 4-Corner."));

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//使用Top Line(长边)求角度
	dTheta = BoundingBox.angle;

	//异常处理
	if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
	if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;

	//角度太大时出错。(严重警报)
	if (abs(dTheta) > dAngleError)
		return E_ERROR_CODE_ALIGN_ANGLE_RANGE_ERROR;

	//角度太大时出错。(警报)
	else if (abs(dTheta) > dAngleWarning)
		return E_ERROR_CODE_ALIGN_ANGLE_RANGE_WARNING;

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//查找APP检查区域
long CInspectAlign::DoFindActive_APP(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, int nRatio, double dCamResolution, double dPannelSizeX, double dPannelSizeY, cv::Point& ptCenter, wchar_t* strID)
{
	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Start."));

	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	cv::Mat matGrayBuf, matTempBuf;

	//如果没有缓冲区。
	if (matSrcBuf.empty())
	{
		return E_ERROR_CODE_EMPTY_BUFFER;
	}

	//缓冲区分配和初始化
	matGrayBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1);
	matTempBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1);

	matSrcBuf.copyTo(matGrayBuf);
	matGrayBuf.copyTo(matTempBuf);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("cvtColor."));

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////
	int		nThreshold = (int)dPara[E_PARA_APP_ALIGN_THRESHOLD];
	int		nDilate = (int)dPara[E_PARA_APP_ALIGN_DILATE];
	int		nPanelEdgeTheta = (int)dPara[E_PARA_APP_ALIGN_PANEL_EDGE_THETA];
	int		nMinSamples = 3;			// 通用
	double	distThreshold = 5;			// 通用
	double	dAngleError = (double)dPara[E_PARA_APP_ALIGN_ANGLE_ERR];
	double	dAngleWarning = (double)dPara[E_PARA_APP_ALIGN_ANGLE_WAR];
	double	fAvgOffset = (double)dPara[E_PARA_APP_ALIGN_AVG_OFFSET];
	bool	bUseOverRange = (bool)dPara[E_PARA_APP_ALIGN_USE_OVERRANGE];
	int		dLengthError = (int)dPara[E_PARA_APP_ALIGN_RANGE_OVER];

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;
	long nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	cv::Rect rtCell = cv::Rect(0, 0, matGrayBuf.cols, matGrayBuf.rows);
	long double ldValueB[4] = { 0.0, };
	long double	ldValueA[4] = { 0.0, };

	try
	{
		//进化后只留下最大的Blob(消除Stage背景噪音)
		//使用Morphlogy消除Cell近距离噪音,填充Cell面包,增大And使用的Mask Size大小

		bool bState = false;

		Mat matThreshold;
		cv::threshold(matTempBuf, matThreshold, nThreshold, 255, THRESH_BINARY);

		Mat matBiggest;
		if (FindBiggestBlob_APP(matThreshold, matBiggest) == E_ERROR_CODE_FALSE)
		{
			return E_ERROR_CODE_FALSE;
		}
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("FindBiggestBlob_APP End."));

		GetCheckMaskSize(nDilate);
		if (nDilate - 4 >= 3)
		{
			cv::morphologyEx(matBiggest, matBiggest, MORPH_ERODE, cv::getStructuringElement(MORPH_RECT, cv::Size(nDilate - 4, nDilate - 4)));
		}
		cv::morphologyEx(matBiggest, matBiggest, MORPH_DILATE, cv::getStructuringElement(MORPH_RECT, cv::Size(nDilate, nDilate)));

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Morphlogy End."));

		cv::bitwise_and(matTempBuf, matBiggest, matTempBuf);
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("bitwise_and End."));

		cv::blur(matTempBuf, matTempBuf, cv::Size(3, 3));
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("blur End."));

		//////////////////////////////////////////////////////////////////////////
				//查找Line&RANSAC
		//////////////////////////////////////////////////////////////////////////	
#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(4)
#endif
		for (int nType = E_ALIGN_TYPE_LEFT; nType <= E_ALIGN_TYPE_BOTTOM; nType++)
		{
			nErrorCode |= ObjectOutAreaGetLine(matTempBuf, rtCell, ldValueA[nType], ldValueB[nType], nMinSamples, distThreshold, nType, nPanelEdgeTheta, fAvgOffset);
		}
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("FitLine."));

		//如果有错误,则输出错误代码
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			return nErrorCode;
		}

		//////////////////////////////////////////////////////////////////////////
				//查找4个拐角转角
		//////////////////////////////////////////////////////////////////////////
		double dA1, dB1, dA2, dB2 = 0.0;

		for (int i = E_CORNER_LEFT_TOP; i <= E_CORNER_LEFT_BOTTOM; i++)
		{
			dA1 = ldValueA[i];
			dB1 = ldValueB[i];
			dA2 = ldValueA[(i + 1) < 4 ? i + 1 : 0];
			dB2 = ldValueB[(i + 1) < 4 ? i + 1 : 0];

			ptResCorner[i].x = (int)(((dB2 - dB1) / (dA1 - dA2)) + 0.5f);
			ptResCorner[i].y = (int)(((dA2 * dB1 - dA1 * dB2) / (dA2 - dA1)) + 0.5f);

			if (ptResCorner[i].x < 0)			return E_ERROR_CODE_ALIGN_IMAGE_OVER;
			if (ptResCorner[i].y < 0)			return E_ERROR_CODE_ALIGN_IMAGE_OVER;
			if (ptResCorner[i].x >= nWidth)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;
			if (ptResCorner[i].y >= nHeight)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
		}
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Find 4-Corner."));

		//使用Top Line(长边)求角度
		dTheta = atan(ldValueA[E_ALIGN_TYPE_TOP]) * 180. / PI;
		if (abs(dTheta) > dAngleError)									//角度太大时出错。
		{
			return E_ERROR_CODE_ALIGN_ANGLE_RANGE_ERROR;
		}
		else if (abs(dTheta) > dAngleWarning)					//角度太大时出错。(警报)
		{
			return E_ERROR_CODE_ALIGN_ANGLE_RANGE_WARNING;
		}

		if (DoRotateImage(matTempBuf, matTempBuf, dTheta) == E_ERROR_CODE_EMPTY_BUFFER)
		{
			return E_ERROR_CODE_EMPTY_BUFFER;
		}
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("RotateImage End."));

		cv::Point ptRotateCorner[E_CORNER_END];
		for (int nCornerInx = E_CORNER_LEFT_TOP; nCornerInx <= E_CORNER_LEFT_BOTTOM; nCornerInx++)
		{
			DoRotatePoint(ptResCorner[nCornerInx], ptRotateCorner[nCornerInx], cv::Point(matTempBuf.cols / 2, matTempBuf.rows / 2), dTheta);
		}
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("RotatePoint End."));

		int nNormalLength = (int)(dPannelSizeX * 1000 / dCamResolution);
		int nHorizenLength = ptRotateCorner[E_CORNER_RIGHT_TOP].x - ptRotateCorner[E_CORNER_LEFT_TOP].x;

		if (bUseOverRange == true)
		{
			int OverRange = abs(nNormalLength - nHorizenLength);
			if (OverRange > dLengthError)
			{
				return E_ERROR_CODE_ALIGN_LENGTH_RANGE_ERROR;
			}
		}

		//int nADThresh = (int)dPara[E_PARA_APP_AD_THRESHOLD];
		//int dCompare_Theta = (double)dPara[E_PARA_APP_PAD_EDGE_THETA];
		//cv::Rect rtObject = cv::Rect(ptRotateCorner[E_CORNER_LEFT_TOP], ptRotateCorner[E_CORNER_RIGHT_BOTTOM]);

		//cv::threshold(matTempBuf, matTempBuf, nADThresh, 255, THRESH_BINARY);

				//查找PAD Cutting异常情况
		//if (Check_Abnormal_PADEdge(matTempBuf, nThreshold, dCompare_Theta, rtObject) == E_ERROR_CODE_FALSE)
		//{
		//	return E_ERROR_CODE_ALIGN_DISPLAY;
		//}
	}
	catch (const std::exception&)
	{
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("DoFindActive_APP."));

		return E_ERROR_CODE_FALSE;
	}

	return E_ERROR_CODE_TRUE;
}

//查找AVI扫描区域
long CInspectAlign::DoFindActive_SVI(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, int nCameraNum, int nRatio, cv::Point& ptCenter, wchar_t* strID)
{
	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nThreshold = dPara[E_PARA_SVI_ALIGN_THRESHOLD];

	//点灯:点灯面积(Pixel数量)
	int		nMinArea = (int)(dPara[E_PARA_SVI_CELL_COAX_SIZE_X] * dPara[E_PARA_SVI_CELL_COAX_SIZE_Y] * nRatio * nRatio);

	//Side相机
	if (nCameraNum == 1)
		nMinArea = (int)(dPara[E_PARA_SVI_CELL_SIDE_SIZE_X] * dPara[E_PARA_SVI_CELL_SIDE_SIZE_Y] * nRatio * nRatio);

	//向内检查
//int		nRoundIn		= (int)dPara[E_PARA_SVI_ROUND_IN];

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////
		//Gray&二进制
	//////////////////////////////////////////////////////////////////////////

	// Color -> Gray
	cv::Mat matGrayBuf;
	cv::cvtColor(matSrcBuf, matGrayBuf, COLOR_RGB2GRAY);

	// Threshold
	cv::Mat matTempBuf;
	cv::threshold(matGrayBuf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);

	//////////////////////////////////////////////////////////////////////////
		//查找Cell区域
	//////////////////////////////////////////////////////////////////////////

		//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matTempBuf, nMinArea, rectCell);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matGrayBuf.release();
		matTempBuf.release();
		return nErrorCode;
	}

	//Cell中心点
	ptCenter.x = rectCell.x + rectCell.width / 2;
	ptCenter.y = rectCell.y + rectCell.height / 2;

	//////////////////////////////////////////////////////////////////////////
		//Convex Hull-用于填充凹槽部分
	//////////////////////////////////////////////////////////////////////////

		//拯救Contours
	vector<vector<cv::Point>>	contours;
	vector<vector<cv::Point>>().swap(contours);

	cv::findContours(matTempBuf, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

	matTempBuf.setTo(0);

	vector<cv::Point> ptContours, ptConvexHull;
	for (int m = 0; m < contours.size(); m++)
	{
		for (int k = 0; k < contours.at(m).size(); k++)
			ptContours.push_back(contours[m][k]);
	}

	// convexHull
	cv::convexHull(ptContours, ptConvexHull);
	cv::fillConvexPoly(matTempBuf, ptConvexHull, cv::Scalar(255, 255, 255));

	vector<vector<cv::Point>>().swap(contours);
	vector<cv::Point>().swap(ptContours);
	vector<cv::Point>().swap(ptConvexHull);

	//////////////////////////////////////////////////////////////////////////
		//查找点亮区域转角点(Fit)
	//////////////////////////////////////////////////////////////////////////

	long double	dValueAA[E_ALIGN_TYPE_END], dValueBB[E_ALIGN_TYPE_END];

	//4方向的管线管接头
	for (int nType = E_ALIGN_TYPE_LEFT; nType <= E_ALIGN_TYPE_BOTTOM; nType++)
	{
		nErrorCode = RobustFitLine(matTempBuf, rectCell, dValueAA[nType], dValueBB[nType], nMinSamples, distThreshold, nType, 10);

		//如果有错误,则输出错误代码
		if (nErrorCode != E_ERROR_CODE_TRUE)
			return nErrorCode;
	}

	//查找4个拐角转角
	{
		for (int i = E_CORNER_LEFT_TOP; i <= E_CORNER_LEFT_BOTTOM; i++)
		{
			double dA1 = dValueAA[i];
			double dB1 = dValueBB[i];
			double dA2 = dValueAA[(i + 1) % E_CORNER_END];
			double dB2 = dValueBB[(i + 1) % E_CORNER_END];

			ptResCorner[i].x = (int)((dB2 - dB1) / (dA1 - dA2));
			ptResCorner[i].y = (int)((dA2 * dB1 - dA1 * dB2) / (dA2 - dA1));
		}
	}

	//求角度
	dTheta = atan(dValueAA[E_ALIGN_TYPE_TOP]) * 180.0 / PI;

	//异常处理
	if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
	if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;

	//////////////////////////////////////////////////////////////////////////
		//稍微缩小到区域内部...
	//////////////////////////////////////////////////////////////////////////

	//ptResCorner[E_CORNER_LEFT_TOP].x		+= nRoundIn;
	//ptResCorner[E_CORNER_LEFT_TOP].y		+= nRoundIn;
	//ptResCorner[E_CORNER_RIGHT_TOP].x		-= nRoundIn;
	//ptResCorner[E_CORNER_RIGHT_TOP].y		+= nRoundIn;
	//ptResCorner[E_CORNER_LEFT_BOTTOM].x	+= nRoundIn;
	//ptResCorner[E_CORNER_LEFT_BOTTOM].y	-= nRoundIn;
	//ptResCorner[E_CORNER_RIGHT_BOTTOM].x	-= nRoundIn;
	//ptResCorner[E_CORNER_RIGHT_BOTTOM].y	-= nRoundIn;

		//禁用内存
	matGrayBuf.release();
	matTempBuf.release();

	//////////////////////////////////////////////////////////////////////////
		//检查结果
	//////////////////////////////////////////////////////////////////////////

	if (FALSE)
		//if( TRUE )
	{
		cv::Mat matRectBuf;

		//如果是颜色
		if (matSrcBuf.channels() != 1)	matRectBuf = matSrcBuf.clone();

		//黑白
		else	cv::cvtColor(matSrcBuf, matRectBuf, COLOR_GRAY2RGB);

		for (int k = 0; k < E_CORNER_END; k++)
			cv::line(matRectBuf, ptResCorner[k], ptResCorner[(k + 1) % E_CORNER_END], cv::Scalar(0, 255, 0), 1);		// Fit

		//可执行驱动器D:\不固定-必要时利用InspectLibLog的GETDRV()
		// 		CString strTemp;
		// 		strTemp.Format(_T("E:\\IMTC\\Active.bmp"));
		// 		cv::imwrite( (cv::String)(CStringA)strTemp, matRectBuf);
	}

	return E_ERROR_CODE_TRUE;
}

//设置外围Round&Camera Hole曲线&保存文件
long CInspectAlign::SetFindContour(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRountROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath)
{

	//nRatio:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//排除模式
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_BLACK)		return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_PCD)	return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_VINIT)	return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_128)	return E_ERROR_CODE_TRUE;

	//17.11.14-不包括Dust模式
	//如果是Round,则存在背景
	//复制和使用White图案轮廓
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST)		return E_ERROR_CODE_TRUE;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)		return E_ERROR_CODE_TRUE; //跳过背光DUST和DUST圆角区域设置 hjf
	cv::Mat matTempBuf;

	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);
	int		nThreshold = dPara[E_PARA_ALIGN_THRESHOLD];
	int		nMorp = dPara[E_PARA_ALIGN_MORP];
	double	dAngleError = dPara[E_PARA_ALIGN_ANGLE_ERR];
	double	dAngleWarning = dPara[E_PARA_ALIGN_ANGLE_WAR];

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////
		//仅用于AVI,用于在Cell数组之间填充空格
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matSrc8bit = cv::Mat::zeros(matSrcBuf.size(), CV_8UC1);		//内存分配

	if (matSrcBuf.type() == CV_8UC1)
		matSrcBuf.copyTo(matSrc8bit);
	else
		matSrcBuf.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

		//Morphology Close(在Cell之间填充空格)
		cv::morphologyEx(matSrc8bit, matTempBuf, MORPH_CLOSE, StructElem);

		StructElem.release();
	}
	else
		matTempBuf = matSrc8bit.clone();		//内存分配

	// Threshold
	cv::threshold(matTempBuf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);

	//////////////////////////////////////////////////////////////////////////
		//查找Cell区域
	//////////////////////////////////////////////////////////////////////////

		//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matTempBuf, nMinArea, rectCell);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matTempBuf.release();
		return nErrorCode;
	}

	//////////////////////////////////////////////////////////////////////////
		//画面旋转
	//////////////////////////////////////////////////////////////////////////

	long double	dValueA, dValueB;

	//查找Top直线
	nErrorCode = RobustFitLine(matTempBuf, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_TOP);

	//计算旋转坐标时,使用
	double dTheta = atan(dValueA) * 180. / PI;
	DoRotateImage(matTempBuf, matTempBuf, dTheta);

	//cv::Mat matSaveBuf = matTempBuf.clone();

	//////////////////////////////////////////////////////////////////////////
		//查找点亮区域
	//////////////////////////////////////////////////////////////////////////	

		//整个外角线
	vector< vector< cv::Point2i > > contours;
	cv::findContours(matTempBuf, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	cv::RotatedRect BoundingBox = cv::minAreaRect(contours[0]);

	cv::Point2f vertices[E_CORNER_END];
	BoundingBox.points(vertices);

	//////////////////////////////////////////////////////////////////////////
		//查找4个拐角转角
	//////////////////////////////////////////////////////////////////////////

	cv::Point	ptCorner[E_CORNER_END];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	nErrorCode = FindCornerPoint(vertices, ptCorner, nWidth, nHeight);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//////////////////////////////////////////////////////////////////////////
	// Contour
	//////////////////////////////////////////////////////////////////////////

		//Pixel范围
	int nOffsetL = 0;

	//将范围设置为大范围
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x) - nOffsetL,
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y) - nOffsetL,
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x) + nOffsetL,
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y) + nOffsetL);

	//需要校正曲线区域设置区域(顶点校正)
	CRect	rectTemp[MAX_MEM_SIZE_E_INSPECT_AREA];
	bool	nInside[MAX_MEM_SIZE_E_INSPECT_AREA][E_CORNER_END];

	for (int i = 0; i < nRountROICnt; i++)
	{
		//导入UI设置的区域
		rectTemp[i] = RoundROI[i].rectROI;

		//PS模式校正
		rectTemp[i].left *= nRatio;
		rectTemp[i].top *= nRatio;
		rectTemp[i].right *= nRatio;
		rectTemp[i].bottom *= nRatio;

		//以Left-Top坐标为原点的坐标值
		//根据当前点亮区域进行校正
		rectTemp[i].OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

		//异常处理
		if (rectTemp[i].left < 0)		rectTemp[i].left = 0;
		if (rectTemp[i].top < 0)		rectTemp[i].top = 0;
		if (rectTemp[i].right < 0)		rectTemp[i].right = 0;
		if (rectTemp[i].bottom < 0)		rectTemp[i].bottom = 0;

		if (rectTemp[i].left >= matTempBuf.cols)	rectTemp[i].left = matTempBuf.cols - 1;
		if (rectTemp[i].top >= matTempBuf.rows)	rectTemp[i].top = matTempBuf.rows - 1;
		if (rectTemp[i].right >= matTempBuf.cols)	rectTemp[i].right = matTempBuf.cols - 1;
		if (rectTemp[i].bottom >= matTempBuf.rows)	rectTemp[i].bottom = matTempBuf.rows - 1;

		//检查顶点是否存在于Cell点灯区域内
		nInside[i][E_CORNER_LEFT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].left) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_LEFT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].left) != 0) ? 1 : 0;
	}

	//用于检查结果
//{
//	for (int i=0 ; i<MAX_MEM_SIZE_E_INSPECT_AREA ; i++)
//	{
//		if( RoundROI[i].bUseROI )
//		{
//			CRect rect = rectTemp[i];
//
//			cv::rectangle(matSaveBuf, cv::Rect(rect.left, rect.top, rect.Width(), rect.Height()), cv::Scalar(128, 128, 128) );
//		}
//	}
//
//	cv::imwrite("E:\\IMTC\\temp.bmp", matSaveBuf);
//	matSaveBuf.release();
//}

	//曲线外角线
	vector< cv::Point2i > ptContours[MAX_MEM_SIZE_E_INSPECT_AREA];
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		vector< cv::Point2i >().swap(ptContours[i]);

	//顶点数量
	for (int i = 0; i < contours.size(); i++)
	{
		for (int j = 0; j < contours[i].size(); j++)
		{
			//UI设置的区域数量
			for (int k = 0; k < nRountROICnt; k++)
			{
				//不使用
				if (!RoundROI[k].bUseROI)	continue;

				//如果坐标在校正区域内
				if (rectTemp[k].PtInRect(CPoint(contours[i][j].x, contours[i][j].y)))
				{
					//添加坐标
					ptContours[k].push_back(cv::Point2i(contours[i][j]));
					break;
				}
			}
		}
		//初始化
		vector< cv::Point2i >().swap(contours[i]);
	}

	//////////////////////////////////////////////////////////////////////////
		//排序
		//多边形&填充内部颜色时,需要按顺序...
	//////////////////////////////////////////////////////////////////////////

	cv::Point2i ptTempS;
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		for (int j = 0; j < ptContours[i].size(); j++)
		{
			for (int m = j + 1; m < ptContours[i].size(); m++)
			{
				//对齐y坐标,使其向上移动
				if (ptContours[i][j].y > ptContours[i][m].y)
				{
					ptTempS = ptContours[i][j];
					ptContours[i][j] = ptContours[i][m];
					ptContours[i][m] = ptTempS;
				}
				//如果y坐标相同
				else if (ptContours[i][j].y == ptContours[i][m].y)
				{
					//更改距离转角点较远的情况
					if (abs(ptCorner[i].x - ptContours[i][j].x) < abs(ptCorner[i].x - ptContours[i][m].x))
					{
						ptTempS = ptContours[i][j];
						ptContours[i][j] = ptContours[i][m];
						ptContours[i][m] = ptTempS;
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Save
	//////////////////////////////////////////////////////////////////////////

	char szPath[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strPath, -1, szPath, sizeof(szPath), NULL, NULL);

	//文件存储路径
	CString str;
	str.Format(_T("%s\\CornerEdge"), strPath);
	CreateDirectory(str, NULL);

	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		//如果没有个数,则排除
		if (ptContours[i].size() <= 0)	continue;

		//////////////////////////////////////////////////////////////////////////
				//查找设置区域距离点亮的Cell顶点最近的顶点
		//////////////////////////////////////////////////////////////////////////

		int nCx = (rectTemp[i].left + rectTemp[i].right) / 2;
		int nCy = (rectTemp[i].top + rectTemp[i].bottom) / 2;

		//与LT的距离
		int nLength = abs(ptCorner[E_CORNER_LEFT_TOP].x - nCx) + abs(ptCorner[E_CORNER_LEFT_TOP].y - nCy);
		int	nIndex = E_CORNER_LEFT_TOP;
		for (int j = 1; j < E_CORNER_END; j++)
		{
			int nTempLenght = abs(ptCorner[j].x - nCx) + abs(ptCorner[j].y - nCy);

			//寻找小距离的顶点
			if (nLength > nTempLenght)
			{
				nLength = nTempLenght;
				nIndex = j;
			}
		}

		//////////////////////////////////////////////////////////////////////////		

				//文件存储路径
		CStringA strTemp;
		strTemp.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, GetPatternStringA(nAlgImg), i);

		//打开文件(Unicode环境"t"->"wt")
		FILE* out = NULL;
		fopen_s(&out, strTemp, "wt");

		if (out != NULL)
		{
			//近索引标记
			fprintf_s(out, "CornerIndex%d\n", nIndex);

			//检查顶点是否存在于Cell区域中
			fprintf_s(out, "CornerInside%d,%d,%d,%d\n", nInside[i][0], nInside[i][1], nInside[i][2], nInside[i][3]);

			for (int j = 0; j < ptContours[i].size(); j++)
			{
				//拐角的原点&角度为0度
				fprintf_s(out, "%d,%d\n", ptContours[i][j].x - ptCorner[nIndex].x, ptContours[i][j].y - ptCorner[nIndex].y);
			}

			//关闭文件
			fclose(out);
			out = NULL;

			//17.11.14-White时复制到Dust模式
			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_DUST), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}

			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CornerEdge\\%sPS_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_GRAY_64), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}
		}

		//初始化
		vector< cv::Point2i >().swap(ptContours[i]);
	}

	//禁用内存
	matTempBuf.release();

	return E_ERROR_CODE_TRUE;
}
long CInspectAlign::SetFindContour_APP(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRountROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath, Point* ptAlignCorner, CStringA strImageName, double dAlignTH, bool bImageSave)
{
	int nImageSaveInx = 0;

	CString strSavePath;
	CString strSaveName;
	strSavePath.Format(_T("E:\\IMTC\\ActiveMaskPrc2"));
	if (bImageSave)
	{
		//检查路径
		DWORD result;
		if (((result = GetFileAttributes(strSavePath)) == -1) || !(result & FILE_ATTRIBUTE_DIRECTORY)) {
			CreateDirectory(strSavePath, NULL);
		}
	}

	long nErrorCode = E_ERROR_CODE_TRUE;

	float	fResize = (float)dPara[E_PARA_APP_POLYGON_RESIZE];							//要重置的画面大小
	int		nGausBlurSize = (int)dPara[E_PARA_APP_POLYGON_GAUS_SIZE];							//用于去除噪音的Gaussian Blur Mask Size
	float	fGausSigma = (float)dPara[E_PARA_APP_POLYGON_GAUS_SIGMA];						//用于去除噪音的Gaussian Blur Sigma值
	int		nThreshold = (int)dPara[E_PARA_APP_POLYGON_THRESHOLD];							//Sobel Edge处理后仅提取所需线条的阈值
	int		nOpenSize = (int)dPara[E_PARA_APP_POLYGON_OPEN_SIZE];							//用于明确Panel轮廓,BM轮廓和Active轮廓的线边界的开放运算大小
	int		nSelectBlob = (int)dPara[E_PARA_APP_POLYGON_SELECT_BLOB];						//在顺从的Blob中选择第几大
	int		nDilateSize = (int)dPara[E_PARA_APP_POLYGON_DILATE_SIZE];						//用于填充Active Mask上的孔的膨胀运算大小
	int		nErodeSize = (int)dPara[E_PARA_APP_POLYGON_ERODE_SIZE];						//侵蚀计算大小,用于将Active Mask中的孔填充时增大的大小恢复到原来的大小。

	//图像处理Mask Size异常处理
	GetCheckMaskSize(nGausBlurSize);
	GetCheckMaskSize(nOpenSize);
	GetCheckMaskSize(nErodeSize);

	Point ptPanelCorner[E_CORNER_END];
	for (int nCorner = E_CORNER_LEFT_TOP; nCorner < E_CORNER_END; nCorner++)
	{
		ptPanelCorner[nCorner] = ptAlignCorner[nCorner];
	}

	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	nErrorCode = E_ERROR_CODE_ALIGN_ROUND_SETTING;

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, matSrcBuf);
	}

	Mat OrgImg;
	matSrcBuf.copyTo(OrgImg);

	DoRotateImage(OrgImg, OrgImg, dAlignTH);

	Mat OrgRotateImage;
	OrgImg.copyTo(OrgRotateImage);

	cv::resize(OrgImg, OrgImg, cv::Size(), fResize, fResize);

	cv::GaussianBlur(OrgImg, OrgImg, Size(nGausBlurSize, nGausBlurSize), fGausSigma);

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, OrgImg);
	}

	Mat grad_X, grad_Y;
	Mat abs_grad_X, abs_grad_Y;
	int nDepth = CV_16S;

	Sobel(OrgImg, grad_X, nDepth, 1, 0, 3);
	convertScaleAbs(grad_X, abs_grad_X);

	Sobel(OrgImg, grad_Y, nDepth, 0, 1, 3);
	convertScaleAbs(grad_Y, abs_grad_Y);

	cv::addWeighted(abs_grad_X, 0.5, abs_grad_Y, 0.5, 0, OrgImg);

	cv::threshold(OrgImg, OrgImg, nThreshold, 255, CV_THRESH_BINARY);

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, OrgImg);
	}

	//返回最大的Blob
	Mat mtBiggist;
	mtBiggist = Mat::zeros(OrgImg.size(), OrgImg.type());
	FindBiggestBlob(OrgImg, mtBiggist);

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, mtBiggist);
	}

	//反转
	cv::bitwise_not(mtBiggist, mtBiggist);

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, mtBiggist);
	}

	//断开活动区域
	cv::morphologyEx(mtBiggist, mtBiggist, cv::MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(nOpenSize, nOpenSize)));

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, mtBiggist);
	}

	//返回选定顺序中最大的Blob
	Mat mtSelectBig;
	mtSelectBig = Mat::zeros(mtBiggist.size(), mtBiggist.type());
	SelectBiggestBlob(mtBiggist, mtSelectBig, nSelectBlob);

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, mtSelectBig);
	}

	//在二进制活动区域中填充漏洞
	cv::morphologyEx(mtSelectBig, mtSelectBig, cv::MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(nDilateSize, nDilateSize)));
	cv::morphologyEx(mtSelectBig, mtSelectBig, cv::MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(nErodeSize, nErodeSize)));

	if (bImageSave)
	{
		strSaveName.Format(_T("%s\\%d.bmp"), strSavePath, nImageSaveInx++);
		imwrite((cv::String)(CStringA)strSaveName, mtSelectBig);
	}

	//恢复到原始画面大小
	cv::resize(mtSelectBig, mtSelectBig, cv::Size(), 1 / fResize, 1 / fResize);

	vector<vector<cv::Point>> ptSelectBig;
	findContours(mtSelectBig, ptSelectBig, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

	Mat mtActiveArea;
	mtActiveArea = Mat::zeros(mtSelectBig.size(), mtSelectBig.type());
	drawContours(mtActiveArea, ptSelectBig, 0, Scalar(255));

	//////////////////////////////////////////////////////////////////////////
	// Save
	//////////////////////////////////////////////////////////////////////////

	char szPath[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strPath, -1, szPath, sizeof(szPath), NULL, NULL);

	//文件存储路径
	CString str;
	str.Format(_T("%s\\CornerEdge"), strPath);
	CreateDirectory(str, NULL);

	//获取Polygon ROI
	Rect* rtPolygonROI;
	rtPolygonROI = new Rect[nRountROICnt];

	for (int nROIInx = 0; nROIInx < nRountROICnt; nROIInx++)
	{
		rtPolygonROI[nROIInx] = Rect(Point(RoundROI[nROIInx].rectROI.left + ptPanelCorner[E_CORNER_LEFT_TOP].x,
			RoundROI[nROIInx].rectROI.top + ptPanelCorner[E_CORNER_LEFT_TOP].y),
			Point(RoundROI[nROIInx].rectROI.right + ptPanelCorner[E_CORNER_LEFT_TOP].x,
				RoundROI[nROIInx].rectROI.bottom + ptPanelCorner[E_CORNER_LEFT_TOP].y));

		GetCheckROIOver(rtPolygonROI[nROIInx], Rect(0, 0, matSrcBuf.cols, matSrcBuf.rows), rtPolygonROI[nROIInx]);

		CStringA strROI;
		strROI.Format("%s\\CornerEdge\\%d.bmp", szPath, nROIInx);
		imwrite((cv::String)strROI, OrgRotateImage(rtPolygonROI[nROIInx]));

		//文件存储路径
		CStringA strTemp;
		strTemp.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, strImageName, nROIInx);

		//打开文件(Unicode环境"t"->"wt")
		FILE* out = NULL;
		fopen_s(&out, strTemp, "wt");

		if (out != NULL)
		{
			vector<vector<cv::Point>> ptActiveArea;

			findContours(mtActiveArea(rtPolygonROI[nROIInx]), ptActiveArea, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

			//如果没有Contour
			if (ptActiveArea.size() < 1)
			{
				//关闭文件
				fclose(out);
				out = NULL;
				continue;
			}

			int nPanelCenterX = ptPanelCorner[E_CORNER_LEFT_TOP].x + (ptPanelCorner[E_CORNER_RIGHT_TOP].x - ptPanelCorner[E_CORNER_LEFT_TOP].x) / 2;

			Point ptTempS;

			for (int j = 0; j < ptActiveArea[0].size(); j++)
			{
				for (int m = j + 1; m < ptActiveArea[0].size(); m++)
				{
					if (nPanelCenterX > rtPolygonROI[nROIInx].tl().x)
					{
						//y坐标升序
						if (ptActiveArea[0][j].y > ptActiveArea[0][m].y)
						{
							ptTempS = ptActiveArea[0][j];
							ptActiveArea[0][j] = ptActiveArea[0][m];
							ptActiveArea[0][m] = ptTempS;
						}
					}
					else
					{
						//y坐标降序
						if (ptActiveArea[0][j].y < ptActiveArea[0][m].y)
						{
							ptTempS = ptActiveArea[0][j];
							ptActiveArea[0][j] = ptActiveArea[0][m];
							ptActiveArea[0][m] = ptTempS;
						}
					}

				}
			}

			fprintf_s(out, "CornerIndex %d\n", (int)ptActiveArea[0].size());

			for (int j = 0; j < ptActiveArea[0].size(); j++)
			{
				fprintf_s(out, "%d,%d\n", ptActiveArea[0][j].x, ptActiveArea[0][j].y);
			}

			//关闭文件
			fclose(out);
			out = NULL;
		}
	}

	delete[] rtPolygonROI;

	return E_ERROR_CODE_TRUE;
}

//设置外围Round&Camera Hole曲线&保存文件
long CInspectAlign::SetFindContour_(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nCHoleROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath)
{
	//排除模式
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_BLACK)		return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_PCD)	return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_VINIT) return E_ERROR_CODE_TRUE;

	//2022.10.14 G3外围填充test
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_128)
	{
		double dMulti = 0;
		int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);	// 3800 * 1900;	// APP
		dMulti = CenterMeanGV(matSrcBuf, nMinArea);

		matSrcBuf *= dMulti;
	}

	//17.11.14-不包括Dust模式
	//如果是Round,则存在背景
	//复制和使用White图案轮廓
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST)		return E_ERROR_CODE_TRUE;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)		return E_ERROR_CODE_TRUE; //跳过背光DUST和DUST圆角区域设置 hjf
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nMorp = dPara[E_PARA_ALIGN_MORP];
	int		nThreshold = dPara[E_PARA_ALIGN_THRESHOLD];
	int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);

	bool	bRoundSet = (dPara[E_PARA_ROUND_SETTING] > 0) ? true : false;
	bool	bCHoleSet = (dPara[E_PARA_CHOLE_SETTING] > 0) ? true : false;
	bool	bRoundAuto = (dPara[E_PARA_ROUND_AUTO] > 0) ? true : false;
	//////////////////////////////////////////////////////////////////////////

		//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//缓冲区分配和初始化

//////////////////////////////////////////////////////////////////////////
// ShiftCopy
//////////////////////////////////////////////////////////////////////////
	cv::Mat matDstBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type());
	//需要添加参数

	//稍微缩小到区域内部...(区域)
	int nInPixel = 3;

	//如果处于PS模式
	if (nRatio != 1)
	{
		//获取Shift Copy Parameter
		int		nRedPattern = (int)dPara[E_PARA_SHIFT_COPY_R];
		int		nGreenPattern = (int)dPara[E_PARA_SHIFT_COPY_G];
		int		nBluePattern = (int)dPara[E_PARA_SHIFT_COPY_B];

		int nCpyX = 0, nCpyY = 0, nLoopX = 0, nLoopY = 0;

		//按模式...
		switch (nAlgImg)
		{
		case E_IMAGE_CLASSIFY_AVI_R:
		{
			if (nRedPattern == 0) break;
			ShiftCopyParaCheck(nRedPattern, nCpyX, nCpyY, nLoopX, nLoopY);
			nErrorCode = AlgoBase::ShiftCopy(matSrcBuf, matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
			matDstBuf.copyTo(matSrcBuf);
			matDstBuf.release();
		}
		break;

		case E_IMAGE_CLASSIFY_AVI_G:
		{
			if (nGreenPattern == 0) break;
			ShiftCopyParaCheck(nGreenPattern, nCpyX, nCpyY, nLoopX, nLoopY);
			nErrorCode = AlgoBase::ShiftCopy(matSrcBuf, matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
			matDstBuf.copyTo(matSrcBuf);
			matDstBuf.release();
		}
		break;

		case E_IMAGE_CLASSIFY_AVI_B:
		{
			if (nBluePattern == 0) break;
			ShiftCopyParaCheck(nBluePattern, nCpyX, nCpyY, nLoopX, nLoopY);
			nErrorCode = AlgoBase::ShiftCopy(matSrcBuf, matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
			matDstBuf.copyTo(matSrcBuf);
			matDstBuf.release();
		}
		break;

		default:
			break;
		}

	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("ShiftCopy."));

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		return nErrorCode;
	}

	//////////////////////////////////////////////////////////////////////////
		//仅用于AVI,用于在Cell数组之间填充空格
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matSrc8bit = cMem->GetMat(matSrcBuf.size(), CV_8UC1);

	if (matSrcBuf.type() == CV_8UC1)
		matSrcBuf.copyTo(matSrc8bit);
	else
		matSrcBuf.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

	cv::Mat matTempBuf = cMem->GetMat(matSrc8bit.size(), CV_8UC1);

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

		//Morphology Close(在Cell之间填充空格)
		cv::morphologyEx(matSrc8bit, matTempBuf, MORPH_CLOSE, StructElem);

		StructElem.release();
	}
	else
		matSrc8bit.copyTo(matTempBuf);

	cv::threshold(matTempBuf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);

	//////////////////////////////////////////////////////////////////////////
		//查找Cell区域
	//////////////////////////////////////////////////////////////////////////

		//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matTempBuf, nMinArea, rectCell);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matTempBuf.release();
		return nErrorCode;
	}

	//////////////////////////////////////////////////////////////////////////
		//画面旋转
	//////////////////////////////////////////////////////////////////////////

	long double	dValueA, dValueB;

	//查找Top直线
	nErrorCode = RobustFitLine(matTempBuf, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_TOP);

	//计算旋转坐标时,使用
	double dTheta = atan(dValueA) * 180. / PI;
	DoRotateImage(matTempBuf, matTempBuf, dTheta);

	//////////////////////////////////////////////////////////////////////////
	//查找点亮区域&查找Hole
	//////////////////////////////////////////////////////////////////////////	

	//外部外角线&内部外角线
	vector< vector< cv::Point2i > > contours;
	int nContourSize = 0;
	int nContourIdx = 0;
	cv::findContours(matTempBuf, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++)
	{
		if (nContourSize < contours[i].size())
		{
			nContourSize = contours[i].size();
			nContourIdx = i;
		}
	}

	cv::RotatedRect BoundingBox = cv::minAreaRect(contours[nContourIdx]);

	cv::Point2f vertices[E_CORNER_END];
	BoundingBox.points(vertices);

	//////////////////////////////////////////////////////////////////////////
	//查找4个拐角转角
	//////////////////////////////////////////////////////////////////////////

	cv::Point	ptCorner[E_CORNER_END];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	nErrorCode = FindCornerPoint(vertices, ptCorner, nWidth, nHeight);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//////////////////////////////////////////////////////////////////////////
	// Contour
	//////////////////////////////////////////////////////////////////////////

	if (bRoundSet) {
		if (!bRoundAuto) {
			SetFindRound(matTempBuf, contours, ptCorner, RoundROI, nRoundROICnt, nContourIdx, nAlgImg, nRatio, strPath);
		}
		else {
			SetFindRoundAuto(matTempBuf, contours, ptCorner, nContourIdx, nAlgImg, nRatio, dTheta, strPath);
		}
	}

	if (bCHoleSet)
		SetFindCHole(matTempBuf, contours, ptCorner, CHoleROI, nCHoleROICnt, nContourIdx, nAlgImg, nRatio, strPath);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return nErrorCode;
}

//设置外围Round&Camera Hole曲线&保存文件
long CInspectAlign::SetFindContour_2(cv::Mat* matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nCHoleROICnt, double* dPara, int nAlgImg, int nRatio, CString strPath)
{
	//排除模式
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_BLACK)		return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_PCD)	return E_ERROR_CODE_TRUE;
	else if (nAlgImg == E_IMAGE_CLASSIFY_AVI_VINIT) return E_ERROR_CODE_TRUE;

	//17.11.14-不包括Dust模式
	//如果是Round,则存在背景
	//复制和使用White图案轮廓
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST)		return E_ERROR_CODE_TRUE;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)		return E_ERROR_CODE_TRUE; //跳过背光DUST和DUST圆角区域设置 hjf

	//如果没有缓冲区。
//if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

//////////////////////////////////////////////////////////////////////////
	//参数
//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nMorp = dPara[E_PARA_ALIGN_MORP];
	int		nThreshold = dPara[E_PARA_ALIGN_THRESHOLD];
	int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);

	bool	bCHoleSet = (dPara[E_PARA_CHOLE_POINT_SETTING] > 0) ? true : false;
	//////////////////////////////////////////////////////////////////////////

		//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//缓冲区分配和初始化

//////////////////////////////////////////////////////////////////////////
// ShiftCopy
//////////////////////////////////////////////////////////////////////////
	cv::Mat matDstBuf = cMem->GetMat(matSrcBuf[0].size(), matSrcBuf[0].type());
	//需要添加参数

	//稍微缩小到区域内部...(区域)
	int nInPixel = 3;

	//如果处于PS模式
	if (nRatio != 1)
	{
		//获取Shift Copy Parameter
		int		nRedPattern = (int)dPara[E_PARA_SHIFT_COPY_R];
		int		nGreenPattern = (int)dPara[E_PARA_SHIFT_COPY_G];
		int		nBluePattern = (int)dPara[E_PARA_SHIFT_COPY_B];

		int nCpyX = 0, nCpyY = 0, nLoopX = 0, nLoopY = 0;

		//按模式...

		ShiftCopyParaCheck(nRedPattern, nCpyX, nCpyY, nLoopX, nLoopY);
		nErrorCode = AlgoBase::ShiftCopy(matSrcBuf[1], matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
		matDstBuf.copyTo(matSrcBuf[1]);
		matDstBuf.release();

	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("ShiftCopy."));

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		return nErrorCode;
	}

	//////////////////////////////////////////////////////////////////////////
		//仅用于AVI,用于在Cell数组之间填充空格
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matSrc8bit = cMem->GetMat(matSrcBuf[1].size(), CV_8UC1);

	if (matSrcBuf[1].type() == CV_8UC1)
		matSrcBuf[1].copyTo(matSrc8bit);
	else
		matSrcBuf[1].convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

	cv::Mat matTempBuf = cMem->GetMat(matSrc8bit.size(), CV_8UC1);

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

		//Morphology Close(在Cell之间填充空格)
		cv::morphologyEx(matSrc8bit, matTempBuf, MORPH_CLOSE, StructElem);

		StructElem.release();
	}
	else
		matSrc8bit.copyTo(matTempBuf);

	cv::Mat matThr;
	cv::threshold(matTempBuf, matThr, nThreshold, 255.0, THRESH_BINARY);

	//////////////////////////////////////////////////////////////////////////
		//查找Cell区域
	//////////////////////////////////////////////////////////////////////////

		//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matThr, nMinArea, rectCell);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matThr.release();
		return nErrorCode;
	}

	//////////////////////////////////////////////////////////////////////////
		//画面旋转
	//////////////////////////////////////////////////////////////////////////

	long double	dValueA, dValueB;

	//查找Top直线
	nErrorCode = RobustFitLine(matTempBuf, rectCell, dValueA, dValueB, nMinSamples, distThreshold, E_ALIGN_TYPE_TOP);

	//计算旋转坐标时,使用
	double dTheta = atan(dValueA) * 180. / PI;
	DoRotateImage(matTempBuf, matTempBuf, dTheta);

	//////////////////////////////////////////////////////////////////////////
		//查找点亮区域&查找Hole
	//////////////////////////////////////////////////////////////////////////	
		//外部外角线&内部外角线
	vector< vector< cv::Point2i > > contours;
	int nContourSize = 0;
	int nContourIdx = 0;

	cv::RotatedRect BoundingBox;
	cv::Point2f vertices[E_CORNER_END];

	//外部外角线&内部外角线

	cv::findContours(matThr, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++)
	{
		if (nContourSize < contours[i].size())
		{
			nContourSize = contours[i].size();
			nContourIdx = i;
		}
	}

	BoundingBox = cv::minAreaRect(contours[nContourIdx]);

	BoundingBox.points(vertices);

	//////////////////////////////////////////////////////////////////////////
		//查找4个拐角转角
	//////////////////////////////////////////////////////////////////////////

	cv::Point	ptCorner[E_CORNER_END];

	long	nWidth = (long)matSrcBuf[1].cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf[1].rows;	// 图像垂直尺寸

	nErrorCode = FindCornerPoint(vertices, ptCorner, nWidth, nHeight);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//在收到的图像中,Chole在R模式中成为诗人,以R模式为基准制作
	cv::bitwise_and(matSrcBuf[1], matThr, matTempBuf);

	cv::GaussianBlur(matTempBuf, matTempBuf, cv::Size(31, 31), 5);

	double dAvg = cv::mean(matTempBuf, matThr)[0];
	matTempBuf -= (dAvg - 10); // 适当的Offset

	double dAvg2 = cv::mean(matTempBuf, matThr)[0];
	matTempBuf *= (dAvg * 2 / dAvg2);	// 用于提高Chole的可视性(使用平方会更好)

	//cv::threshold(matTempBuf, matTempBuf, dAvg, 255.0, THRESH_BINARY_INV);

	///////////////////////////////////////////////////////////////////////////////////

	CRect	rectTemp[MAX_MEM_SIZE_E_INSPECT_AREA];
	//bool	nInside[MAX_MEM_SIZE_E_INSPECT_AREA][E_CORNER_END];
	for (int i = 0; i < nCHoleROICnt; i++)
	{
		//不使用
		if (!CHoleROI[i].bUseROI)	continue;

		//导入UI设置的区域
		rectTemp[i] = CHoleROI[i].rectROI;

		//PS模式校正
		rectTemp[i].left *= nRatio;
		rectTemp[i].top *= nRatio;
		rectTemp[i].right *= nRatio;
		rectTemp[i].bottom *= nRatio;

		//以Left-Top坐标为原点的坐标值
		//根据当前点亮区域进行校正
		rectTemp[i].OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

		//异常处理
		if (rectTemp[i].left < 0)		rectTemp[i].left = 0;
		if (rectTemp[i].top < 0)		rectTemp[i].top = 0;
		if (rectTemp[i].right < 0)		rectTemp[i].right = 0;
		if (rectTemp[i].bottom < 0)		rectTemp[i].bottom = 0;

		if (rectTemp[i].left >= matTempBuf.cols)	rectTemp[i].left = matTempBuf.cols - 1;
		if (rectTemp[i].top >= matTempBuf.rows)	rectTemp[i].top = matTempBuf.rows - 1;
		if (rectTemp[i].right >= matTempBuf.cols)	rectTemp[i].right = matTempBuf.cols - 1;
		if (rectTemp[i].bottom >= matTempBuf.rows)	rectTemp[i].bottom = matTempBuf.rows - 1;

		vector<Vec3f> circles;

		cv::HoughCircles(matTempBuf(Rect(rectTemp[i].left, rectTemp[i].top, rectTemp[i].right - rectTemp[i].left, rectTemp[i].bottom - rectTemp[i].top)), circles, HOUGH_GRADIENT, 1, 100, 25, 60, 60, 0);

		for (int i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center(c[0], c[1]);
			int radius = c[2];

			cv::circle(matThr(Rect(rectTemp[i].left, rectTemp[i].top, rectTemp[i].right - rectTemp[i].left, rectTemp[i].bottom - rectTemp[i].top)), center, radius, 0, -1, -1, 0);
		}

	}

	cv::findContours(matThr, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	nContourSize = 0;
	nContourIdx = 0;

	for (int i = 0; i < contours.size(); i++)
	{
		if (nContourSize < contours[i].size())
		{
			nContourSize = contours[i].size();
			nContourIdx = i;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////
// 	cv::Mat matCHole;
// 	cv::threshold(matTempBuf, matCHole, 45, 255.0, THRESH_BINARY_INV);
// 	cv::threshold(matTempBuf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);
// 
//	//如果左边的部分与Chole区域的亮度相同,则会进行相同的二进制处理
// 	Rect rectROI;
// 	rectROI.x = matCHole.cols / 3;
// 	rectROI.y = 0;
// 	rectROI.width = matCHole.cols * 2 / 3;
// 	rectROI.height = matCHole.rows;
// 	matCHole(rectROI).setTo(0);
// 
// 	cv::bitwise_and(matTempBuf, matCHole, matCHole);
// 	//cv::subtract(matTempBuf, matCHole, matTempBuf);
// 
// 	//////////////////////////////////////////////////////////////////////////
//	//查找Hole
// 	//////////////////////////////////////////////////////////////////////////	
// 
// 	
// 	cv::findContours(matCHole, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
// 
// 	for (int i = 0; i < contours.size(); i++)
// 	{
// 		if (nContourSize > contours[i].size() && contours[i].size()>500)
// 		{
// 			nContourSize = contours[i].size();
// 			nContourIdx = i;
// 		}
// 	}
// 
// 
// 
// 	
// 	BoundingBox.points(vertices);
// 
// 	int nRadius;
// 
// 	if (BoundingBox.size.width >= BoundingBox.size.height)
// 		nRadius = (int)(BoundingBox.size.width / 2);
// 	else
// 		nRadius = (int)(BoundingBox.size.height / 2);
// 
// 	cv::circle(matTempBuf, Point(BoundingBox.center.x, BoundingBox.center.y), nRadius, 0, -1, -1, 0);
// 
// 	matCHole.release();

	//////////////////////////////////////////////////////////////////////////
	// Contour
	//////////////////////////////////////////////////////////////////////////

	if (bCHoleSet)
		SetFindCHole2(matThr, contours, ptCorner, CHoleROI, nCHoleROICnt, nContourIdx, nAlgImg, nRatio, strPath);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return nErrorCode;
}

void CInspectAlign::SetFindRound(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nContourIdx, int nAlgImg, int nRatio, CString strPath)
{
	// Image Save 0 On / -1 Off
	int nSaveImg = -1;
	//需要校正曲线区域设置区域(顶点校正)
	CRect	rectTemp[MAX_MEM_SIZE_E_INSPECT_AREA];
	bool	nInside[MAX_MEM_SIZE_E_INSPECT_AREA][E_CORNER_END];

	for (int i = 0; i < nRoundROICnt; i++)
	{
		//导入UI设置的区域
		rectTemp[i] = RoundROI[i].rectROI;

		//PS模式校正
		rectTemp[i].left *= nRatio;
		rectTemp[i].top *= nRatio;
		rectTemp[i].right *= nRatio;
		rectTemp[i].bottom *= nRatio;

		//以Left-Top坐标为原点的坐标值
		//根据当前点亮区域进行校正
		rectTemp[i].OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

		//异常处理
		if (rectTemp[i].left < 0)		rectTemp[i].left = 0;
		if (rectTemp[i].top < 0)		rectTemp[i].top = 0;
		if (rectTemp[i].right < 0)		rectTemp[i].right = 0;
		if (rectTemp[i].bottom < 0)		rectTemp[i].bottom = 0;

		if (rectTemp[i].left >= matTempBuf.cols)	rectTemp[i].left = matTempBuf.cols - 1;
		if (rectTemp[i].top >= matTempBuf.rows)	rectTemp[i].top = matTempBuf.rows - 1;
		if (rectTemp[i].right >= matTempBuf.cols)	rectTemp[i].right = matTempBuf.cols - 1;
		if (rectTemp[i].bottom >= matTempBuf.rows)	rectTemp[i].bottom = matTempBuf.rows - 1;

		//检查顶点是否存在于Cell点灯区域内
		nInside[i][E_CORNER_LEFT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].left) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_LEFT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].left) != 0) ? 1 : 0;
	}
	//用于检查结果
	if (nSaveImg >= 0)
	{
		cv::Mat matSaveBuf = matTempBuf.clone();
		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			if (RoundROI[i].bUseROI)
			{
				CRect rect = rectTemp[i];

				cv::rectangle(matSaveBuf, cv::Rect(rect.left, rect.top, rect.Width(), rect.Height()), cv::Scalar(128, 128, 128));
			}
		}

		cv::imwrite("E:\\IMTC\\Round.bmp", matSaveBuf);
		matSaveBuf.release();
	}

	//曲线外角线
	vector< cv::Point2i > ptContours[MAX_MEM_SIZE_E_INSPECT_AREA];
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		vector< cv::Point2i >().swap(ptContours[i]);

	//顶点数量
	for (int i = 0; i < contours.size(); i++)
	{
		if (i != nContourIdx) continue;
		for (int j = 0; j < contours[i].size(); j++)
		{
			//UI设置的区域数量
			for (int k = 0; k < nRoundROICnt; k++)
			{
				//不使用
				if (!RoundROI[k].bUseROI)	continue;

				//如果坐标在校正区域内
				if (rectTemp[k].PtInRect(CPoint(contours[i][j].x, contours[i][j].y)))
				{
					//添加坐标
					ptContours[k].push_back(cv::Point2i(contours[i][j]));
					break;
				}
			}
		}
		//初始化
		vector< cv::Point2i >().swap(contours[i]);
	}

	//////////////////////////////////////////////////////////////////////////
		//排序
		//多边形&填充内部颜色时,需要按顺序...
	//////////////////////////////////////////////////////////////////////////

	cv::Point2i ptTempS;
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		for (int j = 0; j < ptContours[i].size(); j++)
		{
			for (int m = j + 1; m < ptContours[i].size(); m++)
			{
				//对齐y坐标,使其向上移动
				if (ptContours[i][j].y > ptContours[i][m].y)
				{
					ptTempS = ptContours[i][j];
					ptContours[i][j] = ptContours[i][m];
					ptContours[i][m] = ptTempS;
				}
				//如果y坐标相同
				else if (ptContours[i][j].y == ptContours[i][m].y)
				{
					//更改距离转角点较远的情况
					if (abs(ptCorner[i].x - ptContours[i][j].x) < abs(ptCorner[i].x - ptContours[i][m].x))
					{
						ptTempS = ptContours[i][j];
						ptContours[i][j] = ptContours[i][m];
						ptContours[i][m] = ptTempS;
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Save
	//////////////////////////////////////////////////////////////////////////

	char szPath[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strPath, -1, szPath, sizeof(szPath), NULL, NULL);

	//文件存储路径
	CString str;
	str.Format(_T("%s\\CornerEdge"), strPath);
	CreateDirectory(str, NULL);

	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		//如果没有个数,则排除
		if (ptContours[i].size() <= 0)	continue;

		//////////////////////////////////////////////////////////////////////////
				//查找设置区域距离点亮的Cell顶点最近的顶点
		//////////////////////////////////////////////////////////////////////////

		int nCx = (rectTemp[i].left + rectTemp[i].right) / 2;
		int nCy = (rectTemp[i].top + rectTemp[i].bottom) / 2;

		//与LT的距离
		int nLength = abs(ptCorner[E_CORNER_LEFT_TOP].x - nCx) + abs(ptCorner[E_CORNER_LEFT_TOP].y - nCy);
		int	nIndex = E_CORNER_LEFT_TOP;
		for (int j = 1; j < E_CORNER_END; j++)
		{
			int nTempLenght = abs(ptCorner[j].x - nCx) + abs(ptCorner[j].y - nCy);

			//寻找小距离的顶点
			if (nLength > nTempLenght)
			{
				nLength = nTempLenght;
				nIndex = j;
			}
		}

		//////////////////////////////////////////////////////////////////////////		

				//文件存储路径
		CStringA strTemp;
		strTemp.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, GetPatternStringA(nAlgImg), i);

		//打开文件(Unicode环境"t"->"wt")
		FILE* out = NULL;
		fopen_s(&out, strTemp, "wt");

		if (out != NULL)
		{
			//近索引标记
			fprintf_s(out, "CornerIndex%d\n", nIndex);

			//检查顶点是否存在于Cell区域中
			fprintf_s(out, "CornerInside%d,%d,%d,%d\n", nInside[i][0], nInside[i][1], nInside[i][2], nInside[i][3]);

			for (int j = 0; j < ptContours[i].size(); j++)
			{
				//拐角的原点&角度为0度
				fprintf_s(out, "%d,%d\n", ptContours[i][j].x - ptCorner[nIndex].x, ptContours[i][j].y - ptCorner[nIndex].y);
			}

			//关闭文件
			fclose(out);
			out = NULL;

			//17.11.14-White时复制到Dust模式
			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_DUST), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}

			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CornerEdge\\%sPS_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_GRAY_64), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}
		}

		//初始化
		vector< cv::Point2i >().swap(ptContours[i]);
	}
}

// 自动检测Round区并写文件
void CInspectAlign::SetFindRoundAuto(cv::Mat& matTempBuf, vector<vector<cv::Point2i>> contours, cv::Point ptCorner[E_CORNER_END], int nContourIdx, int nAlgImg, int nRatio, float dTheta, CString strPath)
{
	// Image Save 0 On / -1 Off
	int nSaveImg = 1;
	CRect	rectTemp[MAX_MEM_SIZE_E_INSPECT_AREA];
	bool	nInside[MAX_MEM_SIZE_E_INSPECT_AREA][E_CORNER_END];

	const int DIST_THRESHOLD = 300;
	const int ROI_EXPAND = 150;
	const int MIN_ROI_PT_CNT = 20;

	// 找最大轮廓
	int maxContIdx = 0;
	int maxCount = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > maxCount) {
			maxContIdx = i;
			maxCount = contours[i].size();
		}
	}

	// 近似轮廓点
	std::vector<cv::Point> approxPoints;
	cv::approxPolyDP(contours[maxContIdx], approxPoints, 1, true);
	std::reverse(approxPoints.begin(), approxPoints.end());

	// 找第一个点集的起点
	int ptSetStart = -1;
	for (size_t i = 0; i < approxPoints.size(); ++i) {
		int nextIdx = (i + 1) % approxPoints.size();
		const cv::Point& curPoint = approxPoints[i];
		const cv::Point& nextPoint = approxPoints[nextIdx];

		double distance = cv::norm(nextPoint - curPoint);
		if (distance > DIST_THRESHOLD) {
			ptSetStart = nextIdx;
			break;
		}
	}

	// 调整轮廓点的顺序保证轮廓中的第一个点是某个点集的起点
	std::rotate(approxPoints.begin(), approxPoints.begin() + ptSetStart, approxPoints.end());

	// 利用弧线区域点密集，直线区域点稀疏特点，找圆角位置
	std::vector<CRect> roundRectList;
	std::vector<std::vector<cv::Point>> collectedPointSets;
	std::vector<cv::Point> currentPointSet;

	// 遍历轮廓点集
	for (size_t i = 0; i < approxPoints.size(); ++i) {
		int nextIdx = (i + 1) % approxPoints.size();
		const cv::Point& curPoint = approxPoints[i];
		const cv::Point& nextPoint = approxPoints[nextIdx];

		// 计算当前点与下一个点之间的距离
		double distance = cv::norm(nextPoint - curPoint);

		// 如果距离小于阈值，则将当前点添加到当前点集
		if (distance < DIST_THRESHOLD) {
			currentPointSet.push_back(curPoint);
		}
		// 距离大于等于阈值时，将当前点集添加到点集合，并重新创建一个空的当前点集
		else {
			if (!currentPointSet.empty()) {
				collectedPointSets.push_back(currentPointSet);
				currentPointSet.clear();
			}
		}
	}

	// 如果当前点集不为空，则将其添加到点集合
	if (!currentPointSet.empty()) {
		collectedPointSets.push_back(currentPointSet);
	}

	//cv::Mat drawImg;
	//cv::cvtColor(matTempBuf, drawImg, cv::COLOR_GRAY2BGR);

	std::vector<CRect> rndRoiList;
	int ltRoiIdx = -1;
	int ltPosVal = 0;
	for (int i = 0; i < collectedPointSets.size(); i++) {
		if (collectedPointSets[i].size() < MIN_ROI_PT_CNT) continue;

		cv::Rect bbox = cv::boundingRect(collectedPointSets[i]);
		bbox.x -= ROI_EXPAND;
		bbox.y -= ROI_EXPAND;
		bbox.width += ROI_EXPAND * 2;
		bbox.height += ROI_EXPAND * 2;

		std::vector<cv::Point> roiPts = {
			{ bbox.x, bbox.y },
			{ bbox.x + bbox.width, bbox.y },
			{ bbox.x + bbox.width, bbox.y + bbox.height },
			{ bbox.x, bbox.y + bbox.height }
		};
		std::vector<cv::Point> rotRoiPts(4);

		for (int ptIdx = 0; ptIdx < roiPts.size(); ptIdx++) {
			DoRotatePoint(roiPts[ptIdx], rotRoiPts[ptIdx], cv::Point(matTempBuf.cols / 2, matTempBuf.rows / 2), -dTheta);
		}

		cv::Rect roiBox = cv::boundingRect(rotRoiPts);
		CRect roi(roiBox.x, roiBox.y, roiBox.x + roiBox.width, roiBox.y + roiBox.height);

		if (roi.left < 0) roi.left = 0;
		if (roi.top < 0) roi.top = 0;
		if (roi.right >= matTempBuf.cols) roi.right = matTempBuf.cols - 1;
		if (roi.bottom >= matTempBuf.cols) roi.bottom = matTempBuf.rows - 1;

		if (ltRoiIdx == -1) {
			ltPosVal = roi.CenterPoint().x + roi.CenterPoint().y;
			ltRoiIdx = 0;
		}
		else if (rndRoiList.size() > 0 && (roi.CenterPoint().x + roi.CenterPoint().y) < ltPosVal) {
			ltRoiIdx = rndRoiList.size();
			ltPosVal = roi.CenterPoint().x + roi.CenterPoint().y;
		}

		roi.OffsetRect(CPoint(-ptCorner[E_CORNER_LEFT_TOP].x, -ptCorner[E_CORNER_LEFT_TOP].y));
		rndRoiList.push_back(roi);

		//cv::Scalar colors[9] = {
		//	cv::Scalar(255, 0, 0),
		//	cv::Scalar(0, 255, 0),
		//	cv::Scalar(0, 0, 255),
		//	cv::Scalar(255, 255, 0),
		//	cv::Scalar(255, 0, 255),
		//	cv::Scalar(0, 255, 255),
		//	cv::Scalar(100, 255, 0),
		//	cv::Scalar(0, 100, 255),
		//	cv::Scalar(100, 0, 255)
		//};
		//for (auto pt : collectedPointSets[i]) {
		//	cv::circle(drawImg, pt, 1, colors[i], cv::FILLED);
		//}
		//cv::rectangle(drawImg, bbox, colors[i], 2);
	}

	// 以左上角为第一个，顺时针方向调整RndRoi顺序
	if (ltRoiIdx > 0) {
		std::rotate(rndRoiList.begin(), rndRoiList.begin() + ltRoiIdx, rndRoiList.end());
	}

	for (int i = 0; i < rndRoiList.size(); i++)
	{
		//导入UI设置的区域
		rectTemp[i] = rndRoiList[i];

		//PS模式校正
		rectTemp[i].left *= nRatio;
		rectTemp[i].top *= nRatio;
		rectTemp[i].right *= nRatio;
		rectTemp[i].bottom *= nRatio;

		//以Left-Top坐标为原点的坐标值
		//根据当前点亮区域进行校正
		rectTemp[i].OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

		//异常处理
		if (rectTemp[i].left < 0)		rectTemp[i].left = 0;
		if (rectTemp[i].top < 0)		rectTemp[i].top = 0;
		if (rectTemp[i].right < 0)		rectTemp[i].right = 0;
		if (rectTemp[i].bottom < 0)		rectTemp[i].bottom = 0;

		if (rectTemp[i].left >= matTempBuf.cols)	rectTemp[i].left = matTempBuf.cols - 1;
		if (rectTemp[i].top >= matTempBuf.rows)	rectTemp[i].top = matTempBuf.rows - 1;
		if (rectTemp[i].right >= matTempBuf.cols)	rectTemp[i].right = matTempBuf.cols - 1;
		if (rectTemp[i].bottom >= matTempBuf.rows)	rectTemp[i].bottom = matTempBuf.rows - 1;

		//检查顶点是否存在于Cell点灯区域内
		nInside[i][E_CORNER_LEFT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].left) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_LEFT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].left) != 0) ? 1 : 0;
	}
	//用于检查结果
	if (nSaveImg >= 0)
	{
		//cv::Mat matSaveBuf = matTempBuf.clone();
		//for (int i = 0; i < rndRoiList.size(); i++)
		//{
		//	CRect rect = rectTemp[i];
		//	cv::rectangle(matSaveBuf, cv::Rect(rect.left, rect.top, rect.Width(), rect.Height()), cv::Scalar(128, 128, 128));
		//}
		//cv::imwrite("E:\\IMTC\\Round.bmp", matSaveBuf);
		//matSaveBuf.release();
	}

	//曲线外角线
	vector< cv::Point2i > ptContours[MAX_MEM_SIZE_E_INSPECT_AREA];
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		vector< cv::Point2i >().swap(ptContours[i]);

	//顶点数量
	for (int i = 0; i < contours.size(); i++)
	{
		if (i != nContourIdx) continue;
		for (int j = 0; j < contours[i].size(); j++)
		{
			//UI设置的区域数量
			for (int k = 0; k < rndRoiList.size(); k++)
			{
				//如果坐标在校正区域内
				if (rectTemp[k].PtInRect(CPoint(contours[i][j].x, contours[i][j].y)))
				{
					//添加坐标
					ptContours[k].push_back(cv::Point2i(contours[i][j]));
					break;
				}
			}
		}
		//初始化
		vector< cv::Point2i >().swap(contours[i]);
	}

	//////////////////////////////////////////////////////////////////////////
		//排序
		//多边形&填充内部颜色时,需要按顺序...
	//////////////////////////////////////////////////////////////////////////

	cv::Point2i ptTempS;
	for (int i = 0; i < rndRoiList.size(); i++)
	{
		for (int j = 0; j < ptContours[i].size(); j++)
		{
			for (int m = j + 1; m < ptContours[i].size(); m++)
			{
				//对齐y坐标,使其向上移动
				if (ptContours[i][j].y > ptContours[i][m].y)
				{
					ptTempS = ptContours[i][j];
					ptContours[i][j] = ptContours[i][m];
					ptContours[i][m] = ptTempS;
				}
				//如果y坐标相同
				else if (ptContours[i][j].y == ptContours[i][m].y)
				{
					//更改距离转角点较远的情况
					if (abs(ptCorner[i].x - ptContours[i][j].x) < abs(ptCorner[i].x - ptContours[i][m].x))
					{
						ptTempS = ptContours[i][j];
						ptContours[i][j] = ptContours[i][m];
						ptContours[i][m] = ptTempS;
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Save
	//////////////////////////////////////////////////////////////////////////

	char szPath[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strPath, -1, szPath, sizeof(szPath), NULL, NULL);

	//文件存储路径
	CString str;
	str.Format(_T("%s\\CornerEdge"), strPath);
	CreateDirectory(str, NULL);

	CStringA strTemp;
	strTemp.Format(("%s\\CornerEdge\\%s_ROIList.txt"), szPath, GetPatternStringA(nAlgImg));
	FILE* rndRoiFile = NULL;
	fopen_s(&rndRoiFile, strTemp, "wt");

	for (int i = 0; i < rndRoiList.size(); i++)
	{
		//如果没有个数,则排除
		if (ptContours[i].size() <= 0)	continue;

		if (rndRoiFile != NULL)
		{
			fprintf_s(rndRoiFile, "%d,%d,%d,%d\n", rndRoiList[i].left, rndRoiList[i].top, rndRoiList[i].right, rndRoiList[i].bottom);
		}

		//////////////////////////////////////////////////////////////////////////
				//查找设置区域距离点亮的Cell顶点最近的顶点
		//////////////////////////////////////////////////////////////////////////

		int nCx = (rectTemp[i].left + rectTemp[i].right) / 2;
		int nCy = (rectTemp[i].top + rectTemp[i].bottom) / 2;

		//与LT的距离
		int nLength = abs(ptCorner[E_CORNER_LEFT_TOP].x - nCx) + abs(ptCorner[E_CORNER_LEFT_TOP].y - nCy);
		int	nIndex = E_CORNER_LEFT_TOP;
		for (int j = 1; j < E_CORNER_END; j++)
		{
			int nTempLenght = abs(ptCorner[j].x - nCx) + abs(ptCorner[j].y - nCy);

			//寻找小距离的顶点
			if (nLength > nTempLenght)
			{
				nLength = nTempLenght;
				nIndex = j;
			}
		}

		//////////////////////////////////////////////////////////////////////////		

				//文件存储路径
		CStringA strTemp;
		strTemp.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, GetPatternStringA(nAlgImg), i);

		//打开文件(Unicode环境"t"->"wt")
		FILE* out = NULL;
		fopen_s(&out, strTemp, "wt");

		if (out != NULL)
		{
			//近索引标记
			fprintf_s(out, "CornerIndex%d\n", nIndex);

			//检查顶点是否存在于Cell区域中
			fprintf_s(out, "CornerInside%d,%d,%d,%d\n", nInside[i][0], nInside[i][1], nInside[i][2], nInside[i][3]);

			for (int j = 0; j < ptContours[i].size(); j++)
			{
				//拐角的原点&角度为0度
				fprintf_s(out, "%d,%d\n", ptContours[i][j].x - ptCorner[nIndex].x, ptContours[i][j].y - ptCorner[nIndex].y);
			}

			//关闭文件
			fclose(out);
			out = NULL;

			//17.11.14-White时复制到Dust模式
			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CornerEdge\\%s_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_DUST), i);
				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}

			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CornerEdge\\%sPS_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_GRAY_64), i);
				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}
		}

		//初始化
		vector< cv::Point2i >().swap(ptContours[i]);
	}

	if (rndRoiFile != NULL)
	{
		fclose(rndRoiFile);
		rndRoiFile = NULL;
	}
}

//保存Camera Hole坐标
void CInspectAlign::SetFindCHole(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nCHoleROICnt, int nContourIdx, int nAlgImg, int nRatio, CString strPath)
{
	// Image Save 0 On / -1 Off
	int		nSaveImg = -1;

	//需要校正曲线区域设置区域(顶点校正)
	CRect	rectTemp[MAX_MEM_SIZE_E_INSPECT_AREA];
	bool	nInside[MAX_MEM_SIZE_E_INSPECT_AREA][E_CORNER_END];

	for (int i = 0; i < nCHoleROICnt; i++)
	{
		//导入UI设置的区域
		rectTemp[i] = CHoleROI[i].rectROI;

		//PS模式校正
		rectTemp[i].left *= nRatio;
		rectTemp[i].top *= nRatio;
		rectTemp[i].right *= nRatio;
		rectTemp[i].bottom *= nRatio;

		//以Left-Top坐标为原点的坐标值
		//根据当前点亮区域进行校正
		rectTemp[i].OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

		//异常处理
		if (rectTemp[i].left < 0)		rectTemp[i].left = 0;
		if (rectTemp[i].top < 0)		rectTemp[i].top = 0;
		if (rectTemp[i].right < 0)		rectTemp[i].right = 0;
		if (rectTemp[i].bottom < 0)		rectTemp[i].bottom = 0;

		if (rectTemp[i].left >= matTempBuf.cols)	rectTemp[i].left = matTempBuf.cols - 1;
		if (rectTemp[i].top >= matTempBuf.rows)	rectTemp[i].top = matTempBuf.rows - 1;
		if (rectTemp[i].right >= matTempBuf.cols)	rectTemp[i].right = matTempBuf.cols - 1;
		if (rectTemp[i].bottom >= matTempBuf.rows)	rectTemp[i].bottom = matTempBuf.rows - 1;

		//检查顶点是否存在于Cell点灯区域内
		nInside[i][E_CORNER_LEFT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].left) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_LEFT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].left) != 0) ? 1 : 0;
	}
	//用于检查结果
	if (nSaveImg >= 0)
	{
		cv::Mat matSaveBuf = matTempBuf.clone();

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			if (CHoleROI[i].bUseROI)
			{
				CRect rect = rectTemp[i];

				cv::rectangle(matSaveBuf, cv::Rect(rect.left, rect.top, rect.Width(), rect.Height()), cv::Scalar(128, 128, 128));
			}
		}
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_CHole_Coord.bmp"), nAlgImg);
		cv::imwrite((cv::String)(CStringA)strTemp, matSaveBuf);
		matSaveBuf.release();
	}

	//曲线外角线
	vector< cv::Point2i > ptContours[MAX_MEM_SIZE_E_INSPECT_AREA];
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		vector< cv::Point2i >().swap(ptContours[i]);

	//顶点数量
	for (int i = 0; i < contours.size(); i++)
	{
		if (i == nContourIdx) continue;
		for (int j = 0; j < contours[i].size(); j++)
		{
			//UI设置的区域数量
			for (int k = 0; k < nCHoleROICnt; k++)
			{
				//不使用
				if (!CHoleROI[k].bUseROI)	continue;

				//如果坐标在校正区域内
				if (rectTemp[k].PtInRect(CPoint(contours[i][j].x, contours[i][j].y)))
				{
					//添加坐标
					ptContours[k].push_back(cv::Point2i(contours[i][j]));
					break;
				}
			}
		}
		//初始化
		vector< cv::Point2i >().swap(contours[i]);
	}

	//////////////////////////////////////////////////////////////////////////
		//排序
		//多边形&填充内部颜色时,需要按顺序...
	//////////////////////////////////////////////////////////////////////////

	cv::Point2i ptTempS;
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		for (int j = 0; j < ptContours[i].size(); j++)
		{
			for (int m = j + 1; m < ptContours[i].size(); m++)
			{
				//对齐y坐标,使其向上移动
				if (ptContours[i][j].y > ptContours[i][m].y)
				{
					ptTempS = ptContours[i][j];
					ptContours[i][j] = ptContours[i][m];
					ptContours[i][m] = ptTempS;
				}
				//如果y坐标相同
				else if (ptContours[i][j].y == ptContours[i][m].y)
				{
					//更改距离转角点较远的情况
					if (abs(ptCorner[i].x - ptContours[i][j].x) < abs(ptCorner[i].x - ptContours[i][m].x))
					{
						ptTempS = ptContours[i][j];
						ptContours[i][j] = ptContours[i][m];
						ptContours[i][m] = ptTempS;
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Save
	//////////////////////////////////////////////////////////////////////////

	char szPath[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strPath, -1, szPath, sizeof(szPath), NULL, NULL);

	//文件存储路径
	CString str;
	str.Format(_T("%s\\CameraHole"), strPath);
	CreateDirectory(str, NULL);

	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		//如果没有个数,则排除
		if (ptContours[i].size() <= 0)	continue;

		//////////////////////////////////////////////////////////////////////////
				//查找设置区域距离点亮的Cell顶点最近的顶点
		//////////////////////////////////////////////////////////////////////////

		int nCx = (rectTemp[i].left + rectTemp[i].right) / 2;
		int nCy = (rectTemp[i].top + rectTemp[i].bottom) / 2;

		//与LT的距离
		int nLength = abs(ptCorner[E_CORNER_LEFT_TOP].x - nCx) + abs(ptCorner[E_CORNER_LEFT_TOP].y - nCy);
		int	nIndex = E_CORNER_LEFT_TOP;
		for (int j = 1; j < E_CORNER_END; j++)
		{
			int nTempLenght = abs(ptCorner[j].x - nCx) + abs(ptCorner[j].y - nCy);

			//寻找小距离的顶点
			if (nLength > nTempLenght)
			{
				nLength = nTempLenght;
				nIndex = j;
			}
		}

		//////////////////////////////////////////////////////////////////////////		

				//文件存储路径
		CStringA strTemp;
		strTemp.Format(("%s\\CameraHole\\%s_%02d.EdgePT"), szPath, GetPatternStringA(nAlgImg), i);

		//打开文件(Unicode环境"t"->"wt")
		FILE* out = NULL;
		fopen_s(&out, strTemp, "wt");

		if (out != NULL)
		{
			//近索引标记
			fprintf_s(out, "CameraHoleIndex%d\n", nIndex);

			//检查顶点是否存在于Cell区域中
			fprintf_s(out, "CameraHoleInside%d,%d,%d,%d\n", nInside[i][0], nInside[i][1], nInside[i][2], nInside[i][3]);

			for (int j = 0; j < ptContours[i].size(); j++)
			{
				//拐角的原点&角度为0度
				fprintf_s(out, "%d,%d\n", ptContours[i][j].x - ptCorner[nIndex].x, ptContours[i][j].y - ptCorner[nIndex].y);
			}

			//关闭文件
			fclose(out);
			out = NULL;

			//17.11.14-White时复制到Dust模式
			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CameraHole\\%s_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_DUST), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}

			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_64)
			{
				CStringA strCopy;
				strCopy.Format(("%s\\CameraHole\\%sPS_%02d.EdgePT"), szPath, GetPatternStringA(E_IMAGE_CLASSIFY_AVI_GRAY_64), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);
			}
		}

		//初始化
		vector< cv::Point2i >().swap(ptContours[i]);
	}
}

//保存Camera Hole坐标
void CInspectAlign::SetFindCHole2(cv::Mat& matTempBuf, vector< vector< cv::Point2i > > contours, cv::Point ptCorner[E_CORNER_END], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nCHoleROICnt, int nContourIdx, int nAlgImg, int nRatio, CString strPath)
{
	// Image Save 0 On / -1 Off
	int		nSaveImg = -1;

	//需要校正曲线区域设置区域(顶点校正)
	CRect	rectTemp[MAX_MEM_SIZE_E_INSPECT_AREA];
	bool	nInside[MAX_MEM_SIZE_E_INSPECT_AREA][E_CORNER_END];

	for (int i = 0; i < nCHoleROICnt; i++)
	{
		//导入UI设置的区域
		rectTemp[i] = CHoleROI[i].rectROI;

		//PS模式校正
		rectTemp[i].left *= nRatio;
		rectTemp[i].top *= nRatio;
		rectTemp[i].right *= nRatio;
		rectTemp[i].bottom *= nRatio;

		//以Left-Top坐标为原点的坐标值
		//根据当前点亮区域进行校正
		rectTemp[i].OffsetRect(CPoint(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_TOP].y));

		//异常处理
		if (rectTemp[i].left < 0)		rectTemp[i].left = 0;
		if (rectTemp[i].top < 0)		rectTemp[i].top = 0;
		if (rectTemp[i].right < 0)		rectTemp[i].right = 0;
		if (rectTemp[i].bottom < 0)		rectTemp[i].bottom = 0;

		if (rectTemp[i].left >= matTempBuf.cols)	rectTemp[i].left = matTempBuf.cols - 1;
		if (rectTemp[i].top >= matTempBuf.rows)	rectTemp[i].top = matTempBuf.rows - 1;
		if (rectTemp[i].right >= matTempBuf.cols)	rectTemp[i].right = matTempBuf.cols - 1;
		if (rectTemp[i].bottom >= matTempBuf.rows)	rectTemp[i].bottom = matTempBuf.rows - 1;

		//检查顶点是否存在于Cell点灯区域内
		nInside[i][E_CORNER_LEFT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].left) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_TOP] = (matTempBuf.at<uchar>(rectTemp[i].top, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_RIGHT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].right) != 0) ? 1 : 0;
		nInside[i][E_CORNER_LEFT_BOTTOM] = (matTempBuf.at<uchar>(rectTemp[i].bottom, rectTemp[i].left) != 0) ? 1 : 0;
	}
	//用于检查结果
	if (nSaveImg >= 0)
	{
		cv::Mat matSaveBuf = matTempBuf.clone();

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			if (CHoleROI[i].bUseROI)
			{
				CRect rect = rectTemp[i];

				cv::rectangle(matSaveBuf, cv::Rect(rect.left, rect.top, rect.Width(), rect.Height()), cv::Scalar(128, 128, 128));
			}
		}
		CString strTemp;
		strTemp.Format(_T("D:\\CHole\\%02d_CHole_Coord.bmp"), nAlgImg);
		cv::imwrite((cv::String)(CStringA)strTemp, matSaveBuf);
		matSaveBuf.release();
	}

	//曲线外角线
	vector< cv::Point2i > ptContours[MAX_MEM_SIZE_E_INSPECT_AREA];
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		vector< cv::Point2i >().swap(ptContours[i]);

	//顶点数量
	for (int i = 0; i < contours.size(); i++)
	{
		if (i == nContourIdx) continue;
		for (int j = 0; j < contours[i].size(); j++)
		{
			//UI设置的区域数量
			for (int k = 0; k < nCHoleROICnt; k++)
			{
				//不使用
				if (!CHoleROI[k].bUseROI)	continue;

				//如果坐标在校正区域内
				if (rectTemp[k].PtInRect(CPoint(contours[i][j].x, contours[i][j].y)))
				{
					//添加坐标
					ptContours[k].push_back(cv::Point2i(contours[i][j]));
					break;
				}
			}
		}
		//初始化
		vector< cv::Point2i >().swap(contours[i]);
	}

	//////////////////////////////////////////////////////////////////////////
		//排序
		//多边形&填充内部颜色时,需要按顺序...
	//////////////////////////////////////////////////////////////////////////

	cv::Point2i ptTempS;
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		for (int j = 0; j < ptContours[i].size(); j++)
		{
			for (int m = j + 1; m < ptContours[i].size(); m++)
			{
				//对齐y坐标,使其向上移动
				if (ptContours[i][j].y > ptContours[i][m].y)
				{
					ptTempS = ptContours[i][j];
					ptContours[i][j] = ptContours[i][m];
					ptContours[i][m] = ptTempS;
				}
				//如果y坐标相同
				else if (ptContours[i][j].y == ptContours[i][m].y)
				{
					//更改距离转角点较远的情况
					if (abs(ptCorner[i].x - ptContours[i][j].x) < abs(ptCorner[i].x - ptContours[i][m].x))
					{
						ptTempS = ptContours[i][j];
						ptContours[i][j] = ptContours[i][m];
						ptContours[i][m] = ptTempS;
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Save
	//////////////////////////////////////////////////////////////////////////

	char szPath[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strPath, -1, szPath, sizeof(szPath), NULL, NULL);

	//文件存储路径
	CString str;
	str.Format(_T("%s\\CameraHole"), strPath);
	CreateDirectory(str, NULL);

	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		//如果没有个数,则排除
		if (ptContours[i].size() <= 0)	continue;

		//////////////////////////////////////////////////////////////////////////
				//查找设置区域距离点亮的Cell顶点最近的顶点
		//////////////////////////////////////////////////////////////////////////

		int nCx = (rectTemp[i].left + rectTemp[i].right) / 2;
		int nCy = (rectTemp[i].top + rectTemp[i].bottom) / 2;

		//与LT的距离
		int nLength = abs(ptCorner[E_CORNER_LEFT_TOP].x - nCx) + abs(ptCorner[E_CORNER_LEFT_TOP].y - nCy);
		int	nIndex = E_CORNER_LEFT_TOP;
		for (int j = 1; j < E_CORNER_END; j++)
		{
			int nTempLenght = abs(ptCorner[j].x - nCx) + abs(ptCorner[j].y - nCy);

			//寻找小距离的顶点
			if (nLength > nTempLenght)
			{
				nLength = nTempLenght;
				nIndex = j;
			}
		}

		//////////////////////////////////////////////////////////////////////////		

				//文件存储路径
		CStringA strTemp;
		strTemp.Format(("%s\\CameraHole\\%s_%02d.EdgePT"), szPath, GetPatternStringA(nAlgImg), i);

		//打开文件(Unicode环境"t"->"wt")
		FILE* out = NULL;
		fopen_s(&out, strTemp, "wt");

		if (out != NULL)
		{
			//近索引标记
			fprintf_s(out, "CameraHoleIndex%d\n", nIndex);

			//检查顶点是否存在于Cell区域中
			fprintf_s(out, "CameraHoleInside%d,%d,%d,%d\n", nInside[i][0], nInside[i][1], nInside[i][2], nInside[i][3]);

			for (int j = 0; j < ptContours[i].size(); j++)
			{
				//拐角的原点&角度为0度
				fprintf_s(out, "%d,%d\n", ptContours[i][j].x - ptCorner[nIndex].x, ptContours[i][j].y - ptCorner[nIndex].y);
			}

			//关闭文件
			fclose(out);
			out = NULL;

			//17.11.14-从Dust模式复制所有模式
			for (int j = 0; j < E_IMAGE_CLASSIFY_AVI_COUNT; j++)
			{
				if (j == E_IMAGE_CLASSIFY_AVI_DUST)
					continue;
				CStringA strCopy;
				strCopy.Format(("%s\\CameraHole\\%s_%02d.EdgePT"), szPath, GetPatternStringA(j), i);

				CopyFile((CString)strTemp, (CString)strCopy, FALSE);

				//客户请求保存G64PS模式
				if (j == E_IMAGE_CLASSIFY_AVI_GRAY_64)
				{
					strCopy.Format(("%s\\CameraHole\\%s_%02d.EdgePT"), "G64PS", i);
					CopyFile((CString)strTemp, (CString)strCopy, FALSE);
				}

			}

		}

		//初始化
		vector< cv::Point2i >().swap(ptContours[i]);
	}
}

long CInspectAlign::GetBMCorner(cv::Mat Src, double* dAlgPara, Point* ptPanelCorner, cv::Rect& rtBMCorner)
{
	int		nGausBlurSize = (int)dAlgPara[E_PARA_APP_POLYGON_GAUS_SIZE];
	float	fGausSigma = (float)dAlgPara[E_PARA_APP_POLYGON_GAUS_SIGMA];
	int		nCornerROISize = (int)dAlgPara[E_PARA_APP_POLYGON_CORNER_ROI_SIZE];					//Corner区域的ROI大小
	float	fBM_LT_Theta = (float)dAlgPara[E_PARA_APP_POLYGON_BM_THETA_LT];						//BM Line Search的基准角度
	float	fBM_RB_Theta = (float)dAlgPara[E_PARA_APP_POLYGON_BM_THETA_RB];
	int		nBMIgnore = (int)dAlgPara[E_PARA_APP_POLYGON_BM_IGNORE_GV];					//BM Line搜索基准角度点的最小GV值条件
	float	fBMThreshold_Ratio = (float)dAlgPara[E_PARA_APP_POLYGON_BM_PRE_THRESH_RATIO];			//在Threshold为平均值时,平均值的使用比例大小。

	Mat mtLeftTop;
	Src(Rect(ptPanelCorner[E_CORNER_LEFT_TOP].x,
		ptPanelCorner[E_CORNER_LEFT_TOP].y,
		nCornerROISize, nCornerROISize)).copyTo(mtLeftTop);

	Mat mtRightBottom;
	Src(Rect(ptPanelCorner[E_CORNER_RIGHT_BOTTOM].x - nCornerROISize,
		ptPanelCorner[E_CORNER_RIGHT_BOTTOM].y - nCornerROISize,
		nCornerROISize, nCornerROISize)).copyTo(mtRightBottom);

	GaussianBlur(mtLeftTop, mtLeftTop, Size(nGausBlurSize, nGausBlurSize), fGausSigma);
	GaussianBlur(mtRightBottom, mtRightBottom, Size(nGausBlurSize, nGausBlurSize), fGausSigma);

	Scalar mean_LT, mean_RB, std_LT, std_RB;

	cv::meanStdDev(mtLeftTop, mean_LT, std_LT);
	cv::meanStdDev(mtRightBottom, mean_RB, std_RB);

	Mat mtThreshold_LT, mtThreshold_RB;
	cv::threshold(mtLeftTop, mtThreshold_LT, mean_LT[0] * 0.8, 255, CV_THRESH_BINARY);
	cv::threshold(mtRightBottom, mtThreshold_RB, mean_RB[0] * 0.8, 255, CV_THRESH_BINARY);

	//只保留最大的块,用作Mask
	Mat mtBigLT, mtBigRB;
	mtBigLT = Mat::zeros(mtLeftTop.size(), mtLeftTop.type());
	mtBigRB = Mat::zeros(mtLeftTop.size(), mtLeftTop.type());

	FindBiggestBlob(mtThreshold_LT, mtBigLT);
	FindBiggestBlob(mtThreshold_RB, mtBigRB);

	Mat mtMaskLT, mtMaskRB;
	mtLeftTop.copyTo(mtMaskLT, mtBigLT);
	mtRightBottom.copyTo(mtMaskRB, mtBigRB);

	vector<vector<Point>> ptLT(1), ptRB(1);

	int nDeltaX = 3;
	int nDeltaY = 3;

	for (int nY = 0; nY < mtMaskLT.rows; nY++)
	{
		uchar* ucDataLT = mtMaskLT.data + nY * mtMaskLT.step;
		uchar* ucDataRB = mtMaskRB.data + nY * mtMaskRB.step;

		for (int nX = 0; nX < mtMaskLT.cols; nX++)
		{
			if (nX + nDeltaX > mtMaskLT.cols - 1)
				continue;

			int nGV1_LT, nGV2_LT;
			nGV1_LT = (int)*(ucDataLT + nX);
			nGV2_LT = (int)*(ucDataLT + nX + nDeltaX);

			double fThetaLT = atan((float)(nGV2_LT - nGV1_LT) / nDeltaX) * 180 / CV_PI;

			if (90 > fThetaLT && fThetaLT > fBM_LT_Theta) {
				if (nBMIgnore > nGV2_LT)
					continue;
				ptLT[0].push_back(Point(nX + nDeltaX, nY));
				break;
			}

		}

		for (int nX2 = mtMaskRB.cols - 1; nX2 >= 0; nX2--)
		{
			if (nX2 - nDeltaX < 0)
				continue;

			int nGV1_RB, nGV2_RB;
			nGV1_RB = (int)*(ucDataRB + nX2);
			nGV2_RB = (int)*(ucDataRB + nX2 - nDeltaX);

			double fThetaRB = atan((float)(nGV2_RB - nGV1_RB) / nDeltaX) * 180 / CV_PI;

			if (90 > fThetaRB && fThetaRB > fBM_RB_Theta) {
				if (nBMIgnore > nGV2_RB)
					continue;
				ptRB[0].push_back(Point(nX2 - nDeltaX, nY));
				break;
			}
		}
	}

	for (int nX = 0; nX < mtMaskLT.cols; nX++)
	{
		uchar* ucDataLT = mtMaskLT.data + nX;
		uchar* ucDataRB = mtMaskRB.data + nX;

		for (int nY = 0; nY < mtMaskLT.rows; nY++)
		{
			if (nY + nDeltaY > mtMaskLT.rows - 1)
				continue;

			int nGV1_LT, nGV2_LT;
			nGV1_LT = (int)*(ucDataLT + nY * mtMaskLT.step);
			nGV2_LT = (int)*(ucDataLT + (nY + nDeltaY) * mtMaskLT.step);

			double fThetaLT = (float)atan((float)(nGV2_LT - nGV1_LT) / nDeltaY) * 180 / CV_PI;

			if (90 > fThetaLT && fThetaLT > fBM_LT_Theta) {
				if (nBMIgnore > nGV2_LT)
					continue;
				ptLT[0].push_back(Point(nX, nY + nDeltaY));
				break;
			}
		}

		for (int nY2 = mtMaskRB.rows; nY2 >= 0; nY2--)
		{
			if (nY2 - nDeltaY < 0)
				continue;

			int nGV1_RB, nGV2_RB;

			nGV1_RB = (int)*(ucDataRB + nY2 * mtMaskRB.step);
			nGV2_RB = (int)*(ucDataRB + (nY2 - nDeltaY) * mtMaskRB.step);

			double fThetaRB = (float)atan((float)(nGV2_RB - nGV1_RB) / nDeltaY) * 180 / CV_PI;
			if (90 > fThetaRB && fThetaRB > fBM_RB_Theta)
			{
				if (nBMIgnore > nGV2_RB)
					continue;
				ptRB[0].push_back(Point(nX, nY2 - nDeltaY));
				break;
			}
		}
	}

	//如果Countour坐标数低于50个,则不进行检查。
	if (ptLT[0].size() < 50 || ptRB[0].size() < 50)
		return E_ERROR_CODE_FALSE;

	cv::Rect rtLT, rtRB;
	rtLT = boundingRect(ptLT[0]);
	rtRB = boundingRect(ptRB[0]);

	rtBMCorner = Rect(Point(rtLT.tl().x + ptPanelCorner[E_CORNER_LEFT_TOP].x,
		rtLT.tl().y + ptPanelCorner[E_CORNER_LEFT_TOP].y),
		Point(rtRB.br().x + ptPanelCorner[E_CORNER_RIGHT_BOTTOM].x - nCornerROISize,
			rtRB.br().y + ptPanelCorner[E_CORNER_RIGHT_BOTTOM].y - nCornerROISize));

	return E_ERROR_CODE_TRUE;
}

//找到大致的Cell位置
long CInspectAlign::FindCellEdge_For_Morphology(cv::Mat matSrc, int nThreshold, cv::Rect& rcFindCellROI)
{

	//////////////////////////////////////////////////////////////////////////////////////////////////
		//只为提高速度的外围寻找局部的运算-寻找局部的运算点。
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//double dblTmpSum;

	//cv::Rect rcTempFindEdgeROI;

	cv::Size szTempBufSize = matSrc.size();
	int nTempBufWidth = szTempBufSize.width;
	int nTempBufHeight = szTempBufSize.height;

	//////////////////////////////////
		//首先查找Y轴方向的Edge。//
	//////////////////////////////////

	int nHalfWidth = nTempBufWidth / 2;

	cv::Mat  matTempFindEdgeYROI = matSrc(cv::Rect(nHalfWidth - (nHalfWidth / 2), 0, nHalfWidth, nTempBufHeight));
	int nROIWidth = matTempFindEdgeYROI.size().width;

	int nFindStart = -1;
	int nFindEnd = -1;

#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int roop = 0; roop < 2; roop++)
	{
		if (roop == 0)
		{
			double dblTmpSum;
			//Y轴顶部

			for (int i = 0; i < nTempBufHeight; i++)
			{
				dblTmpSum = cv::sum(matTempFindEdgeYROI.row(i))[0] / nROIWidth;
				if (dblTmpSum > nThreshold)
				{
					nFindStart = i;
					break;
				}
			}
		}

		if (roop == 1)
		{
			double dblTmpSum;
			//Y轴底部
			for (int i = nTempBufHeight - 1; i > 0; i--)
			{
				dblTmpSum = cv::sum(matTempFindEdgeYROI.row(i))[0] / nROIWidth;
				if (dblTmpSum > nThreshold)
				{
					nFindEnd = i;
					break;
				}
			}
		}
	}

	if (nFindStart == -1 | nFindEnd == -1)//如果找不到
		return E_ERROR_CODE_ALIGN_LENGTH_RANGE_ERROR;

	rcFindCellROI.y = nFindStart;
	rcFindCellROI.height = nFindEnd - rcFindCellROI.y;

	/////////////////////////////
		//查找X轴方向的Edge	//
	/////////////////////////////

	cv::Mat  matTempFindEdgeXROI = matSrc(cv::Rect(0, rcFindCellROI.y, nTempBufWidth, rcFindCellROI.height));
	int nROIHeight = matTempFindEdgeXROI.size().height;

	nFindStart = -1;
	nFindEnd = -1;

#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int roop = 0; roop < 2; roop++)
	{
		if (roop == 0)
		{
			double dblTmpSum;
			//左查找
			for (int i = 0; i < nTempBufWidth; i++)
			{
				dblTmpSum = cv::sum(matTempFindEdgeXROI.col(i))[0] / nROIHeight;
				if (dblTmpSum > nThreshold)
				{
					nFindStart = i;
					break;
				}
			}
		}

		if (roop == 1)
		{
			double dblTmpSum;
			//右查找
			for (int i = nTempBufWidth - 1; i > 0; i--)
			{
				dblTmpSum = cv::sum(matTempFindEdgeXROI.col(i))[0] / nROIHeight;
				if (dblTmpSum > nThreshold)
				{
					nFindEnd = i;
					break;
				}
			}
		}
	}

	if (nFindStart == -1 | nFindEnd == -1)//如果找不到
		return E_ERROR_CODE_ALIGN_LENGTH_RANGE_ERROR;

	rcFindCellROI.x = nFindStart;
	rcFindCellROI.width = nFindEnd - rcFindCellROI.x;

	matTempFindEdgeYROI.release();
	matTempFindEdgeXROI.release();

	return E_ERROR_CODE_TRUE;
}

void CInspectAlign::RecalRect(cv::Rect& rcRect, cv::Size szLimit)
{
	if (rcRect.x < 0)
		rcRect.x = 0;

	if (rcRect.y < 0)
		rcRect.y = 0;

	if (rcRect.width > szLimit.width - rcRect.x)
		rcRect.width = szLimit.width - rcRect.x;

	if (rcRect.height > szLimit.height - rcRect.y)
		rcRect.height = szLimit.height - rcRect.y;
}

//创建要填充的区域
long CInspectAlign::MakeRoI_For_Morphology(cv::Rect rcFindCellROI,
	int nExtROI_Outer, int nExtROI_Inner_L, int nExtROI_Inner_R, int nExtROI_Inner_T, int nExtROI_Inner_B,
	cv::Size rcLimit, cv::Rect* prcMorpROI)
{
	//Cell外部Outer
	//Cell内部为Inner

	int nAddInspArea = 100; // 重复检查区域

	//Left
	prcMorpROI[0].x = rcFindCellROI.x - nExtROI_Outer;
	prcMorpROI[0].y = rcFindCellROI.y - nExtROI_Outer;
	prcMorpROI[0].width = nExtROI_Inner_L + nExtROI_Outer;
	prcMorpROI[0].height = rcFindCellROI.height + (nExtROI_Outer * 2);

	//right
	prcMorpROI[1].x = rcFindCellROI.x + rcFindCellROI.width - nExtROI_Inner_R;
	prcMorpROI[1].y = rcFindCellROI.y - nExtROI_Outer;
	prcMorpROI[1].width = nExtROI_Inner_R + nExtROI_Outer;
	prcMorpROI[1].height = rcFindCellROI.height + (nExtROI_Outer * 2);

	//top
	prcMorpROI[2].x = (prcMorpROI[0].x + prcMorpROI[0].width) - nAddInspArea;	// 请参阅Left数据-重复的波罗数为150pxl
	prcMorpROI[2].y = prcMorpROI[0].y;			// 请参阅Left数据
	prcMorpROI[2].width = prcMorpROI[1].x - prcMorpROI[2].x + nAddInspArea;	// 请参阅Right数据-重复的波罗数为150pxl
	prcMorpROI[2].height = nExtROI_Inner_T + nExtROI_Outer;

	//bottom
	prcMorpROI[3].x = (prcMorpROI[0].x + prcMorpROI[0].width) - nAddInspArea;		// 请参阅Left数据-重复的波罗数为150pxl
	prcMorpROI[3].y = rcFindCellROI.y + rcFindCellROI.height - nExtROI_Inner_B;
	prcMorpROI[3].width = prcMorpROI[1].x - prcMorpROI[3].x + nAddInspArea;	// 请参阅Right数据-重复的波罗数为150pxl
	prcMorpROI[3].height = nExtROI_Inner_B + nExtROI_Outer;

	//范围异常处理
	RecalRect(prcMorpROI[0], rcLimit);
	RecalRect(prcMorpROI[1], rcLimit);
	RecalRect(prcMorpROI[2], rcLimit);
	RecalRect(prcMorpROI[3], rcLimit);

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::Partial_Morphology(cv::Mat matSrc, cv::Mat matDst, int nMorpType, cv::Mat StructElem, cv::Rect* prcMorpROI)
{
	//每个输入的区域要做4次mopology。-上下同时,左右同时并行处理		

	cv::Mat matTmpROI_Src_L = matSrc(prcMorpROI[0]);
	cv::Mat matTmpROI_Src_R = matSrc(prcMorpROI[1]);
	cv::Mat matTmpROI_Src_T = matSrc(prcMorpROI[2]);
	cv::Mat matTmpROI_Src_B = matSrc(prcMorpROI[3]);

	cv::Mat matTmpROI_Dst_L = matDst(prcMorpROI[0]);
	cv::Mat matTmpROI_Dst_R = matDst(prcMorpROI[1]);
	cv::Mat matTmpROI_Dst_T = matDst(prcMorpROI[2]);
	cv::Mat matTmpROI_Dst_B = matDst(prcMorpROI[3]);

	//左右冒号-并行处理。
#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int roop = 0; roop < 2; roop++)
	{
		if (roop == 0)
		{
			//左边的冒号
			cv::morphologyEx(matTmpROI_Src_L, matTmpROI_Dst_L, nMorpType, StructElem);
		}

		if (roop == 1)
		{
			//右边的阿波罗
			cv::morphologyEx(matTmpROI_Src_R, matTmpROI_Dst_R, nMorpType, StructElem);
		}
	}

	//上下角冒号-并行处理。
#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int roop = 0; roop < 2; roop++)
	{
		if (roop == 0)
		{
			//左边的冒号
			cv::morphologyEx(matTmpROI_Src_T, matTmpROI_Dst_T, nMorpType, StructElem);
		}

		if (roop == 1)
		{
			//右边的阿波罗
			cv::morphologyEx(matTmpROI_Src_B, matTmpROI_Dst_B, nMorpType, StructElem);
		}
	}

	matTmpROI_Src_L.release();
	matTmpROI_Src_R.release();
	matTmpROI_Src_T.release();
	matTmpROI_Src_B.release();

	matTmpROI_Dst_L.release();
	matTmpROI_Dst_R.release();
	matTmpROI_Dst_T.release();
	matTmpROI_Dst_B.release();

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::Partial_Laplacian(cv::Mat matSrc, cv::Mat matDst, cv::Rect* prcMorpROI)
{
	//每个输入的区域要做4次mopology。-上下同时,左右同时并行处理		

	cv::Mat matTmpROI_Src_L = matSrc(prcMorpROI[0]);
	cv::Mat matTmpROI_Src_R = matSrc(prcMorpROI[1]);
	cv::Mat matTmpROI_Src_T = matSrc(prcMorpROI[2]);
	cv::Mat matTmpROI_Src_B = matSrc(prcMorpROI[3]);

	cv::Mat matTmpROI_Dst_L = matDst(prcMorpROI[0]);
	cv::Mat matTmpROI_Dst_R = matDst(prcMorpROI[1]);
	cv::Mat matTmpROI_Dst_T = matDst(prcMorpROI[2]);
	cv::Mat matTmpROI_Dst_B = matDst(prcMorpROI[3]);

	//左右冒号-并行处理。
#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int roop = 0; roop < 2; roop++)
	{
		if (roop == 0)
		{
			//左Laplacian
			cv::Laplacian(matTmpROI_Src_L, matTmpROI_Dst_L, CV_8U);
		}

		if (roop == 1)
		{
			//右Laplacian
			cv::Laplacian(matTmpROI_Src_R, matTmpROI_Dst_R, CV_8U);
		}
	}

	//上下角冒号-并行处理。
#ifdef _DEBUG
#else
#pragma omp parallel for num_threads(2)
#endif
	for (int roop = 0; roop < 2; roop++)
	{
		if (roop == 0)
		{
			//左Laplacian
			cv::Laplacian(matTmpROI_Src_T, matTmpROI_Dst_T, CV_8U);
		}

		if (roop == 1)
		{
			//右Laplacian
			cv::Laplacian(matTmpROI_Src_B, matTmpROI_Dst_B, CV_8U);
		}
	}

	matTmpROI_Src_L.release();
	matTmpROI_Src_R.release();
	matTmpROI_Src_T.release();
	matTmpROI_Src_B.release();

	matTmpROI_Dst_L.release();
	matTmpROI_Dst_R.release();
	matTmpROI_Dst_T.release();
	matTmpROI_Dst_B.release();

	return E_ERROR_CODE_TRUE;
}

double CInspectAlign::CenterMeanGV(cv::Mat& matSrcBuf, int nMinArea)
{
	//必须找到点灯区域(希望除区域外没有杂音)
	cv::Mat matMask = cv::Mat::zeros(matSrcBuf.size(), CV_8UC1);
	// 阈值由5改到2，解决blob个数太多导致速度慢问题 20230406.xb
	cv::threshold(matSrcBuf, matMask, 2, 255, CV_THRESH_BINARY);

	/*cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(31, 31));

	cv::morphologyEx(matMask, matMask, MORPH_CLOSE, StructElem);*/

	//查找活动区域
	cv::Rect rectCell;
	FindCellArea(matMask, nMinArea, rectCell);

	//需要知道活动区域的中心亮度
	cv::Rect rectCenter = Rect(rectCell.x + rectCell.width / 2 - rectCell.width / 6, rectCell.y + rectCell.height / 2 - rectCell.height / 6, rectCell.width / 6 * 2, rectCell.height / 6 * 2);
	cv::Mat matCenterROI = matSrcBuf(rectCenter);

	int nMean = cv::mean(matCenterROI)[0];

	if (nMean == 0)
	{
		cv::Rect rectCenter = Rect(matSrcBuf.cols / 2 - 100 / 6, matSrcBuf.rows / 2 - 100, 200, 200);
		cv::Mat matCenterROI = matSrcBuf(rectCenter);

		nMean = cv::mean(matCenterROI)[0];
	}

	// Fix 除0 crash. 20230406.xb
	if (nMean == 0) {
		return 1;
	}
	double dMulti = 50 / nMean;				// GV为80

	return dMulti;

}
//外围处理
long CInspectAlign::DoFillOutArea(cv::Mat& matSrcBuf, cv::Mat& MatDrawBuffer, cv::Mat& matBKGBuf, cv::Point ptResCornerOrigin[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA], cv::Mat* matCHoleROIBuf, cv::Rect* rcCHoleROI, bool* bCHoleAD, double* dPara, int nAlgImg, int nRatio, wchar_t* strID)
{
	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Start."));
	//2021.02.24 G3相关测试
	//不包括G3模式(2021.02.24将G128用作G3模式)
	//2022.10.14 G3外围填充test
	double dMulti = 0;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_128)
	{
		int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);	// 3800 * 1900;	// APP
		dMulti = CenterMeanGV(matSrcBuf, nMinArea);

		matSrcBuf *= dMulti;
	}

	//不包括Black模式
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_BLACK)		return E_ERROR_CODE_TRUE;

	//排除PCD模式
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_PCD)		return E_ERROR_CODE_TRUE;

	//排除VINIT模式
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_VINIT)		return E_ERROR_CODE_TRUE;
	//17.11.14-Dust模式->使用DoFillOutAreaDust()
	//如果是Round,则存在背景
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST)		return E_ERROR_CODE_TRUE;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)		return E_ERROR_CODE_TRUE;//跳过背光画面 hjf
	//如果没有缓冲区。
	if (matSrcBuf.empty())							return E_ERROR_CODE_EMPTY_BUFFER;

	//-1:不保存文件
	//0:保存文件
	int nSaveImageCount = -1;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	int		nMinSamples = 3;	// 固定
	double	distThreshold = 10;	// 固定

	int		nMinArea = (int)(dPara[E_PARA_CELL_SIZE_X] * dPara[E_PARA_CELL_SIZE_Y] * nRatio * nRatio);
	int		nThreshold = dPara[E_PARA_ALIGN_THRESHOLD];
	int		nMorp = dPara[E_PARA_ALIGN_MORP];
	double	dAngleError = dPara[E_PARA_ALIGN_ANGLE_ERR];
	double	dAngleWarning = dPara[E_PARA_ALIGN_ANGLE_WAR];

	//设置查找Round区域的范围(上,下,左,右)
	int		nFindRoundOffset = (int)dPara[E_PARA_ROUND_FIND_OFFSET];
	int		nFindCHoleOffset = (int)dPara[E_PARA_CHOLE_FIND_OFFSET];

	//填充轮廓平均值时,设置最小平均GV
	int nLabel_Flag = dPara[E_PARA_AVI_Label_Flag];   //yuxuefei add
	int nPolNum_Flag = dPara[E_PARA_AVI_PolNum_Flag];
	int nPolSign_Flag = dPara[E_PARA_AVI_PolSign_Flag];
	int bPolSaveTemplate = dPara[E_PARA_AVI_Pol_Save_Template];

	int nRotate_Use = dPara[E_PARA_AVI_Rotate_Image];  //yuxuefei add
	int nLabelArea = (int)dPara[E_PARA_AVI_Label_Width] * (int)dPara[E_PARA_AVI_Label_Height];
	// 외곽 평균 채울때, 최소 평균 GV 설정
	//在Round&CHole区域外平均填充时,设置最小平均GV
	int		nRoundMinGV = (int)dPara[E_PARA_ROUND_OTHER_MIN_GV];
	int		nCHoleMinGV = (int)dPara[E_PARA_CHOLE_ROI_MIN_GV];

	//使用Round&CHole Cell有/无
	bool	bRoundUse = (dPara[E_PARA_ROUND_USE] > 0) ? true : false;
	bool	bCHoleUse = (dPara[E_PARA_CHOLE_USE] > 0) ? true : false;

	// 2021.11.23- Chole Point
	bool	bCholePointUse = (dPara[E_PARA_CHOLE_POINT_USE] > 0) ? true : false;

	//Round&CHole Cell里面能进多少Pixel......(只有曲线部分...)
	int		nRoundIn = (int)(dPara[E_PARA_ROUND_IN]);
	int		nCHoleIn = (int)(dPara[E_PARA_CHOLE_IN]);

	nCHoleIn = nCHoleIn * 2 + 1;

	int nShiftX = 0;
	int nShiftY = 0;

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////
	// ShiftCopy
	//////////////////////////////////////////////////////////////////////////
	cv::Mat matDstBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type());
	//需要添加参数

	//稍微缩小到区域内部...(区域)
	int nInPixel = 3;

	//如果处于PS模式
	if (nRatio != 1)
	{
		//获取Shift Copy Parameter
		int		nRedPattern = (int)dPara[E_PARA_SHIFT_COPY_R];
		int		nGreenPattern = (int)dPara[E_PARA_SHIFT_COPY_G];
		int		nBluePattern = (int)dPara[E_PARA_SHIFT_COPY_B];

		int nCpyX = 0, nCpyY = 0, nLoopX = 0, nLoopY = 0;

		//按模式...
		switch (nAlgImg)
		{
		case E_IMAGE_CLASSIFY_AVI_R:
		{
			if (nRedPattern == 0) break;
			ShiftCopyParaCheck(nRedPattern, nCpyX, nCpyY, nLoopX, nLoopY);
			nErrorCode = AlgoBase::ShiftCopy(matSrcBuf, matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
			matDstBuf.copyTo(matSrcBuf);
			matDstBuf.release();
		}
		break;

		case E_IMAGE_CLASSIFY_AVI_G:
		{
			if (nGreenPattern == 0) break;
			ShiftCopyParaCheck(nGreenPattern, nCpyX, nCpyY, nLoopX, nLoopY);
			nErrorCode = AlgoBase::ShiftCopy(matSrcBuf, matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
			matDstBuf.copyTo(matSrcBuf);
			matDstBuf.release();
		}
		break;

		case E_IMAGE_CLASSIFY_AVI_B:
		{
			if (nBluePattern == 0) break;
			ShiftCopyParaCheck(nBluePattern, nCpyX, nCpyY, nLoopX, nLoopY);
			nErrorCode = AlgoBase::ShiftCopy(matSrcBuf, matDstBuf, nCpyX, nCpyY, nLoopX, nLoopY);
			matDstBuf.copyTo(matSrcBuf);
			matDstBuf.release();
		}
		break;

		default:
			break;
		}

		nShiftX = nCpyX;
		nShiftY = nCpyY;

		if (nShiftX % 2 == 1) nShiftX += 1;
		if (nShiftY % 2 == 1) nShiftY += 1;

	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("ShiftCopy."));

	if (nSaveImageCount >= 0)
	{
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_ShiftCopy.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matSrcBuf);
	}

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		return nErrorCode;
	}

	//////////////////////////////////////////////////////////////////////////
		//仅用于AVI,用于在Cell数组之间填充空格
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matSrc_8bit = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);

	if (matSrcBuf.type() == CV_8UC1)
		matSrcBuf.copyTo(matSrc_8bit);
	else
		matSrcBuf.convertTo(matSrc_8bit, CV_8UC1, 1. / 16.);

	cv::Mat matTempBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1);

	//////////////////////////////////////////////////////////////////////////////
		//只为提高速度的外围寻找局部的运算-寻找局部的运算点。
	//////////////////////////////////////////////////////////////////////////////

	bool bUsePartialMorp = ((int)dPara[E_PARA_ALIGN_PARTIAL_USE] > 0) ? true : false;		//= true;
	int nFindCellROI_Threshold = (int)dPara[E_PARA_ALIGN_PARTIAL_THRESHOLD];						//= 10;
	int nExtROI_Outer = (int)dPara[E_PARA_ALIGN_PARTIAL_OUTER];							//= 200;
	int nExtROI_Inner_L = (int)dPara[E_PARA_ALIGN_PARTIAL_LEFT_OFFSET];					//= 800;
	int nExtROI_Inner_R = (int)dPara[E_PARA_ALIGN_PARTIAL_RIGHT_OFFSET];					//= 800;
	int nExtROI_Inner_T = (int)dPara[E_PARA_ALIGN_PARTIAL_TOP_OFFSET];					//= 300;
	int nExtROI_Inner_B = (int)dPara[E_PARA_ALIGN_PARTIAL_BOTTOM_OFFSET];					//= 300;

	cv::Rect rcFindCellROI;
	cv::Rect rcMorpROI[4]; // 0 : Left , 1 : Right , 2 : Top , 3 : Bottom

	bool bFindPartialMorpROI = false;

	//用作局部mopology-以后可能要关掉watch
	if (bUsePartialMorp)
	{
		cv::Mat matFindCellEdge_For_Morphology = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);
		bFindPartialMorpROI = true;

		cv::threshold(matSrc_8bit, matFindCellEdge_For_Morphology, nThreshold, 255.0, THRESH_BINARY); // 很容易找到,进化后再找。

		nErrorCode = FindCellEdge_For_Morphology(matFindCellEdge_For_Morphology, nFindCellROI_Threshold, rcFindCellROI);

		//如果是错误,则不执行局部mopology而执行错误代码输出,而是执行整个mopology。
		if (nErrorCode != E_ERROR_CODE_TRUE ||
			dPara[E_PARA_CELL_SIZE_X] * nRatio > rcFindCellROI.width ||
			dPara[E_PARA_CELL_SIZE_Y] * nRatio > rcFindCellROI.height)
		{
			bFindPartialMorpROI = false;
		}

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Find Morphology ROI"));

		if (nSaveImageCount >= 0)
		{
			CString strTemp;
			cv::Mat matSaveTmpROI = matFindCellEdge_For_Morphology(rcFindCellROI);
			strTemp.Format(_T("E:\\IMTC\\%02d_%02d_FindROI.jpg"), nAlgImg, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, matSaveTmpROI);
			matSaveTmpROI.release();
		}

		//创建要填充的区域-成功查找区域时
		if (bFindPartialMorpROI)
		{
			//将mopology区域分割为left,right,top和bottom区域。
			MakeRoI_For_Morphology(rcFindCellROI, nExtROI_Outer, nExtROI_Inner_L, nExtROI_Inner_R, nExtROI_Inner_T, nExtROI_Inner_B, matSrcBuf.size(), rcMorpROI);

			//因为只将外围区域填充为父区域,所以中间区域只填充灰色
			//=>以后只使用Thresholding后的Edge,不需要中间区域
			cv::Rect rcTempDrawRect;

			rcTempDrawRect.x = rcMorpROI[0].x + rcMorpROI[0].width;
			rcTempDrawRect.y = rcMorpROI[2].y + rcMorpROI[2].height;
			rcTempDrawRect.width = rcMorpROI[1].x - rcTempDrawRect.x;
			rcTempDrawRect.height = rcMorpROI[3].y - rcTempDrawRect.y;

			cv::Mat matTmpDrawRect = matTempBuf(rcTempDrawRect); // 涂上TempBuf才能涂到正确的目的地。

			//复制源文件以查找Cell中央Camera Hole

//matSrcBuf(rcTempDrawRect).copyTo(matTmpDrawRect);

			matTmpDrawRect.setTo(128); // 中间区域涂成128GV。

			matTmpDrawRect.release();
		}

		//查看mopology ROI结果
		if (nSaveImageCount >= 0)
		{
			cv::Mat matColor;
			cv::cvtColor(matSrc_8bit, matColor, COLOR_GRAY2RGB);

			cv::rectangle(matColor, rcMorpROI[0], cv::Scalar(255, 0, 0)); // Left
			cv::rectangle(matColor, rcMorpROI[1], cv::Scalar(255, 0, 0)); // Right

			cv::rectangle(matColor, rcMorpROI[2], cv::Scalar(0, 255, 0)); // Top
			cv::rectangle(matColor, rcMorpROI[3], cv::Scalar(0, 255, 0)); // Bottom

			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\%02d_%02d_MorpROI_Rect.jpg"), nAlgImg, nSaveImageCount++);
			cv::imwrite((cv::String)(CStringA)strTemp, matColor);

			matColor.release();
		}

		matFindCellEdge_For_Morphology.release();
	}

	///////////////////////////////////////////////////////////////////////////////

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));

		//使用部分冒号
		if (bUsePartialMorp && bFindPartialMorpROI)
		{
			Partial_Morphology(matSrc_8bit, matTempBuf, MORPH_CLOSE, StructElem, rcMorpROI); // 使用部分毛孔纸。
		}
		else
		{
			//Morphology Close(在Cell之间填充空格)
			cv::morphologyEx(matSrc_8bit, matTempBuf, MORPH_CLOSE, StructElem);
		}
		StructElem.release();
	}
	else
		matSrc_8bit.copyTo(matTempBuf);

	if (nSaveImageCount >= 0)
	{
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_Morp1.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf);
	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Morphology 1."));

	//////////////////////////////////////////////////////////////////////////
	// 2021.11.23- chole Point
		//查找Chole Roi(除了在图像中查找Chole之外,仅根据存储在Model中的ROI查找ROI)
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
		//查找Camera Hole
		//使用Morphology 2时,Camera Hole导致Contour异常
		//Morphology 2之前填充Camera Hole的目的
		//Morphology 1->Threshold->获取拐角坐标->角度和位置校正->CHole Find->CHole填充(原始)
		//在Parallel Morphology ROI中必须有Camera Hole
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matLabelBuf;
	if (nLabel_Flag)
	{
		matLabelBuf = cMem->GetMat(matTempBuf.size(), CV_8UC1, false);  //yuxuefei
	}

	if (bCHoleUse || bCholePointUse)
	{
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("CHole Fill Area Start."));

		cv::Mat matCHoleBuf;
		cv::Mat matCHoleBuf2;
		cv::Mat matCHoleBuf3;

		// Image Save 0 On / -1 Off
		//int nImgSave = -1;
		int nImgSave = (int)dPara[E_PARA_CHOLE_TEXT];

		matCHoleBuf = cMem->GetMat(matTempBuf.size(), CV_8UC1, false);
		matCHoleBuf2 = cMem->GetMat(matTempBuf.size(), CV_8UC1, false);
		matCHoleBuf3 = cMem->GetMat(matTempBuf.size(), CV_8UC1);
		cv::threshold(matTempBuf, matCHoleBuf, nThreshold, 255, THRESH_BINARY);

		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole Thres.bmp"), nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matCHoleBuf);
		}

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Threshold."));

		cv::Laplacian(matCHoleBuf, matCHoleBuf2, CV_8UC1);

		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole Raplacian.bmp"), nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matCHoleBuf2);
		}

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Laplacian."));

		//////////////////////////////////////////////////////////////////////////
		//查找点灯区域
		//////////////////////////////////////////////////////////////////////////	

		//整个外角线
		vector< vector< cv::Point2i > > contours;
		cv::findContours(matCHoleBuf2, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		int nMaxIndex = 0;
		double dMaxSize = 0.0;
		for (int i = 0; i < (int)contours.size(); i++)
		{
			double dValue = cv::contourArea(contours[i]);
			if (dValue > dMaxSize)
			{
				dMaxSize = dValue;
				nMaxIndex = i;
			}
		}
		//2022.11.10确认没有图像会爆炸的现象
		//如果找不到Cell,则跳过而不填充外围
		if (contours.size() == 0)
			return E_ERROR_CODE_TRUE;

		cv::RotatedRect BoundingBox = cv::minAreaRect(contours[nMaxIndex]);

		cv::Point2f vertices[E_CORNER_END];
		BoundingBox.points(vertices);

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Active Area Find."));

		//////////////////////////////////////////////////////////////////////////
		//查找4个拐角转角
		//////////////////////////////////////////////////////////////////////////

		cv::Point ptResCorner[E_CORNER_END], ptCornerAlign[E_CORNER_END];

		long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
		long	nHeight = (long)matSrcBuf.rows;	// 影像垂直尺寸	

		nErrorCode = FindCornerPoint(vertices, ptResCorner, nWidth, nHeight);

		//如果有错误,则输出错误代码
		if (nErrorCode != E_ERROR_CODE_TRUE)
			return nErrorCode;

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Find 4-Corner."));

		if (nImgSave >= 1)
		{
			cv::Mat matColor;
			cv::cvtColor(matSrcBuf, matColor, COLOR_GRAY2RGB);

			cv::line(matColor, ptResCorner[0], ptResCorner[1], cv::Scalar(255, 0, 0));
			cv::line(matColor, ptResCorner[1], ptResCorner[2], cv::Scalar(255, 0, 0));
			cv::line(matColor, ptResCorner[2], ptResCorner[3], cv::Scalar(255, 0, 0));
			cv::line(matColor, ptResCorner[3], ptResCorner[0], cv::Scalar(255, 0, 0));

			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_ROI_Rect.bmp"), nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matColor);

			matColor.release();
		}

		//////////////////////////////////////////////////////////////////////////
		//倾斜校正时,获取转角坐标
		//注册的曲线&ROI区域基于转角坐标
		//////////////////////////////////////////////////////////////////////////

		//计算旋转坐标时,使用
		double	dTheta = -BoundingBox.angle;

		//异常处理
		if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
		if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;

		dTheta *= PI;
		dTheta /= 180.0;
		double	dSin = sin(dTheta);
		double	dCos = cos(dTheta);
		double	dSin_ = sin(-dTheta);
		double	dCos_ = cos(-dTheta);
		int		nCx = matSrcBuf.cols / 2;
		int		nCy = matSrcBuf.rows / 2;

		for (int i = 0; i < E_CORNER_END; i++)
		{
			//旋转时计算预测坐标
			ptCornerAlign[i].x = (int)(dCos * (ptResCorner[i].x - nCx) - dSin * (ptResCorner[i].y - nCy) + nCx);
			ptCornerAlign[i].y = (int)(dSin * (ptResCorner[i].x - nCx) + dCos * (ptResCorner[i].y - nCy) + nCy);
		}

		//////////////////////////////////////////////////////////////////////////
		//注册的曲线&ROI区域->根据当前Grab进行坐标校正
		//////////////////////////////////////////////////////////////////////////

		//重新计算
		dTheta = BoundingBox.angle;

		//异常处理
		if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
		if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;

		dTheta *= PI;
		dTheta /= 180.0;
		dSin = sin(dTheta);
		dCos = cos(dTheta);
		dSin_ = sin(-dTheta);
		dCos_ = cos(-dTheta);

		//////////////////////////////////////////////////////////////////////////
		//查找注册曲线和ROI区域的位置
		//////////////////////////////////////////////////////////////////////////

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			//如果没有曲线数,则判断区域未注册/下一步...
			if (tCHoleSet[i].nContourCount <= 0)	continue;

			//左右上下移动,寻找偏差最大的地方(上下左右-pixel)
			const int nArrSize = (nFindCHoleOffset * 2 + 1) * (nFindCHoleOffset * 2 + 1);
			__int64* nSum = (__int64*)malloc(sizeof(__int64) * nArrSize);
			memset(nSum, 0, sizeof(__int64) * nArrSize);
			cv::Point2i ptTemp;

			//距离Align转角点最近的转角点
			int nCornerAlign = tCHoleSet[i].nCornerMinLength;

			const int nCount = (int)tCHoleSet[i].nContourCount;
			cv::Point* ptPoly = new cv::Point[nCount];
			memset(ptPoly, 0, sizeof(cv::Point) * nCount);

			int nMinX = matCHoleBuf.cols, nMaxX = 0;
			int nMinY = matCHoleBuf.rows, nMaxY = 0;

			//////////////////////////////////////////////////////////////////////////
			//虚线坐标校正
			//////////////////////////////////////////////////////////////////////////

			//如果存在曲线数量
			for (int j = 0; j < tCHoleSet[i].nContourCount; j++)
			{
				//从虚线坐标到曲线设置区域的最短距离校正
				//设备&每个舞台都存在偏差
				int XX = tCHoleSet[i].ptContours[j].x + ptCornerAlign[nCornerAlign].x;
				int YY = tCHoleSet[i].ptContours[j].y + ptCornerAlign[nCornerAlign].y;

				//旋转时,计算预测坐标
				ptTemp.x = (int)(dCos * (XX - nCx) - dSin * (YY - nCy) + nCx);
				ptTemp.y = (int)(dSin * (XX - nCx) + dCos * (YY - nCy) + nCy);

				//插入校正坐标
				ptPoly[j].x = ptTemp.x;
				ptPoly[j].y = ptTemp.y;

				//matCHoleBuf3.at<uchar>(ptPoly[j].y, ptPoly[j].x) = (uchar)255;

				if (nMinX > ptPoly[j].x) nMinX = ptPoly[j].x;
				if (nMaxX < ptPoly[j].x) nMaxX = ptPoly[j].x;
				if (nMinY > ptPoly[j].y) nMinY = ptPoly[j].y;
				if (nMaxY < ptPoly[j].y) nMaxY = ptPoly[j].y;

				if (bCholePointUse)
				{
					matCHoleBuf3.at<uchar>(ptPoly[j].y, ptPoly[j].x) = (uchar)255;
				}

			}

			//////////////////////////////////////////////////////////////////////////
			//CHole ROI最低GV检查
			//////////////////////////////////////////////////////////////////////////

			int nMeanROIOffset = 40;
			rcCHoleROI[i].x = nMinX - nMeanROIOffset;
			rcCHoleROI[i].y = nMinY - nMeanROIOffset;
			rcCHoleROI[i].width = (nMaxX - nMinX + 1) + nMeanROIOffset * 2;
			rcCHoleROI[i].height = (nMaxY - nMinY + 1) + nMeanROIOffset * 2;

			if (bCholePointUse)
			{
				matCHoleROIBuf[i] = matCHoleBuf3(rcCHoleROI[i]).clone();
				cv::floodFill(matCHoleROIBuf[i], cv::Point(matCHoleROIBuf[i].cols / 2, matCHoleROIBuf[i].rows / 2), cv::Scalar(255));

				cv::Mat	StructElem = cv::getStructuringElement(MORPH_CROSS, cv::Size(nCHoleIn + nShiftX, nCHoleIn + nShiftY));
				cv::morphologyEx(matCHoleROIBuf[i], matCHoleROIBuf[i], MORPH_DILATE, StructElem);
				StructElem.release();
			}
			if (!bCholePointUse)
			{

				//每个模式的原始画面GV差异->在二进制画面中进行比较(检查CHole Area大小？)
				cv::Scalar scrSrcMeanGV;
				scrSrcMeanGV = cv::mean(matCHoleBuf(rcCHoleROI[i]));

				//低于最低GV时CHole AD不良
				if (scrSrcMeanGV[0] < nCHoleMinGV) bCHoleAD[i] = true;

				// Log
				wchar_t wcLogTemp[MAX_PATH] = { 0 };
				swprintf_s(wcLogTemp, _T("CHole Mean (%.5f)"), scrSrcMeanGV[0]);
				writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, wcLogTemp);

				//////////////////////////////////////////////////////////////////////////
				//查找CHole
				//////////////////////////////////////////////////////////////////////////

				//如果存在曲线数量
				for (int j = 0; j < tCHoleSet[i].nContourCount; j++)
				{
					//左右上下移动,寻找偏差最大的地方
					int m = 0;
					for (int y = -nFindCHoleOffset; y <= nFindCHoleOffset; y++)
					{
						for (int x = -nFindCHoleOffset; x <= nFindCHoleOffset; x++, m++)
						{
							int indexX = ptPoly[j].x + x;
							int indexY = ptPoly[j].y + y;

							//异常处理
							if (indexX < 0)						continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;
							if (indexY < 0)						continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;
							if (indexX >= matCHoleBuf.cols)		continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;
							if (indexY >= matCHoleBuf.rows)		continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;

							nSum[m] += matCHoleBuf2.at<uchar>(indexY, indexX);
						}
					}
				}
				//左右上下移动,寻找偏差最大的地方
				long nMax = 0;
				int m = 0;
				int xx = 0;
				int yy = 0;
				for (int y = -nFindCHoleOffset; y <= nFindCHoleOffset; y++)
				{
					for (int x = -nFindCHoleOffset; x <= nFindCHoleOffset; x++, m++)
					{
						if (nMax < nSum[m])
						{
							nMax = nSum[m];
							xx = x;
							yy = y;
						}
					}
				}

				free(nSum);
				nSum = NULL;

				nMinX = matCHoleBuf.cols;
				nMinY = matCHoleBuf.rows;
				nMaxX = 0;
				nMaxY = 0;

				//修改位置
				for (int j = 0; j < tCHoleSet[i].nContourCount; j++)
				{
					ptPoly[j].x += xx;
					ptPoly[j].y += yy;

					//异常处理
					if (ptPoly[j].x < 0)					return E_ERROR_CODE_ALIGN_IMAGE_OVER;
					if (ptPoly[j].y < 0)					return E_ERROR_CODE_ALIGN_IMAGE_OVER;
					if (ptPoly[j].x >= matCHoleBuf2.cols)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
					if (ptPoly[j].y >= matCHoleBuf2.rows)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

					if (nMinX > ptPoly[j].x) nMinX = ptPoly[j].x;
					if (nMaxX < ptPoly[j].x) nMaxX = ptPoly[j].x;
					if (nMinY > ptPoly[j].y) nMinY = ptPoly[j].y;
					if (nMaxY < ptPoly[j].y) nMaxY = ptPoly[j].y;

					matCHoleBuf3.at<uchar>(ptPoly[j].y, ptPoly[j].x) = (uchar)255;
				}

				delete ptPoly;
				ptPoly = NULL;

				writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Find CHole."));

				//////////////////////////////////////////////////////////////////////////
								//画面CHole填充
				//////////////////////////////////////////////////////////////////////////

				int nCHoleROIOffSet = 40 + nCHoleIn;

				cv::Rect rectCHoleROI;
				rectCHoleROI.x = nMinX - nCHoleROIOffSet;
				rectCHoleROI.y = nMinY - nCHoleROIOffSet;
				rectCHoleROI.width = (nMaxX - nMinX + 1) + nCHoleROIOffSet * 2;
				rectCHoleROI.height = (nMaxY - nMinY + 1) + nCHoleROIOffSet * 2;

				//原始画面
				cv::Mat matsrcCHoleROI = matSrcBuf(rectCHoleROI);

				//二进制画面(平均Mask)
				cv::Mat matCHoleROI = cMem->GetMat(matsrcCHoleROI.size(), matsrcCHoleROI.type(), false);

				//原始二进制画面
				cv::Mat matSrcCHoleBuf = cMem->GetMat(matsrcCHoleROI.size(), matsrcCHoleROI.type(), false);

				//填充CHole的平均画面
				cv::Mat matCHoleFill = cMem->GetMat(matsrcCHoleROI.size(), matsrcCHoleROI.type());
				cv::Mat matCHoleFillX = cMem->GetMat(matsrcCHoleROI.size(), matsrcCHoleROI.type());
				cv::Mat matCHoleFillY = cMem->GetMat(matsrcCHoleROI.size(), matsrcCHoleROI.type());

				//CHole坐标画面
				cv::Mat matCHoleFillBuf = matCHoleBuf3(rectCHoleROI);

				//外角线连接&CHolein
				if (nCHoleIn >= 3)
				{
					cv::Mat	StructElem = cv::getStructuringElement(MORPH_CROSS, cv::Size(nCHoleIn + nShiftX, nCHoleIn + nShiftY));
					cv::morphologyEx(matCHoleFillBuf, matCHoleFillBuf, MORPH_DILATE, StructElem);
					StructElem.release();
				}
				else
				{
					cv::Mat	StructElem = cv::getStructuringElement(MORPH_CROSS, cv::Size(3, 3));
					cv::morphologyEx(matCHoleFillBuf, matCHoleFillBuf, MORPH_CLOSE, StructElem);
					StructElem.release();
				}

				cv::floodFill(matCHoleFillBuf, cv::Point(matCHoleFillBuf.cols / 2, matCHoleFillBuf.rows / 2), cv::Scalar(255));

				writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("CHoleIn."));

				//不是Pattern特定AD时
				if (!bCHoleAD[i])
				{
					nCHoleROIOffSet = 40;
					rcCHoleROI[i].x = nMinX - nCHoleROIOffSet;
					rcCHoleROI[i].y = nMinY - nCHoleROIOffSet;
					rcCHoleROI[i].width = (nMaxX - nMinX + 1) + nCHoleROIOffSet * 2;
					rcCHoleROI[i].height = (nMaxY - nMinY + 1) + nCHoleROIOffSet * 2;

					matCHoleROIBuf[i] = matCHoleBuf3(rcCHoleROI[i]).clone();
				}

				//创建CHole周围的掩码
				cv::blur(matCHoleFillBuf, matCHoleROI, cv::Size(61, 61));
				cv::subtract(matCHoleROI, matCHoleFillBuf, matCHoleROI);
				cv::threshold(matCHoleROI, matCHoleROI, 0.0, 255.0, THRESH_BINARY);

				cv::threshold(matsrcCHoleROI, matSrcCHoleBuf, nThreshold, 255.0, THRESH_BINARY);

				cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(5, 5));
				cv::morphologyEx(matSrcCHoleBuf, matSrcCHoleBuf, MORPH_CLOSE, StructElem);
				StructElem.release();

				cv::bitwise_and(matCHoleBuf(rectCHoleROI), matCHoleROI, matCHoleROI);

				cv::bitwise_and(matSrcCHoleBuf, matCHoleROI, matCHoleROI);

				//原始CHole填充宽度
				for (int nY = 2; nY < matsrcCHoleROI.rows - 2; nY += 3)
				{
					//原始画面
					cv::Mat matsrcFillROI = matsrcCHoleROI(cv::Rect(0, nY - 2, matsrcCHoleROI.cols, 5));
					// Mask
					cv::Mat matCHoleFillROI1 = matCHoleROI(cv::Rect(0, nY - 2, matsrcCHoleROI.cols, 5));

					cv::Scalar CHoleMeanGV = cv::mean(matsrcFillROI, matCHoleFillROI1);

					cv::Mat matCHoleFillROI2 = matCHoleFillX(cv::Rect(0, nY - 1, matsrcCHoleROI.cols, 3));

					matCHoleFillROI2.setTo(CHoleMeanGV[0]);
				}

				//原始CHole填充深度
				for (int nX = 2; nX < matsrcCHoleROI.cols - 2; nX += 3)
				{
					//原始画面
					cv::Mat matsrcFillROI = matsrcCHoleROI(cv::Rect(nX - 2, 0, 5, matsrcCHoleROI.rows));
					// Mask
					cv::Mat matCHoleFillROI1 = matCHoleROI(cv::Rect(nX - 2, 0, 5, matsrcCHoleROI.rows));

					cv::Scalar CHoleMeanGV = cv::mean(matsrcFillROI, matCHoleFillROI1);

					cv::Mat matCHoleFillROI2 = matCHoleFillY(cv::Rect(nX - 1, 0, 3, matsrcCHoleROI.rows));

					matCHoleFillROI2.setTo(CHoleMeanGV[0]);
				}

				// Minimum
				cv::min(matCHoleFillX, matCHoleFillY, matCHoleFill);

				// Aver
				//matCHoleFill = (matCHoleFillX + matCHoleFillY) / 2;

				// TEST
				//cv::Mat matSrcThresbuf = cMem->GetMat(matsrcCHoleROI.size(), matsrcCHoleROI.type(), false);
				//cv::threshold(matsrcCHoleROI, matSrcThresbuf, 30, 255, CV_THRESH_BINARY_INV);
				//
				//cv::bitwise_and(matSrcThresbuf, matCHoleFillBuf, matSrcThresbuf);

				//TEST
				//cv::bitwise_and(matCHoleFill, matSrcThresbuf, matCHoleFill);

								//只保留CHole区域
				cv::bitwise_and(matCHoleFill, matCHoleFillBuf, matCHoleFill);
				cv::blur(matCHoleFill, matCHoleFill, cv::Size(5, 5));
				cv::bitwise_and(matCHoleFill, matCHoleFillBuf, matCHoleFill);

				//填充原始画面
				cv::max(matsrcCHoleROI, matCHoleFill, matsrcCHoleROI);

				//为Morphology 2填充CHole
				cv::add(matTempBuf(rectCHoleROI), matCHoleFillBuf, matTempBuf(rectCHoleROI));

				if (nImgSave >= 1)
				{
					CString strTemp;
					strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole_Mean.bmp"), nAlgImg, nImgSave++);
					cv::imwrite((cv::String)(CStringA)strTemp, matCHoleFill);
				}

				if (nImgSave >= 1)
				{
					CString strTemp;
					strTemp.Format(_T("E:\\IMTC\\CHole\\%s_%02d_%02d_CHole_src_ROI.bmp"), strID, nAlgImg, nImgSave++);
					cv::imwrite((cv::String)(CStringA)strTemp, matsrcCHoleROI);
				}

				if (nImgSave >= 1)
				{
					CString strTemp;
					strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole_ROI.bmp"), nAlgImg, nImgSave++);
					cv::imwrite((cv::String)(CStringA)strTemp, matCHoleROI);
				}
			}
		}

		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole_Line.bmp"), nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matCHoleBuf2);
		}

		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole_Line_Fill.bmp"), nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matCHoleBuf3);
		}

		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%02d_%02d_CHole_FillArea.bmp"), nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf);
		}

		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%s_%02d_%02d_CHole_FillArea_src.bmp"), strID, nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matSrcBuf);
		}

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("CHole Fill Area End."));
	}

	//////////////////////////////////////////////////////////////////////////
		//18.10.31-如果缺陷在外围较亮,则亮点会导致转角位置错误
		//(以最小矩形查找转角点...也存在转角点位置相差50 pixel以上的情况)
		//从原始点亮区域中删除突出区域的目的
		//不妨增加寻找顶点的参数......(速度增加约1秒)
		//=>不能在矩形Cell中使用
	//////////////////////////////////////////////////////////////////////////

	cv::Mat matTemp2Buf = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false);
	matTempBuf.copyTo(matTemp2Buf);

	if (false)
	{
		//cv::Mat matBuf = matTempBuf.clone();

		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(101, 1));

		//使用部分冒号
		if (bUsePartialMorp && bFindPartialMorpROI)
		{
			Partial_Morphology(matTemp2Buf, matTempBuf, MORPH_OPEN, StructElem, rcMorpROI); // 使用部分毛孔纸。
		}
		else
		{
			cv::morphologyEx(matTemp2Buf, matTempBuf, MORPH_OPEN, StructElem);
		}

		StructElem.release();

		//////////////////////////////////////////////////////////////////////////

		StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(1, 101));

		//使用部分冒号
		if (bUsePartialMorp && bFindPartialMorpROI)
		{
			Partial_Morphology(matTempBuf, matTemp2Buf, MORPH_OPEN, StructElem, rcMorpROI); // 使用部分毛孔纸。
		}
		else
		{
			cv::morphologyEx(matTempBuf, matTemp2Buf, MORPH_OPEN, StructElem);
		}

		StructElem.release();

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Morphology 2."));
	}

	if (nSaveImageCount >= 0)
	{
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_Morp2.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTemp2Buf);
	}

	// Threshold
	cv::threshold(matTemp2Buf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);
	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("threshold."));

	if (nSaveImageCount >= 0)
	{
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_threshold.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf);
	}

	//////////////////////////////////////////////////////////////////////////
		//查找Cell区域
	//////////////////////////////////////////////////////////////////////////

		//检查区域Rect
	cv::Rect rectCell;
	nErrorCode = FindCellArea(matTempBuf, nMinArea, rectCell);

	if (nSaveImageCount >= 0)
	{
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_Area.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matTempBuf);
	}

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("FindCellArea."));

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
	{
		matTempBuf.release();
		return nErrorCode;
	}

	//Temp内存分配
	cv::Mat matThreshBuf = cMem->GetMat(matTempBuf.size(), matTempBuf.type());

	//使用Edge查找偏差较大的地方
//matTempBuf.copyTo(matThreshBuf);
	//成功查找区域时,拉普拉西亚也会部分执行。
	if (bUsePartialMorp && bFindPartialMorpROI)
	{
		Partial_Laplacian(matTempBuf, matThreshBuf, rcMorpROI);
	}
	else
	{
		cv::Laplacian(matTempBuf, matThreshBuf, CV_8U);
	}

	if (nSaveImageCount >= 0)
	{
		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_Laplacian.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matThreshBuf);
	}

	//////////////////////////////////////////////////////////////////////////
	//查找点亮区域
	//////////////////////////////////////////////////////////////////////////	

	//整个外角线
	vector< vector< cv::Point2i > > contours;
	cv::findContours(matTempBuf, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	int nMaxIndex = 0;
	double dMaxSize = 0.0;
	for (int i = 0; i < (int)contours.size(); i++)
	{
		double dValue = cv::contourArea(contours[i]);
		if (dValue > dMaxSize)
		{
			dMaxSize = dValue;
			nMaxIndex = i;
		}
	}

	cv::RotatedRect BoundingBox = cv::minAreaRect(contours[nMaxIndex]);

	cv::Point2f vertices[E_CORNER_END];
	BoundingBox.points(vertices);

	//////////////////////////////////////////////////////////////////////////
	//查找4个拐角转角
	//////////////////////////////////////////////////////////////////////////
	cv::Point ptResCorner[E_CORNER_END], ptCornerAlign[E_CORNER_END];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 影像垂直尺寸	

	nErrorCode = FindCornerPoint(vertices, ptResCorner, nWidth, nHeight);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Find 4-Corner."));

	//检查Align结果
	if (nSaveImageCount >= 0)
	{
		cv::Mat matColor;
		cv::cvtColor(matSrcBuf, matColor, COLOR_GRAY2RGB);

		cv::line(matColor, ptResCorner[0], ptResCorner[1], cv::Scalar(255, 0, 0));
		cv::line(matColor, ptResCorner[1], ptResCorner[2], cv::Scalar(255, 0, 0));
		cv::line(matColor, ptResCorner[2], ptResCorner[3], cv::Scalar(255, 0, 0));
		cv::line(matColor, ptResCorner[3], ptResCorner[0], cv::Scalar(255, 0, 0));

		CString strTemp;
		strTemp.Format(_T("E:\\IMTC\\%02d_%02d_Line.jpg"), nAlgImg, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matColor);

		matColor.release();
	}

	//////////////////////////////////////////////////////////////////////////
	//稍微缩小到区域内部...
	//////////////////////////////////////////////////////////////////////////

	ptResCorner[E_CORNER_LEFT_TOP].x += nInPixel;
	ptResCorner[E_CORNER_LEFT_TOP].y += nInPixel;
	ptResCorner[E_CORNER_RIGHT_TOP].x -= nInPixel;
	ptResCorner[E_CORNER_RIGHT_TOP].y += nInPixel;
	ptResCorner[E_CORNER_LEFT_BOTTOM].x += nInPixel;
	ptResCorner[E_CORNER_LEFT_BOTTOM].y -= nInPixel;
	ptResCorner[E_CORNER_RIGHT_BOTTOM].x -= nInPixel;
	ptResCorner[E_CORNER_RIGHT_BOTTOM].y -= nInPixel;

	//////////////////////////////////////////////////////////////////////////
	//检查Rect范围

	//设置大范围(两侧必须有200 Pixel空背景)
	//填充空的平均值
	int offset = 100;

	CRect rectROI = new CRect(
		min(ptResCorner[E_CORNER_LEFT_TOP].x, ptResCorner[E_CORNER_LEFT_BOTTOM].x) - offset,
		min(ptResCorner[E_CORNER_LEFT_TOP].y, ptResCorner[E_CORNER_RIGHT_TOP].y) - offset,
		max(ptResCorner[E_CORNER_RIGHT_TOP].x, ptResCorner[E_CORNER_RIGHT_BOTTOM].x) + offset,
		max(ptResCorner[E_CORNER_LEFT_BOTTOM].y, ptResCorner[E_CORNER_RIGHT_BOTTOM].y) + offset);

	//如果扫描区域超出画面大小
	if (rectROI.left < 0)	rectROI.left = 0;
	if (rectROI.top < 0)	rectROI.top = 0;
	if (rectROI.right >= nWidth)	rectROI.right = nWidth - 1;
	if (rectROI.bottom >= nHeight)	rectROI.bottom = nHeight - 1;

	//如果扫描区域超出画面大小
//if( rectROI.left	<	0		||
//	rectROI.top		<	0		||
//	rectROI.right	>= 	nWidth	||
//	rectROI.bottom	>= 	nHeight	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//////////////////////////////////////////////////////////////////////////
		//曲线处理

		//初始化掩码
	matTempBuf.setTo(0);

	int npt[] = { E_CORNER_END };

	//禁用曲线时(如果是Rect)
	if (!bRoundUse)
	{
		const cv::Point* ppt[1] = { ptResCorner };

		//点亮区域掩码
		cv::fillPoly(matTempBuf, ppt, npt, 1, cv::Scalar(255, 255, 255));
	}

	//曲线部分处理
	else
	{
		//分配Temp内存
		cv::Mat matRndBuf = cMem->GetMat(matTempBuf.size(), matTempBuf.type());

		//////////////////////////////////////////////////////////////////////////
		//倾斜校正时,获取转角坐标
		//注册的曲线&ROI区域基于转角坐标
		//////////////////////////////////////////////////////////////////////////

		//计算旋转坐标时,使用
		double	dTheta = -BoundingBox.angle;

		//异常处理
		if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
		if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;

		dTheta *= PI;
		dTheta /= 180.0;
		double	dSin = sin(dTheta);
		double	dCos = cos(dTheta);
		double	dSin_ = sin(-dTheta);
		double	dCos_ = cos(-dTheta);
		int		nCx = matSrcBuf.cols / 2;
		int		nCy = matSrcBuf.rows / 2;

		for (int i = 0; i < E_CORNER_END; i++)
		{
			//旋转时计算预测坐标
			ptCornerAlign[i].x = (int)(dCos * (ptResCorner[i].x - nCx) - dSin * (ptResCorner[i].y - nCy) + nCx);
			ptCornerAlign[i].y = (int)(dSin * (ptResCorner[i].x - nCx) + dCos * (ptResCorner[i].y - nCy) + nCy);
		}

		//////////////////////////////////////////////////////////////////////////
		//注册的曲线&ROI区域->根据当前Grab进行坐标校正和定位
		//////////////////////////////////////////////////////////////////////////

		//重新计算
		dTheta = BoundingBox.angle;

		//异常处理
		if (45.0 < dTheta && dTheta < 135.0)	dTheta -= 90.0;
		if (-45.0 > dTheta && dTheta > -135.0)	dTheta += 90.0;

		dTheta *= PI;
		dTheta /= 180.0;
		dSin = sin(dTheta);
		dCos = cos(dTheta);
		dSin_ = sin(-dTheta);
		dCos_ = cos(-dTheta);

		cv::Point	ptSE[2], ptTempLine[2];
		int			nSE;

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			//如果没有曲线数,则判断区域未注册/下一步...
			if (tRoundSet[i].nContourCount <= 0)	continue;

			//左右上下移动,寻找偏差最大的地方(上下左右-pixel)
			const int nArrSize = (nFindRoundOffset * 2 + 1) * (nFindRoundOffset * 2 + 1);
			__int64* nSum = (__int64*)malloc(sizeof(__int64) * nArrSize);
			memset(nSum, 0, sizeof(__int64) * nArrSize);
			cv::Point2i ptTemp;

			//点灯区域外的转角点数
			int nOutsideCnt = 0;
			for (int j = 0; j < E_CORNER_END; j++)
			{
				if (!tRoundSet[i].nCornerInside[j])
					nOutsideCnt++;
			}

			//如果一个都没有,则设置错误
// 			if( nOutsideCnt == 0 )
// 				AfxMessageBox(_T("Set Corner Err !!!"));

			const int nCount = (int)tRoundSet[i].nContourCount;
			cv::Point* ptPoly = new cv::Point[nCount];
			memset(ptPoly, 0, sizeof(cv::Point) * nCount);

			// Align顶点最近的顶点
			int nCornerAlign = tRoundSet[i].nCornerMinLength;

			//如果存在曲线数量
			for (int j = 0; j < tRoundSet[i].nContourCount; j++)
			{
				//从虚线坐标到曲线设置区域的最短距离…校正
				//设备&每个舞台都存在偏差
				int XX = tRoundSet[i].ptContours[j].x + ptCornerAlign[nCornerAlign].x;
				int YY = tRoundSet[i].ptContours[j].y + ptCornerAlign[nCornerAlign].y;

				// 旋转时，计算预测坐标
				ptTemp.x = (int)(dCos * (XX - nCx) - dSin * (YY - nCy) + nCx);
				ptTemp.y = (int)(dSin * (XX - nCx) + dCos * (YY - nCy) + nCy);

				//插入校正坐标
				ptPoly[j].x = ptTemp.x;
				ptPoly[j].y = ptTemp.y;

				// 左右上下移动，寻找偏差最大的地方
				int m = 0;
				for (int y = -nFindRoundOffset; y <= nFindRoundOffset; y++)
				{
					for (int x = -nFindRoundOffset; x <= nFindRoundOffset; x++, m++)
					{
						int indexX = ptTemp.x + x;
						int indexY = ptTemp.y + y;

						//异常处理
						if (indexX < 0)						continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;
						if (indexY < 0)						continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;
						if (indexX >= matThreshBuf.cols)		continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;
						if (indexY >= matThreshBuf.rows)		continue;	//return E_ERROR_CODE_ALIGN_IMAGE_OVER;

						//17.07.19-使用二进制画面计算偏差/作为源,在单一模式中查找空白部分时发生
	//int k	= abs(matThreshBuf.at<uchar>(indexY+1, indexX) - matThreshBuf.at<uchar>(indexY-1, indexX))
	//		+ abs(matThreshBuf.at<uchar>(indexY, indexX+1) - matThreshBuf.at<uchar>(indexY, indexX-1));

	//nSum[m++] += k;

						nSum[m] += matThreshBuf.at<uchar>(indexY, indexX);
					}
				}
			}

			//左右上下移动,寻找偏差最大的地方
			long nMax = 0;
			int m = 0;
			int xx = 0;
			int yy = 0;
			for (int y = -nFindRoundOffset; y <= nFindRoundOffset; y++)
			{
				for (int x = -nFindRoundOffset; x <= nFindRoundOffset; x++, m++)
				{
					if (nMax < nSum[m])
					{
						nMax = nSum[m];
						xx = x;
						yy = y;
					}
				}
			}
			free(nSum);
			nSum = NULL;

			//曲线部分,轻轻地放在点灯区域内(取决于情况)
			cv::Point ptRnd = calcRoundIn(tRoundSet, i, nRoundIn);
			xx += ptRnd.x;
			yy += ptRnd.y;

			//如果在该范围内未找到
			//设备&每个舞台都存在偏差
			//提高查找范围(nFindOffset)或在原始位置绘制
// 			if( nMax == 0 )
// 				AfxMessageBox(_T("Find Range Over Err !!!"));

						//如果找到
			{
				//修改位置
				int j = 0;
				for (; j < tRoundSet[i].nContourCount; j++)
				{
					ptPoly[j].x += xx;
					ptPoly[j].y += yy;

					//异常处理
					if (ptPoly[j].x < 0)					return E_ERROR_CODE_ALIGN_IMAGE_OVER;
					if (ptPoly[j].y < 0)					return E_ERROR_CODE_ALIGN_IMAGE_OVER;
					if (ptPoly[j].x >= matRndBuf.cols)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;
					if (ptPoly[j].y >= matRndBuf.rows)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;

					matRndBuf.at<uchar>(ptPoly[j].y, ptPoly[j].x) = (uchar)255;
				}
			}

			//////////////////////////////////////////////////////////////////////////
			//非曲线直线连接
			//////////////////////////////////////////////////////////////////////////

			//结束点
			ptTempLine[0] = ptPoly[0];
			ptTempLine[1] = ptPoly[nCount - 1];

			delete ptPoly;
			ptPoly = NULL;

			//注册结束点
			if (i == 0)
			{
				ptSE[0] = ptTempLine[0];
				ptSE[1] = ptTempLine[1];
			}
			//连接端点
			else
			{
				//非曲线直线连接
				calcLineConnect(matRndBuf, ptSE, ptTempLine, nSE, i);
			}

			// 			delete ptPoly;
			// 			ptPoly = NULL;
		}
		//连接最后一个端点
		cv::line(matRndBuf, ptSE[0], ptSE[1], cv::Scalar(255, 255, 255));

		//17.11.12-连接断点
		cv::dilate(matRndBuf, matTempBuf, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

		//向轮廓内填充(4方向？)
		cv::floodFill(matTempBuf, BoundingBox.center, cv::Scalar(255, 255, 255));

		matRndBuf.release();
	}
	//17.07.17曲线部分处理
//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Edge Corner Find."));

	// Label PolMark 添加 2024.07
	InspectLabelPol inpectLabelPol;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE && (nLabel_Flag == 1 || nPolNum_Flag == 1 || nPolSign_Flag == 1)) {
		if (inpectLabelPol.DoFindLabelMark(matSrcBuf, labelMarkParams, labelMarkInfo) == 0) {
			labelMarkInfo.bFindEnd = true;
		}
	}

	if (!bPolSaveTemplate && (nLabel_Flag == 1 || nPolNum_Flag == 1 || nPolSign_Flag == 1)) {
		// 等待白画面定位PolMark结束
		while (!labelMarkInfo.bFindEnd) {
			Sleep(20);
		}

		inpectLabelPol.DoFillLabelMark(matSrcBuf, labelMarkParams, labelMarkInfo);
	}
	

	cv::bitwise_not(matTempBuf, matBKGBuf);

	cv::Mat matEdgeArea = cMem->GetMat(matTempBuf.size(), CV_8UC1, false);

	//留下一点点亮区域轮廓(点亮区域平均时使用)
	nErrorCode = FindEdgeArea(matTempBuf, matEdgeArea, offset, cMem);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	cv::imwrite(("E:\\IMTC\\0.EdgeArea.bmp"), matEdgeArea);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Edge Area Find."));

	//平均缓冲内存分配
	cv::Mat matMeanBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	matSrcBuf.copyTo(matMeanBuf);

	//Offset(点亮区域平均大小)
	//nSeg x nSeg区域内必须存在点亮区域
	//6.18"需要增加大小(300->500)
	//左右上下offset(目前为70)
	//x,y分割使用
	int nSegX = 3;
	int nSegY = offset + offset;

	nErrorCode = FillAreaMeanX(matMeanBuf, matEdgeArea, rectROI, nSegX, nSegY, nRoundMinGV);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//cv::imwrite(("E:\\IMTC\\1.Mean.bmp"), matMeanBuf);
	//cv::imwrite(("E:\\IMTC\\1.EdgeArea.bmp"), matEdgeArea);

	//Offset(点亮区域平均大小)
	//nSeg x nSeg区域内必须存在点亮区域
	//6.18"需要增加大小(300->500)
	//左右上下offset(目前为70)
	//x,y分割使用
	nSegX = 800;
	nSegY = 3;

	nErrorCode = FillAreaMeanY(matMeanBuf, matEdgeArea, rectROI, nSegX, nSegY, nRoundMinGV);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//cv::imwrite(("E:\\IMTC\\2.Mean.bmp"), matMeanBuf);
	//cv::imwrite(("E:\\IMTC\\2.EdgeArea.bmp"), matEdgeArea);

	//只保留已点亮的部分
	nErrorCode = FillMerge(matSrcBuf, matMeanBuf, matTempBuf, nAlgImg, cMem);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//2022.10.14 G3外围填充test
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_GRAY_128)
	{
		matSrcBuf /= dMulti;
	}
	//cv::imwrite(("E:\\IMTC\\3.Mean.bmp"), matMeanBuf);
	//cv::imwrite(("E:\\IMTC\\3.EdgeArea.bmp"), matEdgeArea);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Edge Fill Average."));

	//禁用内存
	matMeanBuf.release();
	matTempBuf.release();
	matThreshBuf.release();

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//外围处理
long CInspectAlign::DoFillOutArea_SVI(cv::Mat& matSrcBuf, cv::Mat& matBKGBuf, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA],
	double* dPara, int nAlgImg, int nCameraNum, int nRatio, wchar_t* strID, cv::Point* ptCorner)
{
	//不包括Black模式
	if (nAlgImg == E_IMAGE_CLASSIFY_SVI_BLACK)		return E_ERROR_CODE_TRUE;

	//如果没有缓冲区。
	if (matSrcBuf.empty())							return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

		//二进制
	//int		nThreshold		= (int)dPara[E_PARA_SVI_ALIGN_THRESHOLD];

		//设置Round区域范围
	int		nCornerSize = (int)dPara[E_PARA_SVI_ROUND_SIZE];

	//使用Round Cell有/无
	int		nRoundUse = (int)dPara[E_PARA_SVI_ROUND_USE];

	//向内检查
	int		nRoundIn = (int)dPara[E_PARA_SVI_ROUND_IN];

	//Round区域拐角设置高度
	int		nDiagonal = (int)dPara[E_PARA_SVI_ROUND_DIAGONAL];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////
		//稍微缩小到Align区域的内部...
	//////////////////////////////////////////////////////////////////////////
	cv::Point ptCornerIn[E_CORNER_END];

	ptCornerIn[E_CORNER_LEFT_TOP].x = ptCorner[E_CORNER_LEFT_TOP].x + nRoundIn;
	ptCornerIn[E_CORNER_LEFT_TOP].y = ptCorner[E_CORNER_LEFT_TOP].y + nRoundIn;
	ptCornerIn[E_CORNER_RIGHT_TOP].x = ptCorner[E_CORNER_RIGHT_TOP].x - nRoundIn;
	ptCornerIn[E_CORNER_RIGHT_TOP].y = ptCorner[E_CORNER_RIGHT_TOP].y + nRoundIn;
	ptCornerIn[E_CORNER_LEFT_BOTTOM].x = ptCorner[E_CORNER_LEFT_BOTTOM].x + nRoundIn;
	ptCornerIn[E_CORNER_LEFT_BOTTOM].y = ptCorner[E_CORNER_LEFT_BOTTOM].y - nRoundIn;
	ptCornerIn[E_CORNER_RIGHT_BOTTOM].x = ptCorner[E_CORNER_RIGHT_BOTTOM].x - nRoundIn;
	ptCornerIn[E_CORNER_RIGHT_BOTTOM].y = ptCorner[E_CORNER_RIGHT_BOTTOM].y - nRoundIn;

	//////////////////////////////////////////////////////////////////////////
		//点亮区域
	//////////////////////////////////////////////////////////////////////////

		//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	CRect rectROI = CRect(
		min(ptCornerIn[E_CORNER_LEFT_TOP].x, ptCornerIn[E_CORNER_LEFT_BOTTOM].x),
		min(ptCornerIn[E_CORNER_LEFT_TOP].y, ptCornerIn[E_CORNER_RIGHT_TOP].y),
		max(ptCornerIn[E_CORNER_RIGHT_TOP].x, ptCornerIn[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCornerIn[E_CORNER_LEFT_BOTTOM].y, ptCornerIn[E_CORNER_RIGHT_BOTTOM].y));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	cv::Mat matTempBuf = cv::Mat::zeros(matSrcBuf.size(), CV_8UC1);

	//Cell点亮区域
	for (int i = 0; i < E_CORNER_END; i++)
		cv::line(matTempBuf, ptCornerIn[i], ptCornerIn[(i + 1) % 4], cv::Scalar(255, 255, 255), 1);

	//////////////////////////////////////////////////////////////////////////
		//顶点4方向Round Cell
	//////////////////////////////////////////////////////////////////////////

	if (nRoundUse > 0)
	{
		//使用外接十字标记清除
		for (int i = 0; i < E_CORNER_END; i++)
		{
			cv::line(matTempBuf, cv::Point(ptCornerIn[i].x - nCornerSize, ptCornerIn[i].y), cv::Point(ptCornerIn[i].x + nCornerSize, ptCornerIn[i].y), cv::Scalar(0, 0, 0), 1);
			cv::line(matTempBuf, cv::Point(ptCornerIn[i].x, ptCornerIn[i].y - nCornerSize), cv::Point(ptCornerIn[i].x, ptCornerIn[i].y + nCornerSize), cv::Scalar(0, 0, 0), 1);
		}

		cv::Point pt1, pt2, pt3;

		// E_CORNER_LEFT_TOP
		pt1 = cv::Point(ptCornerIn[E_CORNER_LEFT_TOP].x + nCornerSize, ptCornerIn[E_CORNER_LEFT_TOP].y);
		pt2 = cv::Point(ptCornerIn[E_CORNER_LEFT_TOP].x, ptCornerIn[E_CORNER_LEFT_TOP].y + nCornerSize);
		pt3.x = (pt1.x + pt2.x) / 2 - nDiagonal;
		pt3.y = (pt1.y + pt2.y) / 2 - nDiagonal;
		cv::line(matTempBuf, pt1, pt3, cv::Scalar(255, 255, 255), 1);
		cv::line(matTempBuf, pt2, pt3, cv::Scalar(255, 255, 255), 1);

		// E_CORNER_RIGHT_TOP
		pt1 = cv::Point(ptCornerIn[E_CORNER_RIGHT_TOP].x - nCornerSize, ptCornerIn[E_CORNER_RIGHT_TOP].y);
		pt2 = cv::Point(ptCornerIn[E_CORNER_RIGHT_TOP].x, ptCornerIn[E_CORNER_RIGHT_TOP].y + nCornerSize);
		pt3.x = (pt1.x + pt2.x) / 2 + nDiagonal;
		pt3.y = (pt1.y + pt2.y) / 2 - nDiagonal;
		cv::line(matTempBuf, pt1, pt3, cv::Scalar(255, 255, 255), 1);
		cv::line(matTempBuf, pt2, pt3, cv::Scalar(255, 255, 255), 1);

		// E_CORNER_RIGHT_BOTTOM
		pt1 = cv::Point(ptCornerIn[E_CORNER_RIGHT_BOTTOM].x - nCornerSize, ptCornerIn[E_CORNER_RIGHT_BOTTOM].y);
		pt2 = cv::Point(ptCornerIn[E_CORNER_RIGHT_BOTTOM].x, ptCornerIn[E_CORNER_RIGHT_BOTTOM].y - nCornerSize);
		pt3.x = (pt1.x + pt2.x) / 2 + nDiagonal;
		pt3.y = (pt1.y + pt2.y) / 2 + nDiagonal;
		cv::line(matTempBuf, pt1, pt3, cv::Scalar(255, 255, 255), 1);
		cv::line(matTempBuf, pt2, pt3, cv::Scalar(255, 255, 255), 1);

		// E_CORNER_LEFT_BOTTOM
		pt1 = cv::Point(ptCornerIn[E_CORNER_LEFT_BOTTOM].x + nCornerSize, ptCornerIn[E_CORNER_LEFT_BOTTOM].y);
		pt2 = cv::Point(ptCornerIn[E_CORNER_LEFT_BOTTOM].x, ptCornerIn[E_CORNER_LEFT_BOTTOM].y - nCornerSize);
		pt3.x = (pt1.x + pt2.x) / 2 - nDiagonal;
		pt3.y = (pt1.y + pt2.y) / 2 + nDiagonal;
		cv::line(matTempBuf, pt1, pt3, cv::Scalar(255, 255, 255), 1);
		cv::line(matTempBuf, pt2, pt3, cv::Scalar(255, 255, 255), 1);
	}

	//////////////////////////////////////////////////////////////////////////
		//凹槽(Notch)Round Cell
	//////////////////////////////////////////////////////////////////////////

	if (nRoundUse > 1)
	{
		//横向60 pixel
		int nSizeX = 60;

		//垂直方向70%
		int nSizeY = (int)(rectROI.Height() * 0.7);

		//空白
		int nEmptyY = (rectROI.Height() - nSizeY) / 2;

		//剪切Notch部分
		cv::Mat matNotchBuf = matSrcBuf(cv::Rect(rectROI.left, rectROI.top + nEmptyY, nSizeX, nSizeY));

		// Gray -> Color
		cv::Mat matNotchGrayBuf, matNotchEdgeBuf;
		cv::cvtColor(matNotchBuf, matNotchGrayBuf, COLOR_RGB2GRAY);

		// Blur
		//cv::blur(matNotchGrayBuf, matNotchEdgeBuf, cv::Size(3, 3));

		// Threshold
		cv::threshold(matNotchGrayBuf, matNotchEdgeBuf, 10, 255.0, THRESH_BINARY);

		// Edge
		cv::Laplacian(matNotchEdgeBuf, matNotchEdgeBuf, CV_8UC1);

		//////////////////////////////////////////////////////////////////////////

		int nMax, nTemp, nIndexX = 0, nIndexY1 = 0, nIndexY2 = 0;

		nMax = 0;
		for (int y = 1; y < matNotchEdgeBuf.rows / 2 - 1; y++)
		{
			nTemp = (int)(cv::sum(matNotchEdgeBuf.row(y - 1))[0]);
			nTemp += (int)(cv::sum(matNotchEdgeBuf.row(y))[0]);
			nTemp += (int)(cv::sum(matNotchEdgeBuf.row(y + 1))[0]);

			if (nMax < nTemp)
			{
				nMax = nTemp;
				nIndexY1 = y;
			}
		}

		nMax = 0;
		for (int y = matNotchEdgeBuf.rows / 2 + 1; y < matNotchEdgeBuf.rows - 1; y++)
		{
			nTemp = (int)(cv::sum(matNotchEdgeBuf.row(y - 1))[0]);
			nTemp += (int)(cv::sum(matNotchEdgeBuf.row(y))[0]);
			nTemp += (int)(cv::sum(matNotchEdgeBuf.row(y + 1))[0]);

			if (nMax < nTemp)
			{
				nMax = nTemp;
				nIndexY2 = y;
			}
		}

		nMax = 0;
		for (int x = 0; x < matNotchEdgeBuf.cols; x++)
		{
			nTemp = (int)(cv::sum(matNotchEdgeBuf.col(x))[0]);

			if (nMax < nTemp)
			{
				nMax = nTemp;
				nIndexX = x;
			}
		}

		//////////////////////////////////////////////////////////////////////////

		cv::Mat matNotchROIBuf = matTempBuf(cv::Rect(rectROI.left, rectROI.top + nEmptyY, nSizeX, nSizeY));

		cv::line(matNotchROIBuf, cv::Point(0, nIndexY1 - 9), cv::Point(0, nIndexY2 + 9), cv::Scalar(0, 0, 0), 1);

		cv::line(matNotchROIBuf, cv::Point(0, nIndexY1 - 10), cv::Point(nIndexX + 5, nIndexY1 - 10), cv::Scalar(255, 255, 255), 1);
		cv::line(matNotchROIBuf, cv::Point(nIndexX + 5, nIndexY2 + 10), cv::Point(nIndexX + 5, nIndexY1 - 10), cv::Scalar(255, 255, 255), 1);
		cv::line(matNotchROIBuf, cv::Point(0, nIndexY2 + 10), cv::Point(nIndexX + 5, nIndexY2 + 10), cv::Scalar(255, 255, 255), 1);
	}

	//////////////////////////////////////////////////////////////////////////
		//提取背景区域
	//////////////////////////////////////////////////////////////////////////

		//向轮廓内填充
	cv::floodFill(matTempBuf, cv::Point(rectROI.CenterPoint().x, rectROI.CenterPoint().y), cv::Scalar(255, 255, 255));

	//背景区域
	cv::bitwise_not(matTempBuf, matBKGBuf);

	//////////////////////////////////////////////////////////////////////////
		//填充周围
	//////////////////////////////////////////////////////////////////////////	

		//设置大范围(两侧必须有200 Pixel空背景)
	int offset = 70;

	rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x) - offset,
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y) - offset,
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x) + offset,
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y) + offset);

	//如果扫描区域超出画面大小
	if (rectROI.left < 0)	rectROI.left = 0;
	if (rectROI.top < 0)	rectROI.top = 0;
	if (rectROI.right >= nWidth)	rectROI.right = nWidth - 1;
	if (rectROI.bottom >= nHeight)	rectROI.bottom = nHeight - 1;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//留下一点点亮区域轮廓(点亮区域平均时使用)
	cv::Mat matEdgeArea;
	nErrorCode = FindEdgeArea_SVI(matTempBuf, matEdgeArea, 30);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//平均缓冲区
	cv::Mat matMeanBuf = matSrcBuf.clone();

	//横向平均填充
	nErrorCode = FillAreaMeanX_SVI(matMeanBuf, matEdgeArea, rectROI, 3, 100, 0);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//垂直平均填充
	nErrorCode = FillAreaMeanY_SVI(matMeanBuf, matEdgeArea, rectROI, 200, 3, 0);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	// Gray -> Color
	cv::cvtColor(matBKGBuf, matTempBuf, COLOR_GRAY2RGB);

	//只保留已点亮的部分
	nErrorCode = FillMerge_SVI(matSrcBuf, matMeanBuf, matTempBuf);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//取消分配
	if (!matTempBuf.empty())	matTempBuf.release();
	if (!matEdgeArea.empty())	matEdgeArea.release();
	if (!matMeanBuf.empty())	matMeanBuf.release();

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::DoFillOutArea_APP(cv::Mat& matSrcBuf, cv::Mat& matBKGBuf, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], double* dPara, int nAlgImg, int nCameraNum, int nRatio, wchar_t* strID, cv::Point* ptCorner, vector<vector<Point2i>>& ptActive, double dAlignTH, CString strPath, bool bImageSave)
{
	Mat mtOrg;
	matSrcBuf.copyTo(mtOrg);

	Mat mtContour = Mat::zeros(mtOrg.size(), mtOrg.type());

	DoRotateImage(mtOrg, mtOrg, dAlignTH);

	//减少Tact时间
	cv::resize(mtOrg, mtOrg, Size(), 0.25, 0.25);

	Mat mtRef[MAX_MEM_SIZE_E_INSPECT_AREA];
	Mat mtMatchRst;
	CString strRefImagePath;
	strRefImagePath.Format(_T("%s\\CornerEdge"), strPath);

	vector<vector<Point2i>> ptContours;
	vector<Point2i>			pt;

	double dMatchRate;
	Point  ptMatch;
	Point  ptMatchResize;
	for (int nROIInx = 0; nROIInx < MAX_MEM_SIZE_E_INSPECT_AREA; nROIInx++)
	{
		strRefImagePath.Format(_T("%s\\CornerEdge\\%d.bmp"), strPath, nROIInx);
		mtRef[nROIInx] = imread((cv::String)(CStringA)strRefImagePath, -1);
		if (mtRef[nROIInx].empty())
			continue;

		//减少Tact时间
		cv::resize(mtRef[nROIInx], mtRef[nROIInx], Size(), 0.25, 0.25);

		cv::matchTemplate(mtOrg, mtRef[nROIInx], mtMatchRst, CV_TM_CCORR_NORMED);
		cv::minMaxLoc(mtMatchRst, NULL, &dMatchRate, NULL, &ptMatchResize);

		ptMatch = Point(ptMatchResize.x * 4, ptMatchResize.y * 4);

		if (dMatchRate < 0.6)
			return E_ERROR_CODE_FALSE;

		vector<Point>().swap(pt);

		for (int n = 0; n < tRoundSet[nROIInx].nContourCount; n++)
		{
			//将ROI区域内的坐标更改为整个画面的坐标
			Point ptTlans = Point(tRoundSet[nROIInx].ptContours[n].x + ptMatch.x,
				tRoundSet[nROIInx].ptContours[n].y + ptMatch.y);
			if (ptTlans.x < 0 || ptTlans.x >= matSrcBuf.cols || ptTlans.y < 0 || ptTlans.y >= matSrcBuf.rows)
				return E_ERROR_CODE_ALIGN_IMAGE_OVER;

			pt.push_back(ptTlans);
			cv::circle(mtContour, ptTlans, 1, Scalar(255));
		}

		ptContours.push_back(pt);
	}

	for (int nContourInx = 0; nContourInx < ptContours.size(); nContourInx++)
	{
		int nNextContourInx = nContourInx + 1;
		if (nNextContourInx >= ptContours.size())
			nNextContourInx = 0;
		cv::line(mtContour, ptContours[nContourInx][ptContours[nContourInx].size() - 1], ptContours[nNextContourInx][0], Scalar(255));
	}

	cv::findContours(mtContour, ptActive, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::DoFillOutAreaDust(cv::Mat& matSrcBuf, cv::Mat& MatDrawBuffer, cv::Point ptResCorner[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, double dAngle, cv::Rect* rcCHoleROI, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA], double* dPara, int nAlgImg, int nRatio, wchar_t* strID)
{
	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Start."));

	//如果没有缓冲区。
	if (matSrcBuf.empty())							return E_ERROR_CODE_EMPTY_BUFFER;

	//如果不是Dust模式
	if (nAlgImg != E_IMAGE_CLASSIFY_AVI_DUST)		return E_ERROR_CODE_TRUE;
	if (nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)		return E_ERROR_CODE_TRUE;//跳过背光画面 hjf
	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

		//使用Round Cell&CHole有/无
	bool	bRoundUse = (dPara[E_PARA_ROUND_USE] > 0) ? true : false;
	bool	bCHoleUse = (dPara[E_PARA_CHOLE_USE] > 0) ? true : false;
	bool	bCHolePointUse = (dPara[E_PARA_CHOLE_POINT_USE] > 0) ? true : false;

	// CHole Threshold value
	double dThresValue = dPara[E_PARA_ALIGN_THRESHOLD];

	//Round Cell&CHole里面有多少Pixel...(只有曲线部分...)
	int		nRoundIn = (int)(dPara[E_PARA_ROUND_IN]);
	int		nCHoleIn = (int)dPara[E_PARA_CHOLE_IN];
	//Label填充
	int nRotate_Use = dPara[E_PARA_AVI_Rotate_Image];//yuxuefei
	// Find CHole Offset
	int		nFindCHoleOffset = (int)dPara[E_PARA_CHOLE_FIND_OFFSET];

	//填充轮廓平均值时,设置最小平均GV
	int		nRoundMinGV = (int)dPara[E_PARA_ROUND_DUST_MIN_GV];

	//错误代码
	long nErrorCode = E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////
		//查找4个拐角转角
	//////////////////////////////////////////////////////////////////////////
	cv::Point ptCornerAlign[E_CORNER_END];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////
		//稍微缩小到区域内部...
	//////////////////////////////////////////////////////////////////////////

		//稍微缩小到区域内部...(区域)
	int nInPixel = 3;

	ptResCorner[E_CORNER_LEFT_TOP].x += nInPixel;
	ptResCorner[E_CORNER_LEFT_TOP].y += nInPixel;
	ptResCorner[E_CORNER_RIGHT_TOP].x -= nInPixel;
	ptResCorner[E_CORNER_RIGHT_TOP].y += nInPixel;
	ptResCorner[E_CORNER_LEFT_BOTTOM].x += nInPixel;
	ptResCorner[E_CORNER_LEFT_BOTTOM].y -= nInPixel;
	ptResCorner[E_CORNER_RIGHT_BOTTOM].x -= nInPixel;
	ptResCorner[E_CORNER_RIGHT_BOTTOM].y -= nInPixel;

	//////////////////////////////////////////////////////////////////////////
		//检查Rect范围

		//设置大范围(两侧必须有200 Pixel空背景)
	int offset = 70;

	CRect rectROI = new CRect(
		min(ptResCorner[E_CORNER_LEFT_TOP].x, ptResCorner[E_CORNER_LEFT_BOTTOM].x) - offset,
		min(ptResCorner[E_CORNER_LEFT_TOP].y, ptResCorner[E_CORNER_RIGHT_TOP].y) - offset,
		max(ptResCorner[E_CORNER_RIGHT_TOP].x, ptResCorner[E_CORNER_RIGHT_BOTTOM].x) + offset,
		max(ptResCorner[E_CORNER_LEFT_BOTTOM].y, ptResCorner[E_CORNER_RIGHT_BOTTOM].y) + offset);

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//////////////////////////////////////////////////////////////////////////
		//曲线处理

		//转换原始画面8位
	cv::Mat matSrc_8bit = cMem->GetMat(matSrcBuf.size(), CV_8UC1, false); // 使用X

	if (matSrcBuf.type() == CV_8UC1)
		matSrcBuf.copyTo(matSrc_8bit);
	else
		matSrcBuf.convertTo(matSrc_8bit, CV_8UC1, 1. / 16.);

	cv::Mat matCHoleBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1);
	//是否使用CHole
	if (bCHoleUse || bCHolePointUse)
	{
		// Image Save 0 On / -1 Off
		//int nImgSave = -1;
		int nImgSave = (int)dPara[E_PARA_CHOLE_TEXT];

		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("CHole FillArea Start."));
		//////////////////////////////////////////////////////////////////////////
				//倾斜校正时,获取转角坐标
				//注册的曲线&ROI区域基于转角坐标
		//////////////////////////////////////////////////////////////////////////

				//计算旋转坐标时,使用
		double	dTheta = -dAngle;
		dTheta *= PI;
		dTheta /= 180.0;
		double	dSin = sin(dTheta);
		double	dCos = cos(dTheta);
		double	dSin_ = sin(-dTheta);
		double	dCos_ = cos(-dTheta);
		int		nCx = matSrcBuf.cols / 2;
		int		nCy = matSrcBuf.rows / 2;

		for (int i = 0; i < E_CORNER_END; i++)
		{
			//旋转时计算预测坐标
			ptCornerAlign[i].x = (int)(dCos * (ptResCorner[i].x - nCx) - dSin * (ptResCorner[i].y - nCy) + nCx);
			ptCornerAlign[i].y = (int)(dSin * (ptResCorner[i].x - nCx) + dCos * (ptResCorner[i].y - nCy) + nCy);
		}

		//////////////////////////////////////////////////////////////////////////
				//注册的曲线&ROI区域->根据当前Grab进行坐标校正和定位
		//////////////////////////////////////////////////////////////////////////

				//重新计算
		dTheta = dAngle;
		dTheta *= PI;
		dTheta /= 180.0;
		dSin = sin(dTheta);
		dCos = cos(dTheta);
		dSin_ = sin(-dTheta);
		dCos_ = cos(-dTheta);

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			//如果没有曲线数,则判断区域未注册/下一步...
			if (tCHoleSet[i].nContourCount <= 0)	continue;

			cv::Point2i ptTemp;

			//点灯区域外的转角点数
			int nOutsideCnt = 0;
			for (int j = 0; j < E_CORNER_END; j++)
			{
				if (!tCHoleSet[i].nCornerInside[j])
					nOutsideCnt++;
			}

			const int nCount = (int)tCHoleSet[i].nContourCount;
			cv::Point* ptPoly = new cv::Point[nCount];
			memset(ptPoly, 0, sizeof(cv::Point) * nCount);

			//距离Align转角点最近的转角点
			int nCornerAlign = tCHoleSet[i].nCornerMinLength;

			int nMaxX = 0, nMinX = matSrc_8bit.cols;
			int nMaxY = 0, nMinY = matSrc_8bit.rows;

			//如果存在曲线数量
			for (int j = 0; j < tCHoleSet[i].nContourCount; j++)
			{
				//从虚线坐标到曲线设置区域的最短距离...校正
				int XX = tCHoleSet[i].ptContours[j].x + ptCornerAlign[nCornerAlign].x;
				int YY = tCHoleSet[i].ptContours[j].y + ptCornerAlign[nCornerAlign].y;

				//旋转时,计算预测坐标
				ptTemp.x = (int)(dCos * (XX - nCx) - dSin * (YY - nCy) + nCx);
				ptTemp.y = (int)(dSin * (XX - nCx) + dCos * (YY - nCy) + nCy);

				//插入校正坐标
				ptPoly[j].x = ptTemp.x;
				ptPoly[j].y = ptTemp.y;

				matCHoleBuf.at<uchar>(ptPoly[j].y, ptPoly[j].x) = (uchar)255;

				if (nMaxX < ptPoly[j].x) nMaxX = ptPoly[j].x;
				if (nMinX > ptPoly[j].x) nMinX = ptPoly[j].x;
				if (nMaxY < ptPoly[j].y) nMaxY = ptPoly[j].y;
				if (nMinY > ptPoly[j].y) nMinY = ptPoly[j].y;
			}

			int nDustOffset = 20;

			rcCHoleROI[i].x = nMinX - nDustOffset;
			rcCHoleROI[i].y = nMinY - nDustOffset;
			rcCHoleROI[i].width = (nMaxX - nMinX + 1) + nDustOffset * 2;
			rcCHoleROI[i].height = (nMaxY - nMinY + 1) + nDustOffset * 2;
		}
		if (nImgSave >= 1)
		{
			CString strTemp;
			strTemp.Format(_T("E:\\IMTC\\CHole\\%s_%02d_%02d_Dust_CHole_src.bmp"), strID, nAlgImg, nImgSave++);
			cv::imwrite((cv::String)(CStringA)strTemp, matSrcBuf);
		}
	}

	//掩码
	cv::Mat matTempBuf = cMem->GetMat(matSrcBuf.size(), CV_8UC1);

	int npt[] = { E_CORNER_END };

	//禁用曲线时(如果是Rect)
	if (!bRoundUse)
	{
		const cv::Point* ppt[1] = { ptResCorner };

		//点亮区域掩码
		cv::fillPoly(matTempBuf, ppt, npt, 1, cv::Scalar(255, 255, 255));
	}

	//曲线部分处理
	else
	{
		//分配Temp内存
		cv::Mat matRndBuf = cMem->GetMat(matTempBuf.size(), matTempBuf.type());

		//////////////////////////////////////////////////////////////////////////
				//倾斜校正时,获取转角坐标
				//注册的曲线&ROI区域基于转角坐标
		//////////////////////////////////////////////////////////////////////////

				//计算旋转坐标时,使用
		double	dTheta = -dAngle;
		dTheta *= PI;
		dTheta /= 180.0;
		double	dSin = sin(dTheta);
		double	dCos = cos(dTheta);
		double	dSin_ = sin(-dTheta);
		double	dCos_ = cos(-dTheta);
		int		nCx = matSrcBuf.cols / 2;
		int		nCy = matSrcBuf.rows / 2;

		for (int i = 0; i < E_CORNER_END; i++)
		{
			//旋转时计算预测坐标
			ptCornerAlign[i].x = (int)(dCos * (ptResCorner[i].x - nCx) - dSin * (ptResCorner[i].y - nCy) + nCx);
			ptCornerAlign[i].y = (int)(dSin * (ptResCorner[i].x - nCx) + dCos * (ptResCorner[i].y - nCy) + nCy);
		}

		//////////////////////////////////////////////////////////////////////////
				//注册的曲线&ROI区域->根据当前Grab进行坐标校正和定位
		//////////////////////////////////////////////////////////////////////////

				//重新计算
		dTheta = dAngle;
		dTheta *= PI;
		dTheta /= 180.0;
		dSin = sin(dTheta);
		dCos = cos(dTheta);
		dSin_ = sin(-dTheta);
		dCos_ = cos(-dTheta);

		cv::Point	ptSE[2], ptTempLine[2];
		int			nSE;

		for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
		{
			//如果没有曲线数,则判断区域未注册/下一步...
			if (tRoundSet[i].nContourCount <= 0)	continue;

			cv::Point2i ptTemp;

			//点灯区域外的转角点数
			int nOutsideCnt = 0;
			for (int j = 0; j < E_CORNER_END; j++)
			{
				if (!tRoundSet[i].nCornerInside[j])
					nOutsideCnt++;
			}

			//如果一个都没有,则设置错误
// 			if( nOutsideCnt == 0 )
// 				AfxMessageBox(_T("Set Corner Err !!!"));

			const int nCount = (int)tRoundSet[i].nContourCount;
			cv::Point* ptPoly = new cv::Point[nCount];
			memset(ptPoly, 0, sizeof(cv::Point) * nCount);

			//距离Align转角点最近的转角点
			int nCornerAlign = tRoundSet[i].nCornerMinLength;

			//如果存在曲线数量
			for (int j = 0; j < tRoundSet[i].nContourCount; j++)
			{
				//从虚线坐标到曲线设置区域的最短距离...校正
				int XX = tRoundSet[i].ptContours[j].x + ptCornerAlign[nCornerAlign].x;
				int YY = tRoundSet[i].ptContours[j].y + ptCornerAlign[nCornerAlign].y;

				//旋转时,计算预测坐标
				ptTemp.x = (int)(dCos * (XX - nCx) - dSin * (YY - nCy) + nCx);
				ptTemp.y = (int)(dSin * (XX - nCx) + dCos * (YY - nCy) + nCy);

				//插入校正坐标
				ptPoly[j].x = ptTemp.x;
				ptPoly[j].y = ptTemp.y;
			}

			//曲线部分,轻轻地放在点灯区域内(取决于情况)
			cv::Point ptRnd = calcRoundIn(tRoundSet, i, nRoundIn);

			//修改位置
			int j = 0;
			for (; j < tRoundSet[i].nContourCount; j++)
			{
				ptPoly[j].x += ptRnd.x;
				ptPoly[j].y += ptRnd.y;

				//异常处理
				if (ptPoly[j].x < 0)					return E_ERROR_CODE_ALIGN_IMAGE_OVER;
				if (ptPoly[j].y < 0)					return E_ERROR_CODE_ALIGN_IMAGE_OVER;
				if (ptPoly[j].x >= matRndBuf.cols)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;
				if (ptPoly[j].y >= matRndBuf.rows)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;

				matRndBuf.at<uchar>(ptPoly[j].y, ptPoly[j].x) = (uchar)255;
			}

			//////////////////////////////////////////////////////////////////////////
						//非曲线直线连接
			//////////////////////////////////////////////////////////////////////////

						//结束点
			ptTempLine[0] = ptPoly[0];
			ptTempLine[1] = ptPoly[nCount - 1];

			delete ptPoly;
			ptPoly = NULL;

			//注册结束点
			if (i == 0)
			{
				ptSE[0] = ptTempLine[0];
				ptSE[1] = ptTempLine[1];
			}
			//连接端点
			else
			{
				//非曲线直线连接
				calcLineConnect(matRndBuf, ptSE, ptTempLine, nSE, i);
			}
		}
		//连接最后一个端点
		cv::line(matRndBuf, ptSE[0], ptSE[1], cv::Scalar(255, 255, 255));

		//17.11.12-连接断点
		cv::dilate(matRndBuf, matTempBuf, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

		//向轮廓内填充(4方向？)
		cv::floodFill(matTempBuf, cv::Point(nWidth / 2, nHeight / 2), cv::Scalar(255, 255, 255));

		matRndBuf.release();
	}
	//17.07.17曲线部分处理
//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Edge Corner Find."));

	cv::Mat matEdgeArea = cMem->GetMat(matTempBuf.size(), CV_8UC1, false);

	// Label PolMark 添加 2024.07
	InspectLabelPol inpectLabelPol;
	int nLabel_Flag = dPara[E_PARA_AVI_Label_Flag];   //yuxuefei add
	int nPolNum_Flag = dPara[E_PARA_AVI_PolNum_Flag];
	int nPolSign_Flag = dPara[E_PARA_AVI_PolSign_Flag];
	int bPolSaveTemplate = dPara[E_PARA_AVI_Pol_Save_Template];
	if (!bPolSaveTemplate && (nLabel_Flag == 1 || nPolNum_Flag == 1 || nPolSign_Flag == 1)) {
		// 等待白画面定位PolMark结束
		while (!labelMarkInfo.bFindEnd) {
			Sleep(20);
		}

		inpectLabelPol.DoFillLabelMark(matSrcBuf, labelMarkParams, labelMarkInfo);
	}
	//留下一点点亮区域轮廓(点亮区域平均时使用)
	nErrorCode = FindEdgeArea(matTempBuf, matEdgeArea, offset, cMem);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//cv::imwrite(("E:\\IMTC\\0.EdgeArea.bmp"), matEdgeArea);

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Edge Area Find."));

	//平均缓冲内存分配
	cv::Mat matMeanBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	matSrcBuf.copyTo(matMeanBuf);

	//Offset(点亮区域平均大小)
	//nSeg x nSeg区域内必须存在点亮区域
	//6.18"需要增加大小(300->500)
	//左右上下offset(目前为70)
	//x,y分割使用
	int nSegX = 3;
	int nSegY = offset + offset / 2;

	//横向平均填充
	nErrorCode = FillAreaMeanX(matMeanBuf, matEdgeArea, rectROI, nSegX, nSegY, nRoundMinGV);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//cv::imwrite(("E:\\IMTC\\1.Mean.bmp"), matMeanBuf);
	//cv::imwrite(("E:\\IMTC\\1.EdgeArea.bmp"), matEdgeArea);

		//Offset(点亮区域平均大小)
		//nSeg x nSeg区域内必须存在点亮区域
		//6.18"需要增加大小(300->500)
		//左右上下offset(目前为70)
		//x,y分割使用
	nSegX = 800;
	nSegY = 3;

	//垂直平均填充
	nErrorCode = FillAreaMeanY(matMeanBuf, matEdgeArea, rectROI, nSegX, nSegY, nRoundMinGV);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	//cv::imwrite(("E:\\IMTC\\2.Mean.bmp"), matMeanBuf);
	//cv::imwrite(("E:\\IMTC\\2.EdgeArea.bmp"), matEdgeArea);
	//cv::imwrite(("E:\\IMTC\\2.Temp.bmp"), matTempBuf);

		//只保留已点亮的部分
	nErrorCode = FillMerge(matSrcBuf, matMeanBuf, matTempBuf, nAlgImg, cMem);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("Edge Fill Average."));

	//cv::imwrite(("E:\\IMTC\\3.Mean.bmp"), matMeanBuf);
	//cv::imwrite(("E:\\IMTC\\3.EdgeArea.bmp"), matEdgeArea);

		//禁用内存
	matMeanBuf.release();
	matTempBuf.release();

	writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("End."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//画面旋转
long CInspectAlign::DoRotateImage(cv::Mat matSrcBuffer, cv::Mat& matDstBuffer, double dAngle)
{
	//如果没有缓冲区。错误代码
	if (matSrcBuffer.empty())
		return E_ERROR_CODE_EMPTY_BUFFER;

	cv::Mat matTempBuffer = cv::Mat::zeros(matSrcBuffer.size(), CV_8UC1);

	//创建旋转矩阵
	cv::Mat matRotation = cv::getRotationMatrix2D(Point(matSrcBuffer.cols / 2, matSrcBuffer.rows / 2), dAngle, 1.0);

	//画面旋转
	cv::warpAffine(matSrcBuffer, matTempBuffer, matRotation, matTempBuffer.size());

	//取消分配后,放入结果
	if (!matDstBuffer.empty())
		matDstBuffer.release();

	matDstBuffer = matTempBuffer.clone();

	//取消分配内存
	matTempBuffer.release();
	matRotation.release();

	return E_ERROR_CODE_TRUE;
}

//旋转坐标
long CInspectAlign::DoRotatePoint(cv::Point ptSrcPoint, cv::Point& ptDstPoint, cv::Point ptCenter, double dAngle)
{
	//OpenCV画面旋转反向(设置符号"-")
	double dTheta = -dAngle * PI / 180.;

	double dSin = sin(dTheta);
	double dCos = cos(dTheta);

	//旋转时计算预测坐标
	int X = (int)(dCos * (ptSrcPoint.x - ptCenter.x) - dSin * (ptSrcPoint.y - ptCenter.y) + ptCenter.x);
	int Y = (int)(dSin * (ptSrcPoint.x - ptCenter.x) + dCos * (ptSrcPoint.y - ptCenter.y) + ptCenter.y);

	//放入结果
	ptDstPoint.x = X;
	ptDstPoint.y = Y;

	return E_ERROR_CODE_TRUE;
}

//AVI AD检查/dResult:当前Cell匹配率
long CInspectAlign::DoFindDefectAD(cv::Mat matSrcBuf, double* dPara, double* dResult, int nRatio)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果参数为NULL。
	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	cv::Mat matGrayBuf;

	//缓冲区分配和初始化
	matGrayBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

	//颜色(SVI)
	if (matSrcBuf.channels() != 1)
	{
		//cv::cvtColor(matSrcBuf, matGrayBuf, COLOR_RGB2GRAY);
				//	AD和Align相关序列上的tac time目前可能不是一个大问题,所以直接使用这个结构。
				//	如果进一步细化AD检查,将独立于SVI和AVI函数。

		cv::Mat matLab;
		cv::cvtColor(matSrcBuf, matLab, CV_BGR2Lab);
		vector<Mat> vLab(3);
		cv::split(matLab, vLab);
		vLab.at(0).convertTo(matGrayBuf, CV_8UC1, 2.55);

		if (!matLab.empty())			matLab.release();
		if (!vLab.at(0).empty())		vLab.at(0).release();
		if (!vLab.at(1).empty())		vLab.at(1).release();
		if (!vLab.at(2).empty())		vLab.at(2).release();
		vector<Mat>().swap(vLab);
	}
	//黑白(AVI,APP)
	else
		matSrcBuf.copyTo(matGrayBuf);
	//matGrayBuf = matSrcBuf.clone();

//////////////////////////////////////////////////////////////////////////
	//17.08.07-如果根本没有点亮,请退出
//////////////////////////////////////////////////////////////////////////

	//获取stdDev
	cv::Scalar m, s;
	cv::meanStdDev(matGrayBuf, m, s);

	//1:平均亮度(Mean GV)
	dResult[1] = double(m[0]);

	//2:标准偏差(Std)
	dResult[2] = double(s[0]);

	//可能会有噪音......(好像也会出现0.xxx...)
	//如果小于整个画面的平均亮度1,则判断为未点亮
	if (dResult[1] < 1.0)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_OFF;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	double	dRate = dPara[E_PARA_AD_RATE];
	double	dMinGV = dPara[E_PARA_AD_MIN_GV];
	double	dMaxGV = dPara[E_PARA_AD_MAX_GV];
	double	dMaxStd = dPara[E_PARA_AD_MAX_STD];
	int		nLineYCount = (int)dPara[E_PARA_AD_Y_LINE];

	//检查Block
	int		nBlockGV_X = (int)(dPara[E_PARA_AD_BLOCK_X_GV]);
	float	fBlockArea_X = (float)(dPara[E_PARA_AD_BLOCK_X_AREA]);
	int		nBlockGV_Y = (int)(dPara[E_PARA_AD_BLOCK_Y_GV]);
	float	fBlockArea_Y = (float)(dPara[E_PARA_AD_BLOCK_Y_AREA]);

	double	dZoom = dPara[E_PARA_AD_ZOOM];
	int		nModelX = (int)dPara[E_PARA_CELL_SIZE_X] * nRatio;
	int		nModelY = (int)dPara[E_PARA_CELL_SIZE_Y] * nRatio;

	//参数异常
	if (dZoom < 1)	return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//禁用时退出
	if (dRate <= 0)
	{
		dResult[0] = 100.0;
		dResult[1] = 0;
		dResult[2] = 0;

		return E_ERROR_CODE_TRUE;
	}

	//缩小的图像大小
	int nWidth = matGrayBuf.cols / dZoom;
	int nHeight = matGrayBuf.rows / dZoom;

	//缩小原始大小
	cv::Mat matResizeBuf, matResBuf, matModelBuf;

	//缓冲区分配和初始化
	matResizeBuf = cMem->GetMat(nHeight, nWidth, matSrcBuf.type(), false);
	matResBuf = cMem->GetMat(nHeight, nWidth, CV_32FC1, false);
	matModelBuf = cMem->GetMat((int)(nModelY / dZoom), (int)(nModelX / dZoom), CV_8UC1, false);

	cv::resize(matGrayBuf, matResizeBuf, cv::Size(nWidth, nHeight));

	//创建恒定亮度模型(Gray Value:50)
	matModelBuf.setTo(50);

	if (matResizeBuf.type() == CV_8UC1)
	{
		//匹配
		cv::matchTemplate(matResizeBuf, matModelBuf, matResBuf, CV_TM_CCORR_NORMED);
	}
	else
	{
		//12bit-->8bit转换
		cv::Mat matTemp_8bit = cMem->GetMat(nHeight, nWidth, CV_8UC1, false);
		matResizeBuf.convertTo(matTemp_8bit, CV_8UC1, 1. / 16.);

		//匹配
		cv::matchTemplate(matTemp_8bit, matModelBuf, matResBuf, CV_TM_CCORR_NORMED);

		matTemp_8bit.release();
	}

	//匹配率
//cv::minMaxLoc(matResBuf, NULL, &dResult);

	int xx = 0, yy = 0;
	dResult[0] = 0.0;
	for (int y = 0; y < matResBuf.rows; y++)
	{
		float* ptr = (float*)matResBuf.ptr(y);
		for (int x = 0; x < matResBuf.cols; x++, ptr++)
		{
			if (dResult[0] < *ptr)
			{
				dResult[0] = *ptr;
				xx = x;
				yy = y;
			}
		}
	}

	//0:匹配率(Rate)
	//更改为百分比
	dResult[0] *= 100.0;

	//取消分配内存
	matResizeBuf.release();
	matResBuf.release();
	matModelBuf.release();
	matGrayBuf.release();

	//////////////////////////////////////////////////////////////////////////
		//如果找不到点亮区域
		//如果不是要检查的亮度
	//////////////////////////////////////////////////////////////////////////

		//如果最大值匹配率低于设置的值,则显示异常
	if (dResult[0] < dRate)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//检查匹配的部分统计信息
	cv::Rect rect(xx * dZoom, yy * dZoom, nModelX, nModelY);

	//如果扫描区域超出画面大小
	if (rect.x < 0 ||
		rect.y < 0 ||
		rect.x + rect.width >= matSrcBuf.cols ||
		rect.y + rect.height >= matSrcBuf.rows)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	if (rect.width <= 1)									return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rect.height <= 1)									return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//剪切匹配的局部源
	matGrayBuf = matSrcBuf(rect);

	//获取stdDev
	cv::meanStdDev(matGrayBuf, m, s);

	//1:平均亮度(Mean GV)
	dResult[1] = double(m[0]);

	//2:标准偏差(Std)
	dResult[2] = double(s[0]);

	//偏差较大时显示异常
	if (dResult[2] > dMaxStd)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较亮时,显示或更高
	if (dResult[1] > dMaxGV)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较暗时,显示异常
	if (dResult[1] < dMinGV)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//////////////////////////////////////////////////////////////////////////
		//18.11.05-如果由于上下文不良导致竖线过多,则上下文异常
	//////////////////////////////////////////////////////////////////////////

		//剪切匹配的局部源
	//cv::Mat matROIBuf = matGrayBuf(rect);

		//消除噪音
	int nGauSize = 31;
	double dGauSig = 2;
	cv::Mat matGauBuf = cMem->GetMat(matGrayBuf.size(), matGrayBuf.type(), false);
	cv::GaussianBlur(matGrayBuf, matGauBuf, cv::Size(nGauSize, nGauSize), dGauSig, dGauSig);
	//cv::imwrite("E:\\IMTC\\01.Gau.bmp", matGauBuf);

	// Edge
	cv::Mat matEdgeBuf = cMem->GetMat(matGauBuf.size(), matGauBuf.type(), false);
	cv::Sobel(matGauBuf, matEdgeBuf, CV_8UC1, 1, 0);
	//cv::imwrite("E:\\IMTC\\02.Edge.bmp", matEdgeBuf);

		//基准值
	float fAvg = (float)(cv::mean(matEdgeBuf)[0] * 2.75f);

	//纵向累计平均值
	matGauBuf.setTo(0);
	int nCount = 0;
	for (int x = 0; x < matEdgeBuf.cols; x++)
	{
		float fTemp = (float)cv::sum(matEdgeBuf.col(x))[0] / (float)matEdgeBuf.rows;

		if (fTemp > fAvg)
		{
			cv::line(matGauBuf, cv::Point(x, 0), cv::Point(x, matGauBuf.rows), cv::Scalar(255), 1);
			nCount++;
		}
	}
	//cv::imwrite("E:\\IMTC\\03.Line.bmp", matGauBuf);

	matEdgeBuf.release();
	matGauBuf.release();

	//如果竖线太多,则判断为上下文异常
	if (nLineYCount > 0 &&
		nCount > nLineYCount)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//////////////////////////////////////////////////////////////////////////
		//如果区域被分割成两部分,亮度不同
	//////////////////////////////////////////////////////////////////////////

	matModelBuf = cMem->GetMat(matGrayBuf.size(), matGrayBuf.type(), false);
	cv::blur(matGrayBuf, matModelBuf, cv::Size(5, 5));

	matResizeBuf = cMem->GetMat(cv::Size(matModelBuf.cols / 10, matModelBuf.rows / 10), matModelBuf.type(), false);
	cv::resize(matModelBuf, matResizeBuf, matResizeBuf.size());

	//初始化数组
	int* nCntX = new int[matResizeBuf.cols];
	int* nCntY = new int[matResizeBuf.rows];
	memset(nCntX, 0, sizeof(int) * matResizeBuf.cols);
	memset(nCntY, 0, sizeof(int) * matResizeBuf.rows);

	//横向累计
	int nAvgY = 0;
	int nAvgY_Up = 0;
	int nAvgY_Down = 0;
	int nCntY_Up = 0;
	int nCntY_Down = 0;

	//纵向累计
	int nAvgX = 0;
	int nAvgX_Up = 0;
	int nAvgX_Down = 0;
	int nCntX_Up = 0;
	int nCntX_Down = 0;

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
			nAvgY = 0;
			for (int y = 0; y < matResizeBuf.rows; y++)
			{
				nCntY[y] = (int)(cv::sum(matResizeBuf.row(y))[0] / matResizeBuf.cols);
				nAvgY += nCntY[y];
			}
			nAvgY /= matResizeBuf.rows;
		}
		break;
		case 1:
		{
			nAvgX = 0;
			for (int x = 0; x < matResizeBuf.cols; x++)
			{
				nCntX[x] = (int)(cv::sum(matResizeBuf.col(x))[0] / matResizeBuf.rows);
				nAvgX += nCntX[x];
			}
			nAvgX /= matResizeBuf.cols;
		}
		break;
		}
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
			nAvgY_Up = 0;
			nAvgY_Down = 0;
			nCntY_Up = 0;
			nCntY_Down = 0;

			for (int y = 0; y < matResizeBuf.rows; y++)
			{
				if (nCntY[y] >= nAvgY)
				{
					nAvgY_Up += nCntY[y];
					nCntY_Up++;
				}
				else
				{
					nAvgY_Down += nCntY[y];
					nCntY_Down++;
				}
			}
			if (nCntY_Up > 0)	nAvgY_Up /= nCntY_Up;
			if (nCntY_Down > 0)	nAvgY_Down /= nCntY_Down;
		}
		break;
		case 1:
		{
			nAvgX_Up = 0;
			nAvgX_Down = 0;
			nCntX_Up = 0;
			nCntX_Down = 0;
			for (int x = 0; x < matResizeBuf.cols; x++)
			{
				if (nCntX[x] >= nAvgX)
				{
					nAvgX_Up += nCntX[x];
					nCntX_Up++;
				}
				else
				{
					nAvgX_Down += nCntX[x];
					nCntX_Down++;
				}
			}
			if (nCntX_Up > 0)	nAvgX_Up /= nCntX_Up;
			if (nCntX_Down > 0)	nAvgX_Down /= nCntX_Down;
		}
		break;
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////

		//取消分配内存
	delete[] nCntX;		nCntX = NULL;
	delete[] nCntY;		nCntY = NULL;
	matModelBuf.release();
	matGrayBuf.release();
	matResizeBuf.release();

	//GV差异较大时
	if (nAvgX_Up - nAvgX_Down > nBlockGV_X)
	{
		int nTemp = (nCntX_Up > nCntX_Down) ? nCntX_Down : nCntX_Up;

		//面积百分比超过20%
		if (nTemp / (float)(nCntX_Up + nCntX_Down) > fBlockArea_X)
		{
			dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
			return E_ERROR_CODE_ALIGN_DISPLAY;
		}
	}

	//GV差异较大时
	if (nAvgY_Up - nAvgY_Down > nBlockGV_Y)
	{
		int nTemp = (nCntY_Up > nCntY_Down) ? nCntY_Down : nCntY_Up;

		//面积百分比超过20%
		if (nTemp / (float)(nCntY_Up + nCntY_Down) > fBlockArea_Y)
		{
			dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
			return E_ERROR_CODE_ALIGN_DISPLAY;
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//SVI AD检查/dResult:当前Cell匹配率
long CInspectAlign::DoFindDefectAD_SVI(cv::Mat matSrcBuf, double* dPara, double* dResult, int nCameraNum, int nRatio)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果参数为NULL。
	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//////////////////////////////////////////////////////////////////////////
		//如果根本没有点灯,请退出
	//////////////////////////////////////////////////////////////////////////

	// Color -> Gray
	cv::Mat matGrayBuf;
	cv::cvtColor(matSrcBuf, matGrayBuf, COLOR_RGB2GRAY);

	//获取stdDev
	cv::Scalar m, s;
	cv::meanStdDev(matGrayBuf, m, s);

	//1:平均亮度(Mean GV)
	dResult[1] = double(m[0]);

	//2:标准偏差(Std)
	dResult[2] = double(s[0]);

	//可能会有噪音......(好像也会出现0.xxx...)
	//如果小于整个画面的平均亮度1,则判断为未点亮
	if (dResult[1] < 1.0)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_OFF;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	double	dRate = dPara[E_PARA_SVI_AD_RATE];
	double	dMinGV = dPara[E_PARA_SVI_AD_MIN_GV];
	double	dMaxGV = dPara[E_PARA_SVI_AD_MAX_GV];
	double	dMaxStd = dPara[E_PARA_SVI_AD_MAX_STD];

	double	dZoom = dPara[E_PARA_SVI_AD_ZOOM];
	int		nModelX = (int)dPara[E_PARA_SVI_CELL_COAX_SIZE_X] * nRatio;
	int		nModelY = (int)dPara[E_PARA_SVI_CELL_COAX_SIZE_Y] * nRatio;

	//Side相机
	if (nCameraNum == 1)
	{
		nModelX = (int)dPara[E_PARA_SVI_CELL_SIDE_SIZE_X] * nRatio;
		nModelY = (int)dPara[E_PARA_SVI_CELL_SIDE_SIZE_Y] * nRatio;
	}

	//参数异常
	if (dZoom < 1)	return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//禁用时退出
	if (dRate <= 0)
	{
		dResult[0] = 100.0;
		dResult[1] = 0;
		dResult[2] = 0;
		return E_ERROR_CODE_TRUE;
	}

	//////////////////////////////////////////////////////////////////////////
		//查找Cell位置
	//////////////////////////////////////////////////////////////////////////

		//缩小的图像大小
	int nWidth = matGrayBuf.cols / dZoom;
	int nHeight = matGrayBuf.rows / dZoom;

	//缩小原始大小
	cv::Mat matResizeBuf;
	cv::resize(matGrayBuf, matResizeBuf, cv::Size(nWidth, nHeight));

	//创建恒定亮度模型(Gray Value:50)
	cv::Mat matModelBuf = cv::Mat::zeros((int)(nModelY / dZoom), (int)(nModelX / dZoom), CV_8UC1);
	matModelBuf.setTo(50);

	//匹配
	cv::Mat matResBuf = cv::Mat::zeros(nHeight, nWidth, CV_32FC1);
	cv::matchTemplate(matResizeBuf, matModelBuf, matResBuf, CV_TM_CCORR_NORMED);

	//匹配率
//cv::minMaxLoc(matResBuf, NULL, &dResult);

	int xx = 0, yy = 0;
	dResult[0] = 0.0;
	for (int y = 0; y < matResBuf.rows; y++)
	{
		float* ptr = (float*)matResBuf.ptr(y);
		for (int x = 0; x < matResBuf.cols; x++, ptr++)
		{
			if (dResult[0] < *ptr)
			{
				dResult[0] = *ptr;
				xx = x;
				yy = y;
			}
		}
	}

	//0:匹配率(Rate)
	//更改为百分比
	dResult[0] *= 100.0;

	//取消分配内存
	matResizeBuf.release();
	matResBuf.release();
	matModelBuf.release();
	matGrayBuf.release();

	//如果最大值匹配率低于设置的值,则显示异常
	if (dResult[0] < dRate)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//检查匹配的部分统计信息
	cv::Rect rect(xx * dZoom, yy * dZoom, nModelX, nModelY);

	//如果扫描区域超出画面大小
	if (rect.x < 0 ||
		rect.y < 0 ||
		rect.x + rect.width >= matSrcBuf.cols ||
		rect.y + rect.height >= matSrcBuf.rows)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	if (rect.width <= 1)									return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rect.height <= 1)									return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//剪切匹配的局部源
	matGrayBuf = matSrcBuf(rect);

	//获取stdDev
	cv::meanStdDev(matGrayBuf, m, s);

	//1:平均亮度(Mean GV)
	dResult[1] = double(m[0]);

	//2:标准偏差(Std)
	dResult[2] = double(s[0]);

	matGrayBuf.release();

	//////////////////////////////////////////////////////////////////////////
		//AD检查
	//////////////////////////////////////////////////////////////////////////

		//偏差较大时显示异常
	if (dResult[2] > dMaxStd)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_ABNORMAL;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较亮时,显示或更高
	if (dResult[1] > dMaxGV)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较暗时,显示异常
	if (dResult[1] < dMinGV)
	{
		dResult[3] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::DoFindDefectAD_APP(cv::Mat MatOrgImage, double* dAlgPara, double* dResult, int nRatio)
{
	long nErrorCode;

	if (MatOrgImage.empty())
		nErrorCode = E_ERROR_CODE_EMPTY_BUFFER;

	cv::Scalar m, s;
	cv::meanStdDev(MatOrgImage, m, s);

	if (m[0] == 0)
		return E_ERROR_CODE_EMPTY_BUFFER;

	int nBlurSize = (int)dAlgPara[E_PARA_APP_AD_BLUR_SIZE];		//21
	int nThreshold = (int)dAlgPara[E_PARA_APP_AD_THRESHOLD];		//20
	double dLimitSize = dAlgPara[E_PARA_APP_AD_LIMIT_AREA];			//10000
	double dCompare_Theta = (double)dAlgPara[E_PARA_APP_PAD_EDGE_THETA];
	int	nADGV = (int)dAlgPara[E_PARA_APP_AD_GV];

	if (nBlurSize % 2 == 0)
		nBlurSize++;

	cv::Mat temp;
	cv::blur(MatOrgImage, temp, cv::Size(nBlurSize, nBlurSize));
	Mat mtThresh;
	cv::threshold(temp, mtThresh, nThreshold, 255, CV_THRESH_BINARY);
	bool flag = false;

	vector<vector<cv::Point> > contours;

	FindBiggestContour(mtThresh, contours);

	if (contours.size() < 1)
	{
		return E_ERROR_CODE_FALSE;
	}

	double size = contourArea(contours[0]);
	if (size > dLimitSize)	flag = true;

	double dImageArea = MatOrgImage.cols * MatOrgImage.rows;

	//如果Panel占据画面的95%以上,则处理Display异常
	if (size / dImageArea > 0.95)
		return E_ERROR_CODE_FALSE;

	Rect rtObject = boundingRect(contours[0]);

	vector<cv::Rect> pADCheck;
	pADCheck.push_back(Rect(rtObject.x, rtObject.y, rtObject.width / 4, rtObject.height));   //左侧照明
	pADCheck.push_back(Rect(rtObject.x + rtObject.width * 3 / 4, rtObject.y, rtObject.width / 4, rtObject.height)); //右侧照明
	pADCheck.push_back(Rect(rtObject.x, rtObject.y, rtObject.width, rtObject.height / 4));   //上方照明
	pADCheck.push_back(Rect(rtObject.x, rtObject.y + rtObject.height * 3 / 4, rtObject.width, rtObject.height / 4)); //下部照明

	//检查某些照明异常
	for (int i = 0; i < pADCheck.size(); i++)
	{
		Scalar m, s;
		GetCheckROIOver(pADCheck[i], Rect(0, 0, MatOrgImage.cols, MatOrgImage.rows), pADCheck[i]);
		Mat mtLightcheck;
		MatOrgImage(pADCheck[i]).copyTo(mtLightcheck);
		cv::meanStdDev(mtLightcheck, m, s);
		if (m[0] < nADGV)
			flag = false;
	}

	if (flag == true)	nErrorCode = E_ERROR_CODE_TRUE;
	else				nErrorCode = E_ERROR_CODE_FALSE;

	return nErrorCode;
}

long CInspectAlign::Check_Abnormal_PADEdge(cv::Mat MatOrgImage, int nThreshold, double dCompare_Theta, Rect rtObject)
{
	//检查Pad Edge部分Cutting异常

	//前往Object外围区域Edge要搜索的区域
	Rect rtSearchArea = Rect(rtObject.x, rtObject.y + 100, rtObject.width + 100, rtObject.height - 100);

	GetCheckROIOver(rtSearchArea, Rect(0, 0, MatOrgImage.cols, MatOrgImage.rows), rtSearchArea);

	vector<cv::Point> pt[E_ABNORMAL_PAD_EDGE];

	uchar* ucImgData;
	int nGV;
	for (int nY = rtSearchArea.y; nY < rtSearchArea.height; nY += 10)
	{
		ucImgData = MatOrgImage.data + (nY * MatOrgImage.step);
		for (int nX = rtSearchArea.width; nX >= 0; nX--)
		{
			nGV = (int)*(nX + ucImgData);
			if (nGV == 255)
			{
				pt[E_ABNORMAL_PAD_EDGE_TOTAL].push_back(cv::Point(nX, nY));
				break;
			}
		}
	}

	//复制矢量,分成Top,Bottom和Middle 3个
	pt[E_ABNORMAL_PAD_EDGE_TOP] = pt[E_ABNORMAL_PAD_EDGE_TOTAL];
	pt[E_ABNORMAL_PAD_EDGE_BOTTOM] = pt[E_ABNORMAL_PAD_EDGE_TOTAL];
	pt[E_ABNORMAL_PAD_EDGE_MIDDLE] = pt[E_ABNORMAL_PAD_EDGE_TOTAL];

	pt[E_ABNORMAL_PAD_EDGE_TOP].erase(pt[E_ABNORMAL_PAD_EDGE_TOP].begin() + pt[E_ABNORMAL_PAD_EDGE_TOP].size() / 2
		, pt[E_ABNORMAL_PAD_EDGE_TOP].end());
	pt[E_ABNORMAL_PAD_EDGE_BOTTOM].erase(pt[E_ABNORMAL_PAD_EDGE_BOTTOM].begin()
		, pt[E_ABNORMAL_PAD_EDGE_BOTTOM].begin() + pt[E_ABNORMAL_PAD_EDGE_BOTTOM].size() / 2);
	pt[E_ABNORMAL_PAD_EDGE_MIDDLE].erase(pt[E_ABNORMAL_PAD_EDGE_MIDDLE].begin()
		, pt[E_ABNORMAL_PAD_EDGE_MIDDLE].begin() + pt[E_ABNORMAL_PAD_EDGE_TOTAL].size() / 3);
	pt[E_ABNORMAL_PAD_EDGE_MIDDLE].erase(pt[E_ABNORMAL_PAD_EDGE_MIDDLE].end() - pt[E_ABNORMAL_PAD_EDGE_TOTAL].size() / 3
		, pt[E_ABNORMAL_PAD_EDGE_MIDDLE].end());

	double dSlope[E_ABNORMAL_PAD_EDGE];

	//用最小自乘法求直线方程的斜率
	for (int i = 0; i < E_ABNORMAL_PAD_EDGE; i++)
	{
		if (MethodOfLeastSquares(pt[i], dSlope[i]) == E_ERROR_CODE_FALSE)
			return E_ERROR_CODE_FALSE;
	}

	//求角度
	double dTheth[E_ABNORMAL_PAD_EDGE];

	for (int i = 0; i < E_ABNORMAL_PAD_EDGE; i++)
	{
		dTheth[i] = atan(dSlope[i]) * 180. / PI;
		dTheth[i] += 90;
	}

	//角度范围
	double dMin = dTheth[E_ABNORMAL_PAD_EDGE_TOTAL] - dCompare_Theta;
	double dMax = dTheth[E_ABNORMAL_PAD_EDGE_TOTAL] + dCompare_Theta;

	//杀死向量
	for (int i = 0; i < E_ABNORMAL_PAD_EDGE; i++)
	{
		vector<cv::Point>().swap(pt[i]);
	}

	//比较角度
	if ((dTheth[E_ABNORMAL_PAD_EDGE_TOP]    < dMin || dTheth[E_ABNORMAL_PAD_EDGE_TOP]    > dMax)
		|| (dTheth[E_ABNORMAL_PAD_EDGE_BOTTOM] < dMin || dTheth[E_ABNORMAL_PAD_EDGE_BOTTOM] > dMax)
		|| (dTheth[E_ABNORMAL_PAD_EDGE_MIDDLE] < dMin || dTheth[E_ABNORMAL_PAD_EDGE_BOTTOM] > dMax))
		return E_ERROR_CODE_FALSE;

	return E_ERROR_CODE_TRUE;

}

long CInspectAlign::DoFindDefectLabel(cv::Mat matSrcBuf, double* dPara, cv::Point* ptCorner, double dAngel, CRect rectLabelArea[MAX_MEM_SIZE_LABEL_COUNT])
{

	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	int nLabelWidth = (int)dPara[E_PARA_AVI_Label_Width];
	int nLabelHeight = (int)dPara[E_PARA_AVI_Label_Height];

	cv::Mat matInputRotate;

	// 图像旋转
	matInputRotate = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	cv::Mat matRotation = cv::getRotationMatrix2D(Point(matSrcBuf.cols / 2, matSrcBuf.rows / 2), dAngel, 1.0);
	cv::warpAffine(matSrcBuf, matInputRotate, matRotation, matSrcBuf.size());


	cv::Mat matTempBuf;
	matTempBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

	int nMorp = (int)31;  //闭运算
	int nThreshold = (int)15;  //二值化

	if (nMorp > 0)
	{
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nMorp, nMorp), cv::Point(nMorp / 2, nMorp / 2));
		cv::morphologyEx(matInputRotate, matTempBuf, MORPH_CLOSE, StructElem);

		StructElem.release();
	}
	else
		matInputRotate.copyTo(matTempBuf);


	cv::threshold(matTempBuf, matTempBuf, nThreshold, 255.0, THRESH_BINARY);

	vector< vector< cv::Point2i > > contours;
	vector<Vec4i> hierarchy;


	findContours(matTempBuf, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if ((int)contours.size() < 2)
	{
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("The number of Label Contours < 2."));
		return E_ERROR_CODE_FALSE;
	}
	//寻找最大面积轮廓
	int nMaxIndex = 0;
	double dMaxSize = 0.0;
	for (int i = 0; i < (int)contours.size(); i++)
	{
		double dValue = cv::contourArea(contours[i]);
		if (dValue > dMaxSize)
		{
			dMaxSize = dValue;
			nMaxIndex = i;
		}
	}

	//寻找Label区域轮廓
	int nLabelIndex = 0;
	double dLabelSize = 0.0;
	for (int i = 0; i < (int)contours.size(); i++)
	{
		if (i == nMaxIndex) continue;
		double dValue = cv::contourArea(contours[i]);
		if (dValue > dLabelSize)
		{
			dLabelSize = dValue;
			nLabelIndex = i;
		}
	}

	int nRealLabelArea = cv::contourArea(contours[nLabelIndex]);

	if (nRealLabelArea < (nLabelWidth * nLabelHeight))
	{
		return E_ERROR_CODE_TRUE;
	}

	cv::Rect LabelRect;
	LabelRect = boundingRect(contours[nLabelIndex]);

	int offset = 60;
	LabelRect.x -= offset;
	LabelRect.y -= offset;
	LabelRect.width += 2 * offset;
	LabelRect.height += 2 * offset;

	if (LabelRect.x <= 0) LabelRect.x = 0;
	if (LabelRect.y <= 0) LabelRect.y = 0;

	rectLabelArea[0].left = LabelRect.x;
	rectLabelArea[0].top = LabelRect.y;
	rectLabelArea[0].right = LabelRect.x + LabelRect.width;
	rectLabelArea[0].bottom = LabelRect.y + LabelRect.height;

	{
		matInputRotate.release();
		matRotation.release();
		matTempBuf.release();
	}

	return E_ERROR_CODE_TRUE;
}

//yuxuefei for Mark
long CInspectAlign::DoFindDefectMark(cv::Mat matSrcBuf, double* dPara, cv::Point* ptCorner, double dAngel, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT], int nMarkROICnt)
{

	int nErrorCode = E_ERROR_CODE_TRUE;
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	float fVal = 4.f * PI;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	cv::Mat matInputRotate;

	// 图像旋转
	matInputRotate = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	cv::Mat matRotation = cv::getRotationMatrix2D(Point(matSrcBuf.cols / 2, matSrcBuf.rows / 2), dAngel, 1.0);
	cv::warpAffine(matSrcBuf, matInputRotate, matRotation, matSrcBuf.size());
	matRotation.release();


	for (int nROICnt = 0; nROICnt < nMarkROICnt; nROICnt++)
	{
		switch (nROICnt)
		{
		case E_MARK_RIGHT_TOP:
			//右上角Mark检测
			nErrorCode = DoFindMarkTop(matInputRotate, dPara, ptCorner, rcMarkROI, rectMarkArea);
			break;
		case E_MARK_RIGHT_BOTTOM:
			//右下角Mark检测
			nErrorCode = DoFindMarkBottom(matInputRotate, dPara, ptCorner, rcMarkROI, rectMarkArea);
			break;
		}

	}

	matInputRotate.release();
	return E_ERROR_CODE_TRUE;
}

//yuxuefei for Mark right top
long CInspectAlign::DoFindMarkTop(cv::Mat& matSrcBuf, double* dPara, cv::Point* ptCorner, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT])
{

	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//Judge Whether use the Area
	if (rectMarkArea[E_MARK_RIGHT_TOP].Width() == 0 || rectMarkArea[E_MARK_RIGHT_TOP].Height() == 0)
	{
		return E_ERROR_CODE_TRUE;
	}

	float fVal = 4.f * PI;

	int nErrorCode = E_ERROR_CODE_TRUE;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	long	nWidth = (long)matSrcBuf.cols;
	long	nHeight = (long)matSrcBuf.rows;


	cv::Mat matMark, matTemple;
	cv::Rect recMark;
	recMark.x = ptCorner[E_CORNER_LEFT_TOP].x + rectMarkArea[E_MARK_RIGHT_TOP].left;
	recMark.y = ptCorner[E_CORNER_LEFT_TOP].y + rectMarkArea[E_MARK_RIGHT_TOP].top;
	recMark.width = rectMarkArea[E_MARK_RIGHT_TOP].Width();
	recMark.height = rectMarkArea[E_MARK_RIGHT_TOP].Height();

	//超出区域判断
	if (recMark.x < ptCorner[E_CORNER_LEFT_TOP].x || recMark.x > ptCorner[E_CORNER_RIGHT_TOP].x) return E_ERROR_CODE_ROI_OVER;
	if (recMark.y <  ptCorner[E_CORNER_LEFT_TOP].y || recMark.y > ptCorner[E_CORNER_LEFT_BOTTOM].y) return E_ERROR_CODE_ROI_OVER;
	if ((recMark.x + recMark.width) > ptCorner[E_CORNER_RIGHT_TOP].x || (recMark.y + recMark.height) > ptCorner[E_CORNER_LEFT_BOTTOM].y) return E_ERROR_CODE_ROI_OVER;

	matMark = matSrcBuf(recMark);

	//背景区域提取

	cv::Mat matGauBuf = cMem->GetMat(matMark.size(), matMark.type(), false);;
	cv::GaussianBlur(matMark, matGauBuf, cv::Size(31, 31), 5, 5);

	cv::Mat matBK;
	matGauBuf.copyTo(matBK);
	cv::Mat matBGBuf = cMem->GetMat(matMark.size(), matMark.type(), false);
	Estimation_XY(matGauBuf, matBGBuf, dPara, &cMatBufTemp);

	cv::Mat matSubDaBuf;
	cv::subtract(matBGBuf, matBK, matSubDaBuf);


	// 二值化
	cv::threshold(matSubDaBuf, matTemple, 2, 255.0, CV_THRESH_BINARY);

	cv::Mat erodemgKsize = getStructuringElement(MORPH_RECT, Size(5, 5));
	cv::morphologyEx(matTemple, matTemple, MORPH_ERODE, erodemgKsize, Point(-1, -1));

	cv::Mat morImgKsize = getStructuringElement(MORPH_RECT, Size(31, 31));
	cv::morphologyEx(matTemple, matTemple, MORPH_CLOSE, morImgKsize, Point(-1, -1), 2);

	//膨胀
	cv::Mat dilateImgKsize = getStructuringElement(MORPH_RECT, Size(19, 19));
	cv::morphologyEx(matTemple, matTemple, MORPH_DILATE, dilateImgKsize, Point(-1, -1));


	//释放图像缓存
	{
		matGauBuf.release();
		matBGBuf.release();
		matBK.release();
		matSubDaBuf.release();
		erodemgKsize.release();
		morImgKsize.release();
		dilateImgKsize.release();
	}

	int nTotalLabel = 0;
	cv::Mat matLabel, matStats, matCentroid;
	nTotalLabel = cv::connectedComponentsWithStats(matTemple, matLabel, matStats, matCentroid, 8, CV_32S) - 1;

	m_BlobMark.resize(nTotalLabel);

	if (nTotalLabel < 1)
	{
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("The number of Mrak Components < 1."));
		return E_ERROR_CODE_FALSE;
	}
	else
	{
		for (int idx = 1; idx <= nTotalLabel; idx++)
		{

			int nBlobNum = idx - 1;

			m_BlobMark.at(nBlobNum).rectBox.x = matStats.at<int>(idx, CC_STAT_LEFT);
			m_BlobMark.at(nBlobNum).rectBox.y = matStats.at<int>(idx, CC_STAT_TOP);
			m_BlobMark.at(nBlobNum).rectBox.width = matStats.at<int>(idx, CC_STAT_WIDTH);
			m_BlobMark.at(nBlobNum).rectBox.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

			// 按眉 林函 ( 硅版 GV侩档 )
			int nOffSet = 20;

			int nSX = m_BlobMark.at(nBlobNum).rectBox.x - nOffSet;
			int nSY = m_BlobMark.at(nBlobNum).rectBox.y - nOffSet;
			int nEX = m_BlobMark.at(nBlobNum).rectBox.x + m_BlobMark.at(nBlobNum).rectBox.width + nOffSet + nOffSet;
			int nEY = m_BlobMark.at(nBlobNum).rectBox.y + m_BlobMark.at(nBlobNum).rectBox.height + nOffSet + nOffSet;

			if (nSX < 0)	nSX = 0;
			if (nSY < 0)	nSY = 0;
			if (nEX >= matMark.cols)	nEX = matMark.cols - 1;
			if (nEY >= matMark.rows)	nEY = matMark.rows - 1;

			cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);

			__int64 nCount_in = 0;
			__int64 nCount_out = 0;
			__int64 nSum_in = 0;
			__int64 nSum_out = 0;

			cv::Mat matTmp_src = matMark(rectTemp);
			cv::Mat matTmp_label = matLabel(rectTemp);
			cv::Mat matTemp = cv::Mat::zeros(rectTemp.height, rectTemp.width, CV_8UC1);

			for (int y = 0; y < rectTemp.height; y++)
			{
				int* ptrLabel = (int*)matTmp_label.ptr(y);
				uchar* ptrGray = (uchar*)matTmp_src.ptr(y);
				uchar* ptrTemp = (uchar*)matTemp.ptr(y);

				for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++)
				{

					if (*ptrLabel == idx)
					{
						nSum_in += *ptrGray;
						nCount_in++;


						m_BlobMark.at(nBlobNum).ptIndexs.push_back(cv::Point(nSX + x, nSY + y));

						*ptrTemp = (uchar)255;

						m_BlobMark.at(nBlobNum).nHist[*ptrGray]++;
					}
					// 促弗版快 硅版
					else
					{
						// 扼骇 锅龋啊 0牢 版快...
						// 促弗 按眉 曼炼瞪荐 乐栏骨肺.... 炼扒 眠啊
						if (*ptrLabel == 0)
						{
							nSum_out += *ptrGray;
							nCount_out++;
						}
					}
				}
			}


			m_BlobMark.at(nBlobNum).nSumGV = nSum_in;


			m_BlobMark.at(nBlobNum).nArea = nCount_in;	//matStats.at<int>(idx, CC_STAT_AREA);

			if (m_BlobMark.at(nBlobNum).nArea < 1000)
				continue;

			m_BlobMark.at(nBlobNum).nBoxArea = m_BlobMark.at(nBlobNum).rectBox.width * m_BlobMark.at(nBlobNum).rectBox.height;


			cv::Scalar m, s;
			cv::meanStdDev(matTmp_src, m, s, matTemp);
			m_BlobMark.at(nBlobNum).fStdDev = float(s[0]);

			// Contours 备窍扁
			vector<vector<cv::Point>>	ptContours;
			vector<vector<cv::Point>>().swap(ptContours);
			cv::findContours(matTemp, ptContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			// Perimeter 备窍扁
			if (ptContours.size() != 0)
			{
				// ROI 康惑捞骨肺, 焊沥 鞘夸
				// 寇胞 谅钎 搬苞 汗荤
				for (int m = 0; m < ptContours.size(); m++)
				{
					for (int k = 0; k < ptContours.at(m).size(); k++)
						m_BlobMark.at(nBlobNum).ptContours.push_back(cv::Point(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY));
				}
			}
			else
			{
				// 盔夯 谅钎捞骨肺, 焊沥 救秦档 凳.
				// 寇胞 谅钎 搬苞 汗荤
				m_BlobMark.at(nBlobNum).ptContours.resize((int)m_BlobMark.at(nBlobNum).ptIndexs.size());
				std::copy(m_BlobMark.at(nBlobNum).ptIndexs.begin(), m_BlobMark.at(nBlobNum).ptIndexs.end(), m_BlobMark.at(nBlobNum).ptContours.begin());
			}
			m_BlobMark.at(nBlobNum).fPerimeter = float(cv::arcLength(m_BlobMark.at(nBlobNum).ptContours, true));
			vector<vector<cv::Point>>().swap(ptContours);

			// Roundness 备窍扁
			m_BlobMark.at(nBlobNum).fRoundness = (fVal * m_BlobMark.at(nBlobNum).nArea)
				/ (m_BlobMark.at(nBlobNum).fPerimeter * m_BlobMark.at(nBlobNum).fPerimeter);

			// 按眉啊 盔 葛剧俊 倔付唱 啊鳖款啊? ( 笛饭^2 / 4 * Pi * 搁利 )
			m_BlobMark.at(nBlobNum).fCompactness = (m_BlobMark.at(nBlobNum).fPerimeter * m_BlobMark.at(nBlobNum).fPerimeter)
				/ (fVal * float(m_BlobMark.at(nBlobNum).nArea));

			// Defect GV 备窍扁
			m_BlobMark.at(nBlobNum).fMeanGV = nSum_in / (float)nCount_in;

			//	硅版 GV 备窍扁
			m_BlobMark.at(nBlobNum).fBKGV = nSum_out / (float)nCount_out;

			// GV 瞒捞蔼 备窍扁 ( 硅版 - 按眉 )
			m_BlobMark.at(nBlobNum).fDiffGV = m_BlobMark.at(nBlobNum).fBKGV - m_BlobMark.at(nBlobNum).fMeanGV;

			// min, max GV 备窍扁
			double valMin, valMax, ta, tb;
			cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
			cv::minMaxIdx(matTmp_src, &ta, &tb);
			m_BlobMark.at(nBlobNum).nMinGV = (long)valMin;
			m_BlobMark.at(nBlobNum).nMaxGV = (long)valMax;

			// 按眉 弥家 灌扁 / 按眉 乞闭 灌扁
			m_BlobMark.at(nBlobNum).nMinGVRatio = m_BlobMark.at(nBlobNum).nMinGV / m_BlobMark.at(nBlobNum).fBKGV;

			// 按眉 弥措 灌扁 / 按眉 乞闭 灌扁
			m_BlobMark.at(nBlobNum).nMaxGVRatio = m_BlobMark.at(nBlobNum).nMaxGV / m_BlobMark.at(nBlobNum).fBKGV;

			//      硅版 灌扁 / 按眉 乞闭 灌扁
			m_BlobMark.at(nBlobNum).fDiffGVRatio = m_BlobMark.at(nBlobNum).fMeanGV / m_BlobMark.at(nBlobNum).fBKGV;

			// Center Point 备窍扁
			m_BlobMark.at(nBlobNum).ptCenter.x = (int)matCentroid.at<double>(idx, 0);
			m_BlobMark.at(nBlobNum).ptCenter.y = (int)matCentroid.at<double>(idx, 1);


		}

	}
	//选择面积最大的进行保留
	if (m_BlobMark.size() >= 2)
	{
		int nMaxIndex = 0;
		int nMaxArea = 0;
		for (int k = 0; k < m_BlobMark.size(); k++)
		{
			int nValue = m_BlobMark[k].nArea;
			if (nValue > nMaxArea)
			{
				nMaxIndex = k;
				nMaxArea = nValue;
			}
		}

		int nSecondIndex = 0;
		int nSecondArea = 0;
		for (int j = 0; j < m_BlobMark.size(); j++)
		{
			if (j == nMaxIndex) continue;
			int nValue = m_BlobMark[j].nArea;
			if (nValue > nSecondArea)
			{
				nSecondArea = nValue;
				nSecondIndex = j;
			}
		}

		int nAreaRation = nMaxArea / nSecondArea;

		if (nAreaRation > 2)
		{
			int offset = 50;
			m_BlobMark[nMaxIndex].rectBox.x -= offset;
			m_BlobMark[nMaxIndex].rectBox.y -= offset;
			m_BlobMark[nMaxIndex].rectBox.width += 2 * offset;
			m_BlobMark[nMaxIndex].rectBox.height += 2 * offset;

			if (m_BlobMark[nMaxIndex].rectBox.x <= 0) m_BlobMark[nMaxIndex].rectBox.x = 0;
			if (m_BlobMark[nMaxIndex].rectBox.y <= 0) m_BlobMark[nMaxIndex].rectBox.y = 0;

			//将Mark值赋值给Align参数中
			rcMarkROI[E_MARK_RIGHT_TOP].x = recMark.x + m_BlobMark[nMaxIndex].rectBox.x;
			rcMarkROI[E_MARK_RIGHT_TOP].y = recMark.y + m_BlobMark[nMaxIndex].rectBox.y;
			rcMarkROI[E_MARK_RIGHT_TOP].width = m_BlobMark[nMaxIndex].rectBox.width;
			rcMarkROI[E_MARK_RIGHT_TOP].height = m_BlobMark[nMaxIndex].rectBox.height;

		}
		else
		{
			if (m_BlobMark[nMaxIndex].fDiffGV > m_BlobMark[nSecondIndex].fDiffGV)
			{
				int offset = 50;
				m_BlobMark[nMaxIndex].rectBox.x -= offset;
				m_BlobMark[nMaxIndex].rectBox.y -= offset;
				m_BlobMark[nMaxIndex].rectBox.width += 2 * offset;
				m_BlobMark[nMaxIndex].rectBox.height += 2 * offset;

				if (m_BlobMark[nMaxIndex].rectBox.x <= 0) m_BlobMark[nMaxIndex].rectBox.x = 0;
				if (m_BlobMark[nMaxIndex].rectBox.y <= 0) m_BlobMark[nMaxIndex].rectBox.y = 0;

				//将Mark值赋值给Align参数中
				rcMarkROI[E_MARK_RIGHT_TOP].x = recMark.x + m_BlobMark[nMaxIndex].rectBox.x;
				rcMarkROI[E_MARK_RIGHT_TOP].y = recMark.y + m_BlobMark[nMaxIndex].rectBox.y;
				rcMarkROI[E_MARK_RIGHT_TOP].width = m_BlobMark[nMaxIndex].rectBox.width;
				rcMarkROI[E_MARK_RIGHT_TOP].height = m_BlobMark[nMaxIndex].rectBox.height;
			}
			else
			{
				int offset = 50;
				m_BlobMark[nSecondIndex].rectBox.x -= offset;
				m_BlobMark[nSecondIndex].rectBox.y -= offset;
				m_BlobMark[nSecondIndex].rectBox.width += 2 * offset;
				m_BlobMark[nSecondIndex].rectBox.height += 2 * offset;

				if (m_BlobMark[nSecondIndex].rectBox.x <= 0) m_BlobMark[nSecondIndex].rectBox.x = 0;
				if (m_BlobMark[nSecondIndex].rectBox.y <= 0) m_BlobMark[nSecondIndex].rectBox.y = 0;

				//将Mark值赋值给Align参数中
				rcMarkROI[E_MARK_RIGHT_TOP].x = recMark.x + m_BlobMark[nSecondIndex].rectBox.x;
				rcMarkROI[E_MARK_RIGHT_TOP].y = recMark.y + m_BlobMark[nSecondIndex].rectBox.y;
				rcMarkROI[E_MARK_RIGHT_TOP].width = m_BlobMark[nSecondIndex].rectBox.width;
				rcMarkROI[E_MARK_RIGHT_TOP].height = m_BlobMark[nSecondIndex].rectBox.height;
			}
		}

	}
	else
	{
		int nMaxIndex = 0;
		int offset = 50;
		m_BlobMark[nMaxIndex].rectBox.x -= offset;
		m_BlobMark[nMaxIndex].rectBox.y -= offset;
		m_BlobMark[nMaxIndex].rectBox.width += 2 * offset;
		m_BlobMark[nMaxIndex].rectBox.height += 2 * offset;

		if (m_BlobMark[nMaxIndex].rectBox.x <= 0) m_BlobMark[nMaxIndex].rectBox.x = 0;
		if (m_BlobMark[nMaxIndex].rectBox.y <= 0) m_BlobMark[nMaxIndex].rectBox.y = 0;



		//将Mark值赋值给Align参数中
		rcMarkROI[E_MARK_RIGHT_TOP].x = recMark.x + m_BlobMark[nMaxIndex].rectBox.x;
		rcMarkROI[E_MARK_RIGHT_TOP].y = recMark.y + m_BlobMark[nMaxIndex].rectBox.y;
		rcMarkROI[E_MARK_RIGHT_TOP].width = m_BlobMark[nMaxIndex].rectBox.width;
		rcMarkROI[E_MARK_RIGHT_TOP].height = m_BlobMark[nMaxIndex].rectBox.height;
	}

	return nErrorCode;
}

//yuxuefei for Mark right bottom
long CInspectAlign::DoFindMarkBottom(cv::Mat& matSrcBuf, double* dPara, cv::Point* ptCorner, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT])
{
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;
	//Judge Whether use the Area
	if (rectMarkArea[E_MARK_RIGHT_BOTTOM].Width() == 0 || rectMarkArea[E_MARK_RIGHT_BOTTOM].Height() == 0)
	{
		return E_ERROR_CODE_TRUE;
	}

	float fVal = 4.f * PI;

	int nErrorCode = E_ERROR_CODE_TRUE;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	long	nWidth = (long)matSrcBuf.cols;
	long	nHeight = (long)matSrcBuf.rows;


	cv::Mat matMark, matTemple;
	cv::Rect recMark;
	recMark.x = ptCorner[E_CORNER_LEFT_TOP].x + rectMarkArea[E_MARK_RIGHT_BOTTOM].left;
	recMark.y = ptCorner[E_CORNER_LEFT_TOP].y + rectMarkArea[E_MARK_RIGHT_BOTTOM].top;
	recMark.width = rectMarkArea[E_MARK_RIGHT_BOTTOM].Width();
	recMark.height = rectMarkArea[E_MARK_RIGHT_BOTTOM].Height();

	//超出区域判断
	if (recMark.x < ptCorner[E_CORNER_LEFT_TOP].x || recMark.x > ptCorner[E_CORNER_RIGHT_TOP].x) return E_ERROR_CODE_ROI_OVER;
	if (recMark.y <  ptCorner[E_CORNER_LEFT_TOP].y || recMark.y > ptCorner[E_CORNER_LEFT_BOTTOM].y) return E_ERROR_CODE_ROI_OVER;
	if ((recMark.x + recMark.width) > ptCorner[E_CORNER_RIGHT_TOP].x || (recMark.y + recMark.height) > ptCorner[E_CORNER_LEFT_BOTTOM].y) return E_ERROR_CODE_ROI_OVER;

	matMark = matSrcBuf(recMark);


	//背景区域提取

	cv::Mat matGauBuf = cMem->GetMat(matMark.size(), matMark.type(), false);;
	cv::GaussianBlur(matMark, matGauBuf, cv::Size(31, 31), 5, 5);

	cv::Mat matBK;
	matGauBuf.copyTo(matBK);
	cv::Mat matBGBuf = cMem->GetMat(matMark.size(), matMark.type(), false);
	Estimation_XY(matGauBuf, matBGBuf, dPara, &cMatBufTemp);

	cv::Mat matSubDaBuf;
	cv::subtract(matBGBuf, matBK, matSubDaBuf);


	// 二值化
	cv::threshold(matSubDaBuf, matTemple, 2, 255.0, CV_THRESH_BINARY);

	cv::Mat erodemgKsize = getStructuringElement(MORPH_RECT, Size(5, 5));
	cv::morphologyEx(matTemple, matTemple, MORPH_ERODE, erodemgKsize, Point(-1, -1));

	cv::Mat morImgKsize = getStructuringElement(MORPH_RECT, Size(31, 31));
	cv::morphologyEx(matTemple, matTemple, MORPH_CLOSE, morImgKsize, Point(-1, -1), 2);

	//膨胀
	cv::Mat dilateImgKsize = getStructuringElement(MORPH_RECT, Size(5, 5));
	cv::morphologyEx(matTemple, matTemple, MORPH_DILATE, dilateImgKsize, Point(-1, -1));

	//释放图像缓存
	{
		matGauBuf.release();
		matBGBuf.release();
		matBK.release();
		matSubDaBuf.release();
		erodemgKsize.release();
		morImgKsize.release();
		dilateImgKsize.release();
	}

	int nTotalLabel = 0;
	cv::Mat matLabel, matStats, matCentroid;
	nTotalLabel = cv::connectedComponentsWithStats(matTemple, matLabel, matStats, matCentroid, 8, CV_32S) - 1;

	m_BlobMark.resize(nTotalLabel);

	if (nTotalLabel < 1)
	{
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("The number of Mrak Components < 1."));
		return E_ERROR_CODE_FALSE;
	}
	else
	{
		for (int idx = 1; idx <= nTotalLabel; idx++)
		{

			int nBlobNum = idx - 1;

			m_BlobMark.at(nBlobNum).rectBox.x = matStats.at<int>(idx, CC_STAT_LEFT);
			m_BlobMark.at(nBlobNum).rectBox.y = matStats.at<int>(idx, CC_STAT_TOP);
			m_BlobMark.at(nBlobNum).rectBox.width = matStats.at<int>(idx, CC_STAT_WIDTH);
			m_BlobMark.at(nBlobNum).rectBox.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

			// 按眉 林函 ( 硅版 GV侩档 )
			int nOffSet = 20;

			int nSX = m_BlobMark.at(nBlobNum).rectBox.x - nOffSet;
			int nSY = m_BlobMark.at(nBlobNum).rectBox.y - nOffSet;
			int nEX = m_BlobMark.at(nBlobNum).rectBox.x + m_BlobMark.at(nBlobNum).rectBox.width + nOffSet + nOffSet;
			int nEY = m_BlobMark.at(nBlobNum).rectBox.y + m_BlobMark.at(nBlobNum).rectBox.height + nOffSet + nOffSet;

			if (nSX < 0)	nSX = 0;
			if (nSY < 0)	nSY = 0;
			if (nEX >= matMark.cols)	nEX = matMark.cols - 1;
			if (nEY >= matMark.rows)	nEY = matMark.rows - 1;

			cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);

			__int64 nCount_in = 0;
			__int64 nCount_out = 0;
			__int64 nSum_in = 0;
			__int64 nSum_out = 0;

			cv::Mat matTmp_src = matMark(rectTemp);
			cv::Mat matTmp_label = matLabel(rectTemp);
			cv::Mat matTemp = cv::Mat::zeros(rectTemp.height, rectTemp.width, CV_8UC1);

			for (int y = 0; y < rectTemp.height; y++)
			{
				int* ptrLabel = (int*)matTmp_label.ptr(y);
				uchar* ptrGray = (uchar*)matTmp_src.ptr(y);
				uchar* ptrTemp = (uchar*)matTemp.ptr(y);

				for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++)
				{

					if (*ptrLabel == idx)
					{
						nSum_in += *ptrGray;
						nCount_in++;


						m_BlobMark.at(nBlobNum).ptIndexs.push_back(cv::Point(nSX + x, nSY + y));

						*ptrTemp = (uchar)255;

						m_BlobMark.at(nBlobNum).nHist[*ptrGray]++;
					}
					// 促弗版快 硅版
					else
					{
						// 扼骇 锅龋啊 0牢 版快...
						// 促弗 按眉 曼炼瞪荐 乐栏骨肺.... 炼扒 眠啊
						if (*ptrLabel == 0)
						{
							nSum_out += *ptrGray;
							nCount_out++;
						}
					}
				}
			}


			m_BlobMark.at(nBlobNum).nSumGV = nSum_in;


			m_BlobMark.at(nBlobNum).nArea = nCount_in;	//matStats.at<int>(idx, CC_STAT_AREA);

			if (m_BlobMark.at(nBlobNum).nArea < 1000)
				continue;

			m_BlobMark.at(nBlobNum).nBoxArea = m_BlobMark.at(nBlobNum).rectBox.width * m_BlobMark.at(nBlobNum).rectBox.height;


			cv::Scalar m, s;
			cv::meanStdDev(matTmp_src, m, s, matTemp);
			m_BlobMark.at(nBlobNum).fStdDev = float(s[0]);

			// Contours 备窍扁
			vector<vector<cv::Point>>	ptContours;
			vector<vector<cv::Point>>().swap(ptContours);
			cv::findContours(matTemp, ptContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

			// Perimeter 备窍扁
			if (ptContours.size() != 0)
			{
				// ROI 康惑捞骨肺, 焊沥 鞘夸
				// 寇胞 谅钎 搬苞 汗荤
				for (int m = 0; m < ptContours.size(); m++)
				{
					for (int k = 0; k < ptContours.at(m).size(); k++)
						m_BlobMark.at(nBlobNum).ptContours.push_back(cv::Point(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY));
				}
			}
			else
			{
				// 盔夯 谅钎捞骨肺, 焊沥 救秦档 凳.
				// 寇胞 谅钎 搬苞 汗荤
				m_BlobMark.at(nBlobNum).ptContours.resize((int)m_BlobMark.at(nBlobNum).ptIndexs.size());
				std::copy(m_BlobMark.at(nBlobNum).ptIndexs.begin(), m_BlobMark.at(nBlobNum).ptIndexs.end(), m_BlobMark.at(nBlobNum).ptContours.begin());
			}
			m_BlobMark.at(nBlobNum).fPerimeter = float(cv::arcLength(m_BlobMark.at(nBlobNum).ptContours, true));
			vector<vector<cv::Point>>().swap(ptContours);

			// Roundness 备窍扁
			m_BlobMark.at(nBlobNum).fRoundness = (fVal * m_BlobMark.at(nBlobNum).nArea)
				/ (m_BlobMark.at(nBlobNum).fPerimeter * m_BlobMark.at(nBlobNum).fPerimeter);

			// 按眉啊 盔 葛剧俊 倔付唱 啊鳖款啊? ( 笛饭^2 / 4 * Pi * 搁利 )
			m_BlobMark.at(nBlobNum).fCompactness = (m_BlobMark.at(nBlobNum).fPerimeter * m_BlobMark.at(nBlobNum).fPerimeter)
				/ (fVal * float(m_BlobMark.at(nBlobNum).nArea));

			// Defect GV 备窍扁
			m_BlobMark.at(nBlobNum).fMeanGV = nSum_in / (float)nCount_in;

			//	硅版 GV 备窍扁
			m_BlobMark.at(nBlobNum).fBKGV = nSum_out / (float)nCount_out;

			// GV 瞒捞蔼 备窍扁 ( 硅版 - 按眉 )
			m_BlobMark.at(nBlobNum).fDiffGV = m_BlobMark.at(nBlobNum).fBKGV - m_BlobMark.at(nBlobNum).fMeanGV;

			// min, max GV 备窍扁
			double valMin, valMax, ta, tb;
			cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
			cv::minMaxIdx(matTmp_src, &ta, &tb);
			m_BlobMark.at(nBlobNum).nMinGV = (long)valMin;
			m_BlobMark.at(nBlobNum).nMaxGV = (long)valMax;

			// 按眉 弥家 灌扁 / 按眉 乞闭 灌扁
			m_BlobMark.at(nBlobNum).nMinGVRatio = m_BlobMark.at(nBlobNum).nMinGV / m_BlobMark.at(nBlobNum).fBKGV;

			// 按眉 弥措 灌扁 / 按眉 乞闭 灌扁
			m_BlobMark.at(nBlobNum).nMaxGVRatio = m_BlobMark.at(nBlobNum).nMaxGV / m_BlobMark.at(nBlobNum).fBKGV;

			//      硅版 灌扁 / 按眉 乞闭 灌扁
			m_BlobMark.at(nBlobNum).fDiffGVRatio = m_BlobMark.at(nBlobNum).fMeanGV / m_BlobMark.at(nBlobNum).fBKGV;

			// Center Point 备窍扁
			m_BlobMark.at(nBlobNum).ptCenter.x = (int)matCentroid.at<double>(idx, 0);
			m_BlobMark.at(nBlobNum).ptCenter.y = (int)matCentroid.at<double>(idx, 1);


			if (m_BlobMark.at(nBlobNum).fDiffGV == 0.0)
			{
				if (m_BlobMark.at(nBlobNum).fBKGV == 0)
				{
					m_BlobMark.at(nBlobNum).fSEMU = 1.0
						/ (1.97f / (cv::pow((float)m_BlobMark.at(nBlobNum).nArea, 0.33f) + 0.72f));
				}
				else
				{
					m_BlobMark.at(nBlobNum).fSEMU = (0.000001 / m_BlobMark.at(nBlobNum).fBKGV)
						/ (1.97 / (cv::pow((float)m_BlobMark.at(nBlobNum).nArea, 0.33f) + 0.72f));
				}
			}
			else
			{
				if (m_BlobMark.at(nBlobNum).fBKGV == 0)
				{
					m_BlobMark.at(nBlobNum).fSEMU = (fabs(m_BlobMark.at(nBlobNum).fMeanGV - m_BlobMark.at(nBlobNum).fBKGV) / 0.000001)
						/ (1.97 / (cv::pow((float)m_BlobMark.at(nBlobNum).nArea, 0.33f) + 0.72f));
				}
				else
				{
					m_BlobMark.at(nBlobNum).fSEMU = (fabs(m_BlobMark.at(nBlobNum).fMeanGV - m_BlobMark.at(nBlobNum).fBKGV) / m_BlobMark.at(nBlobNum).fBKGV)
						/ (1.97 / (cv::pow((float)m_BlobMark.at(nBlobNum).nArea, 0.33f) + 0.72f));
				}
			}

			cv::RotatedRect BoundingBox = cv::minAreaRect(m_BlobMark.at(nBlobNum).ptIndexs);

			// 雀傈等 荤阿屈 怖瘤痢 4俺
			//cv::Point2f vertices[4];
			//BoundingBox.points(vertices);

			// Box width and length
			m_BlobMark.at(nBlobNum).BoxSize = BoundingBox.size;

			// Angle between the horizontal axis
			m_BlobMark.at(nBlobNum).fAngle = BoundingBox.angle;

			// Minor Axis & Major Axis
			if (BoundingBox.size.width > BoundingBox.size.height)
			{
				m_BlobMark.at(nBlobNum).fMinorAxis = BoundingBox.size.width;
				m_BlobMark.at(nBlobNum).fMajorAxis = BoundingBox.size.height;
			}
			else
			{
				m_BlobMark.at(nBlobNum).fMinorAxis = BoundingBox.size.height;
				m_BlobMark.at(nBlobNum).fMajorAxis = BoundingBox.size.width;
			}

			// Feret’s area
			m_BlobMark.at(nBlobNum).fMinBoxArea = m_BlobMark.at(nBlobNum).fMinorAxis * m_BlobMark.at(nBlobNum).fMajorAxis;

			// Axis Ratio
			if (m_BlobMark.at(nBlobNum).fMajorAxis > 0)
				m_BlobMark.at(nBlobNum).fAxisRatio = m_BlobMark.at(nBlobNum).fMinorAxis / m_BlobMark.at(nBlobNum).fMajorAxis;
			else
				m_BlobMark.at(nBlobNum).fAxisRatio = 0.f;

			// Min Bounding Box 搁利 厚啦 / 按眉 搁利 ( Area porosity )
			m_BlobMark.at(nBlobNum).fMinBoxRatio = m_BlobMark.at(nBlobNum).fMinBoxArea / (float)m_BlobMark.at(nBlobNum).nArea;
			//choikwangil
			m_BlobMark.at(nBlobNum).fMeanAreaRatio = m_BlobMark.at(nBlobNum).fMeanGV / (float)m_BlobMark.at(nBlobNum).nArea;
			// 且寸 秦力



		}

	}
	//选择面积最大的进行保留
	if (m_BlobMark.size() >= 2)
	{
		int nMaxIndex = 0;
		int nMaxArea = 0;
		for (int k = 0; k < m_BlobMark.size(); k++)
		{
			int nValue = m_BlobMark[k].nArea;
			if (nValue > nMaxArea)
			{
				nMaxIndex = k;
				nMaxArea = nValue;
			}
		}

		int nSecondIndex = 0;
		int nSecondArea = 0;
		for (int j = 0; j < m_BlobMark.size(); j++)
		{
			if (j == nMaxIndex) continue;
			int nValue = m_BlobMark[j].nArea;
			if (nValue > nSecondArea)
			{
				nSecondArea = nValue;
				nSecondIndex = j;
			}
		}

		int nAreaRation = nMaxArea / nSecondArea;

		if (nAreaRation > 2)
		{
			int offset = 50;
			m_BlobMark[nMaxIndex].rectBox.x -= offset;
			m_BlobMark[nMaxIndex].rectBox.y -= offset;
			m_BlobMark[nMaxIndex].rectBox.width += 2 * offset;
			m_BlobMark[nMaxIndex].rectBox.height += 2 * offset;

			if (m_BlobMark[nMaxIndex].rectBox.x <= 0) m_BlobMark[nMaxIndex].rectBox.x = 0;
			if (m_BlobMark[nMaxIndex].rectBox.y <= 0) m_BlobMark[nMaxIndex].rectBox.y = 0;

			//将Mark值赋值给Align参数中
			rcMarkROI[E_MARK_RIGHT_BOTTOM].x = recMark.x + m_BlobMark[nMaxIndex].rectBox.x;
			rcMarkROI[E_MARK_RIGHT_BOTTOM].y = recMark.y + m_BlobMark[nMaxIndex].rectBox.y;
			rcMarkROI[E_MARK_RIGHT_BOTTOM].width = m_BlobMark[nMaxIndex].rectBox.width;
			rcMarkROI[E_MARK_RIGHT_BOTTOM].height = m_BlobMark[nMaxIndex].rectBox.height;

		}
		else
		{
			if (m_BlobMark[nMaxIndex].fDiffGV > m_BlobMark[nSecondIndex].fDiffGV)
			{
				int offset = 50;
				m_BlobMark[nMaxIndex].rectBox.x -= offset;
				m_BlobMark[nMaxIndex].rectBox.y -= offset;
				m_BlobMark[nMaxIndex].rectBox.width += 2 * offset;
				m_BlobMark[nMaxIndex].rectBox.height += 2 * offset;

				if (m_BlobMark[nMaxIndex].rectBox.x <= 0) m_BlobMark[nMaxIndex].rectBox.x = 0;
				if (m_BlobMark[nMaxIndex].rectBox.y <= 0) m_BlobMark[nMaxIndex].rectBox.y = 0;

				//将Mark值赋值给Align参数中
				rcMarkROI[E_MARK_RIGHT_BOTTOM].x = recMark.x + m_BlobMark[nMaxIndex].rectBox.x;
				rcMarkROI[E_MARK_RIGHT_BOTTOM].y = recMark.y + m_BlobMark[nMaxIndex].rectBox.y;
				rcMarkROI[E_MARK_RIGHT_BOTTOM].width = m_BlobMark[nMaxIndex].rectBox.width;
				rcMarkROI[E_MARK_RIGHT_BOTTOM].height = m_BlobMark[nMaxIndex].rectBox.height;
			}
			else
			{
				int offset = 50;
				m_BlobMark[nSecondIndex].rectBox.x -= offset;
				m_BlobMark[nSecondIndex].rectBox.y -= offset;
				m_BlobMark[nSecondIndex].rectBox.width += 2 * offset;
				m_BlobMark[nSecondIndex].rectBox.height += 2 * offset;

				if (m_BlobMark[nSecondIndex].rectBox.x <= 0) m_BlobMark[nSecondIndex].rectBox.x = 0;
				if (m_BlobMark[nSecondIndex].rectBox.y <= 0) m_BlobMark[nSecondIndex].rectBox.y = 0;

				//将Mark值赋值给Align参数中
				rcMarkROI[E_MARK_RIGHT_BOTTOM].x = recMark.x + m_BlobMark[nSecondIndex].rectBox.x;
				rcMarkROI[E_MARK_RIGHT_BOTTOM].y = recMark.y + m_BlobMark[nSecondIndex].rectBox.y;
				rcMarkROI[E_MARK_RIGHT_BOTTOM].width = m_BlobMark[nSecondIndex].rectBox.width;
				rcMarkROI[E_MARK_RIGHT_BOTTOM].height = m_BlobMark[nSecondIndex].rectBox.height;
			}
		}

	}
	else
	{
		int nMaxIndex = 0;
		int offset = 50;
		m_BlobMark[nMaxIndex].rectBox.x -= offset;
		m_BlobMark[nMaxIndex].rectBox.y -= offset;
		m_BlobMark[nMaxIndex].rectBox.width += 2 * offset;
		m_BlobMark[nMaxIndex].rectBox.height += 2 * offset;

		if (m_BlobMark[nMaxIndex].rectBox.x <= 0) m_BlobMark[nMaxIndex].rectBox.x = 0;
		if (m_BlobMark[nMaxIndex].rectBox.y <= 0) m_BlobMark[nMaxIndex].rectBox.y = 0;



		//将Mark值赋值给Align参数中
		rcMarkROI[E_MARK_RIGHT_BOTTOM].x = recMark.x + m_BlobMark[nMaxIndex].rectBox.x;
		rcMarkROI[E_MARK_RIGHT_BOTTOM].y = recMark.y + m_BlobMark[nMaxIndex].rectBox.y;
		rcMarkROI[E_MARK_RIGHT_BOTTOM].width = m_BlobMark[nMaxIndex].rectBox.width;
		rcMarkROI[E_MARK_RIGHT_BOTTOM].height = m_BlobMark[nMaxIndex].rectBox.height;
	}

	return nErrorCode;
}

//检查AVI AD GV(8bit和12bit)
long CInspectAlign::DoFindDefectAD_GV(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner, CDefectCCD* cCCD)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果参数为NULL。
	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	double	dADGVMin = dPara[E_PARA_CHECK_MIN_GV];
	double	dADGVMax = dPara[E_PARA_CHECK_MAX_GV];
	double	dADGVAVG = dPara[E_PARA_CHECK_AVG_GV];
	double	dADGVLAB = dPara[E_PARA_CHECK_PATTERN_LABEL];

	bool	bAreaFlag = (dPara[E_PARA_CHECK_AREA_FLAG] > 0) ? true : false;
	int		nAreaGV = (int)dPara[E_PARA_CHECK_AREA_GV];
	double	dAreaRatio = dPara[E_PARA_CHECK_AREA_RATIO];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////

		//禁用时退出
		//if( dUse <= 0 )		return E_ERROR_CODE_TRUE;	// 已禁用

		//临时缩小范围
	int nOffset = 5;
	CRect rectROI = new CRect(
		max(ptCorner[E_CORNER_LEFT_TOP].x + nOffset, ptCorner[E_CORNER_LEFT_BOTTOM].x + nOffset),
		max(ptCorner[E_CORNER_LEFT_TOP].y + nOffset, ptCorner[E_CORNER_RIGHT_TOP].y + nOffset),
		min(ptCorner[E_CORNER_RIGHT_TOP].x - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].x - nOffset),
		min(ptCorner[E_CORNER_LEFT_BOTTOM].y - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].y - nOffset));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ALIGN_DISPLAY;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_DISPLAY;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_DISPLAY;

	int nROIWidth = rectROI.Width();
	int nROIHeight = rectROI.Height();

	//创建Image Buff
	cv::Mat matSrcCopyBuf = cMem->GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
	cv::Mat matSubBuf = cMem->GetMat(nROIHeight, nROIWidth, matSrcBuf.type(), false);

	//获取画面信息
	matSrcBuf.copyTo(matSrcCopyBuf);

	cv::Mat matMeanBuf = matSrcCopyBuf(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	//求相等区域的平均值
	cv::Scalar m = cv::mean(matMeanBuf);

	//平均结果
	dResult[0] = m[0];

	//亮度较亮时,显示或更高
	if (dResult[0] > dADGVMax)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较暗时,显示异常
	if (dResult[0] < dADGVMin)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//////////////////////////////////////////////////////////////////////////

		//如果在Black模式下点亮,Point算法需要太长时间
		//将AD排除在黑色模式按已点亮区域比率
		//18.03.14修改PNZ Black Pattern对等区域AD Check
		//现有:一定GV以上Pixel数量Count
		//现象:1GV左右前Pattern有带状炎
		//修改:正等影像2进化获得膨胀和占据的区域,以当前总区域15%为基准设置
	if (bAreaFlag)
	{
		__int64 nCount = 0;

		//对点灯画面求Mean,StdDev,对没有Line,点灯的画面进行统计
		cv::Scalar m, s;
		cv::meanStdDev(matMeanBuf, m, s);

		double dMeanGV = m[0];
		double dStdGV = s[0];

		__int64 nTotalLabel = 0;

		//如果是原始8U
		if ((matMeanBuf.type() == CV_8U) && (dMeanGV > 0.8 || dStdGV > 1.0))
		{
			MatIterator_<uchar> itSrc, endSrc;
			itSrc = matMeanBuf.begin<uchar>();
			endSrc = matMeanBuf.end<uchar>();

			//Pixel计数高于设置的GV
			for (; itSrc != endSrc; itSrc++)
				(*itSrc > nAreaGV) ? nCount++ : NULL;
		}

		//如果满足条件
		else if (matMeanBuf.type() == CV_8U && dMeanGV <= 0.8 && dStdGV <= 1.0)
		{
			int nPS = 2;

			if (cCCD != NULL)
			{
				//CCD不良位置校正
				//17.07.08只注册Black pattern,因此只在Black Pattern中使用
				//17.07.11直接修改原始画面
			//PS模式下有1 pixel的误差,周围再校正1 pixel
				cCCD->OffsetDefectCCD(matSrcCopyBuf, 1, nPS);

				//清除CCD不良位置
					//17.07.08只注册Black pattern,因此只在Black Pattern中使用
					//17.07.11直接修改原始画面
				//PS模式下会有1个pixel的误差,所以周围再删除1个pixel
				cCCD->DeleteDefectCCD(matSrcCopyBuf, 1, nPS);
			}

			//2进化
			cv::threshold(matMeanBuf, matSubBuf, 1.0, 255.0, THRESH_BINARY);

			//用于Labeling的Buff
			cv::Mat matLabel = cMem->GetMat(nROIHeight, nROIWidth, CV_32SC1, false);
			//cv::Mat matStats		= cMem->GetMat(nROIHeight, nROIWidth, matSrcBuf.type());
			//cv::Mat matCentroid		= cMem->GetMat(nROIHeight, nROIWidth, matSrcBuf.type());

			//nTotalLabel = cv::connectedComponentsWithStats(matMeanBuf, matLabel, matStats, matCentroid, 8, CV_32S) - 1;
			nTotalLabel = cv::connectedComponents(matSubBuf, matLabel, 8, CV_32S, CCL_GRANA) - 1;
		}
		//如果是原始16U
		else
		{
			MatIterator_<ushort> itSrc, endSrc;
			itSrc = matMeanBuf.begin<ushort>();
			endSrc = matMeanBuf.end<ushort>();

			//Pixel计数高于设置的GV
			for (; itSrc != endSrc; itSrc++)
				(*itSrc > nAreaGV) ? nCount++ : NULL;
		}

		//所占比率
		double dRatio = nCount / (double)(matMeanBuf.rows * matMeanBuf.cols);

		//如果高于设置的值
		if (dRatio > dAreaRatio)
		{
			dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
			return E_ERROR_CODE_ALIGN_DISPLAY;
		}

		//如果高于设置的值
		if (nTotalLabel > dADGVLAB)
		{
			dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
			dResult[2] = (double)nTotalLabel;
			return E_ERROR_CODE_ALIGN_DISPLAY;
		}
	}

	//////////////////////////////////////////////////////////////////////////

		//将画面更改为所需亮度
	if (dADGVAVG > 0)
		cv::multiply(matSrcBuf, dADGVAVG / dResult[0], matSrcBuf);

	matMeanBuf.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long	CInspectAlign::DoFindPGAnomal_AVI(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner)
{
	//临时跳过压接异显算法 hjf 
	return E_ERROR_CODE_TRUE;
	long	nWidth = (long)matSrcBuf.cols;
	long	nHeight = (long)matSrcBuf.rows;

	//double	dPGGVMean = dPara[E_PARA_CHECK_PG_FRINGE_MEANGV];
	//double	dPGGVArea = dPara[E_PARA_CHECK_PG_FRINGE_AREA];
	double	dPGGVMean = 100;
	double	dPGGVArea = 1500;

	int nOffset = 5;
	CRect rectROI = new CRect(
		max(ptCorner[E_CORNER_LEFT_TOP].x + nOffset, ptCorner[E_CORNER_LEFT_BOTTOM].x + nOffset),
		max(ptCorner[E_CORNER_LEFT_TOP].y + nOffset, ptCorner[E_CORNER_RIGHT_TOP].y + nOffset),
		min(ptCorner[E_CORNER_RIGHT_TOP].x - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].x - nOffset),
		min(ptCorner[E_CORNER_LEFT_BOTTOM].y - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].y - nOffset));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ALIGN_DISPLAY;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_DISPLAY;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_DISPLAY;


	int nROIWidth = rectROI.Width();
	int nROIHeight = rectROI.Height();

	cv::Mat matLeftFringeBuf = matSrcBuf(cv::Rect(rectROI.left, rectROI.top, rectROI.Width() / 20, rectROI.Height() / 3));
	cv::Mat matRightFringeBuf = matSrcBuf(cv::Rect(rectROI.left, rectROI.top / 3, rectROI.Width() / 20, rectROI.Height() / 3));
	//求左右刘海区域的平均值
	int mL = cv::mean(matLeftFringeBuf)[0];
	int mR = cv::mean(matLeftFringeBuf)[0];
	//平均结果
	dResult[0] = 0.0;

	//亮度较亮时,显示或更高
	if (abs(mL - dPGGVMean) > 20)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	cv::Mat maskL, maskR;
	cv::threshold(matLeftFringeBuf, maskL, 0, 255, cv::THRESH_BINARY);
	cv::threshold(matRightFringeBuf, maskR, 0, 255, cv::THRESH_BINARY);

	// 计算图像的均值
	cv::Scalar meanValueL = cv::mean(matLeftFringeBuf, maskL);
	cv::Scalar meanValueR = cv::mean(matRightFringeBuf, maskR);
	double thresholdValue = 20.0;  // 阈值为20

	// 寻找轮廓
	std::vector<std::vector<cv::Point>> contoursL, contoursR;
	std::vector<cv::Vec4i> hierarchyL, hierarchyR;

	cv::findContours(maskL, contoursL, hierarchyL, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(maskR, contoursR, hierarchyR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// 遍历轮廓，提取大于阈值均值或小于阈值均值的区域并计算统计面积
	int totalAreaL = 0;
	int totalAreaR = 0;
	//std::vector<std::vector<cv::Point>> selectedContoursL;
	//std::vector<std::vector<cv::Point>> selectedContoursR;

	for (size_t i = 0; i < contoursL.size(); i++) {
		double area = cv::contourArea(contoursL[i]);
		cv::Scalar meanValue = cv::mean(matLeftFringeBuf, contoursL[i]);  // 计算面积区域对应原图像素的均值
		if (meanValue[0] > meanValueL[0] + thresholdValue || meanValue[0] < meanValueL[0] - thresholdValue) {
			//selectedContoursL.push_back(contoursL[i]);
			totalAreaL += static_cast<int>(area);
		}
	}

	for (size_t i = 0; i < contoursR.size(); i++) {
		double area = cv::contourArea(contoursR[i]);
		cv::Scalar meanValue = cv::mean(matRightFringeBuf, contoursR[i]);  // 计算面积区域对应原图像素的均值
		if (meanValue[0] > meanValueR[0] + thresholdValue || meanValue[0] < meanValueR[0] - thresholdValue) {
			//selectedContoursR.push_back(contoursR[i]);
			totalAreaR += static_cast<int>(area);
		}
	}

	// 在原图上绘制提取的区域
	//cv::Mat resultL = cv::Mat::zeros(matLeftFringeBuf.size(), CV_8UC1);
	//cv::Mat resultR = cv::Mat::zeros(matRightFringeBuf.size(), CV_8UC1);

	//cv::drawContours(resultL, selectedContoursL, -1, cv::Scalar(255), cv::FILLED);
	//cv::drawContours(resultR, selectedContoursR, -1, cv::Scalar(255), cv::FILLED);


	// 判断总面积是否大于600个像素，并打印该值
	int totalArea = totalAreaL + totalAreaR;
	if (totalArea > dPGGVArea) {
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	return E_ERROR_CODE_TRUE;
}

//LJH Dust 4区域平均
long CInspectAlign::DoFindDefectAD_GV_DUST(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果参数为NULL。
	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	double	dADGVMin = dPara[E_PARA_CHECK_MIN_GV];
	double	dADGVMax = dPara[E_PARA_CHECK_MAX_GV];
	double	dADGVAVG = dPara[E_PARA_CHECK_AVG_GV];

	bool	bAreaFlag = (dPara[E_PARA_CHECK_AREA_FLAG] > 0) ? true : false;
	int		nAreaGV = (int)dPara[E_PARA_CHECK_AREA_GV];
	double	dAreaRatio = dPara[E_PARA_CHECK_AREA_RATIO];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////

		//禁用时退出
		//if( dUse <= 0 )		return E_ERROR_CODE_TRUE;	// 已禁用

		//临时缩小范围
	int nOffset = 5;
	CRect rectROI = new CRect(
		max(ptCorner[E_CORNER_LEFT_TOP].x + nOffset, ptCorner[E_CORNER_LEFT_BOTTOM].x + nOffset),
		max(ptCorner[E_CORNER_LEFT_TOP].y + nOffset, ptCorner[E_CORNER_RIGHT_TOP].y + nOffset),
		min(ptCorner[E_CORNER_RIGHT_TOP].x - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].x - nOffset),
		min(ptCorner[E_CORNER_LEFT_BOTTOM].y - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].y - nOffset));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ALIGN_DISPLAY;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_DISPLAY;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_DISPLAY;

	//求LJH点等面积的四等分平均值
	int halfWidth = rectROI.Width() / 2;
	int halfHeight = rectROI.Height() / 2;

	cv::Mat matMeanBuf1 = matSrcBuf(cv::Rect(rectROI.left, rectROI.top, halfWidth, halfHeight)); //左上角区域
	cv::Mat matMeanBuf2 = matSrcBuf(cv::Rect(rectROI.left + halfWidth, rectROI.top, halfWidth, halfHeight)); //右上角区域
	cv::Mat matMeanBuf3 = matSrcBuf(cv::Rect(rectROI.left, rectROI.top + halfHeight, halfWidth, halfHeight)); //左下角区域
	cv::Mat matMeanBuf4 = matSrcBuf(cv::Rect(rectROI.left + halfWidth, rectROI.top + halfHeight, halfWidth, halfHeight)); //右下角区域

	cv::Scalar m1 = cv::mean(matMeanBuf1);	//左上角区域平均
	cv::Scalar m2 = cv::mean(matMeanBuf2);	//右上角区域平均
	cv::Scalar m3 = cv::mean(matMeanBuf3);	//左下角区域平均
	cv::Scalar m4 = cv::mean(matMeanBuf4);	//右下角区域平均

	//获得平均分区后,释放缓冲区
	matMeanBuf1.release();
	matMeanBuf2.release();
	matMeanBuf3.release();
	matMeanBuf4.release();

	//亮度较亮时,显示或更高
	if (m1[0] > dADGVMax)
	{
		dResult[0] = m1[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	else if (m2[0] > dADGVMax)
	{
		dResult[0] = m2[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	else if (m3[0] > dADGVMax)
	{
		dResult[0] = m3[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	else if (m4[0] > dADGVMax)
	{
		dResult[0] = m4[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较暗时,显示异常
	if (m1[0] < dADGVMin)
	{
		dResult[0] = m1[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	else if (m2[0] < dADGVMin)
	{
		dResult[0] = m2[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	else if (m3[0] < dADGVMin)
	{
		dResult[0] = m3[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}
	else if (m4[0] < dADGVMin)
	{
		dResult[0] = m4[0];
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//求平均值
	//LJH在每个4等分区域亮度不超过时运行整个区域的平均逻辑
	cv::Mat matMeanBuf = matSrcBuf(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	cv::Scalar m = cv::mean(matMeanBuf);

	//平均结果
	dResult[0] = m[0];

	//亮度较亮时,显示或更高
	if (dResult[0] > dADGVMax)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较暗时,显示异常
	if (dResult[0] < dADGVMin)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//////////////////////////////////////////////////////////////////////////

		//如果在Black模式下点亮,Point算法需要太长时间
		//将AD排除在黑色模式按已点亮区域比率
	if (bAreaFlag)
	{
		__int64 nCount = 0;

		//如果是原始8U
		if (matMeanBuf.type() == CV_8U)
		{
			MatIterator_<uchar> itSrc, endSrc;
			itSrc = matMeanBuf.begin<uchar>();
			endSrc = matMeanBuf.end<uchar>();

			//Pixel计数高于设置的GV
			for (; itSrc != endSrc; itSrc++)
				(*itSrc > nAreaGV) ? nCount++ : NULL;
		}
		//如果是原始16U
		else
		{
			MatIterator_<ushort> itSrc, endSrc;
			itSrc = matMeanBuf.begin<ushort>();
			endSrc = matMeanBuf.end<ushort>();

			//Pixel计数高于设置的GV
			for (; itSrc != endSrc; itSrc++)
				(*itSrc > nAreaGV) ? nCount++ : NULL;
		}

		//所占比率
		double dRatio = nCount / (double)(matMeanBuf.rows * matMeanBuf.cols);

		//如果高于设置的值
		if (dRatio > dAreaRatio)
		{
			dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
			return E_ERROR_CODE_ALIGN_DISPLAY;
		}
	}

	//////////////////////////////////////////////////////////////////////////

		//将画面更改为所需亮度
	if (dADGVAVG > 0)
		cv::multiply(matSrcBuf, dADGVAVG / dResult[0], matSrcBuf);

	matMeanBuf.release();

	return E_ERROR_CODE_TRUE;
}

//SVI AD GV检查
long CInspectAlign::DoFindDefectAD_GV_SVI(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//如果参数为NULL。
	if (dPara == NULL)			return E_ERROR_CODE_ALIGN_WARNING_PARA;

	//////////////////////////////////////////////////////////////////////////
		//参数
	//////////////////////////////////////////////////////////////////////////

	double	dADGVMin = dPara[E_PARA_SVI_CHECK_MIN_GV];
	double	dADGVMax = dPara[E_PARA_SVI_CHECK_MAX_GV];

	long	nWidth = (long)matSrcBuf.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuf.rows;	// 图像垂直尺寸

	//////////////////////////////////////////////////////////////////////////

		//临时缩小范围
	int nOffset = 5;
	CRect rectROI = new CRect(
		max(ptCorner[E_CORNER_LEFT_TOP].x + nOffset, ptCorner[E_CORNER_LEFT_BOTTOM].x + nOffset),
		max(ptCorner[E_CORNER_LEFT_TOP].y + nOffset, ptCorner[E_CORNER_RIGHT_TOP].y + nOffset),
		min(ptCorner[E_CORNER_RIGHT_TOP].x - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].x - nOffset),
		min(ptCorner[E_CORNER_LEFT_BOTTOM].y - nOffset, ptCorner[E_CORNER_RIGHT_BOTTOM].y - nOffset));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ALIGN_DISPLAY;

	if (rectROI.left >= rectROI.right)	return E_ERROR_CODE_ALIGN_DISPLAY;
	if (rectROI.top >= rectROI.bottom)	return E_ERROR_CODE_ALIGN_DISPLAY;

	//求平均值
	cv::Mat matMeanBuf = matSrcBuf(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	cv::Scalar m = cv::mean(matMeanBuf);
	matMeanBuf.release();

	//平均结果
	dResult[0] = (m[0] + m[1] + m[2]) / 3;

	//亮度较亮时,显示或更高
	if (dResult[0] > dADGVMax)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_BRIGHT;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	//亮度较暗时,显示异常
	if (dResult[0] < dADGVMin)
	{
		dResult[1] = E_DEFECT_JUDGEMENT_DISPLAY_DARK;
		return E_ERROR_CODE_ALIGN_DISPLAY;
	}

	return E_ERROR_CODE_TRUE;
}

//查找Cell区域
long CInspectAlign::FindCellArea(cv::Mat matThreshBuf, int nMinArea, cv::Rect& rectCell)
{
	//如果没有缓冲区。
	if (matThreshBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matThreshBuf.size(), CV_32SC1);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matThreshBuf, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

	//大于一定面积的Blob数
	int nLabelCount = 0;

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
		if (nArea <= nMinArea)
		{
			//初始化为0GV后,跳过
			cv::Mat matTempROI = matThreshBuf(rectTemp);
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

		//Cell区域Rect
		rectCell = rectTemp;

		//大于一定面积的Blob数
		nLabelCount++;
	}

	//取消分配内存
	matLabel.release();
	matStats.release();
	matCentroid.release();

	//如果数量不是1
	if (nLabelCount != 1)	return E_ERROR_CODE_ALIGN_NO_FIND_CELL;

	//18.04.11-改变方式
/************************************************************************
	//如果没有缓冲区。
if( matThreshBuf.empty() )		return E_ERROR_CODE_EMPTY_BUFFER;

CMatBuf cMatBufTemp;

	//缓冲区分配和初始化
cMatBufTemp.SetMem(cMem);
cv::Mat matLabelBuffer = cMatBufTemp.GetMat(matThreshBuf.size(), CV_32SC1);

matThreshBuf.convertTo(matLabelBuffer, CV_32SC1);
matThreshBuf.setTo(0);

// starts at 2 because 0,1 are used already
int nLabelCount = 2;

	//检查区域Rect
cv::Rect rectTemp;
int nTempCount = 0;

for(int y=0 ; y<matLabelBuffer.rows ; y++)
{
	int *row = (int*)matLabelBuffer.ptr(y);

	for(int x=0 ; x<matLabelBuffer.cols ; x++)
	{
		if(row[x] != 255)	continue;

		nTempCount++;

					//对象面积
		long nArea = cv::floodFill(matLabelBuffer, cv::Point(x, y), nTempCount, &rectTemp);

					//面积限制
		if( nArea <  nMinArea )	continue;

					//Cell区域Rect
		rectCell = rectTemp;

		int nEndXX = rectTemp.x+rectTemp.width;
		int nEndYY = rectTemp.y+rectTemp.height;

					//异常处理
		if( nEndYY >= matLabelBuffer.rows )	nEndYY = matLabelBuffer.rows - 1;
		if( nEndXX >= matLabelBuffer.cols )	nEndXX = matLabelBuffer.cols - 1;

					//每个标签的像素坐标
		for(int j=rectTemp.y ; j<=nEndYY ; j++)
		{
			int		*row2 = (int *)matLabelBuffer.ptr(j);
			BYTE	*row3 = (BYTE *)matThreshBuf.ptr(j);

			for(int i=rectTemp.x ; i<=nEndXX ; i++)
			{
				if(row2[i] != nTempCount)	continue;

				row3[i] = (BYTE)255;
			}
		}

		nLabelCount++;
	}
}

matLabelBuffer.release();

	//如果数量不是1个。(从nLabelCount2开始)
if( nLabelCount != 3 )	return E_ERROR_CODE_ALIGN_NO_FIND_CELL;
************************************************************************/

	return E_ERROR_CODE_TRUE;
}

//按方向获取数据
long CInspectAlign::RobustFitLine(cv::Mat& matTempBuf, cv::Rect rectCell, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType, int nSamp)
{
	//画面大小
	int nW = matTempBuf.cols;
	int nH = matTempBuf.rows;

	//异常处理
	if (rectCell.x < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectCell.x >= nW)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectCell.y < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectCell.y >= nH)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//异常处理
	if (rectCell.x + rectCell.width < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectCell.x + rectCell.width >= nW)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectCell.y + rectCell.height < 0)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
	if (rectCell.y + rectCell.height >= nH)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

	//设置范围
	int nSrartX = rectCell.x + rectCell.width / 4;	//  宽1/4点
	int nEndX = nSrartX + rectCell.width / 2;		//  水平3/4点

	int nStartY = rectCell.y + rectCell.height / 4;	//  垂直1/4点
	int nEndY = nStartY + rectCell.height / 2;	//  垂直3/4点

	int x, y;

	vector<cv::Point2i>	ptSrcIndexs;
	vector<cv::Point2i>().swap(ptSrcIndexs);

	//提取行数据向量
	switch (nType)
	{
	case E_ALIGN_TYPE_LEFT:
	{
		for (y = nStartY; y <= nEndY; y += nSamp)
		{
			for (x = rectCell.x; x <= nEndX; x++)
			{
				//如果有值
				if (matTempBuf.at<uchar>(y, x))
				{
					//添加坐标向量
					ptSrcIndexs.push_back(cv::Point2i(x, y));

					break;
				}
			}
		}
	}
	break;

	case E_ALIGN_TYPE_TOP:
	{
		for (x = nSrartX; x <= nEndX; x += nSamp)
		{
			for (y = rectCell.y; y <= nEndY; y++)
			{
				//如果有值
				if (matTempBuf.at<uchar>(y, x))
				{
					//添加坐标向量
					ptSrcIndexs.push_back(cv::Point2i(x, y));

					break;
				}
			}
		}
	}
	break;

	case E_ALIGN_TYPE_RIGHT:
	{
		for (y = nStartY; y <= nEndY; y += nSamp)
		{
			for (x = rectCell.x + rectCell.width; x >= nSrartX; x--)
			{
				//如果有值
				if (matTempBuf.at<uchar>(y, x))
				{
					//添加坐标向量
					ptSrcIndexs.push_back(cv::Point2i(x, y));

					break;
				}
			}
		}
	}
	break;

	case E_ALIGN_TYPE_BOTTOM:
	{
		for (x = nSrartX; x <= nEndX; x += nSamp)
		{
			for (y = rectCell.y + rectCell.height; y >= nStartY; y--)
			{
				//如果有值
				if (matTempBuf.at<uchar>(y, x))
				{
					//添加坐标向量
					ptSrcIndexs.push_back(cv::Point2i(x, y));

					break;
				}
			}
		}
	}
	break;

	//如果参数高于
	default:
	{
		return E_ERROR_CODE_ALIGN_WARNING_PARA;
	}
	break;
	}

	//使用提取的数据查找行
	long nErrorCode = AlgoBase::calcRANSAC(ptSrcIndexs, dA, dB, nMinSamples, distThreshold);

	//删除矢量数据
	vector<cv::Point2i>().swap(ptSrcIndexs);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	return E_ERROR_CODE_TRUE;
}

//在二进制画面中从物体的内部到外部导航坐标
long CInspectAlign::ObjectInAreaGetLine(cv::Mat& matTempBuf, cv::Rect rectImgSize, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType)
{
	//画面大小
	int nW = matTempBuf.cols;
	int nH = matTempBuf.rows;
	uchar* ucImagedata = matTempBuf.data;

	//设置范围
	int nStartX = 0;								//初始化
	int nEndX = 0;
	int nStartY = 0;
	int nEndY = 0;

	vector<cv::Point2i>	ptSrcIndexs;//坐标获取部
	vector<cv::Point2i>().swap(ptSrcIndexs);
	int nX = 0, nY = 0, nSample = 50;
	int nStPtOffset = 100;			//在Panel的内侧搜索外侧时,从外侧到内侧起始点的Offset值
	uchar* p = 0;						//画面Data Pointer

	switch (nType)
	{
	case E_ALIGN_TYPE_LEFT:
		nStartX = rectImgSize.width / 2;							//画面的水平中央
		nEndX = 0;											//画面的横向起点
		nStartY = rectImgSize.height / 3;							//画面的垂直1/3点
		nEndY = rectImgSize.height * 2 / 3;						//面板的垂直3/4点

		for (nY = nStartY; nY < nEndY; nY += nSample)
		{
			p = ucImagedata + (nW * nY);
			for (nX = nStartX; nX > nEndX; nX--)
			{
				if ((int)*(p + nX) == 0)
				{
					ptSrcIndexs.push_back(cv::Point2i(nX, nY));
					break;
				}
			}
		}
		break;
	case E_ALIGN_TYPE_TOP:
		nStartX = rectImgSize.width / 3;							//画面的三分之一宽
		nEndX = rectImgSize.width * 2 / 3;						//画面的2/3宽
		nStartY = rectImgSize.height / 2;							//图像的垂直中心
		nEndY = 0;											//画面的垂直起点

		for (nX = nStartX; nX < nEndX; nX += nSample)
		{
			p = ucImagedata + nX;
			for (nY = nStartY; nY > nEndY; nY--)
			{

				if ((int)*(p + nY * nW) == 0)
				{
					ptSrcIndexs.push_back(cv::Point2i(nX, nY));
					break;
				}
			}

		}
		break;
	case E_ALIGN_TYPE_RIGHT:
		nStartX = rectImgSize.width / 2;						//画面的水平中央
		nEndX = matTempBuf.cols;							//画面的水平端点
		nStartY = rectImgSize.height / 3;						//画面的垂直1/3点
		nEndY = rectImgSize.height / 3 * 2;					//画面的垂直2/3点

		for (nY = nStartY; nY < nEndY; nY += nSample)
		{
			p = ucImagedata + (nW * nY);
			for (nX = nStartX; nX < nEndX; nX++)
			{
				if ((int)*(p + nX) == 0)
				{
					ptSrcIndexs.push_back(cv::Point2i(nX, nY));
					break;
				}
			}
		}
		break;
	case E_ALIGN_TYPE_BOTTOM:
		nStartX = rectImgSize.width / 3;							//画面的三分之一宽
		nEndX = rectImgSize.width / 3 * 2;						//画面的2/3宽
		nStartY = rectImgSize.height / 2;							//图像的垂直中心
		nEndY = matTempBuf.rows;								//画面的垂直端点

		for (nX = nStartX; nX < nEndX; nX += nSample)
		{
			p = ucImagedata + nX;
			for (nY = nStartY; nY < nEndY; nY++)
			{
				if ((int)*(p + nY * nW) == 0)
				{
					ptSrcIndexs.push_back(cv::Point2i(nX, nY));
					break;
				}
			}
		}
		break;

	default:
		return E_ERROR_CODE_ALIGN_WARNING_PARA;
		break;
	}

	//使用提取的数据查找行
	long nErrorCode = AlgoBase::calcRANSAC(ptSrcIndexs, dA, dB, nMinSamples, distThreshold);

	//删除矢量数据
	vector<cv::Point2i>().swap(ptSrcIndexs);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	return E_ERROR_CODE_TRUE;
}

//返回最大Blob的APP
long CInspectAlign::FindBiggestBlob_APP(cv::Mat& src, cv::Mat& dst)
{
	try
	{
		Mat matSrc;
		src.copyTo(matSrc);

		dst = Mat::zeros(matSrc.size(), matSrc.type());

		double dLargest_Area = 0;
		int nLargest_Contour_Index = 0;

		vector<vector<cv::Point> > vContours; // Vector for storing contour
		vector<cv::Vec4i> vHierarchy;

		findContours(matSrc, vContours, vHierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("findContours END."));

		cv::Rect rtArea = cv::Rect(0, 0, 0, 0);
		for (int i = 0; i < (int)vContours.size(); i++)
		{
			// iterate through each contour. 
			double dArea = contourArea(vContours[i], false);		// Find the area of contour
			if (dArea > dLargest_Area)
			{
				dLargest_Area = (int)dArea;
				nLargest_Contour_Index = i;							// Store the index of largest contour

				rtArea = boundingRect(vContours[i]);				// Find the bounding rectangle for biggest contour
			}
		}

		drawContours(dst, vContours, nLargest_Contour_Index, Scalar(255), CV_FILLED, 8, vHierarchy); // Draw the largest contour using previously stored index.
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("drawContours END."));
	}
	catch (const std::exception&)
	{
		writeInspectLog(E_ALG_TYPE_COMMON_ALIGN, __FUNCTION__, _T("ERROR."));

		return E_ERROR_CODE_FALSE;
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::ObjectOutAreaGetLine(cv::Mat& matTempBuf, cv::Rect rectCell, long double& dA, long double& dB, int nMinSamples, double distThreshold, int nType, int nThreshold_Theta, float fAvgOffset)
{
	//画面大小
	int nW = matTempBuf.cols;
	int nH = matTempBuf.rows;

	//异常处理
// 	if( rectCell.x <	0	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
// 	if( rectCell.x >=	nW	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
// 	if( rectCell.y <	0	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
// 	if( rectCell.y >=	nH	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

		//异常处理
// 	if( rectCell.x + rectCell.width <	0	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
// 	if( rectCell.x + rectCell.width >=	nW	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
// 	if( rectCell.y + rectCell.height <	0	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
// 	if( rectCell.y + rectCell.height >=	nH	)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

		//设置范围
	int nStartX = rectCell.x + rectCell.width / 4;	//  宽1/4点
	int nEndX = nStartX + rectCell.width / 4 * 2;		//  水平3/4点

	int nStartY = rectCell.y + rectCell.height / 4;	//  垂直1/4点
	int nEndY = nStartY + rectCell.height / 4 * 2;	//  垂直3/4点

	int nSamp = 50;								//  每50 Pixel取样

	uchar* ucImgData;

	int nGV_Level, nGV_Level2;

	vector<cv::Point> ptProfile;
	vector<cv::Point> pt;
	long double dA_, dB_;

	double dX, dY;

	float fSum;
	float fAvg;
	int	  nCnt;

	float fTheta;
	int nDeltaOffset = 3;

	switch (nType)
	{
	case E_ALIGN_TYPE_LEFT:
		vector<cv::Point>().swap(ptProfile);
		//LEFT
		for (int nY = nStartY; nY < nEndY; nY += nSamp)
		{
			ucImgData = matTempBuf.data + nY * matTempBuf.step;
			for (int nX = 0; nX < nEndX; nX++)
			{
				nGV_Level = *(ucImgData + nX);
				int nDeltaX = nX + nDeltaOffset;
				if (nDeltaX > matTempBuf.cols - 1)
					nDeltaX = matTempBuf.cols - 1;

				nGV_Level2 = *(ucImgData + nDeltaX);

				fTheta = atan((float)(nGV_Level2 - nGV_Level) / nDeltaOffset) * 180 / PI;

				if (90 > fTheta && fTheta > nThreshold_Theta)
				{
					nCnt = 0;
					fSum = 0;
					for (int nX2 = nDeltaX - 3; nX2 <= nDeltaX + 2; nX2++)
					{
						nGV_Level = *(ucImgData + nX2);
						fSum += nGV_Level;
						nCnt++;
						ptProfile.push_back(cv::Point(nX2, nGV_Level));
					}
					AlgoBase::calcLineFit(ptProfile, dA_, dB_);
					if (dA_ == 0)
						break;
					fAvg = fSum / nCnt * fAvgOffset;
					dX = (fAvg - dB_) / dA_;

					pt.push_back(cv::Point2f(dX, nY));
					vector<cv::Point>().swap(ptProfile);
					break;
				}
			}
		}
		break;
	case E_ALIGN_TYPE_TOP:
		vector<cv::Point>().swap(ptProfile);
		//TOP
		for (int nX = nStartX; nX < nEndX; nX += nSamp)
		{
			ucImgData = matTempBuf.data + nX;
			for (int nY = 0; nY < nEndY; nY++)
			{
				nGV_Level = *(ucImgData + nY * matTempBuf.step);
				int nDeltaY = nY + nDeltaOffset;
				if (nDeltaY > matTempBuf.rows - 1)
					nDeltaY = matTempBuf.rows - 1;
				nGV_Level2 = *(ucImgData + nDeltaY * matTempBuf.step);

				fTheta = atan((float)(nGV_Level2 - nGV_Level) / nDeltaOffset) * 180 / PI;

				if (90 > fTheta && fTheta > nThreshold_Theta)
				{
					fSum = 0;
					nCnt = 0;
					for (int nY2 = nDeltaY - 2; nY2 <= nDeltaY + 3; nY2++)
					{
						nGV_Level = *(ucImgData + nY2 * matTempBuf.step);
						fSum += nGV_Level;
						nCnt++;
						ptProfile.push_back(cv::Point(nY2, nGV_Level));
					}
					if (AlgoBase::calcLineFit(ptProfile, dA_, dB_) != 0)
						break;

					if (dA_ == 0)
						break;
					fAvg = fSum / nCnt * fAvgOffset;
					dY = (fAvg - dB_) / dA_;
					pt.push_back(cv::Point2f(nX, dY));
					vector<cv::Point>().swap(ptProfile);
					break;
				}
			}
		}
		break;
	case E_ALIGN_TYPE_RIGHT:
		vector<cv::Point>().swap(ptProfile);
		//Right
		for (int nY = nStartY; nY < nEndY; nY += nSamp)
		{
			ucImgData = matTempBuf.data + nY * matTempBuf.step;
			for (int nX = matTempBuf.cols - 1; nX > nStartX; nX--)
			{
				nGV_Level = *(ucImgData + nX);
				int nDeltaX = nX - nDeltaOffset;
				if (nDeltaX < 0)
					nDeltaX = 0;
				nGV_Level2 = *(ucImgData + nDeltaX);
				fTheta = atan((float)(nGV_Level2 - nGV_Level) / nDeltaOffset) * 180 / PI;

				if (90 > fTheta && fTheta > nThreshold_Theta)
				{
					fSum = 0;
					nCnt = 0;
					for (int nX2 = nDeltaX - 2; nX2 <= nDeltaX + 3; nX2++)
					{
						nGV_Level = *(ucImgData + nX2);
						fSum += nGV_Level;
						nCnt++;
						ptProfile.push_back(cv::Point(nX2, nGV_Level));
					}
					if (AlgoBase::calcLineFit(ptProfile, dA_, dB_) != 0)
						break;

					if (dA_ == 0)
						break;
					fAvg = fSum / nCnt * fAvgOffset;
					dX = (fAvg - dB_) / dA_;
					pt.push_back(cv::Point2f(dX, nY));
					vector<cv::Point>().swap(ptProfile);
					break;

				}
			}
		}
		break;
	case E_ALIGN_TYPE_BOTTOM:
		vector<cv::Point>().swap(ptProfile);
		//BOTTOM
		for (int nX = nStartX; nX < nEndX; nX += 50)
		{
			ucImgData = matTempBuf.data + nX;
			for (int nY = matTempBuf.rows - 1; nY > nStartY; nY--)
			{
				nGV_Level = *(ucImgData + nY * matTempBuf.step);
				int nDeltaY = nY - nDeltaOffset;
				if (nDeltaY < 0)
					nDeltaY = 0;
				nGV_Level2 = *(ucImgData + nDeltaY * matTempBuf.step);
				fTheta = atan((float)(nGV_Level2 - nGV_Level) / nDeltaOffset) * 180 / PI;
				if (90 > fTheta && fTheta > nThreshold_Theta)
				{
					fSum = 0;
					nCnt = 0;
					for (int nY2 = nDeltaY - 3; nY2 <= nDeltaY + 2; nY2++)
					{
						nGV_Level = *(ucImgData + nY2 * matTempBuf.step);
						fSum += nGV_Level;
						nCnt++;
						ptProfile.push_back(cv::Point(nY2, nGV_Level));
					}
					if (AlgoBase::calcLineFit(ptProfile, dA_, dB_) != 0)
						break;

					if (dA_ == 0)
						break;
					fAvg = fSum / nCnt * fAvgOffset;
					dY = (fAvg - dB_) / dA_;
					pt.push_back(cv::Point2f(nX, dY));

					vector<cv::Point>().swap(ptProfile);
					break;
				}
			}
		}
		break;
	}

	//使用提取的数据查找行
	long nErrorCode = AlgoBase::calcRANSAC(pt, dA, dB, nMinSamples, distThreshold);

	//删除矢量数据
	vector<cv::Point2i>().swap(pt);

	//如果有错误,则输出错误代码
	if (nErrorCode != E_ERROR_CODE_TRUE)
		return nErrorCode;

	return E_ERROR_CODE_TRUE;
}

//查找4个转角位置
long CInspectAlign::FindCornerPoint(cv::Point2f ptSrc[E_CORNER_END], cv::Point ptDst[E_CORNER_END], long nWidth, long nHeight)
{
	int nHx = nWidth / 2;
	int nHy = nHeight / 2;

	for (int i = 0; i < E_CORNER_END; i++)
	{
		if (ptSrc[i].x <= 0)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;
		if (ptSrc[i].y <= 0)		return E_ERROR_CODE_ALIGN_IMAGE_OVER;
		if (ptSrc[i].x >= nWidth)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;
		if (ptSrc[i].y >= nHeight)	return E_ERROR_CODE_ALIGN_IMAGE_OVER;

		if (ptSrc[i].x < nHx &&
			ptSrc[i].y < nHy)
		{
			ptDst[E_CORNER_LEFT_TOP].x = ptSrc[i].x;
			ptDst[E_CORNER_LEFT_TOP].y = ptSrc[i].y;
		}
		else if (ptSrc[i].x > nHx &&
			ptSrc[i].y > nHy)
		{
			ptDst[E_CORNER_RIGHT_BOTTOM].x = ptSrc[i].x;
			ptDst[E_CORNER_RIGHT_BOTTOM].y = ptSrc[i].y;
		}
		else if (ptSrc[i].x > nHx &&
			ptSrc[i].y < nHy)
		{
			ptDst[E_CORNER_RIGHT_TOP].x = ptSrc[i].x;
			ptDst[E_CORNER_RIGHT_TOP].y = ptSrc[i].y;
		}
		else
		{
			ptDst[E_CORNER_LEFT_BOTTOM].x = ptSrc[i].x;
			ptDst[E_CORNER_LEFT_BOTTOM].y = ptSrc[i].y;
		}
	}

	return E_ERROR_CODE_TRUE;
}

//留下点亮区域的外围部分
long CInspectAlign::FindEdgeArea(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nLength, CMatBuf* cMemSub)
{
	//如果参数为NULL
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	int nSize = nLength * 2 + 1;
	cv::Mat matTempBuf = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

	//填充外围平均值的掩码
	cv::blur(matSrcBuf, matTempBuf, cv::Size(nSize, nSize));
	cv::subtract(matSrcBuf, matTempBuf, matDstBuf);
	cv::threshold(matDstBuf, matDstBuf, 0, 255, THRESH_BINARY);

	matTempBuf.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}
	//18.03.28-更改需要很长时间
/************************************************************************
cv::Mat matDistBuf1 = cMatBufTemp.GetMat(matSrcBuf.size(), CV_32FC1);
cv::Mat matDistBuf2 = cMatBufTemp.GetMat(matSrcBuf.size(), CV_32FC1);
cv::Mat matDistBuf3 = cMatBufTemp.GetMat(matSrcBuf.size(), CV_8UC1);

cv::distanceTransform(matSrcBuf, matDistBuf1, CV_DIST_L2, 3);

cv::threshold(matDistBuf1, matDistBuf2, nLength, 255.0, THRESH_BINARY);

	//bit转换
matDistBuf2.convertTo(matDistBuf3, matDistBuf3.type());

	//保留最外围70 Pixel左右
cv::bitwise_xor(matDistBuf3, matSrcBuf, matDstBuf);

matDistBuf1.release();
matDistBuf2.release();
matDistBuf3.release();
************************************************************************/

	return E_ERROR_CODE_TRUE;
}

//留下点亮区域的外围部分
long CInspectAlign::FindEdgeArea_SVI(cv::Mat matSrcBuf, cv::Mat& matDstBuf, int nLength)
{
	//如果参数为NULL
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	int nSize = nLength * 2 + 1;
	cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(nSize, nSize), Point(nLength, nLength));

	cv::Mat matTempBuf = cv::Mat::zeros(matSrcBuf.size(), matSrcBuf.type());

	cv::morphologyEx(matSrcBuf, matTempBuf, MORPH_ERODE, StructElem);

	cv::subtract(matSrcBuf, matTempBuf, matDstBuf);

	StructElem.release();
	matTempBuf.release();

	//18.03.28-更改需要很长时间
/************************************************************************
cv::Mat matDistBuf1, matDistBuf2, matDistBuf3;

cv::distanceTransform(matSrcBuf, matDistBuf1, CV_DIST_L2, 3);

cv::threshold(matDistBuf1, matDistBuf2, nLength, 255.0, THRESH_BINARY);

	//bit转换
matDistBuf2.convertTo(matDistBuf3, CV_8UC1);

	//保留最外围
cv::bitwise_xor(matDistBuf3, matSrcBuf, matDstBuf);

matDistBuf1.release();
matDistBuf2.release();
matDistBuf3.release();
************************************************************************/

	return E_ERROR_CODE_TRUE;
}

//横向平均填充
long CInspectAlign::FillAreaMeanX(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV)
{
	//如果参数为NULL
	if (matMeanBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matEdgeBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	for (int y = 0; y < 2; y++)
	{
		long nStart_Y, nEnd_Y;

		//计算y范围
		if (y == 0)
		{
			nStart_Y = rectROI.top;
			nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nEnd_Y = rectROI.bottom;
			nStart_Y = nEnd_Y - nSegY;
		}

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (int x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			nStart_X = x * nSegX + rectROI.left;
			if (x == nRangeX - 1)		nEnd_X = rectROI.right;
			else					nEnd_X = nStart_X + nSegX;

			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = nStart_X;
			rectTemp.y = nStart_Y;
			rectTemp.width = nSegX;
			rectTemp.height = nSegY;

			//离开画面时进行异常处理
			if (rectTemp.x < 0)	rectTemp.x = 0;
			if (rectTemp.y < 0)	rectTemp.y = 0;
			if (rectTemp.x + rectTemp.width >= matMeanBuf.cols)	rectTemp.width = matMeanBuf.cols - rectTemp.x - 1;
			if (rectTemp.y + rectTemp.height >= matMeanBuf.rows)	rectTemp.height = matMeanBuf.rows - rectTemp.y - 1;

			//范围异常时,异常处理
			if (rectTemp.width <= 0)	continue;
			if (rectTemp.height <= 0)	continue;

			//画面ROI
			cv::Mat matTempROIBuf = matMeanBuf(rectTemp);

			//掩码ROI
			cv::Mat matMaskROIBuf = matEdgeBuf(rectTemp);

			//求平均值
			cv::Scalar m = cv::mean(matTempROIBuf, matMaskROIBuf);

			//仅在大于nMinGV时填充平均值
			//设置范围,平均测量时,如果不需要点亮区域,则存在(边部分)
			//有原始值,但如果用0值覆盖则存在
			if (m[0] > nMinGV)
			{
				matMaskROIBuf.setTo(255);
				matTempROIBuf.setTo((unsigned int)m[0]);
			}
			else
			{
				matMaskROIBuf.setTo(0);
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}

//横向平均填充
long CInspectAlign::FillAreaMeanX_SVI(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV)
{
	//如果参数为NULL
	if (matMeanBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matEdgeBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	for (int y = 0; y < 2; y++)
	{
		long nStart_Y, nEnd_Y;

		//计算y范围
		if (y == 0)
		{
			nStart_Y = rectROI.top;
			nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nEnd_Y = rectROI.bottom;
			nStart_Y = nEnd_Y - nSegY;
		}

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (int x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			nStart_X = x * nSegX + rectROI.left;
			if (x == nRangeX - 1)		nEnd_X = rectROI.right;
			else					nEnd_X = nStart_X + nSegX;

			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = nStart_X;
			rectTemp.y = nStart_Y;
			rectTemp.width = nSegX;
			rectTemp.height = nSegY;

			//离开画面时进行异常处理
			if (rectTemp.x < 0)	rectTemp.x = 0;
			if (rectTemp.y < 0)	rectTemp.y = 0;
			if (rectTemp.x + rectTemp.width >= matMeanBuf.cols)	rectTemp.width = matMeanBuf.cols - rectTemp.x - 1;
			if (rectTemp.y + rectTemp.height >= matMeanBuf.rows)	rectTemp.height = matMeanBuf.rows - rectTemp.y - 1;

			//范围异常时,异常处理
			if (rectTemp.width <= 0)	continue;
			if (rectTemp.height <= 0)	continue;

			//画面ROI
			cv::Mat matTempROIBuf = matMeanBuf(rectTemp);

			//掩码ROI
			cv::Mat matMaskROIBuf = matEdgeBuf(rectTemp);

			//求平均值
			cv::Scalar m = cv::mean(matTempROIBuf, matMaskROIBuf);

			//仅在大于nMinGV时填充平均值
			//设置范围,平均测量时,如果不需要点亮区域,则存在(边部分)
			//有原始值,但如果用0值覆盖则存在
			if ((m[0] + m[1] + m[2]) / 3 > nMinGV)
			{
				//平均填充
				matTempROIBuf.setTo(m);

				//填充掩码
				matMaskROIBuf.setTo(255);
			}
			else
			{
				matMaskROIBuf.setTo(0);
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}

//垂直平均填充
long CInspectAlign::FillAreaMeanY(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV)
{
	//如果参数为NULL
	if (matMeanBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matEdgeBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	for (int x = 0; x < 2; x++)
	{
		long nStart_X, nEnd_X;

		//计算x范围
		if (x == 0)
		{
			nStart_X = rectROI.left;
			nEnd_X = nStart_X + nSegX;
		}
		else
		{
			nEnd_X = rectROI.right;
			nStart_X = nEnd_X - nSegX;
		}

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (int y = 0; y < nRangeY; y++)
		{
			long nStart_Y, nEnd_Y;

			//计算y范围
			nStart_Y = y * nSegY + rectROI.top;
			if (y == nRangeY - 1)		nEnd_Y = rectROI.bottom;
			else					nEnd_Y = nStart_Y + nSegY;

			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = nStart_X;
			rectTemp.y = nStart_Y;
			rectTemp.width = nSegX;
			rectTemp.height = nSegY;

			//离开画面时进行异常处理
			if (rectTemp.x < 0)	rectTemp.x = 0;
			if (rectTemp.y < 0)	rectTemp.y = 0;
			if (rectTemp.x + rectTemp.width >= matMeanBuf.cols)	rectTemp.width = matMeanBuf.cols - rectTemp.x - 1;
			if (rectTemp.y + rectTemp.height >= matMeanBuf.rows)	rectTemp.height = matMeanBuf.rows - rectTemp.y - 1;

			//范围异常时,异常处理
			if (rectTemp.width <= 0)	continue;
			if (rectTemp.height <= 0)	continue;

			//画面ROI
			cv::Mat matTempROIBuf = matMeanBuf(rectTemp);

			//掩码ROI
			cv::Mat matMaskROIBuf = matEdgeBuf(rectTemp);

			//求平均值
			cv::Scalar m = cv::mean(matTempROIBuf, matMaskROIBuf);

			//仅在大于nMinGV时填充平均值
			//设置范围,平均测量时,如果不需要点亮区域,则存在(边部分)
			//有原始值,但如果用0值覆盖则存在
			if (m[0] > nMinGV)
			{
				//在x方向优先处理后禁用
//matMaskROIBuf.setTo( 255 );
				matTempROIBuf.setTo((unsigned int)m[0]);
			}
			//else
			//{
			//	matMaskROIBuf.setTo( 0 );
			//}
		}
	}

	return E_ERROR_CODE_TRUE;
}

//垂直平均填充
long CInspectAlign::FillAreaMeanY_SVI(cv::Mat& matMeanBuf, cv::Mat& matEdgeBuf, CRect rectROI, int nSegX, int nSegY, int nMinGV)
{
	//如果参数为NULL
	if (matMeanBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matEdgeBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	for (int x = 0; x < 2; x++)
	{
		long nStart_X, nEnd_X;

		//计算x范围
		if (x == 0)
		{
			nStart_X = rectROI.left;
			nEnd_X = nStart_X + nSegX;
		}
		else
		{
			nEnd_X = rectROI.right;
			nStart_X = nEnd_X - nSegX;
		}

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (int y = 0; y < nRangeY; y++)
		{
			long nStart_Y, nEnd_Y;

			//计算y范围
			nStart_Y = y * nSegY + rectROI.top;
			if (y == nRangeY - 1)		nEnd_Y = rectROI.bottom;
			else					nEnd_Y = nStart_Y + nSegY;

			//设置范围
			cv::Rect rectTemp;
			rectTemp.x = nStart_X;
			rectTemp.y = nStart_Y;
			rectTemp.width = nSegX;
			rectTemp.height = nSegY;

			//离开画面时进行异常处理
			if (rectTemp.x < 0)	rectTemp.x = 0;
			if (rectTemp.y < 0)	rectTemp.y = 0;
			if (rectTemp.x + rectTemp.width >= matMeanBuf.cols)	rectTemp.width = matMeanBuf.cols - rectTemp.x - 1;
			if (rectTemp.y + rectTemp.height >= matMeanBuf.rows)	rectTemp.height = matMeanBuf.rows - rectTemp.y - 1;

			//范围异常时,异常处理
			if (rectTemp.width <= 0)	continue;
			if (rectTemp.height <= 0)	continue;

			//画面ROI
			cv::Mat matTempROIBuf = matMeanBuf(rectTemp);

			//掩码ROI
			cv::Mat matMaskROIBuf = matEdgeBuf(rectTemp);

			//求平均值
			cv::Scalar m = cv::mean(matTempROIBuf, matMaskROIBuf);

			//仅在大于nMinGV时填充平均值
			//设置范围,平均测量时,如果不需要点亮区域,则存在(边部分)
			//有原始值,但如果用0值覆盖则存在
			if ((m[0] + m[1] + m[2]) / 3 > nMinGV)
			{
				//平均填充
				matTempROIBuf.setTo(m);

				//填充掩码
	//matMaskROIBuf.setTo( 255 );
			}
			//else
			//{
			//	matMaskROIBuf.setTo( 0 );
			//}
		}
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectAlign::FillMerge(cv::Mat& matSrcBuf, cv::Mat matMeanBuf, cv::Mat matMaskBuf, int nAlgImg, CMatBuf* cMemSub)
{
	//如果参数为NULL
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matMeanBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matMaskBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	int Threshold = 0;
	// 8bit
	if (matSrcBuf.type() == CV_8U)
	{
		/*	cv::bitwise_and(matSrcBuf, matMaskBuf, matSrcBuf);

		//反转
		cv::bitwise_not(matMaskBuf, matMaskBuf);

		//只留下背景平均部分
		cv::bitwise_and(matMeanBuf, matMaskBuf, matMeanBuf);

		//合并
		cv::add(matSrcBuf, matMeanBuf, matSrcBuf);
		*/

		////////////////////////////////////////////////////////////////////////// 
		if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE || nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST || nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN) {//将背光画面也加入图像增强里面 hjf

			cv::bitwise_and(matSrcBuf, matMaskBuf, matSrcBuf);

			//反转
			cv::bitwise_not(matMaskBuf, matMaskBuf);

			//只留下背景平均部分
			cv::bitwise_and(matMeanBuf, matMaskBuf, matMeanBuf);

			//合并
			cv::add(matSrcBuf, matMeanBuf, matSrcBuf);
		}

		else {

			cv::Mat Test_tmp = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

			//反转
			cv::bitwise_not(matMaskBuf, matMaskBuf);

			//只留下背景平均部分
			cv::bitwise_and(matMeanBuf, matMaskBuf, matMeanBuf);

			cv::min(matMeanBuf, matSrcBuf, Test_tmp);

			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_G) { //choi 06.04
				Threshold = 5;
			}
			else {
				Threshold = 20; //pwj 12.04.14改为10->20
			}

			cv::threshold(Test_tmp, Test_tmp, Threshold, 255.0, THRESH_BINARY);
			//cv::imwrite("E:\\IMTC\\notch\\1_threshold.bmp", Test_tmp);

			cv::subtract(matMeanBuf, Test_tmp, matMeanBuf);
			//cv::imwrite("E:\\IMTC\\notch\\2_subtract.bmp", matMeanBuf);
							//合并	
			cv::max(matSrcBuf, matMeanBuf, matSrcBuf);
			//cv::imwrite("E:\\IMTC\\notch\\3_max.bmp", matSrcBuf);

		}

	}
	// 12bit
	else
	{
		cv::Mat matTempBuf16 = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);
		matMaskBuf.convertTo(matTempBuf16, matSrcBuf.type());

		// threshold
		MatIterator_<ushort> it, end;
		for (it = matTempBuf16.begin<ushort>(), end = matTempBuf16.end<ushort>(); it != end; it++)
			*it = (*it) ? 4095 : 0;

		/*cv::bitwise_and(matSrcBuf, matTempBuf16, matSrcBuf);

		//反转
		cv::bitwise_not(matTempBuf16, matTempBuf16);

		//只留下背景平均部分
		cv::bitwise_and(matMeanBuf, matTempBuf16, matMeanBuf);

		//合并
		cv::add(matSrcBuf, matMeanBuf, matSrcBuf);*/

		////////////////////////////////////////////////////////////////////////// choikwangil 04.06 Test
		if (nAlgImg == E_IMAGE_CLASSIFY_AVI_WHITE || nAlgImg == E_IMAGE_CLASSIFY_AVI_DUST || nAlgImg == E_IMAGE_CLASSIFY_AVI_DUSTDOWN) {//将背光画面也加入图像增强里面 hjf

			cv::bitwise_and(matSrcBuf, matMaskBuf, matSrcBuf);

			//反转
			cv::bitwise_not(matMaskBuf, matMaskBuf);

			//只留下背景平均部分
			cv::bitwise_and(matMeanBuf, matMaskBuf, matMeanBuf);

			//合并
			cv::add(matSrcBuf, matMeanBuf, matSrcBuf);
		}

		else {

			cv::Mat Test_tmp = cMatBufTemp.GetMat(matSrcBuf.size(), matSrcBuf.type(), false);

			//反转
			cv::bitwise_not(matMaskBuf, matMaskBuf);

			//只留下背景平均部分
			cv::bitwise_and(matMeanBuf, matMaskBuf, matMeanBuf);

			cv::min(matMeanBuf, matSrcBuf, Test_tmp);

			if (nAlgImg == E_IMAGE_CLASSIFY_AVI_G) { //choi 06.04
				Threshold = 5;
			}
			else {
				Threshold = 20;
			}

			cv::threshold(Test_tmp, Test_tmp, Threshold, 255.0, THRESH_BINARY);

			cv::subtract(matMeanBuf, Test_tmp, matMeanBuf);
			//合并	
			cv::max(matSrcBuf, matMeanBuf, matSrcBuf);
		}
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//只留下点亮的部分
long CInspectAlign::FillMerge_SVI(cv::Mat& matSrcBuf, cv::Mat matMeanBuf, cv::Mat matMaskBuf)
{
	//如果参数为NULL
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matMeanBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matMaskBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	cv::Mat matTemp;
	cv::subtract(matSrcBuf, matMaskBuf, matTemp);

	//只保留背景平均部分
	cv::bitwise_and(matMeanBuf, matMaskBuf, matMeanBuf);

	//合并
	cv::add(matTemp, matMeanBuf, matSrcBuf);

	matTemp.release();

	return E_ERROR_CODE_TRUE;
}

//将曲线部分,点灯区域内侧轻轻放入(视情况而定)
cv::Point CInspectAlign::calcRoundIn(ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], int nIndex, int nRoundIn)
{
	// Top
	if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(0, nRoundIn);
	}
	// Left
	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(-nRoundIn, 0);
	}
	// Right
	else if (!tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(nRoundIn, 0);
	}
	// Bottom
	else if (!tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(0, -nRoundIn);
	}

	//////////////////////////////////////////////////////////////////////////

	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(-nRoundIn, -nRoundIn);
	}
	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(nRoundIn, -nRoundIn);
	}
	else if (!tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(nRoundIn, nRoundIn);
	}
	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP] &&
		!tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM] &&
		tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(-nRoundIn, nRoundIn);
	}

	//////////////////////////////////////////////////////////////////////////

	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_TOP])
	{
		return cv::Point(-nRoundIn, -nRoundIn);
	}
	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_TOP])
	{
		return cv::Point(nRoundIn, -nRoundIn);
	}
	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_RIGHT_BOTTOM])
	{
		return cv::Point(nRoundIn, nRoundIn);
	}
	else if (tRoundSet[nIndex].nCornerInside[E_CORNER_LEFT_BOTTOM])
	{
		return cv::Point(-nRoundIn, nRoundIn);
	}

	//////////////////////////////////////////////////////////////////////////
	else
	{
		// 		AfxMessageBox(_T("Set Corner Err !!!"));
	}

	return cv::Point(0, 0);
}

//曲线以外的直线连接
long CInspectAlign::calcLineConnect(cv::Mat& matSrcBuf, cv::Point ptSE[2], cv::Point ptPoly[2], int& nSE, int nSetArea)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//确认距离
	int nLength[4] = { 0, };
	nLength[0] = abs(ptSE[0].x - ptPoly[0].x) + abs(ptSE[0].y - ptPoly[0].y);
	nLength[1] = abs(ptSE[0].x - ptPoly[1].x) + abs(ptSE[0].y - ptPoly[1].y);
	nLength[2] = abs(ptSE[1].x - ptPoly[0].x) + abs(ptSE[1].y - ptPoly[0].y);
	nLength[3] = abs(ptSE[1].x - ptPoly[1].x) + abs(ptSE[1].y - ptPoly[1].y);

	int nMinIndex, nMinValue;

	if (nSetArea == 1)
	{
		nMinIndex = 0;
		nMinValue = nLength[0];

		for (int k = 1; k < 4; k++)
		{
			if (nMinValue > nLength[k])
			{
				nMinValue = nLength[k];
				nMinIndex = k;
			}
		}

		//仅在上次未使用时使用
		nSE = nMinIndex / 2;
	}
	else
	{
		//继续比较ptSE[0]
		if (nSE == 0)
		{
			nMinIndex = (nLength[0] > nLength[1]) ? 1 : 0;
		}
		else
		{
			nMinIndex = (nLength[2] > nLength[3]) ? 3 : 2;
		}
	}

	//短距离连接
	switch (nMinIndex)
	{
		// nLength[0] = abs(ptSE[0].x - ptPoly[0].x)	+  abs(ptSE[0].y - ptPoly[0].y);
	case 0:
		cv::line(matSrcBuf, ptSE[0], ptPoly[0], cv::Scalar(255, 255, 255));
		ptSE[0] = ptPoly[1];
		break;

		// nLength[1] = abs(ptSE[0].x - ptPoly[1].x)	+  abs(ptSE[0].y - ptPoly[1].y);
	case 1:
		cv::line(matSrcBuf, ptSE[0], ptPoly[1], cv::Scalar(255, 255, 255));
		ptSE[0] = ptPoly[0];
		break;

		// nLength[2] = abs(ptSE[1].x - ptPoly[0].x)	+  abs(ptSE[1].y - ptPoly[0].y);
	case 2:
		cv::line(matSrcBuf, ptSE[1], ptPoly[0], cv::Scalar(255, 255, 255));
		ptSE[1] = ptPoly[1];
		break;

		// nLength[3] = abs(ptSE[1].x - ptPoly[1].x)	+  abs(ptSE[1].y - ptPoly[1].y);
	case 3:
		cv::line(matSrcBuf, ptSE[1], ptPoly[1], cv::Scalar(255, 255, 255));
		ptSE[1] = ptPoly[0];
		break;
	}

	return E_ERROR_CODE_TRUE;
}

//填充拐角部分区域(Color)
long CInspectAlign::FillCorner(cv::Mat& matSrcROIBuf, cv::Mat& matMaskROIBuf, int nType)
{
	//如果没有缓冲区。
	if (matSrcROIBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matMaskROIBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	bool bLoop = true;

	switch (nType)
	{
	case E_CORNER_LEFT_TOP:
		while (bLoop)
		{
			bLoop = false;

			for (int j = 0; j < matMaskROIBuf.rows - 1; j++)
			{
				for (int i = 1; i < matMaskROIBuf.cols - 1; i++)
				{
					if (matMaskROIBuf.at<uchar>(j, i - 1) == 255)
						continue;

					if (matMaskROIBuf.at<uchar>(j + 1, i - 1) == 0)
						continue;

					if (matMaskROIBuf.at<uchar>(j, i) == 0)
						continue;

					matMaskROIBuf.at<uchar>(j, i - 1) = 255;

					matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[0] = (matSrcROIBuf.at<cv::Vec3b>(j + 1, i - 1)[0] + matSrcROIBuf.at<cv::Vec3b>(j, i)[0]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[1] = (matSrcROIBuf.at<cv::Vec3b>(j + 1, i - 1)[1] + matSrcROIBuf.at<cv::Vec3b>(j, i)[1]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[2] = (matSrcROIBuf.at<cv::Vec3b>(j + 1, i - 1)[2] + matSrcROIBuf.at<cv::Vec3b>(j, i)[2]) / 2;

					bLoop = true;
				}
			}
		}
		break;

	case E_CORNER_RIGHT_TOP:
		while (bLoop)
		{
			bLoop = false;

			for (int j = 0; j < matMaskROIBuf.rows - 1; j++)
			{
				for (int i = 1; i < matMaskROIBuf.cols; i++)
				{
					if (matMaskROIBuf.at<uchar>(j, i) == 255)
						continue;

					if (matMaskROIBuf.at<uchar>(j + 1, i) == 0)
						continue;

					if (matMaskROIBuf.at<uchar>(j, i - 1) == 0)
						continue;

					matMaskROIBuf.at<uchar>(j, i) = 255;

					matSrcROIBuf.at<cv::Vec3b>(j, i)[0] = (matSrcROIBuf.at<cv::Vec3b>(j + 1, i)[0] + matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[0]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i)[1] = (matSrcROIBuf.at<cv::Vec3b>(j + 1, i)[1] + matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[1]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i)[2] = (matSrcROIBuf.at<cv::Vec3b>(j + 1, i)[2] + matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[2]) / 2;

					bLoop = true;
				}
			}
		}
		break;

	case E_CORNER_RIGHT_BOTTOM:
		while (bLoop)
		{
			bLoop = false;

			for (int j = 1; j < matMaskROIBuf.rows; j++)
			{
				for (int i = 1; i < matMaskROIBuf.cols; i++)
				{
					if (matMaskROIBuf.at<uchar>(j, i) == 255)
						continue;

					if (matMaskROIBuf.at<uchar>(j, i - 1) == 0)
						continue;

					if (matMaskROIBuf.at<uchar>(j - 1, i) == 0)
						continue;

					matMaskROIBuf.at<uchar>(j, i) = 255;

					matSrcROIBuf.at<cv::Vec3b>(j, i)[0] = (matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[0] + matSrcROIBuf.at<cv::Vec3b>(j - 1, i)[0]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i)[1] = (matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[1] + matSrcROIBuf.at<cv::Vec3b>(j - 1, i)[1]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i)[2] = (matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[2] + matSrcROIBuf.at<cv::Vec3b>(j - 1, i)[2]) / 2;

					bLoop = true;
				}
			}
		}
		break;

	case E_CORNER_LEFT_BOTTOM:
		while (bLoop)
		{
			bLoop = false;

			for (int j = 1; j < matMaskROIBuf.rows; j++)
			{
				for (int i = 1; i < matMaskROIBuf.cols; i++)
				{
					if (matMaskROIBuf.at<uchar>(j, i - 1) == 255)
						continue;

					if (matMaskROIBuf.at<uchar>(j, i) == 0)
						continue;

					if (matMaskROIBuf.at<uchar>(j - 1, i - 1) == 0)
						continue;

					matMaskROIBuf.at<uchar>(j, i - 1) = 255;

					matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[0] = (matSrcROIBuf.at<cv::Vec3b>(j, i)[0] + matSrcROIBuf.at<cv::Vec3b>(j - 1, i - 1)[0]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[1] = (matSrcROIBuf.at<cv::Vec3b>(j, i)[1] + matSrcROIBuf.at<cv::Vec3b>(j - 1, i - 1)[1]) / 2;
					matSrcROIBuf.at<cv::Vec3b>(j, i - 1)[2] = (matSrcROIBuf.at<cv::Vec3b>(j, i)[2] + matSrcROIBuf.at<cv::Vec3b>(j - 1, i - 1)[2]) / 2;

					bLoop = true;
				}
			}
		}
		break;
	}

	return E_ERROR_CODE_TRUE;
}

// PNZ ShiftCopy Parameter Check ( 18.10.18 )
long	CInspectAlign::ShiftCopyParaCheck(int ShiftValue, int& nCpyX, int& nCpyY, int& nLoopX, int& nLoopY)
{
	if (ShiftValue == 0) return false;
	// 
	// 	nCpyX	= (int)	(ShiftValue											)	/ 1000	;	// X方向单元
	// 	nCpyY	= (int) (ShiftValue - nCpyX*1000							)	/ 100	;	// Y方向Unit
	// 	nLoopX	= (int) (ShiftValue	- nCpyX*1000 - nCpyY*100				)	/ 10	;	// X方向Loop
	// 	nLoopY	= (int) (ShiftValue - nCpyX*1000 - nCpyY*100 - nLoopX*10	)	/ 1		;	// Y方向Loop

	nCpyX = (int)(ShiftValue / 1000 % 10);	// X方向单元
	nCpyY = (int)(ShiftValue / 100 % 10);	// Y方向Unit
	nLoopX = (int)(ShiftValue / 10 % 10);	// X方向Loop
	nLoopY = (int)(ShiftValue / 1 % 10);	// Y方向Loop

	return E_ERROR_CODE_TRUE;
}

//获取模式名称
CString	CInspectAlign::GetPatternString(int nPattern)
{
	switch (nPattern)
	{
	case E_IMAGE_CLASSIFY_AVI_R:
		return _T("R");
		break;

	case E_IMAGE_CLASSIFY_AVI_G:
		return _T("G");
		break;

	case E_IMAGE_CLASSIFY_AVI_B:
		return _T("B");
		break;

	case E_IMAGE_CLASSIFY_AVI_BLACK:
		return _T("BLACK");
		break;

	case E_IMAGE_CLASSIFY_AVI_WHITE:
		return _T("WHITE");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
		return _T("G32");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
		return _T("G64");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
		return _T("G87");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
		return _T("G128");
		break;

	case E_IMAGE_CLASSIFY_AVI_VTH:
		return _T("VTH");
		break;

	case E_IMAGE_CLASSIFY_AVI_DUST:
		return _T("DUST");
		break;

	case E_IMAGE_CLASSIFY_AVI_PCD:
		return _T("PCD");
		break;

	case E_IMAGE_CLASSIFY_AVI_VINIT:
		return _T("VINIT");
		break;
	case E_IMAGE_CLASSIFY_AVI_DUSTDOWN:
		return _T("DUSTDOWN");
		break;
	default:
		return _T("NULL");
		break;
	}

	return _T("NULL");
}

//获取模式名称
CStringA CInspectAlign::GetPatternStringA(int nPattern)
{
	switch (nPattern)
	{
	case E_IMAGE_CLASSIFY_AVI_R:
		return ("R");
		break;

	case E_IMAGE_CLASSIFY_AVI_G:
		return ("G");
		break;

	case E_IMAGE_CLASSIFY_AVI_B:
		return ("B");
		break;

	case E_IMAGE_CLASSIFY_AVI_BLACK:
		return ("BLACK");
		break;

	case E_IMAGE_CLASSIFY_AVI_WHITE:
		return ("WHITE");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
		return ("G32");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_64:
		return ("G64");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
		return ("G87");
		break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
		return ("G128");
		break;

	case E_IMAGE_CLASSIFY_AVI_VTH:
		return ("VTH");
		break;

	case E_IMAGE_CLASSIFY_AVI_DUST:
		return ("DUST");
		break;

	case E_IMAGE_CLASSIFY_AVI_PCD:
		return ("PCD");
		break;

	case E_IMAGE_CLASSIFY_AVI_VINIT:
		return ("VINIT");
		break;
	case E_IMAGE_CLASSIFY_AVI_DUSTDOWN:
		return ("DUSTDOWN");
		break;
	default:
		return ("NULL");
		break;
	}

	return ("NULL");
}

long CInspectAlign::CurlJudge(cv::Mat matSrcBuf, double* dpara, cv::Point* ptCorner, BOOL& bCurl, stMeasureInfo* stCurlMeasure, BOOL bSaveImage, CString strPath)
{
	if (matSrcBuf.empty())
		return E_ERROR_CODE_EMPTY_BUFFER;

	Mat matResult;
	cv::cvtColor(matSrcBuf, matResult, CV_GRAY2RGB);

	//参数
	BOOL bCurlUse = (BOOL)dpara[E_PARA_APP_CURL_USE];         //是否使用卷判定
	if (bCurlUse < 1)
	{
		bCurl = FALSE;
		return E_ERROR_CODE_TRUE;
	}
	int nXRange = (int)dpara[E_PARA_APP_CURL_RANGE_X];                 // x轴检测范围100
	int nYRangeTop = (int)dpara[E_PARA_APP_CURL_RANGE_Y_TOP];             // y轴检测范围TOP200
	int nYRangeBottom = (int)dpara[E_PARA_APP_CURL_RANGE_Y_BOTTOM];          // y轴检测范围Bottom 200(5.99Q的Y轴比率Bottom较小)
	int nDetectionCondition = (int)dpara[E_PARA_APP_CURL_CONDITION_COUNT];         // 检测条件2
	double dSearchRatio = dpara[E_PARA_APP_CURL_SEARCH_RATIO];            // 检出率1.5
	double dStandardGVRatio = dpara[E_PARA_APP_CURL_STANDARD_GV_RATIO];       // 平均GV比率0.85
	int nCornerSearchCount = (int)dpara[E_PARA_APP_CURL_CORNER_SEARCH_COUNT];     // 拐角检测条件5
	double dCornerSearchRatio = dpara[E_PARA_APP_CURL_CORNER_SEARCH_RATIO];     // 拐角检出率2.75
	int nStartOffsetLeft = (int)dpara[E_PARA_APP_CURL_START_OFFSET_LEFT];       // 检查开始偏移		300
	int nStartOffsetRight = (int)dpara[E_PARA_APP_CURL_END_OFFSET_RIGHT];      // 检查结束偏移
	int nBlurSize = (int)dpara[E_PARA_APP_CURL_BLUR_SIZE];               // 腮红		11
	float nGausSigma = (float)dpara[E_PARA_APP_CURL_GAUSSIGMA];				 // 高斯安布勒Sigma	5
	int	  nProfileOffset_W = (int)dpara[E_PARA_APP_CURL_PROFILE_SEARCH_OFFSET];	//在Profile中,将Parameter值用作最小值后的宽度以保留值,删除宽度后的值。

	if (nBlurSize % 2 == 0)
		nBlurSize++;
	Size szGaussianBlurSize(nBlurSize, nBlurSize);

	Mat mtSrc;
	matSrcBuf.copyTo(mtSrc);

	cv::GaussianBlur(mtSrc, mtSrc, szGaussianBlurSize, nGausSigma);             // 高西安·布勒

#ifdef _DEBUG
	//可执行驱动器D:\不固定-必要时利用InspectLibLog的GETDRV()
//imwrite("E:\\IMTC\\IDB\\CurlBlur.bmp", mtSrc);  
#endif

	std::vector<int> nGV[LINE_END];
	std::vector<int> nDistance[LINE_END];
	std::vector<int> nCornerSearchDistance[LINE_END];
	Point ptLocation[4] = { ptCorner[0], ptCorner[3], ptCorner[1], ptCorner[2] };
	int nMaxGV[LINE_END] = { 0,0 };
	int nMinGV[LINE_END] = { 255,255 };
	int nMinGVInx[LINE_END] = { 0,0 };
	/*double nMeanGV[LINE_END];*/
	double nDetectionCount[LINE_END];
	int nImageGV[LINE_END];
	int nCurlJudge = 0;
	double dMeanDistance[LINE_END];
	double nMaxDistance[LINE_END];
	int nCornerMaxDistance[LINE_END];
	int nCount[LINE_END];

	Mat mtCurl;

	matSrcBuf.copyTo(mtCurl);
	cv::cvtColor(mtCurl, mtCurl, CV_GRAY2RGB);

	CString strTest;

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
	for (int nDirection = 0; nDirection < LINE_END; nDirection++)
	{
		int nStageNumOffset = 10;
		double dStageAvg = 0;

		for (int nX = ptLocation[nDirection].x + nStartOffsetLeft; nX < ptLocation[nDirection + 2].x - nStartOffsetRight; nX += nXRange)
		{
			for (int nY = ptLocation[nDirection].y - nYRangeTop; nY < ptLocation[nDirection].y + nYRangeBottom; nY++)
			{
				Scalar m, s;
				cv::meanStdDev(mtSrc(cv::Rect(nX, nY, nXRange, 1)), m, s);  //nX Range长度的平均GV
				nImageGV[nDirection] = (int)m[0];
				nGV[nDirection].push_back(nImageGV[nDirection]);
				if (nMinGV[nDirection] > nGV[nDirection][nGV[nDirection].size() - 1])
				{
					nMinGV[nDirection] = nGV[nDirection][nGV[nDirection].size() - 1];
					nMinGVInx[nDirection] = nGV[nDirection].size() - 1;
				}
			}
			if (nDirection == 0) // TOP
			{
				for (int nStage = 0; nStage < nStageNumOffset; nStage++)
					dStageAvg += nGV[nDirection][nStage];
				dStageAvg /= nStageNumOffset;
			}

			if (nDirection == 1) // BOTTOM
			{
				for (int nStage = nGV[nDirection].size() - 1; nStage > nGV[nDirection].size() - 1 - 10; nStage--)
					dStageAvg += nGV[nDirection][nStage];
				dStageAvg /= nStageNumOffset;
			}

			cv::Point pt1, pt2;
			//测量TOM方向的CUR时
			if (nDirection == 0)
			{
				for (int n = 0; n < nGV[nDirection].size(); n++)
				{
					if (nGV[nDirection][n] <= dStageAvg * dStandardGVRatio)
					{
						pt1 = cv::Point(nX, ptLocation[nDirection].y - nYRangeTop + n);
						nDetectionCount[nDirection] = 0;
						for (int m = n; m < nGV[nDirection].size(); m++)
						{
							nDetectionCount[nDirection]++;  // 距离测量
							if (nGV[nDirection][m] > dStageAvg * dStandardGVRatio)
							{
								pt2 = cv::Point(nX, ptLocation[nDirection].y - nYRangeTop + m);
								break;
							}
						}
						break;
					}
				}
				CString test_dist;
				test_dist.Format(_T("%.1lf"), nDetectionCount[nDirection]);
				cv::line(mtCurl, pt1, pt2, Scalar(255, 0, 0));
				cv::putText(mtCurl, ((cv::String)(CStringA)test_dist), cv::Point(pt1.x - 5, pt1.y - 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 0, 0), 1);
			}

			//对于BUTTOM方向的CUR测量
			else if (nDirection == 1)
			{
				for (int n = nGV[nDirection].size() - 1; n >= 0; n--)
				{
					if (nGV[nDirection][n] <= dStageAvg * dStandardGVRatio)
					{
						pt1 = cv::Point(nX, ptLocation[nDirection].y - nYRangeTop + n);
						nDetectionCount[nDirection] = 0;
						for (int m = n; m >= 0; m--)
						{
							nDetectionCount[nDirection]++;  // 距离测量
							if (nGV[nDirection][m] > dStageAvg * dStandardGVRatio)
							{
								pt2 = cv::Point(nX, ptLocation[nDirection].y - nYRangeTop + m);
								break;
							}
						}
						break;
					}
				}
				CString test_dist;
				test_dist.Format(_T("%.1lf"), nDetectionCount[nDirection]);
				cv::line(mtCurl, pt1, pt2, Scalar(255, 0, 0));
				cv::putText(mtCurl, ((cv::String)(CStringA)test_dist), cv::Point(pt1.x - 5, pt1.y - 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 0, 0), 1);
			}
			nDistance[nDirection].push_back(nDetectionCount[nDirection]);
			nGV[nDirection].erase(nGV[nDirection].begin(), nGV[nDirection].end());
		}

		//单独检查拐角部分
		for (int n = 0; n < nCornerSearchCount; n++)
		{
			nCornerSearchDistance[nDirection].push_back(nDistance[nDirection][n]);
		}
		for (int n = 0; n < nCornerSearchCount; n++)
		{
			nCornerSearchDistance[nDirection].push_back(nDistance[nDirection][nDistance[nDirection].size() - (n + 1)]);
		}

		nCornerMaxDistance[nDirection] = *max_element(nCornerSearchDistance[nDirection].begin(), nCornerSearchDistance[nDirection].end());// 拐角部分的最大值
		nDistance[nDirection].erase(nDistance[nDirection].begin(), nDistance[nDirection].begin() + nCornerSearchCount);
		nDistance[nDirection].erase(nDistance[nDirection].end() - nCornerSearchCount, nDistance[nDirection].end());

		nMaxDistance[nDirection] = *max_element(nDistance[nDirection].begin(), nDistance[nDirection].end());
		dMeanDistance[nDirection] = 0;
		for (int n = 0; n < nDistance[nDirection].size(); n++)
		{
			dMeanDistance[nDirection] += nDistance[nDirection][n]; //距离平均
		}
		dMeanDistance[nDirection] = dMeanDistance[nDirection] / nDistance[nDirection].size();
		nCount[nDirection] = 0;
		for (int n = 0; n < nDistance[nDirection].size(); n++)
		{
			if (nDistance[nDirection][n] >= dMeanDistance[nDirection] * dSearchRatio)//(平均距离*检测率)的距离计数
				nCount[nDirection]++;
		}
		for (int n = 0; n < nCornerSearchDistance[nDirection].size(); n++)
		{
			if (nCornerSearchDistance[nDirection][n] >= dMeanDistance[nDirection] * dCornerSearchRatio)//(平均距离*检测率)的距离计数(拐角部分)
				nCount[nDirection]++;
		}
		if (nDirection == 0 && nCount[nDirection] >= nDetectionCondition)
			nCurlJudge++;//卷判定

		else if (nDirection == 1 && nCount[nDirection] >= nDetectionCondition)
			nCurlJudge++;//卷判定	
	}

	if (bSaveImage)
	{
		CString strSavePath;
		strSavePath.Format(_T("%s\\Curl_0.bmp"), strPath);
		imwrite((cv::String)(CStringA)strSavePath, mtCurl);
	}

	//杀死向量
	for (int i = 0; i < 2; i++)
	{
		vector<int>().swap(nGV[i]);
		vector<int>().swap(nDistance[i]);
		vector<int>().swap(nCornerSearchDistance[i]);
	}

	for (int n = 0; n < LINE_END; n++)
	{
		stCurlMeasure->dMeasureValue[n] = nMaxDistance[n];
		stCurlMeasure->dMeasureValue[n + 2] = nCornerMaxDistance[n];
	}

	if (nCurlJudge >= 1)
	{
		bCurl = TRUE;
		stCurlMeasure->bJudge = TRUE;
	}
	else
	{
		bCurl = FALSE;
		stCurlMeasure->bJudge = FALSE;
	}

	return E_ERROR_CODE_TRUE;
}


long CInspectAlign::Estimation_XY(cv::Mat matSrcBuf, cv::Mat& matDstBuf, double* dPara, CMatBuf* cMemSub)
{
	// 에러 코드
	long	nErrorCode = E_ERROR_CODE_TRUE;

	// 예외 처리
	if (matSrcBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstBuf.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matSrcBuf.channels() != 1)	return E_ERROR_CODE_IMAGE_GRAY;

	int		nEstiDimX = (int)1;
	int		nEstiDimY = (int)1;
	int		nEstiStepX = (int)100;
	int		nEstiStepY = (int)100;

	double	dEstiBright = double(1.5);
	double	dEstiDark = double(0.5);

	// 예외 처리
	if (nEstiDimX <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiDimY <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiStepX <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;
	if (nEstiStepY <= 0)	return E_ERROR_CODE_MURA_WRONG_PARA;

	// 버퍼 할당 & 초기화
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