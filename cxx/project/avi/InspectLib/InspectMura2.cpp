/************************************************************************
Mura2算法相关源
************************************************************************/

#include "StdAfx.h"
#include "InspectMura2.h"
#include "AlgoBase.h"

CInspectMura2::CInspectMura2(void)
{
}

CInspectMura2::~CInspectMura2(void)
{
}

long CInspectMura2::FindMuraDefect(cv::Mat matSrcBuffer, cv::Mat& matDrawBuffer, cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath,
	stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt)
{
	///////////////////////////////////////////////////////
	//
		//	function	:Main函数
		//	made by		:金振英
	//				: 2017. 04. 17
	//
		//	[in]	param1	:	原始画面		/	CV_8UC1
	//
		//	[out]	param2:	二进制画面输出/	CV_8UC1
		//						两个尺寸,函数调用前必须声明以下内容。
	//
	//						vector<cv::Mat> matBins(3);
	//						matBins.at(0) = cv::Mat(matSrcImage.size(), CV_8UC1, Scalar(0));
	//						matBins.at(1) = cv::Mat(matSrcImage.size(), CV_8UC1, Scalar(0));
	//
	//						0 : Dark
	//						1 : Bright
	//
		//	[in]	param3:	检查区域ROI
	//
		//	[in]	param4	:	参数(仍在使用X)
	//
		//	[in]	param5	:	图像文件存储路径
	//
		//	[in]	param6	:	选择是否保存图像文件
	//
	//	[return]		:	true / false
	//
	////////////////////////////////////////////////////////

		//如果参数为NULL。
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

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
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;

	long	nWidth = (long)matSrcBuffer.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuffer.rows;	// 图像垂直尺寸

	//临时缩小范围
	CRect rectROI = new CRect(
		max(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		min(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	//如果扫描区域超出画面大小
	if (rectROI.left < 0 ||
		rectROI.top < 0 ||
		rectROI.right >= nWidth ||
		rectROI.bottom >= nHeight)	return E_ERROR_CODE_ROI_OVER;

	cv::Rect _rectROI(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height());

	Mat matDarkResultImage_01;
	Mat matBrightResultImage_01;

	LogicStart_Gray(matSrcBuffer, matDarkResultImage_01, matBrightResultImage_01, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge);

	//Bright
	if (!matBrightResultImage_01.empty())
	{
		//标签
		CFeatureExtraction cFeatureExtraction;

		//只使用EMD过滤器...
		cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer, matBrightResultImage_01, matDrawBuffer,
			nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("M2B_"), EngineerBlockDefectJudge, pResultBlob);

		//绘制结果轮廓(Light SkyBlue)
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

		//保存结果轮廓
		cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt);
	}

	//Dark
	if (!matDarkResultImage_01.empty())
	{
		//标签
		CFeatureExtraction cFeatureExtraction;

		//只使用EMD过滤器...
		cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer, matDarkResultImage_01, matDrawBuffer,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("M2D_"), EngineerBlockDefectJudge, pResultBlob);

		//绘制结果轮廓(Light SkyBlue)
		cFeatureExtraction.DrawBlob(matDrawBuffer, cv::Scalar(135, 206, 250), BLOB_DRAW_BLOBS_CONTOUR, true);

		//保存结果轮廓
		cFeatureExtraction.SaveTxt(nCommonPara, strContourTxt);
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectMura2::LogicStart_Gray(cv::Mat& matSrcImage, cv::Mat& matDarkResultImage_01, cv::Mat& matBrightResultImage_01,
	CRect rectROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge)
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

	bImageSave = true;

	//////////////////////////////////////////////////////////////////////////
	int nPIdx = 0;
	float fBrightThresholdRatio_M1 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_M1 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的暗不良Threshold ratio

	float fBrightThresholdRatio_OutLine_M1 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_OutLine_M1 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的暗不良Threshold ratio

	int nBirghtFilterSize_M1 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小
	int nDarkFilterSize_M1 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小

	nPIdx++;//  Dummy => --------------------------------------------

	float fBrightThresholdRatio_M2 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_M2 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的暗不良Threshold ratio

	float fBrightThresholdRatio_OutLine_M2 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_OutLine_M2 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的暗不良Threshold ratio

	int nBirghtFilterSize_M2 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小
	int nDarkFilterSize_M2 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小

	nPIdx++;//  Dummy => --------------------------------------------

	float fBrightThresholdRatio_L1 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_L1 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的暗不良Threshold ratio

	float fBrightThresholdRatio_OutLine_L1 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_OutLine_L1 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的暗不良Threshold ratio

	int nBirghtFilterSize_L1 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小
	int nDarkFilterSize_L1 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小

	nPIdx++;//  Dummy => --------------------------------------------

	float fBrightThresholdRatio_L2 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_L2 = (float)dPara[nPIdx++] / 100.0;	// 主检查区域的暗不良Threshold ratio

	float fBrightThresholdRatio_OutLine_L2 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的亮不良Threshold ratio
	float fDarkThresholdRatio_OutLine_L2 = (float)dPara[nPIdx++] / 100.0; // 外围检查区域的暗不良Threshold ratio

	int nBirghtFilterSize_L2 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小
	int nDarkFilterSize_L2 = (int)dPara[nPIdx++]; // 要过滤第一次的不良大小

	nPIdx++;//  Dummy => --------------------------------------------

	double dblTargetMeanGV = (double)dPara[nPIdx++]; // 当前Parameter设置的Cell的标准GV。如果输入画面的平均GV高于相应的GV,则整体缩放亮度。
	int nBlur_Base = (int)dPara[nPIdx++];			// 放大前要运行的Blur大小
	int nBlur_Final = (int)dPara[nPIdx++];		// 放大后运行Blur大小以创建扫描目标图像
	int nBlur_Back = (int)dPara[nPIdx++];		// 放大后运行Blur大小以创建背景图像

	nPIdx++;//  Dummy => --------------------------------------------

	int nAveMinStart = (int)dPara[nPIdx++];	//检查时抓住Threshold时,以平均GV的比率决定。这时会生成直方图进行计算,这时至少只使用比设置GV亮的Pixel。
	double dblAveCutOffCountR_Min = (double)dPara[nPIdx++] / 100.0; // 检查时抓住Threshold时,以平均GV的比率决定。这时会制作直方图进行计算,认为只有直方图整体Pixel个数超过设定值的GV才有效。

	int nMaxGVAreaPartX = (int)dPara[nPIdx++]; // 点击与放大前的底座相差太大的Pixel时使用的Parameter-按区域进行底座计算,这时会设置X轴方向的区域分成几等分。
	int nMaxGVAreaPartY = (int)dPara[nPIdx++]; // 点击与放大前的基础相差太大的Pixel时使用的Parameter-按区域进行基础计算,这时设置Y轴方向的区域分成几等分。
	double dblMinGVR = (double)dPara[nPIdx++] / 100.0; // 点击与放大前的基础相差太多的Pixel时使用的Parameter-你会点击与基础相差多少%的家伙吗？阴暗的不良行为。1.0最多
	double dblMaxGVR = (double)dPara[nPIdx++] / 100.0; // 点击与放大前的基础相差太多的Pixel时使用的Parameter-你会点击与基础相差多少%的家伙吗？明亮的不良。1.0最多

	int nMaxGVAreaPartX_BGI = (int)dPara[nPIdx++];  // 放大后,在生成基础图像时,会按与基础相差太大的Pixel时使用的Parameter-按区域进行基础计算,这时会设置X轴方向的区域分成多少等分。
	int nMaxGVAreaPartY_BGI = (int)dPara[nPIdx++];  // 放大后,在生成基础图像时,按下与基础相差太大的Pixel时使用的Parameter-按区域进行基础计算,此时将Y轴方向的区域分成几等分。
	double dblMinGVR_BGI = (double)dPara[nPIdx++] / 100.0; // 放大后,点击生成基础图像时与基础相差太大的Pixel时使用的Parameter-点击与基础相差多少%的家伙。阴暗的不良行为。1.0最多
	double dblMaxGVR_BGI = (double)dPara[nPIdx++] / 100.0; // 放大后,点击生成基础图像时与基础相差太大的Pixel时使用的Parameter-点击与基础相差多少%的家伙。明亮的不良。1.0最多

	nPIdx++;//  Dummy => --------------------------------------------

	double dblResizeRatio = (double)dPara[nPIdx++];

	long	nOutLineForDelete = (long)dPara[nPIdx++];		// 删除轮廓Pixel	

	int nOutLineArea_01 = (int)dPara[nPIdx++];		// 外围检查区域的检查规格不同。这时会决定到离外围有多远的地方给不同的。
	int nOutLineArea_02 = (int)dPara[nPIdx++];		// 外围检查区域的检查规格不同。这时会决定到离外围有多远的地方给不同的。

	////////////////////////
		//缩小检查区域的轮廓//
	////////////////////////

	CRect rectTemp(rectROI);
	rectTemp.left += nOutLineForDelete;
	rectTemp.top += nOutLineForDelete;
	rectTemp.right -= nOutLineForDelete;
	rectTemp.bottom -= nOutLineForDelete;

	cv::Rect cropRect = cv::Rect(rectTemp.left, rectTemp.top, rectTemp.Width(), rectTemp.Height());

	cv::Mat matSrcCrop = matSrcImage(cropRect);
	cv::Mat matResizeTmp1;
	cv::Mat matResizeTmp2;
	cv::Mat matResizeTmp3;
	cv::Mat mat16Tmp1;
	cv::Mat mat16Tmp2;
	cv::Mat mat16Tmp3;
	cv::Mat mat16Tmp4;
	cv::Mat mat16InspImage;
	cv::Mat mat16BackgroundImage_M;
	cv::Mat mat16BackgroundImage_L;
	cv::Mat mat16BinBright_M1;
	cv::Mat mat16BinDark_M1;
	cv::Mat mat16BinBright_M2;
	cv::Mat mat16BinDark_M2;
	cv::Mat mat16BinBright_L1;
	cv::Mat mat16BinDark_L1;
	cv::Mat mat16BinBright_L2;
	cv::Mat mat16BinDark_L2;
	cv::Mat mat16BinResult_Bright;
	cv::Mat mat16BinResult_Dark;
	double dblBrightAverage_M;
	double dblDarkAverage_M;
	double dblBrightAverage_L;
	double dblDarkAverage_L;

	Size szBlur_Base = Size(nBlur_Base, nBlur_Base);
	Size szBlur_Final = Size(nBlur_Final, nBlur_Final);
	Size sznBlur_Back = Size(nBlur_Back, nBlur_Back);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_Main_%02d_Src.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matSrcCrop);
	}

	//////////////////////
		//重新调整图像大小//
	//////////////////////

	cv::resize(matSrcCrop, matResizeTmp1, Size(matSrcCrop.cols / dblResizeRatio, matSrcCrop.rows / dblResizeRatio), CV_INTER_AREA);

	/////////////////////
		//目标平均GV应用//
	/////////////////////

	AlgoBase::ApplyMeanGV(matResizeTmp1, dblTargetMeanGV);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_Resize.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTmp1);
	}

	//////////////////////
		//给出默认的Blur。//
	//////////////////////
	cv::blur(matResizeTmp1, matResizeTmp2, szBlur_Base);
	cv::blur(matResizeTmp2, matResizeTmp3, szBlur_Base);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_Filter_Base.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTmp3);
	}

	////////////////////////////////////////////////////////////////////////
		//按与第一次周边相比差异很大的GV-放大前			//
		//特定GV以上或低于分区平均值的家伙会设置阈值。//
	////////////////////////////////////////////////////////////////////////

	SetMinMax(matResizeTmp3, nMaxGVAreaPartX, nMaxGVAreaPartY, dblMinGVR, dblMaxGVR);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_Filter_MinMax_01.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, matResizeTmp3);
	}

	/////////////
		//16 bit转换//
	/////////////

	matResizeTmp3.convertTo(mat16Tmp1, CV_16U);

	/////////////////
		//平方放大//
	/////////////////

	cv::pow(mat16Tmp1, 2, mat16Tmp1);

	////////////////////
		//生成扫描图像//	
	////////////////////

		//通过给一个小blur生成最终图像-当前25 mask
	cv::blur(mat16Tmp1, mat16InspImage, szBlur_Final);

	////////////////////
		//创建跟踪图像//
	////////////////////

		//设定特定GV以上或低于区域平均值的家伙的阈值。在这里做基础之前再做一次。
	SetMinMax(mat16Tmp1, nMaxGVAreaPartX_BGI, nMaxGVAreaPartY_BGI, dblMinGVR_BGI, dblMaxGVR_BGI);

	//生成桌面
	MakeBGImage(BG_METHOD_BLUR, mat16Tmp1, mat16BackgroundImage_M, sznBlur_Back); // 平均桌面-当前150 mask
	MakeBGImage(BG_METHOD_PROFILE, mat16Tmp1, mat16BackgroundImage_L, sznBlur_Back); // 基于lINE PROFILE的桌面

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_InspImage.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16InspImage / 32);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_Back_M.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BackgroundImage_M / 32);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_Back_L.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BackgroundImage_L / 32);
	}

	/////////////
		//最终结果//
	/////////////

		//基于Mean过滤器的桌面错误操作
	mat16Tmp1 = mat16InspImage - mat16BackgroundImage_M; //从检查图像中删除背景图像。-亮不良	
	mat16Tmp2 = mat16BackgroundImage_M - mat16InspImage; //从跟踪图像中删除扫描图像。-黑暗不良

	//基于Line profile的桌面错误操作
	mat16Tmp3 = mat16InspImage - mat16BackgroundImage_L; //从检查图像中删除背景图像。-亮不良	
	mat16Tmp4 = mat16BackgroundImage_L - mat16InspImage; //从跟踪图像中删除扫描图像。-黑暗不良

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_FinBright_M.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16Tmp1);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_FinDark_M.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16Tmp2);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_FinBright_L.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16Tmp3);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_FinDark_L.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16Tmp4);
	}

	//Threshold计算-剪切并求出上下特定比例。基于Mean过滤器
	dblBrightAverage_M = AlgoBase::GetAverage(mat16Tmp1, dblAveCutOffCountR_Min, nAveMinStart, 255);
	dblDarkAverage_M = AlgoBase::GetAverage(mat16Tmp2, dblAveCutOffCountR_Min, nAveMinStart, 255);

	//Threshold计算-剪切并求出上下特定比例。基于Line profile过滤器
	dblBrightAverage_L = AlgoBase::GetAverage(mat16Tmp3, dblAveCutOffCountR_Min, nAveMinStart, 255);
	dblDarkAverage_L = AlgoBase::GetAverage(mat16Tmp4, dblAveCutOffCountR_Min, nAveMinStart, 255);

	//基于Mean过滤器的计算
	MakeThresholdImage(mat16Tmp1, mat16BinBright_M1, dblBrightAverage_M, fBrightThresholdRatio_M1, fBrightThresholdRatio_OutLine_M1, nOutLineArea_01); // -明亮的不良
	MakeThresholdImage(mat16Tmp1, mat16BinBright_M2, dblBrightAverage_M, fBrightThresholdRatio_M2, fBrightThresholdRatio_OutLine_M2, nOutLineArea_02); // 弱视人-明亮的不良
	MakeThresholdImage(mat16Tmp2, mat16BinDark_M1, dblDarkAverage_M, fDarkThresholdRatio_M1, fDarkThresholdRatio_OutLine_M1, nOutLineArea_01); // -黑暗不良
	MakeThresholdImage(mat16Tmp2, mat16BinDark_M2, dblDarkAverage_M, fDarkThresholdRatio_M2, fDarkThresholdRatio_OutLine_M2, nOutLineArea_02); // 弱视人-暗不良

	//基于Line profile的计算	
	MakeThresholdImage(mat16Tmp3, mat16BinBright_L1, dblBrightAverage_L, fBrightThresholdRatio_L1, fBrightThresholdRatio_OutLine_L1, nOutLineArea_01); // -明亮的不良
	MakeThresholdImage(mat16Tmp3, mat16BinBright_L2, dblBrightAverage_L, fBrightThresholdRatio_L2, fBrightThresholdRatio_OutLine_L2, nOutLineArea_02); // 弱视人-亮不良	
	MakeThresholdImage(mat16Tmp4, mat16BinDark_L1, dblDarkAverage_L, fDarkThresholdRatio_L1, fDarkThresholdRatio_OutLine_L1, nOutLineArea_01); // -黑暗不良
	MakeThresholdImage(mat16Tmp4, mat16BinDark_L2, dblDarkAverage_L, fDarkThresholdRatio_L2, fDarkThresholdRatio_OutLine_L2, nOutLineArea_02); // 弱视人-暗不良

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinBright_M1.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinBright_M1);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinDark_M1.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinDark_M1);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinBright_M2.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinBright_M2);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinDark_M2.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinDark_M2);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinBright_L1.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinBright_L1);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinDark_L1.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinDark_L1);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinBright_L2.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinBright_L2);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinDark_L2.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinDark_L2);
	}

	///////////////////////////////////////////////
		//首先过滤第一个尺寸->强诗人,弱诗人//
	///////////////////////////////////////////////
	mat16BinResult_Bright = cv::Mat::zeros(mat16BinBright_M1.size(), CV_8UC1);
	mat16BinResult_Dark = cv::Mat::zeros(mat16BinDark_M1.size(), CV_8UC1);;

	CFeatureExtraction cFeatureExtraction;
	/////////////////////////////////////////////
	// //替换STRU_DEFECT_ITEM变量，改为stPanelBlockJudgeInfo.stDefectItem[0]访问 [hjf]
	//STRU_DEFECT_ITEM FilterJudgment;
	stPanelBlockJudgeInfo FilterJudgment;
	////////////////////////////////////
	FilterJudgment.stDefectItem[0].bDefectItemUse = TRUE;
	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].bUse = TRUE;
	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].nSign = E_SIGN_GREATER;

	double dblScale_Area = dblResizeRatio * dblResizeRatio;
	//亮不良
	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nBirghtFilterSize_M1 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinBright_M1, mat16BinBright_M1, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Bright, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nBirghtFilterSize_M2 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinBright_M2, mat16BinBright_M2, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Bright, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nBirghtFilterSize_L1 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinBright_L1, mat16BinBright_L1, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Bright, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nBirghtFilterSize_L2 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinBright_L2, mat16BinBright_L2, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Bright, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	//暗不良
	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nDarkFilterSize_M1 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinDark_M1, mat16BinDark_M1, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Dark, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nDarkFilterSize_M2 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinDark_M2, mat16BinDark_M2, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Dark, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nDarkFilterSize_L1 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinDark_L1, mat16BinDark_L1, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Dark, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	FilterJudgment.stDefectItem[0].Judgment[E_FEATURE_AREA].dValue = nDarkFilterSize_L2 / dblScale_Area;
	cFeatureExtraction.DoDefectBlobSingleJudgment(mat16BinDark_L2, mat16BinDark_L2, &FilterJudgment);
	cFeatureExtraction.DrawBlob(mat16BinResult_Dark, cv::Scalar(255, 255, 255), BLOB_DRAW_BLOBS, true);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinBright_01_FAfter.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinResult_Bright);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_MURA2_%02d_BinDark_01_FAfter.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		cv::imwrite((cv::String)(CStringA)strTemp, mat16BinResult_Dark);
	}

	//设置为原始大小
	matBrightResultImage_01 = cv::Mat::zeros(matSrcImage.size(), CV_8UC1);
	cv::resize(mat16BinResult_Bright, matBrightResultImage_01(cv::Rect(rectTemp.left, rectTemp.top, rectTemp.Width(), rectTemp.Height())), Size(rectTemp.Width(), rectTemp.Height()), 0, 0, CV_INTER_AREA);

	matDarkResultImage_01 = cv::Mat::zeros(matSrcImage.size(), CV_8UC1);
	cv::resize(mat16BinResult_Dark, matDarkResultImage_01(cv::Rect(rectTemp.left, rectTemp.top, rectTemp.Width(), rectTemp.Height())), Size(rectTemp.Width(), rectTemp.Height()), 0, 0, CV_INTER_AREA);

	matSrcCrop.release();
	matResizeTmp1.release();
	matResizeTmp2.release();
	matResizeTmp3.release();
	mat16Tmp1.release();
	mat16Tmp2.release();
	mat16Tmp3.release();
	mat16Tmp4.release();
	mat16InspImage.release();
	mat16BackgroundImage_M.release();
	mat16BinBright_M1.release();
	mat16BinDark_M1.release();
	mat16BinBright_M2.release();
	mat16BinDark_M2.release();
	mat16BinBright_L1.release();
	mat16BinDark_L1.release();
	mat16BinBright_L2.release();
	mat16BinDark_L2.release();
	mat16BinResult_Bright.release();;
	mat16BinResult_Dark.release();;

	return E_ERROR_CODE_TRUE;
}

void CInspectMura2::MakeThresholdImage(Mat& matOri, Mat& matDst, double dblAverage, float fThresholdR, double fThresholdR_Outline, int nOutLineArea)
{
	//Threshold计算//
	int nThreshold = (int)(dblAverage * fThresholdR);
	int nThreshold_OutLine = (int)(dblAverage * fThresholdR_Outline);

	//16bit图像是不直接二进制的过程
	matDst = matOri.clone();
	MakeThresholdImageWithCenterAndOut(matDst, nOutLineArea, nThreshold, nThreshold_OutLine);

	//转换为8位
	matDst.convertTo(matDst, CV_8U);

	//二进制
	cv::threshold(matDst, matDst, 0, 255.0, THRESH_BINARY);
}

void CInspectMura2::MakeBGImage(int nMethod, Mat& matSrcImage, Mat& matBGImage, Size szParam01)
{
	if (nMethod == BG_METHOD_BLUR)
	{
		MakeBGImage_Blur(matSrcImage, matBGImage, szParam01);
	}
	else if (nMethod == BG_METHOD_PROFILE)
	{
		MakeBGImage_Profile(matSrcImage, matBGImage);
	}
}

void CInspectMura2::MakeBGImage_Blur(Mat& matSrcImage, Mat& matBGImage, Size szParam01)
{
	cv::blur(matSrcImage, matBGImage, szParam01);
}
void CInspectMura2::MakeBGImage_Profile(Mat& matSrcImage, Mat& matBGImage)
{
	int nRow, nCol;
	Mat matRow, matCol;

	nRow = matSrcImage.rows;
	nCol = matSrcImage.cols;

	AlgoBase::MakeProfile(PROFILE_ROW, matSrcImage, matRow);
	AlgoBase::MakeProfile(PROFILE_COL, matSrcImage, matCol);

	matBGImage.create(nRow, nCol, CV_16U);

	for (int i = 0; i < nRow; i++)
	{
		matCol.row(0).copyTo(matBGImage.row(i));
	}

	for (int i = 0; i < nCol; i++)
	{
		cv::add(matBGImage.col(i), matRow.col(0), matBGImage.col(i));
	}
	matBGImage /= 2.0;
}

//用于分别处理外围和中央的Threshold函数
void CInspectMura2::MakeThresholdImageWithCenterAndOut(Mat& matSrcImage, int nOutLine, int nThresholdCen, int nThresholdOut)
{
	Mat matImageTmp;
	int nImageSizeX = matSrcImage.cols;
	int nImageSizeY = matSrcImage.rows;

	matImageTmp = matSrcImage.colRange(nOutLine, nImageSizeX - nOutLine).rowRange(0, nOutLine); // 上面
	matImageTmp = matImageTmp - nThresholdOut;

	matImageTmp = matSrcImage.colRange(nOutLine, nImageSizeX - nOutLine).rowRange(nImageSizeY - nOutLine, nImageSizeY); // 下面
	matImageTmp = matImageTmp - nThresholdOut;

	matImageTmp = matSrcImage.colRange(0, nOutLine); // 左边
	matImageTmp = matImageTmp - nThresholdOut;

	matImageTmp = matSrcImage.colRange(nImageSizeX - nOutLine, nImageSizeX); // 右边
	matImageTmp = matImageTmp - nThresholdOut;

	matImageTmp = matSrcImage.rowRange(nOutLine, nImageSizeY - nOutLine).colRange(nOutLine, nImageSizeX - nOutLine); // 中间
	matImageTmp = matImageTmp - nThresholdCen;
}

//给出各区间的平均值,将平均GV的特定比率以上的值按为Min,Max值。
void CInspectMura2::SetMinMax(Mat& matSrcImage, int nMaxGVAreaPartX, int nMaxGVAreaPartY, double dblMinGVR, double dblMaxGVR)
{
	int nImageSizeX = matSrcImage.cols;
	int nImageSizeY = matSrcImage.rows;
	int nImageTermX = nImageSizeX / nMaxGVAreaPartX;
	int nImageTermY = nImageSizeY / nMaxGVAreaPartY;
	int nAreaStartX, nAreaEndX;
	int nAreaStartY, nAreaEndY;
	double dblMean, dblMin, dblMax;
	Mat tmpImage;

	for (int y = 0; y < nMaxGVAreaPartY; y++)
	{
		nAreaStartY = y * nImageTermY;
		nAreaEndY = (y + 1) * nImageTermY;

		if (nAreaEndY > nImageSizeY - 1)
			nAreaEndY = nImageSizeY - 1;

		for (int x = 0; x < nMaxGVAreaPartX; x++)
		{
			nAreaStartX = x * nImageTermX;
			nAreaEndX = (x + 1) * nImageTermX;

			if (nAreaEndX > nImageSizeX - 1)
				nAreaEndX = nImageSizeX - 1;

			tmpImage = matSrcImage.colRange(nAreaStartX, nAreaEndX).rowRange(nAreaStartY, nAreaEndY);

			Scalar m;
			m = cv::mean(tmpImage);
			dblMean = m[0];

			dblMin = dblMean - (dblMean * dblMinGVR);
			dblMax = dblMean + (dblMean * dblMaxGVR);

			cv::max(tmpImage, dblMin, tmpImage);
			cv::min(tmpImage, dblMax, tmpImage);
		}
	}
}
