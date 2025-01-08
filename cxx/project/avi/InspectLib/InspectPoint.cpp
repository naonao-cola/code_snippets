
/************************************************************************/
//Point不良检测相关源
//修改日期:17.07.10
/************************************************************************/

#include "StdAfx.h"
#include "InspectPoint.h"
#include "AlgoBase.h"

CInspectPoint::CInspectPoint(void)
{
	cMem[0] = NULL;
	cMem[1] = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;
}

CInspectPoint::~CInspectPoint(void)
{

}

long CInspectPoint::DoFindPointDefect(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat& matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dAlignPara, cv::Rect* rcCHoleROI, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, CDefectCCD* cCCD, cv::Mat* matCholeBuffer)
{
	//如果参数为NULL
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcBuffer.empty())			return E_ERROR_CODE_EMPTY_BUFFER;

	long	nWidth = (long)matSrcBuffer.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcBuffer.rows;	// 图像垂直尺寸

	//画面号码
	long	nImageClassify = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];

	//查找ID
	CString strPathTemp;

	//删除右边的"\ \"
	strPathTemp.Format(_T("%s"), strAlgPath.Left(strAlgPath.GetLength() - 1));

	//在右侧查找"\ \"
	int nLength = strPathTemp.ReverseFind(_T('\\'));

	//只查找ID
	wchar_t	strID[MAX_PATH] = { 0, };
	swprintf_s(strID, MAX_PATH, L"%s", (LPCWSTR)strPathTemp.Right(strPathTemp.GetLength() - nLength - 1));

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//11客户要求Vinit模式外壳点检测不到,经确认ptCorner值小,外壳不良被剪掉
	//可能会在非活动区域检测到错误
	if (nImageClassify == E_IMAGE_CLASSIFY_AVI_VINIT)
	{
		ptCorner[E_CORNER_LEFT_TOP].x -= 10;
		ptCorner[E_CORNER_LEFT_TOP].y -= 10;
		ptCorner[E_CORNER_LEFT_BOTTOM].x -= 10;
		ptCorner[E_CORNER_LEFT_BOTTOM].y += 10;
		ptCorner[E_CORNER_RIGHT_TOP].x += 10;
		ptCorner[E_CORNER_RIGHT_TOP].y -= 10;
		ptCorner[E_CORNER_RIGHT_BOTTOM].x += 10;
		ptCorner[E_CORNER_RIGHT_BOTTOM].y += 10;
	}

	//将范围设置为大范围
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

	cv::Mat matDstImage[E_DEFECT_COLOR_COUNT];

	//缓冲区分配和初始化
	//17.09.26-Dust模式下的Dark	-Area大漂浮物
	//17.09.26-Dust模式下BRIGHT	-所有浮游生物
	matDstImage[E_DEFECT_COLOR_DARK] = cMem[0]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
	matDstImage[E_DEFECT_COLOR_BRIGHT] = cMem[0]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());

	//如果结果缓冲区不是NULL,则出现错误
	if (matDstImage[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	if (matDstImage[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	//图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)
	int nPS = (int)nCommonPara[E_PARA_COMMON_PS_MODE];

	//是否使用CCD(按模式参数index错误)
	int	nDelFlag = 0;

	//等待
//while( matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].empty() || matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].empty() || matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].empty())
//	Sleep(10);

	//每个画面的算法都不同。

	bool bGray = false;

	switch (nImageClassify)
	{
		//检查RGB Pattern
	case E_IMAGE_CLASSIFY_AVI_R:
	case E_IMAGE_CLASSIFY_AVI_G:
	case E_IMAGE_CLASSIFY_AVI_B:

	{
		cv::Mat matDstImage_RGB[E_DEFECT_COLOR_COUNT];
		matDstImage_RGB[E_DEFECT_COLOR_DARK] = cMem[1]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
		matDstImage_RGB[E_DEFECT_COLOR_BRIGHT] = cMem[1]->GetMat(matSrcBuffer.size(), matSrcBuffer.type());
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
				//RGB检查暗点/明点
				nErrorCode = LogicStart_RGB(matSrcBuffer, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, rcCHoleROI, matCholeBuffer, EngineerBlockDefectJudge);
			}
			break;
			case 1:
			{
				//检查PNZ RGB亮点(对PS画面效果更好)
				nErrorCode = LogicStart_RGBTest(matSrcBufferRGB, matDstImage_RGB, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, nImageClassify, rcCHoleROI, matCholeBuffer);
			}
			break;
			}
		}
		cv::bitwise_or(matDstImage_RGB[E_DEFECT_COLOR_BRIGHT], matDstImage[E_DEFECT_COLOR_BRIGHT], matDstImage[E_DEFECT_COLOR_BRIGHT]);

		matDstImage_RGB[E_DEFECT_COLOR_BRIGHT].release();
		matDstImage_RGB[E_DEFECT_COLOR_DARK].release();

		//是否使用CCD
		nDelFlag = (int)dPara[E_PARA_POINT_RGB_DEL_CCD_DELETE_FLAG];
	}
	break;

	//无需删除Black-Pol标记
	case E_IMAGE_CLASSIFY_AVI_BLACK:
	case E_IMAGE_CLASSIFY_AVI_VINIT:
	{

		//只留下点灯区域
		nErrorCode = DeleteOutArea(matSrcBuffer, ptCorner, cMem[0]);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteOutArea."));

		//有关CCD...
		if (nImageClassify == E_IMAGE_CLASSIFY_AVI_BLACK)
		{
			nDelFlag = (int)dPara[E_PARA_POINT_BLACK_DEL_CCD_DELETE_FLAG];
			int	nOffsetFlag = (int)dPara[E_PARA_POINT_BLACK_DEL_CCD_OFFSET_FLAG];
			int	nAutoDelFlag = (int)dPara[E_PARA_POINT_BLACK_DEL_CCD_AUTO_FLAG];
			int nAutoGV = (int)dPara[E_PARA_POINT_BLACK_DEL_CCD_AUTO_GV];
			int	nAutoBkGV = (int)dPara[E_PARA_POINT_BLACK_DEL_CCD_AUTO_BKGV];

			//CCD不良位置校正
			//17.07.08只注册Black pattern,因此只在Black Pattern中使用
			//17.07.11直接修改原始画面
		//PS模式下有1 pixel的误差,周围再校正1 pixel
			if (nOffsetFlag > 0)
				cCCD->OffsetDefectCCD(matSrcBuffer, 1, nPS);

			//清除CCD不良位置
				//17.07.08只注册Black pattern,因此只在Black Pattern中使用
				//17.07.11直接修改原始画面
			//PS模式下会有1个pixel的误差,所以周围再删除1个pixel
			if (nDelFlag > 0)
				cCCD->DeleteDefectCCD(matSrcBuffer, 1, nPS);

			//17.08.10-自动清除弹出的CCD故障
			if (nAutoDelFlag > 0)
			{
				long nCountCCD = cCCD->DeleteAutoDefectCCD(matSrcBuffer, nAutoGV, nAutoBkGV, nPS, cMem[0]);

				//文件打开后无限等待？
			//EnterCritical Section(&m_csCoordFile);

				//保存CCD数量
				CStdioFile	fileWriter;
				CString		strTemp;

				SYSTEMTIME time;
				::GetLocalTime(&time);

				CString strTimePath;
				strTimePath.Format(_T("E:\\IMTC\\DATA\\LOG\\Algorithm\\CCD_AUTO_DELETE_%04d%02d%02d.csv"), time.wYear, time.wMonth, time.wDay);

				//打开文件(如果文件未打开,则忽略)
				if (fileWriter.Open(strTimePath, CFile::modeCreate | CFile::modeNoTruncate | CFile::modeWrite))
				{
					strTemp.Format(_T("%02d:%02d:%02d, %s, %d\n"), time.wHour, time.wMinute, time.wSecond, strID, nCountCCD);
					fileWriter.SeekToEnd();
					fileWriter.WriteString(strTemp);

					//仅在文件打开时关闭
					fileWriter.Close();
				}

				//LeaveCriticalSection(&m_csCoordFile);
			}
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("OffsetDefectCCD."));

		//开始检查Black Pattern
		nErrorCode = LogicStart_Black(matSrcBuffer, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge);
	}
	break;

	//无需删除Dust-Pol标记
	case E_IMAGE_CLASSIFY_AVI_DUST:
	{
		//Dust算法集成
		nErrorCode = LogicStart_DustALL(matSrcBuffer, matDstImage, rectROI, rcCHoleROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge);

		//////////////////////////////////////////////////////////////////////////
					//添加气泡(用于在SVI中删除气泡检测)
					//当前未使用/验证后使用

		if (0)
		{
			cv::Mat matBubbleResult = cMem[0]->GetMat(matSrcBuffer.size(), matSrcBuffer.type(), false);

			nErrorCode = FindBubble_DustImage(matSrcBuffer, matBubbleResult, cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()), rcCHoleROI, dPara, nCommonPara, strAlgPath);

			cv::bitwise_or(matBubbleResult, matDstImage[E_DEFECT_COLOR_DARK], matDstImage[E_DEFECT_COLOR_DARK]);
		}
	}
	break;

	case E_IMAGE_CLASSIFY_AVI_GRAY_32:
	case E_IMAGE_CLASSIFY_AVI_GRAY_64:

	case E_IMAGE_CLASSIFY_AVI_GRAY_128:
	case E_IMAGE_CLASSIFY_AVI_WHITE:
	{
		bGray = true;
		nErrorCode = LogicStart_Gray(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, rcCHoleROI, matCholeBuffer);

		//是否使用CCD
		nDelFlag = (int)dPara[E_PARA_POINT_RGB_DEL_CCD_DELETE_FLAG];
	}
	break;
	case E_IMAGE_CLASSIFY_AVI_GRAY_87:
	{
		//RGB检查暗点/明点
		nErrorCode = LogicStart_CholePoint_G87(matSrcBuffer, matDstImage, rectROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, rcCHoleROI, matCholeBuffer);

	}
	break;
	case E_IMAGE_CLASSIFY_AVI_PCD:
	{

	}
	break;
	//如果画面号码输入错误。
	default:
		return E_ERROR_CODE_TRUE;
	}

	//移除最外围阵列的暗点
	long	nDeleteOutLine = (long)dPara[E_PARA_POINT_BLACK_DELETE_PIXEL];
	if (nDeleteOutLine > 0)
	{
		cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_LEFT_TOP], ptCorner[E_CORNER_RIGHT_TOP], cv::Scalar(0, 0, 0), nDeleteOutLine);
		cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_RIGHT_TOP], ptCorner[E_CORNER_RIGHT_BOTTOM], cv::Scalar(0, 0, 0), nDeleteOutLine);
		cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_RIGHT_BOTTOM], ptCorner[E_CORNER_LEFT_BOTTOM], cv::Scalar(0, 0, 0), nDeleteOutLine);
		cv::line(matDstImage[E_DEFECT_COLOR_DARK], ptCorner[E_CORNER_LEFT_BOTTOM], ptCorner[E_CORNER_LEFT_TOP], cv::Scalar(0, 0, 0), nDeleteOutLine);
	}
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteOutLine."));

	//choikwangil
	if (nImageClassify == E_IMAGE_CLASSIFY_AVI_BLACK || nImageClassify == E_IMAGE_CLASSIFY_AVI_VINIT) {
		int Black_roi_outline_offset = (int)dPara[E_PARA_POINT_BLACK_ROI_OUTLINE_OFFSET];

		vector<vector<cv::Point>> contours;
		findContours(matBKBuffer, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

		cv::drawContours(matBKBuffer, contours, -1, 0, Black_roi_outline_offset);

	}

	//17.12.13-删除点亮区域检测
	if (!matBKBuffer.empty())
	{
		if (bGray)
		{
			int		nRoundIn = (int)(dAlignPara[E_PARA_ROUND_IN]);
			int		nMaskSize = nRoundIn * 2 + 1;
			cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDstImage[E_DEFECT_COLOR_DARK]);

			cv::Mat matBkBufferForB = cMem[0]->GetMat(matDstImage[E_DEFECT_COLOR_BRIGHT].size(), matDstImage[E_DEFECT_COLOR_BRIGHT].type());
			cv::blur(matBKBuffer, matBkBufferForB, cv::Size(nMaskSize, nMaskSize));
			cv::threshold(matBkBufferForB, matBkBufferForB, 254, 255, CV_THRESH_BINARY);

			cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBkBufferForB, matDstImage[E_DEFECT_COLOR_BRIGHT]);

			//	cv::imwrite("E:\\IMTC\\AlignTmp\\matBKBuffer.bmp", matBKBuffer);
			//	cv::imwrite("E:\\IMTC\\AlignTmp\\matBkBufferForB.bmp", matBkBufferForB);
		}
		else
		{
			if (nImageClassify != E_IMAGE_CLASSIFY_AVI_BLACK && nImageClassify != E_IMAGE_CLASSIFY_AVI_VINIT)//更改为不删除背景...背景值听起来很奇怪...pwj
			{
				cv::subtract(matDstImage[E_DEFECT_COLOR_DARK], matBKBuffer, matDstImage[E_DEFECT_COLOR_DARK]);
				cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT], matBKBuffer, matDstImage[E_DEFECT_COLOR_BRIGHT]);
			}
		}
	}
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Sub BK."));

	//删除CCD Defect
	if (nDelFlag > 0)
	{
		//清除CCD不良位置
		//PS模式下会有1 pixel的误差,所以周围再删除1 pixel
		cCCD->DeleteDefectCCD(matDstImage[E_DEFECT_COLOR_DARK], 2, nPS);
		cCCD->DeleteDefectCCD(matDstImage[E_DEFECT_COLOR_BRIGHT], 2, nPS);
	}

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteDefectCCD."));

	//	//转交结果

			//正在测试pwj.clone()
	matDstImage[E_DEFECT_COLOR_DARK].copyTo(matDarkBuffer);  //matDarkBuffer已经有内存了,所以copyto更快
	matDstImage[E_DEFECT_COLOR_BRIGHT].copyTo(matBrightBuffer);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Result Copy."));

	//取消分配
	matDstImage[E_DEFECT_COLOR_DARK].release();
	matDstImage[E_DEFECT_COLOR_BRIGHT].release();

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("End."));

	return nErrorCode;
}

//删除Dust后,转交结果向量
long CInspectPoint::GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer)
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
	int		nUIImageNumber = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];

	//将范围设置为大范围
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//Dust模式没有问题
	if (nImageNum == E_IMAGE_CLASSIFY_AVI_DUST)
		return E_ERROR_CODE_TRUE;

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_DustImage_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDustBuffer[E_DEFECT_COLOR_BRIGHT]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_DustImage_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDustBuffer[E_DEFECT_COLOR_DARK]);
	}

	//如果有Dust画面,则直接运行除尘逻辑
	if (!matDustBuffer[E_DEFECT_COLOR_BRIGHT].empty())
	{
		cv::Mat matDustTemp;

		CMatBuf cMatBufTemp;

		//缓冲区分配和初始化
		cMatBufTemp.SetMem(cMem[0]);
		matDustTemp = cMatBufTemp.GetMat(matDustBuffer[E_DEFECT_COLOR_BRIGHT].size(), matDustBuffer[E_DEFECT_COLOR_BRIGHT].type(), false);

		//修改17.06.27
		//Size错误
		if (matDustBuffer[E_DEFECT_COLOR_BRIGHT].rows != matDstBuffer[E_DEFECT_COLOR_DARK].rows ||
			matDustBuffer[E_DEFECT_COLOR_BRIGHT].cols != matDstBuffer[E_DEFECT_COLOR_DARK].cols)
		{
			cv::resize(matDustBuffer[E_DEFECT_COLOR_BRIGHT], matDustTemp, matDstBuffer[E_DEFECT_COLOR_DARK].size());
		}
		else
		{
			matDustBuffer[E_DEFECT_COLOR_BRIGHT].copyTo(matDustTemp);
			//matDustTemp = matDustBuffer.clone();
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Resize & Copy."));

		//清除Dark不良Dust//Sub(不良-灰尘=真性不良)
		cv::subtract(matDstBuffer[E_DEFECT_COLOR_DARK], matDustTemp, matDstBuffer[E_DEFECT_COLOR_DARK]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("subtract (DARK)."));

		//需要从Bright中删除Dust吗？(用Dust遮挡会变暗)
		//清除Bright不良Dust
		//如果有Dust,则不亮/亮不良可以不清除dust吗？
		//17.03.02:存在因Dust而被点亮的情况
		//清除Dark不良Dust//Sub(不良-灰尘=真性不良)
		//Black 亮点过检
	//cv::subtract(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matDustTemp, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("subtract (BRIGHT)."));

		//绘制Dust结果
		if (bDrawDust)
		{
			if (!matDrawBuffer.empty())
			{
				cv::Vec3b color(0, 128, 128);

				if (matDustTemp.type() == CV_8U)
				{
					for (int y = 0; y < matDustTemp.rows; y++)
					{
						//获取竖线的第一个数组地址
						BYTE* ptr = (BYTE*)matDustTemp.ptr(y);

						for (int x = 0; x < matDustTemp.cols; x++)
						{
							//如果有Dust Image值,则添加matDraw颜色
							if (ptr[x] != 0)
								matDrawBuffer.at<cv::Vec3b>(y, x) = color;
						}
					}
				}
				else
				{
					for (int y = 0; y < matDustTemp.rows; y++)
					{
						//获取竖线的第一个数组地址
						ushort* ptr = (ushort*)matDustTemp.ptr(y);

						for (int x = 0; x < matDustTemp.cols; x++)
						{
							//如果有Dust Image值,则添加matDraw颜色
							if (ptr[x] != 0)
								matDrawBuffer.at<cv::Vec3b>(y, x) = color;
						}
					}
				}

			}
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DrawDust."));
		matDustTemp.release();

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}

	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_ResThreshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_ResThreshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);
	}

	//错误判定&发送结果
	{
		cv::Rect rectBlobROI;
		Insp_RectSet(rectBlobROI, rectROI, matSrcBuffer.cols, matSrcBuffer.rows);

		//标签
		CFeatureExtraction cFeatureExtraction;
		cFeatureExtraction.SetMem(cMem[0]);
		cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_POINT, m_tInitTime, m_tBeforeTime, m_strAlgLog);
		//默认块0 [hjf]
		STRU_DEFECT_ITEM* EngineerDefectJudgment = EngineerBlockDefectJudge[0].stDefectItem;
		if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_RETEST_POINT_DARK].bDefectItemUse)
		{
			//获取指定的Edge区域
			int nArea = (int)dPara[E_PARA_POINT_RGB_EDGE_AREA];

			//要排除在检查之外的Active Area
			cv::Rect rectEdge = rectBlobROI;
			rectEdge.y += nArea;
			rectEdge.height -= (nArea * 2);

			//只保留Edge区域
			cv::Mat matEdge = matDstBuffer[E_DEFECT_COLOR_DARK].clone();
			matEdge(rectEdge).setTo(0);

			//只保留活动区域
			cv::subtract(matDstBuffer[E_DEFECT_COLOR_DARK], matEdge, matDstBuffer[E_DEFECT_COLOR_DARK]);

			//Edge判定

			//E_DEFECT_COLOR_DARK结果
			nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matEdge(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI, nCommonPara, E_DEFECT_COLOR_DARK, _T("DP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_RETEST_POINT_DARK, FALSE);

		}

		//G87模式Point错误判定Point_Chole_Dark和Point_RGB_Dark判定
		if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_RGB_DARK].bDefectItemUse || EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_CHOLE_DARK].bDefectItemUse)
		{
			//只留下Chole区域的错误
			cv::Mat matChole = cv::Mat::zeros(matDstBuffer[E_DEFECT_COLOR_DARK].size(), CV_8UC1);

			for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
			{
				if (!matCholeBuffer[j].empty())
				{
					cv::Mat matAND;
					cv::bitwise_and(matDstBuffer[E_DEFECT_COLOR_DARK](rcCHoleROI[j]), matCholeBuffer[j], matAND);
					cv::add(matAND, matChole(rcCHoleROI[j]), matChole(rcCHoleROI[j]));
				}
			}

			//只删除和保留Chole区域
			cv::subtract(matDstBuffer[E_DEFECT_COLOR_DARK], matChole, matDstBuffer[E_DEFECT_COLOR_DARK]);

			//判定
			//G87判定
			if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_CHOLE_DARK].bDefectItemUse)
			{
				nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matChole(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI, nCommonPara, E_DEFECT_COLOR_DARK, _T("DP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_POINT_CHOLE_DARK, FALSE);
			}
			else if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_RGB_DARK].bDefectItemUse)
			{
				nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matChole(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI, nCommonPara, E_DEFECT_COLOR_DARK, _T("DP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_POINT_RGB_DARK, FALSE);
			}
		}
		//E_DEFECT_COLOR_DARK结果
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstBuffer[E_DEFECT_COLOR_DARK](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_DARK, _T("DP_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("BlobJudgment (DARK)."));

		//确认画面比例是否相同
		int nModePS = 1;
		if (matSrcBuffer.rows == matDustBuffer[E_DEFECT_COLOR_DARK].rows &&
			matSrcBuffer.cols == matDustBuffer[E_DEFECT_COLOR_DARK].cols)
			nModePS = 1;
		else
			nModePS = 2;

		//暗点-在面积较大的Dust附近移除
		//Dust无法完全检测出异物
		nErrorCode = DeleteCompareDarkPoint(matDustBuffer[E_DEFECT_COLOR_DARK], 30, pResultBlob, nModePS);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Compare Delete (Big Area Dust - Dark Point)."));

		if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT].bDefectItemUse)
		{
			//获取指定的Edge区域
			int nArea = (int)dPara[E_PARA_POINT_RGB_EDGE_AREA];

			if (nImageNum == E_IMAGE_CLASSIFY_AVI_BLACK || nImageNum == E_IMAGE_CLASSIFY_AVI_VINIT)
				nArea = (int)dPara[E_PARA_POINT_BLACK_EDGE_AREA];

			//要排除在检查之外的Active Area
			cv::Rect rectEdge = rectBlobROI;
			rectEdge.y += nArea;
			rectEdge.height -= (nArea * 2);

			//只保留Edge区域
			cv::Mat matEdge = matDstBuffer[E_DEFECT_COLOR_BRIGHT].clone();
			matEdge(rectEdge).setTo(0);

			//只保留活动区域
			cv::subtract(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matEdge, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);

			//Edge判定

			//E_DEFECT_COLOR_BRIGHT结果
			nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matEdge(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI, nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT, FALSE);

		}

		//G87模式Point错误判定Point_Chole_Dark和Point_RGB_Dark判定
		if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_RGB_BRIGHT].bDefectItemUse || EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_CHOLE_BRIGHT].bDefectItemUse)
		{
			//只留下Chole区域的错误
			cv::Mat matChole = cv::Mat::zeros(matDstBuffer[E_DEFECT_COLOR_BRIGHT].size(), CV_8UC1);

			for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
			{
				if (!matCholeBuffer[j].empty())
				{
					cv::Mat matAND;
					cv::bitwise_and(matDstBuffer[E_DEFECT_COLOR_BRIGHT](rcCHoleROI[j]), matCholeBuffer[j], matAND);
					cv::add(matAND, matChole(rcCHoleROI[j]), matChole(rcCHoleROI[j]));
				}
			}

			//只删除和保留Chole区域
			cv::subtract(matDstBuffer[E_DEFECT_COLOR_BRIGHT], matChole, matDstBuffer[E_DEFECT_COLOR_BRIGHT]);

			//判定
			//G87判定
			if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_CHOLE_BRIGHT].bDefectItemUse)
			{
				nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matChole(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI, nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_POINT_CHOLE_BRIGHT, FALSE);
			}
			else if (EngineerDefectJudgment[E_DEFECT_JUDGEMENT_POINT_RGB_BRIGHT].bDefectItemUse)
			{
				nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matChole(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI, nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BP_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_POINT_RGB_BRIGHT, FALSE);
			}
		}

		//E_DEFECT_COLOR_BRIGHT结果
		nErrorCode = cFeatureExtraction.DoDefectBlobJudgment(matSrcBuffer(rectBlobROI), matDstBuffer[E_DEFECT_COLOR_BRIGHT](rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
			nCommonPara, E_DEFECT_COLOR_BRIGHT, _T("BP_"), EngineerBlockDefectJudge, pResultBlob);
		if (nErrorCode != E_ERROR_CODE_TRUE)
		{
			//禁用内存
			matSrcBuffer.release();
			matDstBuffer[E_DEFECT_COLOR_DARK].release();
			matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

			return nErrorCode;
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("BlobJudgment (BRIGHT)."));
	}

	//禁用内存
	matSrcBuffer.release();
	matDstBuffer[E_DEFECT_COLOR_DARK].release();
	matDstBuffer[E_DEFECT_COLOR_BRIGHT].release();

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("End."));

	return E_ERROR_CODE_TRUE;
}

//检测气泡后,交出结果向量
long CInspectPoint::GetDefectList_Bubble(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob)
{
	//灰尘:设置为255GV
	//气泡:设置为200GV

	long nErrorCode = E_ERROR_CODE_TRUE;

	//将范围设置为大范围
	CRect rectROI = CRect(
		min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
		min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
		max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
		max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

	//灰尘检测
	cv::Mat matTempBuf1 = cMem[0]->GetMat(matDstBuffer[E_DEFECT_COLOR_DARK].size(), matDstBuffer[E_DEFECT_COLOR_DARK].type(), false);
	cv::threshold(matDstBuffer[E_DEFECT_COLOR_DARK], matTempBuf1, 230, 255, THRESH_BINARY);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Dust threshold."));

	//除尘
	cv::Mat matTempBuf2 = cMem[0]->GetMat(matDstBuffer[E_DEFECT_COLOR_DARK].size(), matDstBuffer[E_DEFECT_COLOR_DARK].type(), false);
	cv::subtract(matDstBuffer[E_DEFECT_COLOR_DARK], matTempBuf1, matTempBuf2);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("subtract."));

	//气泡检测
	cv::threshold(matTempBuf2, matTempBuf1, 100, 255, THRESH_BINARY);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Bubble threshold."));

	cv::Rect rectBlobROI;
	Insp_RectSet(rectBlobROI, rectROI, matSrcBuffer.cols, matSrcBuffer.rows);

	//标签
	CFeatureExtraction cFeatureExtraction;
	cFeatureExtraction.SetMem(cMem[0]);
	cFeatureExtraction.SetLog(m_cInspectLibLog, E_ALG_TYPE_AVI_POINT, m_tInitTime, m_tBeforeTime, m_strAlgLog);

	//BUBBLE结果
	nErrorCode = cFeatureExtraction.DoDefectBlobSingleJudgment(matSrcBuffer(rectBlobROI), matTempBuf1(rectBlobROI), matDrawBuffer(rectBlobROI), rectROI,
		nCommonPara, E_DEFECT_COLOR_DARK, _T("Bubble_"), EngineerBlockDefectJudge, pResultBlob, E_DEFECT_JUDGEMENT_APP_ACTIVE_BUBBLE, true);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Bubble Blob."));

	return nErrorCode;
}

//R,G,B画面检测算法
long CInspectPoint::LogicStart_RGB(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, stPanelBlockJudgeInfo* EngineerBlockDefectJudge)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	double	dDarkDist = dPara[E_PARA_POINT_RGB_COMMON_DARK_DIST];

	long	nDelLineBrightCntX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_X];		// 删除行x方向计数
	long	nDelLineBrightCntY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_Y];		// 删除行y方向计数
	long	nDelLineBrightThickX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_X];	// 删除行x厚度
	long	nDelLineBrightThickY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_Y];	// 删除行y厚度

	long	nDelLineDarkCntX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_X];		// 删除行x方向计数
	long	nDelLineDarkCntY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_Y];		// 删除行y方向计数
	long	nDelLineDarkThickX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_X];		// 删除行x厚度
	long	nDelLineDarkThickY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_Y];		// 删除行y厚度

	// Chole Point
	bool	bCholePointFlag = (dPara[E_PARA_POINT_RGB_CHOLE_POINT_FLAG] > 0) ? true : false;	// 是否使用Chole Pint
	double	nCholePoint_Ratio_B = (double)dPara[E_PARA_POINT_RGB_CHOLE_POINT_TBRIGHT_RATIO];					// Brignt Ratio
	double	nCholePoint_Ratio_D = (double)dPara[E_PARA_POINT_RGB_CHOLE_POINT_TDARK_RATIO];					// Dark Tatio

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

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	int nOffSet = 100;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectTemp.left - nOffSet;
	//rtInspROI.y = rectTemp.top - nOffSet;
	//rtInspROI.width = rectTemp.Width() + nOffSet * 2;
	//rtInspROI.height = rectTemp.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectTemp, matSrcImage.cols, matSrcImage.rows, nOffSet);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	//范围二进制化
	nErrorCode = RangeAvgThreshold(matSrcImage, matDstImage, rectTemp, dPara, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("RangeAvgThreshold."));

	if (bCholePointFlag)
	{
		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Chole Point."));

		for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
		{
			if (!matCholeBuffer[j].empty())
			{
				//检查中间映像
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Chole Area.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matSrcImage(rcCHoleROI[j]));
				}

				cv::Mat matCholeArea;
				cv::Mat matDark;
				cv::Mat matBright;

				long	nBlurLoop_5x5 = (long)dPara[E_PARA_POINT_RGB_COMMON_BLUR_LOOP];
				int nBlur = 3;

				if (nBlurLoop_5x5 > 0)
				{
					cv::blur(matSrcImage(rcCHoleROI[j]), matCholeArea, cv::Size(nBlur, nBlur));

					if (nBlurLoop_5x5 > 1)
					{
						// Avg
						for (int i = 1; i < nBlurLoop_5x5; i++)
						{
							cv::blur(matCholeArea, matCholeArea, cv::Size(nBlur, nBlur));
						}
					}
				}

				cv::bitwise_and(matCholeArea, matCholeBuffer[j], matCholeArea);

				//检查中间映像
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Chole Area_Blur.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matCholeArea);
				}

				double dAvg = cv::mean(matCholeArea, matCholeBuffer[j])[0];
				double dTh_B = dAvg * nCholePoint_Ratio_B;
				double dTh_D = dAvg * nCholePoint_Ratio_D;

				cv::threshold(matCholeArea, matBright, dTh_B, 255, THRESH_BINARY);
				cv::threshold(matCholeArea, matDark, dTh_D, 255, THRESH_BINARY_INV);

				cv::bitwise_and(matBright, matCholeBuffer[j], matBright);
				cv::bitwise_and(matDark, matCholeBuffer[j], matDark);

				//检查中间映像
				if (bImageSave)
				{
					CString strTemp;
					strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02dChole_Dark_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matDark);

					strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02dChole_Bright_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
					ImageSave(strTemp, matBright);
				}

				cv::subtract(matDstImage[E_DEFECT_COLOR_BRIGHT](rcCHoleROI[j]), matCholeBuffer[j], matDstImage[E_DEFECT_COLOR_BRIGHT](rcCHoleROI[j]));
				cv::subtract(matDstImage[E_DEFECT_COLOR_DARK](rcCHoleROI[j]), matCholeBuffer[j], matDstImage[E_DEFECT_COLOR_DARK](rcCHoleROI[j]));

				cv::add(matDstImage[E_DEFECT_COLOR_BRIGHT](rcCHoleROI[j]), matBright, matDstImage[E_DEFECT_COLOR_BRIGHT](rcCHoleROI[j]));
				cv::add(matDstImage[E_DEFECT_COLOR_DARK](rcCHoleROI[j]), matDark, matDstImage[E_DEFECT_COLOR_DARK](rcCHoleROI[j]));

				matCholeArea.release();
				matDark.release();
				matBright.release();

			}
		}
	}

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	// Distance Transform & Threshold
	nErrorCode = AlgoBase::DistanceTransformThreshold(matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), dDarkDist, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DistanceTransformThreshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Dist.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Dist.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//Bright消除小面积不良
	//Blob的速度慢。
	nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), 3, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (BRIGHT)."));

	//清除Dark小面积故障
	//Blob的速度慢。
	nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), 0, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (DARK)."));

	//删除line错误
	if (nDelLineBrightCntX > 0 || nDelLineBrightCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_BRIGHT], nDelLineBrightCntX, nDelLineBrightCntY, nDelLineBrightThickX, nDelLineBrightThickY);
	if (nDelLineDarkCntX > 0 || nDelLineDarkCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_DARK], nDelLineDarkCntX, nDelLineDarkCntY, nDelLineDarkThickX, nDelLineDarkThickY);

	//粘贴不良
//Morphology(matDstImage[E_DEFECT_COLOR_DARK],	matDstImage[E_DEFECT_COLOR_DARK],	5, 5, E_MORP_CLOSE, cMem[0]);
//Morphology(matDstImage[E_DEFECT_COLOR_BRIGHT],	matDstImage[E_DEFECT_COLOR_BRIGHT],	5, 5, E_MORP_CLOSE, cMem[0]);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Projection."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	return E_ERROR_CODE_TRUE;
}

//PNZ RGB Pattern名点未检Test Alg。
long CInspectPoint::LogicStart_RGBTest(cv::Mat** matSrcBufferRGBAdd, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, int Type, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer)
{
	//如果参数为NULL
	if (dPara == NULL)					return E_ERROR_CODE_EMPTY_PARA;
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	cv::Mat matSrcBufferRGB[3];

	//matSrcBufferRGB[0]	= *matSrcBufferRGBAdd[0];
	//matSrcBufferRGB[1]	= *matSrcBufferRGBAdd[1];
	//matSrcBufferRGB[2]	= *matSrcBufferRGBAdd[2];

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("RGB Buff Copy Start."));

	//复制RGB之前
	int TimeCount = 0;
	//while(	matSrcBufferRGB[0].empty()	|| matSrcBufferRGB[1].empty()	||	matSrcBufferRGB[2].empty()	)
	while (matSrcBufferRGBAdd[0]->empty() || matSrcBufferRGBAdd[1]->empty() || matSrcBufferRGBAdd[2]->empty())
	{
		TimeCount++;
		Sleep(10);

		// 18.05.30
		if (TimeCount >= 1000)
			return E_ERROR_CODE_FALSE;
	}

	matSrcBufferRGB[0] = *matSrcBufferRGBAdd[0];
	matSrcBufferRGB[1] = *matSrcBufferRGBAdd[1];
	matSrcBufferRGB[2] = *matSrcBufferRGBAdd[2];

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("RGB Buff Copy End."));

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	// Parameter
	double	fApplyWeight = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_APPLYGV_WEIGHT];

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

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[1]);

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	int nOffSet = 10;

	cv::Rect rtInspROI;

	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	if (Type == E_IMAGE_CLASSIFY_AVI_R && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].empty())
	{
		Insp_RectSet(rtInspROI, rectTemp, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].cols, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].rows, nOffSet);
		//创建Red Pattern Temp
		cv::Mat mat_Org = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].type(), false);
		cv::Mat matTemp_1 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].type(), false);
		cv::Mat matTemp_2 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].type(), false);
		cv::Mat matTemp_3 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].type(), false);
		cv::Mat matTemp16_1 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].size(), CV_16UC1, false);
		cv::Mat matTemp16_2 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].size(), CV_16UC1, false);

		matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].copyTo(mat_Org);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Red Pattern MatCreate."));

		// Apply Enhancement
		ApplyEnhancement(mat_Org, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G], matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B], matTemp_1, matTemp_3, dPara, nCommonPara, strAlgPath, E_IMAGE_CLASSIFY_AVI_R, &cMatBufTemp);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_ApplyEhance.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_1);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_EnhancedOrg.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_3);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Red Pattern ApplyEnhancement."));

		cv::Mat tempMean, tempStd;
		cv::meanStdDev(mat_Org, tempMean, tempStd);

		double Meanvalue = tempMean.at<double>(0, 0);	// 平均
		double Stdvalue = tempStd.at<double>(0, 0);		// 标准偏差

		AlgoBase::ApplyMeanGV(matTemp_1, Meanvalue * fApplyWeight);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_ApplyGV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_1);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Red Pattern ApplyMeanGV."));

		//创建Sub画面
		cv::blur(matTemp_3(rtInspROI), mat_Org(rtInspROI), cv::Size(3, 3));
		cv::blur(matTemp_1(rtInspROI), matTemp_2(rtInspROI), cv::Size(3, 3));

		matTemp_3(rtInspROI) = mat_Org(rtInspROI) - matTemp_2(rtInspROI);
		////////////////////////////////////////////////////////////////////////// choi 06.08
		for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
		{
			if (!matCholeBuffer[j].empty() && !rcCHoleROI[j].empty()) {
				writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("IN_!matCholeBuffer[j].empty() && !rcCHoleROI[j].empty()."));
				cv::Mat matChole_Temp = cMatBufTemp.GetMat(matCholeBuffer[j].size(), matCholeBuffer[j].type(), false);
				Scalar scMean, scStdev;
				matCholeBuffer[j].copyTo(matChole_Temp);
				cv::meanStdDev(matTemp_3(rtInspROI), scMean, scStdev);
				for (int i = 0; i < matChole_Temp.rows * matChole_Temp.cols; i++) {
					if (matChole_Temp.data[i] == 255) {
						matChole_Temp.data[i] = scMean[0];
					}
				}
				cv::Mat matTempRoi = matTemp_3(rcCHoleROI[j]);
				cv::max(matTempRoi, matChole_Temp, matTempRoi);
				matChole_Temp.release();
				matTempRoi.release();
				writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("OUT_!matCholeBuffer[j].empty() && !rcCHoleROI[j].empty()."));
			}
		}
		//////////////////////////////////////////////////////////////////////////
				//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_3);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Red Pattern Sub."));

		matTemp_3.convertTo(matTemp16_1, CV_16UC1);

		AlgoBase::Pow(matTemp16_1(rtInspROI), matTemp16_2(rtInspROI), 2.0, 4095, &cMatBufTemp);

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp16_2);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Red Pattern Pow."));

		//范围进化
		nErrorCode = RangeAvgThreshold_RGB(matTemp16_2, matDstImage, rectTemp, dPara, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_B_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);

			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Red Pattern_D_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Red Pattern Process."));
	}

	if (Type == E_IMAGE_CLASSIFY_AVI_G && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].empty())
	{
		Insp_RectSet(rtInspROI, rectTemp, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].cols, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].rows, nOffSet);
		//创建Green Pattern Temp
		cv::Mat mat_Org = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].type(), false);
		cv::Mat matTemp_1 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].type(), false);
		cv::Mat matTemp_2 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].type(), false);
		cv::Mat matTemp_3 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].type(), false);
		cv::Mat matTemp16_1 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].size(), CV_16UC1, false);
		cv::Mat matTemp16_2 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].size(), CV_16UC1, false);

		matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].copyTo(mat_Org);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Green Pattern MatCreate."));

		// Apply Enhancement
		ApplyEnhancement(mat_Org, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R], matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B], matTemp_1, matTemp_3, dPara, nCommonPara, strAlgPath, E_IMAGE_CLASSIFY_AVI_G, &cMatBufTemp);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_ApplyEhance.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_1);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_EnhancedOrg.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_3);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Green Pattern ApplyEnhancement."));

		cv::Mat tempMean, tempStd;
		cv::meanStdDev(mat_Org, tempMean, tempStd);

		double Meanvalue = tempMean.at<double>(0, 0);	// 平均
		double Stdvalue = tempStd.at<double>(0, 0);		// 标准偏差

		AlgoBase::ApplyMeanGV(matTemp_1, Meanvalue * fApplyWeight);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_ApplyGV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_1);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Green Pattern ApplyMeanGV."));

		//创建Sub画面
		//创建Sub画面
		cv::blur(matTemp_3(rtInspROI), mat_Org(rtInspROI), cv::Size(3, 3));
		cv::blur(matTemp_1(rtInspROI), matTemp_2(rtInspROI), cv::Size(3, 3));

		matTemp_3(rtInspROI) = mat_Org(rtInspROI) - matTemp_2(rtInspROI);
		//////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
		{
			if (!matCholeBuffer[j].empty() && !rcCHoleROI[j].empty()) {
				cv::Mat matChole_Temp = cMatBufTemp.GetMat(matCholeBuffer[j].size(), matCholeBuffer[j].type(), false);
				Scalar scMean, scStdev;
				matCholeBuffer[j].copyTo(matChole_Temp);
				cv::meanStdDev(matTemp_3(rtInspROI), scMean, scStdev);
				for (int i = 0; i < matChole_Temp.rows * matChole_Temp.cols; i++) {
					if (matChole_Temp.data[i] == 255) {
						matChole_Temp.data[i] = scMean[0];
					}
				}
				cv::Mat matTempRoi = matTemp_3(rcCHoleROI[j]);
				cv::max(matTempRoi, matChole_Temp, matTempRoi);
				matChole_Temp.release();
				matTempRoi.release();
			}
		}
		//////////////////////////////////////////////////////////////////////////
				//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_3);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Green Pattern Sub."));

		matTemp_3.convertTo(matTemp16_1, CV_16UC1);

		AlgoBase::Pow(matTemp16_1(rtInspROI), matTemp16_2(rtInspROI), 2.0, 4095, &cMatBufTemp);

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp16_2);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Green Pattern Pow."));

		//范围进化
		nErrorCode = RangeAvgThreshold_RGB(matTemp16_2, matDstImage, rectTemp, dPara, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_B_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);

			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Green Pattern_D_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Green Pattern Process."));
	}

	if (Type == E_IMAGE_CLASSIFY_AVI_B && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].empty())
	{
		Insp_RectSet(rtInspROI, rectTemp, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].cols, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].rows, nOffSet);
		//创建Blue Pattern Temp
		cv::Mat mat_Org = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].type(), false);
		cv::Mat matTemp_1 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].type(), false);
		cv::Mat matTemp_2 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].type(), false);
		cv::Mat matTemp_3 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].size(), matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].type(), false);
		cv::Mat matTemp16_1 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].size(), CV_16UC1, false);
		cv::Mat matTemp16_2 = cMatBufTemp.GetMat(matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].size(), CV_16UC1, false);

		matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].copyTo(mat_Org);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Blue Pattern MatCreate."));

		// Apply Enhancement
		ApplyEnhancement(mat_Org, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R], matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G], matTemp_1, matTemp_3, dPara, nCommonPara, strAlgPath, E_IMAGE_CLASSIFY_AVI_B, &cMatBufTemp);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_ApplyEhance.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_1);
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_EnhancedOrg.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_3);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Blue Pattern ApplyEnhancement."));

		cv::Mat tempMean, tempStd;
		cv::meanStdDev(mat_Org, tempMean, tempStd);

		double Meanvalue = tempMean.at<double>(0, 0);	// 平均
		double Stdvalue = tempStd.at<double>(0, 0);		// 标准偏差

		AlgoBase::ApplyMeanGV(matTemp_1, Meanvalue * fApplyWeight);

		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_ApplyGV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_1);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Blue Pattern ApplyMeanGV."));

		//创建Sub画面
		cv::blur(matTemp_3(rtInspROI), mat_Org(rtInspROI), cv::Size(3, 3));
		cv::blur(matTemp_1(rtInspROI), matTemp_2(rtInspROI), cv::Size(3, 3));

		matTemp_3(rtInspROI) = mat_Org(rtInspROI) - matTemp_2(rtInspROI);

		//cv::GaussianBlur(matTemp_2,	matTemp_1, cv::Size(5, 5), 3);
		//////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
		{
			if (!matCholeBuffer[j].empty() && !rcCHoleROI[j].empty()) {
				cv::Mat matChole_Temp = cMatBufTemp.GetMat(matCholeBuffer[j].size(), matCholeBuffer[j].type(), false);
				Scalar scMean, scStdev;
				matCholeBuffer[j].copyTo(matChole_Temp);
				cv::meanStdDev(matTemp_3(rtInspROI), scMean, scStdev);
				for (int i = 0; i < matChole_Temp.rows * matChole_Temp.cols; i++) {
					if (matChole_Temp.data[i] == 255) {
						matChole_Temp.data[i] = scMean[0];
					}
				}
				cv::Mat matTempRoi = matTemp_3(rcCHoleROI[j]);
				cv::max(matTempRoi, matChole_Temp, matTempRoi);
				matChole_Temp.release();
				matTempRoi.release();
			}
		}
		//////////////////////////////////////////////////////////////////////////

				//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp_3);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Blue Pattern Sub."));

		matTemp_3.convertTo(matTemp16_1, CV_16UC1);

		AlgoBase::Pow(matTemp16_1(rtInspROI), matTemp16_2(rtInspROI), 2.0, 4095, &cMatBufTemp);

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTemp16_2);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Blue Pattern Pow."));

		//范围进化
		nErrorCode = RangeAvgThreshold_RGB(matTemp16_2, matDstImage, rectTemp, dPara, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_B_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);

			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Blue Pattern_D_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
		}

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Blue Pattern Process."));
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//Black模式检测算法
long CInspectPoint::LogicStart_Black(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	long	nBlur1 = (long)dPara[E_PARA_POINT_BLACK_ACTIVE_BLUR_1];				// Blur1
	long	nBlur2 = (long)dPara[E_PARA_POINT_BLACK_ACTIVE_BLUR_2];				// Blur2
	long	nThresholdBright = (long)dPara[E_PARA_POINT_BLACK_ACTIVE_THRESHOLD];

	long	nDelLineBrightCntX = (long)dPara[E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_CNT_X];		// 删除行x方向计数
	long	nDelLineBrightCntY = (long)dPara[E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_CNT_Y];		// 删除行y方向计数
	long	nDelLineBrightThickX = (long)dPara[E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_THICK_X];		// 删除行x厚度
	long	nDelLineBrightThickY = (long)dPara[E_PARA_POINT_BLACK_DEL_LINE_BRIGHT_THICK_Y];		// 删除行y厚度

	long	nBigFlag = (long)dPara[E_PARA_POINT_BLACK_BIG_FLAG];						// 使用大不良&低GV不良检测有/无
	long	nBigMinArea = (long)dPara[E_PARA_POINT_BLACK_BIG_MIN_AREA];					// 大不良的最小面积
	long	nBigMaxGV = (long)dPara[E_PARA_POINT_BLACK_BIG_MAX_GV];					// 低GV的最大GV

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

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	int nOffSet = 50;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nOffSet);

	// Blur
	cv::Mat matTempBuf1, matTempBuf2;
	matTempBuf1 = cMem[0]->GetMat(matSrcImage.size(), matSrcImage.type(), false);
	matTempBuf2 = cMem[0]->GetMat(matSrcImage.size(), matSrcImage.type(), false);

	cv::blur(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), cv::Size(nBlur1, nBlur1));
	cv::blur(matTempBuf1(rtInspROI), matTempBuf2(rtInspROI), cv::Size(nBlur2, nBlur2));

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("blur."));

	cv::subtract(matTempBuf1, matTempBuf2, matTempBuf1);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("subtract."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Sub.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf1);
	}

	// Threshold
	cv::Mat matROI1 = matTempBuf1(cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));
	cv::Mat matROI2 = matDstImage[E_DEFECT_COLOR_BRIGHT](cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()));

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Mat Create."));

	// Threshold
	nErrorCode = AlgoBase::Binary(matROI1, matROI2, nThresholdBright, false, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("threshold."));

	//删除line错误
	if (nDelLineBrightCntX > 0 || nDelLineBrightCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_BRIGHT], nDelLineBrightCntX, nDelLineBrightCntY, nDelLineBrightThickX, nDelLineBrightThickY);

	//粘贴不良
//Morphology(matDstImage[E_DEFECT_COLOR_BRIGHT],	matDstImage[E_DEFECT_COLOR_BRIGHT],	5, 5, E_MORP_CLOSE, cMem[0]);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Projection."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//////////////////////////////////////////////////////////////////////////
		//查找区域较大,GV值较低的情况
	//////////////////////////////////////////////////////////////////////////
	if (nBigFlag > 0)
	{
		// Threshold
		nErrorCode = AlgoBase::Binary(matSrcImage(rtInspROI), matTempBuf2(rtInspROI), 0, false, cMem[0]);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		//粘贴不良内容
		nErrorCode = Morphology(matTempBuf2(rtInspROI), matTempBuf1(rtInspROI), 5, 5, E_MORP_CLOSE);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Big_Point.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTempBuf1);
		}

		nErrorCode = DeleteMinArea(matSrcImage(rtInspROI), matTempBuf1(rtInspROI), nBigMinArea, nBigMaxGV, cMem[0]);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		cv::bitwise_or(matTempBuf1, matDstImage[E_DEFECT_COLOR_BRIGHT], matDstImage[E_DEFECT_COLOR_BRIGHT]);

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Big_Point_MinArea.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matTempBuf1);

			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Big_Point_or.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
		}
	}
	//////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Find Big Area."));

	matROI1.release();
	matROI2.release();
	matTempBuf1.release();
	matTempBuf2.release();

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Release."));

	return E_ERROR_CODE_TRUE;
}

//Gray模式检测算法
long CInspectPoint::LogicStart_Gray(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGBAdd, cv::Mat matBKBuffer, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	double	dPow = dPara[E_PARA_POINT_RGB_COMMON_POW];
	double	dDarkDist = dPara[E_PARA_POINT_RGB_COMMON_DARK_DIST];

	long	nDelLineBrightCntX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_X];		// 删除行x方向计数
	long	nDelLineBrightCntY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_Y];		// 删除行y方向计数
	long	nDelLineBrightThickX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_X];	// 删除行x厚度
	long	nDelLineBrightThickY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_Y];	// 删除行y厚度

	long	nDelLineDarkCntX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_X];		// 删除行x方向计数
	long	nDelLineDarkCntY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_Y];		// 删除行y方向计数
	long	nDelLineDarkThickX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_X];		// 删除行x厚度
	long	nDelLineDarkThickY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_Y];		// 删除行y厚度

	double	bAdjustGrayR_AdjustRatio = (double)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_ADJUST_RATIO];
	int		bAdjustGrayR_AdjustCutMinGV = (int)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_CUT_MINGV];

	double	bAdjustGrayG_AdjustRatio = (double)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_ADJUST_RATIO];
	int		bAdjustGrayG_AdjustCutMinGV = (int)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_CUT_MINGV];

	double	bAdjustGrayB_AdjustRatio = (double)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_USE];
	int		bAdjustGrayB_AdjustCutMinGV = (int)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_CUT_MINGV];

	bool	bAdjustGrayR_Use = (dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_USE] > 0) ? true : false;
	bool	bAdjustGrayG_Use = (dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_USE] > 0) ? true : false;
	bool	bAdjustGrayB_Use = (dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_USE] > 0) ? true : false;

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
	int		nRatio = nCommonPara[E_PARA_COMMON_PS_MODE];

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	cv::Mat matSrcBufferRGB[3];

	matSrcBufferRGB[0] = *matSrcBufferRGBAdd[0];
	matSrcBufferRGB[1] = *matSrcBufferRGBAdd[1];
	matSrcBufferRGB[2] = *matSrcBufferRGBAdd[2];

	//预先复制,以防禁用Adjust功能
	cv::Mat matAdjustTempBuf = cMem[0]->GetMat(matSrcImage.size(), matSrcImage.type(), false);
	matSrcImage.copyTo(matAdjustTempBuf);

	CRect rectTemp_For_Adj;
	rectTemp_For_Adj.left = 0;
	rectTemp_For_Adj.top = 0;
	rectTemp_For_Adj.right = nWidth;
	rectTemp_For_Adj.bottom = nHeight;

	//由于校正前后亮度略有变化,外围区域也有所变化,因此增加了校正值
	double dblAve_BEF = AlgoBase::GetAverage(matAdjustTempBuf);
	double dblAve_AFT, dblAve_ADJ_Ratio;

	//使用R模式校正
	if (bAdjustGrayR_Use && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].empty())
	{
		AdjustImageWithRGB(matAdjustTempBuf, matAdjustTempBuf, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R], bAdjustGrayR_AdjustRatio, bAdjustGrayR_AdjustCutMinGV, rectTemp_For_Adj, cMem[0]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Adjust R."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Adjust_R.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matAdjustTempBuf);
		}
	}

	//使用G模式校正
	if (bAdjustGrayG_Use && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].empty())
	{
		AdjustImageWithRGB(matAdjustTempBuf, matAdjustTempBuf, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G], bAdjustGrayG_AdjustRatio, bAdjustGrayG_AdjustCutMinGV, rectTemp_For_Adj, cMem[0]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Adjust G."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Adjust_G.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matAdjustTempBuf);
		}
	}

	//使用B模式校正
	if (bAdjustGrayB_Use && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].empty())
	{
		AdjustImageWithRGB(matAdjustTempBuf, matAdjustTempBuf, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B], bAdjustGrayB_AdjustRatio, bAdjustGrayB_AdjustCutMinGV, rectTemp_For_Adj, cMem[0]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Adjust B."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Adjust_B.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matAdjustTempBuf);
		}
	}

	//由于校正前后亮度略有变化,外围区域也有所变化,因此增加了校正值
	dblAve_AFT = AlgoBase::GetAverage(matAdjustTempBuf);
	dblAve_ADJ_Ratio = dblAve_AFT / dblAve_BEF;

	//////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////// choi 06.08
	for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
	{
		if (!matBKBuffer.empty() && !matCholeBuffer[j].empty()) {
			cv::Mat matChole_Temp = cMem[0]->GetMat(matCholeBuffer[j].size(), matCholeBuffer[j].type(), false);
			Scalar scMean, scStdev;
			matCholeBuffer[j].copyTo(matChole_Temp);
			cv::Mat matTempRoi = matBKBuffer(rcCHoleROI[j]);
			cv::add(matTempRoi, matChole_Temp, matTempRoi);

			matChole_Temp.release();
			matTempRoi.release();
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

		//对齐外围区域的GV。
	AlgoBase::AdjustBkGV(matAdjustTempBuf, matBKBuffer, dblAve_ADJ_Ratio, cMem[0]);

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_AdjustBK.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matAdjustTempBuf);
	}

	int nOffSet = 100;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nOffSet);

	//17.09.05-添加Pow
	cv::Mat matTempBuf = cMem[0]->GetMat(matSrcImage.size(), CV_16U);
	nErrorCode = AlgoBase::Pow(matAdjustTempBuf(rtInspROI), matTempBuf(rtInspROI), dPow, 4095, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Pow."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf);
	}

	//范围二进制化
	nErrorCode = RangeAvgThreshold(matTempBuf, matDstImage, rectTemp, dPara, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	matTempBuf.release();

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("RangeAvgThreshold."));

	nOffSet = 10;

	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nOffSet);

	// Distance Transform & Threshold
	nErrorCode = AlgoBase::DistanceTransformThreshold(matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), dDarkDist, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DistanceTransformThreshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DistanceTransformThreshold Off"));

		//Bright消除小面积不良
		//Blob的速度慢。
		//17.12.01-去除气泡亮点时,必须在White模式中检测并进行比较
		//如果不是P/S模式,也存在在White模式中检测到1 Pixel的情况
	if (nRatio == 2 || nImageNum != E_IMAGE_CLASSIFY_AVI_WHITE)
		nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), 2, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (BRIGHT)."));

	//清除Dark小面积故障
	//Blob的速度慢。
	nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), 2, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (DARK)."));

	//删除line错误
	if (nDelLineBrightCntX > 0 || nDelLineBrightCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_BRIGHT], nDelLineBrightCntX, nDelLineBrightCntY, nDelLineBrightThickX, nDelLineBrightThickY);
	if (nDelLineDarkCntX > 0 || nDelLineDarkCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_DARK], nDelLineDarkCntX, nDelLineDarkCntY, nDelLineDarkThickX, nDelLineDarkThickY);

	//粘贴不良
//Morphology(matDstImage[E_DEFECT_COLOR_DARK],	matDstImage[E_DEFECT_COLOR_DARK],	5, 5, E_MORP_CLOSE, cMem[0]);
//Morphology(matDstImage[E_DEFECT_COLOR_BRIGHT],	matDstImage[E_DEFECT_COLOR_BRIGHT],	5, 5, E_MORP_CLOSE, cMem[0]);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Projection."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	return E_ERROR_CODE_TRUE;
}

//Gray模式检测算法
long CInspectPoint::LogicStart_Gray2(cv::Mat& matSrcImage, cv::Mat** matSrcBufferRGBAdd, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	double	dPow = dPara[E_PARA_POINT_RGB_COMMON_POW];
	double	dDarkDist = dPara[E_PARA_POINT_RGB_COMMON_DARK_DIST];

	long	nDelLineBrightCntX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_X];		// 删除行x方向计数
	long	nDelLineBrightCntY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_CNT_Y];		// 删除行y方向计数
	long	nDelLineBrightThickX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_X];	// 删除行x厚度
	long	nDelLineBrightThickY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_BRIGHT_THICK_Y];	// 删除行y厚度

	long	nDelLineDarkCntX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_X];		// 删除行x方向计数
	long	nDelLineDarkCntY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_CNT_Y];		// 删除行y方向计数
	long	nDelLineDarkThickX = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_X];		// 删除行x厚度
	long	nDelLineDarkThickY = (long)dPara[E_PARA_POINT_RGB_DEL_LINE_DARK_THICK_Y];		// 删除行y厚度

	double	bAdjustGrayR_AdjustRatio = (double)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_ADJUST_RATIO];
	int		bAdjustGrayR_AdjustCutMinGV = (int)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_CUT_MINGV];

	double	bAdjustGrayG_AdjustRatio = (double)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_ADJUST_RATIO];
	int		bAdjustGrayG_AdjustCutMinGV = (int)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_CUT_MINGV];

	double	bAdjustGrayB_AdjustRatio = (double)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_ADJUST_RATIO];
	int		bAdjustGrayB_AdjustCutMinGV = (int)dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_CUT_MINGV];

	bool	bAdjustGrayR_Use = (dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_R_USE] > 0) ? true : false;
	bool	bAdjustGrayG_Use = (dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_G_USE] > 0) ? true : false;
	bool	bAdjustGrayB_Use = (dPara[E_PARA_POINT_RGB_ADJUST_GRAY_WITH_RGB_B_USE] > 0) ? true : false;

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
	int		nRatio = nCommonPara[E_PARA_COMMON_PS_MODE];

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	cv::Mat matSrcBufferRGB[3];

	matSrcBufferRGB[0] = *matSrcBufferRGBAdd[0];
	matSrcBufferRGB[1] = *matSrcBufferRGBAdd[1];
	matSrcBufferRGB[2] = *matSrcBufferRGBAdd[2];

	//预先复制,以防禁用Adjust功能
	cv::Mat matAdjustTempBuf = cMem[0]->GetMat(matSrcImage.size(), matSrcImage.type(), false);
	matSrcImage.copyTo(matAdjustTempBuf);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cv::Mat weak_test_ori;
	cv::Mat weak_test_R;
	cv::Mat weak_test_G;
	cv::Mat weak_test_B;
	matSrcImage.copyTo(weak_test_ori);
	matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].copyTo(weak_test_R);
	matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].copyTo(weak_test_G);
	matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].copyTo(weak_test_B);

	//从subtract gray中获取R/G/B

// 	for (int i= rectROI.TopLeft().y; i< rectROI.BottomRight().y; i++)
// 	{
// 		for (int j= rectROI.TopLeft().x; j< rectROI.BottomRight().x; j++)
// 		{
// 			weak_test_ori.data[i*matSrcImage.cols + j]= 255;
// 		}
// 	}

// 	if (bImageSave)
// 	{
// 		CString strTemp;
// 		strTemp.Format(_T("%s\\1_ori.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, weak_test_ori);
// 
// 		strTemp.Format(_T("%s\\1_R.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, weak_test_R);
// 	
// 
// 		strTemp.Format(_T("%s\\1_G.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, weak_test_G);
// 	
// 
// 		strTemp.Format(_T("%s\\1_B.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
// 		ImageSave(strTemp, weak_test_B);
// 	}

	////////////直接用于第一个threshold源,过滤点亮区域
	for (int i = 0; i < weak_test_R.rows * weak_test_R.cols; i++)
	{
		if (weak_test_R.data[i] < (double)30.0)
		{
			weak_test_R.data[i] = (double)0.0;
		}

	}

	for (int i = 0; i < weak_test_G.rows * weak_test_G.cols; i++)
	{
		if (weak_test_G.data[i] < (double)25.0)
		{
			weak_test_G.data[i] = (double)0.0;
		}

	}

	for (int i = 0; i < weak_test_B.rows * weak_test_B.cols; i++)
	{
		if (weak_test_B.data[i] < (double)40.0)
		{
			weak_test_B.data[i] = (double)0.0;
		}

	}

	//	cv::threshold(weak_test_G,weak_test_G,30,255, THRESH_BINARY);

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\1_1_pre_threshold_R.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_R);

		strTemp.Format(_T("%s\\1_1_pre_threshold_G.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_G);

		strTemp.Format(_T("%s\\1_1_pre_threshold_B.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_B);

	}

	//R获取
	cv::subtract(matSrcImage, weak_test_B, weak_test_ori);
	cv::subtract(weak_test_ori, weak_test_G, weak_test_R);

	//G获取
	cv::subtract(matSrcImage, weak_test_R, weak_test_ori);
	cv::subtract(weak_test_ori, weak_test_B, weak_test_G);

	//B获取
	cv::subtract(matSrcImage, weak_test_R, weak_test_ori);
	cv::subtract(weak_test_ori, weak_test_G, weak_test_B);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\1_2_subtract_R.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_R);
		strTemp.Format(_T("%s\\1_2_subtract_G.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_G);
		strTemp.Format(_T("%s\\1_2_subtract_B.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_B);
	}

	////////////提取从二次threshold gray中提取的R/G/B点亮区域
	int bright_count = 0;

	for (int i = 0; i < weak_test_R.rows * weak_test_R.cols; i++)
	{
		if (weak_test_R.data[i] < (double)35.0)
		{
			weak_test_R.data[i] = (double)0.0;
		}

	}

	for (int i = 0; i < weak_test_G.rows * weak_test_G.cols; i++)
	{
		if (weak_test_G.data[i] < (double)25.0)
		{
			weak_test_G.data[i] = (double)0.0;
		}

	}

	for (int i = 0; i < weak_test_B.rows * weak_test_B.cols; i++)
	{
		if (weak_test_B.data[i] < (double)40.0)
		{
			weak_test_B.data[i] = (double)0.0;
		}

	}

	//	cv::threshold(weak_test_G,weak_test_G,30,255, THRESH_BINARY);

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\1_3_pre_threshold_R.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_R);

		strTemp.Format(_T("%s\\1_3_pre_threshold_G.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_G);

		strTemp.Format(_T("%s\\1_3_pre_threshold_B.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_B);

	}

	//开始区域分割
	int test_width = (rectROI.BottomRight().x - rectROI.TopLeft().x) / 100;
	int test_height = (rectROI.BottomRight().y - rectROI.TopLeft().y) / 100;

	// 	 	for (int i= rectROI.TopLeft().y; i< rectROI.BottomRight().y; i++)
	// 	 	{
	// 	 		for (int j= rectROI.TopLeft().x; j< rectROI.BottomRight().x; j++)
	// 	 		{
	// 				for (int x = 0; x < test_height; x++)
	// 				{
	// 					for (int y = 0; x < test_width; y++)
	// 					{
	// 						weak_test_ori.data[(i+x )*weak_test_ori.cols + (j+y)] = 255;
	// 					}
	// 				}
	// 	 		}
	// 		}

	// 	for (int i = rectROI.TopLeft().y; i < rectROI.BottomRight().y - test_height; i = i + test_height)
	// 	{
	// 		for (int j = rectROI.TopLeft().x; j < rectROI.BottomRight().x - test_width; j = j + test_width)
	// 		{
	// 			for (int x = 0; x < test_height; x++)
	// 			{
	// 				for (int y = 0; y < test_width; y++)
	// 				{
	// 					if (weak_test_ori.data[(x + i)*matSrcImage.cols + (y + j)] <= 50)
	// 						weak_test_ori.data[(x + i)*matSrcImage.cols + (y + j)] = 0;
	// 				}
	// 			}
	// 
	// 		}
	// 	}
	double test_count = 0;
	double test_mean = 0;
	double test_gv_sum = 0;
	int test_hist[256] = { 0 };

	for (int i = rectROI.TopLeft().y; i < rectROI.BottomRight().y - test_height; i = i + test_height)
	{
		for (int j = rectROI.TopLeft().x; j < rectROI.BottomRight().x - test_width; j = j + test_width)
		{

			/////过滤第一个点亮区域-G模式
			for (int x = 0; x < test_height; x++)
			{
				for (int y = 0; y < test_width; y++)
				{
					if (weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] <= 25)
						weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] = 0;
					else
					{

						test_hist[weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)]]++;
						test_count++;
						test_gv_sum = test_gv_sum + weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)];
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//choikwangil

			// 			cv::Mat histo_weak;
			// 		    cv::Point2i weak_Point_LT(j, i);
			// 			cv::Point2i weak_Point_RB(j + test_width, i + test_height);
			// 			cv::Rect weak_rect(weak_Point_LT,weak_Point_RB);
			// 
			// 			GetHistogram(weak_test_G(weak_rect), histo_weak);

			int h_count_sum = 0;

			for (int h = 0; h < 256; h++) {
				if (test_hist[h] != 0) {
					h_count_sum += test_hist[h];
				}
			}

			double high_sumGV = 0;
			double high_meanGV = 0;
			int com_count = 0;

			h_count_sum *= 0.1;

			for (int h = 255; h >= 0; h--) {
				if (test_hist[h] != 0) {
					com_count += test_hist[h];
					if (com_count <= h_count_sum) {
						high_sumGV += test_hist[h] * h;
						high_meanGV = high_sumGV / com_count;
					}
					else {
						break;
					}
				}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
						///////////启动第二个threshold-
			/*if (test_count > 0)
			test_mean = (test_gv_sum / test_count) + 20;
			else
			test_mean = 0;*/

			for (int x = 0; x < test_height; x++)
			{
				for (int y = 0; y < test_width; y++)
				{
					if (weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] <= high_meanGV + 15)//choikwangil更改test_mean->high_meanGV
					{
						weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] = 0;
					}
					//weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] = 0;

				}
			}

			test_hist[256] = { 0 };
			test_gv_sum = 0;
			test_count = 0;

			///////////////////////////////////////////////////////////////////
						/////过滤第一个点等区域-R模式////////////////////////////////////////////////////////////////////////////////////
			for (int x = 0; x < test_height; x++)
			{
				for (int y = 0; y < test_width; y++)
				{
					if (weak_test_R.data[(x + i) * matSrcImage.cols + (y + j)] <= 30)
						weak_test_R.data[(x + i) * matSrcImage.cols + (y + j)] = 0;
					else
					{
						test_count++;
						test_gv_sum = test_gv_sum + weak_test_R.data[(x + i) * matSrcImage.cols + (y + j)];
					}
				}
			}

			///////////启动第二个threshold-R
			if (test_count > 0)
				test_mean = (test_gv_sum / test_count) + 25;
			else
				test_mean = 0;

			for (int x = 0; x < test_height; x++)
			{
				for (int y = 0; y < test_width; y++)
				{
					if (weak_test_R.data[(x + i) * matSrcImage.cols + (y + j)] <= test_mean)
					{
						weak_test_R.data[(x + i) * matSrcImage.cols + (y + j)] = 0;
					}
					//weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] = 0;

				}
			}

			test_gv_sum = 0;
			test_count = 0;

			///////////////////////////////////////////////////////////////////
						/////过滤第一个点等区域-B模式////////////////////////////////////////////////////////
			for (int x = 0; x < test_height; x++)
			{
				for (int y = 0; y < test_width; y++)
				{
					if (weak_test_B.data[(x + i) * matSrcImage.cols + (y + j)] <= 40)
						weak_test_B.data[(x + i) * matSrcImage.cols + (y + j)] = 0;
					else
					{
						test_count++;
						test_gv_sum = test_gv_sum + weak_test_B.data[(x + i) * matSrcImage.cols + (y + j)];
					}
				}
			}

			///////////启动第二个threshold-B
			if (test_count > 0)
				test_mean = (test_gv_sum / test_count) + 30;
			else
				test_mean = 0;

			for (int x = 0; x < test_height; x++)
			{
				for (int y = 0; y < test_width; y++)
				{
					if (weak_test_B.data[(x + i) * matSrcImage.cols + (y + j)] <= test_mean)
					{
						weak_test_B.data[(x + i) * matSrcImage.cols + (y + j)] = 0;
					}
					//weak_test_G.data[(x + i) * matSrcImage.cols + (y + j)] = 0;

				}
			}

			test_gv_sum = 0;
			test_count = 0;

		}
	}

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\1_4_range_threshold_G.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_G);
	}

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\1_4_range_threshold_R.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_R);
	}

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\1_4_range_threshold_B.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_B);
	}

	//	cv::add(weak_test_G, weak_test_R, weak_test_G);
	//	cv::add(weak_test_G, weak_test_B, weak_test_G);

	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\1_5_final.bmp"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, weak_test_G);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//使用	//R模式校正
	if (bAdjustGrayR_Use && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R].empty())
	{
		AdjustImageWithRGB(matAdjustTempBuf, matAdjustTempBuf, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_R], bAdjustGrayR_AdjustRatio, bAdjustGrayR_AdjustCutMinGV, rectTemp, cMem[0]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Adjust R."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Adjust_R.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matAdjustTempBuf);
		}
	}

	//使用	//G模式校正
	if (bAdjustGrayG_Use && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G].empty())
	{
		AdjustImageWithRGB(matAdjustTempBuf, matAdjustTempBuf, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_G], bAdjustGrayG_AdjustRatio, bAdjustGrayG_AdjustCutMinGV, rectTemp, cMem[0]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Adjust G."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Adjust_G.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matAdjustTempBuf);
		}
	}

	//使用	//B模式校正
	if (bAdjustGrayB_Use && !matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B].empty())
	{
		AdjustImageWithRGB(matAdjustTempBuf, matAdjustTempBuf, matSrcBufferRGB[E_IMAGE_CLASSIFY_AVI_B], bAdjustGrayB_AdjustRatio, bAdjustGrayB_AdjustCutMinGV, rectTemp, cMem[0]);

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Adjust B."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Adjust_B.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matAdjustTempBuf);
		}
	}

	int nOffSet = 100;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nOffSet);

	//17.09.05-添加Pow
	cv::Mat matTempBuf = cMem[0]->GetMat(matSrcImage.size(), CV_16U);
	nErrorCode = AlgoBase::Pow(matAdjustTempBuf(rtInspROI), matTempBuf(rtInspROI), dPow, 4095, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Pow."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Pow.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf);
	}

	//范围二进制化
	nErrorCode = RangeAvgThreshold(matTempBuf, matDstImage, rectTemp, dPara, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	matTempBuf.release();

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("RangeAvgThreshold."));

	nOffSet = 10;

	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nOffSet);

	// Distance Transform & Threshold
	nErrorCode = AlgoBase::DistanceTransformThreshold(matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), dDarkDist, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DistanceTransformThreshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Threshold.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DistanceTransformThreshold Off"));

		//Bright消除小面积不良
		//Blob的速度慢。
		//17.12.01-去除气泡亮点时,必须在White模式中检测并进行比较
		//如果不是P/S模式,也存在在White模式中检测到1 Pixel的情况
	if (nRatio == 2 || nImageNum != E_IMAGE_CLASSIFY_AVI_WHITE)
		nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), 2, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (BRIGHT)."));

	//清除Dark小面积故障
	//Blob的速度慢。
	nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), 1, cMem[0]); // PWJ12.10源->nErrorCode=DeleteArea(matDstImage[E_DEFECT_COLOR_DARK](rtInsproi),2,cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (DARK)."));

	//删除line错误
	if (nDelLineBrightCntX > 0 || nDelLineBrightCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_BRIGHT], nDelLineBrightCntX, nDelLineBrightCntY, nDelLineBrightThickX, nDelLineBrightThickY);
	if (nDelLineDarkCntX > 0 || nDelLineDarkCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_DARK], nDelLineDarkCntX, nDelLineDarkCntY, nDelLineDarkThickX, nDelLineDarkThickY);

	//粘贴不良
//Morphology(matDstImage[E_DEFECT_COLOR_DARK],	matDstImage[E_DEFECT_COLOR_DARK],	5, 5, E_MORP_CLOSE, cMem[0]);
//Morphology(matDstImage[E_DEFECT_COLOR_BRIGHT],	matDstImage[E_DEFECT_COLOR_BRIGHT],	5, 5, E_MORP_CLOSE, cMem[0]);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Projection."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Dark_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	return E_ERROR_CODE_TRUE;
}

//Dust模式检测算法

long CInspectPoint::LogicStart_DustALL(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, cv::Rect* rcCHoleROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//////////////////////////////////////////////////////////////////////////
	// Dust Parameter
	int		nShiftRanage = (int)dPara[E_PARA_POINT_DUST_ENHANCMENT_SHIFT_RANGE];
	int		nGaussianSize = (int)dPara[E_PARA_POINT_DUST_ENHANCMENT_GAUSSIAN_SIZE];
	float	fGaussianSigma = (float)dPara[E_PARA_POINT_DUST_ENHANCMENT_GAUSSIAN_SIGMA];
	int		nMinMaxSize = (int)dPara[E_PARA_POINT_DUST_ENHANCMENT_MINMAX_SIZE];
	int		nDeletArea = (int)dPara[E_PARA_POINT_DUST_LOGIC_DELET_AREA];
	int		nDustDilate = (int)dPara[E_PARA_POINT_DUST_LOGIC_MORP_RANGE];
	int		nFindMinArea = (int)dPara[E_PARA_POINT_DUST_LOGIC_BIG_AREA];

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

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Src.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matSrcImage);
	}

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem[0]);

	//复制到缓冲区
	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matTempBuf2 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	cv::Mat matTempBuf3 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	matSrcImage.copyTo(matTempBuf1);

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	int nOffSet = 100;

	cv::Rect rtInspROI;

	//rtInspROI.x = rectROI.left - nOffSet;
	//rtInspROI.y = rectROI.top - nOffSet;
	//rtInspROI.width = rectROI.Width() + nOffSet * 2;
	//rtInspROI.height = rectROI.Height() + nOffSet * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nOffSet);

	//////////////////////////////////////////////////////////////////////////
	//开始预处理

		// Image Enhancement
	AlgoBase::ShiftCopy(matTempBuf1(rtInspROI), matTempBuf2(rtInspROI), nShiftRanage, 0, 1, 0, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("ShiftCopy."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_ShiftCopy.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf2);
	}

	//rtInspROI.x = rectROI.left - nGaussianSize;
	//rtInspROI.y = rectROI.top - nGaussianSize;
	//rtInspROI.width = rectROI.Width() + nGaussianSize * 2;
	//rtInspROI.height = rectROI.Height() + nGaussianSize * 2;

	Insp_RectSet(rtInspROI, rectROI, matTempBuf2.cols, matTempBuf2.rows, nGaussianSize);

	// Blur
	cv::GaussianBlur(matTempBuf2(rtInspROI), matTempBuf3(rtInspROI), cv::Size(nGaussianSize, nGaussianSize), fGaussianSigma);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("blur."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_GaussianBlur.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf3);
	}

	//18.05.21异常处理
	if (nMinMaxSize < 3)	nMinMaxSize = 3;

	//rtInspROI.x = rectROI.left - nMinMaxSize;
	//rtInspROI.y = rectROI.top - nMinMaxSize;
	//rtInspROI.width = rectROI.Width() + nMinMaxSize * 2;
	//rtInspROI.height = rectROI.Height() + nMinMaxSize * 2;

	Insp_RectSet(rtInspROI, rectROI, matTempBuf3.cols, matTempBuf3.rows, nMinMaxSize);

	matTempBuf1.setTo(0);
	matTempBuf2.setTo(0);
	// Min-Max Filtering
	cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(nMinMaxSize, nMinMaxSize), cv::Point(-1, -1));

	cv::morphologyEx(matTempBuf3(rtInspROI), matTempBuf1(rtInspROI), MORPH_ERODE, StructElem);	// MORPH_ERODE  (Min)
	cv::morphologyEx(matTempBuf3(rtInspROI), matTempBuf2(rtInspROI), MORPH_DILATE, StructElem);	// MORPH_DILATE (Max)

	///2020202.05.23用于清除因DUST图案Min-Max膨胀而产生的轮廓
	cv::Mat matThreshold = cv::Mat::zeros(matTempBuf3.size(), CV_8UC1);
	cv::Mat matThreshold_min = cv::Mat::zeros(matTempBuf3.size(), CV_8UC1);
	cv::Mat matThreshold_max = cv::Mat::zeros(matTempBuf3.size(), CV_8UC1);

	cv::threshold(matTempBuf3, matThreshold, 20, 255, THRESH_BINARY);

	cv::morphologyEx(matThreshold(rtInspROI), matThreshold_min(rtInspROI), MORPH_ERODE, StructElem);	// MORPH_ERODE  (Min)
	cv::morphologyEx(matThreshold(rtInspROI), matThreshold_max(rtInspROI), MORPH_DILATE, StructElem);	// MORPH_DILATE (Max)

	matThreshold = matThreshold_max - matThreshold_min;

	matThreshold_min.release();
	matThreshold_max.release();
	/////////////////////////////////////////////////////////////////////////////

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Min-Max Filter."));

	StructElem.release();

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_MinFilter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf1);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_MaxFilter.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf2);

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_MaxFilter_Edge.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matThreshold);
	}

	// Max - Min SubImage Calculation
	matTempBuf3 = matTempBuf2 - matTempBuf1;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Min-Max Substract."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Sub Image.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf3);
	}

	//////////////////////////////////////////////////////////////////////////

	nErrorCode = RangeAvgThreshold_DUST(matTempBuf3, matDstImage, rectTemp, dPara, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DUST RangeAvgThreshold."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Threshold_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Threshold_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
	}

	//内存故障
	matTempBuf1.release();
	matTempBuf2.release();
	matTempBuf3.release();

	//删除CHole
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		cv::Point ptCHoleCenter;
		ptCHoleCenter.x = rcCHoleROI[i].x + rcCHoleROI[i].width / 2;
		ptCHoleCenter.y = rcCHoleROI[i].y + rcCHoleROI[i].height / 2;
		int nCHoleraius = (rcCHoleROI[i].width + rcCHoleROI[i].height) / 2;

		cv::circle(matDstImage[E_DEFECT_COLOR_BRIGHT], ptCHoleCenter, nCHoleraius, cv::Scalar(0), -1);
	}

	//////////////////////////////////////////////////////////////////////////

			//只保留点灯区域
	nErrorCode = DeleteOutArea(matDstImage[E_DEFECT_COLOR_BRIGHT], rectROI, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteOutArea."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_DeletOutArea_Image.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//rtInspROI.x = rectROI.left;
	//rtInspROI.y = rectROI.top;
	//rtInspROI.width = rectROI.Width();
	//rtInspROI.height = rectROI.Height();

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows);

	//Bright消除小面积不良
	//Blob的速度慢。
	nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), nDeletArea, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	//Dust不好的情况很多,E级判定检查函数
	nErrorCode = JudgementCheckE(matDstImage[E_DEFECT_COLOR_BRIGHT], dPara, rectROI, &cMatBufTemp);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("E Check."));

	//除尘时,稍微放大一点即可。
	if (nDustDilate > 0)
	{
		//rtInspROI.x = rectROI.left - nDustDilate;
		//rtInspROI.y = rectROI.top - nDustDilate;
		//rtInspROI.width = rectROI.Width() + nDustDilate * 2;
		//rtInspROI.height = rectROI.Height() + nDustDilate * 2;

		Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nDustDilate);

		nErrorCode = Morphology(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), nDustDilate, nDustDilate, E_MORP_DILATE);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Morphology."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Dilate.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
		}
	}

	//灰尘大时-->用于清除周围暗点
	//对于灰尘,目前Dust照明无法全部检测到
	{
		//rtInspROI.x = rectROI.left;
		//rtInspROI.y = rectROI.top;
		//rtInspROI.width = rectROI.Width();
		//rtInspROI.height = rectROI.Height();

		Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows);

		nErrorCode = FindBigAreaDust(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), matDstImage[E_DEFECT_COLOR_DARK](rtInspROI), nFindMinArea, &cMatBufTemp);
		if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

		writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Find Big Area."));

		//检查中间映像
		if (bImageSave)
		{
			CString strTemp;
			strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Big_Area.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
			ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
		}
	}

	return E_ERROR_CODE_TRUE;
}

//保存8bit和12bit画面
long CInspectPoint::ImageSave(CString strPath, cv::Mat matSrcBuf)
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

//分区进化
long CInspectPoint::RangeAvgThreshold(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	if (matSrcImage.type() == CV_8U)
		nErrorCode = RangeAvgThreshold_8bit(matSrcImage, matDstImage, rectROI, dPara, cMemSub);
	else
		nErrorCode = RangeAvgThreshold_16bit(matSrcImage, matDstImage, rectROI, dPara, cMemSub);

	return nErrorCode;
}

//分区进化
long CInspectPoint::RangeAvgThreshold_8bit(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub)
{
	long	nBlurLoop_5x5 = (long)dPara[E_PARA_POINT_RGB_COMMON_BLUR_LOOP];
	long	nSegX = (long)dPara[E_PARA_POINT_RGB_COMMON_SEG_X];
	long	nSegY = (long)dPara[E_PARA_POINT_RGB_COMMON_SEG_Y];
	long	nEdgeArea = (long)dPara[E_PARA_POINT_RGB_EDGE_AREA];

	float	fActiveDarkRatio = (float)dPara[E_PARA_POINT_RGB_ACTIVE_DARK_RATIO];
	float	fActiveBrightRatio = (float)dPara[E_PARA_POINT_RGB_ACTIVE_BRIGHT_RATIO];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_POINT_RGB_EDGE_DARK_RATIO];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_POINT_RGB_EDGE_BRIGHT_RATIO];

	int		nMinThGV = (long)dPara[E_PARA_POINT_RGB_COMMON_MIN_THRESHOLD];
	int		nMedian = (long)dPara[E_PARA_POINT_RGB_COMMON_MEDIAN];

	//如果设置值小于10。
	if (nSegX <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSegY <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matBlurBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, matSrcImage.type(), false);

	int nBlur = 5;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nBlur;
	//rtInspROI.y = rectROI.top - nBlur;
	//rtInspROI.width = rectROI.Width() + nBlur * 2;
	//rtInspROI.height = rectROI.Height() + nBlur * 2;

	//Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, 0);
	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nBlur);

	cv::Mat mattest = matSrcImage(rtInspROI);
	if (nBlurLoop_5x5 > 0)
	{
		cv::blur(matSrcImage(rtInspROI), matBlurBuf(rtInspROI), cv::Size(nBlur, nBlur));

		if (nBlurLoop_5x5 > 1)
		{
			// Avg
			for (int i = 1; i < nBlurLoop_5x5; i++)
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

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("BlurLoop."));

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
			nStart_Y = y * nSegY + rectROI.top;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = rectROI.bottom - nEdgeArea * nSegY;
			else							nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			if (y == 0)
			{
				nStart_Y = rectROI.top;
				nEnd_Y = nStart_Y + nSegY;
			}
			else
			{
				nStart_Y = rectROI.bottom - (nRangeY - y) * nSegY;

				if (nStart_Y < rectROI.top)
					nStart_Y = rectROI.top;

				nEnd_Y = nStart_Y + nSegY;
			}
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX + rectROI.left;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = rectROI.right - nEdgeArea * nSegX;
				else							nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = rectROI.right - (nRangeX - x) * nSegX;

				if (nStart_X < rectROI.left)
					nStart_X = rectROI.left;

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
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//获取mean&stdDev
//	cv::Scalar m, s;
//	cv::meanStdDev(matTempBuf, m, s);

				//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);
			double dblAverage;
			double dblStdev;
			AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

			//设置平均范围
//	int nMinGV = (int)(m[0] - s[0]);
//	int nMaxGV = (int)(m[0] + s[0]);
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

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("MedianFilter."));

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		BYTE* ptr = (BYTE*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
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
			cv::threshold(matTempBuf, matTempBufT, nDarkTemp, 255.0, THRESH_BINARY_INV);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, nBrightTemp, 255.0, THRESH_BINARY);
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
long CInspectPoint::RangeAvgThreshold_DUST(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub)
{
	long	nBlurLoop_5x5 = (long)dPara[E_PARA_POINT_DUST_BINARY_BLUR_LOOP];
	long	nSegX = (long)dPara[E_PARA_POINT_DUST_BINARY_SEG_X];
	long	nSegY = (long)dPara[E_PARA_POINT_DUST_BINARY_SEG_Y];
	long	nEdgeArea = (long)dPara[E_PARA_POINT_DUST_BINARY_EDGE_AREA];

	float	fActiveDarkRatio = (float)dPara[E_PARA_POINT_DUST_BINARY_ACTIVE_DARK_RATIO];
	float	fActiveBrightRatio = (float)dPara[E_PARA_POINT_DUST_BINARY_ACTIVE_BRIGHT_RATIO];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_POINT_DUST_BINARY_EDGE_DARK_RATIO];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_POINT_DUST_BINARY_EDGE_BRIGHT_RATIO];

	int		nMinThGV = (long)dPara[E_PARA_POINT_DUST_BINARY_MIN_THRESHOLD_GV];
	int		nMedian = (long)dPara[E_PARA_POINT_DUST_BINARY_MEDIAN];

	//如果设置值小于10。
	if (nSegX <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSegY <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	// Active Area
	rectROI.top += nEdgeArea;
	rectROI.left += nEdgeArea;
	rectROI.right -= nEdgeArea;
	rectROI.bottom -= nEdgeArea;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DUST Start."));

	//Temp内存分配
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	cv::Mat matBlurBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, matSrcImage.type(), false);

	int nBlur = 5;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nBlur;
	//rtInspROI.y = rectROI.top - nBlur;
	//rtInspROI.width = rectROI.Width() + nBlur * 2;
	//rtInspROI.height = rectROI.Height() + nBlur * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nBlur);

	if (nBlurLoop_5x5 > 0)
	{
		cv::blur(matSrcImage(rtInspROI), matBlurBuf(rtInspROI), cv::Size(nBlur, nBlur));

		if (nBlurLoop_5x5 > 1)
		{
			// Avg
			for (int i = 1; i < nBlurLoop_5x5; i++)
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

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DUST BlurLoop."));

	// Area
		//////////////////////////////按照客户的要求Edge Area Pixel进行操作
	// Range Rect
	cv::Rect* rectRange = new cv::Rect[nRangeX * nRangeY + 4]; // Edge区域上分为下,左,右
	memset(rectRange, 0, sizeof(cv::Rect) * (nRangeX * nRangeY + 4));

	// Area
		//计算范围
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

		//计算y范围
		nStart_Y = y * nSegY + rectROI.top;
		nEnd_Y = nStart_Y + nSegY;

		if (nEnd_Y > rectROI.bottom)
		{
			nStart_Y = rectROI.bottom - nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			nStart_X = x * nSegX + rectROI.left;
			nEnd_X = nStart_X + nSegX;

			if (nEnd_X > rectROI.right)
			{
				nStart_X = rectROI.right - nSegX;
				nEnd_X = nStart_X + nSegX;
			}

			//设置范围
			rectPtr[x].x = nStart_X;
			rectPtr[x].y = nStart_Y;
			rectPtr[x].width = nEnd_X - nStart_X + 1;
			rectPtr[x].height = nEnd_Y - nStart_Y + 1;
		}
	}
	// Area
		//计算Edge范围
	if (nEdgeArea > 1)
	{
		cv::Rect* rectPtr = &rectRange[nRangeY * nRangeX];	// Top

		//设置范围
		rectPtr[0].x = rectROI.left;
		rectPtr[0].y = rectROI.top;
		rectPtr[0].width = rectROI.Width();
		rectPtr[0].height = nEdgeArea;

		//设置范围
		rectPtr[1].x = rectROI.left;
		rectPtr[1].y = rectROI.top + rectROI.Height() - nEdgeArea;
		rectPtr[1].width = rectROI.Width();
		rectPtr[1].height = nEdgeArea;

		//设置范围
		rectPtr[2].x = rectROI.left;
		rectPtr[2].y = rectROI.top;
		rectPtr[2].width = nEdgeArea;
		rectPtr[2].height = rectROI.Height();

		//设置范围
		rectPtr[3].x = rectROI.left + rectROI.Width() - nEdgeArea;
		rectPtr[3].y = rectROI.top;
		rectPtr[3].width = nEdgeArea;
		rectPtr[3].height = rectROI.Height();
	}
	// Area
		///////////////////////////////源////////////////////////////////////////////////////////////////////////

// 	cv::Rect *rectRange = new cv::Rect[nRangeX * nRangeY];
// 	memset(rectRange, 0, sizeof(cv::Rect) * nRangeX * nRangeY);

//	范围计算
// 		for (long y=0 ; y<nRangeY ; y++)

// 			cv::Rect *rectPtr = &rectRange[nRangeX * y];

		//y范围计算
// 			if ( y < nRangeY - nEdgeArea )

// 				nStart_Y = y * nSegY + rectROI.top;

// 				if( y==nRangeY-1-nEdgeArea)		nEnd_Y = rectROI.bottom - nEdgeArea * nSegY;
// 				else							nEnd_Y = nStart_Y + nSegY;

// 				nStart_Y = rectROI.bottom - (nRangeY - y) * nSegY;
// 				nEnd_Y	 = nStart_Y + nSegY;

// 			for (long x=0 ; x<nRangeX ; x++)

		//x范围计算
// 				if ( x < nRangeX - nEdgeArea )

// 					nStart_X = x * nSegX + rectROI.left;

// 					if( x==nRangeX-1-nEdgeArea)		nEnd_X = rectROI.right - nEdgeArea * nSegX;
// 					else							nEnd_X = nStart_X + nSegX;

// 					nStart_X = rectROI.right - (nRangeX - x) * nSegX;
// 					nEnd_X	 = nStart_X + nSegX;

//			//设置范围
// 				rectPtr[x].x		= nStart_X;
// 				rectPtr[x].y		= nStart_Y;
// 				rectPtr[x].width	= nEnd_X - nStart_X + 1;
// 				rectPtr[x].height	= nEnd_Y - nStart_Y + 1;

		//平均计算
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		BYTE* ptr = (BYTE*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//获取mean&stdDev
//	cv::Scalar m, s;
//	cv::meanStdDev(matTempBuf, m, s);

			//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);

			double dblAverage;
			double dblStdev;
			AlgoBase::GetMeanStdDev_From_Histo(matHisto, 0, 255, dblAverage, dblStdev);

			//设置平均范围
//	int nMinGV = (int)(m[0] - s[0]);
//	int nMaxGV = (int)(m[0] + s[0]);
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

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DUST MedianFilter."));

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		BYTE* ptr = (BYTE*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//平均*Ratio
			long nDarkTemp, nBrightTemp;

			// Area
						//单独设置Edge->Edge不在这里进行
// 			if (x < nEdgeArea				||
// 				y < nEdgeArea				||
// 				x >= nRangeX - nEdgeArea	||
// 				y >= nRangeY - nEdgeArea	)

		//Dust的平均值为0

			nDarkTemp = (long)(ptr[x] * fActiveDarkRatio);
			nBrightTemp = (long)(ptr[x] * fActiveBrightRatio);
			//}			

						//Gray有太暗的情况。
						//(平均GV2~3*fBrightRatio->二进制:噪声全部上升)
						//在Dust中可能会找到非常弱的东西,所以在0时不起作用
			if (nMinThGV != 0)
			{
				if (nBrightTemp < nMinThGV)	nBrightTemp = nMinThGV;
			}

			//异常处理
			if (nDarkTemp < 0)		nDarkTemp = 0;
			if (nBrightTemp > 255)	nBrightTemp = 255;

			//参数0时异常处理
			if (fActiveDarkRatio == 0)		nDarkTemp = -1;
			if (fActiveBrightRatio == 0)	nBrightTemp = 256;

			// E_DEFECT_COLOR_DARK Threshold
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, nDarkTemp, 255.0, THRESH_BINARY_INV);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			cv::threshold(matTempBuf, matTempBufT, nBrightTemp, 255.0, THRESH_BINARY);
		}
	}
	// Area
	if (nEdgeArea > 1)
	{
		cv::Rect* rectPtr = &rectRange[nRangeX * nRangeY];

		for (int i = 0; i < 4; i++)
		{
			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[i]);

			//二进制

// E_DEFECT_COLOR_DARK Threshold
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectPtr[i]);
			cv::threshold(matTempBuf, matTempBufT, fEdgeDarkRatio, 255.0, THRESH_BINARY_INV);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[i]);
			cv::threshold(matTempBuf, matTempBufT, fEdgeBrightRatio, 255.0, THRESH_BINARY);
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
long CInspectPoint::RangeAvgThreshold_16bit(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub)
{
	long	nBlurLoop_5x5 = (long)dPara[E_PARA_POINT_RGB_COMMON_BLUR_LOOP];
	long	nSegX = (long)dPara[E_PARA_POINT_RGB_COMMON_SEG_X];
	long	nSegY = (long)dPara[E_PARA_POINT_RGB_COMMON_SEG_Y];
	long	nEdgeArea = (long)dPara[E_PARA_POINT_RGB_EDGE_AREA];

	float	fActiveDarkRatio = (float)dPara[E_PARA_POINT_RGB_ACTIVE_DARK_RATIO];
	float	fActiveBrightRatio = (float)dPara[E_PARA_POINT_RGB_ACTIVE_BRIGHT_RATIO];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_POINT_RGB_EDGE_DARK_RATIO];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_POINT_RGB_EDGE_BRIGHT_RATIO];

	int		nMinThGV = (long)dPara[E_PARA_POINT_RGB_COMMON_MIN_THRESHOLD];
	int		nMedian = (long)dPara[E_PARA_POINT_RGB_COMMON_MEDIAN];

	//如果设置值小于10。
	if (nSegX <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSegY <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//Temp内存分配	
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());
	cv::Mat matBlurBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, matSrcImage.type(), false);

	int nBlur = 5;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nBlur;
	//rtInspROI.y = rectROI.top - nBlur;
	//rtInspROI.width = rectROI.Width() + nBlur * 2;
	//rtInspROI.height = rectROI.Height() + nBlur * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nBlur);

	if (nBlurLoop_5x5 > 0)
	{
		cv::blur(matSrcImage(rtInspROI), matBlurBuf(rtInspROI), cv::Size(nBlur, nBlur));

		if (nBlurLoop_5x5 > 1)
		{
			// Avg
			for (int i = 1; i < nBlurLoop_5x5; i++)
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

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("BlurLoop."));

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
			nStart_Y = y * nSegY + rectROI.top;

			if (y == nRangeY - 1 - nEdgeArea)		
				nEnd_Y = rectROI.bottom - nEdgeArea * nSegY;
			else							
				nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = rectROI.bottom - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX + rectROI.left;

				if (x == nRangeX - 1 - nEdgeArea)		
					nEnd_X = rectROI.right - nEdgeArea * nSegX;
				else							
					nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = rectROI.right - (nRangeX - x) * nSegX;
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
		ushort* ptr = (ushort*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//获取mean&stdDev
			cv::Scalar m, s;
			cv::meanStdDev(matTempBuf, m, s);

			//设置平均范围
			int nMinGV = (int)(m[0] - s[0]);
			int nMaxGV = (int)(m[0] + s[0]);
			if (nMinGV < 0)	nMinGV = 0;
			if (nMaxGV > 4095)	nMaxGV = 4095;

			//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);
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
				ptr[x] = (ushort)(nPixelSum / (float)nPixelCount);
		}
	}

	//设置为周边平均值->中间值
	AlgoBase::MedianFilter(matAvgBuf, matAvgBuf, nMedian, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("MedianFilter."));

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		ushort* ptr = (ushort*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
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
			if (nDarkTemp < 0)			nDarkTemp = 0;
			if (nBrightTemp > 4095)	nBrightTemp = 4095;

			//参数0时异常处理
			if (fActiveDarkRatio == 0)		nDarkTemp = -1;
			if (fActiveBrightRatio == 0)	nBrightTemp = 4096;

			// E_DEFECT_COLOR_DARK Threshold
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectPtr[x]);
			AlgoBase::Binary_16bit(matTempBuf, matTempBufT, nDarkTemp, true, &cMatBufTemp);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			AlgoBase::Binary_16bit(matTempBuf, matTempBufT, nBrightTemp, false, &cMatBufTemp);
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
long CInspectPoint::RangeAvgThreshold_RGB(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara, CMatBuf* cMemSub)
{
	long	nBlurLoop_5x5 = (long)dPara[E_PARA_POINT_RGB_COMMON_BLUR_LOOP];
	long	nSegX = (long)dPara[E_PARA_POINT_RGB_COMMON_SEG_X];
	long	nSegY = (long)dPara[E_PARA_POINT_RGB_COMMON_SEG_Y];
	long	nEdgeArea = (long)dPara[E_PARA_POINT_RGB_EDGE_AREA];

	float	fActiveDarkRatio = (float)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_DARK_RATIO_ACTIVE];
	float	fActiveBrightRatio = (float)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BRIGHT_RATIO_ACTIVE];
	float	fEdgeDarkRatio = (float)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_DARK_RATIO_EDGE];
	float	fEdgeBrightRatio = (float)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BRIGHT_RATIO_EDGE];

	int		nMinThGV = (long)dPara[E_PARA_POINT_RGB_COMMON_MIN_THRESHOLD];
	int		nMedian = (long)dPara[E_PARA_POINT_RGB_COMMON_MEDIAN];

	//如果设置值小于10。
	if (nSegX <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSegY <= 10)		return E_ERROR_CODE_POINT_WARNING_PARA;

	//图像大小
	long	nWidth = (long)matSrcImage.cols;	// 图像宽度大小
	long	nHeight = (long)matSrcImage.rows;	// 图像垂直尺寸

	long nStart_Y, nEnd_Y;

	//仅检查活动区域
	int nRangeX = rectROI.Width() / nSegX + 1;
	int nRangeY = rectROI.Height() / nSegY + 1;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Start."));

	//Temp内存分配	
	cv::Mat matBlurBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matBlurBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	// Range Avg
	cv::Mat matAvgBuf = cMatBufTemp.GetMat(nRangeY, nRangeX, matSrcImage.type(), false);

	int nBlur = 5;

	cv::Rect rtInspROI;
	//rtInspROI.x = rectROI.left - nBlur;
	//rtInspROI.y = rectROI.top - nBlur;
	//rtInspROI.width = rectROI.Width() + nBlur * 2;
	//rtInspROI.height = rectROI.Height() + nBlur * 2;

	Insp_RectSet(rtInspROI, rectROI, matSrcImage.cols, matSrcImage.rows, nBlur);

	if (nBlurLoop_5x5 > 0)
	{
		cv::blur(matSrcImage(rtInspROI), matBlurBuf(rtInspROI), cv::Size(nBlur, nBlur));

		if (nBlurLoop_5x5 > 1)
		{
			// Avg
			for (int i = 1; i < nBlurLoop_5x5; i++)
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

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("BlurLoop."));

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
			nStart_Y = y * nSegY + rectROI.top;

			if (y == nRangeY - 1 - nEdgeArea)		nEnd_Y = rectROI.bottom - nEdgeArea * nSegY;
			else							nEnd_Y = nStart_Y + nSegY;
		}
		else
		{
			nStart_Y = rectROI.bottom - (nRangeY - y) * nSegY;
			nEnd_Y = nStart_Y + nSegY;
		}

		for (long x = 0; x < nRangeX; x++)
		{
			long nStart_X, nEnd_X;

			//计算x范围
			if (x < nRangeX - nEdgeArea)
			{
				nStart_X = x * nSegX + rectROI.left;

				if (x == nRangeX - 1 - nEdgeArea)		nEnd_X = rectROI.right - nEdgeArea * nSegX;
				else							nEnd_X = nStart_X + nSegX;
			}
			else
			{
				nStart_X = rectROI.right - (nRangeX - x) * nSegX;
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
		ushort* ptr = (ushort*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
			//画面ROI
			cv::Mat matTempBuf = matBlurBuf(rectPtr[x]);

			//获取mean&stdDev
			cv::Scalar m, s;
			cv::meanStdDev(matTempBuf, m, s);

			//设置平均范围
			int nMinGV = (int)(m[0] - s[0]);
			int nMaxGV = (int)(m[0] + s[0]);
			if (nMinGV < 0)	nMinGV = 0;
			if (nMaxGV > 4095)	nMaxGV = 4095;

			//直方图
			cv::Mat matHisto;
			AlgoBase::GetHistogram(matTempBuf, matHisto, false);
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
				ptr[x] = (ushort)(nPixelSum / (float)nPixelCount);
		}
	}

	//设置为周边平均值->中间值
	AlgoBase::MedianFilter(matAvgBuf, matAvgBuf, nMedian, &cMatBufTemp);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("MedianFilter."));

	//二进制
	for (long y = 0; y < nRangeY; y++)
	{
		// Range Avg
		ushort* ptr = (ushort*)matAvgBuf.ptr(y);

		// Range Rect
		cv::Rect* rectPtr = &rectRange[nRangeX * y];

#ifdef _DEBUG
#else
#pragma omp parallel for  num_threads(2)
#endif
		for (long x = 0; x < nRangeX; x++)
		{
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
			if (nDarkTemp < 0)			nDarkTemp = 0;
			if (nBrightTemp > 4095)	nBrightTemp = 4095;

			//参数0时异常处理
			if (fActiveDarkRatio == 0)		nDarkTemp = -1;
			if (fActiveBrightRatio == 0)	nBrightTemp = 4096;

			// E_DEFECT_COLOR_DARK Threshold
			cv::Mat matTempBufT = matDstImage[E_DEFECT_COLOR_DARK](rectPtr[x]);
			AlgoBase::Binary_16bit(matTempBuf, matTempBufT, nDarkTemp, true, &cMatBufTemp);

			// E_DEFECT_COLOR_BRIGHT Threshold
			matTempBufT = matDstImage[E_DEFECT_COLOR_BRIGHT](rectPtr[x]);
			AlgoBase::Binary_16bit(matTempBuf, matTempBufT, nBrightTemp, false, &cMatBufTemp);
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

//莫波罗吉
long CInspectPoint::Morphology(cv::Mat& matSrcImage, cv::Mat& matDstImage, long nSizeX, long nSizeY, ENUM_MORP nOperation)
{
	if (nSizeX < 1)			return E_ERROR_CODE_POINT_WARNING_PARA;
	if (nSizeY < 1)			return E_ERROR_CODE_POINT_WARNING_PARA;
	if (matSrcImage.empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, Size(nSizeX, nSizeY), Point(nSizeX / 2, nSizeY / 2));

	switch (nOperation)
	{
	case E_MORP_ERODE:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_ERODE, StructElem);
		break;

	case E_MORP_DILATE:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_DILATE, StructElem);
		break;

	case E_MORP_OPEN:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_OPEN, StructElem);
		break;

	case E_MORP_CLOSE:
		cv::morphologyEx(matSrcImage, matDstImage, MORPH_CLOSE, StructElem);
		break;

	default:
		StructElem.release();
		return E_ERROR_CODE_POINT_WARNING_PARA;
		break;
	}

	StructElem.release();
	return E_ERROR_CODE_TRUE;
}

//去除外壳
long CInspectPoint::DeleteOutArea(cv::Mat& matSrcImage, cv::Point* ptCorner, CMatBuf* cMemSub)
{

	CMatBuf cMatBufTemp;

	//Temp内存分配
	cMatBufTemp.SetMem(cMemSub);

	//初始化掩码
	cv::Mat matTempBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	int npt[] = { E_CORNER_END };
	const cv::Point* ppt[1] = { ptCorner };

	//点亮区域掩码
	if (matSrcImage.type() == CV_8U)
		cv::fillPoly(matTempBuf, ppt, npt, 1, cv::Scalar(255, 255, 255));
	else
		cv::fillPoly(matTempBuf, ppt, npt, 1, cv::Scalar(4095, 4095, 4095));

	// AND
	cv::bitwise_and(matSrcImage, matTempBuf, matSrcImage);

	//取消分配
	matTempBuf.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//去除外壳
long CInspectPoint::DeleteOutArea(cv::Mat& matSrcImage, CRect rectROI, CMatBuf* cMemSub)
{
	CMatBuf cMatBufTemp;

	//Temp内存分配
	cMatBufTemp.SetMem(cMemSub);

	//初始化掩码
	cv::Mat matTempBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type());

	//点亮区域掩码
	if (matSrcImage.type() == CV_8U)
		cv::rectangle(matTempBuf, cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()), cv::Scalar(255, 255, 255), -1);
	else
		cv::rectangle(matTempBuf, cv::Rect(rectROI.left, rectROI.top, rectROI.Width(), rectROI.Height()), cv::Scalar(4095, 4095, 4095), -1);

	// AND
	cv::bitwise_and(matSrcImage, matTempBuf, matSrcImage);

	//取消分配
	matTempBuf.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

//删除小面积
long CInspectPoint::DeleteArea(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	if (matSrcImage.type() == CV_8U)
		nErrorCode = DeleteArea_8bit(matSrcImage, nCount, cMemSub);
	else
		nErrorCode = DeleteArea_16bit(matSrcImage, nCount, cMemSub);

	return nErrorCode;
}

//删除小面积
long CInspectPoint::DeleteArea_8bit(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
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
long CInspectPoint::DeleteArea_16bit(cv::Mat& matSrcImage, int nCount, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//8 bit转换
	cv::Mat matSrc8bit = cMatBufTemp.GetMat(matSrcImage.size(), CV_8UC1, false);
	matSrcImage.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

	//内存分配
	cv::Mat matLabel, matStats, matCentroid;
	matLabel = cMatBufTemp.GetMat(matSrc8bit.size(), CV_32SC1, false);

	//Blob计数
	__int64 nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

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
			cv::Mat matTempROI = matSrc8bit(rectTemp);
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

	//16bit转换
	matSrc8bit.convertTo(matSrcImage, CV_16UC1, 16.);

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
		if( matSrcImage.at<ushort>(y, x) == 0)		continue;

		memset(bConnect, 0, sizeof(bool) * 8);
		nConnectCnt = 1;

					//确定周围的数量
		for (int z=0 ; z<8 ; z++)
		{
			if( matSrcImage.at<ushort>(y + nConnectY[z], x + nConnectX[z]) != 0)
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

			DstBuffer.at<ushort>(y + nConnectY[z], x + nConnectX[z]) = (ushort)4095;
		}

					//绘制中心
		DstBuffer.at<ushort>(y, x) = (ushort)4095;
	}
}

matSrcImage = DstBuffer.clone();

DstBuffer.release();
************************************************************************/

	return E_ERROR_CODE_TRUE;
}

long CInspectPoint::JudgementCheckE(cv::Mat& matSrcBuf, double* dPara, CRect rectROI, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//参数	
	int nFlag = (int)dPara[E_PARA_POINT_DUST_GROUP_USE];
	int nDel = (int)dPara[E_PARA_POINT_DUST_GROUP_EDGE_DEL_OUTLINE];
	int nEdgeArea = (int)dPara[E_PARA_POINT_DUST_GROUP_EDGE_AREA];
	int nMinAreaEdge = (int)dPara[E_PARA_POINT_DUST_GROUP_MIN_AREA_EDGE];
	int nMinAreaActive = (int)dPara[E_PARA_POINT_DUST_GROUP_MIN_AREA_ACTIVE];
	int nSetEdgeCount = (int)dPara[E_PARA_POINT_DUST_GROUP_EDGE_COUNT];
	int nSetActiveCount = (int)dPara[E_PARA_POINT_DUST_GROUP_ACTIVE_COUNT];
	int nSetTotalCount = (int)dPara[E_PARA_POINT_DUST_GROUP_TOTAL_COUNT];

	//如果禁用
//if( nFlag <= 0 )		return E_ERROR_CODE_EMPTY_BUFFER;
	if (nFlag <= 0)		return E_ERROR_CODE_TRUE;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//预分配
	cv::Mat matStats, matCentroid;
	cv::Mat matLabel = cMatBufTemp.GetMat(matSrcBuf.size(), CV_32SC1, false);

	//Label运算
	__int64 nTotalLabel = 0;

	if (matSrcBuf.type() == CV_8U)
	{
		nTotalLabel = cv::connectedComponentsWithStats(matSrcBuf, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA);
	}
	else
	{
		cv::Mat matSrc8bit = cMatBufTemp.GetMat(matSrcBuf.size(), CV_8UC1, false);
		matSrcBuf.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

		nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA);

		matSrc8bit.release();
	}

	//区域不良数量
	__int64 nEdgeCount = 0;
	__int64 nActiveCount = 0;

	//Total区域(覆盖区域Offset)
	CRect rectTotalArea(rectROI.left + nDel, rectROI.top + nDel, rectROI.right - nDel, rectROI.bottom - nDel);

	//活动区域
	CRect rectActiveArea(rectTotalArea.left + nEdgeArea, rectTotalArea.top + nEdgeArea, rectTotalArea.right - nEdgeArea, rectTotalArea.bottom - nEdgeArea);

	//Dust不良坐标
	CPoint ptCenter;

	//Dust个数...
	for (int idx = 1; idx < nTotalLabel; idx++)
	{
		//获取Center Point
		ptCenter.x = (int)matCentroid.at<double>(idx, 0);
		ptCenter.y = (int)matCentroid.at<double>(idx, 1);
		int nArea = (int)matStats.at<int>(idx, CC_STAT_AREA);

		//Edge覆盖区域外路径
		if (!rectTotalArea.PtInRect(ptCenter))
			continue;

		//活动范围内有不良存在吗？
		if (rectActiveArea.PtInRect(ptCenter))
		{
			//如果小于设置的坏大小,则跳过
			if (nMinAreaActive > nArea)
				continue;

			//如果存在于Active区域中
			nActiveCount++;
		}
		//Edge范围内不良
		else
		{
			//如果小于设置的坏大小,则跳过
			if (nMinAreaEdge > nArea)
				continue;

			//如果存在于Edge区域中
			nEdgeCount++;
		}
	}

	//检查活动设置(添加E_DEFECT_JUDGEMENT_DUST_GROUP故障)
	if (nActiveCount > nSetActiveCount)
		nErrorCode = E_ERROR_CODE_POINT_JUDEGEMENT_E;

	//检查Edge设置(添加E_DEFECT_JUDGEMENT_DUST_GROUP故障)
	if (nEdgeCount > nSetEdgeCount)
		nErrorCode = E_ERROR_CODE_POINT_JUDEGEMENT_E;

	//检查总体设置(添加E_DEFECT_JUDGEMENT_DUST_GROUP错误)
	if (nActiveCount + nEdgeCount > nSetTotalCount)
		nErrorCode = E_ERROR_CODE_POINT_JUDEGEMENT_E;

	//CString str;
	//str.Format(_T("E%d, A%d"), nEdgeCount, nActiveCount);
	//AfxMessageBox(str);

		//禁用
	matStats.release();
	matCentroid.release();
	matLabel.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//在原始图像中加入RGB模式中的一个来提供校正值的功能->在Gray中使用
long CInspectPoint::AdjustImageWithRGB(cv::Mat& matSrcImage, cv::Mat& matDstImage, cv::Mat& matAdjustSrcImage, double dblAdjustRatio, int nCutMinGVForAdjust, CRect rectROI, CMatBuf* cMemSub)
{
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//画面大小不同时不可用
	if (matSrcImage.rows != matAdjustSrcImage.rows)	return E_ERROR_CODE_TRUE;
	if (matSrcImage.cols != matAdjustSrcImage.cols)	return E_ERROR_CODE_TRUE;

	cv::Rect rcROI;
	rcROI.x = rectROI.left;
	rcROI.y = rectROI.top;
	rcROI.width = rectROI.Width();
	rcROI.height = rectROI.Height();

	//Temp内存分配	
	cv::Mat matTempBuf = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	//调整ROI
	cv::Mat matSrcROIImage = matSrcImage(rcROI);
	cv::Mat matDstROIImage = matTempBuf(rcROI);
	cv::Mat matAdjSrcROIImage = matAdjustSrcImage(rcROI);

	cv::Mat matAdjImage = cMatBufTemp.GetMat(matSrcROIImage.size(), matSrcROIImage.type(), false);
	cv::Mat matAdjImageBackup = cMatBufTemp.GetMat(matSrcROIImage.size(), matSrcROIImage.type(), false);

	double dblAveSrc, dblAveAdjustSrc, dblApplyRatio;
	CString strTemp;

	///////////////////////////////////////////////////////////////////
		//获取参考图像。每个像素的亮度与输入的Gray图像相同。
	////////////////////////////////////////////////////////////////////

		//求原始图像的平均值和校正时要参照的图像的平均值。
	dblAveSrc = AlgoBase::GetAverage(matSrcROIImage);

	//求接近每个像素平均值的平均值。
	dblAveAdjustSrc = AlgoBase::GetAverageForRGB(matAdjSrcROIImage);

	//求相对值。
	dblApplyRatio = dblAveSrc / dblAveAdjustSrc;

	//获取应用校正值的参照图像。-能够将每个像素的亮度与输入的灰度图像对齐
	matAdjImage = matAdjSrcROIImage * dblApplyRatio;

	//应用补偿值备份图像-用于以后填充空洞。
	matAdjImage.copyTo(matAdjImageBackup);

	/////////////////////////////////////////////////////////////////////
		//对最终获得的校正图像应用用户校正值。
	/////////////////////////////////////////////////////////////////////

		//应用用户补偿
	matAdjImage *= dblAdjustRatio;

	//断开最小GV值(像素之间的亮区干扰)
	matAdjImage -= nCutMinGVForAdjust;

	//	//应用补救图像

	//	//应用补正值图像后,填补中间的黑洞。

	matDstROIImage = matSrcROIImage - matAdjImageBackup;
	cv::max(matDstROIImage, matAdjImage, matDstROIImage);

	//复制最终结果
	matTempBuf.copyTo(matDstImage);

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

long CInspectPoint::PatternSubstraction(cv::Mat& matSrcImage, cv::Mat* matDstImage, int type, CMatBuf* cMemSub)
{
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//Temp内存分配
	cv::Mat matTempBuf_Temp = cMatBufTemp.GetMat(matDstImage[E_IMAGE_CLASSIFY_AVI_R].size(), matDstImage[E_IMAGE_CLASSIFY_AVI_R].type(), false);
	cv::Mat matTempBuf_Output = cMatBufTemp.GetMat(matDstImage[E_IMAGE_CLASSIFY_AVI_B].size(), matDstImage[E_IMAGE_CLASSIFY_AVI_B].type(), false);

	if (type == E_IMAGE_CLASSIFY_AVI_R)
	{
		matTempBuf_Temp = matDstImage[E_IMAGE_CLASSIFY_AVI_R] - matDstImage[E_IMAGE_CLASSIFY_AVI_G];
		matTempBuf_Output = matTempBuf_Temp - matDstImage[E_IMAGE_CLASSIFY_AVI_B];

	}

	else if (type == E_IMAGE_CLASSIFY_AVI_G)
	{
		matTempBuf_Temp = matDstImage[E_IMAGE_CLASSIFY_AVI_G] - matDstImage[E_IMAGE_CLASSIFY_AVI_R];
		matTempBuf_Output = matTempBuf_Temp - matDstImage[E_IMAGE_CLASSIFY_AVI_B];
	}

	else if (type == E_IMAGE_CLASSIFY_AVI_B)
	{
		matTempBuf_Temp = matDstImage[E_IMAGE_CLASSIFY_AVI_B] - matDstImage[E_IMAGE_CLASSIFY_AVI_G];
		matTempBuf_Output = matTempBuf_Temp - matDstImage[E_IMAGE_CLASSIFY_AVI_R];
	}

	//复制最终结果
//matTempBuf.copyTo(matDstImage);
	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return E_ERROR_CODE_TRUE;
}

void CInspectPoint::ApplyEnhancement(cv::Mat matSrcImage, cv::Mat matBuff1, cv::Mat matBuff2, cv::Mat& matDstImage1, cv::Mat& matDstImage2,
	double* dPara, int* nCommonPara, CString strAlgPath, int Type, CMatBuf* cMemSub)
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

	//使用的Parameter
	double	dSubRatio1;
	double	dMulRatio1;
	double	dSubRatio2;
	double	dMulRatio2;
	double	dSubRatioTemp;
	double	dMulRatioTemp;

	if (Type == E_IMAGE_CLASSIFY_AVI_R)
	{
		dSubRatio1 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_SUB_RATIO];
		dMulRatio1 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_MLT_RATIO];
		dSubRatio2 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_SUB_RATIO];
		dMulRatio2 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_MLT_RATIO];

		dSubRatioTemp = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_RED_SUB_RATIO];
		dMulRatioTemp = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_RED_MLT_RATIO];
	}

	if (Type == E_IMAGE_CLASSIFY_AVI_G)
	{
		dSubRatio1 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_RED_SUB_RATIO];
		dMulRatio1 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_RED_MLT_RATIO];
		dSubRatio2 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_SUB_RATIO];
		dMulRatio2 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_MLT_RATIO];

		dSubRatioTemp = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_SUB_RATIO];
		dMulRatioTemp = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_MLT_RATIO];
	}

	if (Type == E_IMAGE_CLASSIFY_AVI_B)
	{
		dSubRatio1 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_RED_SUB_RATIO];
		dMulRatio1 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_RED_MLT_RATIO];
		dSubRatio2 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_SUB_RATIO];
		dMulRatio2 = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_GREEN_MLT_RATIO];

		dSubRatioTemp = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_SUB_RATIO];
		dMulRatioTemp = (double)dPara[E_PARA_POINT_RGB_BRIGHT_INSP_BLUE_MLT_RATIO];
	}

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//Temp内存分配
	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);
	cv::Mat matTempBuf2 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	//Pattern Sub for Buff1, Buff2SubPatternEhancemnt(
	SubPatternEhancemnt(matBuff1, matTempBuf1, dSubRatio1, dMulRatio1, nCommonPara, strAlgPath, &cMatBufTemp);
	SubPatternEhancemnt(matBuff2, matTempBuf2, dSubRatio2, dMulRatio2, nCommonPara, strAlgPath, &cMatBufTemp);

	SubPatternEhancemnt(matSrcImage, matDstImage2, dSubRatioTemp, dMulRatioTemp, nCommonPara, strAlgPath, &cMatBufTemp);

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_SubPattern1.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf1);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_SubPattern2.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf2);
	}

	matDstImage1 = matDstImage2 - matTempBuf1 - matTempBuf2;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_SubPattern3.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage1);
	}

	matTempBuf1.release();
	matTempBuf2.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

}

void CInspectPoint::SubPatternEhancemnt(cv::Mat matSrcImage, cv::Mat& matDstImage, double dSubWeight, double dEnhanceWeight, int* nCommonPara, CString strAlgPath, CMatBuf* cMemSub)
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

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	//Temp内存分配
	cv::Mat matTempBuf1 = cMatBufTemp.GetMat(matSrcImage.size(), matSrcImage.type(), false);

	double ImagelMean;
	cv::Scalar m = cv::mean(matSrcImage);
	ImagelMean = m[0];

	matTempBuf1 = matSrcImage - ImagelMean * dSubWeight;
	matDstImage = matTempBuf1 * dEnhanceWeight;

	if (bImageSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Enhancem1.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matTempBuf1);
		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Enhancem2.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage);
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}
}

//在Dust中查找面积较大的情况
long CInspectPoint::FindBigAreaDust(cv::Mat& matSrcBuf, cv::Mat& matDstBuf, long nFindArea, CMatBuf* cMemSub)
{
	//如果没有缓冲区。
	if (matSrcBuf.empty())				return E_ERROR_CODE_EMPTY_BUFFER;

	//仅响应8bit
//if( matSrcBuf.type() != CV_8U )		return E_ERROR_CODE_TRUE;

	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matLabelBuffer = cMatBufTemp.GetMat(matSrcBuf.size(), CV_32SC1, false);

	matSrcBuf.convertTo(matLabelBuffer, CV_32SC1);
	matDstBuf.setTo(0);

	//检查区域Rect
	cv::Rect rectTemp;
	int nTempCount = 0;

	// 8 bit
	if (matDstBuf.type() == CV_8U)
	{
		for (int y = 0; y < matLabelBuffer.rows; y++)
		{
			int* row = (int*)matLabelBuffer.ptr(y);

			for (int x = 0; x < matLabelBuffer.cols; x++)
			{
				if (row[x] != 255)	continue;

				nTempCount++;

				//对象面积
				long nArea = cv::floodFill(matLabelBuffer, Point(x, y), nTempCount, &rectTemp);

				//面积限制
				if (nArea < nFindArea)	continue;

				int nEndXX = rectTemp.x + rectTemp.width;
				int nEndYY = rectTemp.y + rectTemp.height;

				//异常处理
				if (nEndYY >= matLabelBuffer.rows)	nEndYY = matLabelBuffer.rows - 1;
				if (nEndXX >= matLabelBuffer.cols)	nEndXX = matLabelBuffer.cols - 1;

				//每个标签的像素坐标
				for (int j = rectTemp.y; j <= nEndYY; j++)
				{
					int* row2 = (int*)matLabelBuffer.ptr(j);
					BYTE* row3 = (BYTE*)matDstBuf.ptr(j);

					for (int i = rectTemp.x; i <= nEndXX; i++)
					{
						if (row2[i] != nTempCount)	continue;

						row3[i] = (BYTE)255;
					}
				}
			}
		}
	}
	// 12 bit
	else
	{
		for (int y = 0; y < matLabelBuffer.rows; y++)
		{
			int* row = (int*)matLabelBuffer.ptr(y);

			for (int x = 0; x < matLabelBuffer.cols; x++)
			{
				if (row[x] != 4095)	continue;

				nTempCount++;

				//对象面积
				long nArea = cv::floodFill(matLabelBuffer, Point(x, y), nTempCount, &rectTemp);

				//面积限制
				if (nArea < nFindArea)	continue;

				int nEndXX = rectTemp.x + rectTemp.width;
				int nEndYY = rectTemp.y + rectTemp.height;

				//异常处理
				if (nEndYY >= matLabelBuffer.rows)	nEndYY = matLabelBuffer.rows - 1;
				if (nEndXX >= matLabelBuffer.cols)	nEndXX = matLabelBuffer.cols - 1;

				//每个标签的像素坐标
				for (int j = rectTemp.y; j <= nEndYY; j++)
				{
					int* row2 = (int*)matLabelBuffer.ptr(j);
					ushort* row3 = (ushort*)matDstBuf.ptr(j);

					for (int i = rectTemp.x; i <= nEndXX; i++)
					{
						if (row2[i] != nTempCount)	continue;

						row3[i] = (ushort)4095;
					}
				}
			}
		}
	}

	matLabelBuffer.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//删除暗点-Dust面积较大的周边(8bit&12bit)
long CInspectPoint::DeleteCompareDarkPoint(cv::Mat& matSrcBuffer, int nOffset, stDefectInfo* pResultBlob, int nModePS)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL
	if (matSrcBuffer.empty())		return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)		return E_ERROR_CODE_EMPTY_PARA;

	//检测到的不良数量
	int nDefectCount = pResultBlob->nDefectCount;

	//根据不良数量...
	for (int i = 0; i < nDefectCount; )
	{
		//不包括暗点不良
		if (pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_POINT_DARK &&
			pResultBlob->nDefectJudge[i] != E_DEFECT_JUDGEMENT_POINT_GROUP_DARK)
		{
			i++;
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

		//根据画面大小进行比较(nModePS)
//rect.x		/= nModePS;
//rect.y		/= nModePS;
//rect.width	/= nModePS;
//rect.height	/= nModePS;

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
				//检查暗点周围是否存在Dust
		//////////////////////////////////////////////////////////////////////////

				//寻找暗点周围->Dust画面Max GV
		double valMax;
		cv::minMaxLoc(matTempBuf, NULL, &valMax);

		//检查Dust时,没有太大的不良(存在太大的不良,存在255/4096值)
		if (valMax == 0)
		{
			i++;
			continue;
		}

		//删除相应的暗点
		//最后一个index错误-->放入当前index
		{
			pResultBlob->nDefectJudge[i] = pResultBlob->nDefectJudge[nDefectCount - 1];
			pResultBlob->nDefectColor[i] = pResultBlob->nDefectColor[nDefectCount - 1];
			pResultBlob->nPatternClassify[i] = pResultBlob->nPatternClassify[nDefectCount - 1];
			pResultBlob->nArea[i] = pResultBlob->nArea[nDefectCount - 1];
			pResultBlob->ptLT[i] = pResultBlob->ptLT[nDefectCount - 1];
			pResultBlob->ptRT[i] = pResultBlob->ptRT[nDefectCount - 1];
			pResultBlob->ptRB[i] = pResultBlob->ptRB[nDefectCount - 1];
			pResultBlob->ptLB[i] = pResultBlob->ptLB[nDefectCount - 1];
			pResultBlob->dMeanGV[i] = pResultBlob->dMeanGV[nDefectCount - 1];
			pResultBlob->dSigma[i] = pResultBlob->dSigma[nDefectCount - 1];
			pResultBlob->nMinGV[i] = pResultBlob->nMinGV[nDefectCount - 1];
			pResultBlob->nMaxGV[i] = pResultBlob->nMaxGV[nDefectCount - 1];
			pResultBlob->dBackGroundGV[i] = pResultBlob->dBackGroundGV[nDefectCount - 1];
			pResultBlob->nCenterx[i] = pResultBlob->nCenterx[nDefectCount - 1];
			pResultBlob->nCentery[i] = pResultBlob->nCentery[nDefectCount - 1];
			pResultBlob->dBreadth[i] = pResultBlob->dBreadth[nDefectCount - 1];
			pResultBlob->dCompactness[i] = pResultBlob->dCompactness[nDefectCount - 1];
			pResultBlob->dF_Elongation[i] = pResultBlob->dF_Elongation[nDefectCount - 1];
			pResultBlob->dF_Min[i] = pResultBlob->dF_Min[nDefectCount - 1];
			pResultBlob->dF_Max[i] = pResultBlob->dF_Max[nDefectCount - 1];
			pResultBlob->Lab_avg_L[i] = pResultBlob->Lab_avg_L[nDefectCount - 1];
			pResultBlob->Lab_avg_a[i] = pResultBlob->Lab_avg_a[nDefectCount - 1];
			pResultBlob->Lab_avg_b[i] = pResultBlob->Lab_avg_b[nDefectCount - 1];
			pResultBlob->Lab_diff_L[i] = pResultBlob->Lab_diff_L[nDefectCount - 1];
			pResultBlob->Lab_diff_a[i] = pResultBlob->Lab_diff_a[nDefectCount - 1];
			pResultBlob->Lab_diff_b[i] = pResultBlob->Lab_diff_b[nDefectCount - 1];

#if USE_ALG_HIST
			memcpy(pResultBlob->nHist[i], pResultBlob->nHist[nDefectCount - 1], sizeof(__int64) * IMAGE_MAX_GV);
#endif

			pResultBlob->bUseResult[i] = pResultBlob->bUseResult[nDefectCount - 1];

			//清除一个不良总数
			nDefectCount--;
		}

		//需要确认是否需要初始化最后一个Index(可以不初始化吗？？)
	}

	//重置最终不良计数
	pResultBlob->nDefectCount = nDefectCount;

	return nErrorCode;
}

//只查找暗Dust面积小的情况(8 bit)
long CInspectPoint::DarkDustMaxArea(cv::Mat matSrcBuffer[E_DEFECT_COLOR_COUNT], int nMaxArea, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL
	if (matSrcBuffer[E_DEFECT_COLOR_BRIGHT].empty())	return E_ERROR_CODE_EMPTY_PARA;
	if (matSrcBuffer[E_DEFECT_COLOR_DARK].empty())		return E_ERROR_CODE_EMPTY_PARA;

	//仅响应8bit
//if( matSrcBuffer[E_DEFECT_COLOR_DARK].type() != CV_8U )		return E_ERROR_CODE_TRUE;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matLabelBuffer = cMatBufTemp.GetMat(matSrcBuffer[E_DEFECT_COLOR_DARK].size(), CV_32SC1, false);

	matSrcBuffer[E_DEFECT_COLOR_DARK].convertTo(matLabelBuffer, CV_32SC1);

	//检查区域Rect
	cv::Rect rectTemp;
	int nTempCount = 0;

	// 8 bit
	if (matSrcBuffer[E_DEFECT_COLOR_BRIGHT].type() == CV_8U)
	{
		for (int y = 0; y < matLabelBuffer.rows; y++)
		{
			int* row = (int*)matLabelBuffer.ptr(y);

			for (int x = 0; x < matLabelBuffer.cols; x++)
			{
				if (row[x] != 255)	continue;

				nTempCount++;

				//对象面积
				long nArea = cv::floodFill(matLabelBuffer, Point(x, y), nTempCount, &rectTemp);

				//面积限制
				if (nArea > nMaxArea)	continue;

				int nEndXX = rectTemp.x + rectTemp.width;
				int nEndYY = rectTemp.y + rectTemp.height;

				//异常处理
				if (nEndYY >= matLabelBuffer.rows)	nEndYY = matLabelBuffer.rows - 1;
				if (nEndXX >= matLabelBuffer.cols)	nEndXX = matLabelBuffer.cols - 1;

				//每个标签的像素坐标
				for (int j = rectTemp.y; j <= nEndYY; j++)
				{
					int* row2 = (int*)matLabelBuffer.ptr(j);
					BYTE* row3 = (BYTE*)matSrcBuffer[E_DEFECT_COLOR_BRIGHT].ptr(j);

					for (int i = rectTemp.x; i <= nEndXX; i++)
					{
						if (row2[i] != nTempCount)	continue;

						row3[i] = (BYTE)255;
					}
				}
			}
		}
	}
	// 12 bit
	else
	{
		for (int y = 0; y < matLabelBuffer.rows; y++)
		{
			int* row = (int*)matLabelBuffer.ptr(y);

			for (int x = 0; x < matLabelBuffer.cols; x++)
			{
				if (row[x] != 4095)	continue;

				nTempCount++;

				//对象面积
				long nArea = cv::floodFill(matLabelBuffer, Point(x, y), nTempCount, &rectTemp);

				//面积限制
				if (nArea > nMaxArea)	continue;

				int nEndXX = rectTemp.x + rectTemp.width;
				int nEndYY = rectTemp.y + rectTemp.height;

				//异常处理
				if (nEndYY >= matLabelBuffer.rows)	nEndYY = matLabelBuffer.rows - 1;
				if (nEndXX >= matLabelBuffer.cols)	nEndXX = matLabelBuffer.cols - 1;

				//每个标签的像素坐标
				for (int j = rectTemp.y; j <= nEndYY; j++)
				{
					int* row2 = (int*)matLabelBuffer.ptr(j);
					ushort* row3 = (ushort*)matSrcBuffer[E_DEFECT_COLOR_BRIGHT].ptr(j);

					for (int i = rectTemp.x; i <= nEndXX; i++)
					{
						if (row2[i] != nTempCount)	continue;

						row3[i] = (ushort)4095;
					}
				}
			}
		}
	}

	matLabelBuffer.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

//去除小面积
long CInspectPoint::DeleteMinArea(cv::Mat matSrcBuffer, cv::Mat matThBuffer, int nMinArea, int nMaxGV, CMatBuf* cMemSub)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//如果参数为NULL
	if (matSrcBuffer.empty())	return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThBuffer.empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	//缓冲区分配和初始化
	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	cv::Mat matLabelBuffer = cMatBufTemp.GetMat(matThBuffer.size(), CV_32SC1, false);

	//位转换
	matThBuffer.convertTo(matLabelBuffer, CV_32SC1);

	//初始化为0
	matThBuffer.setTo(0);

	//检查区域Rect
	cv::Rect rectTemp;
	int nTempCount = 0;

	// 8 bit
	if (matThBuffer.type() == CV_8U)
	{
		for (int y = 0; y < matLabelBuffer.rows; y++)
		{
			int* row = (int*)matLabelBuffer.ptr(y);

			for (int x = 0; x < matLabelBuffer.cols; x++)
			{
				if (row[x] != 255)	continue;

				nTempCount++;

				//对象面积
				long nArea = cv::floodFill(matLabelBuffer, Point(x, y), nTempCount, &rectTemp);

				//面积限制
				if (nArea < nMinArea)	continue;

				int nEndXX = rectTemp.x + rectTemp.width;
				int nEndYY = rectTemp.y + rectTemp.height;

				//异常处理
				if (nEndYY >= matLabelBuffer.rows)	nEndYY = matLabelBuffer.rows - 1;
				if (nEndXX >= matLabelBuffer.cols)	nEndXX = matLabelBuffer.cols - 1;

				double dMax = 0;
				cv::Mat matTempBuf = matSrcBuffer(rectTemp);
				cv::minMaxLoc(matTempBuf, NULL, &dMax);
				if (dMax > nMaxGV)		continue;

				//每个标签的像素坐标
				for (int j = rectTemp.y; j <= nEndYY; j++)
				{
					int* row2 = (int*)matLabelBuffer.ptr(j);
					BYTE* row3 = (BYTE*)matThBuffer.ptr(j);

					for (int i = rectTemp.x; i <= nEndXX; i++)
					{
						if (row2[i] != nTempCount)	continue;

						row3[i] = (BYTE)255;
					}
				}
			}
		}
	}
	// 12 bit
	else
	{
		for (int y = 0; y < matLabelBuffer.rows; y++)
		{
			int* row = (int*)matLabelBuffer.ptr(y);

			for (int x = 0; x < matLabelBuffer.cols; x++)
			{
				if (row[x] != 4095)	continue;

				nTempCount++;

				//对象面积
				long nArea = cv::floodFill(matLabelBuffer, Point(x, y), nTempCount, &rectTemp);

				//面积限制
				if (nArea > nMinArea)	continue;

				int nEndXX = rectTemp.x + rectTemp.width;
				int nEndYY = rectTemp.y + rectTemp.height;

				//异常处理
				if (nEndYY >= matLabelBuffer.rows)	nEndYY = matLabelBuffer.rows - 1;
				if (nEndXX >= matLabelBuffer.cols)	nEndXX = matLabelBuffer.cols - 1;

				double dMax = 0;
				cv::Mat matTempBuf = matSrcBuffer(rectTemp);
				cv::minMaxLoc(matTempBuf, NULL, &dMax);
				if (dMax > nMaxGV)		continue;

				//每个标签的像素坐标
				for (int j = rectTemp.y; j <= nEndYY; j++)
				{
					int* row2 = (int*)matLabelBuffer.ptr(j);
					ushort* row3 = (ushort*)matThBuffer.ptr(j);

					for (int i = rectTemp.x; i <= nEndXX; i++)
					{
						if (row2[i] != nTempCount)	continue;

						row3[i] = (ushort)4095;
					}
				}
			}
		}
	}

	matLabelBuffer.release();

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
	}

	return nErrorCode;
}

long CInspectPoint::FindBubble_DustImage(cv::Mat matSrcbuffer, cv::Mat& matBubbleResult, cv::Rect rtROI, cv::Rect* rcCHoleROI, double* dPara, int* nCommonPara, CString strAlgPath)
{
	bool bImgSave = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0 ? true : false;

	//参数
	int nBlurSize = dPara[E_PARA_POINT_DUST_BUBBLE_SRC_BLUR_SIZE];
	int nBKBlurSize = dPara[E_PARA_POINT_DUST_BUBBLE_BK_BLUR_SIZE];
	float fResize = dPara[E_PARA_POINT_DUST_BUBBLE_RESIZE];
	int nThreshold = dPara[E_PARA_POINT_DUST_BUBBLE_THRESHOLD];
	int nCloseSize = dPara[E_PARA_POINT_DUST_BUBBLE_CLOSE_SIZE];

	//异常处理(奇数)
	if (nBlurSize % 2 == 0)		nBlurSize++;
	if (nBKBlurSize % 2 == 0)	nBKBlurSize++;
	if (nCloseSize % 2 == 0)		nCloseSize++;

	//////////////////////////////////////////////////////////////////////////

		//点亮区域ROI
	cv::Mat matSrcROIBuf = matSrcbuffer(rtROI);

	if (bImgSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\Dust_Bubble_Org.jpg"), strAlgPath);
		ImageSave(strTemp, matSrcROIBuf);
	}

	double dPow = 1.4;
	cv::Mat matPow = cMem[0]->GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);

	AlgoBase::Pow(matSrcROIBuf, matPow, dPow, 4095, cMem[0]);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Pow."));

	//里萨兹0.25
	cv::Mat matResizeBuf = cMem[0]->GetMat(matSrcROIBuf.rows * fResize, matSrcROIBuf.cols * fResize, matSrcbuffer.type(), false);
	cv::resize(matPow, matResizeBuf, matResizeBuf.size());
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Resize."));

	//用于检测泡沫的配电商
	cv::Mat matResizeTemp1Buf = cMem[0]->GetMat(matResizeBuf.size(), matResizeBuf.type(), false);
	cv::Mat matResizeTemp2Buf = cMem[0]->GetMat(matResizeBuf.size(), matResizeBuf.type(), false);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Create Resize Buf."));

	// Blur
	cv::blur(matResizeBuf, matResizeTemp1Buf, cv::Size(nBlurSize, nBlurSize));		// Src
	cv::blur(matResizeBuf, matResizeTemp2Buf, cv::Size(nBKBlurSize, nBKBlurSize));	// BackGround

	if (bImgSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\Dust_Bubble_Src.jpg"), strAlgPath);
		ImageSave(strTemp, matResizeTemp1Buf);

		strTemp.Format(_T("%s\\Dust_Bubble_BackGround.jpg"), strAlgPath);
		ImageSave(strTemp, matResizeTemp2Buf);
	}

	//气泡暗？(从背景中删除源文件)
	cv::subtract(matResizeTemp2Buf, matResizeTemp1Buf, matResizeBuf);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("subtract."));

	//二进制
	cv::threshold(matResizeBuf, matResizeTemp1Buf, nThreshold, 255, THRESH_BINARY);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("threshold."));

	//填充
	cv::morphologyEx(matResizeTemp1Buf, matResizeTemp2Buf, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT, cv::Size(nCloseSize, nCloseSize)));
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Morp."));

	//Lisaiz元福
	cv::Mat matResROIBuf = cMem[0]->GetMat(matSrcROIBuf.size(), matSrcROIBuf.type(), false);
	cv::resize(matResizeTemp2Buf, matResROIBuf, matResROIBuf.size());
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Resize."));

	//删除CHole
	for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
	{
		cv::Point ptCHoleCenter;
		ptCHoleCenter.x = (rcCHoleROI[i].x + rcCHoleROI[i].width / 2) - rtROI.x;
		ptCHoleCenter.y = (rcCHoleROI[i].y + rcCHoleROI[i].height / 2) - rtROI.y;
		int nCHoleraius = (rcCHoleROI[i].width + rcCHoleROI[i].height) / 2;

		cv::circle(matResROIBuf, ptCHoleCenter, nCHoleraius, cv::Scalar(0), -1);
	}

	if (bImgSave)
	{
		CString strTemp;
		strTemp.Format(_T("%s\\Dust_Bubble_TH.jpg"), strAlgPath);
		ImageSave(strTemp, matResROIBuf);
	}

	//用检测到的气泡200GV填充(仅点亮区域)
	matBubbleResult(rtROI).setTo(200, matResROIBuf);
	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Bubble Set."));

	return E_ERROR_CODE_TRUE;
}

void CInspectPoint::Insp_RectSet(cv::Rect& rectInspROI, CRect& rectROI, int nWidth, int nHeight, int nOffset)
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

long CInspectPoint::LogicStart_CholePoint_G87(cv::Mat& matSrcImage, cv::Mat* matDstImage, CRect rectROI, double* dPara,
	int* nCommonPara, CString strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer)
{
	//错误代码
	long	nErrorCode = E_ERROR_CODE_TRUE;

	//double	dDarkDist = dPara[E_PARA_POINT_RGB_COMMON_DARK_DIST];

	long	nDelLineBrightCntX = (long)dPara[E_PARA_POINT_G87_DEL_LINE_BRIGHT_CNT_X];		// 删除行x方向计数
	long	nDelLineBrightCntY = (long)dPara[E_PARA_POINT_G87_DEL_LINE_BRIGHT_CNT_Y];		// 删除行y方向计数
	long	nDelLineBrightThickX = (long)dPara[E_PARA_POINT_G87_DEL_LINE_BRIGHT_THICK_X];	// 删除行x厚度
	long	nDelLineBrightThickY = (long)dPara[E_PARA_POINT_G87_DEL_LINE_BRIGHT_THICK_Y];	// 删除行y厚度

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

	// 	int		nShift_X = 0;
	// 	int		nShift_Y = 5;

	// 	int		nTh = 20;
	// 	int		nTh_Max = 140;

	int		nGaussianSize = (int)dPara[E_PARA_POINT_G87_GAUSSIAN_SIZE];					// 31
	double	dGaussianSigma = (double)dPara[E_PARA_POINT_G87_GAUSSIAN_SIGMA];				// 5.0

	int		nGaussianSize2 = (int)dPara[E_PARA_POINT_G87_GAUSSIAN_SIZE2];					// 31
	double	dGaussianSigma2 = (double)dPara[E_PARA_POINT_G87_GAUSSIAN_SIGMA];				// 3.0

	int		nDilateSize = (int)dPara[E_PARA_POINT_G87_CHOLE_MASK_SIZE_UP];				// 5

	double  dDarkTemp = (double)dPara[E_PARA_POINT_G87_CHOLE_POINT_DARK_RATIO];		// 0.73
	double  dBrightTemp = (double)dPara[E_PARA_POINT_G87_CHOLE_POINT_BRIGHT_RATIO];		// 2.0

	//缩小检查区域的轮廓
	CRect rectTemp(rectROI);

	int nOffSet = 100;

	cv::Rect rtInspROI;

	Insp_RectSet(rtInspROI, rectTemp, matSrcImage.cols, matSrcImage.rows, nOffSet);

	//G87 Chole检查测试用途
	for (int j = 0; j < MAX_MEM_SIZE_E_INSPECT_AREA; j++)
	{
		if (!matCholeBuffer[j].empty())
		{
			//导入Chole区域

			cv::Mat matCholeArea = matSrcImage(rcCHoleROI[j]).clone();

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Chole Area.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matCholeArea);
			}

			//使用高斯模糊填充空部分
			cv::Mat matGaus;
			cv::GaussianBlur(matCholeArea, matGaus, Size(nGaussianSize, nGaussianSize), dGaussianSigma);

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_GaussianBlur.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matGaus);
			}

			//使用该图像创建掩码
			cv::Mat matMask;
			cv::threshold(matGaus, matMask, cv::mean(matGaus)[0] - 10, 255, THRESH_BINARY_INV);

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Chole Mask.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matMask);
			}

			//增大掩码大小
			cv::morphologyEx(matMask, matMask, cv::MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(nDilateSize, nDilateSize)));

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Chole Mask_Size UP.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matMask);
			}

			//检查的高斯安布尔
			cv::GaussianBlur(matCholeArea, matGaus, Size(nGaussianSize2, nGaussianSize2), dGaussianSigma2);

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_GaussianBlur2.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matGaus);
			}

			//使Chole亮度像周围的亮度一样亮
			double dOffset = cv::mean(matGaus, ~matMask)[0] / cv::mean(matGaus, matMask)[0];

			cv::Mat matBuff;
			cv::bitwise_and(matGaus, matMask, matBuff);
			matBuff *= dOffset;

			cv::subtract(matGaus, matMask, matGaus);
			cv::add(matGaus, matBuff, matGaus);

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_AVG_GV.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matGaus);
			}

			//二进制
			cv::Mat matTh;

			// E_DEFECT_COLOR_DARK Threshold
			cv::threshold(matGaus, matTh, cv::mean(matGaus)[0] * dDarkTemp, 255.0, THRESH_BINARY_INV);

			cv::bitwise_and(matTh, matMask, matTh);

			matTh.copyTo(matDstImage[E_DEFECT_COLOR_DARK](rcCHoleROI[j]));

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Threshole_Dark_ROI.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matTh);

				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Threshole_Dark.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_DARK]);
			}

			// E_DEFECT_COLOR_BRIGHT Threshold
			cv::threshold(matGaus, matTh, cv::mean(matGaus)[0] * dBrightTemp, 255.0, THRESH_BINARY);

			cv::bitwise_and(matTh, matMask, matTh);

			matTh.copyTo(matDstImage[E_DEFECT_COLOR_BRIGHT](rcCHoleROI[j]));

			//检查中间映像
			if (bImageSave)
			{
				CString strTemp;
				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Threshole_Bright_ROI.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matTh);

				strTemp.Format(_T("%s\\%02d_%02d_%02d_Chole_POINT_%02d_Threshole_Bright.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
				ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
			}

			//			//检查中间映像

			//			//检查中间映像

			//			//检查中间映像

			//			//检查中间映像

			//			//检查中间映像

			//			//检查中间映像

			matCholeArea.release();
			//matShiftImage.release();
			matTh.release();
			matGaus.release();
			matMask.release();
			matBuff.release();
		}
	}

	//检查中间映像

	//Bright消除小面积不良
	//Blob的速度慢。
	nErrorCode = DeleteArea(matDstImage[E_DEFECT_COLOR_BRIGHT](rtInspROI), 3, cMem[0]);
	if (nErrorCode != E_ERROR_CODE_TRUE)	return nErrorCode;

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("DeleteArea (BRIGHT)."));

	//删除line错误
	if (nDelLineBrightCntX > 0 || nDelLineBrightCntY > 0)
		AlgoBase::ProjectionLineDelete(matDstImage[E_DEFECT_COLOR_BRIGHT], nDelLineBrightCntX, nDelLineBrightCntY, nDelLineBrightThickX, nDelLineBrightThickY);

	writeInspectLog(E_ALG_TYPE_AVI_POINT, __FUNCTION__, _T("Projection."));

	//检查中间映像
	if (bImageSave)
	{
		CString strTemp;

		strTemp.Format(_T("%s\\%02d_%02d_%02d_POINT_%02d_Bright_Delete.jpg"), strAlgPath, nImageNum, nCamNum, nROINumber, nSaveImageCount++);
		ImageSave(strTemp, matDstImage[E_DEFECT_COLOR_BRIGHT]);
	}

	return E_ERROR_CODE_TRUE;
}

void CInspectPoint::ImageShift(cv::Mat& matSrc, cv::Mat& matDst, int nShift_X, int nShift_Y)
{
	//画面大小
	int nImageSizeX = matSrc.cols;
	int nImageSizeY = matSrc.rows;

	cv::Mat matSrcBuf = matSrc.clone();
	cv::Mat matDstBuf = matDst.clone();

	//临时缓冲区
	cv::Mat matTempBuf1, matTempBuf2;

	//x方向
	int nOffsetX = 0;

	nOffsetX = 1 * nShift_X;

	matTempBuf1 = matDstBuf(cv::Rect(nOffsetX, 0, nImageSizeX - nOffsetX, nImageSizeY));
	matTempBuf2 = matSrcBuf(cv::Rect(0, 0, nImageSizeX - nOffsetX, nImageSizeY));

	//积分不良时,可能会打开不应该点亮的部分的数组
	//如果覆盖的话,不良现象就会消失,所以无法使用
	cv::add(matTempBuf1, matTempBuf2, matTempBuf1);

	//y方向
	int nOffsetY = 0;
	matDstBuf.copyTo(matSrcBuf);

	nOffsetY = 1 * nShift_Y;

	matTempBuf1 = matDst(cv::Rect(0, nOffsetY, nImageSizeX, nImageSizeY - nOffsetY));
	matTempBuf2 = matSrcBuf(cv::Rect(0, 0, nImageSizeX, nImageSizeY - nOffsetY));

	cv::add(matTempBuf1, matTempBuf2, matTempBuf1);

}
