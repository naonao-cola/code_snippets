
/************************************************************************/
//Blob相关源
//修改日期:17.03.08
/************************************************************************/

#include "StdAfx.h"

#include "FeatureExtraction.h"

//并行处理
#include <ppl.h>
using namespace Concurrency;

//////////////////////////////////////////////////////////////////////////

CFeatureExtraction::CFeatureExtraction(void)
{
	m_bComplete = false;

	//初始化向量
	vector<tBLOB_FEATURE>().swap(m_BlobResult);

	InitializeCriticalSectionAndSpinCount(&m_csCoordFile, 4000);

	cMem = NULL;
	m_cInspectLibLog = NULL;
	m_strAlgLog = NULL;
	m_tInitTime = 0;
	m_tBeforeTime = 0;

	//超过设置时间后,类结束(40秒)
	cTimeOut.SetTimeOut(40000);

	///AI Detect/////
	////
	spTaskInfo = std::make_shared<stTaskInfo>();
}

CFeatureExtraction::~CFeatureExtraction(void)
{
	DeleteCriticalSection(&m_csCoordFile);

	Release();
}

bool CFeatureExtraction::Release()
{
	m_bComplete = false;

	//初始化向量
	if (m_BlobResult.size() != 0)
	{
		for (int i = 0; i < m_BlobResult.size(); i++)
		{
			vector<cv::Point>().swap(m_BlobResult[i].ptIndexs);
			vector <cv::Point>().swap(m_BlobResult[i].ptContours);
		}
		vector<tBLOB_FEATURE>().swap(m_BlobResult);
	}
	return true;
}

bool CFeatureExtraction::DoBlobCalculate(cv::Mat ThresholdBuffer, cv::Mat GrayBuffer, int nMaxDefectCount)
{
	//确认Blob是否已完成。
	m_bComplete = false;

	//禁用内存
	Release();

	//初始化向量
	if (m_BlobResult.size() != 0)
	{
		for (int i = 0; i < m_BlobResult.size(); i++)
		{
			vector<cv::Point>().swap(m_BlobResult[i].ptIndexs);
			vector <cv::Point>().swap(m_BlobResult[i].ptContours);
		}
		vector<tBLOB_FEATURE>().swap(m_BlobResult);
	}

		//如果没有画面,则返回
	if (ThresholdBuffer.empty())			return false;

	//如果不是1频道
	if (ThresholdBuffer.channels() != 1)	return false;

	//如果Gray画面不存在X&1频道
	bool bGrayEmpty = false;
	if (GrayBuffer.empty() || GrayBuffer.channels() != 1)
	{
		GrayBuffer = ThresholdBuffer.clone();
		bGrayEmpty = true;
	}

	//画面转换
//cv::Mat LabelBuffer;
//ThresholdBuffer.convertTo(LabelBuffer, CV_32SC1);

	//Label计算(8方向)
//int nTotalLabel = cv::connectedComponents(ThresholdBuffer, LabelBuffer) - 1;

	writeInspectLog(__FUNCTION__, _T("Start."));

	cv::Mat matLabel, matStats, matCentroid;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);
	matLabel = cMatBufTemp.GetMat(ThresholdBuffer.size(), CV_32SC1);

	writeInspectLog(__FUNCTION__, _T("Mat Create."));

	__int64 nTotalLabel = 0;

	if (ThresholdBuffer.type() == CV_8U)
	{
		nTotalLabel = cv::connectedComponentsWithStats(ThresholdBuffer, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;
	}
	else
	{
		cv::Mat matSrc8bit = cMatBufTemp.GetMat(ThresholdBuffer.size(), CV_8UC1);
		ThresholdBuffer.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

		nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

		matSrc8bit.release();
	}

	writeInspectLog(__FUNCTION__, _T("connectedComponents."));

	//异常处理
	if (nTotalLabel < 0)
	{
		//禁用内存
		if (bGrayEmpty)			GrayBuffer.release();
		if (!matLabel.empty())		matLabel.release();
		if (!matStats.empty())		matStats.release();
		if (!matCentroid.empty())	matCentroid.release();

		return false;
	}

	//异常处理
	if (nTotalLabel >= nMaxDefectCount)
		nTotalLabel = nMaxDefectCount - 1;

	//默认Feature计算8bit
	if (GrayBuffer.type() == CV_8UC1)
		DoFeatureBasic_8bit(matLabel, matStats, matCentroid, GrayBuffer, (int)nTotalLabel, &cMatBufTemp);

	//默认Feature计算16bit
	else
		DoFeatureBasic_16bit(matLabel, matStats, matCentroid, GrayBuffer, (int)nTotalLabel, &cMatBufTemp);

	writeInspectLog(__FUNCTION__, _T("FeatureBasic."));

	//禁用内存
	if (bGrayEmpty)			GrayBuffer.release();
	if (!matLabel.empty())		matLabel.release();
	if (!matStats.empty())		matStats.release();
	if (!matCentroid.empty())	matCentroid.release();

	writeInspectLog(__FUNCTION__, _T("Release."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory(), m_nAlgType);
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory(), m_nAlgType);
	}
	//Blob完成
	m_bComplete = true;

	return true;
}
bool CFeatureExtraction::DoBlobCalculate(cv::Mat ThresholdBuffer, CRect rectROI, cv::Mat GrayBuffer, int nMaxDefectCount)
{
	//确认Blob是否已完成。
	m_bComplete = false;

	//禁用内存
	Release();

	//初始化向量
	if (m_BlobResult.size() != 0)
	{
		for (int i = 0; i < m_BlobResult.size(); i++)
		{
			vector<cv::Point>().swap(m_BlobResult[i].ptIndexs);
			vector <cv::Point>().swap(m_BlobResult[i].ptContours);
		}
		vector<tBLOB_FEATURE>().swap(m_BlobResult);
	}

		//如果没有画面,则返回
	if (ThresholdBuffer.empty())			return false;

	//如果不是1频道
	if (ThresholdBuffer.channels() != 1)	return false;

	//如果Gray画面不存在X&1频道
	bool bGrayEmpty = false;
	if (GrayBuffer.empty() || GrayBuffer.channels() != 1)
	{
		GrayBuffer = ThresholdBuffer.clone();
		bGrayEmpty = true;
	}

	writeInspectLog(__FUNCTION__, _T("Start."));

	cv::Mat matLabel, matStats, matCentroid;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMem);
	matLabel = cMatBufTemp.GetMat(ThresholdBuffer.size(), CV_32SC1, false);

	writeInspectLog(__FUNCTION__, _T("Mat Create."));

	__int64 nTotalLabel = 0;

	if (ThresholdBuffer.type() == CV_8U)
	{
		nTotalLabel = cv::connectedComponentsWithStats(ThresholdBuffer, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;
	}
	else
	{
		cv::Mat matSrc8bit = cMatBufTemp.GetMat(ThresholdBuffer.size(), CV_8UC1);
		ThresholdBuffer.convertTo(matSrc8bit, CV_8UC1, 1. / 16.);

		nTotalLabel = cv::connectedComponentsWithStats(matSrc8bit, matLabel, matStats, matCentroid, 8, CV_32S, CCL_GRANA) - 1;

		matSrc8bit.release();
	}

	writeInspectLog(__FUNCTION__, _T("connectedComponents."));

	//异常处理
	if (nTotalLabel < 0)
	{
		//禁用内存
		if (bGrayEmpty)				GrayBuffer.release();
		if (!matLabel.empty())		matLabel.release();
		if (!matStats.empty())		matStats.release();
		if (!matCentroid.empty())	matCentroid.release();

		return false;
	}

	//异常处理
	if (nTotalLabel >= nMaxDefectCount)
		nTotalLabel = nMaxDefectCount - 1;

	//默认Feature计算8bit
	if (GrayBuffer.type() == CV_8UC1)
		DoFeatureBasic_8bit(matLabel, matStats, matCentroid, GrayBuffer, (int)nTotalLabel, &cMatBufTemp);

	//默认Feature计算16bit
	else
		DoFeatureBasic_16bit(matLabel, matStats, matCentroid, GrayBuffer, (int)nTotalLabel, &cMatBufTemp);

	writeInspectLog(__FUNCTION__, _T("FeatureBasic."));

	CoordApply(rectROI, nTotalLabel);

	//禁用内存
	if (bGrayEmpty)			GrayBuffer.release();
	if (!matLabel.empty())		matLabel.release();
	if (!matStats.empty())		matStats.release();
	if (!matCentroid.empty())	matCentroid.release();

	writeInspectLog(__FUNCTION__, _T("Release."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory(), m_nAlgType);
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory(), m_nAlgType);
	}

	//Blob完成
	m_bComplete = true;

	return true;
}

//坐标校正
void	CFeatureExtraction::CoordApply(CRect rectROI, int nTotalLabel)
{
	for (int nBlobNum = 0; nBlobNum < nTotalLabel; nBlobNum++)
	{
		m_BlobResult.at(nBlobNum).rectBox.x += rectROI.left;
		m_BlobResult.at(nBlobNum).rectBox.y += rectROI.top;

		m_BlobResult.at(nBlobNum).ptCenter.x += rectROI.left;
		m_BlobResult.at(nBlobNum).ptCenter.y += rectROI.top;

		for (int idx = 0; idx < m_BlobResult.at(nBlobNum).ptIndexs.size(); idx++)
		{
			m_BlobResult.at(nBlobNum).ptIndexs[idx].x += rectROI.left;
			m_BlobResult.at(nBlobNum).ptIndexs[idx].y += rectROI.top;
		}

		for (int idx = 0; idx < m_BlobResult.at(nBlobNum).ptContours.size(); idx++)
		{
			m_BlobResult.at(nBlobNum).ptContours[idx].x += rectROI.left;
			m_BlobResult.at(nBlobNum).ptContours[idx].y += rectROI.top;
		}
	}
}

//Blob&判定结果
long CFeatureExtraction::DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, int nDefectType, bool bPtRotate)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 				return E_ERROR_CODE_EMPTY_PARA;

	if (nDefectType < 0)							return E_ERROR_CODE_EMPTY_PARA;
	if (nDefectType >= E_DEFECT_JUDGEMENT_COUNT)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())						return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())					return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];						//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;	//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];						//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];						//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];						//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;	//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;	//09:Draw Dust显示
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];						//10:UI上的模式顺序画面编号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;				//11:Cell旋转角度(Align计算值,仅小数点3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//输入的错误数量
////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// /////////////////////////////////
	int& nDefectCount = pResultBlob->nDefectCount;

	//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage, nMaxDefectCount);

	//选择的Defect列表
	int nFork = nDefectType;

	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[m_BlobResult[i].nBlockNum];
				break;
			}

		}
		//仅当选择Defect列表时...
		if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
			return E_ERROR_CODE_TRUE;

		//过滤不良颜色
		if (!DoColorFilter(nFork, nDefectColor))
			continue;

		//每个判定项目2个范围
		int nFeatureCount = E_FEATURE_COUNT * 2;
		bool bFilter = true;
		bool bInit = false;

		for (int nForj = 0; nForj < nFeatureCount; nForj++)
		{

			//仅当选择判定项目时
			if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
				continue;

			//哪怕只有一次动作。
			bInit = true;

			//如果满足设置的过滤,则返回true/如果不满足,则返回false
			if (!DoFiltering(
				m_BlobResult[i],//Blob结果
				nForj / 2,//比较Feature
				blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
				blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
			{

				bFilter = false;
				break;
			}
		}

		//如果满足所有设置的条件,请输入结果
		if (bInit && bFilter)
		{
			m_BlobResult[i].bFiltering = true;

			//转角信息
			if (bPtRotate)
			{
				int nL, nT, nR, nB;

				//旋转时,计算预测坐标
				int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
				int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

				//初始设置
				nL = nR = X;
				nT = nB = Y;

				//外围线数量
				for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
				{
					//旋转时,计算预测坐标
					X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
					Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

					//更新
					if (nL > X)	nL = X;
					if (nR < X)	nR = X;
					if (nT > Y)	nT = Y;
					if (nB < Y)	nB = Y;
				}

				cv::Point ptTemp;

				ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
				pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
				pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
				pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
				pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;
			}
			else
			{
				int nL, nT, nR, nB;
				int X = m_BlobResult[i].ptContours[0].x;
				int Y = m_BlobResult[i].ptContours[0].y;

				//初始设置
				nL = nR = X;
				nT = nB = Y;

				//外围线数量
				for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
				{
					X = m_BlobResult[i].ptContours[k].x;
					Y = m_BlobResult[i].ptContours[k].y;

					//更新
					if (nL > X)	nL = X;
					if (nR < X)	nR = X;
					if (nT > Y)	nT = Y;
					if (nB < Y)	nB = Y;
				}

				pResultBlob->ptLT[nDefectCount].x = nL;
				pResultBlob->ptLT[nDefectCount].y = nT;

				pResultBlob->ptRT[nDefectCount].x = nR;
				pResultBlob->ptRT[nDefectCount].y = nT;

				pResultBlob->ptRB[nDefectCount].x = nR;
				pResultBlob->ptRB[nDefectCount].y = nB;

				pResultBlob->ptLB[nDefectCount].x = nL;
				pResultBlob->ptLB[nDefectCount].y = nB;
			}

			//放入要交给UI的结果
			pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
			pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
			pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
			pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

			pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
			pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

			pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

			pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
			pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
			pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

			pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
			pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
			pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
			pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
			pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;

			pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
			pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
			pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
			//17.06.24对象直方图
			memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
			//17.11.29-外围信息(AVI&SVI其他工具)
			calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

			//绘制错误编号
			if (!matDrawBuffer.empty() && bDefectNum)
			{
				cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

				char str[256] = { 0, };
				sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
				cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
				cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
			}

			//最后的不良计数增加
			nDefectCount++;
		}

		//超过最大不良数量时退出
		if (nDefectCount >= nMaxDefectCount)
			break;

	}





	return E_ERROR_CODE_TRUE;
}

//Blob&判定结果ROI
long CFeatureExtraction::DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, CRect rectROI,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, int nDefectType, bool bPtRotate)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)				return E_ERROR_CODE_EMPTY_PARA;

	if (nDefectType < 0)							return E_ERROR_CODE_EMPTY_PARA;
	if (nDefectType >= E_DEFECT_JUDGEMENT_COUNT)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())						return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())					return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
	//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	////////////////////////////////
	// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// 
	// /////////////////////////////////

		//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;

	//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, rectROI, matSrcImage, nMaxDefectCount);

	//选择的Defect列表
	int nFork = nDefectType;
	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[m_BlobResult[i].nBlockNum];
				break;
			}

		}
		//仅当选择Defect列表时...
		if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
			return E_ERROR_CODE_TRUE;


		//过滤不良颜色
		if (!DoColorFilter(nFork, nDefectColor))
			continue;

		//每个判定项目2个范围
		int nFeatureCount = E_FEATURE_COUNT * 2;
		bool bFilter = true;
		bool bInit = false;

		for (int nForj = 0; nForj < nFeatureCount; nForj++)
		{

			//仅当选择判定项目时
			if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
				continue;

			//哪怕只有一次动作。
			bInit = true;

			//如果满足设置的过滤,则返回true/如果不满足,则返回false
			if (!DoFiltering(
				m_BlobResult[i],//Blob结果
				nForj / 2,//比较Feature
				blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
				blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
			{

				bFilter = false;
				break;
			}
		}

		//如果满足所有设置的条件,请输入结果
		if (bInit && bFilter)
		{
			m_BlobResult[i].bFiltering = true;

			//转角信息
			if (bPtRotate)
			{
				int nL, nT, nR, nB;

				//旋转时,计算预测坐标
				int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
				int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

				//初始设置
				nL = nR = X;
				nT = nB = Y;

				//外围线数量
				for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
				{
					//旋转时,计算预测坐标
					X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
					Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

					//更新
					if (nL > X)	nL = X;
					if (nR < X)	nR = X;
					if (nT > Y)	nT = Y;
					if (nB < Y)	nB = Y;
				}

				cv::Point ptTemp;

				ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
				pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
				pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
				pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
				pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;
			}
			else
			{
				int nL, nT, nR, nB;
				int X = m_BlobResult[i].ptContours[0].x;
				int Y = m_BlobResult[i].ptContours[0].y;

				//初始设置
				nL = nR = X;
				nT = nB = Y;

				//外围线数量
				for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
				{
					X = m_BlobResult[i].ptContours[k].x;
					Y = m_BlobResult[i].ptContours[k].y;

					//更新
					if (nL > X)	nL = X;
					if (nR < X)	nR = X;
					if (nT > Y)	nT = Y;
					if (nB < Y)	nB = Y;
				}

				pResultBlob->ptLT[nDefectCount].x = nL;
				pResultBlob->ptLT[nDefectCount].y = nT;

				pResultBlob->ptRT[nDefectCount].x = nR;
				pResultBlob->ptRT[nDefectCount].y = nT;

				pResultBlob->ptRB[nDefectCount].x = nR;
				pResultBlob->ptRB[nDefectCount].y = nB;

				pResultBlob->ptLB[nDefectCount].x = nL;
				pResultBlob->ptLB[nDefectCount].y = nB;
			}

			//放入要交给UI的结果
			pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
			pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
			pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
			pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

			pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
			pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

			pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

			pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
			pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
			pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

			pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
			pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
			pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
			pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
			pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;


			pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
			pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
			pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
			//17.06.24对象直方图
			memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
			//17.11.29-外围信息(AVI&SVI其他工具)
			calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

			//绘制错误编号
			if (!matDrawBuffer.empty() && bDefectNum)
			{
				cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

				char str[256] = { 0, };
				sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
				cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
				cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
			}

			//最后的不良计数增加
			nDefectCount++;
		}

		//超过最大不良数量时退出
		if (nDefectCount >= nMaxDefectCount)
			break;

	}

	return E_ERROR_CODE_TRUE;
}

//Blob&判定结果
long CFeatureExtraction::DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool bPtRotate)
{
	writeInspectLog(__FUNCTION__, _T("Start."));

	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;
	////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];

	// /////////////////////////////////

		//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage, nMaxDefectCount);

	writeInspectLog(__FUNCTION__, _T("BlobCalculate."));
	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[nBlockNum];
				break;
			}

		}
		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	continue;

		//17.10.16[临时]-E_DEFECT_JUDGEMENT_MURA_MULT_BP优先级
		bool	bMultFlag = false;
		bool	bMultCalcFlag = false;
		if (blockJudgeInfo->stDefectItem[E_DEFECT_JUDGEMENT_MURA_MULT_BP].bDefectItemUse)
			bMultFlag = true;

		//Defect列表的数量
		for (int nFork = 0; nFork < E_DEFECT_JUDGEMENT_COUNT; nFork++)
		{
			//17.10.16[临时]-进行E_DEFECT_JUDGEMENT_MURA_MULT_BP后,从头开始
			if (bMultFlag && bMultCalcFlag)
			{
				bMultFlag = false;
				nFork = 0;
			}

			//17.10.16[临时]-E_DEFECT_JUDGEMENT_MURA_MULT_BP优先
			else if (bMultFlag)
			{
				nFork = E_DEFECT_JUDGEMENT_MURA_MULT_BP;
				bMultCalcFlag = true;
			}

			//仅当选择Defect列表时...
			else if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
				continue;

			//过滤不良颜色
			if (!DoColorFilter(nFork, nDefectColor))
				continue;

			//每个判定项目2个范围
			int nFeatureCount = E_FEATURE_COUNT * 2;
			bool bFilter = true;
			bool bInit = false;
			for (int nForj = 0; nForj < nFeatureCount; nForj++)
			{
				//仅当选择判定项目时
				if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
					continue;

				//哪怕只有一次动作。
				bInit = true;

				//如果满足设置的过滤,则返回true/如果不满足,则返回false
				if (!DoFiltering(
					m_BlobResult[i],//Blob结果
					nForj / 2,//比较Feature
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
				{
					bFilter = false;
					break;
				}
			}

			//如果满足所有设置的条件,请输入结果
			if (bInit && bFilter)
			{
				m_BlobResult[i].bFiltering = true;

				//转角信息
				if (bPtRotate)
				{
					int nL, nT, nR, nB;

					//旋转时,计算预测坐标
					int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
					int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

					//初始设置
					nL = nR = X;
					nT = nB = Y;

					//外围线数量
					for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
					{
						//旋转时,计算预测坐标
						X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
						Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

						//更新
						if (nL > X)	nL = X;
						if (nR < X)	nR = X;
						if (nT > Y)	nT = Y;
						if (nB < Y)	nB = Y;
					}

					cv::Point ptTemp;

					ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
					pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

					ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
					pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

					ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
					pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

					ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
					pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;
				}
				else
				{
					int nL, nT, nR, nB;
					int X = m_BlobResult[i].ptContours[0].x;
					int Y = m_BlobResult[i].ptContours[0].y;

					//初始设置
					nL = nR = X;
					nT = nB = Y;

					//外围线数量
					for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
					{
						X = m_BlobResult[i].ptContours[k].x;
						Y = m_BlobResult[i].ptContours[k].y;

						//更新
						if (nL > X)	nL = X;
						if (nR < X)	nR = X;
						if (nT > Y)	nT = Y;
						if (nB < Y)	nB = Y;
					}

					pResultBlob->ptLT[nDefectCount].x = nL;
					pResultBlob->ptLT[nDefectCount].y = nT;

					pResultBlob->ptRT[nDefectCount].x = nR;
					pResultBlob->ptRT[nDefectCount].y = nT;

					pResultBlob->ptRB[nDefectCount].x = nR;
					pResultBlob->ptRB[nDefectCount].y = nB;

					pResultBlob->ptLB[nDefectCount].x = nL;
					pResultBlob->ptLB[nDefectCount].y = nB;
				}

				//放入要交给UI的结果
				pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
				pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
				pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
				pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

				pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
				pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

				pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

				pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
				pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
				pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
				pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

				pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
				pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
				pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
				pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
				pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
				pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
				pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;

				pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
				pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
				pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
				//17.06.24对象直方图
				memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
				//17.11.29-外围信息(AVI&SVI其他工具)
				calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

				//绘制错误编号
				if (!matDrawBuffer.empty() && bDefectNum)
				{
					cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

					char str[256] = { 0, };
					sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
					cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
					cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
				}

				//最后的不良计数增加
				nDefectCount++;

				break;
			}
		}

		//超过最大不良数量时退出
		if (nDefectCount >= nMaxDefectCount)
			break;

	}
	writeInspectLog(__FUNCTION__, _T("Filtering & Result."));

	//检查时间限制
	if (cTimeOut.GetTimeOutFlag())	return E_ERROR_CODE_TIME_OUT;

	return E_ERROR_CODE_TRUE;
}
//Blob&判定结果ROI
long CFeatureExtraction::DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer, CRect rectROI,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool bPtRotate)
{
	writeInspectLog(__FUNCTION__, _T("Start."));

	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
	//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;
	////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// 
	// /////////////////////////////////

		//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, rectROI, matSrcImage, nMaxDefectCount);

	writeInspectLog(__FUNCTION__, _T("BlobCalculate."));

	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[nBlockNum];
				break;
			}

		}
		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	continue;

		//17.10.16[临时]-E_DEFECT_JUDGEMENT_MURA_MULT_BP优先级
		bool	bMultFlag = false;
		bool	bMultCalcFlag = false;
		if (blockJudgeInfo->stDefectItem[E_DEFECT_JUDGEMENT_MURA_MULT_BP].bDefectItemUse)
			bMultFlag = true;

		//Defect列表的数量
		for (int nFork = 0; nFork < E_DEFECT_JUDGEMENT_COUNT; nFork++)
		{
			//17.10.16[临时]-进行E_DEFECT_JUDGEMENT_MURA_MULT_BP后,从头开始
			if (bMultFlag && bMultCalcFlag)
			{
				bMultFlag = false;
				nFork = 0;
			}

			//17.10.16[临时]-E_DEFECT_JUDGEMENT_MURA_MULT_BP优先
			else if (bMultFlag)
			{
				nFork = E_DEFECT_JUDGEMENT_MURA_MULT_BP;
				bMultCalcFlag = true;
			}

			//仅当选择Defect列表时...
			else if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
				continue;

			//检测不到Retest Point
			//B11客户请求
			else if (nFork == E_DEFECT_JUDGEMENT_RETEST_POINT_DARK || nFork == E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT)
				continue;
			//未检测到POINT_RGB
			//B11客户请求
			else if (nFork == E_DEFECT_JUDGEMENT_POINT_RGB_DARK || nFork == E_DEFECT_JUDGEMENT_POINT_RGB_BRIGHT)
				continue;

			//过滤不良颜色
			if (!DoColorFilter(nFork, nDefectColor))
				continue;

			//每个判定项目2个范围
			int nFeatureCount = E_FEATURE_COUNT * 2;
			bool bFilter = true;
			bool bInit = false;
			for (int nForj = 0; nForj < nFeatureCount; nForj++)
			{
				//仅当选择判定项目时
				if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
					continue;

				////////////////////////////////////////////////////////////////////////// 04.20 choi
				if (nForj == E_FEATURE_GVAREA_RATIO_TEST * 2 || nForj == (E_FEATURE_GVAREA_RATIO_TEST * 2) + 1) {

					double dValue = blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue;

					int nTmp = (int)dValue % 10000;
					double nPer = ((double)dValue - (double)nTmp) / 10000.0;
					double nRatio = nTmp / 1000;

					double Mean_GV = m_BlobResult[i].fBKGV * nRatio;

					if (Mean_GV < 0)				Mean_GV = 0;
					if (Mean_GV > IMAGE_MAX_GV)	Mean_GV = IMAGE_MAX_GV - 1;

					__int64 nHist = 0;
					for (int m = Mean_GV; m <= 255; m++)
						nHist += m_BlobResult[i].nHist[m];

					double Area_per = nHist / m_BlobResult[i].nBoxArea;
					Area_per *= 100;

					m_BlobResult[i].fAreaPer = Area_per;
					m_BlobResult[i].nJudge_GV = Mean_GV;
					m_BlobResult[i].nIn_Count = nHist;
				}

				//////////////////////////////////////////////////////////////////////////
					//哪怕只有一次动作。
				bInit = true;

				//如果满足设置的过滤,则返回true/如果不满足,则返回false
				if (!DoFiltering(
					m_BlobResult[i],//Blob结果
					nForj / 2,//比较Feature
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
				{
					bFilter = false;
					break;
				}
			}

			//如果满足所有设置的条件,请输入结果
			if (bInit && bFilter)
			{
				m_BlobResult[i].bFiltering = true;

				//转角信息
				if (bPtRotate)
				{
					int nL, nT, nR, nB;

					//旋转时,计算预测坐标
					int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
					int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

					//初始设置
					nL = nR = X;
					nT = nB = Y;

					//外围线数量
					for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
					{
						//旋转时,计算预测坐标
						X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
						Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

						//更新
						if (nL > X)	nL = X;
						if (nR < X)	nR = X;
						if (nT > Y)	nT = Y;
						if (nB < Y)	nB = Y;
					}

					cv::Point ptTemp;

					ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
					pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

					ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
					pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

					ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
					pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

					ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
					ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
					pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
					pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;

					//出现坐标0,0的现象Test
					if (pResultBlob->ptLT[nDefectCount].x == pResultBlob->ptRB[nDefectCount].x)
						pResultBlob->ptRB[nDefectCount].x += m_BlobResult[i].rectBox.width;

					if (pResultBlob->ptLT[nDefectCount].y == pResultBlob->ptRB[nDefectCount].y)
						pResultBlob->ptRB[nDefectCount].y += m_BlobResult[i].rectBox.height;
				}
				else
				{
					int nL, nT, nR, nB;
					int X = m_BlobResult[i].ptContours[0].x;
					int Y = m_BlobResult[i].ptContours[0].y;

					//初始设置
					nL = nR = X;
					nT = nB = Y;

					//外围线数量
					for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
					{
						X = m_BlobResult[i].ptContours[k].x;
						Y = m_BlobResult[i].ptContours[k].y;

						//更新
						if (nL > X)	nL = X;
						if (nR < X)	nR = X;
						if (nT > Y)	nT = Y;
						if (nB < Y)	nB = Y;
					}

					pResultBlob->ptLT[nDefectCount].x = nL;
					pResultBlob->ptLT[nDefectCount].y = nT;

					pResultBlob->ptRT[nDefectCount].x = nR;
					pResultBlob->ptRT[nDefectCount].y = nT;

					pResultBlob->ptRB[nDefectCount].x = nR;
					pResultBlob->ptRB[nDefectCount].y = nB;

					pResultBlob->ptLB[nDefectCount].x = nL;
					pResultBlob->ptLB[nDefectCount].y = nB;
				}

				//放入要交给UI的结果
				pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
				pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
				pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
				pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

				pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
				pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

				pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

				pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
				pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
				pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
				pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

				pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
				pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
				pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
				pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
				pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
				pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
				pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;
				pResultBlob->dF_MeanAreaRatio[nDefectCount] = m_BlobResult[i].fMeanAreaRatio; //choikwangil

				pResultBlob->dF_AreaPer[nDefectCount] = m_BlobResult[i].fAreaPer; //choikwangil
				pResultBlob->nJudge_GV[nDefectCount] = m_BlobResult[i].nJudge_GV; //choikwangil
				pResultBlob->nIn_Count[nDefectCount] = m_BlobResult[i].nIn_Count; //choikwangil

				pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
				pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
				pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
				//17.06.24对象直方图
				memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
				//17.11.29-外围信息(AVI&SVI其他工具)
				calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

				//绘制错误编号
				if (!matDrawBuffer.empty() && bDefectNum)
				{
					cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

					char str[256] = { 0, };
					sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
					cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
					cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
				}

				//最后的不良计数增加
				nDefectCount++;

				break;
			}
		}

		//超过最大不良数量时退出
		if (nDefectCount >= nMaxDefectCount)
			break;

	}
	writeInspectLog(__FUNCTION__, _T("Filtering & Result."));

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMem->Get_FixMemory(), m_nAlgType);
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMem->Get_AutoMemory(), m_nAlgType);
	}

	//检查时间限制
	if (cTimeOut.GetTimeOutFlag())	return E_ERROR_CODE_TIME_OUT;

	return E_ERROR_CODE_TRUE;
}

//Blob&判定结果
long CFeatureExtraction::DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, int nDefectType, int nMaxDefectCount)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	if (nDefectType < 0)							return E_ERROR_CODE_EMPTY_PARA;
	if (nDefectType >= E_DEFECT_JUDGEMENT_COUNT)	return E_ERROR_CODE_EMPTY_PARA;

	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage, nMaxDefectCount);

	int nBlockX = 1;
	int nBlockY = 1;
	// //////////////////////
		//选择的Defect列表
	int nFork = nDefectType;

	//仅当选择Defect列表时...
	if (!EngineerBlockDefectJudge->stDefectItem[nFork].bDefectItemUse)
		return E_ERROR_CODE_TRUE;

	//Blob数量
#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		//Defect列表的数量
	//for (int nFork = 0 ; nFork < E_DEFECT_JUDGEMENT_COUNT ; nFork++)		
		{
			//每个判定项目2个范围
			int nFeatureCount = E_FEATURE_COUNT * 2;
			bool bFilter = true;
			bool bInit = false;
			for (int nForj = 0; nForj < nFeatureCount; nForj++)
			{
				//仅当选择判定项目时
				if (!EngineerBlockDefectJudge->stDefectItem[nFork].Judgment[nForj].bUse)
					continue;

				//哪怕只有一次动作。
				bInit = true;

				//如果满足设置的过滤,则返回true/如果不满足,则返回false
				if (!DoFiltering(
					m_BlobResult[i],//Blob结果
					nForj / 2,//比较Feature
					EngineerBlockDefectJudge->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
					EngineerBlockDefectJudge->stDefectItem[nFork].Judgment[nForj].dValue))//值
				{
					bFilter = false;
					break;
				}
			}

			//如果满足所有设置的条件
			if (bInit && bFilter)
			{
				m_BlobResult[i].bFiltering = true;
			}
		}
	}
	return E_ERROR_CODE_TRUE;
}

//Blob&判定结果
long CFeatureExtraction::DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, int nMaxDefectCount)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;
	int nBlockX = 1;
	int nBlockY = 1;
	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage, nMaxDefectCount);
	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;


	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[m_BlobResult[i].nBlockNum];
				break;
			}
			}

		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	continue;
#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		//Defect列表的数量
		for (int nFork = 0; nFork < E_DEFECT_JUDGEMENT_COUNT; nFork++)
		{
			//仅当选择Defect列表时...
			if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
				continue;

			//每个判定项目2个范围
			int nFeatureCount = E_FEATURE_COUNT * 2;
			bool bFilter = true;
			bool bInit = false;
			for (int nForj = 0; nForj < nFeatureCount; nForj++)
			{
				//仅当选择判定项目时
				if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
					continue;

				//哪怕只有一次动作。
				bInit = true;

				//如果满足设置的过滤,则返回true/如果不满足,则返回false
				if (!DoFiltering(
					m_BlobResult[i],//Blob结果
					nForj / 2,//比较Feature
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
				{
					bFilter = false;
					break;
				}
			}

			//如果满足所有设置的条件
			if (bInit && bFilter)
			{
				m_BlobResult[i].bFiltering = true;

				break;
			}
		}

		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	return E_ERROR_CODE_TIME_OUT;
		}


	return E_ERROR_CODE_TRUE;
	}

////////sssy0718检查是否位于外围
//Blob&判定结果
long CFeatureExtraction::DoDefectBlobSingleJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, int nDefectType, bool bPtRotate, CRect prerectROI, int offset)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)				return E_ERROR_CODE_EMPTY_PARA;

	if (nDefectType < 0)							return E_ERROR_CODE_EMPTY_PARA;
	if (nDefectType >= E_DEFECT_JUDGEMENT_COUNT)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())						return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())					return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;
	////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// 
	// /////////////////////////////////

		//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage, nMaxDefectCount);

	//选择的Defect列表
	int nFork = nDefectType;
	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[m_BlobResult[i].nBlockNum];
				break;
			}

		}

		//仅当选择Defect列表时...
		if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
			return E_ERROR_CODE_TRUE;

		//过滤不良颜色
		if (!DoColorFilter(nFork, nDefectColor))
			continue;

		//每个判定项目2个范围
		int nFeatureCount = E_FEATURE_COUNT * 2;
		bool bFilter = true;
		bool bInit = false;
		for (int nForj = 0; nForj < nFeatureCount; nForj++)
		{
			//仅当选择判定项目时
			if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
				continue;

			//哪怕只有一次动作。
			bInit = true;
			if (nForj >= E_FEATURE_IS_EDGE_C * 2)
			{
				//m_BlobResult[i].
				if (is_edge(m_BlobResult[i], nForj / 2, prerectROI, offset))
				{
					bFilter = true;
				}
				else
				{
					bFilter = false;
					break;
				}
			}
			else
			{
				//如果满足设置的过滤,则返回true/如果不满足,则返回false
				if (!DoFiltering(
					m_BlobResult[i],//Blob结果
					nForj / 2,//比较Feature
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
					blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
				{
					bFilter = false;
					break;
				}
			}
		}

		//如果满足所有设置的条件,请输入结果
		if (bInit && bFilter)
		{
			m_BlobResult[i].bFiltering = true;

			//转角信息
			if (bPtRotate)
			{
				int nL, nT, nR, nB;

				//旋转时,计算预测坐标
				int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
				int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

				//初始设置
				nL = nR = X;
				nT = nB = Y;

				//外围线数量
				for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
				{
					//旋转时,计算预测坐标
					X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
					Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

					//更新
					if (nL > X)	nL = X;
					if (nR < X)	nR = X;
					if (nT > Y)	nT = Y;
					if (nB < Y)	nB = Y;
				}

				cv::Point ptTemp;

				ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
				pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
				pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
				pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

				ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
				ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
				pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
				pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;
			}
			else
			{
				int nL, nT, nR, nB;
				int X = m_BlobResult[i].ptContours[0].x;
				int Y = m_BlobResult[i].ptContours[0].y;

				//初始设置
				nL = nR = X;
				nT = nB = Y;

				//外围线数量
				for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
				{
					X = m_BlobResult[i].ptContours[k].x;
					Y = m_BlobResult[i].ptContours[k].y;

					//更新
					if (nL > X)	nL = X;
					if (nR < X)	nR = X;
					if (nT > Y)	nT = Y;
					if (nB < Y)	nB = Y;
				}

				pResultBlob->ptLT[nDefectCount].x = nL;
				pResultBlob->ptLT[nDefectCount].y = nT;

				pResultBlob->ptRT[nDefectCount].x = nR;
				pResultBlob->ptRT[nDefectCount].y = nT;

				pResultBlob->ptRB[nDefectCount].x = nR;
				pResultBlob->ptRB[nDefectCount].y = nB;

				pResultBlob->ptLB[nDefectCount].x = nL;
				pResultBlob->ptLB[nDefectCount].y = nB;
			}

			//放入要交给UI的结果
			pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
			pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
			pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
			pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

			pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
			pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

			pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

			pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
			pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
			pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

			pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
			pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
			pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
			pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
			pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
			pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;

			pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
			pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
			pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
			//17.06.24对象直方图
			memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
			//17.11.29-外围信息(AVI&SVI其他工具)
			calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

			//绘制错误编号
			if (!matDrawBuffer.empty() && bDefectNum)
			{
				cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

				char str[256] = { 0, };
				sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
				cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
				cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
			}

			//最后的不良计数增加
			nDefectCount++;
		}

		//超过最大不良数量时退出
		if (nDefectCount >= nMaxDefectCount)
			break;

	}
	return E_ERROR_CODE_TRUE;
}

/////////SSY1124只将选定的不良品转移到"烧毁检测"基地
//Blob&判定结果
long CFeatureExtraction::DoDefectBlobMultiJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, vector<int> nDefectType, bool bPtRotate, CRect prerectROI, int offset)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)				return E_ERROR_CODE_EMPTY_PARA;

	if (nDefectType.size() == 0)					return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())						return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())					return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;
	////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// 
	// /////////////////////////////////

		//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage, nMaxDefectCount);

	//选择的Defect列表
	vector<int> nForks = nDefectType;
	int nFork = 0;
	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[m_BlobResult[i].nBlockNum];
				break;
			}


			for (int j = 0; j < nForks.size(); j++)
			{
				nFork = nForks[j];

				//过滤不良颜色
				if (!DoColorFilter(nFork, nDefectColor))
					continue;

				//仅当选择Defect列表时...
				if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
					continue;

				//每个判定项目2个范围
				int nFeatureCount = E_FEATURE_COUNT * 2;
				bool bFilter = true;
				bool bInit = false;
				for (int nForj = 0; nForj < nFeatureCount; nForj++)
				{
					//仅当选择判定项目时
					if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
						continue;

					//哪怕只有一次动作。
					bInit = true;
					if (nForj >= E_FEATURE_IS_EDGE_C * 2)
					{
						if (is_edge(m_BlobResult[i], nForj / 2, prerectROI, offset))
						{
							bFilter = true;
						}
						else
						{
							bFilter = false;
							break;
						}
					}
					else
					{
						//如果满足设置的过滤,则返回true/如果不满足,则返回false
						if (!DoFiltering(
							m_BlobResult[i],//Blob结果
							nForj / 2,//比较Feature
							blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
							blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
						{
							bFilter = false;
							break;
						}
					}
				}

				//如果满足所有设置的条件,请输入结果
				if (bInit && bFilter)
				{
					m_BlobResult[i].bFiltering = true;

					//转角信息
					if (bPtRotate)
					{
						int nL, nT, nR, nB;

						//旋转时,计算预测坐标
						int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
						int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

						//初始设置
						nL = nR = X;
						nT = nB = Y;

						//外围线数量
						for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
						{
							//旋转时,计算预测坐标
							X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
							Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

							//更新
							if (nL > X)	nL = X;
							if (nR < X)	nR = X;
							if (nT > Y)	nT = Y;
							if (nB < Y)	nB = Y;
						}

						cv::Point ptTemp;

						ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
						pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

						ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
						pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

						ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
						pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

						ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
						pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;
					}
					else
					{
						int nL, nT, nR, nB;
						int X = m_BlobResult[i].ptContours[0].x;
						int Y = m_BlobResult[i].ptContours[0].y;

						//初始设置
						nL = nR = X;
						nT = nB = Y;

						//外围线数量
						for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
						{
							X = m_BlobResult[i].ptContours[k].x;
							Y = m_BlobResult[i].ptContours[k].y;

							//更新
							if (nL > X)	nL = X;
							if (nR < X)	nR = X;
							if (nT > Y)	nT = Y;
							if (nB < Y)	nB = Y;
						}

						pResultBlob->ptLT[nDefectCount].x = nL;
						pResultBlob->ptLT[nDefectCount].y = nT;

						pResultBlob->ptRT[nDefectCount].x = nR;
						pResultBlob->ptRT[nDefectCount].y = nT;

						pResultBlob->ptRB[nDefectCount].x = nR;
						pResultBlob->ptRB[nDefectCount].y = nB;

						pResultBlob->ptLB[nDefectCount].x = nL;
						pResultBlob->ptLB[nDefectCount].y = nB;
					}

					//放入要交给UI的结果
					pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
					pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
					pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
					pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

					pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
					pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

					pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

					pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
					pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
					pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
					pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

					pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
					pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
					pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
					pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
					pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
					pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
					pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;

					pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
					pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
					pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
					//17.06.24对象直方图
					memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
					//17.11.29-外围信息(AVI&SVI其他工具)
					calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

					//绘制错误编号
					if (!matDrawBuffer.empty() && bDefectNum)
					{
						cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

						char str[256] = { 0, };
						sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
						cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
						cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
					}

					//最后的不良计数增加
					nDefectCount++;
				}

				//超过最大不良数量时退出
				if (nDefectCount >= nMaxDefectCount)
					break;
			}
		}
	}
	return E_ERROR_CODE_TRUE;
}

//Blob&判定结果
long CFeatureExtraction::DoDefectBlobJudgment(cv::Mat& matSrcImage, cv::Mat& matThresholdImage, cv::Mat& matDrawBuffer,
	int* nCommonPara, long nDefectColor, CString strTxt, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, bool bPtRotate, CRect prerectROI, int offset)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
	Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)			return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())			return E_ERROR_CODE_EMPTY_BUFFER;
	if (matThresholdImage.empty())		return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);
	double	dSin_ = sin(-dTheta);
	double	dCos_ = cos(-dTheta);

	//标签开始
	DoBlobCalculate(matThresholdImage, matSrcImage);

	//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;
	////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// 
	// /////////////////////////////////
	//////////////////////
	///   划分m_BlobResult缺陷特征，并更新字段nBlockNum（分区块编号）hjf
	divideBlobResult(matSrcImage.cols, matSrcImage.rows, nBlockX, nBlockY);
	stPanelBlockJudgeInfo* blockJudgeInfo;
	for (int i = 0; i < m_BlobResult.size(); i++)
	{
		for (int nBlockNum = 0; nBlockNum < nBlockX * nBlockY; nBlockNum++)
		{
			if (m_BlobResult[i].nBlockNum == nBlockNum)
			{
				blockJudgeInfo = &EngineerBlockDefectJudge[m_BlobResult[i].nBlockNum];
				break;
			}

			//检查时间限制
			if (cTimeOut.GetTimeOutFlag())	continue;

			//Defect列表的数量
			for (int nFork = 0; nFork < E_DEFECT_JUDGEMENT_COUNT; nFork++)
			{
				//仅当选择Defect列表时...
				if (!blockJudgeInfo->stDefectItem[nFork].bDefectItemUse)
					continue;

				//过滤不良颜色
				if (!DoColorFilter(nFork, nDefectColor))
					continue;

				//每个判定项目2个范围
				int nFeatureCount = E_FEATURE_COUNT * 2;
				bool bFilter = true;
				bool bInit = false;
				for (int nForj = 0; nForj < nFeatureCount; nForj++)
				{
					//仅当选择判定项目时

					if (!blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].bUse)
						continue;

					//哪怕只有一次动作。
					bInit = true;
					if (nForj >= E_FEATURE_IS_EDGE_C * 2)
					{
						//m_BlobResult[i].
						if (is_edge(m_BlobResult[i], nForj / 2, prerectROI, offset))
						{
							bFilter = true;
						}
						else
						{
							bFilter = false;
							break;
						}
					}
					else
					{
						//如果满足设置的过滤,则返回true/如果不满足,则返回false
						if (!DoFiltering(
							m_BlobResult[i],//Blob结果
							nForj / 2,//比较Feature
							blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].nSign,//运算符(<,>,==,<=,>=)
							blockJudgeInfo->stDefectItem[nFork].Judgment[nForj].dValue))//值
						{
							bFilter = false;
							break;
						}
					}

				}

				//如果满足所有设置的条件,请输入结果
				if (bInit && bFilter)
				{
					m_BlobResult[i].bFiltering = true;

					//转角信息
					if (bPtRotate)
					{
						int nL, nT, nR, nB;

						//旋转时,计算预测坐标
						int X = (int)(dCos * (m_BlobResult[i].ptContours[0].x - nCx) - dSin * (m_BlobResult[i].ptContours[0].y - nCy) + nCx);
						int Y = (int)(dSin * (m_BlobResult[i].ptContours[0].x - nCx) + dCos * (m_BlobResult[i].ptContours[0].y - nCy) + nCy);

						//初始设置
						nL = nR = X;
						nT = nB = Y;

						//外围线数量
						for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
						{
							//旋转时,计算预测坐标
							X = (int)(dCos * (m_BlobResult[i].ptContours[k].x - nCx) - dSin * (m_BlobResult[i].ptContours[k].y - nCy) + nCx);
							Y = (int)(dSin * (m_BlobResult[i].ptContours[k].x - nCx) + dCos * (m_BlobResult[i].ptContours[k].y - nCy) + nCy);

							//更新
							if (nL > X)	nL = X;
							if (nR < X)	nR = X;
							if (nT > Y)	nT = Y;
							if (nB < Y)	nB = Y;
						}

						cv::Point ptTemp;

						ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nT - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nT - nCy) + nCy);
						pResultBlob->ptLT[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptLT[nDefectCount].y = (LONG)ptTemp.y;

						ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nT - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nT - nCy) + nCy);
						pResultBlob->ptRT[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptRT[nDefectCount].y = (LONG)ptTemp.y;

						ptTemp.x = (int)(dCos_ * (nR - nCx) - dSin_ * (nB - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nR - nCx) + dCos_ * (nB - nCy) + nCy);
						pResultBlob->ptRB[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptRB[nDefectCount].y = (LONG)ptTemp.y;

						ptTemp.x = (int)(dCos_ * (nL - nCx) - dSin_ * (nB - nCy) + nCx);
						ptTemp.y = (int)(dSin_ * (nL - nCx) + dCos_ * (nB - nCy) + nCy);
						pResultBlob->ptLB[nDefectCount].x = (LONG)ptTemp.x;
						pResultBlob->ptLB[nDefectCount].y = (LONG)ptTemp.y;
					}
					else
					{
						int nL, nT, nR, nB;
						int X = m_BlobResult[i].ptContours[0].x;
						int Y = m_BlobResult[i].ptContours[0].y;

						//初始设置
						nL = nR = X;
						nT = nB = Y;

						//外围线数量
						for (int k = 1; k < m_BlobResult[i].ptContours.size(); k++)
						{
							X = m_BlobResult[i].ptContours[k].x;
							Y = m_BlobResult[i].ptContours[k].y;

							//更新
							if (nL > X)	nL = X;
							if (nR < X)	nR = X;
							if (nT > Y)	nT = Y;
							if (nB < Y)	nB = Y;
						}

						pResultBlob->ptLT[nDefectCount].x = nL;
						pResultBlob->ptLT[nDefectCount].y = nT;

						pResultBlob->ptRT[nDefectCount].x = nR;
						pResultBlob->ptRT[nDefectCount].y = nT;

						pResultBlob->ptRB[nDefectCount].x = nR;
						pResultBlob->ptRB[nDefectCount].y = nB;

						pResultBlob->ptLB[nDefectCount].x = nL;
						pResultBlob->ptLB[nDefectCount].y = nB;
					}

					//放入要交给UI的结果
					pResultBlob->nArea[nDefectCount] = m_BlobResult[i].nArea;
					pResultBlob->nMaxGV[nDefectCount] = m_BlobResult[i].nMaxGV;
					pResultBlob->nMinGV[nDefectCount] = m_BlobResult[i].nMinGV;
					pResultBlob->dMeanGV[nDefectCount] = m_BlobResult[i].fMeanGV;

					pResultBlob->nCenterx[nDefectCount] = m_BlobResult[i].ptCenter.x;
					pResultBlob->nCentery[nDefectCount] = m_BlobResult[i].ptCenter.y;

					pResultBlob->dBackGroundGV[nDefectCount] = m_BlobResult[i].fBKGV;

					pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
					pResultBlob->dSigma[nDefectCount] = m_BlobResult[i].fStdDev;
					pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
					pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;

					pResultBlob->dBreadth[nDefectCount] = m_BlobResult[i].fMajorAxis;
					pResultBlob->dF_Min[nDefectCount] = m_BlobResult[i].fMajorAxis;
					pResultBlob->dF_Max[nDefectCount] = m_BlobResult[i].fMinorAxis;
					pResultBlob->dF_Elongation[nDefectCount] = m_BlobResult[i].fAxisRatio;
					pResultBlob->dCompactness[nDefectCount] = m_BlobResult[i].fCompactness;
					pResultBlob->dRoundness[nDefectCount] = m_BlobResult[i].fRoundness;
					pResultBlob->nBlockNum[nDefectCount] = m_BlobResult[i].nBlockNum;

					pResultBlob->nDefectColor[nDefectCount] = nDefectColor;
					pResultBlob->nDefectJudge[nDefectCount] = nFork;//相关不良
					pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;

#if USE_ALG_HIST
					//17.06.24对象直方图
					memcpy(pResultBlob->nHist[nDefectCount], m_BlobResult[i].nHist, sizeof(__int64) * IMAGE_MAX_GV);
#endif

#if USE_ALG_CONTOURS
					//17.11.29-外围信息(AVI&SVI其他工具)
					calcContours(pResultBlob->nContoursX[nDefectCount], pResultBlob->nContoursY[nDefectCount], i, fAngle, nCx, nCy, nPS);
#endif

					//绘制错误编号
					if (!matDrawBuffer.empty() && bDefectNum)
					{
						cv::rectangle(matDrawBuffer, cv::Rect(pResultBlob->ptRT[nDefectCount].x - 2, pResultBlob->ptRT[nDefectCount].y - 10, 30, 12), cv::Scalar(0, 0, 0), -1);

						char str[256] = { 0, };
						sprintf_s(str, sizeof(str), "%s%d", LPSTR(LPCTSTR(strTxt)), nDefectCount);
						cv::Point ptRT(pResultBlob->ptRT[nDefectCount].x, pResultBlob->ptRT[nDefectCount].y);
						cv::putText(matDrawBuffer, str, ptRT, cv::FONT_HERSHEY_SIMPLEX, 0.4f, cv::Scalar(255, 0, 0));
					}

					//最后的不良计数增加
					nDefectCount++;

					break;
				}
			}

			//超过最大不良数量时退出
			if (nDefectCount >= nMaxDefectCount)
				break;
		}

		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	return E_ERROR_CODE_TIME_OUT;
	}
	return E_ERROR_CODE_TRUE;
}

//检查ssy0718是否位于外围
bool CFeatureExtraction::is_edge(tBLOB_FEATURE& tBlobResult, int nBlobFilter, CRect prerectROI, int offset)
{
	//如果已过滤,则排除
	if (tBlobResult.bFiltering)	return false;
	//int offset = 5;
	bool bRes = false;
	int left = tBlobResult.rectBox.x;
	int top = tBlobResult.rectBox.y;
	int width = tBlobResult.rectBox.width;
	int height = tBlobResult.rectBox.height;

	int ori_left = prerectROI.left;
	int ori_top = prerectROI.top;
	int ori_width = prerectROI.Width();
	int ori_height = prerectROI.Height();

	bool is_c = false;
	bool is_v = false;
	bool is_h = false;
	bool is_cen = false;

	if ((left <= ori_left + offset && top <= ori_top + offset) || ((left + width) >= (ori_left + ori_width) - offset && top <= ori_top + offset) || (left <= ori_left + offset && (top + height) >= (ori_top + ori_height) - offset) || ((left + width) >= (ori_left + ori_width) - offset && (top + height) >= (ori_top + ori_height) - offset))
	{
		is_c = true;
	}
	else if (top <= ori_top + offset || (top + height) >= (ori_top + ori_height) - offset)
	{
		is_v = true;
	}
	else if (left <= ori_left + offset || (left + width) >= (ori_left + ori_width) - offset)
	{
		is_h = true;
	}
	else
	{
		is_cen = true;
	}
	switch (nBlobFilter)
	{
	case E_FEATURE_IS_EDGE_C:
		if (is_c) bRes = true;
		break;

	case E_FEATURE_IS_EDGE_V:
		if (is_v) bRes = true;
		break;

	case E_FEATURE_IS_EDGE_H:
		if (is_h) bRes = true;
		break;

	case E_FEATURE_IS_EDGE_CENTER:
		if (is_cen) bRes = true;
		break;
	}
	return bRes;
}

bool CFeatureExtraction::DrawBlob(cv::Mat& DrawBuffer, CvScalar DrawColor, long nOption, bool bSelect, float fFontSize)
{
	//没有画面时返回
	if (DrawBuffer.empty())		return false;

	//如果没有运行Blob,则返回
	if (!m_bComplete)				return false;

	//Blob返回0个结果
	if (m_BlobResult.size() == 0)	return true;

	//如果没有选项,则返回
	if (nOption == 0)				return true;

	int i, j;

	// Fix OMP_CRASH 20230411.xb

	for (i = 0; i < m_BlobResult.size(); i++)
	{
		//只绘制所选内容
		if (!m_BlobResult[i].bFiltering && bSelect)	continue;

		//绘制旋转框
		if (nOption & BLOB_DRAW_ROTATED_BOX)
		{
			cv::RotatedRect rRect = cv::RotatedRect(m_BlobResult[i].ptCenter, m_BlobResult[i].BoxSize, m_BlobResult[i].fAngle);

			cv::Point2f vertices[4];
			rRect.points(vertices);

			cv::line(DrawBuffer, vertices[0], vertices[1], DrawColor);
			cv::line(DrawBuffer, vertices[1], vertices[2], DrawColor);
			cv::line(DrawBuffer, vertices[2], vertices[3], DrawColor);
			cv::line(DrawBuffer, vertices[3], vertices[0], DrawColor);
		}

		//绘制外框
		if (nOption & BLOB_DRAW_BOUNDING_BOX)
		{
			cv::Rect rect(m_BlobResult[i].rectBox);
			rect.x -= 5;
			rect.y -= 5;
			rect.width += 10;
			rect.height += 10;

			cv::rectangle(DrawBuffer, rect, DrawColor);
		}

		//绘制Blob对象
		if (nOption & BLOB_DRAW_BLOBS)
		{
			//如果是Gray
			if (DrawBuffer.channels() == 1)
			{
				int nGrayColor = (int)(DrawColor.val[0] + DrawColor.val[1] + DrawColor.val[2]) / 3;

				for (j = 0; j < m_BlobResult[i].ptIndexs.size(); j++)
				{
					DrawBuffer.at<uchar>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x) = nGrayColor;
				}
			}
			//如果是RGB
			else
			{
				for (j = 0; j < m_BlobResult[i].ptIndexs.size(); j++)
				{
					DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x)[0] = (int)DrawColor.val[0];
					DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x)[1] = (int)DrawColor.val[1];
					DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptIndexs[j].y, m_BlobResult[i].ptIndexs[j].x)[2] = (int)DrawColor.val[2];
				}
			}
		}

		//绘制Blob轮廓
		if (nOption & BLOB_DRAW_BLOBS_CONTOUR)
		{
			//如果是Gray
			if (DrawBuffer.channels() == 1)
			{
				int nGrayColor = (int)(DrawColor.val[0] + DrawColor.val[1] + DrawColor.val[2]) / 3;

				for (j = 0; j < m_BlobResult[i].ptContours.size(); j++)
				{
					DrawBuffer.at<uchar>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x) = nGrayColor;
				}
			}
			//如果是RGB
			else
			{
				for (j = 0; j < m_BlobResult[i].ptContours.size(); j++)
				{
					DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x)[0] = (int)DrawColor.val[0];
					DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x)[1] = (int)DrawColor.val[1];
					DrawBuffer.at<cv::Vec3b>(m_BlobResult[i].ptContours[j].y, m_BlobResult[i].ptContours[j].x)[2] = (int)DrawColor.val[2];
				}
			}
		}
	}

	return true;
}

//获取分区格位置 hjf
int CFeatureExtraction::GetGridNumber(int imageWidth, int imageHeight, int X, int Y, int center_x, int center_y) {
	int gridWidth = imageWidth / X;
	int gridHeight = imageHeight / Y;

	int gridColumn = center_x / (gridWidth + 1);
	int gridRow = center_y / (gridHeight + 1);

	if (gridColumn >= X) gridColumn = X - 1;
	if (gridRow >= Y) gridRow = Y - 1;
	int gridNumber = gridRow * X + gridColumn;

	return gridNumber;
}



void CFeatureExtraction::divideBlobResult(int imageWidth, int imageHeight, int X, int Y) {

	for (tBLOB_FEATURE& BlobSingle : m_BlobResult) {
		int gridNum = GetGridNumber(imageWidth, imageHeight, X, Y, (BlobSingle.rectBox.x + BlobSingle.rectBox.width / 2), (BlobSingle.rectBox.y + BlobSingle.rectBox.height / 2));
		BlobSingle.nBlockNum = gridNum;
	}

	return;
}

bool CFeatureExtraction::DoColorFilter(int nDefectName, int nDefectColor)
{
	//异常处理
	if (nDefectName < 0)								return false;
	if (nDefectName >= E_DEFECT_JUDGEMENT_COUNT)		return false;

	int nColor = -1;

	switch (nDefectName)
	{
	case E_DEFECT_JUDGEMENT_POINT_DARK://暗点
	case E_DEFECT_JUDGEMENT_POINT_DARK_SP_1:
	case E_DEFECT_JUDGEMENT_POINT_DARK_SP_2:
	case E_DEFECT_JUDGEMENT_POINT_DARK_SP_3:
	case E_DEFECT_JUDGEMENT_POINT_GROUP_DARK://行李暗点
	case E_DEFECT_JUDGEMENT_LINE_X_DARK://暗X线
	case E_DEFECT_JUDGEMENT_LINE_X_DARK_MULT://暗X线
	case E_DEFECT_JUDGEMENT_LINE_Y_DARK://暗Y线
	case E_DEFECT_JUDGEMENT_LINE_Y_DARK_MULT://暗Y线
	case E_DEFECT_JUDGEMENT_MURA_AMORPH_DARK://暗无定形
	case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_DARK://暗弱视视星线
	case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_DARK://暗弱视视星线
	case E_DEFECT_JUDGEMENT_RETEST_POINT_DARK://复查暗点
	case E_DEFECT_JUDGEMENT_RETEST_LINE_DARK://重新扫描暗X线
	case E_DEFECT_JUDGEMENT_MURA_BLACK_SPOT:
	case E_DEFECT_JUDGEMENT_MURA_EMD_DARK:
	case E_DEFECT_JUDGEMENT_MURA_NUGI:
	case E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI:
	case E_DEFECT_JUDGEMENT_MURA_EDGE_NUGI_:
		nColor = E_DEFECT_COLOR_DARK;
		break;

	case E_DEFECT_JUDGEMENT_POINT_BRIGHT://亮点
	case E_DEFECT_JUDGEMENT_POINT_WEAK_BRIGHT://简明点
	case E_DEFECT_JUDGEMENT_POINT_GROUP_BRIGHT://群集名点
	case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT://亮X线
	case E_DEFECT_JUDGEMENT_LINE_X_BRIGHT_MULT://亮X线
	case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT://亮Y线
	case E_DEFECT_JUDGEMENT_LINE_Y_BRIGHT_MULT://亮Y线
	case E_DEFECT_JUDGEMENT_MURA_AMORPH_BRIGHT://明亮的无定形
	case E_DEFECT_JUDGEMENT_LINE_X_EDGE_BRIGHT:
	case E_DEFECT_JUDGEMENT_LINE_Y_EDGE_BRIGHT:
	case E_DEFECT_JUDGEMENT_LINE_X_DEFECT_BRIGHT:
	case E_DEFECT_JUDGEMENT_LINE_Y_DEFECT_BRIGHT://亮弱视视星线
	case E_DEFECT_JUDGEMENT_RETEST_POINT_BRIGHT://复查明细
	case E_DEFECT_JUDGEMENT_RETEST_LINE_BRIGHT://重新扫描亮X线
	case E_DEFECT_JUDGEMENT_MURA_MULT_BP:			// 17.09.27 - MURA_MULT_BP
	case E_DEFECT_JUDGEMENT_LINE_PCD_CRACK://PNZ 17.12.01-新不良
	case E_DEFECT_JUDGEMENT_MURA_WHITE_SPOT:
	case E_DEFECT_JUDGEMENT_MURA_E_WHITE_SPOT:      // 04.16 choi
	case E_DEFECT_JUDGEMENT_MURA_EMD_BRIGHT:
	case E_DEFECT_JUDGEMENT_MURA_BOX_SCRATCH:
		nColor = E_DEFECT_COLOR_BRIGHT;
		break;

	default:
		nColor = -1;
		break;
	}

	//如果指定了不良颜色
	if (nColor != -1)
	{
		//与设置的颜色相同吗？
		if (nColor == nDefectColor)
			return true;

		//不是和设置的颜色一样吗？
		else
			return false;
	}

	//不确定不良颜色的情况下。
	return true;
}

bool CFeatureExtraction::Compare(double dFeatureValue, int nSign, double dValue)
{
	bool bRes = false;

	//运算符(<,>,==,<=,>=)
	switch (nSign)
	{
	case	E_SIGN_EQUAL:				// x == judgment value
		bRes = (dFeatureValue == dValue) ? true : false;
		break;

	case	E_SIGN_NOT_EQUAL:			// x != judgment value
		bRes = (dFeatureValue != dValue) ? true : false;
		break;

	case	E_SIGN_GREATER:				// x >  judgment value
		bRes = (dFeatureValue > dValue) ? true : false;
		break;

	case	E_SIGN_LESS:				// x <  judgment value
		bRes = (dFeatureValue < dValue) ? true : false;
		break;

	case	E_SIGN_GREATER_OR_EQUAL:	// x >= judgment value
		bRes = (dFeatureValue >= dValue) ? true : false;
		break;

	case	E_SIGN_LESS_OR_EQUAL:		// x <= judgment value
		bRes = (dFeatureValue <= dValue) ? true : false;
		break;
	}

	return bRes;
}

bool CFeatureExtraction::DoFiltering(tBLOB_FEATURE& tBlobResult, int nBlobFilter, int nSign, double dValue)
{
	//如果已过滤,则排除
	if (tBlobResult.bFiltering)	return false;

	bool bRes = false;

	switch (nBlobFilter)
	{
	case E_FEATURE_AREA:
		bRes = Compare((double)tBlobResult.nArea, nSign, dValue);
		break;

	case E_FEATURE_BOX_AREA:
		bRes = Compare((double)tBlobResult.nBoxArea, nSign, dValue);
		break;

	case E_FEATURE_BOX_RATIO:
		bRes = Compare((double)tBlobResult.fBoxRatio, nSign, dValue);
		break;

	case E_FEATURE_BOX_X:
		bRes = Compare((double)tBlobResult.rectBox.width, nSign, dValue);
		break;

	case E_FEATURE_BOX_Y:
		bRes = Compare((double)tBlobResult.rectBox.height, nSign, dValue);
		break;

	case E_FEATURE_SUM_GV:
		bRes = Compare((double)tBlobResult.nSumGV, nSign, dValue);
		break;

	case E_FEATURE_MIN_GV:
		bRes = Compare((double)tBlobResult.nMinGV, nSign, dValue);
		break;

	case E_FEATURE_MAX_GV:
		bRes = Compare((double)tBlobResult.nMaxGV, nSign, dValue);
		break;

	case E_FEATURE_MEAN_GV:
		bRes = Compare((double)tBlobResult.fMeanGV, nSign, dValue);
		break;

	case E_FEATURE_DIFF_GV:
		bRes = Compare((double)tBlobResult.fDiffGV, nSign, dValue);
		break;

	case E_FEATURE_BK_GV:
		bRes = Compare((double)tBlobResult.fBKGV, nSign, dValue);
		break;

	case E_FEATURE_STD_DEV:
		bRes = Compare((double)tBlobResult.fStdDev, nSign, dValue);
		break;

	case E_FEATURE_SEMU:
		bRes = Compare((double)tBlobResult.fSEMU, nSign, dValue);
		break;

	case E_FEATURE_COMPACTNESS:
		bRes = Compare((double)tBlobResult.fCompactness, nSign, dValue);
		break;

	case E_FEATURE_MIN_GV_RATIO:
		bRes = Compare((double)tBlobResult.nMinGVRatio, nSign, dValue);
		break;

	case E_FEATURE_MAX_GV_RATIO:
		bRes = Compare((double)tBlobResult.nMaxGVRatio, nSign, dValue);
		break;

	case E_FEATURE_DIFF_GV_RATIO:
		bRes = Compare((double)tBlobResult.fDiffGVRatio, nSign, dValue);
		break;

	case E_FEATURE_PERIMETER:
		bRes = Compare((double)tBlobResult.fPerimeter, nSign, dValue);
		break;

	case E_FEATURE_ROUNDNESS:
		bRes = Compare((double)tBlobResult.fRoundness, nSign, dValue);
		break;

	case E_FEATURE_ELONGATION:
		bRes = Compare((double)tBlobResult.fElongation, nSign, dValue);
		break;

	case E_FEATURE_MIN_BOX_AREA:
		bRes = Compare((double)tBlobResult.fMinBoxArea, nSign, dValue);
		break;

	case E_FEATURE_MINOR_AXIS:
		bRes = Compare((double)tBlobResult.fMinorAxis, nSign, dValue);
		break;

	case E_FEATURE_MAJOR_AXIS:
		bRes = Compare((double)tBlobResult.fMajorAxis, nSign, dValue);
		break;

	case E_FEATURE_AXIS_RATIO:
		bRes = Compare((double)tBlobResult.fAxisRatio, nSign, dValue);
		break;

	case E_FEATURE_MIN_BOX_RATIO:
		bRes = Compare((double)tBlobResult.fMinBoxRatio, nSign, dValue);
		break;

	case E_FEATURE_GV_UP_COUNT_0:
	case E_FEATURE_GV_UP_COUNT_1:
	case E_FEATURE_GV_UP_COUNT_2:
	{
		int nCount = (int)dValue / 10000;
		int nGV = (int)dValue % 10000;

		if (nGV < 0)				nGV = 0;
		if (nGV > IMAGE_MAX_GV)	nGV = IMAGE_MAX_GV - 1;

		__int64 nHist = 0;
		for (int m = nGV; m < IMAGE_MAX_GV; m++)
			nHist += tBlobResult.nHist[m];

		bRes = Compare((double)nHist, nSign, (double)nCount);
	}
	break;

	case E_FEATURE_GV_DOWN_COUNT_0:
	case E_FEATURE_GV_DOWN_COUNT_1:
	case E_FEATURE_GV_DOWN_COUNT_2:
	{
		int nCount = (int)dValue / 10000;
		int nGV = (int)dValue % 10000;

		if (nGV < 0)				nGV = 0;
		if (nGV > IMAGE_MAX_GV)	nGV = IMAGE_MAX_GV - 1;

		__int64 nHist = 0;
		for (int m = 0; m <= nGV; m++)
			nHist += tBlobResult.nHist[m];

		bRes = Compare((double)nHist, nSign, (double)nCount);
	}
	break;

	case E_FEATURE_MEANAREA_RATIO: //choikwangil
		bRes = Compare((double)tBlobResult.fMeanAreaRatio, nSign, dValue);
		break;

	case E_FEATURE_GVAREA_RATIO_TEST: //04.20 choi
	{

		int nTmp = (int)dValue % 10000;
		double nPer = ((double)dValue - (double)nTmp) / 10000.0;
		double nRatio = nTmp / 1000;

		double Mean_GV = tBlobResult.fBKGV * nRatio;

		if (Mean_GV < 0)				Mean_GV = 0;
		if (Mean_GV > IMAGE_MAX_GV)  	Mean_GV = IMAGE_MAX_GV - 1;

		__int64 nHist = 0;
		for (int m = Mean_GV; m <= 255; m++)
			nHist += tBlobResult.nHist[m];

		double Area_per = nHist / tBlobResult.nBoxArea;
		Area_per *= 100;

		bRes = Compare((double)Area_per, nSign, (double)nPer);
	}
	break;

	default:
		bRes = false;
		break;
	}

	return bRes;
}

//移交轮廓坐标
//在P/S模式下:折叠以保存坐标
//Align时通过画面预测保存坐标
bool CFeatureExtraction::SaveTxt(int* nCommonPara, wchar_t* strContourTxt, bool bUse)
{
	//移动轮廓信息->AviInspection::JudgeSaveMuraContours()
	//删除不良时,不要表现出来...
	if (!bUse)		return	true;

	//如果参数为NULL。
	if (nCommonPara == NULL)			return false;

	//如果没有运行Blob,则返回
	if (!m_bComplete)					return false;

	//Blob返回0个结果
	if (m_BlobResult.size() == 0)		return true;

	//如果没有保存轮廓的路径
	if (strContourTxt == NULL)			return false;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);

	EnterCriticalSection(&m_csCoordFile);

	//保存TXT
	CStdioFile	fileWriter;
	CString		strLine;

	//打开文件
	if (fileWriter.Open(strContourTxt, CFile::modeCreate | CFile::modeNoTruncate | CFile::modeWrite))
	{
		for (int i = 0; i < m_BlobResult.size(); i++)
		{
			//只选择...(只有实际被判定为不良的......)
			if (!m_BlobResult[i].bFiltering)	continue;

			//保存轮廓
			for (int j = 0; j < m_BlobResult[i].ptContours.size(); j++)
			{
				//旋转时,计算预测坐标
				int X = (int)(dCos * (m_BlobResult[i].ptContours[j].x - nCx) - dSin * (m_BlobResult[i].ptContours[j].y - nCy) + nCx);
				int Y = (int)(dSin * (m_BlobResult[i].ptContours[j].x - nCx) + dCos * (m_BlobResult[i].ptContours[j].y - nCy) + nCy);

				//根据P/S模式修改坐标(更改为单杆坐标)
				strLine.Format(_T("%d, %d\n"), (int)X / nPS, (int)Y / nPS);

				fileWriter.SeekToEnd();
				fileWriter.WriteString(strLine);
			}
		}

		//仅在文件打开时关闭
		fileWriter.Close();
	}

	LeaveCriticalSection(&m_csCoordFile);

	return true;
}

bool CFeatureExtraction::GetResultblob(vector<tBLOB_FEATURE>& OutBlob)
{
	//矢量初始化
	if (OutBlob.size() != 0)
	{
		for (int i = 0; i < OutBlob.size(); i++)
		{
			vector<cv::Point>().swap(OutBlob[i].ptIndexs);
			vector <cv::Point>().swap(m_BlobResult[i].ptContours);
		}
		vector<tBLOB_FEATURE>().swap(OutBlob);
	}

	//如果没有运行Blob,则返回
	if (!m_bComplete)				return false;

	//Blob返回0个结果
	if (m_BlobResult.size() == 0)	return true;

	OutBlob.resize(m_BlobResult.size());

	copy(m_BlobResult.begin(), m_BlobResult.end(), OutBlob.begin());

	return true;
}

bool CFeatureExtraction::DoFeatureBasic_8bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel, CMatBuf* cMemSub)
{
	//如果有一个结果
	if (nTotalLabel <= 0)	return true;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	float fVal = 4.f * PI;

	//m_BlobResult = vector<tBLOB_FEATURE>(nTotalLabel);
	m_BlobResult.resize(nTotalLabel);

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
	for (int idx = 1; idx <= nTotalLabel; idx++)
	{
		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	continue;

		int nBlobNum = idx - 1;

		m_BlobResult.at(nBlobNum).rectBox.x = matStats.at<int>(idx, CC_STAT_LEFT);
		m_BlobResult.at(nBlobNum).rectBox.y = matStats.at<int>(idx, CC_STAT_TOP);
		m_BlobResult.at(nBlobNum).rectBox.width = matStats.at<int>(idx, CC_STAT_WIDTH);
		m_BlobResult.at(nBlobNum).rectBox.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

		//对象周围(用于背景GV)
		int nOffSet = 20;

		int nSX = m_BlobResult.at(nBlobNum).rectBox.x - nOffSet;
		int nSY = m_BlobResult.at(nBlobNum).rectBox.y - nOffSet;
		int nEX = m_BlobResult.at(nBlobNum).rectBox.x + m_BlobResult.at(nBlobNum).rectBox.width + nOffSet + nOffSet;
		int nEY = m_BlobResult.at(nBlobNum).rectBox.y + m_BlobResult.at(nBlobNum).rectBox.height + nOffSet + nOffSet;

		if (nSX < 0)	nSX = 0;
		if (nSY < 0)	nSY = 0;
		if (nSX >= GrayBuffer.cols) continue;
		if (nSY >= GrayBuffer.rows) continue;

		if (nEX >= GrayBuffer.cols)	nEX = GrayBuffer.cols - 1;
		if (nEY >= GrayBuffer.rows)	nEY = GrayBuffer.rows - 1;

		cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);

		__int64 nCount_in = 0;
		__int64 nCount_out = 0;
		__int64 nSum_in = 0;//defect区域
		__int64 nSum_out = 0;//defect排除区域

		cv::Mat matTmp_src = GrayBuffer(rectTemp);//原始ROI
		cv::Mat matTmp_label = matLabel(rectTemp);//Label的ROI
		cv::Mat matTemp = cMatBufTemp.GetMat(rectTemp.height, rectTemp.width, CV_8UC1);

		for (int y = 0; y < rectTemp.height; y++)
		{
			int* ptrLabel = (int*)matTmp_label.ptr(y);
			uchar* ptrGray = (uchar*)matTmp_src.ptr(y);
			uchar* ptrTemp = (uchar*)matTemp.ptr(y);

			for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++)
			{
				//对象
				if (*ptrLabel == idx)
				{
					nSum_in += *ptrGray;
					nCount_in++;

					//在标签向量中存储像素坐标
					m_BlobResult.at(nBlobNum).ptIndexs.push_back(cv::Point(nSX + x, nSY + y));

					*ptrTemp = (uchar)255;

					m_BlobResult.at(nBlobNum).nHist[*ptrGray]++;
				}
				//其他情况下背景
				else
				{
					//如果标签编号为0...
					//因为可以引用其他对象......添加条件
					if (*ptrLabel == 0)
					{
						nSum_out += *ptrGray;
						nCount_out++;
					}
				}
			}
		}

		//亮度累计值
		m_BlobResult.at(nBlobNum).nSumGV = nSum_in;

		//对象面积
		m_BlobResult.at(nBlobNum).nArea = nCount_in;	//matStats.at<int>(idx, CC_STAT_AREA);

		// Box Area
		m_BlobResult.at(nBlobNum).nBoxArea = m_BlobResult.at(nBlobNum).rectBox.width * m_BlobResult.at(nBlobNum).rectBox.height;

		//Bounding Box面积比率/对象面积(Rectangulaty(=Extent))
		m_BlobResult.at(nBlobNum).fBoxRatio = m_BlobResult.at(nBlobNum).nArea / (float)m_BlobResult.at(nBlobNum).nBoxArea;

		//拯救Elongation
		m_BlobResult.at(nBlobNum).fElongation = m_BlobResult.at(nBlobNum).rectBox.width / (float)m_BlobResult.at(nBlobNum).rectBox.height;

		//获取stdDev
		cv::Scalar m, s;
		cv::meanStdDev(matTmp_src, m, s, matTemp);
		m_BlobResult.at(nBlobNum).fStdDev = float(s[0]);

		//拯救Contours
		vector<vector<cv::Point>>	ptContours;
		vector<vector<cv::Point>>().swap(ptContours);
		cv::findContours(matTemp, ptContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		//获取Perimeter
		if (ptContours.size() != 0)
		{
			//ROI画面,需要校正
			//复制轮廓坐标结果
			for (int m = 0; m < ptContours.size(); m++)
			{
				for (int k = 0; k < ptContours.at(m).size(); k++)
					m_BlobResult.at(nBlobNum).ptContours.push_back(cv::Point(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY));
			}
		}
		else
		{
			//因为是原始坐标,所以不用校正。
			//复制轮廓坐标结果
			m_BlobResult.at(nBlobNum).ptContours.resize((int)m_BlobResult.at(nBlobNum).ptIndexs.size());
			std::copy(m_BlobResult.at(nBlobNum).ptIndexs.begin(), m_BlobResult.at(nBlobNum).ptIndexs.end(), m_BlobResult.at(nBlobNum).ptContours.begin());
		}
		m_BlobResult.at(nBlobNum).fPerimeter = float(cv::arcLength(m_BlobResult.at(nBlobNum).ptContours, true));
		vector<vector<cv::Point>>().swap(ptContours);

		//获取Roundness
		m_BlobResult.at(nBlobNum).fRoundness = (fVal * m_BlobResult.at(nBlobNum).nArea)
			/ (m_BlobResult.at(nBlobNum).fPerimeter * m_BlobResult.at(nBlobNum).fPerimeter);

		//对象有多接近原样？(周长^2/4*Pi*面积)
		m_BlobResult.at(nBlobNum).fCompactness = (m_BlobResult.at(nBlobNum).fPerimeter * m_BlobResult.at(nBlobNum).fPerimeter)
			/ (fVal * float(m_BlobResult.at(nBlobNum).nArea));

		//获取Defect GV
		m_BlobResult.at(nBlobNum).fMeanGV = nSum_in / (float)nCount_in;

		//获取背景GV
		m_BlobResult.at(nBlobNum).fBKGV = nSum_out / (float)nCount_out;

		//求GV差值(背景-对象)
		m_BlobResult.at(nBlobNum).fDiffGV = m_BlobResult.at(nBlobNum).fBKGV - m_BlobResult.at(nBlobNum).fMeanGV;

		//min,获取max GV
		double valMin, valMax;
		cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
		m_BlobResult.at(nBlobNum).nMinGV = (long)valMin;
		m_BlobResult.at(nBlobNum).nMaxGV = (long)valMax;

		//对象最小亮度/对象平均亮度
		m_BlobResult.at(nBlobNum).nMinGVRatio = m_BlobResult.at(nBlobNum).nMinGV / m_BlobResult.at(nBlobNum).fBKGV;

		//对象最大亮度/对象平均亮度
		m_BlobResult.at(nBlobNum).nMaxGVRatio = m_BlobResult.at(nBlobNum).nMaxGV / m_BlobResult.at(nBlobNum).fBKGV;

		//背景亮度/对象平均亮度
		m_BlobResult.at(nBlobNum).fDiffGVRatio = m_BlobResult.at(nBlobNum).fMeanGV / m_BlobResult.at(nBlobNum).fBKGV;

		//获取Center Point
		m_BlobResult.at(nBlobNum).ptCenter.x = (int)matCentroid.at<double>(idx, 0);
		m_BlobResult.at(nBlobNum).ptCenter.y = (int)matCentroid.at<double>(idx, 1);

		//拯救SEMU
		if (m_BlobResult.at(nBlobNum).fDiffGV == 0.0)
		{
			if (m_BlobResult.at(nBlobNum).fBKGV == 0)
			{
				m_BlobResult.at(nBlobNum).fSEMU = 1.0
					/ (1.97f / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
			else
			{
				m_BlobResult.at(nBlobNum).fSEMU = (0.000001 / m_BlobResult.at(nBlobNum).fBKGV)
					/ (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
		}
		else
		{
			if (m_BlobResult.at(nBlobNum).fBKGV == 0)
			{
				m_BlobResult.at(nBlobNum).fSEMU = (fabs(m_BlobResult.at(nBlobNum).fMeanGV - m_BlobResult.at(nBlobNum).fBKGV) / 0.000001)
					/ (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
			else
			{
				m_BlobResult.at(nBlobNum).fSEMU = (fabs(m_BlobResult.at(nBlobNum).fMeanGV - m_BlobResult.at(nBlobNum).fBKGV) / m_BlobResult.at(nBlobNum).fBKGV)
					/ (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
		}

		cv::RotatedRect BoundingBox = cv::minAreaRect(m_BlobResult.at(nBlobNum).ptIndexs);

		//4个旋转矩形转角点
	//cv::Point2f vertices[4];
	//BoundingBox.points(vertices);

	// Box width and length
		m_BlobResult.at(nBlobNum).BoxSize = BoundingBox.size;

		// Angle between the horizontal axis
		m_BlobResult.at(nBlobNum).fAngle = BoundingBox.angle;

		// Minor Axis & Major Axis
		if (BoundingBox.size.width > BoundingBox.size.height)
		{
			m_BlobResult.at(nBlobNum).fMinorAxis = BoundingBox.size.width;
			m_BlobResult.at(nBlobNum).fMajorAxis = BoundingBox.size.height;
		}
		else
		{
			m_BlobResult.at(nBlobNum).fMinorAxis = BoundingBox.size.height;
			m_BlobResult.at(nBlobNum).fMajorAxis = BoundingBox.size.width;
		}

		// Feret’s area
		m_BlobResult.at(nBlobNum).fMinBoxArea = m_BlobResult.at(nBlobNum).fMinorAxis * m_BlobResult.at(nBlobNum).fMajorAxis;

		// Axis Ratio
		if (m_BlobResult.at(nBlobNum).fMajorAxis > 0)
			m_BlobResult.at(nBlobNum).fAxisRatio = m_BlobResult.at(nBlobNum).fMinorAxis / m_BlobResult.at(nBlobNum).fMajorAxis;
		else
			m_BlobResult.at(nBlobNum).fAxisRatio = 0.f;

		//Min Bounding Box面积比/对象面积(区域孔隙率)
		m_BlobResult.at(nBlobNum).fMinBoxRatio = m_BlobResult.at(nBlobNum).fMinBoxArea / (float)m_BlobResult.at(nBlobNum).nArea;
		//choikwangil
		m_BlobResult.at(nBlobNum).fMeanAreaRatio = m_BlobResult.at(nBlobNum).fMeanGV / (float)m_BlobResult.at(nBlobNum).nArea;
		//取消分配
		matTmp_src.release();
		matTmp_label.release();
		matTemp.release();
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory(), m_nAlgType);
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory(), m_nAlgType);
	}

	//检查时间限制
	if (cTimeOut.GetTimeOutFlag())	return false;

	return true;
}

bool CFeatureExtraction::DoFeatureBasic_16bit(cv::Mat& matLabel, cv::Mat& matStats, cv::Mat& matCentroid, cv::Mat& GrayBuffer, int nTotalLabel, CMatBuf* cMemSub)
{
	//如果有一个结果
	if (nTotalLabel <= 0)	return true;

	CMatBuf cMatBufTemp;
	cMatBufTemp.SetMem(cMemSub);

	float fVal = 4.f * PI;

	//m_BlobResult = vector<tBLOB_FEATURE>(nTotalLabel);
	m_BlobResult.resize(nTotalLabel);

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
	for (int idx = 1; idx <= nTotalLabel; idx++)
	{
		//检查时间限制
		if (cTimeOut.GetTimeOutFlag())	continue;

		int nBlobNum = idx - 1;

		m_BlobResult.at(nBlobNum).rectBox.x = matStats.at<int>(idx, CC_STAT_LEFT);
		m_BlobResult.at(nBlobNum).rectBox.y = matStats.at<int>(idx, CC_STAT_TOP);
		m_BlobResult.at(nBlobNum).rectBox.width = matStats.at<int>(idx, CC_STAT_WIDTH);
		m_BlobResult.at(nBlobNum).rectBox.height = matStats.at<int>(idx, CC_STAT_HEIGHT);

		//对象周围(用于背景GV)
		int nOffSet = 20;

		int nSX = m_BlobResult.at(nBlobNum).rectBox.x - nOffSet;
		int nSY = m_BlobResult.at(nBlobNum).rectBox.y - nOffSet;
		int nEX = m_BlobResult.at(nBlobNum).rectBox.x + m_BlobResult.at(nBlobNum).rectBox.width + nOffSet + nOffSet;
		int nEY = m_BlobResult.at(nBlobNum).rectBox.y + m_BlobResult.at(nBlobNum).rectBox.height + nOffSet + nOffSet;

		if (nSX < 0)	nSX = 0;
		if (nSY < 0)	nSY = 0;
		if (nEX >= GrayBuffer.cols)	nEX = GrayBuffer.cols - 1;
		if (nEY >= GrayBuffer.rows)	nEY = GrayBuffer.rows - 1;

		cv::Rect rectTemp(nSX, nSY, nEX - nSX + 1, nEY - nSY + 1);

		__int64 nCount_in = 0;
		__int64 nCount_out = 0;
		__int64 nSum_in = 0;//defect区域
		__int64 nSum_out = 0;//defect排除区域

		cv::Mat matTmp_src = GrayBuffer(rectTemp);//原始ROI
		cv::Mat matTmp_label = matLabel(rectTemp);//Label的ROI
		cv::Mat matTemp = cMatBufTemp.GetMat(rectTemp.height, rectTemp.width, CV_8UC1);

		for (int y = 0; y < rectTemp.height; y++)
		{
			int* ptrLabel = (int*)matTmp_label.ptr(y);
			ushort* ptrGray = (ushort*)matTmp_src.ptr(y);
			uchar* ptrTemp = (uchar*)matTemp.ptr(y);

			for (int x = 0; x < rectTemp.width; x++, ptrLabel++, ptrGray++, ptrTemp++)
			{
				//对象
				if (*ptrLabel == idx)
				{
					nSum_in += *ptrGray;
					nCount_in++;

					//在标签向量中存储像素坐标
					m_BlobResult.at(nBlobNum).ptIndexs.push_back(cv::Point(nSX + x, nSY + y));

					*ptrTemp = (uchar)255;

					m_BlobResult.at(nBlobNum).nHist[(int)(*ptrGray)]++;
				}
				//其他情况下背景
				else
				{
					//如果标签编号为0...
					//因为可以引用其他对象......添加条件
					if (*ptrLabel == 0)
					{
						nSum_out += *ptrGray;
						nCount_out++;
					}
				}
			}
		}

		//亮度累计值
		m_BlobResult.at(nBlobNum).nSumGV = nSum_in;

		//对象面积
		m_BlobResult.at(nBlobNum).nArea = nCount_in;	//matStats.at<int>(idx, CC_STAT_AREA);

		// Box Area
		m_BlobResult.at(nBlobNum).nBoxArea = m_BlobResult.at(nBlobNum).rectBox.width * m_BlobResult.at(nBlobNum).rectBox.height;

		//Bounding Box面积比率/对象面积(Rectangulaty(=Extent))
		m_BlobResult.at(nBlobNum).fBoxRatio = m_BlobResult.at(nBlobNum).nArea / (float)m_BlobResult.at(nBlobNum).nBoxArea;

		//拯救Elongation
		m_BlobResult.at(nBlobNum).fElongation = m_BlobResult.at(nBlobNum).rectBox.width / (float)m_BlobResult.at(nBlobNum).rectBox.height;

		//获取stdDev
		cv::Scalar m, s;
		cv::meanStdDev(matTmp_src, m, s, matTemp);
		m_BlobResult.at(nBlobNum).fStdDev = float(s[0]);

		//拯救Contours
		vector<vector<cv::Point>>	ptContours;
		vector<vector<cv::Point>>().swap(ptContours);
		cv::findContours(matTemp, ptContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		//获取Perimeter
		if (ptContours.size() != 0)
		{
			//ROI画面,需要校正
			//复制轮廓坐标结果
			for (int m = 0; m < ptContours.size(); m++)
			{
				for (int k = 0; k < ptContours.at(m).size(); k++)
					m_BlobResult.at(nBlobNum).ptContours.push_back(cv::Point(ptContours.at(m)[k].x + nSX, ptContours.at(m)[k].y + nSY));
			}
		}
		else
		{
			//因为是原始坐标,所以不用校正。
			//复制轮廓坐标结果
			m_BlobResult.at(nBlobNum).ptContours.resize((int)m_BlobResult.at(nBlobNum).ptIndexs.size());
			std::copy(m_BlobResult.at(nBlobNum).ptIndexs.begin(), m_BlobResult.at(nBlobNum).ptIndexs.end(), m_BlobResult.at(nBlobNum).ptContours.begin());
		}
		m_BlobResult.at(nBlobNum).fPerimeter = float(cv::arcLength(m_BlobResult.at(nBlobNum).ptContours, true));
		vector<vector<cv::Point>>().swap(ptContours);

		//获取Roundness
		m_BlobResult.at(nBlobNum).fRoundness = (fVal * m_BlobResult.at(nBlobNum).nArea)
			/ (m_BlobResult.at(nBlobNum).fPerimeter * m_BlobResult.at(nBlobNum).fPerimeter);

		//对象有多接近原样？(周长^2/4*Pi*面积)
		m_BlobResult.at(nBlobNum).fCompactness = (m_BlobResult.at(nBlobNum).fPerimeter * m_BlobResult.at(nBlobNum).fPerimeter)
			/ (fVal * float(m_BlobResult.at(nBlobNum).nArea));

		//获取Defect GV
		m_BlobResult.at(nBlobNum).fMeanGV = nSum_in / (float)nCount_in;

		//获取背景GV
		m_BlobResult.at(nBlobNum).fBKGV = nSum_out / (float)nCount_out;

		//求GV差值(背景-对象)
		m_BlobResult.at(nBlobNum).fDiffGV = m_BlobResult.at(nBlobNum).fBKGV - m_BlobResult.at(nBlobNum).fMeanGV;

		//min,获取max GV
		double valMin, valMax;
		cv::minMaxLoc(matTmp_src, &valMin, &valMax, 0, 0, matTemp);
		m_BlobResult.at(nBlobNum).nMinGV = (long)valMin;
		m_BlobResult.at(nBlobNum).nMaxGV = (long)valMax;

		//对象最小亮度/对象平均亮度
		m_BlobResult.at(nBlobNum).nMinGVRatio = m_BlobResult.at(nBlobNum).nMinGV / m_BlobResult.at(nBlobNum).fBKGV;

		//对象最大亮度/对象平均亮度
		m_BlobResult.at(nBlobNum).nMaxGVRatio = m_BlobResult.at(nBlobNum).nMaxGV / m_BlobResult.at(nBlobNum).fBKGV;

		//背景亮度/对象平均亮度
		m_BlobResult.at(nBlobNum).fDiffGVRatio = m_BlobResult.at(nBlobNum).fMeanGV / m_BlobResult.at(nBlobNum).fBKGV;

		//获取Center Point
		m_BlobResult.at(nBlobNum).ptCenter.x = (int)matCentroid.at<double>(idx, 0);
		m_BlobResult.at(nBlobNum).ptCenter.y = (int)matCentroid.at<double>(idx, 1);

		//拯救SEMU
		if (m_BlobResult.at(nBlobNum).fDiffGV == 0.0)
		{
			if (m_BlobResult.at(nBlobNum).fBKGV == 0)
			{
				m_BlobResult.at(nBlobNum).fSEMU = 1.0
					/ (1.97f / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
			else
			{
				m_BlobResult.at(nBlobNum).fSEMU = (0.000001 / m_BlobResult.at(nBlobNum).fBKGV)
					/ (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
		}
		else
		{
			if (m_BlobResult.at(nBlobNum).fBKGV == 0)
			{
				m_BlobResult.at(nBlobNum).fSEMU = (fabs(m_BlobResult.at(nBlobNum).fMeanGV - m_BlobResult.at(nBlobNum).fBKGV) / 0.000001)
					/ (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
			else
			{
				m_BlobResult.at(nBlobNum).fSEMU = (fabs(m_BlobResult.at(nBlobNum).fMeanGV - m_BlobResult.at(nBlobNum).fBKGV) / m_BlobResult.at(nBlobNum).fBKGV)
					/ (1.97 / (cv::pow((float)m_BlobResult.at(nBlobNum).nArea, 0.33f) + 0.72f));
			}
		}

		cv::RotatedRect BoundingBox = cv::minAreaRect(m_BlobResult.at(nBlobNum).ptIndexs);

		//4个旋转矩形转角点
	//cv::Point2f vertices[4];
	//BoundingBox.points(vertices);

	// Box width and length
		m_BlobResult.at(nBlobNum).BoxSize = BoundingBox.size;

		// Angle between the horizontal axis
		m_BlobResult.at(nBlobNum).fAngle = BoundingBox.angle;

		// Minor Axis & Major Axis
		if (BoundingBox.size.width > BoundingBox.size.height)
		{
			m_BlobResult.at(nBlobNum).fMinorAxis = BoundingBox.size.width;
			m_BlobResult.at(nBlobNum).fMajorAxis = BoundingBox.size.height;
		}
		else
		{
			m_BlobResult.at(nBlobNum).fMinorAxis = BoundingBox.size.height;
			m_BlobResult.at(nBlobNum).fMajorAxis = BoundingBox.size.width;
		}

		// Feret’s area
		m_BlobResult.at(nBlobNum).fMinBoxArea = m_BlobResult.at(nBlobNum).fMinorAxis * m_BlobResult.at(nBlobNum).fMajorAxis;

		// Axis Ratio
		if (m_BlobResult.at(nBlobNum).fMajorAxis > 0)
			m_BlobResult.at(nBlobNum).fAxisRatio = m_BlobResult.at(nBlobNum).fMinorAxis / m_BlobResult.at(nBlobNum).fMajorAxis;
		else
			m_BlobResult.at(nBlobNum).fAxisRatio = 0.f;

		//Min Bounding Box面积比/对象面积(区域孔隙率)
		m_BlobResult.at(nBlobNum).fMinBoxRatio = m_BlobResult.at(nBlobNum).fMinBoxArea / (float)m_BlobResult.at(nBlobNum).nArea;
		//choikwangil
		m_BlobResult.at(nBlobNum).fMeanAreaRatio = m_BlobResult.at(nBlobNum).fMeanGV / (float)m_BlobResult.at(nBlobNum).nArea;
		//取消分配
		matTmp_src.release();
		matTmp_label.release();
		matTemp.release();
	}

	if (m_cInspectLibLog->Use_AVI_Memory_Log) {
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory(), m_nAlgType);
		writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory(), m_nAlgType);
	}
	//检查时间限制
	if (cTimeOut.GetTimeOutFlag())	return false;

	return true;
}

//17.11.29-外围信息(AVI&SVI其他工具)
bool CFeatureExtraction::calcContours(int* nContoursX, int* nContoursY, int nDefectIndex, float fAngle, int nCx, int nCy, int nPs)
{
	//如果没有运行Blob,则返回
	if (!m_bComplete)					return false;

	//Blob返回0个结果
	if (m_BlobResult.size() == 0)		return true;

	//如果输入了错误的参数
	if (nDefectIndex < 0)						return false;
	if (nDefectIndex >= m_BlobResult.size())	return false;

	//////////////////////////////////////////////////////////////////////////

		//轮廓数量
	int nContoursCount = m_BlobResult[nDefectIndex].ptContours.size();

	//计算旋转坐标时,使用
	double dTheta = -fAngle * PI / 180.;
	double	dSin = sin(dTheta);
	double	dCos = cos(dTheta);

	float fRatio = 1.0;

	//如果高于设置数量
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

		//旋转时,计算预测坐标
		int X = (int)(dCos * (m_BlobResult[nDefectIndex].ptContours[i].x - nCx) - dSin * (m_BlobResult[nDefectIndex].ptContours[i].y - nCy) + nCx);
		int Y = (int)(dSin * (m_BlobResult[nDefectIndex].ptContours[i].x - nCx) + dCos * (m_BlobResult[nDefectIndex].ptContours[i].y - nCy) + nCy);

		//根据P/S模式修改坐标(更改为单杆坐标)
		nContoursX[j] = (int)(X / nPs);
		nContoursY[j] = (int)(Y / nPs);
	}

	return true;
}
long CFeatureExtraction::DoDefectAIDectectJudgment(
	cv::Mat& matSrcImage,
	cv::Mat& matDrawBuffer,
	int* nCommonPara,
	long nDefectColor,
	CString strTxt,
	stPanelBlockJudgeInfo* EngineerBlockDefectJudge,
	stDefectInfo* pResultBlob,
	std::shared_ptr<vector<tBLOB_FEATURE>> detBlob,
	cv::Rect cutRoi,
	int nDefectType)
{
	//开始超时
	cTimeOut.TimeCheckStart();

	//禁用内存
//Release();

	//如果参数为NULL。
	if (nCommonPara == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (pResultBlob == NULL)						return E_ERROR_CODE_EMPTY_PARA;
	if (EngineerBlockDefectJudge == NULL) 				return E_ERROR_CODE_EMPTY_PARA;

	if (nDefectType < 0)							return E_ERROR_CODE_EMPTY_PARA;
	if (nDefectType >= E_DEFECT_JUDGEMENT_COUNT)	return E_ERROR_CODE_EMPTY_PARA;

	//如果画面缓冲区为NULL
	if (matSrcImage.empty())						return E_ERROR_CODE_EMPTY_BUFFER;

	//////////////////////////////////////////////////////////////////////////
		//公共参数
	int		nMaxDefectCount = nCommonPara[E_PARA_COMMON_MAX_DEFECT_COUNT];//00:最大不良数量
	bool	bImageSave = (nCommonPara[E_PARA_COMMON_IMAGE_SAVE_FLAG] > 0) ? true : false;//01:算法中间结果Image Save
	int& nSaveImageCount = nCommonPara[E_PARA_COMMON_IMAGE_SAVE_COUNT];//02:画面存储顺序计数
	int		nImageNum = nCommonPara[E_PARA_COMMON_ALG_IMAGE_NUMBER];//03:当前画面号码
	int		nCamNum = nCommonPara[E_PARA_COMMON_CAMERA_NUMBER];						// 04 : Cam Number
	int		nROINumber = nCommonPara[E_PARA_COMMON_ROI_NUMBER];						// 05 : ROI Number
	int		nAlgorithmNumber = nCommonPara[E_PARA_COMMON_ALG_NUMBER];//06:算法编号
	int		nThrdIndex = nCommonPara[E_PARA_COMMON_THREAD_ID];						// 07 : Thread ID
	bool	bDefectNum = (nCommonPara[E_PARA_COMMON_DRAW_DEFECT_NUM_FLAG] > 0) ? true : false;//08:Draw Defect Num显示
	bool	bDrawDust = (nCommonPara[E_PARA_COMMON_DRAW_DUST_FLAG] > 0) ? true : false;//显示09:Draw Dust
	int		nPatternImageNum = nCommonPara[E_PARA_COMMON_UI_IMAGE_NUMBER];//10:UI上的模式顺序画面号
	float	fAngle = nCommonPara[E_PARA_COMMON_ROTATE_ANGLE] / 1000.f;//11:Cell旋转角度(Align计算值,小数点仅为3位...)
	int		nCx = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_X];//12:Cell旋转中心x坐标
	int		nCy = nCommonPara[E_PARA_COMMON_ROTATE_CENTER_Y];//13:Cell旋转中心y坐标
	int		nPS = nCommonPara[E_PARA_COMMON_PS_MODE];//14:图像比例(Pixel Shift Mode-1:None,2:4-Shot,3:9-Shot)

	//输入的错误数量
	int& nDefectCount = pResultBlob->nDefectCount;
	////////////////////////////////
// 新增common参数 [hjf]
	int		nBlockX = nCommonPara[E_PARA_COMMON_BLOCK_X];
	int		nBlockY = nCommonPara[E_PARA_COMMON_BLOCK_Y];
	// 
	// /////////////////////////////////

		//超过最大不良数量时退出
	if (nDefectCount >= nMaxDefectCount)
		return E_ERROR_CODE_TRUE;

	int nFork = nDefectType;

	//if (!EngineerDefectJudgment[nFork].bDefectItemUse)
	//	return E_ERROR_CODE_TRUE;
		//Blob数量
	int dx, dy;
	dx = cutRoi.x;
	dy = cutRoi.y;
	for (int i = 0; i < detBlob->size(); i++)
	{
		tBLOB_FEATURE temp = (*detBlob)[i];
		long x1, y1, x2, y2;
		x1 = temp.rectBox.x + dx;
		y1 = temp.rectBox.y + dy;

		temp.rectBox.y = y1;
		temp.rectBox.x = x1;

		x2 = x1 + temp.rectBox.width;
		y2 = y1 + temp.rectBox.height;
		POINT ptLT, ptRB;
		ptLT.x = x1; ptLT.y = y1;
		ptRB.x = x2; ptRB.y = y2;
		pResultBlob->ptLT[nDefectCount] = ptLT;
		pResultBlob->ptRB[nDefectCount] = ptRB;
		pResultBlob->nDefectJudge[nDefectCount] = nDefectType;
		pResultBlob->nPatternClassify[nDefectCount] = nPatternImageNum;
		pResultBlob->AI_CODE.push_back(temp.AICode);
		pResultBlob->AI_Confidence.push_back(temp.confidence);
		pResultBlob->AI_Object_Nums++;

		if (!matDrawBuffer.empty())
		{
			cv::rectangle(matDrawBuffer, temp.rectBox, cv::Scalar(255, 0, 0), 1);
		}
		if (nDefectCount >= nMaxDefectCount)
			break;
		nDefectCount++;
	}
	m_bComplete = true;

	return E_ERROR_CODE_TRUE;
}
