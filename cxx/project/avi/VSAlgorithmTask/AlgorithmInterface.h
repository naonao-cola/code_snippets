/*****************************************************************************
  File Name		: InspectAlgorithmInterface.h
  Version		: ver 1.0
  Create Date	: 2015.03.12
  Description	:检查算法接口
  Abbreviations	:
 *****************************************************************************/

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "DefineInterface.h"
#include "InspResultInfo.h"
#include <concurrent_queue.h>
#include "InspThrd.h"

 //添加B11DICS
#include "DICS_B11.h"

//要传递给检测算法的结果数据的最大完美尺寸
//18.02.05-SVI不良发生频率低20000->1000
#define	MAX_DEFECT_COUNT		1000

//Logic使用的参数结构
struct tLogicPara
{
	cv::Mat				MatOrgImage;		//相应的阵列源
	//	cv::Mat				MatOrgRGB[3];		// R,G,B画面
	cv::Mat				MatBKG;				//在Mura中使用(非点等区域)
	cv::Mat* MatOrgRGBAdd[3];		// 
	cv::Mat				MatDust;			//Dust画面
	int					nThreadLog;
	int					nCameraNum;
	CString				strPanelID;
	TCHAR				tszPanelID[MAX_PATH];

	//add AI DICS img
	cv::Mat				MatDics;

	//初始化结构体
	tLogicPara() :
		nThreadLog(0), nCameraNum(0)
	{
		strPanelID.Format(_T(""));		memset(tszPanelID, 0, MAX_PATH * sizeof(TCHAR));
	}
};

//	Class功能	:	总结果数据列表
//主要功能	:
//	创建日期	:2015/03/12
//	作者	:	KYT
//	修改历史记录	:	V.1.0初始创建
//	请参见	:	

class ResultBlob_Total
{
public:
	ResultBlob_Total()
	{
		InitializeCriticalSectionAndSpinCount(&m_cs, 4000);
	}

	~ResultBlob_Total()
	{
		stDefectInfo* pTempResultBlob = NULL;
		DeleteCriticalSection(&m_cs);
		for (POSITION pos = m_ListDefectInfo.GetHeadPosition(); pos != NULL;)
		{
			pTempResultBlob = m_ListDefectInfo.GetNext(pos);

			if (pTempResultBlob != NULL)
			{
				delete pTempResultBlob;
				pTempResultBlob = NULL;
			}
		}

		m_ListDefectInfo.RemoveAll();
	}

	stDefectInfo* GetNext_ResultBlob(POSITION& ps)
	{
		EnterCriticalSection(&m_cs);
		stDefectInfo* rb = m_ListDefectInfo.GetNext(ps);
		LeaveCriticalSection(&m_cs);

		return rb;
	}

	void RemoveAll_ResultBlob()
	{
		stDefectInfo* pTempResultBlob = NULL;

		for (POSITION pos = m_ListDefectInfo.GetHeadPosition(); pos != NULL;)
		{
			pTempResultBlob = m_ListDefectInfo.GetNext(pos);

			if (pTempResultBlob != NULL)
			{
				delete pTempResultBlob;
				pTempResultBlob = NULL;
			}
		}

		m_ListDefectInfo.RemoveAll();
	}

	POSITION GetHeadPosition_ResultBlob()
	{
		EnterCriticalSection(&m_cs);
		POSITION ps = m_ListDefectInfo.GetHeadPosition();
		LeaveCriticalSection(&m_cs);

		return ps;
	}

	POSITION AddTail_ResultBlobAndAddOffset(stDefectInfo* pResultBlob, cv::Point* ptCorner)
	{
		//AVI按原始画面计算,无需校正Corner:NULL
//if( ptCorner )
//{
//	for(int nDefectCnt = 0; nDefectCnt < pResultBlob->nDefectNumber; nDefectCnt++)
//	{
//		pResultBlob->Box_X_Max[nDefectCnt] += rectTeachingROI.left;
//		pResultBlob->Box_X_Min[nDefectCnt] += rectTeachingROI.left;
//		pResultBlob->Box_Y_Max[nDefectCnt] += rectTeachingROI.top;
//		pResultBlob->Box_Y_Min[nDefectCnt] += rectTeachingROI.top;
//	}
//}		

		EnterCriticalSection(&m_cs);
		POSITION ps = m_ListDefectInfo.AddTail(pResultBlob);
		LeaveCriticalSection(&m_cs);

		return ps;
	}

	void ResultBlobAddOffset(stDefectInfo* pResultBlob, CRect& rectTeachingROI)
	{
		//当前禁用Offset

		return;
	}

	POSITION AddTail_ResultBlob(stDefectInfo* pResultBlob)
	{
		EnterCriticalSection(&m_cs);
		POSITION ps = m_ListDefectInfo.AddTail(pResultBlob);
		LeaveCriticalSection(&m_cs);

		return ps;
	}

	BOOL IsEmpty_ResultBlob()
	{
		BOOL bRet = TRUE;

		EnterCriticalSection(&m_cs);

		stDefectInfo* pTempResultBlob = NULL;

		for (POSITION pos = m_ListDefectInfo.GetHeadPosition(); pos != NULL;)
		{
			pTempResultBlob = m_ListDefectInfo.GetNext(pos);

			if (pTempResultBlob->nDefectCount != 0)
			{
				bRet = FALSE;
				break;
			}
		}

		LeaveCriticalSection(&m_cs);
		return bRet;
	}

	void	SetPanelID(CString strPanelID) { m_strPanelID = strPanelID; }
	CString GetPanelID() { return m_strPanelID; }
	void	SetModelID(CString strModelID) { m_strModelID = strModelID; }
	CString GetModelID() { return m_strModelID; }
private:
	CList<stDefectInfo*, stDefectInfo*>	m_ListDefectInfo;
	CRITICAL_SECTION	m_cs;
	CString				m_strPanelID;
	CString				m_strModelID;
};

//	Class功能	:	检查算法接口
//主要功能	:
//	创建日期	:2015/03/12
//	作者	:	KYT
//	修改历史记录	:	V.1.0初始创建
//	请参见	:	

class InspectAlgorithmInterface
{
public:
	InspectAlgorithmInterface();
	~InspectAlgorithmInterface(void);

protected:

	//算法后,保存Defect结果文件
	void			BlobFeatureSave(stDefectInfo* resultBlob, CString strPath, int* nImageDefectCount = NULL);

	/**
	 * 保存每个算法的缺陷特征
	 *
	 * \param resultBlob
	 * \param strPath
	 * \param nImageDefectCount
	 */
	void			AlgoFeatureSave(stDefectInfo* resultBlob, CString strPath, CString strPanelID, int nImageNum, CString strAlgorithmName, int nStageNum, int* nDefectCount = NULL);

	//开始检查单个细胞
	long			StartLogic(CString strPanelID, CString strDrive, tAlignInfo stThrdAlignInfo,
		cv::Mat MatOriginImage[][MAX_CAMERA_COUNT], cv::Mat& MatDrawBuffer, cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE],
		ResultBlob_Total* pResultBlob_Total, const int nImageNum, const int nCameraNum, const int nThreadIndex, bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtReusltInfo, const double* _mtp, STRU_LabelMarkInfo& labelMarkInfo);

	void			DrawAdjustROI(cv::Mat& MatDrawBuffer, cv::Point* pPtCorner, LPCTSTR strROIName, int nCurCount, int nDrawMode);	//添加名称参数

	//自动ROI获取和Align Image
	long			AcquireAutoRoiAndAlignImage(CString strPanelID, CString strDrive, cv::Mat& MatOrgImage, int nRatio, int nImageNum, int nCameraNum, tAlignInfo& stCamAlignInfo, double* dResult, double dCamResolution, double dPannelSizeX, double dPannelSizeY);

	long			PanelCurlJudge(cv::Mat& matSrcBuf, double* dPara, tAlignInfo* stCamAlignInfo, ResultBlob_Total* pResultBlobTotal, int nImageNum, stMeasureInfo* stCurlMeasure, CString strPath);

	////////////////////////虚拟函数
	//开始算法检查
	virtual long	StartLogicAlgorithm(const CString strDrive, const tLogicPara& LogicPara,
		cv::Mat MatResultImg[][MAX_CAMERA_COUNT][MAX_MEM_SIZE_E_ALGORITHM_NUMBER][MAX_MEM_SIZE_E_MAX_INSP_TYPE], cv::Mat& MatDrawBuffer,
		const int nImageNum, const int nROINumber, const int nAlgorithmNumber,
		tAlignInfo stThrdAlignInfo, ResultBlob_Total* pResultBlob_Total,
		bool bpInspectEnd[][MAX_CAMERA_COUNT], int nRatio, ENUM_INSPECT_MODE eInspMode, CWriteResultInfo& WrtResultInfo, const double* _mtp = 0) = 0;
	//AVI:Align信息校准-假设旋转,仅校准坐标
	//SVI:用Warping坐标校正
	//AVI:Align信息校正-假设旋转,坐标校正-之后Rotate

	virtual bool	AdjustAlignInfo(tAlignInfo* pStCamAlignInfo, cv::Point* ptAdjCorner) = 0;

	//原始图像校正
	//AVI:什么都不做/SVI:Warping/APP:Rotate
	virtual bool	AdjustOriginImage(cv::Mat& MatOriginImage, cv::Mat& MatDrawImage, tAlignInfo* pStAlignInfo) { return false; };

	//外围处理
	virtual long	makePolygonCellROI(const tLogicPara& LogicPara, cv::Mat& MatDrawBuffer, tAlignInfo& stThrdAlignInfo, STRU_LabelMarkInfo& labelMarkInfo, int nImageNum, int nCameraNum, double* dAlgPara, int nAlgImg, int nRatio) = 0;
	// yuxuefei add
	virtual long LabelProcess(cv::Mat matSrcBuf, int nImageNum, int nCameraNum, tAlignInfo& stCamAlignInfo) { return E_ERROR_CODE_TRUE; };

	// yuxuefei add
	virtual long  MarkProcess(cv::Mat matSrcBuf, int nImageNum, int  nCameraNum, tAlignInfo& stCamAlignInfo) { return E_ERROR_CODE_TRUE; };
	//Align之前的AD检查
	virtual long	CheckAD(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nImageNum, int nCameraNum, double* dResult, int nRatio) { return E_ERROR_CODE_FALSE; };

	//ROI GV检查
	virtual long	CheckADGV(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nStageNo, int nImageNum, int nCameraNum, int nRatio, cv::Point* ptCorner, ResultBlob_Total* pResultBlobTotal, double* dMeanResult,
		bool& bChkDustEnd, bool& bNeedRetry, bool& bIsNormalDust, bool bUseDustRetry, int nDustRetryCnt, bool& bIsHeavyAlarm, ENUM_INSPECT_MODE eInspMode) {
		return E_ERROR_CODE_TRUE;
	};

	//Align之前的AD检查
	virtual long	CheckPGConnect(CString strPanelID, CString strDrive, cv::Mat MatOrgImage, int nImageNum, int nCameraNum, double* dResult, cv::Point* cvPt) { return E_ERROR_CODE_FALSE; };
};