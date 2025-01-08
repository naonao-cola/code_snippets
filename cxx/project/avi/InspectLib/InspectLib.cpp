
/************************************************************************/
// UI에서 사용할 알고리즘 함수 DLL 정의
// 수정일 : 18.02.07
/************************************************************************/

// InspectLib.cpp : DLL 응용 프로그램을 위해 내보낸 함수를 정의합니다.
//

#include "stdafx.h"
#include "InspectLib.h"

//////////////////////////////////////////////////////////////////////////
// 기타 알고리즘
#include "DefectCCD.h"			// CCD Defect
#include "MatBufferManager.h"	// 메모리 관리
#include "ColorCorrection.h"	// SVI Color 보정

//////////////////////////////////////////////////////////////////////////
// 검출 알고리즘
#include "InspectAlign.h"		// 공통 Align 알고리즘 관련 헤더
#include "InspectPoint.h"		// 점등 Point 알고리즘 관련 헤더
#include "InspectLine.h"		// 점등 Line 알고리즘 관련 헤더
#include "InspectMura.h"		// 점등 Mura 알고리즘 관련 헤더
#include "InspectMura2.h"		// 점등 Mura2 알고리즘 관련 헤더
#include "InspectMuraChole.h"	// 점등 Mura Chole 알고리즘 관련 헤더 //2021.01.06
#include "InspectMuraScratch.h"	// 점등 Mura Chole 알고리즘 관련 헤더 //2021.01.06 
#include "InspectMuraNormal.h"	// 점등 MuraNormal 알고리즘 관련 헤더
#include "InspectMura3.h"
#include "InspectMura4.h"

#include "InspectDustPs.h"

//////////////////////////////////////////////////////////////////////////

#include "InspectLibLog.h"			// 알고리즘 Srart & End 로그
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////////

// CCD Defect Index
CDefectCCD			cCCD;

// 메모리 관리
CMatBufferManager	cMem;

// SVI Color 보정
CColorCorrection	cColor;

// 알고리즘 Srart & End 로그
InspectLibLog		cInspectLibLog;

//////////////////////////////////////////////////////////////////////////

// 유일한 응용 프로그램 개체입니다.

CWinApp theApp;

using namespace std;

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	int nRetCode = 0;

	HMODULE hModule = ::GetModuleHandle(NULL);

	if (hModule != NULL)
	{
		// MFC를 초기화합니다. 초기화하지 못한 경우 오류를 인쇄합니다.
		if (!AfxWinInit(hModule, NULL, ::GetCommandLine(), 0))
		{
			// TODO: 오류 코드를 필요에 따라 수정합니다.
			_tprintf(_T("심각한 오류: MFC를 초기화하지 못했습니다.\n"));
			nRetCode = 1;
		}
		else
		{
			// TODO: 응용 프로그램의 동작은 여기에서 코딩합니다.
		}
	}
	else
	{
		// TODO: 오류 코드를 필요에 따라 수정합니다.
		_tprintf(_T("심각한 오류: GetModuleHandle 실패\n"));
		nRetCode = 1;
	}

	return nRetCode;
}

extern "C" AFX_API_EXPORT void Init_InspectLib(CString initFilePath)
{
	cMem.Initialize(initFilePath);
}

//////////////////////////////////////////////////////////////////////////
// Align 관련
//////////////////////////////////////////////////////////////////////////

// Top Line 각도 찾기
extern "C" AFX_API_EXPORT	long	Align_FindTheta(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point & ptCenter, wchar_t* strID)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

	cInspectAlign.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));

	long nRes = cInspectAlign.DoFindTheta(matSrcBuf, dPara, dTheta, ptCenter, strID);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// 검사 영역 찾기
extern "C" AFX_API_EXPORT	long	Align_FindActive(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point * ptResCorner, cv::Point* ptContCorner, int nCameraNum, int nEQType, double dCamResolution, double dPannelSizeX, double dPannelSizeY, int nRatio, cv::Point & ptCenter, wchar_t* strID)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	// 3채널 이상 검사 안함
	if (matSrcBuf.channels() > 3)
	{
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("channels over."));

		return	E_ERROR_CODE_ALIGN_NO_DATA;
	}

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEQType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf_Low();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindActive(matSrcBuf, dPara, dTheta, ptResCorner, ptContCorner, nRatio, ptCenter, strID);
		cMem.ReleaseFreeBuf_Low(pBuf);
		break;

	case 1://SVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindActive_SVI(matSrcBuf, dPara, dTheta, ptResCorner, nCameraNum, nRatio, ptCenter, strID);
		break;

	case 2://APP
		pBuf = cMem.FindFreeBuf();
		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindActive_APP(matSrcBuf, dPara, dTheta, ptResCorner, nRatio, dCamResolution, dPannelSizeX, dPannelSizeY, ptCenter, strID);
		cMem.ReleaseFreeBuf(pBuf);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// 외곽 곡선 설정 & 파일 저장   设置轮廓曲线&保存文件
extern "C" AFX_API_EXPORT	long	Align_SetFindContour(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA],
	int nRoundROICnt, double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strPath, Point * ptAlignCorner, CStringA strImageName, double dAlignTheta, bool bImageSave)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.SetFindContour(matSrcBuf, RoundROI, nRoundROICnt, dPara, nAlgImg, nRatio, strPath);
		cMem.ReleaseFreeBuf(pBuf);
		break;

	case 1://SVI
		nRes = E_ERROR_CODE_FALSE;
		break;

	case 2://APP
		pBuf = cMem.FindFreeBuf();
		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.SetFindContour_APP(matSrcBuf, RoundROI, nRoundROICnt, dPara, nAlgImg, nRatio, strPath, ptAlignCorner, strImageName, dAlignTheta, bImageSave);
		cMem.ReleaseFreeBuf(pBuf);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

//设置轮廓曲线&保存文件
extern "C" AFX_API_EXPORT	long	Align_SetFindContour_(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA],
	int nRoundROICnt, int nCHoleROICnt, double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strPath, Point * ptAlignCorner, CStringA strImageName, double dAlignTheta, bool bImageSave)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.SetFindContour_(matSrcBuf, RoundROI, CHoleROI, nRoundROICnt, nCHoleROICnt, dPara, nAlgImg, nRatio, strPath);
		cMem.ReleaseFreeBuf(pBuf);
		break;

	case 1://SVI
		nRes = E_ERROR_CODE_FALSE;
		break;

	case 2://APP
		//pBuf = cMem.FindFreeBuf();
		//cInspectAlign.SetMem(pBuf);
		//cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		//nRes = cInspectAlign.SetFindContour_APP(matSrcBuf, RoundROI, nRoundROICnt, dPara, nAlgImg, nRatio, strPath, ptAlignCorner, strImageName, dAlignTheta, bImageSave);
		//cMem.ReleaseFreeBuf(pBuf);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// 외곽 곡선 설정 & 파일 저장
extern "C" AFX_API_EXPORT	long	Align_SetFindContour_2(cv::Mat * matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA],
	int nRoundROICnt, int nCHoleROICnt, double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strPath, Point * ptAlignCorner, CStringA strImageName, double dAlignTheta, bool bImageSave)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.SetFindContour_2(matSrcBuf, RoundROI, CHoleROI, nRoundROICnt, nCHoleROICnt, dPara, nAlgImg, nRatio, strPath);
		cMem.ReleaseFreeBuf(pBuf);
		break;

	case 1://SVI
		nRes = E_ERROR_CODE_FALSE;
		break;

	case 2://APP
		//pBuf = cMem.FindFreeBuf();
		//cInspectAlign.SetMem(pBuf);
		//cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		//nRes = cInspectAlign.SetFindContour_APP(matSrcBuf, RoundROI, nRoundROICnt, dPara, nAlgImg, nRatio, strPath, ptAlignCorner, strImageName, dAlignTheta, bImageSave);
		//cMem.ReleaseFreeBuf(pBuf);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// 외곽 부분 처리
extern "C" AFX_API_EXPORT	long	Align_FindFillOutArea(cv::Mat matSrcBuf, cv::Mat & MatDrawBuffer, cv::Mat matBKGBuf, cv::Point ptResCorner[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA], cv::Mat * matCHoleROIBuf, cv::Rect * rcCHoleROI, bool* bCHoleAD,
	double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strAlgLog, wchar_t* strID, cv::Point * ptCorner, vector<vector<Point2i>> &ptActive, double dAlignTH, CString strPath, bool bImagSave)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

		pBuf = cMem.FindFreeBuf_High();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);
		nRes = cInspectAlign.DoFillOutArea(matSrcBuf, MatDrawBuffer, matBKGBuf, ptResCorner, labelMarkParams, labelMarkInfo, tRoundSet, tCHoleSet, matCHoleROIBuf, rcCHoleROI, bCHoleAD, dPara, nAlgImg, nRatio, strID);
		cMem.ReleaseFreeBuf_High(pBuf);
		break;

	case 1://SVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);
		nRes = cInspectAlign.DoFillOutArea_SVI(matSrcBuf, matBKGBuf, tRoundSet, dPara, nAlgImg, nCameraNum, nRatio, strID, ptCorner);
		break;

	case 2://APP
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);
		nRes = cInspectAlign.DoFillOutArea_APP(matSrcBuf, matBKGBuf, tRoundSet, dPara, nAlgImg, nCameraNum, nRatio, strID, ptCorner, ptActive, dAlignTH, strPath, bImagSave);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// 외곽 부분 처리
extern "C" AFX_API_EXPORT	long	Align_FindFillOutAreaDust(cv::Mat matSrcBuf, cv::Mat& MatDrawBuffer, cv::Point ptResCorner[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, double dAngle, cv::Rect * rcCHoleROI,
	ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA], double* dPara, int nAlgImg, int nRatio, wchar_t* strAlgLog, wchar_t* strID)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectAlign.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectAlign.DoFillOutAreaDust(matSrcBuf, MatDrawBuffer, ptResCorner, labelMarkParams, labelMarkInfo, dAngle, rcCHoleROI, tRoundSet, tCHoleSet, dPara, nAlgImg, nRatio, strID);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// 영상 회전
extern "C" AFX_API_EXPORT	long	Align_RotateImage(cv::Mat matSrcBuffer, cv::Mat & matDstBuffer, double dAngle)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	long nRes = cInspectAlign.DoRotateImage(matSrcBuffer, matDstBuffer, dAngle);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// 좌표 회전
extern "C" AFX_API_EXPORT	long	Align_DoRotatePoint(cv::Point matSrcPoint, cv::Point & matDstPoint, cv::Point ptCenter, double dAngle)
{
	// 시간 별로 안걸림
//clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign	cInspectAlign;
	//cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	long nRes = cInspectAlign.DoRotatePoint(matSrcPoint, matDstPoint, ptCenter, dAngle);

	//cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// AD 검사	 / dResult : 현재 Cell 일치율
extern "C" AFX_API_EXPORT	long	Align_FindDefectAD(cv::Mat matSrcBuf, double* dPara, double* dResult, int nRatio, int nCameraNum, int nEqpType)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	// 3채널 이상 검사 안함
	if (matSrcBuf.channels() > 3)
	{
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("channels over."));

		return	E_ERROR_CODE_ALIGN_NO_DATA;
	}

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf_Low();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindDefectAD(matSrcBuf, dPara, dResult, nRatio);
		cMem.ReleaseFreeBuf_Low(pBuf);
		break;

	case 1://SVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindDefectAD_SVI(matSrcBuf, dPara, dResult, nCameraNum, nRatio);
		break;

	case 2://APP
		pBuf = cMem.FindFreeBuf();
		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindDefectAD_APP(matSrcBuf, dPara, dResult, nRatio);
		cMem.ReleaseFreeBuf(pBuf);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

extern "C" AFX_API_EXPORT	long	Align_FindDefectLabel(cv::Mat matSrcBuf, double* dPara, cv::Point * ptCorner, double dAngel, CRect rectLabelArea[MAX_MEM_SIZE_LABEL_COUNT], int nEqpType)
{

	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	// 3채널 이상 검사 안함
	if (matSrcBuf.channels() > 3)
	{
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("channels over."));

		return	E_ERROR_CODE_ALIGN_NO_DATA;
	}

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf_Low();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindDefectLabel(matSrcBuf, dPara, ptCorner, dAngel, rectLabelArea);
		cMem.ReleaseFreeBuf_Low(pBuf);
		break;
	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// AD 검사	 / dResult : 현재 Cell 일치율
extern "C" AFX_API_EXPORT	long	Align_FindDefectPGConnect(cv::Mat matSrcBuf, double* dPara, double* dResult, cv::Point * nRatio, int nCameraNum, int nEqpType)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	// 3채널 이상 검사 안함
	if (matSrcBuf.channels() > 3)
	{
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("channels over."));

		return	E_ERROR_CODE_ALIGN_NO_DATA;
	}

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf_Low();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindPGAnomal_AVI(matSrcBuf, dPara, dResult, nRatio);
		cMem.ReleaseFreeBuf_Low(pBuf);
		break;
	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}


extern "C" AFX_API_EXPORT	long	Align_FindDefectMark(cv::Mat matSrcBuf, double* dPara, cv::Point * ptCorner, double dAngel, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT], int nEqpType, int nMarkROICnt)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	// 3채널 이상 검사 안함
	if (matSrcBuf.channels() > 3)
	{
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("channels over."));

		return	E_ERROR_CODE_ALIGN_NO_DATA;
	}

	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	CMatBuf* pBuf = NULL;
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"));

		pBuf = cMem.FindFreeBuf_Low();

		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"));

		cInspectAlign.SetMem(pBuf);
		cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
		nRes = cInspectAlign.DoFindDefectMark(matSrcBuf, dPara, ptCorner, dAngel, rcMarkROI, rectMarkArea, nMarkROICnt);
		cMem.ReleaseFreeBuf_Low(pBuf);
		break;
	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

// AD GV 검사
extern "C" AFX_API_EXPORT	long	Align_FindDefectAD_GV(cv::Mat & matSrcBuf, double* dPara, double* dResult, cv::Point * ptCorner, int nEqpType, int nImageNum, wchar_t* strAlgLog)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CMatBuf* pBuf = NULL;
	CInspectAlign	cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);
	long			nRes = E_ERROR_CODE_TRUE;

	switch (nEqpType)
	{
	case 0://AVI
		if (nImageNum == E_IMAGE_CLASSIFY_AVI_DUST || nImageNum == E_IMAGE_CLASSIFY_AVI_DUSTDOWN)//新增背光除尘画面检查 hjf
			nRes = cInspectAlign.DoFindDefectAD_GV_DUST(matSrcBuf, dPara, dResult, ptCorner);
		else
		{
			cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

			pBuf = cMem.FindFreeBuf_Low();

			cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

			cInspectAlign.SetMem(pBuf);

			nRes = cInspectAlign.DoFindDefectAD_GV(matSrcBuf, dPara, dResult, ptCorner, &cCCD);

			cMem.ReleaseFreeBuf_Low(pBuf);
		}
		break;

	case 2://APP
		nRes = cInspectAlign.DoFindDefectAD_GV(matSrcBuf, dPara, dResult, ptCorner);
		break;

	case 1://SVI
		nRes = cInspectAlign.DoFindDefectAD_GV_SVI(matSrcBuf, dPara, dResult, ptCorner);
		break;

	default:
		nRes = E_ERROR_CODE_FALSE;
		break;
	}

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	// 버퍼 관리 필요 없음...
	return nRes;
}

///////////////////////////////////////////////////////////////////////////
// PANEL CURL Defect
///////////////////////////////////////////////////////////////////////////
extern "C" AFX_API_EXPORT long	Panel_Curl_Judge(cv::Mat & matSrcBuf, double* dPara, cv::Point * ptCorner, BOOL & bCurl, stMeasureInfo * stCurlMeasure, BOOL bSaveImage, CString strPath)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	CInspectAlign cInspectAlign;
	cInspectAlign.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

	long nRes = cInspectAlign.CurlJudge(matSrcBuf, dPara, ptCorner, bCurl, stCurlMeasure, bSaveImage, strPath);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

//////////////////////////////////////////////////////////////////////////
// CCD Defect
//////////////////////////////////////////////////////////////////////////

extern "C" AFX_API_EXPORT	long	CCD_IndexLoad(CString strPath, CString strPath2)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	long nRes = cCCD.DefectCCDLoad(strPath, strPath2);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

extern "C" AFX_API_EXPORT	long	CCD_IndexSave(cv::Mat & matSrcBuffer, CString strPath, CString strPath2)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	long nRes = cCCD.DefectCCDSave(matSrcBuffer, strPath, strPath2);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

//////////////////////////////////////////////////////////////////////////
// SVI 컬러 보정 관련
//////////////////////////////////////////////////////////////////////////

extern "C" AFX_API_EXPORT	long	ColorCorrection_Load(wchar_t* strPath)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"));

	long nRes = cColor.ColorCorrectionLoad(strPath);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));

	return nRes;
}

//////////////////////////////////////////////////////////////////////////
// Point Defect
//////////////////////////////////////////////////////////////////////////

// Point 불량 검사
extern "C" AFX_API_EXPORT	long	Point_FindDefect(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dAlignPara, cv::Rect * rcCHoleROI, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, wchar_t* strAlgLog, cv::Mat * matCholeBuffer, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectPoint	cInspectPoint;
	cInspectPoint.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf[2];
	CostTime ct(true);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	pBuf[0] = cMem.FindFreeBuf();
	pBuf[1] = cMem.FindFreeBuf_High();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectPoint.SetMem_Multi(2, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	long nRes = cInspectPoint.DoFindPointDefect(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dAlignPara, rcCHoleROI, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, &cCCD, matCholeBuffer);

	cMem.ReleaseFreeBuf(pBuf[0]);
	cMem.ReleaseFreeBuf_High(pBuf[1]);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// Dust 제거 후, 결과 벡터 넘겨주기
extern "C" AFX_API_EXPORT	long	Point_GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat & matDrawBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Rect * rcCHoleROI, cv::Mat * matCholeBuffer, bool bBubble, stPanelBlockJudgeInfo * EngineerlockDefectJudgment = NULL)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectPoint	cInspectPoint;
	cInspectPoint.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	// 버퍼 셋팅
//CMatBuf* pBuf = cMem.FindFreeBuf();

//cInspectPoint.SetMem( pBuf );

	//버퍼 셋팅
	CMatBuf* pBuf[1];

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	cMem.FindFreeBuf_Multi_High(1, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectPoint.SetMem_Multi(1, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = true;

	if (bBubble)
	{
		nRes = cInspectPoint.GetDefectList_Bubble(matSrcBuffer, matDstBuffer, matDustBuffer, matDrawBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);
	}
	else
	{
		nRes = cInspectPoint.GetDefectList(matSrcBuffer, matDstBuffer, matDustBuffer, matDrawBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, rcCHoleROI, matCholeBuffer);
	}

	cMem.ReleaseFreeBuf_Multi_High(1, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

//////////////////////////////////////////////////////////////////////////
// Line Defect
//////////////////////////////////////////////////////////////////////////

// Line 불량 검사
extern "C" AFX_API_EXPORT	long	Line_FindDefect(cv::Mat matSrcBuffer, cv::Mat & matDrawBuffer, cv::Mat matBKBuffer, vector<int> NorchIndex, CPoint OrgIndex, cv::Point * ptCorner, double* dPara, int* nCommonPara,
	wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Rect * rcCHoleROI, cv::Mat * matCholeBuffer, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI, wchar_t* strContourTxt = NULL)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CostTime ct(true);
	CInspectLine	cInspectLine;
	cInspectLine.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf_High();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	cInspectLine.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectLine.FindLineDefect(matSrcBuffer, matDrawBuffer, matBKBuffer, NorchIndex, OrgIndex, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, rcCHoleROI, matCholeBuffer, strContourTxt);

	cMem.ReleaseFreeBuf_High(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

//////////////////////////////////////////////////////////////////////////
// Mura Defect
//////////////////////////////////////////////////////////////////////////

// Mura 불량 검사
extern "C" AFX_API_EXPORT	long	Mura_FindDefect(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CostTime ct(true);
	CInspectMura	cInspectMura;
	cInspectMura.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf_High();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectMura.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	long nRes = cInspectMura.DoFindMuraDefect(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

	cMem.ReleaseFreeBuf_High(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

//woojin 19.08.27
extern "C" AFX_API_EXPORT	long	Mura_FindDefect2(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CostTime ct(true);
	CInspectMura3	cInspectMura3;
	cInspectMura3.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	cInspectMura3.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMura3.DoFindMuraDefect(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

//woojin 19.08.27
extern "C" AFX_API_EXPORT	long	Mura_FindDefect3(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Mat & matDrawBuffer, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI, wchar_t* strContourTxt)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CostTime ct(true);
	CInspectMura3	cInspectMura3;
	cInspectMura3.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf_Low();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	cInspectMura3.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMura3.DoFindMuraDefect2(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, matDrawBuffer, strContourTxt);

	cMem.ReleaseFreeBuf_Low(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

//woojin 20.11.09
extern "C" AFX_API_EXPORT	long	Mura_FindDefect4(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Mat & matDrawBuffer, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI, wchar_t* strContourTxt)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CostTime ct(true);
	CInspectMura4	cInspectMura4;
	cInspectMura4.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf_Low();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	cInspectMura4.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMura4.DoFindMuraDefect2(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, matDrawBuffer, strContourTxt);

	cMem.ReleaseFreeBuf_Low(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// 2021.01.06 Mura Chole
extern "C" AFX_API_EXPORT	long	Mura_FindDefect_Chole(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, cv::Rect * rcCHoleROI, cv::Mat * matCholeBuffer, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Mat & matDrawBuffer, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CostTime ct(true);
	CInspectMuraChole	cInspectMuraChole;
	cInspectMuraChole.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf_Low();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	cInspectMuraChole.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMuraChole.DoFindMuraDefect(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, rcCHoleROI, matCholeBuffer, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, matDrawBuffer);

	cMem.ReleaseFreeBuf_Low(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

extern "C" AFX_API_EXPORT	long	Mura_FindDefect_Scratch(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Mat & matDrawBuffer, wchar_t* strContourTxt)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectMuraScratch	cInspectMuraScratch;
	cInspectMuraScratch.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectMuraScratch.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMuraScratch.DoFindMuraDefect_Scratch(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, matDrawBuffer, strContourTxt);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

extern "C" AFX_API_EXPORT	long	Dust_FindDefect_PS(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDarkBuffer, cv::Mat & matBrightBuffer,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, cv::Mat & matDrawBuffer, wchar_t* strContourTxt)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectDust	cInspectDust;
	cInspectDust.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectDust.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectDust.DoFindMuraDefect(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDarkBuffer, matBrightBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// Dust 제거 후, 결과 벡터 넘겨주기
extern "C" AFX_API_EXPORT	long	Mura_GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat & matDrawBuffer, cv::Point * ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectMura	cInspectMura;
	cInspectMura.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf_High();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectMura.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMura.GetDefectList(matSrcBuffer, matDstBuffer, matDustBuffer, matDrawBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, strContourTxt);

	cMem.ReleaseFreeBuf_High(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// Dust 제거 후, 결과 벡터 넘겨주기
extern "C" AFX_API_EXPORT	long	Mura_GetDefectList2(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat & matDrawBuffer, cv::Point * ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectMura3	cInspectMura3;
	cInspectMura3.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectMura3.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMura3.GetDefectList(matSrcBuffer, matDstBuffer, matDustBuffer, matDrawBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, strContourTxt);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// 2022.01.27 - 고객사 요청사항
// White 패턴 Amorph Dark 검출시 Dust 패턴에 이물이 있으면 제거
// 확인 결과 이물을 Amorph Dark로 검출하는 거였음

// Dust 제거 후, 결과 벡터 넘겨주기
extern "C" AFX_API_EXPORT	long	Mura_GetDefectList3(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat & matDrawBuffer, cv::Point * ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectMura4	cInspectMura4;
	cInspectMura4.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);
	// 버퍼 셋팅
	CMatBuf* pBuf = cMem.FindFreeBuf();

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectMura4.SetMem(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMura4.GetDefectList(matSrcBuffer, matDstBuffer, matDustBuffer, matDrawBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, strContourTxt);

	cMem.ReleaseFreeBuf(pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

//////////////////////////////////////////////////////////////////////////
// Mura Normal Defect
//////////////////////////////////////////////////////////////////////////

// Mura 불량 검사
extern "C" AFX_API_EXPORT	long	MuraNormal_FindDefect(cv::Mat matSrcBuffer, cv::Mat * *matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat & matDstBright, cv::Mat & matDstDark,
	cv::Point * ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strAlgLog, ST_ALGO_INFO * algoInfo, LogSendToUI * pLogUI)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectMuraNormal	cInspectMuraNormal;
	cInspectMuraNormal.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf[4];
	CostTime ct(true);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	cMem.FindFreeBuf_Multi_Low(4, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);
	pLogUI->SendAlgoStartLog(algoInfo, ct.get_cost_time());

	cInspectMuraNormal.SetMem_Multi(4, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMuraNormal.DoFindMuraDefect(matSrcBuffer, matSrcBufferRGB, matBKBuffer, matDstBright, matDstDark, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob);

	cMem.ReleaseFreeBuf_Multi_Low(4, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}

// Dust 제거 후, 결과 벡터 넘겨주기
extern "C" AFX_API_EXPORT	long	MuraNormal_GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat & matDrawBuffer, cv::Point * ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo * EngineerBlockDefectJudge, stDefectInfo * pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog)
{
	clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);

	CInspectMuraNormal	cInspectMuraNormal;
	cInspectMuraNormal.SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, strAlgLog);

	// 버퍼 셋팅
	CMatBuf* pBuf[1];

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find Start"), strAlgLog);

	cMem.FindFreeBuf_Multi(1, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Memory Find End"), strAlgLog);

	cInspectMuraNormal.SetMem_Multi(1, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"), strAlgLog);

	long nRes = cInspectMuraNormal.GetDefectList(matSrcBuffer, matDstBuffer, matDustBuffer, matDrawBuffer, ptCorner, dPara, nCommonPara, strAlgPath, EngineerBlockDefectJudge, pResultBlob, strContourTxt);

	cMem.ReleaseFreeBuf_Multi(1, pBuf);

	cInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"), strAlgLog);

	return nRes;
}
