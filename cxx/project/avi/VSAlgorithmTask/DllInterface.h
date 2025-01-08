////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef DLL_INTERFACE_H
#define DLL_INTERFACE_H

#pragma once

#include "Define.h"
#include "DefineInterface.h"
#include "../../visualstation/CommonHeader/Class/LogSendToUI.h"

#include<vector>
using namespace std;

//////////////////////////////////////////////////////////////////////////
#pragma comment(lib, "InspectLib.lib")


EXTERNC DLLAPI void	Init_InspectLib(CString initFilePath);

//////////////////////////////////////////////////////////////////////////
//Align相关
//////////////////////////////////////////////////////////////////////////

//查找Top Line角度
EXTERNC DLLAPI long	Align_FindTheta(cv::Mat matSrcBuf, double* dPara, double& dThet, cv::Point& ptCentera=cv::Point(), wchar_t* strID=NULL);

//查找检查区域
EXTERNC DLLAPI long	Align_FindActive(cv::Mat matSrcBuf, double* dPara, double& dTheta, cv::Point* ptResCorner, cv::Point* ptContCorner, int nCameraNum, int nEQType,double dCamResolution,
	double dPannelSizeX,double dPannelSizeY ,int nRatio=1, cv::Point& ptCenter=cv::Point(), wchar_t* strID=NULL);

//设置轮廓曲线&保存文件
EXTERNC DLLAPI long	Align_SetFindContour(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt,
	double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strPath, Point* ptAlignCorner=NULL, CStringA strImageName=NULL, double dAlignTheta=0, bool bIamgeSave=false);

EXTERNC DLLAPI long	Align_SetFindContour_(cv::Mat matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nCHoleROICnt,
	double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strPath, Point* ptAlignCorner = NULL, CStringA strImageName = NULL, double dAlignTheta = 0, bool bIamgeSave = false);

EXTERNC DLLAPI long	Align_SetFindContour_2(cv::Mat *matSrcBuf, INSP_AREA RoundROI[MAX_MEM_SIZE_E_INSPECT_AREA], INSP_AREA CHoleROI[MAX_MEM_SIZE_E_INSPECT_AREA], int nRoundROICnt, int nCHoleROICnt,
	double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strPath, Point* ptAlignCorner = NULL, CStringA strImageName = NULL, double dAlignTheta = 0, bool bIamgeSave = false);

// ¿U°u ºIºÐ A³¸®
EXTERNC DLLAPI long	Align_FindFillOutArea(cv::Mat matSrcBuf, cv::Mat& MatDrawBuffer, cv::Mat matBKGBuf, cv::Point ptResCorner[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA], cv::Mat* matCHoleROIBuf, cv::Rect* rcCHoleROI, bool* bCHoleAD,
	double* dPara, int nAlgImg, int nCameraNum, int nRatio, int nEqpType, wchar_t* strAlgLog, wchar_t* strID=NULL, cv::Point* ptCorner=NULL, vector<vector<Point2i>> &ptActive = vector<vector<Point2i>>(), double dAlignTH=0,  CString strPath=NULL, bool bImageSave = false);

//外围处理
EXTERNC DLLAPI long	Align_FindFillOutAreaDust(cv::Mat matSrcBuf, cv::Mat& MatDrawBuffer, cv::Point ptResCorner[E_CORNER_END], STRU_LabelMarkParams& labelMarkParams, STRU_LabelMarkInfo& labelMarkInfo, double dAngle, cv::Rect* rcCHoleROI, ROUND_SET tRoundSet[MAX_MEM_SIZE_E_INSPECT_AREA], ROUND_SET tCHoleSet[MAX_MEM_SIZE_E_INSPECT_AREA],
	double* dPara, int nAlgImg, int nRatio, wchar_t* strAlgLog, wchar_t* strID = NULL);

//画面旋转
EXTERNC DLLAPI long	Align_RotateImage(cv::Mat matSrcBuffer, cv::Mat& matDstBuffer, double dAngle);

//旋转坐标
EXTERNC DLLAPI long	Align_DoRotatePoint(cv::Point matSrcPoint, cv::Point& matDstPoint, cv::Point ptCenter, double dAngle);

//AD检查(AVI,SVI)/dResult:当前Cell匹配率
EXTERNC DLLAPI long	Align_FindDefectAD(cv::Mat matSrcBuf, double* dPara, double* dResult, int nRatio, int nCamera, int nEqpType);

//PG 压接检查 hjf
EXTERNC DLLAPI long	Align_FindDefectPGConnect(cv::Mat matSrcBuf, double* dPara, double* dResult, cv::Point* nRatio, int nCameraNum, int nEqpType);

// Mark
EXTERNC DLLAPI long Align_FindDefectMark(cv::Mat matSrcBuf, double* dPara, cv::Point* ptCorner, double dAngel, cv::Rect rcMarkROI[MAX_MEM_SIZE_MARK_COUNT], CRect rectMarkArea[MAX_MEM_SIZE_MARK_COUNT], int nEqpType, int nMarkROICnt);

//Label
EXTERNC DLLAPI long Align_FindDefectLabel(cv::Mat matSrcBuf, double* dPara, cv::Point* ptCorner, double dAngel, CRect rectLabelArea[MAX_MEM_SIZE_LABEL_COUNT], int nEqpType);

//AD GV检查
EXTERNC DLLAPI long	Align_FindDefectAD_GV(cv::Mat& matSrcBuf, double* dPara, double* dResult, cv::Point* ptCorner, int nEqpType, int nImageNum, wchar_t* strAlgLog);

EXTERNC DLLAPI long	Panel_Curl_Judge(cv::Mat& matSrcBuf, double* dPara, cv::Point* ptCorner, BOOL &bCurl, stMeasureInfo* stCurlMeasure,BOOL bSaveImage, CString strPath);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

EXTERNC DLLAPI long	PM_ImageLoad(int nImageNum, wchar_t* strPath);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

EXTERNC DLLAPI long	CCD_IndexLoad(CString strPath, CString strPath2);

EXTERNC DLLAPI long	CCD_IndexSave(cv::Mat& matSrcBuffer, CString strPath, CString strPath2);

//////////////////////////////////////////////////////////////////////////
//SVI色彩校正相关
//////////////////////////////////////////////////////////////////////////

EXTERNC DLLAPI long	ColorCorrection_Load(CString strPath);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

//Point不良检查
EXTERNC DLLAPI long	Point_FindDefect(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dAlignPara, cv::Rect* rcCHoleROI, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, wchar_t* strAlgLog , cv::Mat* matCholeBuffer, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI, stPanelBlockJudgeInfo* EngineerlockDefectJudgment = NULL);

//删除Dust后,转交结果向量
EXTERNC DLLAPI long	Point_GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, bool bBubble = false, stPanelBlockJudgeInfo* EngineerlockDefectJudgment = NULL);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

//线路不良检查
EXTERNC DLLAPI long Line_FindDefect(cv::Mat matSrcBuffer, cv::Mat& matDrawBuffer, cv::Mat matBKBuffer, vector<int> NorchIndex, CPoint OrgIndex, cv::Point* ptCorner, double* dPara, int* nCommonPara,
	wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI,wchar_t* strContourTxt = NULL);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

//Mura不良检查
EXTERNC DLLAPI long Mura_FindDefect(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI);

EXTERNC DLLAPI long Mura_FindDefect2(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI);

EXTERNC DLLAPI long Mura_FindDefect3(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Mat& matDrawBuffer, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI,wchar_t* strContourTxt = NULL);

//在Mura4中使用
EXTERNC DLLAPI long Mura_FindDefect4(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Mat& matDrawBuffer, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI, wchar_t* strContourTxt = NULL);

//由Mura Chole使用2021.01.06
EXTERNC DLLAPI long Mura_FindDefect_Chole(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, cv::Rect* rcCHoleROI, cv::Mat* matCholeBuffer, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Mat& matDrawBuffer, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI, wchar_t* strContourTxt = NULL);

EXTERNC DLLAPI long Mura_FindDefect_Scratch(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Mat& matDrawBuffer, wchar_t* strContourTxt = NULL);

EXTERNC DLLAPI long Dust_FindDefect_PS(cv::Mat matSrcBuffer, cv::Mat** matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDarkBuffer, cv::Mat& matBrightBuffer,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Mat& matDrawBuffer, wchar_t* strContourTxt = NULL);

//删除Dust后,转交结果向量
EXTERNC DLLAPI long Mura_GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog);

//删除Dust后,转交结果向量
EXTERNC DLLAPI long Mura_GetDefectList2(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog);

//-客户请求
//白色模式Amorp Dark检测Dust模式中有异物时删除
//确认结果显示,用Amorph Dark检测出了异物

//删除Dust后,转交结果向量
EXTERNC DLLAPI long Mura_GetDefectList3(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////

//Mura不良检查
EXTERNC DLLAPI long MuraNormal_FindDefect(cv::Mat matSrcBuffer, cv::Mat **matSrcBufferRGB, cv::Mat matBKBuffer, cv::Mat& matDstBright, cv::Mat& matDstDark,
	cv::Point* ptCorner, double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, ST_ALGO_INFO* algoInfo, LogSendToUI* pLogUI);

//删除Dust后,转交结果向量
EXTERNC DLLAPI long MuraNormal_GetDefectList(cv::Mat matSrcBuffer, cv::Mat matDstBuffer[2], cv::Mat matDustBuffer[2], cv::Mat& matDrawBuffer, cv::Point* ptCorner,
	double* dPara, int* nCommonPara, wchar_t* strAlgPath, stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strContourTxt, wchar_t* strAlgLog);

//////////////////////////////////////////////////////////////////////////
EXTERNC DLLAPI long AI_InspectDemo(cv::Mat matSrcBuffer, cv::Mat& matSrcBufferRGB, cv::Point* ptCorner, double* dAlgPara,
	stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog);

EXTERNC DLLAPI long AI_DICSInspect(cv::Mat& matSrcBuffer, cv::Mat& matSrcBufferRGB, int* nCommonPara, wchar_t strPath[][1024],
	stPanelBlockJudgeInfo* EngineerBlockDefectJudge, stDefectInfo* pResultBlob, wchar_t* strAlgLog, cv::Rect cutRoi);

EXTERNC DLLAPI long AI_ModelSetParamter(double* dPara);

EXTERNC DLLAPI long AI_Initialization(const std::string& config);

#endif
