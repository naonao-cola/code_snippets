#pragma once

#include "Define.h"
#include "StdAfx.h"
#include "DefineInterface_SITE.h"

//2022.05.25......
#include <ctime>

#include <string>
#include <thread>
using namespace std;
enum INSP_TYPE
{
	INSP_AVI = 0,
	INSP_SVI,
	INSP_APP,
	MAX_INSP_TYPE
};

class DICS_B11
{
public:
	DICS_B11();
	~DICS_B11(void);

	void DICSStart(cv::Mat& DicsImg, cv::Mat& matOrigin, cv::Point* ptCorner, int InspType, CString Path, CString ID, CString Patturn, int nCameraNum, double* dAlgPara, CString strDrive);
	void SaveStart(cv::Mat& matROI, cv::Mat& matDics, CString INIPath, CString ID, CString Patturn, int InspType, int nCameaNum, CString strDrive);
	void Generate(cv::Mat matOrigin, cv::Mat& matDics, cv::Mat& matROI, cv::Point* ptCorner, int InspType, CString INIPath, CString ID, CString Patturn, int nCameraNum, double* dAlgPara, CString strDrive);
	void Preprocessing_AVI_DUST_PAD(cv::Mat& matROI, double* dAlgPara, CString Patturn, CString strDirectoryPath);
	static CString CheckDirectory(CString Path, CString ID, CString strDrive);
protected:

	static  void StartSave(cv::Mat& DicsImg, cv::Mat matOrigin, cv::Point* ptCorner, int InspType, CString INIPath, CString ID, CString Patturn, int nCameraNum, double* dAlgPara, CString strDrive);

	static void Preprocessing_AVI(cv::Mat& matROI, cv::Mat& matPreprocessing, CString Patturn, double* dAlgPara);

	static void Preprocessing_SVI(cv::Mat& matROI, cv::Mat& matPreprocessing, CString Patturn, double* dAlgPara);

	static void SaveImage(cv::Mat& matROI, cv::Mat& matPreprocessing, CString  strDirectoryPath, CString Patturn, int InspType, int nCameaNum);

	static void Contrast(cv::Mat& matSrc, cv::Mat& matDst, int nMin, int nMax, int nCount);
	static void ContrastColor(cv::Mat& matSrc, cv::Mat& matDst, int nMin, int nMax, int nCount);

	//2022.05.25......

	static void DeleteDirectoryFile(CString Path);

	// 2022.07.04
	static bool CheckPattern(CString Patturn, CString INIPath);

	// 2022.07.29
	static void Preprocessing_AVI_DUST(cv::Mat& matROI, cv::Mat& matPreprocessing, double* dAlgPara);
	static void ShiftCopy(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nShiftX, int nShiftY, int nShiftLoopX, int nShiftLoopY);
};